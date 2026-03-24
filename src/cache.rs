use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;

use crate::index::{idf, ScoredResult, VtplIndex};
use crate::posting::ChunkId;
use crate::pq::PqCode;

// ── Trigram-level cache entry ───────────────────────────────────────

/// Per-chunk result from scanning a single trigram's posting list.
#[derive(Clone)]
struct TrigramHit {
    chunk_id: ChunkId,
    pq_code: PqCode,
    idf_weight: f32,
}

/// Cached result of scanning one trigram's posting list.
struct TrigramCacheEntry {
    hits: Arc<Vec<TrigramHit>>,
    confidence: u64,
}

// ── Word-level pattern cache ────────────────────────────────────────

/// Maps a normalized word to the trigrams it decomposes into.
/// "concurrent" → {"con","onc","ncu","cur","urr","rre","ren","ent"}
struct WordEntry {
    grams: Vec<String>,
    confidence: u64,
}

// ── Embedding fingerprint cache ─────────────────────────────────────

/// Quantized embedding fingerprint → per-chunk semantic scores.
/// Embeddings within the same quantization bucket reuse scores.
struct SemanticCacheEntry {
    scores: Arc<HashMap<ChunkId, f32>>,
    confidence: u64,
}

fn embedding_fingerprint(emb: &[f32], resolution: f32) -> Vec<i16> {
    emb.iter().map(|&v| (v / resolution).round() as i16).collect()
}

fn fingerprint_hash(fp: &[i16]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &v in fp {
        h ^= v as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

// ── Config ──────────────────────────────────────────────────────────

/// Tuning knobs for the smart cache.
pub struct CacheConfig {
    /// Max trigram posting list results to keep cached.
    pub trigram_capacity: usize,
    /// Max word decompositions to cache.
    pub word_capacity: usize,
    /// Max embedding fingerprints to cache semantic scores for.
    pub semantic_capacity: usize,
    /// Quantization resolution for embedding fingerprints.
    /// Smaller = more buckets = more precise but fewer cache hits.
    /// 0.05 is a good default (embeddings in [-1,1] → ~40 buckets per dim).
    pub embedding_resolution: f32,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            trigram_capacity: 50_000,
            word_capacity: 100_000,
            semantic_capacity: 5_000,
            embedding_resolution: 0.05,
        }
    }
}

// ── CachedIndex ─────────────────────────────────────────────────────

/// Smart cached wrapper around VtplIndex.
///
/// Instead of caching only identical queries, this caches at three levels:
///
/// 1. **Word cache** — "concurrent" always decomposes to the same trigrams.
///    Any query containing "concurrent" reuses that decomposition instantly.
///
/// 2. **Trigram cache** — each trigram's posting list scan (chunk hits + IDF weights)
///    is cached. "concurrent hash" and "concurrent map" share all "concurrent" trigrams.
///    New queries only scan uncached trigrams from disk.
///
/// 3. **Semantic cache** — embeddings are quantized into fingerprints. Queries with
///    similar embeddings (within resolution bucket) reuse per-chunk cosine scores
///    instead of recomputing distance table lookups.
///
/// Eviction is confidence-based: frequently-accessed entries survive longer.
pub struct CachedIndex {
    index: VtplIndex,

    trigram_cache: RwLock<HashMap<String, TrigramCacheEntry>>,
    word_cache: RwLock<HashMap<String, WordEntry>>,
    semantic_cache: RwLock<HashMap<u64, SemanticCacheEntry>>,

    config: CacheConfig,
    tick: AtomicU64,

    hits_trigram: AtomicU64,
    misses_trigram: AtomicU64,
    hits_semantic: AtomicU64,
    misses_semantic: AtomicU64,
    hits_word: AtomicU64,
    misses_word: AtomicU64,
}

impl CachedIndex {
    pub fn new(index: VtplIndex, config: CacheConfig) -> Self {
        Self {
            index,
            trigram_cache: RwLock::new(HashMap::new()),
            word_cache: RwLock::new(HashMap::new()),
            semantic_cache: RwLock::new(HashMap::new()),
            config,
            tick: AtomicU64::new(0),
            hits_trigram: AtomicU64::new(0),
            misses_trigram: AtomicU64::new(0),
            hits_semantic: AtomicU64::new(0),
            misses_semantic: AtomicU64::new(0),
            hits_word: AtomicU64::new(0),
            misses_word: AtomicU64::new(0),
        }
    }

    pub fn with_defaults(index: VtplIndex) -> Self {
        Self::new(index, CacheConfig::default())
    }

    /// Smart fused query. Reuses cached trigram hits, word decompositions,
    /// and semantic scores from similar prior queries.
    pub fn query(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        lambda: f32,
        top_k: usize,
    ) -> Vec<ScoredResult> {
        let now = self.tick.fetch_add(1, Ordering::Relaxed);

        // Step 1: decompose query into trigrams (word cache accelerates this)
        let grams = self.cached_trigram_decomposition(query_text, now);
        let total_grams = grams.len() as f32;
        if total_grams == 0.0 {
            return Vec::new();
        }

        // Step 2: gather per-chunk pattern scores from trigram cache
        let total_docs = self.index.total_docs();
        let mut accum: HashMap<ChunkId, (f32, u32, PqCode)> = HashMap::new();

        for gram in &grams {
            let hits = self.cached_trigram_scan(gram, total_docs, now);
            for hit in hits.iter() {
                let acc = accum
                    .entry(hit.chunk_id)
                    .or_insert((0.0, 0, hit.pq_code));
                acc.0 += hit.idf_weight;
                acc.1 += 1;
            }
        }

        // Step 3: semantic scores (embedding fingerprint cache)
        let fp = embedding_fingerprint(query_embedding, self.config.embedding_resolution);
        let fp_hash = fingerprint_hash(&fp);
        let cached_semantic = self.get_cached_semantic(fp_hash, now);

        // Build distance table only if we have uncached chunks
        let dt = if cached_semantic.is_none()
            || accum.keys().any(|id| {
                cached_semantic
                    .as_ref()
                    .is_none_or(|s| !s.contains_key(id))
            })
        {
            Some(self.index.codebook.build_distance_table(query_embedding))
        } else {
            None
        };

        // Step 4: assemble final scores
        let mut new_semantic: HashMap<ChunkId, f32> = HashMap::new();
        let mut results: Vec<ScoredResult> = accum
            .into_iter()
            .map(|(chunk_id, (idf_sum, hit_count, pq_code))| {
                let pattern_score =
                    (hit_count as f32 / total_grams) * (idf_sum / hit_count as f32);

                let semantic_score = cached_semantic
                    .as_ref()
                    .and_then(|s| s.get(&chunk_id).copied())
                    .unwrap_or_else(|| {
                        let s = dt.as_ref().unwrap().approximate_cosine(&pq_code);
                        new_semantic.insert(chunk_id, s);
                        s
                    });

                let score = lambda * semantic_score + (1.0 - lambda) * pattern_score;
                ScoredResult { chunk_id, score, semantic_score, pattern_score }
            })
            .collect();

        // Step 5: update semantic cache with newly computed scores
        if !new_semantic.is_empty() {
            self.update_semantic_cache(fp_hash, new_semantic, now);
        }

        results.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(top_k);
        results
    }

    /// Text-only query using the trigram cache.
    pub fn text_query(&self, query_text: &str, top_k: usize) -> Vec<ScoredResult> {
        let now = self.tick.fetch_add(1, Ordering::Relaxed);
        let grams = self.cached_trigram_decomposition(query_text, now);
        let total_grams = grams.len() as f32;
        if total_grams == 0.0 {
            return Vec::new();
        }

        let total_docs = self.index.total_docs();
        let mut accum: HashMap<ChunkId, (f32, u32)> = HashMap::new();

        for gram in &grams {
            let hits = self.cached_trigram_scan(gram, total_docs, now);
            for hit in hits.iter() {
                let acc = accum.entry(hit.chunk_id).or_insert((0.0, 0));
                acc.0 += hit.idf_weight;
                acc.1 += 1;
            }
        }

        let mut results: Vec<ScoredResult> = accum
            .into_iter()
            .map(|(chunk_id, (idf_sum, hit_count))| {
                let pattern_score =
                    (hit_count as f32 / total_grams) * (idf_sum / hit_count as f32);
                ScoredResult { chunk_id, score: pattern_score, semantic_score: 0.0, pattern_score }
            })
            .collect();

        results.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(top_k);
        results
    }

    /// Vector-only query — delegates to index (semantic cache doesn't help much here
    /// since we'd need to scan all posting lists regardless).
    pub fn vector_query(&self, query_embedding: &[f32], top_k: usize) -> Vec<ScoredResult> {
        self.index.vector_query(query_embedding, top_k)
    }

    // ── Word cache ──────────────────────────────────────────────────

    fn cached_trigram_decomposition(&self, text: &str, now: u64) -> Vec<String> {
        let normalized: String = text
            .to_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { ' ' })
            .collect();

        let words: Vec<&str> = normalized.split_whitespace().collect();
        let mut all_grams = std::collections::BTreeSet::new();

        for word in &words {
            // Check word cache
            {
                let cache = self.word_cache.read();
                if let Some(entry) = cache.get(*word) {
                    self.hits_word.fetch_add(1, Ordering::Relaxed);
                    for g in &entry.grams {
                        all_grams.insert(g.clone());
                    }
                    // Amortized confidence bump — only write-lock every 8th tick
                    if now & 7 == 0 {
                        drop(cache);
                        if let Some(entry) = self.word_cache.write().get_mut(*word) {
                            entry.confidence += now;
                        }
                    }
                    continue;
                }
            }

            // Cache miss: compute trigrams for this word
            self.misses_word.fetch_add(1, Ordering::Relaxed);
            let chars: Vec<char> = word.chars().collect();
            let mut word_grams = Vec::new();
            if chars.len() < 3 {
                word_grams.push(word.to_string());
            } else {
                for window in chars.windows(3) {
                    word_grams.push(window.iter().collect());
                }
            }

            for g in &word_grams {
                all_grams.insert(g.clone());
            }

            // Insert into word cache with eviction
            let mut cache = self.word_cache.write();
            if cache.len() >= self.config.word_capacity {
                self.evict_lowest_confidence_word(&mut cache);
            }
            cache.insert(word.to_string(), WordEntry { grams: word_grams, confidence: now });
        }

        all_grams.into_iter().collect()
    }

    // ── Trigram cache ───────────────────────────────────────────────

    fn cached_trigram_scan(&self, gram: &str, total_docs: u32, now: u64) -> Arc<Vec<TrigramHit>> {
        // Check cache
        {
            let cache = self.trigram_cache.read();
            if let Some(entry) = cache.get(gram) {
                self.hits_trigram.fetch_add(1, Ordering::Relaxed);
                let hits = Arc::clone(&entry.hits);
                // Only write-lock to bump confidence every 8th access (amortized)
                if now & 7 == 0 {
                    drop(cache);
                    if let Some(entry) = self.trigram_cache.write().get_mut(gram) {
                        entry.confidence += now;
                    }
                }
                return hits;
            }
        }

        // Cache miss: scan the posting list
        self.misses_trigram.fetch_add(1, Ordering::Relaxed);
        let hits: Vec<TrigramHit> = match self.index.get_posting_list(gram) {
            Some(list) => {
                let idf_weight = idf(total_docs, self.index.get_df(gram));
                list.entries
                    .iter()
                    .map(|entry| TrigramHit {
                        chunk_id: entry.chunk_id,
                        pq_code: entry.pq_code,
                        idf_weight,
                    })
                    .collect()
            }
            None => Vec::new(),
        };

        let arc_hits = Arc::new(hits);

        // Insert with eviction
        let mut cache = self.trigram_cache.write();
        if cache.len() >= self.config.trigram_capacity {
            self.evict_lowest_confidence_trigram(&mut cache);
        }
        cache.insert(
            gram.to_string(),
            TrigramCacheEntry { hits: Arc::clone(&arc_hits), confidence: now },
        );

        arc_hits
    }

    // ── Semantic cache ──────────────────────────────────────────────

    fn get_cached_semantic(&self, fp_hash: u64, now: u64) -> Option<Arc<HashMap<ChunkId, f32>>> {
        let cache = self.semantic_cache.read();
        if let Some(entry) = cache.get(&fp_hash) {
            self.hits_semantic.fetch_add(1, Ordering::Relaxed);
            let scores = Arc::clone(&entry.scores);
            drop(cache);
            if let Some(entry) = self.semantic_cache.write().get_mut(&fp_hash) {
                entry.confidence += now;  // accumulate frequency
            }
            return Some(scores);
        }
        self.misses_semantic.fetch_add(1, Ordering::Relaxed);
        None
    }

    fn update_semantic_cache(
        &self,
        fp_hash: u64,
        new_scores: HashMap<ChunkId, f32>,
        now: u64,
    ) {
        let mut cache = self.semantic_cache.write();

        let entry = cache.entry(fp_hash).or_insert_with(|| SemanticCacheEntry {
            scores: Arc::new(HashMap::new()),
            confidence: now,
        });
        // Merge new scores into existing
        let mut merged = (*entry.scores).clone();
        merged.extend(new_scores);
        entry.scores = Arc::new(merged);
        entry.confidence = now;

        if cache.len() > self.config.semantic_capacity {
            self.evict_lowest_confidence_semantic(&mut cache);
        }
    }

    // ── Confidence-based eviction ───────────────────────────────────

    fn evict_lowest_confidence_trigram(&self, cache: &mut HashMap<String, TrigramCacheEntry>) {
        let to_evict = cache.len() / 10; // evict 10% at a time
        let mut entries: Vec<(String, u64)> = cache
            .iter()
            .map(|(k, v)| (k.clone(), v.confidence))
            .collect();
        entries.sort_unstable_by_key(|(_, c)| *c);
        for (key, _) in entries.into_iter().take(to_evict.max(1)) {
            cache.remove(&key);
        }
    }

    fn evict_lowest_confidence_word(&self, cache: &mut HashMap<String, WordEntry>) {
        let to_evict = cache.len() / 10;
        let mut entries: Vec<(String, u64)> = cache
            .iter()
            .map(|(k, v)| (k.clone(), v.confidence))
            .collect();
        entries.sort_unstable_by_key(|(_, c)| *c);
        for (key, _) in entries.into_iter().take(to_evict.max(1)) {
            cache.remove(&key);
        }
    }

    fn evict_lowest_confidence_semantic(
        &self,
        cache: &mut HashMap<u64, SemanticCacheEntry>,
    ) {
        let to_evict = cache.len() / 10;
        let mut entries: Vec<(u64, u64)> = cache
            .iter()
            .map(|(&k, v)| (k, v.confidence))
            .collect();
        entries.sort_unstable_by_key(|(_, c)| *c);
        for (key, _) in entries.into_iter().take(to_evict.max(1)) {
            cache.remove(&key);
        }
    }

    // ── Stats ───────────────────────────────────────────────────────

    pub fn clear_cache(&self) {
        self.trigram_cache.write().clear();
        self.word_cache.write().clear();
        self.semantic_cache.write().clear();
        self.hits_trigram.store(0, Ordering::Relaxed);
        self.misses_trigram.store(0, Ordering::Relaxed);
        self.hits_word.store(0, Ordering::Relaxed);
        self.misses_word.store(0, Ordering::Relaxed);
        self.hits_semantic.store(0, Ordering::Relaxed);
        self.misses_semantic.store(0, Ordering::Relaxed);
    }

    pub fn trigram_cache_size(&self) -> usize { self.trigram_cache.read().len() }
    pub fn word_cache_size(&self) -> usize { self.word_cache.read().len() }
    pub fn semantic_cache_size(&self) -> usize { self.semantic_cache.read().len() }

    pub fn stats(&self) -> CacheStats {
        CacheStats {
            trigram_hits: self.hits_trigram.load(Ordering::Relaxed),
            trigram_misses: self.misses_trigram.load(Ordering::Relaxed),
            word_hits: self.hits_word.load(Ordering::Relaxed),
            word_misses: self.misses_word.load(Ordering::Relaxed),
            semantic_hits: self.hits_semantic.load(Ordering::Relaxed),
            semantic_misses: self.misses_semantic.load(Ordering::Relaxed),
        }
    }

    pub fn inner(&self) -> &VtplIndex { &self.index }
}

/// Cache performance statistics.
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub trigram_hits: u64,
    pub trigram_misses: u64,
    pub word_hits: u64,
    pub word_misses: u64,
    pub semantic_hits: u64,
    pub semantic_misses: u64,
}

impl CacheStats {
    pub fn trigram_hit_rate(&self) -> f64 {
        let total = self.trigram_hits + self.trigram_misses;
        if total == 0 { 0.0 } else { self.trigram_hits as f64 / total as f64 }
    }
    pub fn word_hit_rate(&self) -> f64 {
        let total = self.word_hits + self.word_misses;
        if total == 0 { 0.0 } else { self.word_hits as f64 / total as f64 }
    }
    pub fn semantic_hit_rate(&self) -> f64 {
        let total = self.semantic_hits + self.semantic_misses;
        if total == 0 { 0.0 } else { self.semantic_hits as f64 / total as f64 }
    }
}

impl std::fmt::Display for CacheStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "trigram {}/{} ({:.0}%) | word {}/{} ({:.0}%) | semantic {}/{} ({:.0}%)",
            self.trigram_hits,
            self.trigram_hits + self.trigram_misses,
            self.trigram_hit_rate() * 100.0,
            self.word_hits,
            self.word_hits + self.word_misses,
            self.word_hit_rate() * 100.0,
            self.semantic_hits,
            self.semantic_hits + self.semantic_misses,
            self.semantic_hit_rate() * 100.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{l2_normalize, PqCodebook};
    use rand::Rng;

    fn random_vec(dim: usize) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        let mut v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        l2_normalize(&mut v);
        v
    }

    fn build_small_index(dim: usize) -> VtplIndex {
        let train: Vec<Vec<f32>> = (0..300).map(|_| random_vec(dim)).collect();
        let cb = PqCodebook::train(&train, dim, 15);
        let mut idx = VtplIndex::new(cb);
        idx.insert(0, "concurrent hash map implementation", &random_vec(dim));
        idx.insert(1, "vector search product quantization", &random_vec(dim));
        idx.insert(2, "concurrent programming patterns", &random_vec(dim));
        idx.insert(3, "hash table with chaining", &random_vec(dim));
        idx.finalize();
        idx
    }

    #[test]
    fn overlapping_queries_share_trigram_cache() {
        let dim = 128;
        let idx = build_small_index(dim);
        let cached = CachedIndex::with_defaults(idx);
        let emb = random_vec(dim);

        // First query populates trigram cache for "concurrent" and "hash" trigrams
        let _r1 = cached.query("concurrent hash", &emb, 0.6, 5);
        let s1 = cached.stats();
        assert!(s1.trigram_misses > 0);

        // Second query shares "concurrent" trigrams — should get hits
        let _r2 = cached.query("concurrent programming", &emb, 0.6, 5);
        let s2 = cached.stats();
        assert!(s2.trigram_hits > 0, "overlapping trigrams should hit cache");
    }

    #[test]
    fn word_cache_accelerates_repeated_words() {
        let dim = 128;
        let idx = build_small_index(dim);
        let cached = CachedIndex::with_defaults(idx);
        let emb = random_vec(dim);

        let _r1 = cached.query("concurrent hash", &emb, 0.6, 5);
        let s1 = cached.stats();
        let word_misses_after_first = s1.word_misses;

        // "concurrent" word decomposition should be cached now
        let _r2 = cached.query("concurrent map", &emb, 0.6, 5);
        let s2 = cached.stats();
        assert!(s2.word_hits > 0, "repeated word should hit word cache");
        assert!(
            s2.word_misses < word_misses_after_first + 2,
            "only new words should miss"
        );
    }

    #[test]
    fn similar_embeddings_share_semantic_cache() {
        let dim = 128;
        let idx = build_small_index(dim);
        let resolution = 0.2;
        let config = CacheConfig {
            embedding_resolution: resolution,
            ..CacheConfig::default()
        };
        let cached = CachedIndex::new(idx, config);

        // Build emb1, then construct emb2 that is guaranteed to be in the same bucket:
        // quantize emb1, then reconstruct the bucket center.
        let emb1 = random_vec(dim);
        let emb2: Vec<f32> = emb1
            .iter()
            .map(|&v| {
                let bucket = (v / resolution).round();
                bucket * resolution // exact bucket center — guaranteed same fingerprint
            })
            .collect();
        // emb1 might differ from emb2, but they share the same fingerprint.
        // To be safe, use emb2 for both queries (same fingerprint guaranteed).
        let emb1_snapped: Vec<f32> = emb1
            .iter()
            .map(|&v| (v / resolution).round() * resolution)
            .collect();

        let _r1 = cached.query("concurrent", &emb1_snapped, 0.6, 5);
        let s1 = cached.stats();
        assert_eq!(s1.semantic_hits, 0);

        let _r2 = cached.query("concurrent", &emb2, 0.6, 5);
        let s2 = cached.stats();
        assert!(
            s2.semantic_hits > 0,
            "same-bucket embedding should hit semantic cache"
        );
    }

    #[test]
    fn confidence_eviction_keeps_hot_entries() {
        let dim = 128;
        let train: Vec<Vec<f32>> = (0..300).map(|_| random_vec(dim)).collect();
        let cb = PqCodebook::train(&train, dim, 15);
        let mut idx = VtplIndex::new(cb);
        idx.insert(0, "aaa bbb ccc", &random_vec(dim));
        idx.insert(1, "ddd eee fff", &random_vec(dim));
        idx.insert(2, "ggg hhh iii", &random_vec(dim));
        idx.finalize();

        let config = CacheConfig {
            trigram_capacity: 4,
            word_capacity: 100,
            semantic_capacity: 100,
            embedding_resolution: 0.05,
        };
        let cached = CachedIndex::new(idx, config);
        let emb = random_vec(dim);

        // Query "aaa" many times to build high confidence (amortized bumps every 8 ticks)
        for _ in 0..50 {
            let _ = cached.query("aaa", &emb, 0.6, 5);
        }

        // Fill cache past capacity with different trigrams
        let _ = cached.query("ddd eee", &emb, 0.6, 5);
        let _ = cached.query("ggg hhh", &emb, 0.6, 5);

        // "aaa" trigram should still be cached (high confidence survives eviction)
        let before = cached.stats().trigram_hits;
        let _ = cached.query("aaa", &emb, 0.6, 5);
        let after = cached.stats().trigram_hits;
        assert!(
            after > before,
            "high-confidence trigrams should survive eviction"
        );
    }

    #[test]
    fn cached_results_match_uncached() {
        let dim = 128;
        let idx = build_small_index(dim);
        let emb = random_vec(dim);

        let uncached = idx.query("concurrent hash", &emb, 0.6, 5);
        let cached_idx = CachedIndex::with_defaults(idx);
        let from_cache = cached_idx.query("concurrent hash", &emb, 0.6, 5);

        assert_eq!(uncached.len(), from_cache.len());
        for (a, b) in uncached.iter().zip(from_cache.iter()) {
            assert_eq!(a.chunk_id, b.chunk_id);
            assert!((a.score - b.score).abs() < 1e-6, "scores should match");
        }
    }
}

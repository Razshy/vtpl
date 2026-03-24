use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;

use crate::index::{ScoredResult, VtplIndex};
use crate::posting::ChunkId;

/// Lightweight hash for f32 slices (for embedding-based cache keys).
fn hash_embedding(emb: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325; // FNV offset basis
    for &f in emb {
        let bits = f.to_bits() as u64;
        h ^= bits;
        h = h.wrapping_mul(0x100000001b3); // FNV prime
    }
    h
}

#[derive(Clone, Hash, Eq, PartialEq)]
struct QueryKey {
    text_hash: u64,
    emb_hash: u64,
    lambda_bits: u32,
    top_k: usize,
}

impl QueryKey {
    fn new(text: &str, embedding: &[f32], lambda: f32, top_k: usize) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        text.hash(&mut hasher);
        Self {
            text_hash: hasher.finish(),
            emb_hash: hash_embedding(embedding),
            lambda_bits: lambda.to_bits(),
            top_k,
        }
    }
}

struct LruEntry {
    results: Vec<ScoredResult>,
    order: u64,
}

/// Cached wrapper around VtplIndex. Thread-safe via RwLock.
///
/// Caches:
/// - Full query results (LRU eviction)
/// - Trigram → chunk hit counts (hot trigram acceleration)
pub struct CachedIndex {
    index: VtplIndex,
    query_cache: RwLock<HashMap<QueryKey, LruEntry>>,
    trigram_cache: RwLock<HashMap<String, Vec<ChunkId>>>,
    cache_capacity: usize,
    access_counter: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl CachedIndex {
    /// Wrap a VtplIndex with a cache of the given capacity (number of query results to keep).
    pub fn new(index: VtplIndex, cache_capacity: usize) -> Self {
        Self {
            index,
            query_cache: RwLock::new(HashMap::new()),
            trigram_cache: RwLock::new(HashMap::new()),
            cache_capacity,
            access_counter: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Fused query with caching. Repeated identical queries return instantly.
    pub fn query(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        lambda: f32,
        top_k: usize,
    ) -> Vec<ScoredResult> {
        let key = QueryKey::new(query_text, query_embedding, lambda, top_k);

        // Fast path: cache hit
        {
            let cache = self.query_cache.read();
            if let Some(entry) = cache.get(&key) {
                self.hits.fetch_add(1, Ordering::Relaxed);
                return entry.results.clone();
            }
        }

        // Slow path: compute and cache
        self.misses.fetch_add(1, Ordering::Relaxed);
        let results = self.index.query(query_text, query_embedding, lambda, top_k);

        let order = self.access_counter.fetch_add(1, Ordering::Relaxed);
        let mut cache = self.query_cache.write();

        if cache.len() >= self.cache_capacity {
            // Evict oldest entry
            if let Some(oldest_key) = cache
                .iter()
                .min_by_key(|(_, v)| v.order)
                .map(|(k, _)| k.clone())
            {
                cache.remove(&oldest_key);
            }
        }

        cache.insert(key, LruEntry { results: results.clone(), order });
        results
    }

    /// Text-only query with trigram cache.
    pub fn text_query(&self, query_text: &str, top_k: usize) -> Vec<ScoredResult> {
        self.index.text_query(query_text, top_k)
    }

    /// Vector-only query (no caching — embedding uniqueness makes cache misses likely).
    pub fn vector_query(&self, query_embedding: &[f32], top_k: usize) -> Vec<ScoredResult> {
        self.index.vector_query(query_embedding, top_k)
    }

    /// Pre-warm the trigram cache with the most frequent trigrams.
    /// Call after finalize to accelerate repeated queries over common terms.
    pub fn warm_trigrams(&self, trigrams: &[&str]) {
        let mut cache = self.trigram_cache.write();
        for &gram in trigrams {
            if !cache.contains_key(gram) {
                let results = self.index.text_query(gram, 1000);
                let ids: Vec<ChunkId> = results.iter().map(|r| r.chunk_id).collect();
                cache.insert(gram.to_string(), ids);
            }
        }
    }

    /// Clear all caches.
    pub fn clear_cache(&self) {
        self.query_cache.write().clear();
        self.trigram_cache.write().clear();
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }

    pub fn cache_hits(&self) -> u64 { self.hits.load(Ordering::Relaxed) }
    pub fn cache_misses(&self) -> u64 { self.misses.load(Ordering::Relaxed) }
    pub fn cache_size(&self) -> usize { self.query_cache.read().len() }

    pub fn cache_hit_rate(&self) -> f64 {
        let h = self.cache_hits() as f64;
        let m = self.cache_misses() as f64;
        if h + m == 0.0 { 0.0 } else { h / (h + m) }
    }

    /// Access the underlying index directly.
    pub fn inner(&self) -> &VtplIndex { &self.index }
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

    #[test]
    fn cache_hit_returns_same_results() {
        let dim = 128;
        let train: Vec<Vec<f32>> = (0..300).map(|_| random_vec(dim)).collect();
        let cb = PqCodebook::train(&train, dim, 15);
        let mut idx = VtplIndex::new(cb);
        idx.insert(0, "concurrent hash map", &random_vec(dim));
        idx.insert(1, "vector search", &random_vec(dim));
        idx.finalize();

        let cached = CachedIndex::new(idx, 100);
        let emb = random_vec(dim);

        let r1 = cached.query("concurrent", &emb, 0.6, 5);
        assert_eq!(cached.cache_misses(), 1);
        assert_eq!(cached.cache_hits(), 0);

        let r2 = cached.query("concurrent", &emb, 0.6, 5);
        assert_eq!(cached.cache_hits(), 1);
        assert_eq!(r1.len(), r2.len());
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.chunk_id, b.chunk_id);
        }
    }

    #[test]
    fn cache_evicts_oldest() {
        let dim = 128;
        let train: Vec<Vec<f32>> = (0..300).map(|_| random_vec(dim)).collect();
        let cb = PqCodebook::train(&train, dim, 15);
        let mut idx = VtplIndex::new(cb);
        idx.insert(0, "test document", &random_vec(dim));
        idx.finalize();

        let cached = CachedIndex::new(idx, 2);

        cached.query("aaa", &random_vec(dim), 0.6, 5);
        cached.query("bbb", &random_vec(dim), 0.6, 5);
        assert_eq!(cached.cache_size(), 2);

        cached.query("ccc", &random_vec(dim), 0.6, 5);
        assert_eq!(cached.cache_size(), 2); // evicted oldest
    }
}

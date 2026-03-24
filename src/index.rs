use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

use crate::ngram::trigrams;
use crate::posting::{ChunkId, PostingList, VtplEntry};
use crate::pq::PqCodebook;

#[derive(Debug, Clone)]
pub struct ScoredResult {
    pub chunk_id: ChunkId,
    pub score: f32,
    pub semantic_score: f32,
    pub pattern_score: f32,
}

/// The VTPL index: trigram → posting list with inline PQ codes.
#[derive(Serialize, Deserialize)]
pub struct VtplIndex {
    pub codebook: PqCodebook,
    postings: BTreeMap<String, PostingList>,
    num_chunks: u32,
    df: HashMap<String, u32>,
}

impl VtplIndex {
    pub fn new(codebook: PqCodebook) -> Self {
        Self {
            codebook,
            postings: BTreeMap::new(),
            num_chunks: 0,
            df: HashMap::new(),
        }
    }

    /// Insert a chunk. Embedding should be L2-normalized.
    pub fn insert(&mut self, chunk_id: ChunkId, text: &str, embedding: &[f32]) {
        let pq_code = self.codebook.encode(embedding);
        let grams = trigrams(text);

        for gram in &grams {
            self.postings
                .entry(gram.clone())
                .or_default()
                .push(VtplEntry::new(chunk_id, pq_code));

            *self.df.entry(gram.clone()).or_insert(0) += 1;
        }

        self.num_chunks += 1;
    }

    /// Sort posting lists for merge-intersection.
    pub fn finalize(&mut self) {
        for list in self.postings.values_mut() {
            list.sort();
        }
    }

    /// Fused query in a single pass over posting lists.
    ///
    /// `lambda` — weight for semantic: score = λ·cosine + (1-λ)·pattern
    pub fn query(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        lambda: f32,
        top_k: usize,
    ) -> Vec<ScoredResult> {
        let grams = trigrams(query_text);
        let dt = self.codebook.build_distance_table(query_embedding);
        let total_query_grams = grams.len() as f32;

        if total_query_grams == 0.0 {
            return Vec::new();
        }

        let mut accum: HashMap<ChunkId, (f32, f32, u32)> = HashMap::new();

        for gram in &grams {
            if let Some(list) = self.postings.get(gram.as_str()) {
                let idf = idf(self.num_chunks, self.df.get(gram.as_str()).copied().unwrap_or(1));
                for entry in &list.entries {
                    let acc = accum.entry(entry.chunk_id).or_insert_with(|| {
                        let sem = dt.approximate_cosine(&entry.pq_code);
                        (sem, 0.0, 0)
                    });
                    acc.1 += idf;
                    acc.2 += 1;
                }
            }
        }

        let mut results: Vec<ScoredResult> = accum
            .into_iter()
            .map(|(chunk_id, (semantic_score, idf_sum, hit_count))| {
                let pattern_score = (hit_count as f32 / total_query_grams) * (idf_sum / hit_count as f32);
                let score = lambda * semantic_score + (1.0 - lambda) * pattern_score;
                ScoredResult { chunk_id, score, semantic_score, pattern_score }
            })
            .collect();

        results.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(top_k);
        results
    }

    /// Text-only query (λ=0).
    pub fn text_query(&self, query_text: &str, top_k: usize) -> Vec<ScoredResult> {
        let grams = trigrams(query_text);
        let total_query_grams = grams.len() as f32;
        if total_query_grams == 0.0 {
            return Vec::new();
        }

        let mut accum: HashMap<ChunkId, (f32, u32)> = HashMap::new();
        for gram in &grams {
            if let Some(list) = self.postings.get(gram.as_str()) {
                let idf = idf(self.num_chunks, self.df.get(gram.as_str()).copied().unwrap_or(1));
                for entry in &list.entries {
                    let acc = accum.entry(entry.chunk_id).or_insert((0.0, 0));
                    acc.0 += idf;
                    acc.1 += 1;
                }
            }
        }

        let mut results: Vec<ScoredResult> = accum
            .into_iter()
            .map(|(chunk_id, (idf_sum, hit_count))| {
                let pattern_score = (hit_count as f32 / total_query_grams) * (idf_sum / hit_count as f32);
                ScoredResult { chunk_id, score: pattern_score, semantic_score: 0.0, pattern_score }
            })
            .collect();

        results.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(top_k);
        results
    }

    /// Vector-only query — scan all unique PQ codes.
    pub fn vector_query(&self, query_embedding: &[f32], top_k: usize) -> Vec<ScoredResult> {
        let dt = self.codebook.build_distance_table(query_embedding);
        let mut seen: HashMap<ChunkId, f32> = HashMap::new();

        for list in self.postings.values() {
            for entry in &list.entries {
                seen.entry(entry.chunk_id)
                    .or_insert_with(|| dt.approximate_cosine(&entry.pq_code));
            }
        }

        let mut results: Vec<ScoredResult> = seen
            .into_iter()
            .map(|(chunk_id, sem)| ScoredResult {
                chunk_id, score: sem, semantic_score: sem, pattern_score: 0.0,
            })
            .collect();

        results.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(top_k);
        results
    }

    pub fn num_postings(&self) -> usize { self.postings.len() }
    pub fn num_chunks(&self) -> u32 { self.num_chunks }
    pub fn total_entries(&self) -> usize { self.postings.values().map(|p| p.len()).sum() }
    pub fn pq_overhead_bytes(&self) -> usize { self.postings.values().map(|p| p.pq_overhead_bytes()).sum() }
}

impl VtplIndex {
    pub fn serialize(&self) -> Vec<u8> { bincode::serialize(self).expect("serialization failed") }
    pub fn deserialize(bytes: &[u8]) -> Self { bincode::deserialize(bytes).expect("deserialization failed") }
    pub fn save_to_file(&self, path: &str) -> std::io::Result<()> { std::fs::write(path, self.serialize()) }
    pub fn load_from_file(path: &str) -> std::io::Result<Self> { Ok(Self::deserialize(&std::fs::read(path)?)) }
}

/// BM25-style IDF: log((N - df + 0.5) / (df + 0.5) + 1)
#[inline]
fn idf(total_docs: u32, doc_freq: u32) -> f32 {
    let n = total_docs as f32;
    let df = doc_freq as f32;
    ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
}

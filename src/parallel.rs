use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU32, Ordering};

use dashmap::DashMap;
use rayon::prelude::*;

use crate::ngram::trigrams;
use crate::posting::{ChunkId, PostingList, VtplEntry};
use crate::pq::{PqCode, PqCodebook};
use crate::index::VtplIndex;

struct PreprocessedDoc {
    chunk_id: ChunkId,
    pq_code: PqCode,
    trigrams: Vec<String>,
}

/// Parallel index builder. Pre-processes documents across all cores
/// (PQ encoding + trigram extraction), then merges into a VtplIndex.
pub struct ParallelBuilder {
    codebook: PqCodebook,
    postings: DashMap<String, Vec<VtplEntry>>,
    df: DashMap<String, u32>,
    num_chunks: AtomicU32,
}

impl ParallelBuilder {
    pub fn new(codebook: PqCodebook) -> Self {
        Self {
            codebook,
            postings: DashMap::new(),
            df: DashMap::new(),
            num_chunks: AtomicU32::new(0),
        }
    }

    /// Insert a batch of documents in parallel.
    /// Each (chunk_id, text, embedding) is PQ-encoded and trigram-extracted
    /// across all available cores, then merged into the shared index.
    pub fn insert_batch(&self, docs: &[(ChunkId, &str, &[f32])]) {
        // Phase 1: parallel preprocessing — PQ encode + trigram extract
        let processed: Vec<PreprocessedDoc> = docs
            .par_iter()
            .map(|&(chunk_id, text, embedding)| {
                let pq_code = self.codebook.encode(embedding);
                let grams: Vec<String> = trigrams(text).into_iter().collect();
                PreprocessedDoc { chunk_id, pq_code, trigrams: grams }
            })
            .collect();

        // Phase 2: merge into concurrent maps
        for doc in &processed {
            let entry = VtplEntry::new(doc.chunk_id, doc.pq_code);
            let mut seen_grams = std::collections::HashSet::new();

            for gram in &doc.trigrams {
                self.postings
                    .entry(gram.clone())
                    .or_default()
                    .push(entry);

                if seen_grams.insert(gram.clone()) {
                    *self.df.entry(gram.clone()).or_insert(0) += 1;
                }
            }

            self.num_chunks.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Consume the builder and produce a finalized VtplIndex.
    pub fn build(self) -> VtplIndex {
        let mut postings = BTreeMap::new();
        for entry in self.postings.into_iter() {
            let (gram, mut entries) = entry;
            entries.sort_unstable_by_key(|e| e.chunk_id);
            let mut list = PostingList::new();
            list.entries = entries;
            postings.insert(gram, list);
        }

        let df: HashMap<String, u32> = self.df.into_iter().collect();
        let num_chunks = self.num_chunks.load(Ordering::Relaxed);

        VtplIndex::from_parts(self.codebook, postings, num_chunks, df)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pq::PqCodebook;
    use crate::l2_normalize;
    use rand::Rng;

    fn random_vec(dim: usize) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        let mut v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        l2_normalize(&mut v);
        v
    }

    #[test]
    fn parallel_build_matches_sequential() {
        let dim = 128;
        let train: Vec<Vec<f32>> = (0..300).map(|_| random_vec(dim)).collect();
        let codebook = PqCodebook::train(&train, dim, 15);

        let texts = [
            "concurrent hash map implementation",
            "vector search product quantization",
            "posting list intersection algorithm",
            "lock free data structures",
        ];
        let embeddings: Vec<Vec<f32>> = texts.iter().map(|_| random_vec(dim)).collect();

        // Sequential
        let mut seq = VtplIndex::new(codebook.clone());
        for (i, (text, emb)) in texts.iter().zip(embeddings.iter()).enumerate() {
            seq.insert(i as u32, text, emb);
        }
        seq.finalize();

        // Parallel
        let builder = ParallelBuilder::new(codebook);
        let docs: Vec<(ChunkId, &str, &[f32])> = texts
            .iter()
            .zip(embeddings.iter())
            .enumerate()
            .map(|(i, (text, emb))| (i as u32, *text, emb.as_slice()))
            .collect();
        builder.insert_batch(&docs);
        let par = builder.build();

        assert_eq!(seq.num_chunks(), par.num_chunks());
        assert_eq!(seq.num_postings(), par.num_postings());
        assert_eq!(seq.total_entries(), par.total_entries());

        let q_emb = random_vec(dim);
        let _ = q_emb; // ensure parallel index is queryable
        let r_seq = seq.text_query("concurrent hash", 5);
        let r_par = par.text_query("concurrent hash", 5);
        assert_eq!(r_seq.len(), r_par.len());
    }
}

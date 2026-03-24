use rand::Rng;
use vtpl::{l2_normalize, PqCodebook, VtplIndex};

fn random_vec(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    l2_normalize(&mut v);
    v
}

fn make_similar_vec(base: &[f32], noise: f32) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut v: Vec<f32> = base.iter().map(|&x| x + rng.gen_range(-noise..noise)).collect();
    l2_normalize(&mut v);
    v
}

fn build_test_index(dim: usize) -> (VtplIndex, Vec<Vec<f32>>) {
    let train_vecs: Vec<Vec<f32>> = (0..300).map(|_| random_vec(dim)).collect();
    let codebook = PqCodebook::train(&train_vecs, dim, 15);

    let texts = [
        "concurrent hash map implementation in rust",
        "vector search using product quantization",
        "posting list intersection algorithm",
        "inverted index with embedded vectors",
        "approximate nearest neighbor search",
        "concurrent programming patterns",
        "rust memory safety guarantees",
        "lock free data structures in concurrent systems",
    ];

    let embeddings: Vec<Vec<f32>> = texts.iter().map(|_| random_vec(dim)).collect();
    let mut index = VtplIndex::new(codebook);
    for (i, (text, emb)) in texts.iter().zip(embeddings.iter()).enumerate() {
        index.insert(i as u32, text, emb);
    }
    index.finalize();
    (index, embeddings)
}

#[test]
fn fused_query_returns_results() {
    let dim = 128;
    let (index, _) = build_test_index(dim);
    let results = index.query("concurrent hash", &random_vec(dim), 0.6, 5);
    assert!(!results.is_empty());
}

#[test]
fn text_query_ranks_exact_matches_high() {
    let dim = 128;
    let (index, _) = build_test_index(dim);
    let results = index.text_query("concurrent hash map", 5);
    assert!(!results.is_empty());
    assert_eq!(results[0].chunk_id, 0);
}

#[test]
fn vector_query_prefers_similar_embeddings() {
    let dim = 128;
    let train: Vec<Vec<f32>> = (0..300).map(|_| random_vec(dim)).collect();
    let cb = PqCodebook::train(&train, dim, 15);
    let mut index = VtplIndex::new(cb);

    let target = random_vec(dim);
    index.insert(0, "document alpha", &make_similar_vec(&target, 0.05));
    index.insert(1, "document beta", &random_vec(dim));
    index.finalize();

    let results = index.vector_query(&target, 2);
    assert_eq!(results[0].chunk_id, 0);
}

#[test]
fn lambda_zero_ignores_semantic() {
    let dim = 128;
    let (index, _) = build_test_index(dim);
    let results = index.query("concurrent", &random_vec(dim), 0.0, 5);
    for r in &results {
        assert_eq!(r.score, r.pattern_score);
    }
}

#[test]
fn lambda_one_ignores_pattern() {
    let dim = 128;
    let (index, _) = build_test_index(dim);
    let results = index.query("concurrent", &random_vec(dim), 1.0, 5);
    for r in &results {
        assert!((r.score - r.semantic_score).abs() < 1e-6);
    }
}

#[test]
fn serialize_roundtrip() {
    let dim = 128;
    let (index, _) = build_test_index(dim);
    let index2 = VtplIndex::deserialize(&index.serialize());

    let r1 = index.text_query("concurrent hash", 8);
    let r2 = index2.text_query("concurrent hash", 8);
    assert_eq!(r1.len(), r2.len());

    let mut ids1: Vec<u32> = r1.iter().map(|r| r.chunk_id).collect();
    let mut ids2: Vec<u32> = r2.iter().map(|r| r.chunk_id).collect();
    ids1.sort();
    ids2.sort();
    assert_eq!(ids1, ids2);
}

#[test]
fn empty_query_returns_nothing() {
    let dim = 128;
    let (index, _) = build_test_index(dim);
    assert!(index.query("", &random_vec(dim), 0.5, 5).is_empty());
}

#[test]
fn index_stats() {
    let dim = 128;
    let (index, _) = build_test_index(dim);
    assert_eq!(index.num_chunks(), 8);
    assert!(index.num_postings() > 0);
    assert!(index.total_entries() > 0);
    assert!(index.pq_overhead_bytes() > 0);
}

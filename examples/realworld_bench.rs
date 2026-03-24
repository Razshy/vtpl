use std::collections::HashMap;
use std::time::Instant;

use vtpl::{l2_normalize, CachedIndex, ParallelBuilder, PqCodebook, VtplIndex};

#[derive(serde::Deserialize)]
struct Dataset {
    n_docs: usize,
    n_queries: usize,
    dim: usize,
    texts: Vec<String>,
    embeddings: Vec<Vec<f32>>,
    #[allow(dead_code)]
    query_indices: Vec<usize>,
    query_texts: Vec<String>,
    query_embeddings: Vec<Vec<f32>>,
    ground_truth_top20: Vec<Vec<usize>>,
}

fn recall_at_k(predicted: &[u32], truth: &[usize], k: usize) -> f32 {
    let truth_set: std::collections::HashSet<u32> =
        truth.iter().take(k).map(|&i| i as u32).collect();
    let hits = predicted.iter().take(k).filter(|id| truth_set.contains(id)).count();
    hits as f32 / k as f32
}

fn ndcg_at_k(predicted: &[u32], truth: &[usize], k: usize) -> f32 {
    let relevance: HashMap<u32, f32> = truth
        .iter()
        .take(k)
        .enumerate()
        .map(|(rank, &doc_id)| (doc_id as u32, 1.0 / (rank as f32 + 1.0).log2().max(1.0)))
        .collect();

    let mut dcg = 0.0f32;
    for (i, &doc_id) in predicted.iter().take(k).enumerate() {
        if let Some(&rel) = relevance.get(&doc_id) {
            dcg += rel / (i as f32 + 2.0).log2();
        }
    }

    let mut idcg = 0.0f32;
    for i in 0..k.min(truth.len()) {
        let rel = 1.0 / (i as f32 + 1.0).log2().max(1.0);
        idcg += rel / (i as f32 + 2.0).log2();
    }

    if idcg == 0.0 { 0.0 } else { dcg / idcg }
}

fn run_benchmark(path: &str) {
    println!("Loading {}...", path);
    let raw = std::fs::read_to_string(path).expect("Failed to read dataset");
    let ds: Dataset = serde_json::from_str(&raw).expect("Failed to parse dataset");
    println!(
        "  {} docs, {} queries, {}-dim embeddings\n",
        ds.n_docs, ds.n_queries, ds.dim
    );

    let mut embeddings = ds.embeddings.clone();
    for emb in &mut embeddings {
        l2_normalize(emb);
    }
    let mut query_embeddings = ds.query_embeddings.clone();
    for emb in &mut query_embeddings {
        l2_normalize(emb);
    }

    // ─── Train PQ codebook ───
    let t0 = Instant::now();
    let codebook = PqCodebook::train(&embeddings, ds.dim, 25);
    println!("  PQ codebook trained in {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);

    // ─── Sequential build ───
    let t0 = Instant::now();
    let mut seq_index = VtplIndex::new(codebook.clone());
    for (i, (text, emb)) in ds.texts.iter().zip(embeddings.iter()).enumerate() {
        seq_index.insert(i as u32, text, emb);
    }
    seq_index.finalize();
    let seq_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  Sequential build: {:.1}ms", seq_ms);

    // ─── Parallel build ───
    let t0 = Instant::now();
    let builder = ParallelBuilder::new(codebook.clone());
    let batch: Vec<(u32, &str, &[f32])> = ds
        .texts
        .iter()
        .zip(embeddings.iter())
        .enumerate()
        .map(|(i, (text, emb))| (i as u32, text.as_str(), emb.as_slice()))
        .collect();
    builder.insert_batch(&batch);
    let par_index = builder.build();
    let par_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "  Parallel build:   {:.1}ms ({:.1}x speedup)\n",
        par_ms,
        seq_ms / par_ms
    );

    // ─── Query speed benchmarks ───
    println!("  --- Query speed (200 iterations, release mode) ---");
    let iterations = 200;

    let t0 = Instant::now();
    for i in 0..iterations {
        let qi = i % ds.n_queries;
        let _ = seq_index.query(
            &ds.query_texts[qi],
            &query_embeddings[qi],
            0.6,
            10,
        );
    }
    let fused_us = t0.elapsed().as_micros() as f64 / iterations as f64;

    let t0 = Instant::now();
    for i in 0..iterations {
        let qi = i % ds.n_queries;
        let _ = seq_index.text_query(&ds.query_texts[qi], 10);
    }
    let text_us = t0.elapsed().as_micros() as f64 / iterations as f64;

    let t0 = Instant::now();
    for i in 0..iterations {
        let qi = i % ds.n_queries;
        let _ = seq_index.vector_query(&query_embeddings[qi], 10);
    }
    let vec_us = t0.elapsed().as_micros() as f64 / iterations as f64;

    println!("    Fused:       {:.0} µs/query", fused_us);
    println!("    Text-only:   {:.0} µs/query", text_us);
    println!("    Vector-only: {:.0} µs/query", vec_us);

    // ─── Cache benchmark ───
    println!("\n  --- Smart cache performance ---");
    let cached = CachedIndex::with_defaults(par_index);

    // Warm-up pass: first iteration populates caches
    for (qi, qe) in query_embeddings.iter().enumerate().take(ds.n_queries) {
        let _ = cached.query(
            &ds.query_texts[qi],
            qe,
            0.6,
            10,
        );
    }
    let warm_stats = cached.stats();
    println!("    Warm-up:  {}", warm_stats);

    // Timed pass: subsequent iterations benefit from trigram/word/semantic cache
    let t0 = Instant::now();
    for i in 0..iterations {
        let qi = i % ds.n_queries;
        let _ = cached.query(
            &ds.query_texts[qi],
            &query_embeddings[qi],
            0.6,
            10,
        );
    }
    let cached_us = t0.elapsed().as_micros() as f64 / iterations as f64;
    let final_stats = cached.stats();
    println!("    Cached fused: {:.0} µs/query", cached_us);
    println!("    Speedup vs uncached: {:.1}x", fused_us / cached_us);
    println!("    Final:    {}", final_stats);
    println!(
        "    Cache sizes: {} trigrams, {} words, {} semantic",
        cached.trigram_cache_size(),
        cached.word_cache_size(),
        cached.semantic_cache_size()
    );

    // ─── PQ quality ───
    println!("\n  --- PQ compression quality (vs exact cosine) ---");
    let mut pq_recall_sum = 0.0f32;
    let mut pq_ndcg_sum = 0.0f32;

    for (qi, qe) in query_embeddings.iter().enumerate() {
        let results = seq_index.vector_query(qe, 10);
        let predicted: Vec<u32> = results.iter().map(|r| r.chunk_id).collect();
        pq_recall_sum += recall_at_k(&predicted, &ds.ground_truth_top20[qi], 10);
        pq_ndcg_sum += ndcg_at_k(&predicted, &ds.ground_truth_top20[qi], 10);
    }

    let pq_recall = pq_recall_sum / ds.n_queries as f32;
    let pq_ndcg = pq_ndcg_sum / ds.n_queries as f32;
    println!("    PQ Recall@10: {:.1}%", pq_recall * 100.0);
    println!("    PQ NDCG@10:   {:.1}%", pq_ndcg * 100.0);

    // ─── Fused vs vector-only vs text-only quality ───
    println!("\n  --- Retrieval quality (Recall@10 vs ground truth) ---");
    let mut fused_recall_sum = 0.0f32;
    let mut text_recall_sum = 0.0f32;
    let mut vec_recall_sum = 0.0f32;

    for (qi, qe) in query_embeddings.iter().enumerate() {
        let fused = seq_index.query(&ds.query_texts[qi], qe, 0.6, 10);
        let text = seq_index.text_query(&ds.query_texts[qi], 10);
        let vec_r = seq_index.vector_query(qe, 10);

        let f_ids: Vec<u32> = fused.iter().map(|r| r.chunk_id).collect();
        let t_ids: Vec<u32> = text.iter().map(|r| r.chunk_id).collect();
        let v_ids: Vec<u32> = vec_r.iter().map(|r| r.chunk_id).collect();

        fused_recall_sum += recall_at_k(&f_ids, &ds.ground_truth_top20[qi], 10);
        text_recall_sum += recall_at_k(&t_ids, &ds.ground_truth_top20[qi], 10);
        vec_recall_sum += recall_at_k(&v_ids, &ds.ground_truth_top20[qi], 10);
    }

    println!(
        "    Text-only:     {:.1}%",
        text_recall_sum / ds.n_queries as f32 * 100.0
    );
    println!(
        "    Fused (λ=0.6): {:.1}%",
        fused_recall_sum / ds.n_queries as f32 * 100.0
    );
    println!(
        "    Vector-only:   {:.1}%",
        vec_recall_sum / ds.n_queries as f32 * 100.0
    );

    // ─── Memory ───
    println!("\n  --- Memory ---");
    let serialized = seq_index.serialize();
    println!("    Serialized index: {:.2} MB", serialized.len() as f64 / 1_048_576.0);
    println!(
        "    PQ overhead: {:.2} MB ({} entries × 32 bytes)",
        seq_index.pq_overhead_bytes() as f64 / 1_048_576.0,
        seq_index.total_entries()
    );
    println!();
}

fn main() {
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  VTPL Real-World Benchmark (AG News + MiniLM-L6-v2) ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    if std::path::Path::new("data/embeddings_1k.json").exists() {
        run_benchmark("data/embeddings_1k.json");
    } else {
        println!("⚠ data/embeddings_1k.json not found — run: python scripts/generate_embeddings.py\n");
    }

    if std::path::Path::new("data/embeddings_10k.json").exists() {
        run_benchmark("data/embeddings_10k.json");
    } else {
        println!("⚠ data/embeddings_10k.json not found — run: python scripts/generate_embeddings.py\n");
    }
}

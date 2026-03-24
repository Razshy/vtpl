use rand::Rng;
use vtpl::{l2_normalize, CachedIndex, ParallelBuilder, PqCodebook, VtplIndex};

fn fake_embedding(dim: usize, seed_text: &str) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let bias: f32 = seed_text.len() as f32 * 0.01;
    let mut v: Vec<f32> = (0..dim).map(|i| {
        rng.gen_range(-1.0..1.0) + if i < seed_text.len() { bias } else { 0.0 }
    }).collect();
    l2_normalize(&mut v);
    v
}

fn main() {
    let dim = 128;

    println!("=== VTPL: Vector-Threaded Posting Lists ===\n");

    let train_vecs: Vec<Vec<f32>> = (0..300).map(|_| fake_embedding(dim, "train")).collect();
    let codebook = PqCodebook::train(&train_vecs, dim, 15);

    let documents = vec![
        (0, "concurrent hash map implementation in rust with lock-free reads"),
        (1, "vector search using product quantization for approximate nearest neighbors"),
        (2, "posting list intersection algorithm for inverted indexes"),
        (3, "building a concurrent B-tree index with optimistic locking"),
        (4, "rust async runtime implementation with work-stealing scheduler"),
        (5, "product quantization compresses high-dimensional vectors into compact codes"),
        (6, "inverted index with skip lists for fast posting list traversal"),
        (7, "concurrent programming patterns in modern systems languages"),
        (8, "approximate nearest neighbor search using locality-sensitive hashing"),
        (9, "the rust borrow checker ensures memory safety without garbage collection"),
    ];

    // --- Sequential build ---
    println!("--- Sequential build ---");
    let mut index = VtplIndex::new(codebook.clone());
    for &(id, text) in &documents {
        index.insert(id, text, &fake_embedding(dim, text));
    }
    index.finalize();
    println!("{} docs, {} posting lists, {} entries\n",
             index.num_chunks(), index.num_postings(), index.total_entries());

    // --- Parallel build ---
    println!("--- Parallel build (rayon + AtomicU32) ---");
    let builder = ParallelBuilder::new(codebook);
    let embeddings: Vec<Vec<f32>> = documents.iter().map(|&(_, t)| fake_embedding(dim, t)).collect();
    let batch: Vec<(u32, &str, &[f32])> = documents.iter()
        .zip(embeddings.iter())
        .map(|(&(id, text), emb)| (id, text, emb.as_slice()))
        .collect();
    builder.insert_batch(&batch);
    let par_index = builder.build();
    println!("{} docs, {} posting lists, {} entries\n",
             par_index.num_chunks(), par_index.num_postings(), par_index.total_entries());

    // --- Smart cached queries ---
    println!("--- Smart cached index ---");
    let cached = CachedIndex::with_defaults(par_index);

    let q_emb = fake_embedding(dim, "concurrent hash");

    println!("Query 1: \"concurrent hash\" (lambda=0.6, top 5)\n");
    let results = cached.query("concurrent hash", &q_emb, 0.6, 5);
    for r in &results {
        println!("  id={:>2}  score={:.4}  sem={:.4}  pat={:.4}  {}",
                 r.chunk_id, r.score, r.semantic_score, r.pattern_score,
                 documents[r.chunk_id as usize].1);
    }
    println!("\n  {}", cached.stats());

    // Overlapping query — shares "concurrent" trigrams from cache
    println!("\nQuery 2: \"concurrent programming\" (shares cached trigrams)\n");
    let results2 = cached.query("concurrent programming", &q_emb, 0.6, 5);
    for r in &results2 {
        println!("  id={:>2}  score={:.4}  sem={:.4}  pat={:.4}  {}",
                 r.chunk_id, r.score, r.semantic_score, r.pattern_score,
                 documents[r.chunk_id as usize].1);
    }
    println!("\n  {}", cached.stats());

    // Third query — "hash" trigrams also cached
    println!("\nQuery 3: \"hash table\" (reuses \"hash\" trigrams)\n");
    let results3 = cached.query("hash table", &q_emb, 0.6, 5);
    for r in &results3 {
        println!("  id={:>2}  score={:.4}  sem={:.4}  pat={:.4}  {}",
                 r.chunk_id, r.score, r.semantic_score, r.pattern_score,
                 documents[r.chunk_id as usize].1);
    }
    println!("\n  {}", cached.stats());

    println!("\nCache sizes: {} trigrams, {} words, {} semantic fingerprints",
             cached.trigram_cache_size(), cached.word_cache_size(), cached.semantic_cache_size());
    println!("Serialize: {} bytes", cached.inner().serialize().len());
    println!("=== Done ===");
}

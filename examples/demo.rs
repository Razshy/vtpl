use rand::Rng;
use vtpl::{l2_normalize, PqCodebook, VtplIndex};

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

    let mut index = VtplIndex::new(codebook);
    for &(id, text) in &documents {
        index.insert(id, text, &fake_embedding(dim, text));
    }
    index.finalize();

    println!("{} documents, {} posting lists, {} total entries\n",
             index.num_chunks(), index.num_postings(), index.total_entries());

    let query = "concurrent hash";
    let q_emb = fake_embedding(dim, query);

    println!("Query: \"{query}\" (lambda=0.6, top 5)\n");
    let results = index.query(query, &q_emb, 0.6, 5);
    for r in &results {
        println!("  id={:>2}  score={:.4}  sem={:.4}  pat={:.4}  {}",
                 r.chunk_id, r.score, r.semantic_score, r.pattern_score,
                 documents[r.chunk_id as usize].1);
    }

    println!("\nSerialize: {} bytes", index.serialize().len());
    println!("=== Done ===");
}

# VTPL — Vector-Threaded Posting Lists

Fused text + vector search in a single index pass. Parallel indexing, smart multi-level caching.

## The idea

Traditional hybrid search runs two separate lookups — a text index and a vector index — then merges results. VTPL embeds PQ-compressed vectors directly inside posting list entries, so text matching and semantic scoring happen in **one pass** with no second index traversal.

```
Traditional:  trigram "con" → [doc_1, doc_7, doc_42]
VTPL:         trigram "con" → [(doc_1, pq₁), (doc_7, pq₇), (doc_42, pq₄₂)]
                                        └── 32-byte PQ-compressed embedding
```

```
Score(chunk) = λ · PQ_cosine(q, chunk) + (1 - λ) · pattern_match_score(chunk)
```

## Benchmarks

All benchmarks on real data: AG News articles + all-MiniLM-L6-v2 embeddings (384-dim). Release mode.

### Indexing speed

| Documents | Sequential | Parallel (rayon) | Speedup |
|-----------|-----------|-----------------|---------|
| 1,000     | 54.8 ms   | 25.9 ms         | **2.1x** |
| 10,000    | 706 ms    | 352 ms          | **2.0x** |

### Query speed (200 iterations)

| Documents | Fused | Cached fused | Text-only | Vector-only |
|-----------|-------|-------------|-----------|-------------|
| 1,000     | 204 µs | **177 µs (1.2x)** | 180 µs | 715 µs |
| 10,000    | 1,902 µs | **1,816 µs** | 1,642 µs | 7,805 µs |

Fused is 3.5–4.3x faster than vector-only because it only scans posting lists matching the query trigrams instead of the full index.

### Smart cache hit rates (steady state)

| Layer | 1k docs | 10k docs |
|-------|---------|----------|
| Trigram cache | **95%** | **95%** |
| Word cache | **94%** | **95%** |
| Semantic cache | **91%** | **91%** |

The cache doesn't require identical queries — "concurrent hash" and "concurrent programming" share all "concurrent" trigrams. Similar embeddings (same quantization bucket) reuse per-chunk cosine scores.

### PQ compression quality

Ground truth = top-k by exact cosine similarity on the raw 384-dim embeddings.

| Documents | PQ Recall@10 | PQ NDCG@10 |
|-----------|-------------|------------|
| 1,000     | **76.5%**   | **88.3%**  |
| 10,000    | 74.5%       | 85.5%      |

PQ compression into 32 bytes preserves most ranking quality. At larger scale quantization noise grows — higher-dim embeddings or larger corpora may need a bigger PQ budget.

### Retrieval quality (Recall@10 vs ground truth)

| Method | 1k docs | 10k docs |
|--------|---------|----------|
| Text-only | 31.0% | 51.5% |
| **Fused (λ=0.6)** | **54.0%** | **64.5%** |
| Vector-only | 76.5% | 74.5% |

Fused beats text-only by +74% (1k) and +25% (10k). Vector-only still wins overall on this dataset — the text signal helps when embeddings miss lexical matches, but can add noise on pure-semantic queries. Fused is strongest when queries have **both** a meaningful text pattern and a semantic intent.

### Memory

| Documents | Serialized index | PQ overhead | Total entries |
|-----------|-----------------|-------------|---------------|
| 1,000     | 4.06 MB         | 3.11 MB     | 101,768       |
| 10,000    | 34.84 MB        | 30.39 MB    | 995,716       |

32 extra bytes per posting-list entry for the inline PQ code.

## When to use VTPL

VTPL is useful when queries have **both a text pattern and a semantic component**:

- **Code search:** function name match + semantically related implementations
- **Document retrieval:** keyword filter + meaning-based ranking
- **Log search:** pattern grep + "find similar errors"
- **E-commerce:** product name match + "things like this"
- **High-throughput workloads:** smart cache gives 95% trigram hit rate on overlapping queries

**Don't use VTPL for:**

- **Pure vector search** — use HNSW, it's O(log n) per query
- **Pure text search** — use a standard inverted index, no PQ overhead needed
- **Very large corpora** (millions of docs) with high-dim embeddings — the 32-byte PQ budget may not preserve enough ranking quality

## Usage

```rust
use vtpl::{PqCodebook, VtplIndex, ParallelBuilder, CachedIndex, CacheConfig, l2_normalize};

let codebook = PqCodebook::train(&training_vectors, dim, 25);

// --- Sequential build ---
let mut index = VtplIndex::new(codebook.clone());
for (id, text, embedding) in &chunks {
    index.insert(*id, text, embedding);
}
index.finalize();

// --- Parallel build (rayon + AtomicU32) ---
let builder = ParallelBuilder::new(codebook);
let batch: Vec<(u32, &str, &[f32])> = /* your docs */;
builder.insert_batch(&batch);
let index = builder.build();

// --- Smart cached queries ---
let cached = CachedIndex::with_defaults(index);

// Different queries sharing words reuse cached trigram scans
let r1 = cached.query("concurrent hash map", &query_embedding, 0.6, 10);
let r2 = cached.query("concurrent programming", &query_embedding, 0.6, 10);
// ^ shares all "concurrent" trigrams from cache

println!("{}", cached.stats());
// trigram 95% | word 94% | semantic 91%

// Text-only / vector-only also available
let text_results = cached.inner().text_query("posting list", 10);
let vec_results  = cached.inner().vector_query(&query_embedding, 10);

// Persist to disk
cached.inner().save_to_file("index.vtpl")?;
let loaded = VtplIndex::load_from_file("index.vtpl")?;
```

## Architecture

```
src/
├── lib.rs        — public API
├── pq.rs         — product quantization: codebook training, encode, asymmetric distance tables
├── ngram.rs      — character trigram extraction
├── posting.rs    — VtplEntry (chunk_id + 32-byte PQ code), PostingList
├── index.rs      — VtplIndex: insert, finalize, query, serialization
├── parallel.rs   — ParallelBuilder: rayon + DashMap + AtomicU32 for multi-core indexing
└── cache.rs      — CachedIndex: trigram/word/semantic 3-level cache with confidence eviction
```

- `VtplEntry` is `#[repr(C)]` / 36 bytes — cache-friendly sequential scans
- Asymmetric distance tables: 32 table lookups per candidate, zero float multiplies at query time
- Cosine computed once per chunk on first posting-list encounter
- IDF weighting: `log((N - df + 0.5) / (df + 0.5) + 1)`
- **Parallel indexing:** PQ encode + trigram extract across all cores via `rayon`, merge into `DashMap` with `AtomicU32` counters
- **Smart cache (3 levels):**
  - **Word cache** — "concurrent" always decomposes to the same trigrams; reused across queries
  - **Trigram cache** — posting list scan results cached per trigram; overlapping queries share work
  - **Semantic cache** — embeddings quantized into fingerprints; similar vectors reuse cosine scores
  - Confidence-based eviction: frequently-accessed entries survive longer
  - `Arc`-backed entries: cache hits are near-zero-cost pointer clones
  - `parking_lot::RwLock` with amortized confidence bumps to minimize write contention

## Running

```bash
cargo test                           # 18 tests
cargo run --example demo --release   # parallel build + smart cache demo
cargo bench                          # criterion benchmarks

# Real-world benchmark (requires Python + sentence-transformers + datasets)
python3 scripts/generate_embeddings.py
cargo run --example realworld_bench --release
```

## License

MIT

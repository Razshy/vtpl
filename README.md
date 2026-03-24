# VTPL — Vector-Threaded Posting Lists

Fused text + vector search in a single index pass.

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

### Speed

Measured against a traditional two-index approach (separate trigram index + `HashMap<ChunkId, PqCode>` vector store with O(1) lookup per candidate + merge). Same PQ codebook, same scoring formula. Release mode, 200 iterations.

| Documents | VTPL fused | Traditional hybrid | Speedup |
|-----------|------------|--------------------|---------|
| 1,000     | 60 µs      | 87 µs              | 1.45x   |
| 5,000     | 245 µs     | 439 µs             | 1.79x   |
| 10,000    | 493 µs     | 817 µs             | 1.66x   |
| 25,000    | 1.47 ms    | 2.38 ms            | 1.63x   |
| 50,000    | 2.96 ms    | 4.86 ms            | 1.64x   |

VTPL and traditional return **identical results** — same top-k, same scores. The speedup comes from eliminating the intermediate candidate set construction and second-pass vector lookups.

### PQ compression quality

Tested with real embeddings (all-MiniLM-L6-v2, 384-dim) on AG News articles. Ground truth = top-k by exact cosine similarity.

| Documents | PQ Recall@10 | PQ NDCG@10 |
|-----------|-------------|------------|
| 50        | 100%        | 100%       |
| 1,000     | 95.0%       | 96.1%      |
| 10,000    | 79.5%       | 84.1%      |

PQ compression into 32 bytes works well at small-to-medium scale. At 10k+ docs with 384-dim embeddings, quantization noise costs ~20% recall. Higher-dimensional embeddings or larger corpora would need a bigger PQ budget.

### Fused vs vector-only quality

On real AG News embeddings (1k docs, 20 queries, Recall@10):

| Method       | Recall@10 |
|-------------|-----------|
| Text-only   | 0.33      |
| Fused (λ=0.6) | 0.51   |
| Vector-only | **0.95**  |

Fused beats text-only by +57%, but vector-only still wins overall on this dataset. The text signal helps when embeddings miss lexical matches, but can also add noise. Fused is strongest when queries have **both** a meaningful text pattern and a semantic intent — e.g., searching for a specific function name while also wanting semantically related code.

### Memory overhead

32 extra bytes per posting-list entry. At 50k docs with ~2M total entries: ~60 MB overhead.

## When to use VTPL

VTPL is useful when queries have **both a text pattern and a semantic component**:

- **Code search:** function name match + semantically related implementations
- **Document retrieval:** keyword filter + meaning-based ranking
- **Log search:** pattern grep + "find similar errors"
- **E-commerce:** product name match + "things like this"

**Don't use VTPL for:**

- **Pure vector search** — use HNSW, it's O(log n) per query
- **Pure text search** — use a standard inverted index, no PQ overhead needed
- **Very large corpora** (millions of docs) with high-dim embeddings — the 32-byte PQ budget may not preserve enough ranking quality

## Usage

```rust
use vtpl::{PqCodebook, VtplIndex, l2_normalize};

let codebook = PqCodebook::train(&training_vectors, dim, 25);

let mut index = VtplIndex::new(codebook);
for (id, text, embedding) in chunks {
    index.insert(id, &text, &embedding);
}
index.finalize();

// Fused query: text + vector in one pass
let results = index.query("concurrent hash map", &query_embedding, 0.6, 10);

// Text-only / vector-only also available
let text_results = index.text_query("posting list", 10);
let vec_results  = index.vector_query(&query_embedding, 10);

// Persist to disk
index.save_to_file("index.vtpl")?;
let loaded = VtplIndex::load_from_file("index.vtpl")?;
```

## Architecture

```
src/
├── lib.rs      — public API
├── pq.rs       — product quantization: codebook training, encode, asymmetric distance tables
├── ngram.rs    — character trigram extraction
├── posting.rs  — VtplEntry (chunk_id + 32-byte PQ code), PostingList
└── index.rs    — VtplIndex: insert, finalize, query, serialization
```

- `VtplEntry` is `#[repr(C)]` / 36 bytes — cache-friendly sequential scans
- Asymmetric distance tables: 32 table lookups per candidate, zero float multiplies at query time
- Cosine computed once per chunk on first posting-list encounter
- IDF weighting: `log((N - df + 0.5) / (df + 0.5) + 1)`

## Running

```bash
cargo test
cargo run --example demo
cargo bench
```

## License

MIT

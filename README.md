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

Measured against a traditional two-index approach (separate trigram index + `HashMap<ChunkId, PqCode>` vector store with O(1) lookup per candidate + merge). Same PQ codebook, same scoring. Release mode, 200 iterations per measurement.

| Documents | VTPL fused | Traditional hybrid | Speedup |
|-----------|------------|--------------------|---------|
| 1,000     | 60 µs      | 87 µs              | 1.45x   |
| 5,000     | 245 µs     | 439 µs             | 1.79x   |
| 10,000    | 493 µs     | 817 µs             | 1.66x   |
| 25,000    | 1.47 ms    | 2.38 ms            | 1.63x   |
| 50,000    | 2.96 ms    | 4.86 ms            | 1.64x   |

The traditional approach pays for building a candidate HashSet from text results, then doing a second pass of HashMap lookups for each candidate's vector, then merging. VTPL reads the PQ code inline during the posting list scan and scores everything in one pass.

**Memory cost:** 32 extra bytes per posting-list entry (the PQ code). At 50k docs with ~2M total entries, that's ~60 MB overhead.

## When to use VTPL

VTPL is useful when your queries have **both a text pattern and a semantic component** — which is most real search traffic:

- **Code search:** user types a function name + wants semantically related implementations
- **Document retrieval:** keyword filter + meaning-based ranking
- **Log search:** grep-like pattern + "find similar errors"
- **E-commerce:** product name match + "things like this"

For **pure vector queries** with no text signal, use a dedicated ANN index (HNSW). For **pure text queries** with no embeddings, use a standard inverted index. VTPL targets the intersection — hybrid queries where you'd otherwise run two indexes and merge.

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

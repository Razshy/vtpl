# VTPL — Vector-Threaded Posting Lists

Fused text + vector search in a single index pass.

## The idea

Traditional hybrid search runs two separate lookups — a text index and a vector index — then merges results. VTPL embeds quantized vectors directly inside posting list entries, eliminating the second traversal.

```
Traditional:  trigram "con" → [doc_1, doc_7, doc_42]
VTPL:         trigram "con" → [(doc_1, pq₁), (doc_7, pq₇), (doc_42, pq₄₂)]
                                        └── 32-byte PQ-compressed embedding
```

At query time, text matching and semantic scoring happen in one pass:

```
Score(chunk) = λ · PQ_cosine(q, chunk) + (1 - λ) · pattern_match_score(chunk)
```

## Usage

```rust
use vtpl::{PqCodebook, VtplIndex, l2_normalize};

let codebook = PqCodebook::train(&training_vectors, dim, 25);

let mut index = VtplIndex::new(codebook);
for (id, text, embedding) in chunks {
    index.insert(id, &text, &embedding);
}
index.finalize();

// Fused query
let results = index.query("concurrent hash map", &query_embedding, 0.6, 10);

// Text-only / vector-only
let text_results = index.text_query("posting list", 10);
let vec_results  = index.vector_query(&query_embedding, 10);

// Persist
index.save_to_file("index.vtpl")?;
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
- Asymmetric distance tables: 32 table lookups per candidate, no float multiplies at query time
- Cosine computed once per chunk (first posting-list encounter)
- IDF weighting: `log((N - df + 0.5) / (df + 0.5) + 1)`

## Running

```bash
cargo test
cargo run --example demo
cargo bench
```

## License

MIT

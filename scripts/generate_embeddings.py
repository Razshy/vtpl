"""
Generate real embeddings for VTPL benchmarking at 1k and 10k scale.
Uses AG News dataset + all-MiniLM-L6-v2 (384-dim).
Produces JSON files with texts, embeddings, and ground-truth nearest neighbors.
"""
import json
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

def generate(n_docs, n_queries, output_path):
    print(f"\n=== Generating {n_docs} docs, {n_queries} queries → {output_path} ===")

    ds = load_dataset("ag_news", split="train")
    texts = [row["text"] for row in ds.select(range(n_docs))]

    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"  Encoding {n_docs} documents...")
    doc_embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    query_indices = np.random.RandomState(42).choice(n_docs, n_queries, replace=False).tolist()
    query_texts = [texts[i] for i in query_indices]
    query_embs = doc_embs[query_indices]

    print("  Computing ground truth (exact cosine top-20)...")
    ground_truth = []
    for qi, qe in enumerate(query_embs):
        sims = [cosine_sim(qe, doc_embs[j]) for j in range(n_docs)]
        ranked = sorted(range(n_docs), key=lambda j: sims[j], reverse=True)
        ground_truth.append(ranked[:20])

    data = {
        "n_docs": n_docs,
        "n_queries": n_queries,
        "dim": int(doc_embs.shape[1]),
        "texts": texts,
        "embeddings": doc_embs.tolist(),
        "query_indices": query_indices,
        "query_texts": query_texts,
        "query_embeddings": query_embs.tolist(),
        "ground_truth_top20": ground_truth,
    }

    with open(output_path, "w") as f:
        json.dump(data, f)
    print(f"  Saved to {output_path}")

if __name__ == "__main__":
    generate(1000, 20, "data/embeddings_1k.json")
    generate(10000, 20, "data/embeddings_10k.json")
    print("\nDone.")

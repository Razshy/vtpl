use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::Rng;
use vtpl::{l2_normalize, PqCodebook, VtplIndex};

fn random_vec(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    l2_normalize(&mut v);
    v
}

const TEXTS: &[&str] = &[
    "concurrent hash map implementation in rust with lock-free reads",
    "vector search using product quantization for approximate nearest neighbors",
    "posting list intersection algorithm for inverted indexes",
    "building a concurrent B-tree index with optimistic locking",
    "rust async runtime implementation with work-stealing scheduler",
];

fn bench_query(c: &mut Criterion) {
    let dim = 128;
    let mut group = c.benchmark_group("vtpl_query");

    for n_docs in [1_000, 5_000, 10_000] {
        let train: Vec<Vec<f32>> = (0..500).map(|_| random_vec(dim)).collect();
        let cb = PqCodebook::train(&train, dim, 10);
        let mut idx = VtplIndex::new(cb);
        for i in 0..n_docs {
            idx.insert(i as u32, TEXTS[i % TEXTS.len()], &random_vec(dim));
        }
        idx.finalize();
        let q = random_vec(dim);

        group.bench_with_input(BenchmarkId::new("fused", n_docs), &(&idx, &q), |b, (idx, q)| {
            b.iter(|| black_box(idx.query("concurrent hash", q, 0.6, 10)))
        });

        group.bench_with_input(BenchmarkId::new("text_only", n_docs), &idx, |b, idx| {
            b.iter(|| black_box(idx.text_query("concurrent hash", 10)))
        });

        group.bench_with_input(BenchmarkId::new("vector_only", n_docs), &(&idx, &q), |b, (idx, q)| {
            b.iter(|| black_box(idx.vector_query(q, 10)))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_query);
criterion_main!(benches);

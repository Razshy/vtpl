use std::ops::Deref;

use serde::{Deserialize, Serialize};

pub const PQ_BYTES: usize = 32;
pub const NUM_SUBSPACES: usize = PQ_BYTES;
pub const CENTROIDS_PER_SUBSPACE: usize = 256;

pub type PqCode = [u8; PQ_BYTES];

/// A product-quantization codebook: NUM_SUBSPACES sub-quantizers,
/// each with 256 centroids of dimension `sub_dim`.
#[derive(Clone, Serialize, Deserialize)]
pub struct PqCodebook {
    pub full_dim: usize,
    pub sub_dim: usize,
    /// Shape: [NUM_SUBSPACES][CENTROIDS_PER_SUBSPACE][sub_dim]
    pub centroids: Vec<Vec<Vec<f32>>>,
}

impl PqCodebook {
    /// Train a PQ codebook from a set of vectors using k-means per subspace.
    pub fn train(vectors: &[Vec<f32>], full_dim: usize, max_iters: usize) -> Self {
        assert!(!vectors.is_empty(), "need at least one training vector");
        assert_eq!(
            full_dim % NUM_SUBSPACES,
            0,
            "full_dim must be divisible by {NUM_SUBSPACES}"
        );
        let sub_dim = full_dim / NUM_SUBSPACES;

        let centroids: Vec<Vec<Vec<f32>>> = (0..NUM_SUBSPACES)
            .map(|m| {
                let start = m * sub_dim;
                let slices: Vec<&[f32]> = vectors.iter().map(|v| &v[start..start + sub_dim]).collect();
                kmeans(&slices, CENTROIDS_PER_SUBSPACE, sub_dim, max_iters)
            })
            .collect();

        Self {
            full_dim,
            sub_dim,
            centroids,
        }
    }

    /// Encode a full-dimensional vector into a PQ code (NUM_SUBSPACES bytes).
    pub fn encode(&self, vector: &[f32]) -> PqCode {
        debug_assert_eq!(vector.len(), self.full_dim);
        let mut code = [0u8; PQ_BYTES];
        for (m, byte) in code.iter_mut().enumerate() {
            let start = m * self.sub_dim;
            let sub = &vector[start..start + self.sub_dim];
            *byte = nearest_centroid(sub, &self.centroids[m]) as u8;
        }
        code
    }

    /// Build an asymmetric distance table (ADT) for a query vector.
    /// Returns [NUM_SUBSPACES][256] table where entry [m][c] is the
    /// dot-product between the query's m-th subvector and centroid c of subspace m.
    pub fn build_distance_table(&self, query: &[f32]) -> DistanceTable {
        debug_assert_eq!(query.len(), self.full_dim);
        let mut table = [[0.0f32; CENTROIDS_PER_SUBSPACE]; NUM_SUBSPACES];
        for (m, row) in table.iter_mut().enumerate() {
            let start = m * self.sub_dim;
            let q_sub = &query[start..start + self.sub_dim];
            let n_centroids = self.centroids[m].len();
            for (c, cell) in row.iter_mut().take(n_centroids).enumerate() {
                *cell = dot(q_sub, &self.centroids[m][c]);
            }
        }
        DistanceTable(table)
    }
}

/// Pre-computed distance table for fast PQ code → similarity lookup.
pub struct DistanceTable(pub [[f32; CENTROIDS_PER_SUBSPACE]; NUM_SUBSPACES]);

impl DistanceTable {
    /// Approximate dot-product between the query and a PQ-encoded vector.
    /// Uses table lookups — no float multiplies at query time.
    #[inline]
    pub fn approximate_dot(&self, code: &PqCode) -> f32 {
        let mut sum = 0.0f32;
        for (m, &byte) in code.iter().enumerate() {
            sum += self.0[m][byte as usize];
        }
        sum
    }

    /// Approximate cosine similarity.  Assumes the query and original vectors
    /// were L2-normalized before encoding.
    #[inline]
    pub fn approximate_cosine(&self, code: &PqCode) -> f32 {
        self.approximate_dot(code)
    }
}

// ── k-means ────────────────────────────────────────────────────────

fn kmeans(data: &[&[f32]], k: usize, dim: usize, max_iters: usize) -> Vec<Vec<f32>> {
    let k = k.min(data.len());
    let mut centroids: Vec<Vec<f32>> = data.iter().take(k).map(|s| s.to_vec()).collect();
    let mut assignments = vec![0usize; data.len()];

    for _ in 0..max_iters {
        let mut changed = false;
        for (i, point) in data.iter().enumerate() {
            let best = nearest_centroid(point, &centroids);
            if best != assignments[i] {
                assignments[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }

        let mut sums = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, point) in data.iter().enumerate() {
            let c = assignments[i];
            counts[c] += 1;
            for (j, &val) in point.iter().enumerate() {
                sums[c][j] += val;
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                let cnt = counts[c] as f32;
                for j in 0..dim {
                    centroids[c][j] = sums[c][j] / cnt;
                }
            }
        }
    }

    centroids
}

fn nearest_centroid(point: &[f32], centroids: &[Vec<f32>]) -> usize {
    centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, sq_dist(point, c.deref())))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
        .0
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline]
fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// L2-normalize a vector in place.
pub fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn random_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut rng = rand::thread_rng();
        (0..n)
            .map(|_| {
                let mut v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
                l2_normalize(&mut v);
                v
            })
            .collect()
    }

    #[test]
    fn encode_decode_roundtrip_preserves_similarity() {
        let dim = 128;
        let vecs = random_vectors(500, dim);
        let cb = PqCodebook::train(&vecs, dim, 20);

        let a = &vecs[0];
        let b = &vecs[1];

        let exact = dot(a, b);
        let dt = cb.build_distance_table(a);
        let code_b = cb.encode(b);
        let approx = dt.approximate_cosine(&code_b);

        let error = (exact - approx).abs();
        assert!(
            error < 0.35,
            "PQ approximation error {error:.4} too large (exact={exact:.4}, approx={approx:.4})"
        );
    }
}

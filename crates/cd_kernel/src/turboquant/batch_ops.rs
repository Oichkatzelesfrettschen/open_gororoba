//! Batch operations for TurboQuant: amortize per-vector overhead.
//!
//! Instead of processing vectors one at a time, batch operations
//! process N vectors simultaneously for better cache utilization
//! and SIMD throughput.
//!
//! # Key batch operations
//!
//! 1. Batch rotation: rotate N vectors with shared rotation matrix
//! 2. Batch quantize: quantize N rotated vectors in one pass
//! 3. Batch norm: compute N vector norms simultaneously (simsimd)

use super::rotation::Rotation;

/// Batch rotate N vectors using the same rotation.
///
/// For Haar: this could be a matrix multiply X @ Pi^T via BLAS.
/// For WHT: apply butterfly to each vector (already O(d log d) each).
/// For E8/F4: apply block rotation to each vector.
///
/// Returns flattened (N * d) rotated coordinates.
pub fn batch_rotate(
    vectors: &[Vec<f64>],
    rotation: &Rotation,
) -> Vec<Vec<f64>> {
    let d = rotation.dim();
    let n = vectors.len();

    // Pre-allocate all output at once
    let mut results = Vec::with_capacity(n);
    let mut buf = vec![0.0f64; d]; // shared scratch (WHT needs this)

    for v in vectors {
        let mut out = vec![0.0f64; d];
        rotation.forward(v, &mut buf, &mut out);
        results.push(out);
    }

    results
}

/// Batch rotate with rayon parallelism.
///
/// Each thread gets its own scratch buffer.
pub fn batch_rotate_parallel(
    vectors: &[Vec<f64>],
    rotation: &Rotation,
) -> Vec<Vec<f64>> {
    use rayon::prelude::*;
    let d = rotation.dim();

    vectors.par_iter().map(|v| {
        let mut buf = vec![0.0f64; d];
        let mut out = vec![0.0f64; d];
        rotation.forward(v, &mut buf, &mut out);
        out
    }).collect()
}

/// Batch quantize: boundary-search quantize N pre-rotated vectors.
///
/// Uses the SIMD codebook for maximum throughput.
pub fn batch_quantize(
    rotated_vectors: &[Vec<f64>],
    boundaries: &[f32],
    bits: u32,
) -> Vec<Vec<u8>> {
    use super::simd_codebook::SimdBoundaries;
    let simd = SimdBoundaries::from_boundaries(boundaries, bits);

    rotated_vectors.iter().map(|v| {
        let v_f32: Vec<f32> = v.iter().map(|&x| x as f32).collect();
        let mut indices = vec![0u8; v.len()];
        simd.quantize_batch(&v_f32, &mut indices);
        indices
    }).collect()
}

/// Batch compute vector norms via simsimd.
pub fn batch_norms(vectors: &[Vec<f64>]) -> Vec<f64> {
    vectors.iter().map(|v| {
        super::simsimd_bridge::dot_f64(v, v).sqrt()
    }).collect()
}

/// Batch normalize vectors to unit norm (in-place).
pub fn batch_normalize(vectors: &mut [Vec<f64>]) -> Vec<f64> {
    let norms: Vec<f64> = vectors.iter().map(|v| {
        v.iter().map(|x| x * x).sum::<f64>().sqrt()
    }).collect();

    for (v, &norm) in vectors.iter_mut().zip(norms.iter()) {
        if norm > 1e-15 {
            let inv = 1.0 / norm;
            for x in v.iter_mut() {
                *x *= inv;
            }
        }
    }

    norms
}

/// Complete batch pipeline: normalize -> rotate -> quantize.
///
/// Returns (indices, norms) for each vector.
pub fn batch_pipeline(
    vectors: &mut [Vec<f64>],
    rotation: &Rotation,
    boundaries: &[f32],
    bits: u32,
) -> (Vec<Vec<u8>>, Vec<f64>) {
    // Step 1: batch normalize
    let norms = batch_normalize(vectors);

    // Step 2: batch rotate (parallel)
    let rotated = batch_rotate_parallel(vectors, rotation);

    // Step 3: batch quantize (SIMD)
    let indices = batch_quantize(&rotated, boundaries, bits);

    (indices, norms)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lloyd_max;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use rand_distr::{Distribution, StandardNormal};

    fn random_vectors(n: usize, d: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let normal = StandardNormal;
        (0..n).map(|_| (0..d).map(|_| normal.sample(&mut rng)).collect()).collect()
    }

    #[test]
    fn test_batch_rotate_matches_individual() {
        let d = 64;
        let n = 20;
        let rotation = Rotation::new_fast_jl(d, 42);
        let vectors = random_vectors(n, d, 42);

        // Batch
        let batch_result = batch_rotate(&vectors, &rotation);

        // Individual
        let mut buf = vec![0.0f64; d];
        let individual_result: Vec<Vec<f64>> = vectors.iter().map(|v| {
            let mut out = vec![0.0f64; d];
            rotation.forward(v, &mut buf, &mut out);
            out
        }).collect();

        for (b, i) in batch_result.iter().zip(individual_result.iter()) {
            for (bv, iv) in b.iter().zip(i.iter()) {
                assert!((bv - iv).abs() < 1e-10, "Batch/individual mismatch");
            }
        }
    }

    #[test]
    fn test_batch_rotate_parallel_matches() {
        let d = 128;
        let n = 100;
        let rotation = Rotation::new_fast_jl(d, 42);
        let vectors = random_vectors(n, d, 42);

        let serial = batch_rotate(&vectors, &rotation);
        let parallel = batch_rotate_parallel(&vectors, &rotation);

        for (s, p) in serial.iter().zip(parallel.iter()) {
            for (sv, pv) in s.iter().zip(p.iter()) {
                assert!((sv - pv).abs() < 1e-10, "Serial/parallel mismatch");
            }
        }
    }

    #[test]
    fn test_batch_pipeline() {
        let d = 64;
        let n = 50;
        let bits = 3;
        let mut vectors = random_vectors(n, d, 42);
        let rotation = Rotation::new_fast_jl(d, 42);
        let codebook = lloyd_max::get_codebook(d, bits);

        let (indices, norms) = batch_pipeline(&mut vectors, &rotation, &codebook.boundaries, bits);

        assert_eq!(indices.len(), n);
        assert_eq!(norms.len(), n);
        for idx in &indices {
            assert_eq!(idx.len(), d);
            assert!(idx.iter().all(|&i| i < (1 << bits)));
        }
        assert!(norms.iter().all(|&n| n > 0.0));
    }

    #[test]
    fn test_batch_norms() {
        let vectors = vec![
            vec![3.0, 4.0],    // norm = 5
            vec![1.0, 0.0],    // norm = 1
            vec![0.0, 0.0, 1.0], // norm = 1
        ];
        let norms = batch_norms(&vectors);
        assert!((norms[0] - 5.0).abs() < 1e-10);
        assert!((norms[1] - 1.0).abs() < 1e-10);
        assert!((norms[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_batch_throughput() {
        use std::time::Instant;
        let d = 128;
        let n = 5000;
        let bits = 3;
        let mut vectors = random_vectors(n, d, 42);
        let rotation = Rotation::new_fast_jl(d, 42);
        let codebook = lloyd_max::get_codebook(d, bits);

        let t0 = Instant::now();
        let (indices, _norms) = batch_pipeline(&mut vectors, &rotation, &codebook.boundaries, bits);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;

        println!("Batch pipeline: {} vectors in {:.1} ms ({:.0} kvec/s)",
            n, ms, n as f64 / ms);
        assert_eq!(indices.len(), n);
    }
}

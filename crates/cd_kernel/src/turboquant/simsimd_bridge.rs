//! simsimd bridge: hardware-accelerated distance and similarity functions.
//!
//! Wraps simsimd's SIMD-optimized L2, cosine, and dot product functions
//! for use in TurboQuant and spatial analytics.
//!
//! simsimd uses hardware-specific kernels (AVX2, AVX-512, NEON) selected
//! at runtime, providing maximum throughput without manual SIMD code.
//!
//! # Use cases in open_gororoba
//!
//! 1. QJL inner product estimation (dot product of projected query and keys)
//! 2. CD fidelity cosine similarity measurement
//! 3. Spatial cone-search dot product screening
//! 4. TurboQuant MSE computation (L2 distance between original and reconstructed)

/// Compute dot product of two f64 slices via simsimd.
///
/// Falls back to scalar if simsimd is not available.
pub fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    simsimd::SpatialSimilarity::dot(a, b)
        .unwrap_or_else(|| a.iter().zip(b.iter()).map(|(x, y)| x * y).sum())
}

/// Compute dot product of two f32 slices via simsimd.
pub fn dot_f32(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    simsimd::SpatialSimilarity::dot(a, b)
        .unwrap_or_else(|| a.iter().zip(b.iter()).map(|(&x, &y)| x as f64 * y as f64).sum())
}

/// Compute L2 squared distance via simsimd.
pub fn l2sq_f64(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    simsimd::SpatialSimilarity::sqeuclidean(a, b)
        .unwrap_or_else(|| a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum())
}

/// Compute cosine distance via simsimd (1 - cosine_similarity).
pub fn cosine_distance_f64(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    simsimd::SpatialSimilarity::cosine(a, b)
        .unwrap_or_else(|| {
            let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
            let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
            if na < 1e-15 || nb < 1e-15 { 1.0 } else { 1.0 - dot / (na * nb) }
        })
}

/// Cosine similarity (1 - cosine_distance).
pub fn cosine_similarity_f64(a: &[f64], b: &[f64]) -> f64 {
    1.0 - cosine_distance_f64(a, b)
}

/// Compute MSE between original and reconstructed vectors.
///
/// MSE = L2_sq / dimension.  Uses simsimd for the L2 computation.
pub fn mse_f64(original: &[f64], reconstructed: &[f64]) -> f64 {
    let d = original.len();
    l2sq_f64(original, reconstructed) / d as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_f64() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let result = dot_f64(&a, &b);
        let expected: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!(
            (result - expected).abs() < 1e-6,
            "simsimd dot: {} vs scalar: {}", result, expected
        );
    }

    #[test]
    fn test_l2sq_f64() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = l2sq_f64(&a, &b);
        let expected: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        assert!(
            (result - expected).abs() < 1e-6,
            "simsimd L2sq: {} vs scalar: {}", result, expected
        );
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let cos = cosine_similarity_f64(&a, &b);
        assert!(cos.abs() < 0.01, "Orthogonal vectors should have cos~0, got {}", cos);

        let c = vec![1.0, 0.0, 0.0];
        let cos_same = cosine_similarity_f64(&a, &c);
        assert!((cos_same - 1.0).abs() < 0.01, "Same direction should have cos~1, got {}", cos_same);
    }

    #[test]
    fn test_mse() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.1, 2.1, 3.1, 4.1];
        let mse = mse_f64(&a, &b);
        let expected = 4.0 * 0.01 / 4.0; // 4 * 0.1^2 / 4
        assert!(
            (mse - expected).abs() < 1e-6,
            "MSE: {} vs expected: {}", mse, expected
        );
    }

    #[test]
    fn test_dot_f32() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let b: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
        let result = dot_f32(&a, &b);
        assert!((result - 70.0).abs() < 0.1, "f32 dot: {}", result);
    }

    #[test]
    fn test_large_vector_dot() {
        // Test with d=128 (TurboQuant head dimension)
        let a: Vec<f64> = (0..128).map(|i| (i as f64 * 0.1).sin()).collect();
        let b: Vec<f64> = (0..128).map(|i| (i as f64 * 0.2).cos()).collect();
        let simsimd_result = dot_f64(&a, &b);
        let scalar_result: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!(
            (simsimd_result - scalar_result).abs() < 1e-4,
            "d=128 dot mismatch: simsimd={}, scalar={}", simsimd_result, scalar_result
        );
    }
}

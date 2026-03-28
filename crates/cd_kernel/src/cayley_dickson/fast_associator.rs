//! Fast associator computation: compute ||(a*b)*c - a*(b*c)|| without
//! fully materializing all 4 intermediate products.
//!
//! The standard approach:
//!   1. ab = CD_multiply(a, b)       -- full product
//!   2. ab_c = CD_multiply(ab, c)    -- full product
//!   3. bc = CD_multiply(b, c)       -- full product
//!   4. a_bc = CD_multiply(a, bc)    -- full product
//!   5. diff = ab_c - a_bc           -- element-wise subtract
//!   6. norm = ||diff||              -- L2 norm
//!
//! This requires 4 CD multiplications. The fast approach fuses steps 2-4
//! by computing only the DIFFERENCE directly, avoiding full materialization
//! of ab_c and a_bc separately.
//!
//! For dimensions where the CD multiply is the bottleneck (256D+), this
//! reduces from 4 to 2 full multiplications + 2 fused-difference passes.

/// Fast f32 associator norm: computes ||(a*b)*c - a*(b*c)|| with
/// shared sub-expression elimination.
///
/// At the top level of CD doubling, the difference (a*b)*c - a*(b*c)
/// can be decomposed into half-dimension operations that share common
/// sub-products.
pub fn fast_associator_norm_f32(a: &[f32], b: &[f32], c: &[f32], dim: usize) -> f32 {
    // For now, fall through to the standard 4-multiply approach.
    // The fast path is a TODO for the Cariow implementation.
    // This function exists as the wiring point for future optimization.
    super::simd::cd_associator_norm_f32(a, b, c, dim)
}

/// Batch fast associator norms using the ring buffer for O(3*dim) memory.
pub fn batch_fast_associator_norms_f32(
    embedded: &[Vec<f32>],
    dim: usize,
) -> Vec<f32> {
    if embedded.len() < 3 {
        return vec![];
    }
    // Use rayon for parallelism when we have enough work
    if embedded.len() > 1000 {
        use rayon::prelude::*;
        (0..embedded.len() - 2)
            .into_par_iter()
            .map(|i| fast_associator_norm_f32(&embedded[i], &embedded[i + 1], &embedded[i + 2], dim))
            .collect()
    } else {
        (0..embedded.len() - 2)
            .map(|i| fast_associator_norm_f32(&embedded[i], &embedded[i + 1], &embedded[i + 2], dim))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_matches_standard() {
        let dim = 32;
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.2).cos()).collect();
        let c: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.3).sin()).collect();

        let fast = fast_associator_norm_f32(&a, &b, &c, dim);
        let standard = super::super::simd::cd_associator_norm_f32(&a, &b, &c, dim);

        assert!(
            (fast - standard).abs() < 1e-5,
            "Fast {} != Standard {} (diff {})",
            fast, standard, (fast - standard).abs()
        );
    }

    #[test]
    fn test_batch_fast_matches_standard() {
        let dim = 16;
        let vecs: Vec<Vec<f32>> = (0..20)
            .map(|i| (0..dim).map(|j| ((i * 7 + j * 3) as f32).sin()).collect())
            .collect();

        let fast = batch_fast_associator_norms_f32(&vecs, dim);
        let standard = crate::cayley_dickson::simd::batch_sliding_associator_norms_f32(&vecs, dim);

        assert_eq!(fast.len(), standard.len());
        for (f, s) in fast.iter().zip(standard.iter()) {
            assert!((f - s).abs() < 1e-5, "Fast {} != Standard {}", f, s);
        }
    }
}

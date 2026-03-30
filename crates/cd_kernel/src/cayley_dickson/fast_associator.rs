//! Fast associator computation with workspace reuse.
//!
//! # Why workspace reuse matters
//!
//! The standard associator ||(a*b)*c - a*(b*c)|| requires 4 CD multiplications
//! and 4 temporary vectors of size `dim`. For a sliding window of N vectors:
//!   Standard: allocates 4 * dim * N vectors (at 4096D f32: ~160 MB for 10K triplets)
//!   Workspace: allocates 3 * dim ONCE, reuses across all triplets (~48 KB at 4096D)
//!
//! The workspace approach also reduces from 4 buffers to 3 by overlapping:
//!   buf1 = a*b           (needed until step 3)
//!   buf2 = b*c           (needed until step 4)
//!   buf3 = (a*b)*c       (final product 1)
//!   buf1 = a*(b*c)       (overwrite: a*b no longer needed; final product 2)
//!   norm = ||buf3 - buf1||
//!
//! # Why not Karatsuba 3-for-4?
//!
//! The Cayley-Dickson doubling formula (a,b)(c,d) = (ac - conj(d)*b, d*a + b*conj(c))
//! involves both conjugation and non-commutative operand ordering. The standard
//! Karatsuba trick computes M3 = (a+b)(c+d) and derives cross-terms as M3 - M1 - M2.
//! This requires a*d = d*a (commutativity), which fails for quaternions and beyond.
//! The conjugation further prevents algebraic simplification at the sub-algebra level.
//!
//! The Cariow (2012, 2013) algorithms work at the REAL-NUMBER level by analyzing
//! the full bilinear form for fixed dimensions (26 real mults for 8D, 84 for 16D).
//! These are documented in `cariow_factorization.rs` but require per-dimension
//! sign-table analysis to implement. See that module for the theoretical speedup
//! table and evidentiary classification.

/// Reusable workspace for the CD associator computation.
///
/// Holds 3 product buffers of size `dim` plus a multiply workspace, all reused
/// across all triplets in a batch. This eliminates BOTH the per-triplet product
/// allocations AND the per-multiply intermediate allocations.
///
/// Total memory: 3 * dim (product buffers) + W(dim) (multiply workspace) f32.
/// At 128D: 384 + 720 = 1104 f32 = 4.3 KB, reused for unlimited triplets.
pub struct AssociatorWorkspace {
    buf1: Vec<f32>,
    buf2: Vec<f32>,
    buf3: Vec<f32>,
    /// Pre-allocated workspace for cd_multiply_f32_fused (zero-alloc multiply).
    mul_ws: Vec<f32>,
    dim: usize,
}

impl AssociatorWorkspace {
    /// Create a new workspace for the given embedding dimension.
    pub fn new(dim: usize) -> Self {
        let mul_ws_size = super::simd::cd_multiply_f32_workspace_size(dim);
        Self {
            buf1: vec![0.0f32; dim],
            buf2: vec![0.0f32; dim],
            buf3: vec![0.0f32; dim],
            mul_ws: vec![0.0f32; mul_ws_size],
            dim,
        }
    }

    /// Compute ||(a*b)*c - a*(b*c)|| using the 3-buffer overlap strategy
    /// and zero-allocation fused multiply.
    ///
    /// Steps:
    ///   1. buf1 = a * b
    ///   2. buf2 = b * c
    ///   3. buf3 = buf1 * c  (= (a*b)*c)
    ///   4. buf1 = a * buf2  (= a*(b*c), overwrites a*b which is no longer needed)
    ///   5. return ||buf3 - buf1||
    pub fn associator_norm(&mut self, a: &[f32], b: &[f32], c: &[f32]) -> f32 {
        let dim = self.dim;
        debug_assert!(a.len() >= dim && b.len() >= dim && c.len() >= dim);

        // Use fused (zero-alloc) multiply for dims < 256, allocating path for 256+
        super::simd::cd_multiply_f32_fused(a, b, &mut self.buf1, dim, &mut self.mul_ws);
        super::simd::cd_multiply_f32_fused(b, c, &mut self.buf2, dim, &mut self.mul_ws);
        super::simd::cd_multiply_f32_fused(&self.buf1, c, &mut self.buf3, dim, &mut self.mul_ws);
        super::simd::cd_multiply_f32_fused(a, &self.buf2, &mut self.buf1, dim, &mut self.mul_ws);

        // ||buf3 - buf1||
        let mut norm_sq = 0.0f32;
        for i in 0..dim {
            let d = self.buf3[i] - self.buf1[i];
            norm_sq += d * d;
        }
        norm_sq.sqrt()
    }
}

/// Fast f32 associator norm: computes ||(a*b)*c - a*(b*c)|| with
/// shared sub-expression elimination.
///
/// For one-off calls, this allocates a workspace internally.
/// For batch use, prefer `batch_fast_associator_norms_f32` which
/// reuses a single workspace across all triplets.
pub fn fast_associator_norm_f32(a: &[f32], b: &[f32], c: &[f32], dim: usize) -> f32 {
    let mut ws = AssociatorWorkspace::new(dim);
    ws.associator_norm(a, b, c)
}

/// Batch fast associator norms with workspace reuse.
///
/// Allocates ONE workspace (3 * dim f32) and reuses it for all triplets.
/// For N vectors at 4096D: 48 KB workspace vs 160 MB with per-call allocation.
pub fn batch_fast_associator_norms_f32(
    embedded: &[Vec<f32>],
    dim: usize,
) -> Vec<f32> {
    if embedded.len() < 3 {
        return vec![];
    }
    let n = embedded.len() - 2;

    // Cache-aware dispatch: at high dims (128D+), the input vectors exceed
    // L1d capacity when accessed randomly. Use the RingEmbeddingCache streaming
    // path which keeps only 3 vectors (3*D*4 bytes) resident at any time.
    // At 128D: ring = 1.5KB (fits any L1d) vs batch = 1MB (overflows 32KB L1d).
    if dim >= 128 {
        // Streaming ring cache: sequential but cache-optimal for high dims.
        // Each vector is copied into the 3-slot ring before computing.
        let mut ring = super::soa_cache::RingEmbeddingCache::new(dim);
        let mut norms = Vec::with_capacity(n);
        for v in embedded.iter() {
            if let Some(norm) = ring.push_and_compute(v) {
                norms.push(norm);
            }
        }
        norms
    } else if n > 1000 {
        // Rayon parallel: each thread gets its own workspace via map_init.
        // At dim < 128, the batch fits in L2/L3 so random access is fast.
        use rayon::prelude::*;
        (0..n)
            .into_par_iter()
            .map_init(
                || AssociatorWorkspace::new(dim),
                |ws, i| ws.associator_norm(&embedded[i], &embedded[i + 1], &embedded[i + 2]),
            )
            .collect()
    } else {
        // Sequential: single workspace for all triplets
        let mut ws = AssociatorWorkspace::new(dim);
        (0..n)
            .map(|i| ws.associator_norm(&embedded[i], &embedded[i + 1], &embedded[i + 2]))
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

    #[test]
    fn test_workspace_reuse_correctness() {
        // Verify that workspace reuse across multiple calls produces correct results
        let dim = 64;
        let vecs: Vec<Vec<f32>> = (0..10)
            .map(|i| {
                let mut v: Vec<f32> = (0..dim).map(|j| ((i * 13 + j * 5) as f32).sin()).collect();
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-10 {
                    for x in v.iter_mut() { *x /= norm; }
                }
                v
            })
            .collect();

        let mut ws = AssociatorWorkspace::new(dim);
        for i in 0..vecs.len() - 2 {
            let ws_result = ws.associator_norm(&vecs[i], &vecs[i + 1], &vecs[i + 2]);
            let std_result = crate::cayley_dickson::simd::cd_associator_norm_f32(
                &vecs[i], &vecs[i + 1], &vecs[i + 2], dim,
            );
            assert!(
                (ws_result - std_result).abs() < 1e-5,
                "Workspace[{}] {} != Standard {} (diff {})",
                i, ws_result, std_result, (ws_result - std_result).abs()
            );
        }
    }

    #[test]
    fn test_high_dim_workspace() {
        // Verify workspace correctness at 256D (where rayon kicks in for CD multiply)
        let dim = 256;
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.02).cos()).collect();
        let c: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.03).sin()).collect();

        let fast = fast_associator_norm_f32(&a, &b, &c, dim);
        let standard = crate::cayley_dickson::simd::cd_associator_norm_f32(&a, &b, &c, dim);

        assert!(
            (fast - standard).abs() < 1e-3,
            "256D: Fast {} != Standard {} (diff {})",
            fast, standard, (fast - standard).abs()
        );
    }

    #[test]
    fn test_fused_multiply_matches_allocating() {
        // Verify cd_multiply_f32_fused matches cd_multiply_f32_into at all power-of-2 dims
        use crate::cayley_dickson::simd::{
            cd_multiply_f32_fused, cd_multiply_f32_into, cd_multiply_f32_workspace_size,
        };
        for &dim in &[2, 4, 8, 16, 32, 64, 128] {
            let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.17).sin()).collect();
            let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.31).cos()).collect();

            let mut out_alloc = vec![0.0f32; dim];
            let mut out_fused = vec![0.0f32; dim];
            let ws_size = cd_multiply_f32_workspace_size(dim);
            let mut ws = vec![0.0f32; ws_size];

            cd_multiply_f32_into(&a, &b, &mut out_alloc, dim);
            cd_multiply_f32_fused(&a, &b, &mut out_fused, dim, &mut ws);

            for i in 0..dim {
                assert!(
                    (out_alloc[i] - out_fused[i]).abs() < 1e-6,
                    "dim={} component {}: alloc {} != fused {} (diff {})",
                    dim, i, out_alloc[i], out_fused[i],
                    (out_alloc[i] - out_fused[i]).abs()
                );
            }
        }
    }

    #[test]
    fn test_workspace_size_growth() {
        use crate::cayley_dickson::simd::cd_multiply_f32_workspace_size;
        // Verify workspace size is monotonically increasing and zero for small dims
        assert_eq!(cd_multiply_f32_workspace_size(1), 0);
        assert_eq!(cd_multiply_f32_workspace_size(2), 0);
        assert_eq!(cd_multiply_f32_workspace_size(4), 0);
        assert_eq!(cd_multiply_f32_workspace_size(8), 0);
        assert!(cd_multiply_f32_workspace_size(16) > 0);
        let mut prev = 0;
        for &d in &[16, 32, 64, 128, 256] {
            let ws = cd_multiply_f32_workspace_size(d);
            assert!(ws > prev, "dim={}: ws {} should be > prev {}", d, ws, prev);
            prev = ws;
        }
    }
}

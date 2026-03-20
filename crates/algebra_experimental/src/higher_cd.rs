// ---------------------------------------------------------------------------
// Trap B: Re-export canonical types from gororoba_algebra::construction::cd_tower
// so that any consumer using `algebra_experimental::higher_cd::*` continues to
// resolve these names after their definitions were removed from this file.
// ---------------------------------------------------------------------------
pub use gororoba_algebra::construction::cd_tower::{
    Pathion, Chingon, Routon, Voudon, Eriston, DekaVoudon,
};

// ---------------------------------------------------------------------------
// AVT types: canonical definitions now live in algebra_analysis::avt.
// `HigherAvt` is a type alias so `HigherAvt::new()`, `HigherAvt::sampled()`,
// and all field accesses continue to work at all existing call sites.
// ---------------------------------------------------------------------------
pub use algebra_analysis::avt::{
    AlternativityViolationTensor, PackedAvt, SampledAvt, associator_basis, index_bits_for_dim,
};
/// Backward-compat alias: `HigherAvt` -> `AlternativityViolationTensor`.
pub type HigherAvt = AlternativityViolationTensor;

/// Universal properties of Cayley-Dickson algebras for n >= 4.
pub struct UniversalCDProperties;

impl UniversalCDProperties {
    /// All CD algebras are power-associative: x^n is well-defined.
    pub fn is_power_associative() -> bool { true }

    /// All CD algebras are flexible: x(yx) = (xy)x.
    pub fn is_flexible() -> bool { true }

    /// CD algebras for dim >= 16 contain zero divisors.
    pub fn has_zero_divisors(dim: usize) -> bool { dim >= 16 }
}

/// Systematic naming for higher Cayley-Dickson algebras.
///
/// Returns the primary (Greek-ordinal) name for each known CD tower dimension.
/// Covers 4D through 16384D for the SIMD/sparse tower; also includes 1D-2D for completeness.
pub fn cd_name(dim: usize) -> &'static str {
    match dim {
        1     => "Real",
        2     => "Complex",
        4     => "Quaternion",
        8     => "Octonion",
        16    => "Sedenion",
        32    => "Pathion",
        64    => "Chingon",
        128   => "Routon",
        256   => "Voudon",
        512   => "Eriston",
        1024  => "DekaVoudon",
        2048  => "Endekavoudon",
        4096  => "Dodekvoudon",
        8192  => "Dekatrisvoudon",
        16384 => "Tessareskaidekavoudon",
        _     => "Higher 2^n-ion",
    }
}

/// Theoretical derivation dimensions for CD algebras.
/// - Der(C) = 0
/// - Der(H) = su(2) (3D)
/// - Der(O) = g2 (14D)
/// - Der(S) = ? (Identified as G2-related in synthesis)
pub fn derivation_dim(dim: usize) -> Option<usize> {
    match dim {
        2 => Some(0),
        4 => Some(3),
        8 => Some(14),
        16 => None, // Open problem
        32 => None, // Under investigation
        _ => None,
    }
}

// SparseApeironState moved to algebra_analysis::sparse::SparseState (Phase 4).
// This backward-compat alias keeps all existing import paths valid.
pub use algebra_analysis::sparse::SparseState as SparseApeironState;

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn profile_higher_avt_construction() {
        let _ = HigherAvt::new(16);
        for dim in [16, 32, 64, 128, 256] {
            let t = Instant::now();
            let avt = HigherAvt::new(dim);
            let elapsed = t.elapsed();
            let mem_bytes =
                avt.violations.len() * std::mem::size_of::<(usize, usize, usize, usize, i32)>();
            eprintln!(
                "HigherAvt::new({:>4}): {:>8} violations, {:>10.3}ms, {:.1} MB",
                dim, avt.violations.len(), elapsed.as_secs_f64() * 1000.0,
                mem_bytes as f64 / 1e6,
            );
        }
    }

    #[test]
    fn profile_sampled_avt_512_1024() {
        for (dim, n_samples) in [(512, 1_000_000), (1024, 1_000_000)] {
            let t = Instant::now();
            let result = HigherAvt::sampled(dim, n_samples, 42);
            let elapsed = t.elapsed();
            eprintln!(
                "HigherAvt::sampled({:>4}, {}): {:>8} violations, hit_rate={:.4}, {:>10.3}ms",
                dim, n_samples, result.avt.violations.len(), result.hit_rate,
                elapsed.as_secs_f64() * 1000.0,
            );
        }
    }

    #[test]
    fn test_sparse_apeiron_zero() {
        let s = SparseApeironState::zero(1024);
        assert_eq!(s.nnz(), 0);
        assert!((s.sparsity() - 1.0).abs() < 1e-10);
        assert!((s.norm() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_apeiron_from_dense() {
        let mut dense = vec![0.0; 256];
        dense[0] = 1.0;
        dense[42] = -3.0;
        dense[100] = 0.5;
        let s = SparseApeironState::from_dense(256, &dense, 1e-10);
        assert_eq!(s.nnz(), 3);
        assert!((s.norm_sq() - (1.0 + 9.0 + 0.25)).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_apeiron_dot_product() {
        let a = SparseApeironState::from_pairs(128, vec![(0, 1.0), (5, 2.0), (10, 3.0)]);
        let b = SparseApeironState::from_pairs(128, vec![(0, 4.0), (5, -1.0), (20, 7.0)]);
        let d = a.dot(&b);
        assert!((d - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_apeiron_add() {
        let a = SparseApeironState::from_pairs(64, vec![(0, 1.0), (5, 2.0)]);
        let b = SparseApeironState::from_pairs(64, vec![(5, -2.0), (10, 3.0)]);
        let c = a.add(&b);
        assert_eq!(c.nnz(), 2);
        let dense = c.to_dense();
        assert!((dense[0] - 1.0).abs() < 1e-10);
        assert!((dense[5] - 0.0).abs() < 1e-10);
        assert!((dense[10] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_apeiron_shannon_entropy() {
        let s1 = SparseApeironState::from_pairs(1024, vec![(42, 1.0)]);
        assert!((s1.shannon_entropy() - 0.0).abs() < 1e-10);
        let s2 = SparseApeironState::from_pairs(1024, vec![(0, 1.0), (1, 1.0)]);
        assert!((s2.shannon_entropy() - 2.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_apeiron_roundtrip() {
        let pairs = vec![(3, 1.5), (17, -0.7), (255, 42.0)];
        let s = SparseApeironState::from_pairs(256, pairs);
        let dense = s.to_dense();
        let s2 = SparseApeironState::from_dense(256, &dense, 1e-10);
        assert_eq!(s.nnz(), s2.nnz());
        assert!((s.dot(&s2) - s.norm_sq()).abs() < 1e-10);
    }
}

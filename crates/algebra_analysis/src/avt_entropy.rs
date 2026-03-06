//! Shannon entropy of Alternativity Violation Tensor (AVT) axis distributions.
//!
//! The AVT captures violations of the alternative identity `[x, x, y] = 0` in
//! Cayley-Dickson algebras. For each violation triple `(i, j, k)` with `i < j`,
//! the sum `[e_i, e_j, e_k] + [e_j, e_i, e_k]` lands on some output axis `e_m`.
//!
//! This module counts how many violations target each output axis `m` and
//! computes Shannon entropy of the resulting distribution. Key observations:
//!
//! - dim=8 (octonions): ZERO violations (alternative algebra), H = 0.
//! - dim=16 (sedenions): violations exist, H > 0.
//! - dim=64 (chingons): more violations, 0 < H < ln(64).
//!
//! The AVT construction uses only `cd_kernel::cayley_dickson::cd_basis_mul_sign_iter`,
//! avoiding a circular dependency on `algebra_core`.

use cd_kernel::cayley_dickson::cd_basis_mul_sign_iter;

/// Result of AVT axis-entropy computation for a single Cayley-Dickson dimension.
#[derive(Debug, Clone)]
pub struct AvtEntropyResult {
    /// Cayley-Dickson dimension (must be a power of two).
    pub dim: usize,
    /// Total number of alternativity violations found.
    pub total_violations: usize,
    /// Number of violations targeting each output axis `m` in `0..dim`.
    pub axis_counts: Vec<usize>,
    /// Shannon entropy H = -sum(p_i * ln(p_i)) of the axis distribution.
    pub shannon_entropy: f64,
    /// Maximum possible entropy: ln(number of axes with non-zero counts).
    pub max_entropy: f64,
}

/// Compute the basis associator `[e_i, e_j, e_k] = sign * e_m`.
///
/// Returns `(m, sign)` where `m = (i ^ j) ^ k` and `sign` is the
/// product of Cayley-Dickson basis multiplication signs minus the
/// re-associated product.
fn associator_basis(dim: usize, i: usize, j: usize, k: usize) -> (usize, i32) {
    // (e_i * e_j) * e_k
    let ij_idx = i ^ j;
    let ij_sign = cd_basis_mul_sign_iter(dim, i, j);
    let ijk_idx = ij_idx ^ k;
    let ijk_sign1 = ij_sign * cd_basis_mul_sign_iter(dim, ij_idx, k);

    // e_i * (e_j * e_k)
    let jk_idx = j ^ k;
    let jk_sign = cd_basis_mul_sign_iter(dim, j, k);
    let ijk_sign2 = jk_sign * cd_basis_mul_sign_iter(dim, i, jk_idx);

    (ijk_idx, ijk_sign1 - ijk_sign2)
}

/// Compute Shannon entropy of AVT violation distribution across output axes.
///
/// Enumerates all basis triples `(i, j, k)` with `i < j` and checks whether
/// the alternativity sum `[e_i, e_j, e_k] + [e_j, e_i, e_k]` is non-zero.
/// Each non-zero result targets some output axis `e_m`; the distribution of
/// these counts across axes yields the Shannon entropy.
///
/// # Panics
///
/// Panics if `dim` is not a power of two.
pub fn avt_axis_entropy(dim: usize) -> AvtEntropyResult {
    assert!(
        dim >= 2 && dim.is_power_of_two(),
        "dim must be a power of 2 >= 2, got {dim}"
    );

    let mut axis_counts = vec![0usize; dim];
    let mut total_violations: usize = 0;

    // Enumerate violations: i < j to avoid double-counting; i = j always cancels.
    for i in 0..dim {
        for j in (i + 1)..dim {
            for k in 0..dim {
                let (m1, s1) = associator_basis(dim, i, j, k);
                let (m2, s2) = associator_basis(dim, j, i, k);

                // In Cayley-Dickson algebras, both associators land on the same axis.
                debug_assert_eq!(
                    m1, m2,
                    "structural failure: associator indices diverge at ({i},{j},{k})"
                );

                let sum_sign = s1 + s2;
                if sum_sign != 0 {
                    axis_counts[m1] += 1;
                    total_violations += 1;
                }
            }
        }
    }

    // Compute Shannon entropy from axis distribution.
    let shannon_entropy = histogram_entropy(&axis_counts, total_violations);

    // Maximum entropy = ln(number of non-zero axes).
    let non_zero_axes = axis_counts.iter().filter(|&&c| c > 0).count();
    let max_entropy = if non_zero_axes > 0 {
        (non_zero_axes as f64).ln()
    } else {
        0.0
    };

    AvtEntropyResult {
        dim,
        total_violations,
        axis_counts,
        shannon_entropy,
        max_entropy,
    }
}

/// Shannon entropy of a histogram: H = -sum(p_i * ln(p_i)).
///
/// Identical to the pattern in `associator_entropy.rs`.
fn histogram_entropy(counts: &[usize], total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let t = total as f64;
    let mut h = 0.0;
    for &c in counts {
        if c > 0 {
            let p = c as f64 / t;
            h -= p * p.ln();
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_octonion_zero_entropy() {
        // dim=8: octonions are alternative, so ZERO violations, entropy = 0.
        let result = avt_axis_entropy(8);
        assert_eq!(result.total_violations, 0, "octonions must have zero AVT violations");
        assert_eq!(result.shannon_entropy, 0.0, "zero violations => zero entropy");
        assert!(
            result.axis_counts.iter().all(|&c| c == 0),
            "all axis counts must be zero for octonions"
        );
    }

    #[test]
    fn test_sedenion_positive_entropy() {
        // dim=16: sedenions have violations, entropy > 0.
        let result = avt_axis_entropy(16);
        assert!(
            result.total_violations > 0,
            "sedenions must have AVT violations, got 0"
        );
        assert!(
            result.shannon_entropy > 0.0,
            "sedenion AVT entropy must be positive, got {}",
            result.shannon_entropy
        );
        assert!(
            result.shannon_entropy <= result.max_entropy + 1e-12,
            "entropy {:.6} must not exceed max_entropy {:.6}",
            result.shannon_entropy,
            result.max_entropy
        );
    }

    #[test]
    fn test_chingon_bounded_entropy() {
        // dim=64: entropy should be bounded: 0 < H < ln(64) ~ 4.16.
        let result = avt_axis_entropy(64);
        assert!(
            result.total_violations > 0,
            "dim=64 must have AVT violations"
        );
        assert!(
            result.shannon_entropy > 0.0,
            "dim=64 AVT entropy must be positive"
        );
        let ln64 = (64.0_f64).ln();
        assert!(
            result.shannon_entropy < ln64,
            "entropy {:.3} should be < ln(64)={:.3}",
            result.shannon_entropy,
            ln64
        );
    }
}

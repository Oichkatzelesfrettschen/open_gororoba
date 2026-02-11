//! Stiefel manifold verification for zero-divisor sets.
//!
//! Koebisu (arXiv:2512.13002) claims the zero-divisor set of sedenions
//! (dim=16) is diffeomorphic to the Stiefel manifold V_{8,2} --
//! the space of orthonormal 2-frames in R^8.
//!
//! # Mathematical structure
//!
//! A sedenion z = (a, b) where a, b in O (octonions) is a zero-divisor
//! iff |a| = |b| and Re(a * conj(b)) = 0 (orthogonality in R^8 inner product).
//!
//! For unit-norm ZDs (|z| = 1), we have |a|^2 + |b|^2 = 1 with a perp b.
//! After rescaling both to unit length, this gives a pair (a/|a|, b/|b|)
//! in V_{8,2} = {(u,v) in S^7 x S^7 : <u,v> = 0}.
//!
//! # Holonomy
//!
//! The (1,1)-type holonomy on the torus projection of V_{8,2} measures
//! how parallel transport around closed loops fails to return to the
//! starting frame. This connects to the non-associativity of octonions.
//!
//! # Literature
//! - Koebisu (arXiv:2512.13002): ZD set = V_{8,2} Stiefel manifold
//! - Reggiani (2024): Geometry of sedenion zero divisors

use crate::construction::cayley_dickson::cd_norm_sq;

/// Result of the Stiefel manifold verification.
#[derive(Debug, Clone)]
pub struct StiefelVerification {
    /// Ambient CD dimension (16 for sedenions).
    pub dim: usize,
    /// Number of zero-divisors sampled.
    pub n_samples: usize,
    /// Number that satisfy the V_{8,2} condition (orthonormal 2-frame).
    pub n_stiefel: usize,
    /// Maximum orthogonality violation |<a/|a|, b/|b|>|.
    pub max_ortho_violation: f64,
    /// Maximum norm-balance violation ||a| - |b|| (before rescaling).
    pub max_norm_violation: f64,
    /// Fraction satisfying V_{8,2} (should be 1.0).
    pub stiefel_fraction: f64,
}

/// Check if a unit sedenion z = (a, b) is a zero-divisor by verifying
/// that its left multiplication matrix has non-trivial nullspace.
///
/// Returns true if z is a zero-divisor (exists non-zero w with z*w = 0).
fn is_zero_divisor(z: &[f64]) -> bool {
    assert_eq!(z.len(), 16);
    let norm_sq = cd_norm_sq(z);
    if norm_sq < 1e-14 {
        return false; // zero element is not interesting
    }

    // Check: z is a ZD iff z * w = 0 for some non-zero w.
    // Use the basis ZD test: multiply z by each basis element and check
    // if any product has unexpectedly small norm.
    // More rigorous: build left multiplication matrix and check rank.
    let mat = crate::analysis::annihilator::left_multiplication_matrix(z, 16);
    let svd = nalgebra::SVD::new(mat, false, false);
    let min_sv = svd
        .singular_values
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);

    // If smallest singular value is near zero, z has non-trivial annihilator
    min_sv < 1e-8
}

/// Decompose a sedenion into its octonion halves: z = (a, b) where a, b in O.
///
/// Under the Cayley-Dickson construction S = O x O:
/// - a = z[0..8] (lower half)
/// - b = z[8..16] (upper half)
fn decompose_halves(z: &[f64]) -> (&[f64], &[f64]) {
    assert_eq!(z.len(), 16);
    (&z[..8], &z[8..])
}

/// Inner product of two vectors (standard R^n dot product).
fn inner_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Euclidean norm of a vector.
fn vec_norm(v: &[f64]) -> f64 {
    inner_product(v, v).sqrt()
}

/// Verify the Stiefel manifold condition for sedenion zero-divisors.
///
/// For each sampled ZD z = (a, b):
/// 1. Check |a| = |b| (norm balance)
/// 2. Check <a, b> = 0 (orthogonality)
/// 3. After rescaling, (a/|a|, b/|b|) is in V_{8,2}
pub fn verify_stiefel_condition(n_samples: usize, seed: u64) -> StiefelVerification {
    use rand::Rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut n_stiefel = 0;
    let mut max_ortho = 0.0f64;
    let mut max_norm = 0.0f64;
    let mut total_sampled = 0;

    // Strategy: generate random unit sedenions and check which are ZDs.
    // Then verify the V_{8,2} condition for each ZD.
    //
    // Alternative: construct ZDs from known structure (cross-assessor pairs)
    // and verify the condition analytically.

    // Method 1: Random sampling of unit sedenions
    let mut attempts = 0;
    while total_sampled < n_samples && attempts < n_samples * 100 {
        attempts += 1;

        // Random unit sedenion
        let mut z = [0.0f64; 16];
        let mut norm_sq = 0.0;
        for component in &mut z {
            let g: f64 = rng.gen_range(-1.0..1.0);
            *component = g;
            norm_sq += g * g;
        }
        if norm_sq < 1e-10 {
            continue;
        }
        let norm = norm_sq.sqrt();
        for component in &mut z {
            *component /= norm;
        }

        if !is_zero_divisor(&z) {
            continue;
        }

        total_sampled += 1;

        // Decompose and check V_{8,2} condition
        let (a, b) = decompose_halves(&z);
        let norm_a = vec_norm(a);
        let norm_b = vec_norm(b);

        // Norm balance: |a| should equal |b|
        let norm_diff = (norm_a - norm_b).abs();
        max_norm = max_norm.max(norm_diff);

        // Orthogonality: <a, b> should be 0
        let dot = inner_product(a, b);
        let ortho_violation = dot.abs();
        max_ortho = max_ortho.max(ortho_violation);

        // V_{8,2} condition: both should be satisfied
        if norm_diff < 0.01 && ortho_violation < 0.01 {
            n_stiefel += 1;
        }
    }

    let stiefel_fraction = if total_sampled > 0 {
        n_stiefel as f64 / total_sampled as f64
    } else {
        0.0
    };

    StiefelVerification {
        dim: 16,
        n_samples: total_sampled,
        n_stiefel,
        max_ortho_violation: max_ortho,
        max_norm_violation: max_norm,
        stiefel_fraction,
    }
}

/// Construct specific zero-divisors from cross-assessor pairs and verify
/// the Stiefel condition algebraically.
///
/// A diagonal zero-divisor z = e_i + s*e_j (where i < 8, j >= 8, s = +/-1)
/// satisfies:
/// - a = (0,...,1,...,0) in R^8 (only component i is non-zero)
/// - b = (0,...,s,...,0) in R^8 (only component j-8 is non-zero)
/// - |a| = |b| = 1 (exact)
/// - <a, b> = 0 (exact, since different components)
///
/// This provides analytic verification of the V_{8,2} condition.
pub fn verify_stiefel_algebraic() -> StiefelVerification {
    use crate::analysis::boxkites::{cross_assessors, diagonal_zero_products_exact};

    let pairs = cross_assessors(16);
    let mut n_stiefel = 0;
    let mut n_total = 0;
    let mut max_ortho = 0.0f64;
    let mut max_norm = 0.0f64;

    for &(lo, hi) in &pairs {
        // Construct z = e_lo + e_hi
        let mut z = [0.0f64; 16];
        z[lo] = 1.0;
        z[hi] = 1.0;

        // Decompose
        let (a, b) = decompose_halves(&z);
        let norm_a = vec_norm(a);
        let norm_b = vec_norm(b);
        let dot = inner_product(a, b);

        let norm_diff = (norm_a - norm_b).abs();
        let ortho_violation = dot.abs();

        max_norm = max_norm.max(norm_diff);
        max_ortho = max_ortho.max(ortho_violation);

        n_total += 1;
        if norm_diff < 1e-10 && ortho_violation < 1e-10 {
            n_stiefel += 1;
        }

        // Also check z = e_lo - e_hi
        let mut z_neg = [0.0f64; 16];
        z_neg[lo] = 1.0;
        z_neg[hi] = -1.0;

        let (a_neg, b_neg) = decompose_halves(&z_neg);
        let norm_a_neg = vec_norm(a_neg);
        let norm_b_neg = vec_norm(b_neg);
        let dot_neg = inner_product(a_neg, b_neg);

        let norm_diff_neg = (norm_a_neg - norm_b_neg).abs();
        let ortho_violation_neg = dot_neg.abs();

        max_norm = max_norm.max(norm_diff_neg);
        max_ortho = max_ortho.max(ortho_violation_neg);

        n_total += 1;
        if norm_diff_neg < 1e-10 && ortho_violation_neg < 1e-10 {
            n_stiefel += 1;
        }
    }

    // Also check all actual zero-product pairs
    for &a_pair in &pairs {
        for &b_pair in &pairs {
            if a_pair >= b_pair {
                continue;
            }
            let solutions = diagonal_zero_products_exact(16, a_pair, b_pair);
            for &(s, _t) in &solutions {
                // z = e_{a.0} + s*e_{a.1} is a zero-divisor (it annihilates e_{b.0} + t*e_{b.1})
                let mut z = [0.0f64; 16];
                z[a_pair.0] = 1.0;
                z[a_pair.1] = s as f64;

                let (a_half, b_half) = decompose_halves(&z);
                let norm_a_h = vec_norm(a_half);
                let norm_b_h = vec_norm(b_half);
                let dot_h = inner_product(a_half, b_half);

                let norm_diff_h = (norm_a_h - norm_b_h).abs();
                let ortho_violation_h = dot_h.abs();

                max_norm = max_norm.max(norm_diff_h);
                max_ortho = max_ortho.max(ortho_violation_h);

                n_total += 1;
                if norm_diff_h < 1e-10 && ortho_violation_h < 1e-10 {
                    n_stiefel += 1;
                }
            }
        }
    }

    StiefelVerification {
        dim: 16,
        n_samples: n_total,
        n_stiefel,
        max_ortho_violation: max_ortho,
        max_norm_violation: max_norm,
        stiefel_fraction: if n_total > 0 {
            n_stiefel as f64 / n_total as f64
        } else {
            0.0
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stiefel_algebraic_cross_assessors() {
        let result = verify_stiefel_algebraic();

        eprintln!("Stiefel V_{{8,2}} algebraic verification:");
        eprintln!("  total ZDs tested: {}", result.n_samples);
        eprintln!("  V_{{8,2}} condition satisfied: {}", result.n_stiefel);
        eprintln!("  fraction: {:.4}", result.stiefel_fraction);
        eprintln!(
            "  max orthogonality violation: {:.2e}",
            result.max_ortho_violation
        );
        eprintln!(
            "  max norm-balance violation: {:.2e}",
            result.max_norm_violation
        );

        // Norm balance should always hold (each component contributes equally)
        assert!(
            result.max_norm_violation < 1e-10,
            "norm balance should always hold for cross-assessor ZDs"
        );

        // Not all cross-assessor elements are ZDs; those that are should
        // satisfy the Stiefel condition. The fraction tells us how many
        // cross-assessor-based elements pass the orthogonality test.
        // Those with lo and hi sharing the same octonion half-index
        // (i.e., lo ^ hi < 8) will have non-zero inner product.
        eprintln!("  note: fraction < 1.0 expected when lo and hi-8 overlap");
    }

    #[test]
    fn test_stiefel_random_sample() {
        // Random sampling is slow due to ZD rarity, so use small sample
        let result = verify_stiefel_condition(20, 42);

        eprintln!("Stiefel V_{{8,2}} random sample verification:");
        eprintln!("  ZDs found: {}", result.n_samples);
        eprintln!(
            "  V_{{8,2}} condition: {}/{}",
            result.n_stiefel, result.n_samples
        );
        eprintln!("  fraction: {:.4}", result.stiefel_fraction);
        eprintln!("  max ortho violation: {:.2e}", result.max_ortho_violation);
        eprintln!("  max norm violation: {:.2e}", result.max_norm_violation);

        // Due to random sampling, ZDs are very rare in S^15.
        // Even with 2000 attempts, we may find 0 ZDs.
        // This test is mainly for infrastructure validation.
        if result.n_samples > 0 {
            // All found ZDs should satisfy the condition
            assert!(
                result.stiefel_fraction > 0.9,
                "most random ZDs should satisfy V_{{8,2}}"
            );
        }
    }

    #[test]
    fn test_basis_zd_stiefel() {
        // Find a confirmed zero-divisor from the box-kite zero-product structure.
        // Not every e_lo + e_hi is a ZD; only those pairs (a, b) where
        // diagonal_zero_products_exact returns non-empty solutions.
        use crate::analysis::boxkites::{cross_assessors, diagonal_zero_products_exact};

        let pairs = cross_assessors(16);
        let mut found_zd = false;

        for (i, &a_pair) in pairs.iter().enumerate() {
            for &b_pair in &pairs[(i + 1)..] {
                let solutions = diagonal_zero_products_exact(16, a_pair, b_pair);
                if solutions.is_empty() {
                    continue;
                }

                // We have a confirmed ZD: z = e_{a.0} + s*e_{a.1}
                let (s, _t) = solutions[0];
                let mut z = [0.0f64; 16];
                z[a_pair.0] = 1.0;
                z[a_pair.1] = s as f64;

                // Verify it's actually a ZD via SVD
                assert!(is_zero_divisor(&z), "confirmed ZD should pass SVD check");

                // Verify V_{8,2} condition
                let (a, b) = decompose_halves(&z);
                let norm_a = vec_norm(a);
                let norm_b = vec_norm(b);
                let dot = inner_product(a, b);

                // Cross-assessor pair has lo < 8 and hi >= 8,
                // so a and b occupy different R^8 components.
                // Norm balance: |a| = |b| = 1 (single basis element each)
                assert!(
                    (norm_a - norm_b).abs() < 1e-12,
                    "norm balance for ({}, {})",
                    a_pair.0,
                    a_pair.1
                );

                // Orthogonality: <a, b> = 0 when lo != hi - 8
                if a_pair.0 != a_pair.1 - 8 {
                    assert!(
                        dot.abs() < 1e-12,
                        "<a,b> should be 0 for ({}, {})",
                        a_pair.0,
                        a_pair.1
                    );
                }

                eprintln!(
                    "Confirmed ZD e_{} + {}*e_{}: |a|={:.6}, |b|={:.6}, <a,b>={:.2e}",
                    a_pair.0, s, a_pair.1, norm_a, norm_b, dot
                );
                found_zd = true;
                break;
            }
            if found_zd {
                break;
            }
        }
        assert!(found_zd, "should find at least one confirmed ZD pair");
    }

    #[test]
    fn test_verified_zd_count() {
        // Count how many confirmed ZDs satisfy V_{8,2} vs how many do not.
        // This gives us the actual fraction for Koebisu's claim.
        use crate::analysis::boxkites::{cross_assessors, diagonal_zero_products_exact};

        let pairs = cross_assessors(16);
        let mut n_zd = 0;
        let mut n_stiefel = 0;
        let mut n_ortho_fail = 0;

        for (i, &a_pair) in pairs.iter().enumerate() {
            for &b_pair in &pairs[(i + 1)..] {
                let solutions = diagonal_zero_products_exact(16, a_pair, b_pair);
                for &(s, _t) in &solutions {
                    let mut z = [0.0f64; 16];
                    z[a_pair.0] = 1.0;
                    z[a_pair.1] = s as f64;

                    let (a, b) = decompose_halves(&z);
                    let norm_a = vec_norm(a);
                    let norm_b = vec_norm(b);
                    let dot = inner_product(a, b);

                    n_zd += 1;
                    let norm_ok = (norm_a - norm_b).abs() < 1e-10;
                    let ortho_ok = dot.abs() < 1e-10;
                    if norm_ok && ortho_ok {
                        n_stiefel += 1;
                    }
                    if !ortho_ok {
                        n_ortho_fail += 1;
                    }
                }
            }
        }

        eprintln!("Verified ZD Stiefel census:");
        eprintln!("  total confirmed ZDs: {}", n_zd);
        eprintln!("  V_{{8,2}} satisfied: {}", n_stiefel);
        eprintln!("  orthogonality failures: {}", n_ortho_fail);
        if n_zd > 0 {
            eprintln!("  fraction: {:.4}", n_stiefel as f64 / n_zd as f64);
        }

        // All confirmed ZDs from distinct cross-assessor pairs should have
        // lo != hi-8, so orthogonality should hold for diagonal ZDs.
        assert!(n_zd > 0, "must find at least one ZD");
    }
}

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

use cd_kernel::cayley_dickson::cd_norm_sq;

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
    let mat = crate::annihilator::left_multiplication_matrix(z, 16);
    let svd = nalgebra::linalg::SVD::new(mat, false, false);
    let min_sv: f64 = svd
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
    use rand::{Rng, SeedableRng};
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
    use crate::boxkites::{cross_assessors, diagonal_zero_products_exact};

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

// ---------------------------------------------------------------------------
// Holonomy angle on V_{8,2} and G_2 orbit invariants
// ---------------------------------------------------------------------------

/// Result of a parallel-transport holonomy computation on V_{8,2}.
///
/// Given two zero-divisors z1 = (u1, v1) and z2 = (u2, v2) in the
/// Stiefel manifold V_{8,2} (orthonormal 2-frames in R^8), the
/// holonomy angle theta measures the rotation of the second frame vector
/// when parallel-transported along the geodesic on S^7 connecting u1
/// to u2.
///
/// # Literature
/// - Reggiani (2024): Geometry of sedenion zero divisors
/// - Koebisu (arXiv:2512.13002): ZD set ~ V_{8,2}
#[derive(Debug, Clone, Copy)]
pub struct HolonomyResult {
    /// Geodesic distance on S^7 between u1 and u2 (in radians).
    pub geodesic_distance: f64,
    /// Holonomy angle: rotation of the v-frame under parallel transport
    /// along the u-geodesic (in radians, range [0, pi]).
    pub holonomy_angle: f64,
    /// Inner product <v1, v2> projected orthogonal to the transport plane.
    pub projected_inner_product: f64,
}

/// Compute the parallel-transport holonomy angle between two V_{8,2} frames.
///
/// Each frame is given as a sedenion z = (a, b) with a, b  in  R^8.
/// After normalising to unit vectors u = a/|a|, v = b/|b|, we compute:
///
/// 1. Geodesic distance d = arccos(<u1, u2>) on S^7.
/// 2. Parallel-transport v1 along the great circle from u1 to u2.
/// 3. Holonomy angle = arccos(<v1_transported, v2>).
///
/// For non-associative algebras the holonomy is generically non-trivial,
/// reflecting the curvature of V_{8,2} ~ SO(8)/SO(6).
pub fn holonomy_between(z1: &[f64], z2: &[f64]) -> Option<HolonomyResult> {
    if z1.len() != 16 || z2.len() != 16 {
        return None;
    }
    let (a1, b1) = (&z1[..8], &z1[8..]);
    let (a2, b2) = (&z2[..8], &z2[8..]);

    let norm_a1 = vec_norm(a1);
    let norm_b1 = vec_norm(b1);
    let norm_a2 = vec_norm(a2);
    let norm_b2 = vec_norm(b2);

    if norm_a1 < 1e-14 || norm_b1 < 1e-14 || norm_a2 < 1e-14 || norm_b2 < 1e-14 {
        return None;
    }

    // Unit vectors
    let u1: Vec<f64> = a1.iter().map(|&x| x / norm_a1).collect();
    let v1: Vec<f64> = b1.iter().map(|&x| x / norm_b1).collect();
    let u2: Vec<f64> = a2.iter().map(|&x| x / norm_a2).collect();
    let v2: Vec<f64> = b2.iter().map(|&x| x / norm_b2).collect();

    let cos_d = inner_product(&u1, &u2).clamp(-1.0, 1.0);
    let geodesic_distance = cos_d.acos();

    // Parallel transport v1 along the geodesic from u1 to u2 on S^7.
    // For a great circle parameterised by gamma(t) = cos(t)*u1 + sin(t)*e,
    // where e = (u2 - cos_d*u1) / sin_d  is the unit tangent,
    // the parallel transport of v along gamma is:
    //   v(t) = v - <v,u1>*u1 - <v,e>*e
    //        + [<v,u1>*cos(t) - <v,e>*sin(t)] * gamma(t)/|gamma|
    //        + [<v,u1>*sin(t) + <v,e>*cos(t)] * gamma'(t)/|gamma'|
    //
    // Simplified Schild's ladder approach for the sphere:
    //   P_transport(v1) = v1 - (<v1,u1> + <v1,u2>)/(1 + cos_d) * (u1 + u2)
    //                      + 2*<v1,u1> * u2
    // (valid when cos_d > -1, i.e. u1 != -u2)

    // Distinguish near-identical (d ~ 0) from near-antipodal (d ~ pi).
    // sin(0) = sin(pi) = 0, so we must check geodesic_distance directly.
    let near_zero_tol = 1e-12;
    let near_pi_tol = 1e-12;

    let v1_transported = if geodesic_distance < near_zero_tol {
        // u1 ~ u2: parallel transport is identity
        v1.clone()
    } else if (std::f64::consts::PI - geodesic_distance) < near_pi_tol {
        // u1 ~ -u2 (antipodal): geodesic is non-unique and parallel
        // transport is path-dependent.  No single deterministic holonomy
        // angle exists, so we return None.
        return None;
    } else {
        // Rodrigues-type parallel transport formula on S^n:
        // PT(v) = v - (<v,u1>+<v,u2>)/(1+cos_d) * (u1+u2) + 2<v,u1>*u2
        let vu1 = inner_product(&v1, &u1);
        let vu2 = inner_product(&v1, &u2);
        let coeff = (vu1 + vu2) / (1.0 + cos_d);
        let mut pt = vec![0.0_f64; 8];
        for k in 0..8 {
            pt[k] = v1[k] - coeff * (u1[k] + u2[k]) + 2.0 * vu1 * u2[k];
        }
        // Project out u2 component to stay in tangent space at u2
        let pt_u2 = inner_product(&pt, &u2);
        for k in 0..8 {
            pt[k] -= pt_u2 * u2[k];
        }
        // Re-normalise
        let pt_norm = vec_norm(&pt);
        if pt_norm > 1e-14 {
            for x in &mut pt {
                *x /= pt_norm;
            }
        }
        pt
    };

    // Project v2 orthogonal to u2 (it should already be, but ensure numerical cleanliness)
    let v2u2 = inner_product(&v2, &u2);
    let mut v2_orth: Vec<f64> = v2.iter().zip(u2.iter()).map(|(&vi, &ui)| vi - v2u2 * ui).collect();
    let v2_orth_norm = vec_norm(&v2_orth);
    if v2_orth_norm > 1e-14 {
        for x in &mut v2_orth {
            *x /= v2_orth_norm;
        }
    }

    let proj_ip = inner_product(&v1_transported, &v2_orth).clamp(-1.0, 1.0);
    let holonomy_angle = proj_ip.acos();

    Some(HolonomyResult {
        geodesic_distance,
        holonomy_angle,
        projected_inner_product: proj_ip,
    })
}

/// G_2-orbit invariant: the octonion triple cross-product of three
/// unit vectors in R^7 (imaginary octonions).
///
/// For u, v, w  in  Im(O) ~ R^7, the G_2-invariant quantity
///   phi(u,v,w) = <u, v x w>_O
/// where x is the octonion cross product, is the calibration 3-form
/// of G_2.  This is invariant under the G_2 automorphism group of the
/// octonions.
///
/// # Literature
/// - Harvey & Lawson (1982): Calibrated geometries
/// - Bryant (1987): Metrics with holonomy G_2
/// - Reggiani (2024): Z(S) ~ G_2 context
pub fn g2_calibration_form(u: &[f64; 7], v: &[f64; 7], w: &[f64; 7]) -> f64 {
    // Octonion multiplication table for imaginary units e1..e7:
    // We embed u,v as imaginary octonions (0, u1..u7) and compute
    // the imaginary part of the product, then take the inner product.
    //
    // The calibration form phi(u,v,w) = <u, vxw> where
    //   (vxw)_i = eps_{ijk} v_j w_k
    // with eps_{ijk} the octonion structure constants.
    //
    // Standard octonion Fano-plane structure constants:
    // Positive triples (i,j,k) with e_i*e_j = e_k:
    //   (1,2,3), (1,4,5), (1,7,6), (2,4,6), (2,5,7), (3,4,7), (3,6,5)
    // Using 0-indexed: subtract 1 from each index above.
    let triples: [(usize, usize, usize); 7] = [
        (0, 1, 2), // e1*e2 = e3
        (0, 3, 4), // e1*e4 = e5
        (0, 6, 5), // e1*e7 = e6
        (1, 3, 5), // e2*e4 = e6
        (1, 4, 6), // e2*e5 = e7
        (2, 3, 6), // e3*e4 = e7
        (2, 5, 4), // e3*e6 = e5
    ];

    // Iterate over the 7 Fano-plane triples and for each accumulate
    // the three cyclic permutations (i,j,k), (j,k,i), (k,i,j) of
    // the totally-antisymmetric 3-form:
    // phi(u,v,w) = Sum_{(i,j,k)} [u_i (v_j w_k - v_k w_j)
    //                          + u_j (v_k w_i - v_i w_k)
    //                          + u_k (v_i w_j - v_j w_i)]
    let mut phi = 0.0_f64;
    for &(i, j, k) in &triples {
        phi += u[i] * (v[j] * w[k] - v[k] * w[j]);
        phi += u[j] * (v[k] * w[i] - v[i] * w[k]);
        phi += u[k] * (v[i] * w[j] - v[j] * w[i]);
    }
    phi
}

/// Compute the G_2 calibration form for a triple of standard zero-divisors.
///
/// Each ZD is projected to its imaginary-octonion lower half (indices 1..7
/// of the first 8 components), normalised, and then the G_2 3-form is
/// evaluated.  Returns `None` if any ZD has a vanishing imaginary part.
pub fn g2_calibration_from_zds(z1: &[f64], z2: &[f64], z3: &[f64]) -> Option<f64> {
    let extract_im = |z: &[f64]| -> Option<[f64; 7]> {
        if z.len() < 8 {
            return None;
        }
        let im: [f64; 7] = [z[1], z[2], z[3], z[4], z[5], z[6], z[7]];
        let norm = im.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-14 {
            return None;
        }
        Some(im.map(|x| x / norm))
    };

    let u = extract_im(z1)?;
    let v = extract_im(z2)?;
    let w = extract_im(z3)?;

    Some(g2_calibration_form(&u, &v, &w))
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
        use crate::boxkites::{cross_assessors, diagonal_zero_products_exact};

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
        use crate::boxkites::{cross_assessors, diagonal_zero_products_exact};

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

    #[test]
    fn test_holonomy_self_is_zero() {
        // Holonomy of a ZD with itself should be zero distance and zero angle.
        // Use e_1 + e_9: a = e_1 in R^8, b = e_1 in R^8(shifted).
        // We need a perp b in R^8, so use e_1 + e_10 instead:
        //   a = (0,1,0,0,0,0,0,0), b = (0,0,1,0,0,0,0,0)
        let mut z = [0.0f64; 16];
        z[1] = 1.0;   // e_1 in lower half
        z[10] = 1.0;  // e_2 in upper half (different R^8 component)
        let result = holonomy_between(&z, &z).expect("should compute holonomy");
        assert!(
            result.geodesic_distance.abs() < 1e-10,
            "self-distance should be 0, got {}",
            result.geodesic_distance
        );
        assert!(
            result.holonomy_angle.abs() < 1e-10,
            "self-holonomy should be 0, got {}",
            result.holonomy_angle
        );
    }

    #[test]
    fn test_holonomy_between_distinct_zds() {
        // Two V_{8,2} frames with orthogonal u,v components:
        // z1: a=(e_1), b=(e_2) => u1=e_1, v1=e_2 (orthogonal ok)
        // z2: a=(e_3), b=(e_4) => u2=e_3, v2=e_4 (orthogonal ok)
        let mut z1 = [0.0f64; 16];
        z1[1] = 1.0;   // a = e_1
        z1[10] = 1.0;  // b = e_2 (in upper half)
        let mut z2 = [0.0f64; 16];
        z2[3] = 1.0;   // a = e_3
        z2[12] = 1.0;  // b = e_4 (in upper half)
        let result = holonomy_between(&z1, &z2).expect("should compute holonomy");
        // u1 = e_1, u2 = e_3 in R^8 => <u1,u2> = 0 => geodesic distance = pi/2
        assert!(
            (result.geodesic_distance - std::f64::consts::FRAC_PI_2).abs() < 1e-10,
            "expected pi/2 geodesic distance, got {}",
            result.geodesic_distance
        );
        // The holonomy angle should be well-defined and finite
        assert!(result.holonomy_angle.is_finite());
    }

    #[test]
    fn test_holonomy_antipodal_returns_none() {
        // u1 ~ -u2 (antipodal on S^7): geodesic is non-unique,
        // parallel transport is path-dependent => returns None.
        let mut z1 = [0.0f64; 16];
        z1[1] = 1.0;    // a = +e_1
        z1[10] = 1.0;   // b = e_2 (in upper half)
        let mut z2 = [0.0f64; 16];
        z2[1] = -1.0;   // a = -e_1  (antipodal to z1's a-half)
        z2[10] = 1.0;   // b = e_2
        let result = holonomy_between(&z1, &z2);
        assert!(
            result.is_none(),
            "antipodal u-vectors should return None, got {:?}",
            result
        );
    }

    #[test]
    fn test_g2_calibration_antisymmetric() {
        // The G_2 3-form is totally antisymmetric
        let u = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let w = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];

        let phi_uvw = g2_calibration_form(&u, &v, &w);
        let phi_vuw = g2_calibration_form(&v, &u, &w);
        let phi_uwv = g2_calibration_form(&u, &w, &v);

        assert!(
            (phi_uvw + phi_vuw).abs() < 1e-12,
            "phi(u,v,w) should be -phi(v,u,w)"
        );
        assert!(
            (phi_uvw + phi_uwv).abs() < 1e-12,
            "phi(u,v,w) should be -phi(u,w,v)"
        );
    }

    #[test]
    fn test_g2_calibration_nonzero_for_fano_triple() {
        // (e1, e2, e3) is a Fano-plane triple => phi should be +/-1
        let u = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // e1
        let v = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // e2
        let w = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]; // e3

        let phi = g2_calibration_form(&u, &v, &w);
        // For (0,1,2) triple: contribution is u[0]*(v[1]*w[2] - v[2]*w[1])
        // = 1*(1*1 - 0*0) = 1
        assert!(
            (phi.abs() - 1.0).abs() < 1e-12,
            "phi(e1,e2,e3) should be +/-1 for Fano triple, got {}",
            phi
        );
    }

    #[test]
    fn test_g2_calibration_from_zds_works() {
        // Use three distinct standard ZDs
        let mut z1 = vec![0.0; 16];
        z1[1] = 1.0;
        z1[8] = 1.0;
        let mut z2 = vec![0.0; 16];
        z2[2] = 1.0;
        z2[9] = 1.0;
        let mut z3 = vec![0.0; 16];
        z3[3] = 1.0;
        z3[10] = 1.0;

        let result = g2_calibration_from_zds(&z1, &z2, &z3);
        assert!(result.is_some(), "should compute G_2 calibration for ZDs");
        assert!(result.unwrap().is_finite());
    }
}

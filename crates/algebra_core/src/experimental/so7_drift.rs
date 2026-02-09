//! SO(7) rotation drift analysis for sedenion zero divisors (C-090).
//!
//! The zero-divisor structure of sedenion algebras is NOT invariant under
//! arbitrary SO(7) rotations of the imaginary octonion subspace. Only the
//! G2 automorphism subgroup (14-dimensional) of SO(7) preserves the full
//! octonionic multiplication table.
//!
//! This module measures the "drift" -- how much the zero-divisor product
//! norm ||a'*b'|| deviates from zero after applying SO(7) rotations to
//! a known ZD pair.
//!
//! # References
//! - Baez, J.C. (2002), "The Octonions", Bull. AMS 39, 145-205
//! - Harvey, F.R. (1990), "Spinors and Calibrations", Ch. 6

use nalgebra::{DMatrix, DVector};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use crate::construction::cayley_dickson::{cd_multiply, cd_norm_sq};

/// Canonical sedenion zero-divisor pair: a = e1 + e10, b = e4 - e15.
///
/// These satisfy a*b = 0 in the standard Cayley-Dickson basis.
pub fn sedenion_zd_pair() -> (Vec<f64>, Vec<f64>) {
    let mut a = vec![0.0; 16];
    a[1] = 1.0;
    a[10] = 1.0;

    let mut b = vec![0.0; 16];
    b[4] = 1.0;
    b[15] = -1.0;

    (a, b)
}

/// Generate a random SO(7) rotation matrix using Mezzadri's QR method.
///
/// `angle_scale` controls the magnitude: 0.0 = identity, 1.0 = full random.
/// For intermediate values, we interpolate: R(s) = (1-s)*I + s*R_random,
/// then re-orthogonalize via QR.
pub fn random_so7_rotation(rng: &mut impl Rng, angle_scale: f64) -> DMatrix<f64> {
    let n = 7;

    if angle_scale < 1e-12 {
        return DMatrix::identity(n, n);
    }

    // Step 1: Generate random 7x7 matrix with normal entries.
    let z = DMatrix::from_fn(n, n, |_, _| {
        use rand_distr::{Distribution, StandardNormal};
        StandardNormal.sample(rng)
    });

    // Step 2: QR decomposition (Mezzadri's algorithm for Haar-uniform).
    let qr = z.qr();
    let mut q: DMatrix<f64> = qr.q();
    let r: DMatrix<f64> = qr.r();

    // Step 3: Adjust signs so diagonal of R is positive (Mezzadri trick).
    for j in 0..n {
        if r[(j, j)] < 0.0 {
            for i in 0..n {
                q[(i, j)] = -q[(i, j)];
            }
        }
    }

    // Step 4: Ensure det(Q) = +1 (SO(7), not O(7)).
    if q.determinant() < 0.0 {
        for i in 0..n {
            q[(i, 0)] = -q[(i, 0)];
        }
    }

    // Step 5: Scale by angle_scale via interpolation then re-orthogonalize.
    if angle_scale < 1.0 {
        let identity = DMatrix::identity(n, n);
        let interpolated = &identity * (1.0 - angle_scale) + &q * angle_scale;
        // Re-orthogonalize via QR.
        let qr2 = interpolated.qr();
        let mut q2 = qr2.q();
        let r2 = qr2.r();
        for j in 0..n {
            if r2[(j, j)] < 0.0 {
                for i in 0..n {
                    q2[(i, j)] = -q2[(i, j)];
                }
            }
        }
        if q2.determinant() < 0.0 {
            for i in 0..n {
                q2[(i, 0)] = -q2[(i, 0)];
            }
        }
        q2
    } else {
        q
    }
}

/// Apply SO(7) rotation to the imaginary octonion part (indices 1..8) of a sedenion.
///
/// The sedenion has 16 components. The rotation acts on indices 1..8 (the 7
/// imaginary parts of the left octonion half). Indices 0, 8..16 are unchanged.
pub fn apply_so7_rotation(x: &[f64], r: &DMatrix<f64>) -> Vec<f64> {
    assert_eq!(x.len(), 16);
    assert_eq!(r.nrows(), 7);
    assert_eq!(r.ncols(), 7);

    let mut result = x.to_vec();

    // Extract imaginary part of left octonion half (indices 1..8).
    let im = DVector::from_fn(7, |i, _| x[i + 1]);
    let rotated = r * im;
    for i in 0..7 {
        result[i + 1] = rotated[i];
    }

    result
}

/// Result of measuring ZD drift under SO(7) rotations at a single angle scale.
#[derive(Debug, Clone)]
pub struct ZdDriftResult {
    /// Angle scale parameter.
    pub angle_scale: f64,
    /// Number of rotations tested.
    pub n_rotations: usize,
    /// Mean product norm ||a'*b'|| across rotations.
    pub mean_product_norm: f64,
    /// Max product norm ||a'*b'|| across rotations.
    pub max_product_norm: f64,
    /// Fraction of rotations where ||a'*b'|| > threshold.
    pub fraction_broken: f64,
}

/// Measure ZD drift under random SO(7) rotations at a single angle scale.
///
/// Returns product norms for each rotation trial.
pub fn measure_zd_drift(
    n_rotations: usize,
    angle_scale: f64,
    seed: u64,
) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let (a, b) = sedenion_zd_pair();

    let mut norms = Vec::with_capacity(n_rotations);
    for _ in 0..n_rotations {
        let r = random_so7_rotation(&mut rng, angle_scale);
        let a_rot = apply_so7_rotation(&a, &r);
        let b_rot = apply_so7_rotation(&b, &r);
        let product = cd_multiply(&a_rot, &b_rot);
        norms.push(cd_norm_sq(&product).sqrt());
    }
    norms
}

/// Sweep angle scales and measure ZD drift at each.
pub fn angle_sweep(
    scales: &[f64],
    n_rotations: usize,
    seed: u64,
) -> Vec<ZdDriftResult> {
    scales
        .iter()
        .map(|&s| {
            let norms = measure_zd_drift(n_rotations, s, seed);
            let n = norms.len() as f64;
            let mean = norms.iter().sum::<f64>() / n;
            let max = norms.iter().cloned().fold(0.0f64, f64::max);
            let broken = norms.iter().filter(|&&x| x > 1e-6).count() as f64 / n;

            ZdDriftResult {
                angle_scale: s,
                n_rotations,
                mean_product_norm: mean,
                max_product_norm: max,
                fraction_broken: broken,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_zd_pair_is_zero_divisor() {
        let (a, b) = sedenion_zd_pair();
        let product = cd_multiply(&a, &b);
        let norm = cd_norm_sq(&product).sqrt();
        assert!(
            norm < 1e-10,
            "Canonical pair should be ZD: ||a*b|| = {norm:.3e}"
        );
    }

    #[test]
    fn test_identity_rotation_preserves_zd() {
        let (a, b) = sedenion_zd_pair();
        let identity = DMatrix::identity(7, 7);
        let a_rot = apply_so7_rotation(&a, &identity);
        let b_rot = apply_so7_rotation(&b, &identity);
        let product = cd_multiply(&a_rot, &b_rot);
        let norm = cd_norm_sq(&product).sqrt();
        assert!(
            norm < 1e-10,
            "Identity rotation should preserve ZD: ||a'*b'|| = {norm:.3e}"
        );
    }

    #[test]
    fn test_zero_angle_scale_preserves_zd() {
        let norms = measure_zd_drift(10, 0.0, 42);
        for (i, n) in norms.iter().enumerate() {
            assert!(
                *n < 1e-10,
                "Zero angle scale trial {i}: norm = {n:.3e}"
            );
        }
    }

    #[test]
    fn test_full_rotation_breaks_zd() {
        // Generic SO(7) rotations should break the ZD condition.
        let norms = measure_zd_drift(50, 1.0, 42);
        let broken = norms.iter().filter(|&&x| x > 1e-6).count();
        assert!(
            broken > 25,
            "Full SO(7) rotations should mostly break ZD: {broken}/50 broken"
        );
    }

    #[test]
    fn test_drift_increases_with_angle_scale() {
        let scales = [0.0, 0.1, 0.5, 1.0];
        let results = angle_sweep(&scales, 50, 42);

        // Mean drift should increase with angle scale.
        assert!(
            results[0].mean_product_norm < 1e-10,
            "Zero scale should have zero drift"
        );
        assert!(
            results[3].mean_product_norm > results[1].mean_product_norm,
            "Full scale should have more drift than small scale"
        );
    }

    #[test]
    fn test_so7_rotation_is_orthogonal() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let r = random_so7_rotation(&mut rng, 1.0);

        // R^T * R should be identity.
        let rtr = r.transpose() * &r;
        let identity = DMatrix::identity(7, 7);
        let diff = (&rtr - &identity).norm();
        assert!(
            diff < 1e-10,
            "R^T R should be identity: ||R^T R - I|| = {diff:.3e}"
        );

        // det(R) should be +1.
        let det = r.determinant();
        assert!(
            (det - 1.0).abs() < 1e-10,
            "det(R) should be +1: det = {det:.6}"
        );
    }
}

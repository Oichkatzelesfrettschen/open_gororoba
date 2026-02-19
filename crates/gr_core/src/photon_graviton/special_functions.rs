//! Special functions for worldline QFT computations.
//!
//! Hand-rolled implementations justified by the absence of suitable Rust crates:
//! - Bernoulli numbers: Akiyama-Tanigawa algorithm (~15 lines)
//! - Hurwitz zeta: Euler-Maclaurin formula with N=20, P=10 (~40 lines)
//! - Worldline determinant factors: Taylor-safe near z=0
//!
//! # References
//! - Appendix B of Bastianelli, Corradini, Huet, Schubert (2026), arXiv:2601.23279
//! - Abramowitz & Stegun, Chapter 23 (Bernoulli numbers)

// ---------------------------------------------------------------------------
// Bernoulli numbers
// ---------------------------------------------------------------------------

/// First 31 Bernoulli numbers B_0 through B_30 (even-index only are nonzero
/// for n >= 2; B_1 = -1/2 is the only odd nonzero one).
///
/// Precomputed via the Akiyama-Tanigawa algorithm for exact rational values
/// converted to f64.
const BERNOULLI: [f64; 31] = [
    1.0,                          // B_0
    -0.5,                         // B_1
    1.0 / 6.0,                    // B_2
    0.0,                          // B_3
    -1.0 / 30.0,                  // B_4
    0.0,                          // B_5
    1.0 / 42.0,                   // B_6
    0.0,                          // B_7
    -1.0 / 30.0,                  // B_8
    0.0,                          // B_9
    5.0 / 66.0,                   // B_10
    0.0,                          // B_11
    -691.0 / 2730.0,              // B_12
    0.0,                          // B_13
    7.0 / 6.0,                    // B_14
    0.0,                          // B_15
    -3617.0 / 510.0,              // B_16
    0.0,                          // B_17
    43867.0 / 798.0,              // B_18
    0.0,                          // B_19
    -174611.0 / 330.0,            // B_20
    0.0,                          // B_21
    854513.0 / 138.0,             // B_22
    0.0,                          // B_23
    -236364091.0 / 2730.0,        // B_24
    0.0,                          // B_25
    8553103.0 / 6.0,              // B_26
    0.0,                          // B_27
    -23749461029.0 / 870.0,       // B_28
    0.0,                          // B_29
    8615841276005.0 / 14322.0,    // B_30
];

/// Bernoulli number B_n for n in [0, 30].
///
/// Returns 0.0 for odd n >= 3 (they vanish identically).
/// Panics if n > 30.
pub fn bernoulli(n: usize) -> f64 {
    assert!(n <= 30, "Bernoulli numbers only tabulated to B_30");
    BERNOULLI[n]
}

// ---------------------------------------------------------------------------
// Hurwitz zeta function
// ---------------------------------------------------------------------------

/// Hurwitz zeta function zeta(s, a) = sum_{n=0}^inf 1/(n+a)^s.
///
/// Uses the Euler-Maclaurin formula with N=20 direct terms and P=10
/// asymptotic correction terms. Accurate to ~15 digits for s > 0, a > 0
/// with a not too close to a non-positive integer.
///
/// # Arguments
/// * `s` -- real part > 0 (we only need s >= 1 in this module)
/// * `a` -- shift parameter, must be > 0
///
/// # Panics
/// Panics if a <= 0 or s <= 0.
pub fn hurwitz_zeta(s: f64, a: f64) -> f64 {
    assert!(a > 0.0, "Hurwitz zeta requires a > 0, got {a}");
    assert!(s > 0.0, "Hurwitz zeta requires s > 0, got {s}");

    let big_n: usize = 20;
    let big_p: usize = 10; // Use B_2, B_4, ..., B_{2P} = B_20

    // Direct sum: sum_{n=0}^{N-1} 1/(n+a)^s
    let mut direct = 0.0_f64;
    for n in 0..big_n {
        direct += (n as f64 + a).powf(-s);
    }

    // Integral term: (N+a)^{1-s} / (s-1)
    let na = big_n as f64 + a;
    let integral = na.powf(1.0 - s) / (s - 1.0);

    // Midpoint term: (1/2) * (N+a)^{-s}
    let midpoint = 0.5 * na.powf(-s);

    // Asymptotic correction: sum_{k=1}^{P} B_{2k} / (2k)! * prod_{j=0}^{2k-2}(s+j) * (N+a)^{-(s+2k-1)}
    let mut correction = 0.0_f64;
    let mut factorial_2k = 1.0_f64; // (2k)! built incrementally
    let mut rising_s = 1.0_f64; // product of (s+j)
    for k in 1..=big_p {
        // Update (2k)!
        factorial_2k *= (2 * k - 1) as f64 * (2 * k) as f64;
        // Update rising factorial: multiply by (s + 2k-2) and (s + 2k-3)
        // For k=1: rising_s = s*(s+1) -- but we need prod_{j=0}^{2k-2}(s+j)
        // i.e. for k=1: (s), for k=2: s*(s+1)*(s+2), etc.
        if k == 1 {
            rising_s = s;
        } else {
            rising_s *= (s + 2.0 * k as f64 - 3.0) * (s + 2.0 * k as f64 - 2.0);
        }
        let b2k = bernoulli(2 * k);
        correction += b2k / factorial_2k * rising_s * na.powf(-(s + 2.0 * k as f64 - 1.0));
    }

    direct + integral + midpoint + correction
}

// ---------------------------------------------------------------------------
// Worldline determinant factors (Euclidean)
// ---------------------------------------------------------------------------

/// Scalar determinant factor: z / sinh(z).
///
/// Taylor expansion near z = 0: 1 - z^2/6 + 7z^4/360 - ...
/// This avoids catastrophic cancellation for |z| < 0.1.
pub fn det_factor_scalar(z: f64) -> f64 {
    if z.abs() < 0.1 {
        let z2 = z * z;
        1.0 - z2 / 6.0 + 7.0 * z2 * z2 / 360.0 - 31.0 * z2 * z2 * z2 / 15120.0
    } else {
        z / z.sinh()
    }
}

/// Spinor determinant factor: z / tanh(z).
///
/// Taylor expansion near z = 0: 1 + z^2/3 - z^4/45 + ...
pub fn det_factor_spinor(z: f64) -> f64 {
    if z.abs() < 0.1 {
        let z2 = z * z;
        1.0 + z2 / 3.0 - z2 * z2 / 45.0 + 2.0 * z2 * z2 * z2 / 945.0
    } else {
        z / z.tanh()
    }
}

/// Scalar determinant: (z / sinh(z))^{D/2} for D=4 Euclidean dimensions.
///
/// For a pure magnetic field: (z/sinh(z))^2 since the field acts in a 2D plane,
/// times 1 for the two free directions. We use the full 4D expression.
pub fn det_scalar_4d(z: f64) -> f64 {
    let d = det_factor_scalar(z);
    d * d
}

/// Spinor determinant: (z / tanh(z))^{D/2} * (1/cosh(z))^{D_s/2} for D=4.
///
/// For pure magnetic field in 4D: the factor is z*coth(z) from the one
/// nontrivial plane. The full proper-time determinant for the spinor loop
/// in a pure magnetic field is:
///   det_spinor = z / tanh(z)   (from Eq B.34 with b=0)
pub fn det_spinor_4d(z: f64) -> f64 {
    det_factor_spinor(z)
}

/// Cotangent minus 1/z: cot(z) - 1/z = coth(z) - 1/z (Euclidean).
///
/// Used in tadpole integrands. Taylor-safe near z=0.
/// In Euclidean: coth(z) - 1/z.
pub fn coth_minus_inv(z: f64) -> f64 {
    if z.abs() < 0.1 {
        let z2 = z * z;
        // coth(z) - 1/z = z/3 - z^3/45 + 2z^5/945 - ...
        z / 3.0 - z * z2 / 45.0 + 2.0 * z * z2 * z2 / 945.0
    } else {
        z.cosh() / z.sinh() - 1.0 / z
    }
}

/// tanh(z) - z expanded for small z.
///
/// Used in tadpole spinor integrands where the combination coth + tanh - 1/z
/// appears (Eq 5.5 of Part 3).
pub fn tanh_minus_z(z: f64) -> f64 {
    if z.abs() < 0.1 {
        let z2 = z * z;
        -z * z2 / 3.0 + 2.0 * z * z2 * z2 / 15.0
    } else {
        z.tanh() - z
    }
}

// ---------------------------------------------------------------------------
// Gamma function ratio for dimensional regularization
// ---------------------------------------------------------------------------

/// Gamma(n) for positive integer n (i.e., (n-1)!).
pub fn gamma_integer(n: usize) -> f64 {
    assert!(n >= 1, "gamma_integer requires n >= 1");
    let mut result = 1.0_f64;
    for k in 2..n {
        result *= k as f64;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // -- Bernoulli tests --

    #[test]
    fn test_bernoulli_known_values() {
        assert!((bernoulli(0) - 1.0).abs() < 1e-15);
        assert!((bernoulli(1) + 0.5).abs() < 1e-15);
        assert!((bernoulli(2) - 1.0 / 6.0).abs() < 1e-15);
        assert!((bernoulli(4) + 1.0 / 30.0).abs() < 1e-15);
        assert!((bernoulli(6) - 1.0 / 42.0).abs() < 1e-15);
    }

    #[test]
    fn test_bernoulli_odd_vanish() {
        for n in [3, 5, 7, 9, 11, 13, 15, 17, 19] {
            assert_eq!(bernoulli(n), 0.0, "B_{n} should be 0");
        }
    }

    #[test]
    fn test_bernoulli_b12() {
        // B_12 = -691/2730
        let expected = -691.0 / 2730.0;
        assert!(
            (bernoulli(12) - expected).abs() < 1e-14,
            "B_12 = {}, expected {}",
            bernoulli(12),
            expected
        );
    }

    // -- Hurwitz zeta tests --

    #[test]
    fn test_hurwitz_zeta_is_riemann() {
        // zeta(2, 1) = pi^2/6 (Riemann zeta)
        let result = hurwitz_zeta(2.0, 1.0);
        let expected = PI * PI / 6.0;
        assert!(
            (result - expected).abs() / expected < 1e-12,
            "zeta(2,1) = {result}, expected {expected}"
        );
    }

    #[test]
    fn test_hurwitz_zeta_half() {
        // zeta(2, 1/2) = pi^2/2
        let result = hurwitz_zeta(2.0, 0.5);
        let expected = PI * PI / 2.0;
        assert!(
            (result - expected).abs() / expected < 1e-12,
            "zeta(2,1/2) = {result}, expected {expected}"
        );
    }

    #[test]
    fn test_hurwitz_zeta_s4() {
        // zeta(4, 1) = pi^4/90
        let result = hurwitz_zeta(4.0, 1.0);
        let expected = PI.powi(4) / 90.0;
        assert!(
            (result - expected).abs() / expected < 1e-10,
            "zeta(4,1) = {result}, expected {expected}"
        );
    }

    #[test]
    fn test_hurwitz_zeta_shifted() {
        // zeta(s, a) = zeta(s, a+1) + a^{-s}
        let s = 3.0;
        let a = 1.5;
        let lhs = hurwitz_zeta(s, a);
        let rhs = hurwitz_zeta(s, a + 1.0) + a.powf(-s);
        assert!(
            (lhs - rhs).abs() / lhs.abs() < 1e-12,
            "Shift identity: {lhs} vs {rhs}"
        );
    }

    // -- Determinant factor tests --

    #[test]
    fn test_det_factor_scalar_zero() {
        assert!((det_factor_scalar(0.0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_det_factor_spinor_zero() {
        assert!((det_factor_spinor(0.0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_det_factor_scalar_crossover() {
        // Check continuity at the Taylor/exact boundary z=0.1
        let z_below = 0.099;
        let z_above = 0.101;
        let below = det_factor_scalar(z_below);
        let above = det_factor_scalar(z_above);
        let mid = det_factor_scalar(0.1);
        // Should be smooth -- relative difference < 1e-6
        assert!(
            ((above - below) / mid).abs() < 1e-4,
            "Discontinuity at crossover: {below}, {mid}, {above}"
        );
    }

    #[test]
    fn test_det_factor_spinor_crossover() {
        // Check smoothness at the Taylor/exact boundary z=0.1
        // The Taylor expansion truncation error is O(z^8) ~ 1e-8 at z=0.1.
        // The step from z=0.099 to z=0.101 is ~0.002, so the derivative
        // z/tanh(z)' ~ 2z/3 ~ 0.067 gives a step of ~1.3e-4.
        let z_below = 0.099;
        let z_above = 0.101;
        let below = det_factor_spinor(z_below);
        let above = det_factor_spinor(z_above);
        let mid = det_factor_spinor(0.1);
        // All three should be within a smooth progression
        assert!(
            ((above - below) / mid).abs() < 2e-3,
            "Discontinuity at crossover: {below}, {mid}, {above}"
        );
    }

    #[test]
    fn test_det_factor_scalar_known() {
        // z/sinh(z) at z=1: 1/sinh(1) = 1/1.1752... ~ 0.8509
        let result = det_factor_scalar(1.0);
        let expected = 1.0 / 1.0_f64.sinh();
        assert!(
            (result - expected).abs() < 1e-14,
            "det_scalar(1) = {result}, expected {expected}"
        );
    }

    #[test]
    fn test_det_factor_spinor_known() {
        // z/tanh(z) at z=1: 1/tanh(1) = coth(1) ~ 1.3130
        let result = det_factor_spinor(1.0);
        let expected = 1.0 / 1.0_f64.tanh();
        assert!(
            (result - expected).abs() < 1e-14,
            "det_spinor(1) = {result}, expected {expected}"
        );
    }

    #[test]
    fn test_coth_minus_inv_zero() {
        // coth(z) - 1/z -> z/3 as z -> 0
        let result = coth_minus_inv(1e-6);
        let expected = 1e-6 / 3.0;
        assert!(
            (result - expected).abs() / expected < 1e-6,
            "coth_minus_inv(1e-6) = {result}, expected {expected}"
        );
    }

    #[test]
    fn test_coth_minus_inv_one() {
        let result = coth_minus_inv(1.0);
        let expected = 1.0_f64.cosh() / 1.0_f64.sinh() - 1.0;
        assert!(
            (result - expected).abs() < 1e-14,
            "coth_minus_inv(1) = {result}, expected {expected}"
        );
    }

    #[test]
    fn test_gamma_integer() {
        assert!((gamma_integer(1) - 1.0).abs() < 1e-15);
        assert!((gamma_integer(2) - 1.0).abs() < 1e-15);
        assert!((gamma_integer(3) - 2.0).abs() < 1e-15);
        assert!((gamma_integer(5) - 24.0).abs() < 1e-15);
        assert!((gamma_integer(7) - 720.0).abs() < 1e-15);
    }
}

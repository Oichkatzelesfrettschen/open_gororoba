//! Algebraic identity tests: power-associativity, flexibility, quadratic (Steps 33-35).
//!
//! Implements the key identities from Schafer (1961) Chapters III and V:
//! - Power-associativity: x^2*x = x*x^2 and x^2*x^2 = x*(x*x^2)
//! - Flexible identity: (xy)x = x(yx)
//! - Quadratic identity: x^2 = t(x)*x - n(x)*1
//! - Norm-conjugate: x*conj(x) = n(x)*1
//!
//! Mirrors: CDPowerAssociative.v, DicksonCDProcess.v.

use cd_kernel::cayley_dickson::{cd_conjugate, cd_multiply, cd_norm_sq};

/// Check third-power associativity: x^2 * x = x * x^2.
/// Schafer (1961) Ch V, eq.3.
///
/// Returns the max component error.
pub fn third_power_error(x: &[f64]) -> f64 {
    let x2 = cd_multiply(x, x);
    let lhs = cd_multiply(&x2, x);
    let rhs = cd_multiply(x, &x2);
    lhs.iter()
        .zip(rhs.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

/// Check fourth-power associativity: x^2 * x^2 = x * (x * x^2).
/// Schafer (1961) Ch V, eq.4.
pub fn fourth_power_error(x: &[f64]) -> f64 {
    let x2 = cd_multiply(x, x);
    let lhs = cd_multiply(&x2, &x2);
    let x_x2 = cd_multiply(x, &x2);
    let rhs = cd_multiply(x, &x_x2);
    lhs.iter()
        .zip(rhs.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

/// Check the flexible identity: (xy)x = x(yx).
/// Holds in alternative algebras (dim <= 8), fails at dim >= 16.
pub fn flexible_error(x: &[f64], y: &[f64]) -> f64 {
    let xy = cd_multiply(x, y);
    let lhs = cd_multiply(&xy, x);
    let yx = cd_multiply(y, x);
    let rhs = cd_multiply(x, &yx);
    lhs.iter()
        .zip(rhs.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

/// Check norm-conjugate identity: x * conj(x) = n(x) * e_0.
/// Schafer (1961) Ch III, eq.27'.
pub fn norm_conjugate_error(x: &[f64]) -> f64 {
    let conj_x = cd_conjugate(x);
    let product = cd_multiply(x, &conj_x);
    let norm = cd_norm_sq(x);
    let mut max_err = 0.0_f64;
    for (i, &v) in product.iter().enumerate() {
        let expected = if i == 0 { norm } else { 0.0 };
        max_err = max_err.max((v - expected).abs());
    }
    max_err
}

/// Check the quadratic identity: x^2 - t(x)*x + n(x)*1 = 0.
/// Schafer (1961) Ch III, eq.25.
/// Where t(x) = 2*x[0] (trace) and n(x) = ||x||^2 (norm).
pub fn quadratic_identity_error(x: &[f64]) -> f64 {
    let x2 = cd_multiply(x, x);
    let trace = 2.0 * x[0];
    let norm = cd_norm_sq(x);

    let mut max_err = 0.0_f64;
    for i in 0..x.len() {
        let x2_i = x2[i];
        let tx_i = trace * x[i];
        let n1_i = if i == 0 { norm } else { 0.0 };
        // x^2 - t(x)*x + n(x)*1 should be 0
        let residual = (x2_i - tx_i + n1_i).abs();
        max_err = max_err.max(residual);
    }
    max_err
}

/// Test all identities for basis elements at a given dimension.
/// Returns (max_3rd_power, max_4th_power, max_flex, max_norm_conj, max_quad).
pub fn test_all_identities(dim: usize) -> (f64, f64, f64, f64, f64) {
    let mut max3 = 0.0_f64;
    let mut max4 = 0.0_f64;
    let mut max_flex = 0.0_f64;
    let mut max_nc = 0.0_f64;
    let mut max_quad = 0.0_f64;

    for i in 0..dim {
        let mut x = vec![0.0; dim];
        x[i] = 1.0;
        max3 = max3.max(third_power_error(&x));
        max4 = max4.max(fourth_power_error(&x));
        max_nc = max_nc.max(norm_conjugate_error(&x));
        max_quad = max_quad.max(quadratic_identity_error(&x));

        for j in 0..dim {
            let mut y = vec![0.0; dim];
            y[j] = 1.0;
            max_flex = max_flex.max(flexible_error(&x, &y));
        }
    }
    (max3, max4, max_flex, max_nc, max_quad)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power_assoc_dim4() {
        let (e3, e4, _, _, _) = test_all_identities(4);
        assert!(e3 < 1e-10, "quaternions: 3rd power assoc");
        assert!(e4 < 1e-10, "quaternions: 4th power assoc");
    }

    #[test]
    fn test_power_assoc_dim8() {
        let (e3, e4, _, _, _) = test_all_identities(8);
        assert!(e3 < 1e-10, "octonions: 3rd power assoc");
        assert!(e4 < 1e-10, "octonions: 4th power assoc");
    }

    #[test]
    fn test_power_assoc_dim16() {
        let (e3, e4, _, _, _) = test_all_identities(16);
        assert!(e3 < 1e-10, "sedenions: 3rd power assoc");
        assert!(e4 < 1e-10, "sedenions: 4th power assoc");
    }

    #[test]
    fn test_flexible_dim4() {
        let (_, _, flex, _, _) = test_all_identities(4);
        assert!(flex < 1e-10, "quaternions: flexible");
    }

    #[test]
    fn test_flexible_dim8() {
        let (_, _, flex, _, _) = test_all_identities(8);
        assert!(flex < 1e-10, "octonions: flexible");
    }

    #[test]
    fn test_norm_conjugate_dim16() {
        let (_, _, _, nc, _) = test_all_identities(16);
        assert!(nc < 1e-10, "sedenions: norm-conjugate");
    }

    #[test]
    fn test_quadratic_identity_dim16() {
        let (_, _, _, _, quad) = test_all_identities(16);
        assert!(quad < 1e-10, "sedenions: quadratic identity");
    }

    #[test]
    fn test_quadratic_identity_random() {
        // Test with a non-basis sedenion element
        let x = vec![
            0.5, 1.0, -0.3, 0.7, 0.0, -1.0, 0.4, 0.2, 0.1, -0.5, 0.8, 0.0, -0.6, 0.3, 0.9,
            -0.1,
        ];
        assert!(
            quadratic_identity_error(&x) < 1e-10,
            "quadratic identity for arbitrary sedenion"
        );
    }

    #[test]
    fn test_norm_conjugate_random() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert!(
            norm_conjugate_error(&x) < 1e-10,
            "norm-conjugate for arbitrary octonion"
        );
    }

    #[test]
    fn test_full_tower() {
        for dim in [1, 2, 4, 8, 16] {
            let (e3, e4, _, nc, quad) = test_all_identities(dim);
            assert!(e3 < 1e-10, "dim {dim}: 3rd power");
            assert!(e4 < 1e-10, "dim {dim}: 4th power");
            assert!(nc < 1e-10, "dim {dim}: norm-conjugate");
            assert!(quad < 1e-10, "dim {dim}: quadratic");
        }
    }
}

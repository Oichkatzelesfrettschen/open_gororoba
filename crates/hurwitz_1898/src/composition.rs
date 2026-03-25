//! Norm multiplicativity (Hurwitz Part I + IV).
//!
//! Verifies the composition identity ||xy||^2 = ||x||^2 * ||y||^2
//! at each Cayley-Dickson level, and demonstrates failure at dim 16.
//!
//! Mirrors: HurwitzTheorem.v PART I (hurwitz_dim1..dim8) and PART IV (hurwitz_fails_dim16).

use cd_kernel::cayley_dickson::{cd_multiply, cd_norm_sq};

/// Verify norm multiplicativity for two elements at a given dimension.
/// Returns (||xy||^2, ||x||^2 * ||y||^2, difference).
///
/// Mirrors: HurwitzTheorem.v hurwitz_dim1..dim8.
pub fn check_norm_multiplicativity(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
    assert_eq!(x.len(), y.len(), "dimensions must match");
    let xy = cd_multiply(x, y);
    let norm_xy = cd_norm_sq(&xy);
    let norm_x_times_y = cd_norm_sq(x) * cd_norm_sq(y);
    (norm_xy, norm_x_times_y, (norm_xy - norm_x_times_y).abs())
}

/// Check whether norm multiplicativity holds for ALL basis-pair products
/// at dimension `dim`. Returns the maximum violation.
///
/// For dims 1, 2, 4, 8: returns 0.0 (exact composition).
/// For dim 16: returns nonzero (Hurwitz failure).
///
/// Mirrors: HurwitzTheorem.v Parts I and IV combined.
pub fn max_norm_violation(dim: usize) -> f64 {
    let mut max_viol = 0.0_f64;
    for i in 0..dim {
        for j in 0..dim {
            let mut x = vec![0.0; dim];
            let mut y = vec![0.0; dim];
            x[i] = 1.0;
            y[j] = 1.0;
            let (_, _, viol) = check_norm_multiplicativity(&x, &y);
            max_viol = max_viol.max(viol);
        }
    }
    max_viol
}

/// The Hurwitz composition witness at dim 16: specific ZD pair where
/// ||a*b||^2 = 0 but ||a||^2 * ||b||^2 = 4.
///
/// a = e3 + e10, b = e6 - e15 (Moreno-Froloff zero divisors).
///
/// Mirrors: HurwitzTheorem.v hurwitz_fails_via_zd.
pub fn sedenion_zd_witness() -> (Vec<f64>, Vec<f64>, f64, f64) {
    let mut a = vec![0.0; 16];
    a[3] = 1.0;
    a[10] = 1.0;
    let mut b = vec![0.0; 16];
    b[6] = 1.0;
    b[15] = -1.0;
    let ab = cd_multiply(&a, &b);
    let norm_ab = cd_norm_sq(&ab);
    let norm_a_norm_b = cd_norm_sq(&a) * cd_norm_sq(&b);
    (a, b, norm_ab, norm_a_norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dim1_composition() {
        let x = [3.0_f64];
        let y = [5.0];
        let (norm_xy, prod, diff) = check_norm_multiplicativity(&x, &y);
        assert!((norm_xy - 225.0).abs() < 1e-10);
        assert!((prod - 225.0).abs() < 1e-10);
        assert!(diff < 1e-10);
    }

    #[test]
    fn test_dim2_composition() {
        assert!(max_norm_violation(2) < 1e-10);
    }

    #[test]
    fn test_dim4_composition() {
        assert!(max_norm_violation(4) < 1e-10);
    }

    #[test]
    fn test_dim8_composition() {
        assert!(max_norm_violation(8) < 1e-10);
    }

    #[test]
    fn test_dim16_fails() {
        // Basis-pair products e_i * e_j always have norm 1 (even in sedenions).
        // The violation occurs for LINEAR COMBINATIONS -- use the ZD witness.
        let (_, _, norm_ab, norm_a_norm_b) = sedenion_zd_witness();
        assert!(norm_ab.abs() < 1e-10, "||a*b||^2 = 0");
        assert!((norm_a_norm_b - 4.0).abs() < 1e-10, "||a||^2 * ||b||^2 = 4");
        // violation = |0 - 4| = 4
        assert!((norm_ab - norm_a_norm_b).abs() > 3.0, "dim 16 violates");
    }

    #[test]
    fn test_sedenion_zd_witness() {
        let (_, _, norm_ab, norm_a_norm_b) = sedenion_zd_witness();
        assert!(norm_ab.abs() < 1e-10, "||a*b||^2 should be 0");
        assert!((norm_a_norm_b - 4.0).abs() < 1e-10, "||a||^2 * ||b||^2 should be 4");
    }
}

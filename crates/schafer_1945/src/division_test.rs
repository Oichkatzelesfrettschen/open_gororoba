//! Division algebra tests (Schafer eq.2 and negative result).
//!
//! Tests whether the right multiplication operator R_z is nonsingular
//! for all nonzero z.  At dim 16 over R, it IS singular (zero divisors).
//!
//! Mirrors: SchaferDivAlg16.v Parts III-IV.

use cd_kernel::cayley_dickson::{cd_multiply, cd_norm_sq};

/// The Moreno-Froloff zero-divisor pair at dim 16.
/// a = e3 + e10, b = e6 - e15.
///
/// Mirrors: SchaferDivAlg16.v schafer_eq2_singular_witness.
pub fn zd_witness_16() -> (Vec<f64>, Vec<f64>) {
    let mut a = vec![0.0; 16];
    a[3] = 1.0;
    a[10] = 1.0;
    let mut b = vec![0.0; 16];
    b[6] = 1.0;
    b[15] = -1.0;
    (a, b)
}

/// Check if a dimension has zero divisors among basis-element sums.
/// Searches for x = e_i + e_j, y = e_k +/- e_l with x*y = 0.
///
/// Returns the first ZD pair found, or None.
///
/// Mirrors: SchaferDivAlg16.v schafer_negative_zd.
pub fn find_basis_sum_zd(dim: usize) -> Option<([usize; 2], [usize; 2], f64)> {
    for i in 1..dim {
        for j in (i + 1)..dim {
            let mut x = vec![0.0; dim];
            x[i] = 1.0;
            x[j] = 1.0;
            for k in 1..dim {
                for l in (k + 1)..dim {
                    for sign in [-1.0_f64, 1.0] {
                        let mut y = vec![0.0; dim];
                        y[k] = 1.0;
                        y[l] = sign;
                        let xy = cd_multiply(&x, &y);
                        let norm = cd_norm_sq(&xy);
                        if norm < 1e-20 {
                            return Some(([i, j], [k, l], sign));
                        }
                    }
                }
            }
        }
    }
    None
}

/// Test the Schafer division condition: is right-multiplication
/// by z injective (no nontrivial kernel) for a specific z?
///
/// For z = sed_zd_b, the kernel contains sed_zd_a (nonzero),
/// so R_z is singular and the algebra is not a division algebra.
///
/// Mirrors: SchaferDivAlg16.v schafer_eq2_singular_witness.
pub fn right_multiplication_is_singular(z: &[f64], witness: &[f64]) -> bool {
    let product = cd_multiply(witness, z);
    let norm = cd_norm_sq(&product);
    let witness_norm = cd_norm_sq(witness);
    norm < 1e-20 && witness_norm > 0.1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zd_witness_product_is_zero() {
        let (a, b) = zd_witness_16();
        let ab = cd_multiply(&a, &b);
        assert!(cd_norm_sq(&ab) < 1e-20, "a*b should be zero");
    }

    #[test]
    fn test_zd_witness_norms() {
        let (a, b) = zd_witness_16();
        assert!((cd_norm_sq(&a) - 2.0).abs() < 1e-10, "||a||^2 = 2");
        assert!((cd_norm_sq(&b) - 2.0).abs() < 1e-10, "||b||^2 = 2");
    }

    #[test]
    fn test_singular_right_multiplication() {
        let (a, b) = zd_witness_16();
        assert!(
            right_multiplication_is_singular(&b, &a),
            "R_b should be singular with a in kernel"
        );
    }

    #[test]
    fn test_find_basis_sum_zd_dim16() {
        let result = find_basis_sum_zd(16);
        assert!(result.is_some(), "dim 16 should have ZD among basis sums");
    }

    #[test]
    fn test_no_basis_sum_zd_dim8() {
        let result = find_basis_sum_zd(8);
        assert!(result.is_none(), "dim 8 should have no ZD among basis sums");
    }

    #[test]
    fn test_no_basis_sum_zd_dim4() {
        let result = find_basis_sum_zd(4);
        assert!(result.is_none(), "dim 4 should have no ZD");
    }
}

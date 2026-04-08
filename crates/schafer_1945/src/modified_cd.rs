//! Modified Cayley-Dickson multiplication (Schafer eq.1).
//!
//! The generalized CD product:
//!   (a + vb)(x + vy) = (ax + g*(y*bS)) + v(aS*y + x*b)
//!
//! where S is the standard involution (conjugate) and g is a fixed
//! element of the base algebra.  Standard CD is the case g = gamma (scalar).
//!
//! Mirrors: SchaferDivAlg16.v Part I.

use cd_kernel::cayley_dickson::{cd_conjugate, cd_multiply};

/// Compute the modified CD product with arbitrary element g.
///
/// Given a, b (left pair) and x, y (right pair) in an algebra of dimension n,
/// and a fixed element g of dimension n:
///
///   lo = a*x + g*(y * conj(b))
///   hi = conj(a)*y + x*b
///
/// Returns a 2n-dimensional result.
///
/// Mirrors: SchaferDivAlg16.v eq.1.
pub fn modified_cd_multiply(a: &[f64], b: &[f64], x: &[f64], y: &[f64], g: &[f64]) -> Vec<f64> {
    let n = a.len();
    assert!(b.len() == n && x.len() == n && y.len() == n && g.len() == n);

    // lo = a*x + g*(y * conj(b))
    let ax = cd_multiply(a, x);
    let conj_b = cd_conjugate(b);
    let y_conj_b = cd_multiply(y, &conj_b);
    let g_y_conj_b = cd_multiply(g, &y_conj_b);
    let lo: Vec<f64> = ax
        .iter()
        .zip(g_y_conj_b.iter())
        .map(|(u, v)| u + v)
        .collect();

    // hi = conj(a)*y + x*b
    let conj_a = cd_conjugate(a);
    let conj_a_y = cd_multiply(&conj_a, y);
    let xb = cd_multiply(x, b);
    let hi: Vec<f64> = conj_a_y.iter().zip(xb.iter()).map(|(u, v)| u + v).collect();

    let mut result = lo;
    result.extend_from_slice(&hi);
    result
}

/// NOTE: The standard CD formula (a,b)(c,d) = (ac - conj(d)*b, da + b*conj(c))
/// is NOT exactly Schafer's eq.1 with g = -1.  Schafer's formula uses
/// g*(y*conj(b)) in the lo component, while standard CD uses conj(y)*b.
/// These differ in non-commutative algebras because y*conj(b) != conj(y)*b.
///
/// The standard CD process corresponds to Schafer's "gamma" case with a
/// specific arrangement of conjugates.  The modified product with non-scalar
/// g is a GENUINE GENERALIZATION, not just a reparameterization.
#[cfg(test)]
mod tests {
    use super::*;
    use cd_kernel::cayley_dickson::cd_norm_sq;

    #[test]
    fn test_modified_cd_nonzero_product() {
        // With g = e_1 (non-scalar), the modified product should be nonzero
        // for generic inputs.
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 0.0];
        let x = vec![0.0, 0.0, 1.0, 0.0];
        let y = vec![0.0, 0.0, 0.0, 1.0];
        let g = vec![0.0, 1.0, 0.0, 0.0]; // e_1

        let result = modified_cd_multiply(&a, &b, &x, &y, &g);
        assert_eq!(result.len(), 8);
        assert!(
            cd_norm_sq(&result) > 0.1,
            "modified product should be nonzero"
        );
    }

    #[test]
    fn test_modified_cd_with_scalar_g() {
        // With g = gamma * e_0 (scalar), the modified product is a
        // variant of standard CD (different conjugate arrangement).
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let x = vec![1.0, 0.0];
        let y = vec![0.0, 0.0];
        let g = vec![-1.0, 0.0]; // scalar -1

        let result = modified_cd_multiply(&a, &b, &x, &y, &g);
        assert_eq!(result.len(), 4);
        // (1+v*i)(1+v*0) with g=-1: lo = 1*1 + (-1)*(0*conj(i)) = 1, hi = conj(1)*0 + 1*i = i
        // So result should be (1, 0, 0, 1) = 1 + v*i
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[3] - 1.0).abs() < 1e-10);
    }
}

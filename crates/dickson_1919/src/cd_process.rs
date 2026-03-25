//! The Cayley-Dickson doubling construction (Dickson eq.6).
//!
//! The CD process: given an algebra A of dimension n, produce an algebra
//! of dimension 2n by taking pairs (a, b) with multiplication:
//!   (a, b)(c, d) = (ac - conj(d)*b, da + b*conj(c))
//!
//! This generates: R(1) -> C(2) -> H(4) -> O(8) -> S(16) -> ...
//!
//! Mirrors: DicksonCDProcess.v Parts I-III; cd_kernel::cayley_dickson.

use cd_kernel::cayley_dickson::{cd_conjugate, cd_multiply, cd_norm_sq};

/// Apply one level of CD doubling: given two elements a, b of dimension n,
/// produce their product under the CD formula (eq.6).
///
/// (a, b)(c, d) = (ac - conj(d)*b, da + b*conj(c))
///
/// Input: a, b are the "left" pair; c, d are the "right" pair.
/// All four must have the same dimension n.
/// Output: a 2n-dimensional element.
///
/// Mirrors: DicksonCDProcess.v eq.6, Sedenion.v oct_mul definition.
pub fn cd_double_multiply(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Vec<f64> {
    let n = a.len();
    assert!(b.len() == n && c.len() == n && d.len() == n);

    // lo = a*c - conj(d)*b
    let ac = cd_multiply(a, c);
    let conj_d = cd_conjugate(d);
    let conj_d_b = cd_multiply(&conj_d, b);
    let lo: Vec<f64> = ac.iter().zip(conj_d_b.iter()).map(|(x, y)| x - y).collect();

    // hi = d*a + b*conj(c)
    let da = cd_multiply(d, a);
    let conj_c = cd_conjugate(c);
    let b_conj_c = cd_multiply(b, &conj_c);
    let hi: Vec<f64> = da.iter().zip(b_conj_c.iter()).map(|(x, y)| x + y).collect();

    let mut result = lo;
    result.extend_from_slice(&hi);
    result
}

/// Verify that the CD doubling produces the same result as the
/// direct cd_multiply at the doubled dimension.
///
/// This validates that cd_multiply at dim 2n is consistent with
/// the pair construction from dim n.
pub fn verify_cd_doubling_consistency(dim: usize) -> f64 {
    let half = dim / 2;
    let mut max_err = 0.0_f64;

    // Test with basis elements
    for i in 0..dim {
        for j in 0..dim {
            let mut x = vec![0.0; dim];
            let mut y = vec![0.0; dim];
            x[i] = 1.0;
            y[j] = 1.0;

            // Direct multiplication at dim
            let direct = cd_multiply(&x, &y);

            // Pair multiplication: x = (a, b), y = (c, d)
            let a = &x[..half];
            let b = &x[half..];
            let c = &y[..half];
            let d = &y[half..];
            let paired = cd_double_multiply(a, b, c, d);

            let err: f64 = direct
                .iter()
                .zip(paired.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            max_err = max_err.max(err);
        }
    }
    max_err
}

/// The Dickson norm identity: N(q+Qe) = N(q) + N(Q) = qq' + QQ'.
///
/// For a 2n-dimensional element x = (a, b):
///   N(x) = N(a) + N(b) = sum of all component squares.
///
/// Mirrors: DicksonCDProcess.v dickson_eq5.
pub fn verify_norm_is_sum_of_halves(x: &[f64]) -> bool {
    let dim = x.len();
    let half = dim / 2;
    let a = &x[..half];
    let b = &x[half..];
    let norm_x = cd_norm_sq(x);
    let norm_a = cd_norm_sq(a);
    let norm_b = cd_norm_sq(b);
    (norm_x - (norm_a + norm_b)).abs() < 1e-12
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cd_doubling_consistency_dim4() {
        assert!(verify_cd_doubling_consistency(4) < 1e-10);
    }

    #[test]
    fn test_cd_doubling_consistency_dim8() {
        assert!(verify_cd_doubling_consistency(8) < 1e-10);
    }

    #[test]
    fn test_cd_doubling_consistency_dim16() {
        assert!(verify_cd_doubling_consistency(16) < 1e-10);
    }

    #[test]
    fn test_norm_sum_of_halves() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert!(verify_norm_is_sum_of_halves(&x));
    }
}

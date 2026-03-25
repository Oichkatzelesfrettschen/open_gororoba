//! Property hierarchy of CD algebras (Dickson p.159).
//!
//! The CD tower progressively loses algebraic properties:
//!   R (dim 1): commutative, associative, ordered, division
//!   C (dim 2): commutative, associative, division (loses ordering)
//!   H (dim 4): associative, division (loses commutativity)
//!   O (dim 8): alternative, division (loses associativity)
//!   S (dim 16): power-associative (loses alternativity, gains zero divisors)
//!
//! This module tests each property at each level.
//!
//! Mirrors: DicksonCDProcess.v Parts II-III + PART V.

use cd_kernel::cayley_dickson::{cd_multiply, cd_norm_sq};

/// Test commutativity: x*y == y*x for all basis pairs.
/// Returns the maximum violation.
///
/// Mirrors: DicksonCDProcess.v dickson_complex_commutative, dickson_quat_not_commutative.
pub fn max_commutativity_violation(dim: usize) -> f64 {
    let mut max_viol = 0.0_f64;
    for i in 0..dim {
        for j in (i + 1)..dim {
            let mut x = vec![0.0; dim];
            let mut y = vec![0.0; dim];
            x[i] = 1.0;
            y[j] = 1.0;
            let xy = cd_multiply(&x, &y);
            let yx = cd_multiply(&y, &x);
            let err: f64 = xy
                .iter()
                .zip(yx.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            max_viol = max_viol.max(err);
        }
    }
    max_viol
}

/// Test associativity: (x*y)*z == x*(y*z) for all basis triples.
/// Returns the maximum violation.
///
/// Mirrors: DicksonCDProcess.v dickson_quat_associative, dickson_oct_not_associative.
pub fn max_associativity_violation(dim: usize) -> f64 {
    let mut max_viol = 0.0_f64;
    for i in 0..dim {
        for j in 0..dim {
            for k in 0..dim {
                let mut x = vec![0.0; dim];
                let mut y = vec![0.0; dim];
                let mut z = vec![0.0; dim];
                x[i] = 1.0;
                y[j] = 1.0;
                z[k] = 1.0;
                let xy = cd_multiply(&x, &y);
                let xy_z = cd_multiply(&xy, &z);
                let yz = cd_multiply(&y, &z);
                let x_yz = cd_multiply(&x, &yz);
                let err: f64 = xy_z
                    .iter()
                    .zip(x_yz.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max);
                max_viol = max_viol.max(err);
            }
        }
    }
    max_viol
}

/// Test norm multiplicativity: ||x*y||^2 == ||x||^2 * ||y||^2.
/// Returns the maximum violation over all basis pairs.
///
/// Mirrors: DicksonCDProcess.v dickson_eq4, dickson_eq5.
pub fn max_norm_violation(dim: usize) -> f64 {
    let mut max_viol = 0.0_f64;
    for i in 0..dim {
        for j in 0..dim {
            let mut x = vec![0.0; dim];
            let mut y = vec![0.0; dim];
            x[i] = 1.0;
            y[j] = 1.0;
            let xy = cd_multiply(&x, &y);
            let norm_xy = cd_norm_sq(&xy);
            let norm_x_y = cd_norm_sq(&x) * cd_norm_sq(&y);
            max_viol = max_viol.max((norm_xy - norm_x_y).abs());
        }
    }
    max_viol
}

/// Test for zero divisors: exists x, y != 0 with x*y = 0.
/// Returns Some((i, j)) for the first ZD basis pair found.
///
/// Mirrors: DicksonCDProcess.v reference to C1538_MorZDSymmetry.v.
pub fn find_zero_divisor_pair(dim: usize) -> Option<(usize, usize)> {
    for i in 1..dim {
        for j in (i + 1)..dim {
            // Try x = e_i + e_j' for various j'
            for k in 1..dim {
                if k == i {
                    continue;
                }
                let mut x = vec![0.0; dim];
                x[i] = 1.0;
                x[k] = 1.0;
                for l in 1..dim {
                    if l == i || l == k {
                        continue;
                    }
                    for sign in [-1.0_f64, 1.0] {
                        let mut y = vec![0.0; dim];
                        y[j] = 1.0;
                        y[l] = sign;
                        let xy = cd_multiply(&x, &y);
                        let norm = cd_norm_sq(&xy);
                        if norm < 1e-20 {
                            return Some((i * dim + k, j * dim + l));
                        }
                    }
                }
            }
        }
    }
    None
}

/// Classify a dimension by the Dickson property hierarchy.
pub fn classify_properties(dim: usize) -> DicksonProperties {
    let commutative = max_commutativity_violation(dim) < 1e-10;
    let associative = max_associativity_violation(dim) < 1e-10;
    let norm_multiplicative = max_norm_violation(dim) < 1e-10;
    let has_zero_divisors = dim >= 16 && find_zero_divisor_pair(dim).is_some();

    DicksonProperties {
        dim,
        commutative,
        associative,
        norm_multiplicative,
        has_zero_divisors,
    }
}

/// The property record for a CD algebra dimension.
#[derive(Debug, Clone)]
pub struct DicksonProperties {
    pub dim: usize,
    pub commutative: bool,
    pub associative: bool,
    pub norm_multiplicative: bool,
    pub has_zero_divisors: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dim1_properties() {
        // R: commutative, associative, norm multiplicative
        assert!(max_commutativity_violation(1) < 1e-10);
        assert!(max_associativity_violation(1) < 1e-10);
        assert!(max_norm_violation(1) < 1e-10);
    }

    #[test]
    fn test_dim2_commutative() {
        // C: commutative
        assert!(max_commutativity_violation(2) < 1e-10);
    }

    #[test]
    fn test_dim4_not_commutative() {
        // H: NOT commutative (Dickson sect.3)
        assert!(max_commutativity_violation(4) > 0.5);
    }

    #[test]
    fn test_dim4_associative() {
        // H: associative
        assert!(max_associativity_violation(4) < 1e-10);
    }

    #[test]
    fn test_dim8_not_commutative() {
        // O: NOT commutative
        assert!(max_commutativity_violation(8) > 0.5);
    }

    #[test]
    fn test_dim8_not_associative() {
        // O: NOT associative (Dickson sect.4)
        assert!(max_associativity_violation(8) > 0.5);
    }

    #[test]
    fn test_dim8_norm_multiplicative() {
        // O: norm IS multiplicative (eight-square theorem)
        assert!(max_norm_violation(8) < 1e-10);
    }

    #[test]
    fn test_dim16_has_zero_divisors() {
        // S: HAS zero divisors
        assert!(find_zero_divisor_pair(16).is_some());
    }

    #[test]
    fn test_dim8_no_zero_divisors() {
        // O: no zero divisors (basis pairs only -- not exhaustive)
        assert!(find_zero_divisor_pair(8).is_none());
    }

    #[test]
    fn test_classify_hierarchy() {
        let c = classify_properties(2);
        assert!(c.commutative && c.associative && c.norm_multiplicative && !c.has_zero_divisors);

        let h = classify_properties(4);
        assert!(!h.commutative && h.associative && h.norm_multiplicative && !h.has_zero_divisors);

        let o = classify_properties(8);
        assert!(!o.commutative && !o.associative && o.norm_multiplicative && !o.has_zero_divisors);
    }
}

//! Moreno (1997) structural predicates for Cayley-Dickson elements.
//!
//! # Mathematical context
//!
//! Every predicate here corresponds to a definition or lemma in:
//! Moreno (1997) "The zero divisors of the Cayley-Dickson algebras
//! over the real numbers", arXiv:q-alg/9710013v1.
//!
//! Page references are to that paper.
//!
//! # Algebraic conventions (fixed throughout this module)
//!
//! | Symbol | Definition |
//! |--------|-----------|
//! | `t(x)` | `2 * x[0]` -- real-part doubling (Moreno's trace, p.5) |
//! | `<x,y>` | `sum_i x[i]*y[i]` -- Euclidean inner product |
//! | `x = (x_1, x_2)` | CD split: `x_1 = x[..half]`, `x_2 = x[half..]` |
//! | `tilde_e0` | `e_{dim/2}` -- second unit introduced at each CD doubling |
//! | `tilde(x)` | `x * tilde_e0 = (-x_2, x_1)` (Prop 1.11, p.13) |
//!
//! # Predicate hierarchy
//!
//! ```text
//! is_purely_imaginary    t(x) = 0
//! is_doubly_pure         t(x_1) = 0 AND t(x_2) = 0  (strictly stronger)
//! is_alternative_element (a,a,e_i) = 0 for all basis e_i
//! is_special_element     alternative AND purely imaginary AND unit norm
//! is_special_couple      both special AND orthogonal
//! is_special_triple      couple {a,b} AND (ay)b = -a(yb)
//! ```

use super::{cd_multiply, cd_norm_sq};

// ──────────────────────────────────────────────────────────────────────────────
// Trace, inner product, and purity tests (Chapter I, p.5)
// ──────────────────────────────────────────────────────────────────────────────

/// Moreno trace: `t(x) = 2 * x[0]`.
///
/// Moreno writes `t(x) = x + bar(x)`, the real-part doubled.
/// Zero iff `x` is purely imaginary.
#[inline]
pub fn cd_trace(x: &[f64]) -> f64 {
    2.0 * x[0]
}

/// Euclidean inner product: `<x, y> = sum_i x[i] * y[i]`.
///
/// Equals `(1/2) * t(x * bar(y))` in Moreno's formulation (Lemma 1.2, p.5).
#[inline]
pub fn cd_inner_product(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
}

/// Returns `true` if `x` is purely imaginary: `t(x) = 0`, i.e. `x[0] ~ 0`.
///
/// Moreno's `circ-A_n` is the unit sphere inside the purely imaginary elements (p.5).
#[inline]
pub fn is_purely_imaginary(x: &[f64]) -> bool {
    x[0].abs() < f64::EPSILON
}

/// Returns `true` if `x` is doubly pure (Def. p.12).
///
/// For `x = (x_1, x_2)` split at `half = x.len() / 2`:
/// doubly pure means `t(x_1) = 0` AND `t(x_2) = 0`,
/// equivalently `x[0] = 0` AND `x[half] = 0`.
///
/// This is strictly stronger than purely imaginary: `e_{dim/2}` is purely
/// imaginary but NOT doubly pure (its second-half trace is 1).
///
/// # Pre-condition
///
/// `x.len()` must be a power of two, >= 2.
#[inline]
pub fn is_doubly_pure(x: &[f64]) -> bool {
    let half = x.len() / 2;
    x[0].abs() < f64::EPSILON && x[half].abs() < f64::EPSILON
}

// ──────────────────────────────────────────────────────────────────────────────
// Tilde transform (Proposition 1.11, p.13)
// ──────────────────────────────────────────────────────────────────────────────

/// Tilde transform: right-multiplication by `tilde_e0 = e_{dim/2}`.
///
/// For `x = (x_1, x_2)` split at `half = dim / 2`:
///
/// `tilde(x) = x * e_{dim/2} = (-x_2, x_1)`
///
/// # Derivation via doubling formula `(a,b)(c,d) = (ac - d*b, da + bc*)`
///
/// With `c = 0_half`, `d = e_0_half` (unit of the lower sub-algebra):
/// - Left half: `x_1 * 0 - conj(e_0) * x_2 = -x_2`  (conj(e_0) = e_0, left-mult by unit)
/// - Right half: `e_0 * x_1 + x_2 * conj(0) = x_1`
///
/// # Properties (Prop 1.11)
///
/// - `tilde(tilde(x)) = -x`
/// - `||tilde(x)|| = ||x||`
/// - For doubly-pure `x`: `x _|_ tilde(x)` and `tilde(x)` is also doubly pure
pub fn cd_tilde(x: &[f64]) -> Vec<f64> {
    let dim = x.len();
    let half = dim / 2;
    let mut result = vec![0.0; dim];
    // Left half: -x_2
    for i in 0..half {
        result[i] = -x[half + i];
    }
    // Right half: x_1
    result[half..].copy_from_slice(&x[..half]);
    result
}

// ──────────────────────────────────────────────────────────────────────────────
// Alternative element predicate (Lemma 2.1, p.19; Chapter II)
// ──────────────────────────────────────────────────────────────────────────────

/// Returns `true` if `a` is an alternative element (Lemma 2.1, p.19).
///
/// An element `a` is alternative iff the associator `(a, a, y) = 0` for all `y`:
///
/// `(a, a, y) = a * (a * y) - (a * a) * y = 0`
///
/// Checking against the `dim` basis vectors `{e_0, ..., e_{dim-1}}` is sufficient:
/// if the associator vanishes on all basis vectors, it vanishes on all of `A_n`
/// by bilinearity of the associator in the first two slots.
///
/// Basis vectors are always alternative (Schafer `[8]`). In `A_3` (octonions)
/// ALL elements are alternative because `A_3` is an alternative algebra.
/// Starting from `A_4` (sedenions), general linear combinations may fail.
///
/// # Parameters
///
/// - `a`: element to test
/// - `atol`: absolute tolerance for zero comparisons
pub fn is_alternative_element(a: &[f64], atol: f64) -> bool {
    let dim = a.len();
    let a_sq = cd_multiply(a, a);
    let mut basis = vec![0.0; dim];
    for i in 0..dim {
        if i > 0 {
            basis[i - 1] = 0.0;
        }
        basis[i] = 1.0;
        let a_ei = cd_multiply(a, &basis);
        let lhs = cd_multiply(a, &a_ei); // a*(a*e_i)
        let rhs = cd_multiply(&a_sq, &basis); // (a*a)*e_i
        for k in 0..dim {
            if (lhs[k] - rhs[k]).abs() > atol {
                return false;
            }
        }
    }
    true
}

// ──────────────────────────────────────────────────────────────────────────────
// Special element and special couple (Chapter II, p.20)
// ──────────────────────────────────────────────────────────────────────────────

/// Returns `true` if `a` is a special element (Ch. II, p.20).
///
/// A special element satisfies all three conditions:
/// 1. Purely imaginary: `a[0] ~ 0`
/// 2. Unit norm: `||a|| = 1`
/// 3. Alternative: `(a, a, e_i) = 0` for all basis `e_i`
///
/// Special elements generate quaternion subalgebras `H_a` (Theorem 1.13).
///
/// # Parameters
///
/// - `a`: element to test
/// - `atol`: absolute tolerance
pub fn is_special_element(a: &[f64], atol: f64) -> bool {
    is_purely_imaginary(a) && (cd_norm_sq(a) - 1.0).abs() < atol && is_alternative_element(a, atol)
}

/// Returns `true` if `{a, b}` is a special couple (Ch. II, p.20).
///
/// A special couple requires:
/// 1. Both `a` and `b` are special elements
/// 2. Orthogonality: `<a, b> = 0`
///
/// For purely imaginary elements, `<a, b> = 0` is equivalent to anticommutativity
/// `a*b = -b*a` (Corollary 1.6, p.5).
///
/// When `{a, b}` is a special couple, `V(a;b) = span{e0, a, b, a*b}` is
/// a 4D copy of `H` (quaternions) inside `A_n` (Proposition 2.2).
///
/// # Parameters
///
/// - `a`, `b`: elements to test
/// - `atol`: absolute tolerance
pub fn is_special_couple(a: &[f64], b: &[f64], atol: f64) -> bool {
    debug_assert_eq!(a.len(), b.len());
    is_special_element(a, atol)
        && is_special_element(b, atol)
        && cd_inner_product(a, b).abs() < atol
}

// ──────────────────────────────────────────────────────────────────────────────
// Special triple (p.29)
// ──────────────────────────────────────────────────────────────────────────────

/// Returns `true` if `{a, y, b}` is a special triple (p.29).
///
/// A special triple requires:
/// 1. `{a, b}` is a special couple
/// 2. The non-associativity condition: `(a*y)*b = -a*(y*b)`
///
/// Condition (2) written in terms of associators:
/// `(a, y, b) = (ay)b - a(yb) = -2 * a*(yb)`,
/// which is equivalent to checking `(ay)b + a(yb) = 0` component-wise.
///
/// When `{a, y, b}` is a special triple, `V(a;y;b)` is a copy of `O`
/// (octonions) embedded in `A_n` (Theorem 2.13, p.30).
///
/// The standard example in `A_3`: `{e_1, e_7, e_2}` (p.30).
///
/// # Parameters
///
/// - `a`, `y`, `b`: elements to test (all must have the same length)
/// - `atol`: absolute tolerance
pub fn is_special_triple(a: &[f64], y: &[f64], b: &[f64], atol: f64) -> bool {
    let dim = a.len();
    debug_assert_eq!(y.len(), dim);
    debug_assert_eq!(b.len(), dim);
    if !is_special_couple(a, b, atol) {
        return false;
    }
    // Check (ay)b + a(yb) = 0 component-wise
    let ay = cd_multiply(a, y);
    let ayb = cd_multiply(&ay, b);
    let yb = cd_multiply(y, b);
    let ayb_neg = cd_multiply(a, &yb);
    for i in 0..dim {
        if (ayb[i] + ayb_neg[i]).abs() > atol {
            return false;
        }
    }
    true
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn basis(dim: usize, i: usize) -> Vec<f64> {
        let mut v = vec![0.0; dim];
        v[i] = 1.0;
        v
    }

    // A_3 = octonions, dim = 8.
    const DIM8: usize = 8;
    const ATOL: f64 = 1e-12;

    #[test]
    fn test_cd_trace_and_inner_product() {
        let e0 = basis(DIM8, 0);
        let e1 = basis(DIM8, 1);
        assert!((cd_trace(&e0) - 2.0).abs() < ATOL);
        assert!(cd_trace(&e1).abs() < ATOL);
        assert!((cd_inner_product(&e0, &e0) - 1.0).abs() < ATOL);
        assert!(cd_inner_product(&e0, &e1).abs() < ATOL);
    }

    #[test]
    fn test_is_purely_imaginary() {
        assert!(!is_purely_imaginary(&basis(DIM8, 0))); // e0 has real part 1
        assert!(is_purely_imaginary(&basis(DIM8, 1))); // e1 is imaginary
        assert!(is_purely_imaginary(&basis(DIM8, 7))); // e7 is imaginary
    }

    #[test]
    fn test_is_doubly_pure() {
        // e1: x[0]=0, x[4]=0 -> doubly pure
        assert!(is_doubly_pure(&basis(DIM8, 1)));
        // e0: x[0]=1 -> NOT doubly pure
        assert!(!is_doubly_pure(&basis(DIM8, 0)));
        // e4 = tilde_e0: x[0]=0, x[4]=1 -> NOT doubly pure (second-half trace nonzero)
        assert!(!is_doubly_pure(&basis(DIM8, 4)));
    }

    #[test]
    fn test_cd_tilde_e1() {
        // e1 = [0,1,0,0,0,0,0,0], half=4
        // x_1 = [0,1,0,0], x_2 = [0,0,0,0]
        // tilde(e1) = (-x_2, x_1) = (0, e1_in_4D) = [0,0,0,0,0,1,0,0] = e5
        let e1 = basis(DIM8, 1);
        let t = cd_tilde(&e1);
        let e5 = basis(DIM8, 5);
        for i in 0..DIM8 {
            assert!(
                (t[i] - e5[i]).abs() < ATOL,
                "tilde(e1)[{i}] = {} (expected {})",
                t[i],
                e5[i]
            );
        }
    }

    #[test]
    fn test_cd_tilde_double_negation() {
        // tilde(tilde(x)) = -x  (Proposition 1.11)
        for i in 1..DIM8 {
            let e = basis(DIM8, i);
            let tt = cd_tilde(&cd_tilde(&e));
            for k in 0..DIM8 {
                assert!(
                    (tt[k] + e[k]).abs() < ATOL,
                    "tilde^2(e_{i})[{k}] = {} (expected {})",
                    tt[k],
                    -e[k]
                );
            }
        }
    }

    #[test]
    fn test_cd_tilde_norm_preserved() {
        // ||tilde(x)|| = ||x||  (Proposition 1.11)
        for i in 0..DIM8 {
            let e = basis(DIM8, i);
            let t = cd_tilde(&e);
            assert!(
                (cd_norm_sq(&t) - cd_norm_sq(&e)).abs() < ATOL,
                "tilde norm not preserved for e_{i}"
            );
        }
    }

    #[test]
    fn test_basis_vectors_alternative_in_a3() {
        // In A_3 (octonions, an alternative algebra), all basis vectors are alternative
        for i in 0..DIM8 {
            assert!(
                is_alternative_element(&basis(DIM8, i), ATOL),
                "e_{i} should be alternative in A_3"
            );
        }
    }

    #[test]
    fn test_e0_not_special() {
        // e0 is NOT special: it is not purely imaginary (x[0] = 1)
        assert!(!is_special_element(&basis(DIM8, 0), ATOL));
    }

    #[test]
    fn test_e1_e2_e7_are_special_in_a3() {
        // e_1, e_2, e_7 are special elements in A_3
        let e1 = basis(DIM8, 1);
        let e2 = basis(DIM8, 2);
        let e7 = basis(DIM8, 7);
        assert!(is_special_element(&e1, ATOL), "e1 should be special in A_3");
        assert!(is_special_element(&e2, ATOL), "e2 should be special in A_3");
        assert!(is_special_element(&e7, ATOL), "e7 should be special in A_3");
    }

    #[test]
    fn test_e1_e2_special_couple_in_a3() {
        // {e1, e2} is a special couple: both special AND orthogonal (e1*e2 = e3, e2*e1 = -e3)
        let e1 = basis(DIM8, 1);
        let e2 = basis(DIM8, 2);
        assert!(
            is_special_couple(&e1, &e2, ATOL),
            "{{e1, e2}} should be a special couple in A_3"
        );
    }

    #[test]
    fn test_e1_e7_e2_special_triple_in_a3() {
        // {e1, e7, e2} is a special triple in A_3 (standard octonion example, p.30)
        // a=e1, y=e7, b=e2: requires {e1, e2} special couple AND (e1*e7)*e2 = -e1*(e7*e2).
        // Verification by hand:
        //   e1*e7 = e6,  e6*e2 = e4
        //   e7*e2 = e5,  e1*e5 = -e4   => -e1*(e7*e2) = e4   MATCH
        let e1 = basis(DIM8, 1);
        let e7 = basis(DIM8, 7);
        let e2 = basis(DIM8, 2);
        assert!(
            is_special_triple(&e1, &e7, &e2, ATOL),
            "{{e1, e7, e2}} should be a special triple in A_3"
        );
    }

    #[test]
    fn test_find_non_alternative_in_a4() {
        // In A_4 (sedenions, dim=16) the algebra is NOT alternative: there exists
        // some a with (a,a,b) != 0 for some b. This test finds such an element
        // by scanning all unit basis pairs.
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let mut found = false;
        'outer: for i in 1..16usize {
            for j in (i + 1)..16usize {
                let mut x = vec![0.0_f64; 16];
                x[i] = s;
                x[j] = s;
                if !is_alternative_element(&x, 1e-10) {
                    found = true;
                    break 'outer;
                }
            }
        }
        assert!(
            found,
            "A_4 must contain a non-alternative element (algebra is not alternative)"
        );
    }
}

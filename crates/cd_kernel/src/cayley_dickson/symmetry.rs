//! Symmetry and observable helpers layered on top of the binary CD kernel.
//!
//! These routines are value-level observables or order-3 / order-2 actions
//! built from the underlying multiplication, conjugation, and associator
//! machinery. They are intentionally kept separate from zero-divisor search so
//! signature-sensitive or phenomenology-facing logic has a distinct surface.

use super::arith::cd_multiply;

/// Gourlay/Gresnigt epsilon automorphism (order 2).
///
///   epsilon(A + B*e_8) = A - B*e_8
///
/// Flips the sign of the upper octonion half.
/// Reference: Gourlay & Gresnigt (arXiv:2407.01580), Eq 4.
#[inline]
pub fn gourlay_epsilon(v: &[f64; 16]) -> [f64; 16] {
    let mut out = *v;
    for x in out[8..].iter_mut() {
        *x = -*x;
    }
    out
}

/// Gourlay/Gresnigt psi automorphism (order 3).
///
///   psi(A + B*e_8) = (1/4)`[A + 3A* + sqrt(3)(B - B*)]`
///                   + (1/4)`[B + 3B* - sqrt(3)(A - A*)]` * e_8
///
/// where A* is the octonion conjugate of A (negate imaginary, keep real).
///
/// This cycles the three generations: psi^3 = Id.
/// Reference: Gourlay & Gresnigt (arXiv:2407.01580), Eq 5.
pub fn gourlay_psi(v: &[f64; 16]) -> [f64; 16] {
    let sqrt3 = 3.0_f64.sqrt();

    let a = &v[..8];
    let b = &v[8..];

    let mut a_conj = [0.0_f64; 8];
    a_conj[0] = a[0];
    for i in 1..8 {
        a_conj[i] = -a[i];
    }

    let mut b_conj = [0.0_f64; 8];
    b_conj[0] = b[0];
    for i in 1..8 {
        b_conj[i] = -b[i];
    }

    let mut out = [0.0_f64; 16];

    for i in 0..8 {
        out[i] = 0.25 * (a[i] + 3.0 * a_conj[i] + sqrt3 * (b[i] - b_conj[i]));
    }

    for i in 0..8 {
        out[8 + i] = 0.25 * (b[i] + 3.0 * b_conj[i] - sqrt3 * (a[i] - a_conj[i]));
    }

    out
}

/// Apply psi n times (for computing psi^2, psi^3, etc.).
pub fn gourlay_psi_n(v: &[f64; 16], n: usize) -> [f64; 16] {
    let mut result = *v;
    for _ in 0..n {
        result = gourlay_psi(&result);
    }
    result
}

/// Cross-generational signed friction between subalgebra O_i and O_j.
///
/// Measures how much topological friction exists between generations by
/// computing the associator `[A_rot, X, Y]` where X is in O_i and Y is in O_j.
///
/// This observable is Euclidean/compact in its current interpretation and
/// should not be reused silently as a split-signature diagnostic.
pub fn cross_generational_friction(
    mode_a: usize,
    mode_b: usize,
    subalgebra_i: &[usize],
    subalgebra_j: &[usize],
) -> f64 {
    let dim = 16_usize;
    let theta = std::f64::consts::FRAC_PI_4;

    let mut a_rot = vec![0.0; dim];
    a_rot[mode_a] = theta.cos();
    a_rot[mode_b] = theta.sin();

    let mut total = 0.0_f64;

    for &x_idx in subalgebra_i {
        if x_idx == 0 || x_idx == mode_a || x_idx == mode_b {
            continue;
        }
        for &y_idx in subalgebra_j {
            if y_idx == 0 || y_idx == mode_a || y_idx == mode_b {
                continue;
            }
            if x_idx == y_idx {
                continue;
            }

            let mut ex = vec![0.0; dim];
            ex[x_idx] = 1.0;
            let mut ey = vec![0.0; dim];
            ey[y_idx] = 1.0;

            let ax = cd_multiply(&a_rot, &ex);
            let axy = cd_multiply(&ax, &ey);
            let xy = cd_multiply(&ex, &ey);
            let a_xy = cd_multiply(&a_rot, &xy);

            for k in 0..dim {
                total += axy[k] - a_xy[k];
            }
        }
    }

    total
}

//! `E_6` Casimir invariants.
//!
//! Computes the quadratic Casimir `C_2(lambda)` of representations of `E_6`
//! using the standard formula
//!
//! ```text
//!   C_2(lambda) = <lambda + 2 rho, lambda>
//! ```
//!
//! where `rho` is the Weyl vector (half-sum of positive roots, equivalently
//! sum of fundamental weights), and the inner product is the standard
//! Euclidean one with long roots normalized to squared length 2.
//!
//! # Computing rho without positivity ordering
//!
//! Identifying positive roots in our `E_8`-ambient embedding requires solving
//! a 6x6 linear system per root (sign of simple-root expansion coefficients).
//! A cleaner approach exploits the identity `rho = sum_i omega_i` where
//! `omega_i` is the i-th fundamental weight, dual to the simple coroots:
//!
//! ```text
//!   <omega_i, beta_j^v> = delta_ij  =>  omega = A^{-1} . simple_roots
//! ```
//!
//! For simply-laced `E_6` the coroots equal the roots and `A` is symmetric.
//! We solve `A x = (1, 1, ..., 1)` once via Gaussian elimination, giving the
//! coefficients of `rho` in the simple-root basis. The rest is one matrix-
//! vector product against `simple_roots` to lift `rho` into the 8D ambient.

use super::root_system::{e6_cartan_matrix, e6_simple_roots, generate_e6_roots};

/// Solve a 6x6 integer linear system `A x = b` over `f64` via plain Gaussian
/// elimination with partial pivoting. Sufficient for the well-conditioned
/// E6 Cartan inverse.
fn solve_6x6(a: &[[i32; 6]; 6], b: &[f64; 6]) -> [f64; 6] {
    let mut m = [[0.0f64; 7]; 6];
    for i in 0..6 {
        for j in 0..6 {
            m[i][j] = a[i][j] as f64;
        }
        m[i][6] = b[i];
    }
    for k in 0..6 {
        let mut piv = k;
        for i in (k + 1)..6 {
            if m[i][k].abs() > m[piv][k].abs() {
                piv = i;
            }
        }
        m.swap(k, piv);
        let inv = 1.0 / m[k][k];
        for j in k..7 {
            m[k][j] *= inv;
        }
        for i in 0..6 {
            if i != k && m[i][k] != 0.0 {
                let f = m[i][k];
                for j in k..7 {
                    m[i][j] -= f * m[k][j];
                }
            }
        }
    }
    let mut x = [0.0f64; 6];
    for i in 0..6 {
        x[i] = m[i][6];
    }
    x
}

/// Weyl vector `rho` of `E_6` in the 8D ambient basis.
///
/// Computed as `rho = sum_i omega_i` where `omega_i = (A^{-1})_ji beta_j` are
/// the fundamental weights in the simple-root basis. Concretely,
/// `rho = sum_j coeff_j * beta_j` where `coeff = A^{-1} * (1, 1, ..., 1)`.
pub fn weyl_vector() -> [f64; 8] {
    let cartan = e6_cartan_matrix();
    let coeffs = solve_6x6(&cartan, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let simple = e6_simple_roots();
    let mut rho = [0.0f64; 8];
    for (j, c) in coeffs.iter().enumerate() {
        for k in 0..8 {
            rho[k] += c * simple[j].coords[k];
        }
    }
    rho
}

/// Compute `C_2(lambda) = <lambda + 2 rho, lambda>` for an 8D-ambient weight `lambda`.
///
/// Returned in the **math normalization** (long roots squared length 2). For
/// the physics normalization (squared length 1) used in the project monograph
/// and in [`crate::lie::f4::casimir`], divide the result by 2 -- or call
/// [`casimir_physics`].
pub fn casimir(lambda: &[f64; 8]) -> f64 {
    let rho = weyl_vector();
    let mut total = 0.0f64;
    for i in 0..8 {
        total += (lambda[i] + 2.0 * rho[i]) * lambda[i];
    }
    total
}

/// Same as [`casimir`] but in the physics normalization (long roots squared
/// length 1). Matches `f4_casimir.rs`'s convention so the two eigenvalues
/// can be compared directly across the magic-square exceptional family.
pub fn casimir_physics(lambda: &[f64; 8]) -> f64 {
    casimir(lambda) * 0.5
}

/// Highest weight of the 27-dim fundamental representation in the 8D ambient
/// basis. This is `omega_1`, the fundamental weight dual to `beta_1`.
pub fn highest_weight_27() -> [f64; 8] {
    let cartan = e6_cartan_matrix();
    // omega_1 has expansion (A^{-1})_{j,1} in the simple-root basis.
    let coeffs = solve_6x6(&cartan, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let simple = e6_simple_roots();
    let mut omega = [0.0f64; 8];
    for (j, c) in coeffs.iter().enumerate() {
        for k in 0..8 {
            omega[k] += c * simple[j].coords[k];
        }
    }
    omega
}

/// Compute `(C_2(27), |Delta+(E_6)|, ratio)` for the 27-dim fundamental rep,
/// in the **physics normalization** (matching `f4_casimir.rs`). Textbook
/// value: `C_2(27) = 26/3`.
pub fn compute_e6_casimir_ratio() -> (f64, usize, f64) {
    let lambda = highest_weight_27();
    let c2 = casimir_physics(&lambda);
    let n_pos = generate_e6_roots().len() / 2;
    let ratio = c2 / n_pos as f64;
    (c2, n_pos, ratio)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `<rho, beta_i> = 1` for each simple root, by definition of the dual basis.
    #[test]
    fn rho_pairs_to_one_with_each_simple_root() {
        let rho = weyl_vector();
        let simple = e6_simple_roots();
        for (i, alpha) in simple.iter().enumerate() {
            let dot: f64 = alpha
                .coords
                .iter()
                .zip(rho.iter())
                .map(|(a, r)| a * r)
                .sum();
            assert!(
                (dot - 1.0).abs() < 1e-9,
                "<rho, beta_{}> = {}, expected 1.0",
                i + 1,
                dot,
            );
        }
    }

    /// `<omega_1, beta_1> = 1` and `<omega_1, beta_j> = 0` for `j > 1`.
    #[test]
    fn fundamental_weight_omega_1_dual_to_beta_1() {
        let omega = highest_weight_27();
        let simple = e6_simple_roots();
        for (i, alpha) in simple.iter().enumerate() {
            let dot: f64 = alpha
                .coords
                .iter()
                .zip(omega.iter())
                .map(|(a, w)| a * w)
                .sum();
            let expected = if i == 0 { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() < 1e-9,
                "<omega_1, beta_{}> = {}, expected {}",
                i + 1,
                dot,
                expected,
            );
        }
    }

    /// Textbook: `C_2(27) = 26/3` in the *physics* normalization (long roots
    /// squared length 1). The math-normalization value (squared length 2) is
    /// `52/3`; the two differ by exactly the factor of 2 in the inner product.
    /// See Slansky 1981, Table 7.
    #[test]
    fn casimir_27_equals_26_thirds() {
        let (c2, n_pos, _ratio) = compute_e6_casimir_ratio();
        assert_eq!(n_pos, 36);
        assert!(
            (c2 - 26.0 / 3.0).abs() < 1e-9,
            "C_2(27) = {}, expected 26/3 = {}",
            c2,
            26.0 / 3.0,
        );
    }

    #[test]
    fn solve_6x6_recovers_identity() {
        let identity = [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = solve_6x6(&identity, &b);
        for i in 0..6 {
            assert!((x[i] - b[i]).abs() < 1e-12);
        }
    }
}

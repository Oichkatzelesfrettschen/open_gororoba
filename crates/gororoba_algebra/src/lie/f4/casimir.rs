//! `F_4` Casimir invariant ratio (Theorem 12.6, project monograph; Rocq claim C-035).
//!
//! Computes `epsilon = C_2(26) / |Delta+(F_4)|` for the 26-dim fundamental
//! representation. Under the **physics normalization** (long roots squared
//! length 1) the ratio is exactly `1/4`.
//!
//! Pure root-system combinatorics: enumerate positives via [`super::root_system`],
//! sum to the Weyl vector `rho`, and evaluate `C_2(lambda) = <lambda + 2 rho, lambda>`
//! at the highest weight `lambda = (1, 0, 0, 0)` of the 26.

use super::root_system::{f4_positive_root_count, weyl_vector};

/// Compute `(C_2(26), |Delta+(F_4)|, epsilon)` for the `F_4` fundamental rep,
/// in physics normalization (long roots squared length 1).
pub fn compute_f4_casimir_ratio() -> (f64, usize, f64) {
    let rho = weyl_vector();
    // The 26 has highest weight lambda = (1, 0, 0, 0) -- omega_4 in our
    // simple-root labeling, which matches Slansky 1981 Table 7.
    let lambda = [1.0, 0.0, 0.0, 0.0];

    // C_2(lambda) = <lambda + 2 rho, lambda> in math normalization.
    let mut c2_math = 0.0;
    for i in 0..4 {
        c2_math += (lambda[i] + 2.0 * rho[i]) * lambda[i];
    }

    // Convert to physics normalization (halve, since long roots scale 2 -> 1).
    let c2 = c2_math * 0.5;

    let n_pos = f4_positive_root_count();
    let epsilon = c2 / n_pos as f64;
    (c2, n_pos, epsilon)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f4_casimir_ratio() {
        let (c2, num_pos_roots, epsilon) = compute_f4_casimir_ratio();

        assert_eq!(num_pos_roots, 24, "F4 must have exactly 24 positive roots");
        assert!((c2 - 6.0).abs() < 1e-9, "C2(26) must be exactly 6");
        assert!((epsilon - 0.25).abs() < 1e-9, "Epsilon must be exactly 1/4");
    }
}

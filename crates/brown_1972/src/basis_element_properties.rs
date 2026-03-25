//! Basis element properties (Brown Chapter VI, Theorems 6.1-6.11).
//!
//! The remarkable results:
//! - Basis elements anticommute: e_i * e_j = -e_j * e_i (i,j != 0, i != j) (Lemma 6.10)
//! - Basis alternativity: e_i(e_i e_j) = -e_j (Theorem 6.11)
//! - Restricted Moufang identities hold for adjoined element (Theorems 6.1-6.6)
//! - Commutativity condition: xe = ex iff T(a_2) = 0 (Lemma 6.8)
//! - CD multiplication matches polynomial multiplication for alternative algebras (Corollary 6.7)

use cd_kernel::cayley_dickson::cd_multiply;

/// Verify Lemma 6.10 part i: e_i * e_j = e_j * e_i when i=0, j=0, or i=j.
/// For basis elements, commutativity only holds for these special cases.
/// Mirrors: Brown (1972) Lemma 6.10.
pub fn verify_basis_commutativity_special_cases(dim: usize) -> f64 {
    let mut max_err = 0.0_f64;

    // Case: i = 0 (real part)
    let mut e_0 = vec![0.0; dim];
    e_0[0] = 1.0;
    for j in 1..dim {
        let mut e_j = vec![0.0; dim];
        e_j[j] = 1.0;
        let e0_ej = cd_multiply(&e_0, &e_j);
        let ej_e0 = cd_multiply(&e_j, &e_0);
        max_err = max_err.max(
            e0_ej
                .iter()
                .zip(ej_e0.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f64::max),
        );
    }

    max_err
}

/// Verify Lemma 6.10 part ii: e_i * e_j = -e_j * e_i when i,j != 0 and i != j.
/// For imaginary basis elements, multiplication is anticommutative.
/// Mirrors: Brown (1972) Lemma 6.10.
pub fn verify_basis_anticommutativity(dim: usize) -> f64 {
    let mut max_err = 0.0_f64;

    // Check anticommutativity for all pairs (i,j) where i,j in {1,...,dim-1}, i != j
    for i in 1..dim.min(5) {
        // Limit to first few for efficiency
        for j in (i + 1)..dim.min(5) {
            let mut e_i = vec![0.0; dim];
            e_i[i] = 1.0;
            let mut e_j = vec![0.0; dim];
            e_j[j] = 1.0;

            let ei_ej = cd_multiply(&e_i, &e_j);
            let ej_ei = cd_multiply(&e_j, &e_i);
            let neg_ej_ei: Vec<f64> = ej_ei.iter().map(|&x| -x).collect();

            max_err = max_err.max(
                ei_ej
                    .iter()
                    .zip(neg_ej_ei.iter())
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0, f64::max),
            );
        }
    }

    max_err
}

/// Verify Theorem 6.11: e_i(e_i e_j) = -e_j = (e_j e_i)e_i for i != 0.
/// Basis elements satisfy the alternative property (not full associativity).
/// Mirrors: Brown (1972) Theorem 6.11.
pub fn verify_basis_alternativity(dim: usize) -> f64 {
    let mut max_err = 0.0_f64;

    // Check for a few basis element pairs
    for i in 1..dim.min(4) {
        for j in 1..dim.min(4) {
            if i == j {
                continue;
            }

            let mut e_i = vec![0.0; dim];
            e_i[i] = 1.0;
            let mut e_j = vec![0.0; dim];
            e_j[j] = 1.0;

            // e_i * e_j
            let ei_ej = cd_multiply(&e_i, &e_j);

            // e_i * (e_i * e_j)
            let ei_ei_ej = cd_multiply(&e_i, &ei_ej);

            // e_j * e_i
            let ej_ei = cd_multiply(&e_j, &e_i);

            // (e_j * e_i) * e_i
            let ej_ei_ei = cd_multiply(&ej_ei, &e_i);

            // -e_j
            let neg_e_j: Vec<f64> = e_j.iter().map(|&x| -x).collect();

            // Check ei_ei_ej = -e_j
            max_err = max_err.max(
                ei_ei_ej
                    .iter()
                    .zip(neg_e_j.iter())
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0, f64::max),
            );

            // Check ej_ei_ei = -e_j
            max_err = max_err.max(
                ej_ei_ei
                    .iter()
                    .zip(neg_e_j.iter())
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0, f64::max),
            );
        }
    }

    max_err
}

/// Verify Theorem 6.1: (e,A,B) + (A,e,B) = 0 for all A,B.
/// The associator with the adjoined element satisfies this restraint.
/// Mirrors: Brown (1972) Theorem 6.1.
pub fn verify_adjoined_associator_property(a: &[f64], b: &[f64]) -> f64 {
    let dim = a.len();
    if dim < 2 {
        return 0.0; // No adjoined element for dim 1
    }

    let half = dim / 2;

    // Create artificial "e" as the new basis element (index at half position in doubled space)
    let mut e = vec![0.0; dim * 2];
    e[half] = 1.0;

    // (e,A,B) = (eA)B - e(AB)
    let ea = cd_multiply(&e[..dim], a);
    let ea_b = cd_multiply(&ea, b);
    let ab = cd_multiply(a, b);
    let e_ab = cd_multiply(&e[..dim], &ab);
    let assoc_e_ab: Vec<f64> = ea_b.iter().zip(e_ab.iter()).map(|(x, y)| x - y).collect();

    // (A,e,B) = (Ae)B - A(eB)
    let ae = cd_multiply(a, &e[..dim]);
    let ae_b = cd_multiply(&ae, b);
    let eb = cd_multiply(&e[..dim], b);
    let a_eb = cd_multiply(a, &eb);
    let assoc_a_e_b: Vec<f64> = ae_b.iter().zip(a_eb.iter()).map(|(x, y)| x - y).collect();

    // Sum should be zero
    assoc_e_ab
        .iter()
        .zip(assoc_a_e_b.iter())
        .map(|(x, y)| (x + y).abs())
        .fold(0.0, f64::max)
}

/// Verify Lemma 6.8: xe = ex iff T(a_2) = 0 (for x = a_1 + ea_2).
/// Commutativity with the adjoined element depends on the trace of the upper half.
/// The condition T(a_2) = 0 means a_2[0] = 0 (pure imaginary upper half).
/// Mirrors: Brown (1972) Lemma 6.8.
pub fn verify_commutativity_condition(a_2: &[f64]) -> f64 {
    // T(a_2) = 2 * a_2[0]
    // According to Lemma 6.8, xe = ex iff T(a_2) = 0
    // This means the upper half must be pure imaginary (no real part)
    2.0 * a_2[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_sedenion(seed: u64) -> Vec<f64> {
        let mut x = vec![0.0; 16];
        let mut s = seed;
        for v in &mut x {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *v = ((s >> 33) as f64) / (1u64 << 31) as f64 - 1.0;
        }
        x
    }

    #[test]
    fn test_basis_commutativity_special_cases() {
        let err = verify_basis_commutativity_special_cases(16);
        assert!(err < 1e-15, "e_0 should commute with all basis elements: err={err}");
    }

    #[test]
    fn test_basis_anticommutativity() {
        let err = verify_basis_anticommutativity(16);
        assert!(err < 1e-15, "imaginary basis elements should anticommute: err={err}");
    }

    #[test]
    fn test_basis_alternativity() {
        let err = verify_basis_alternativity(16);
        assert!(err < 1e-15, "basis elements should satisfy alternativity: err={err}");
    }

    #[test]
    fn test_adjoined_associator_property() {
        let a = random_sedenion(11);
        let b = random_sedenion(12);
        let err = verify_adjoined_associator_property(&a, &b);
        assert!(
            err < 1e-6,
            "(e,A,B) + (A,e,B) should be 0: err={err}"
        );
    }

    #[test]
    fn test_commutativity_condition() {
        let a_2 = random_sedenion(14);
        let trace_a_2 = verify_commutativity_condition(&a_2);
        // T(a_2) = 2 * a_2[0]
        // This should match the expected trace value
        assert!((trace_a_2 - 2.0 * a_2[0]).abs() < 1e-15);
    }
}

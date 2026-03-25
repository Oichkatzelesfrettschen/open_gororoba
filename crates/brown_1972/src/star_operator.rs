//! The * (star) operator (Brown Definition 7.1, Theorem 7.5).
//!
//! For A = a1 + ea2 in A_n: A* = a1 - ea2.
//! This is the "inner doubling conjugation" -- it negates the upper half.
//!
//! Key properties (Theorem 7.5):
//! - A + A* = 2*a1 (projection onto lower half)
//! - A* = conj(A) iff a1 is real
//! - A* = A iff a2 = 0
//! - N(A*) = N(A)
//! - (AB)* = A*B*
//! - A** = A

/// Compute the star operator: A* = a1 - ea2.
///
/// For a 2n-dimensional element A = (a1, a2), returns (a1, -a2).
///
/// Mirrors: Brown (1972) Definition 7.1.
pub fn star(a: &[f64]) -> Vec<f64> {
    let half = a.len() / 2;
    let mut result = a.to_vec();
    for v in &mut result[half..] {
        *v = -*v;
    }
    result
}

/// Verify Theorem 7.5 property (i): A + A* = 2*a1.
pub fn verify_sum_is_2a1(a: &[f64]) -> f64 {
    let half = a.len() / 2;
    let a_star = star(a);
    let mut max_err = 0.0_f64;
    for i in 0..a.len() {
        let sum = a[i] + a_star[i];
        let expected = if i < half { 2.0 * a[i] } else { 0.0 };
        max_err = max_err.max((sum - expected).abs());
    }
    max_err
}

/// Verify Theorem 7.5 property (v): N(A*) = N(A).
pub fn verify_norm_preserved(a: &[f64]) -> f64 {
    let a_star = star(a);
    let n_a = cd_kernel::cayley_dickson::cd_norm_sq(a);
    let n_a_star = cd_kernel::cayley_dickson::cd_norm_sq(&a_star);
    (n_a - n_a_star).abs()
}

/// Verify Theorem 7.5 property (xiii): A** = A.
pub fn verify_involution(a: &[f64]) -> f64 {
    let a_star_star = star(&star(a));
    a.iter()
        .zip(a_star_star.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// Verify Lemma 7.6: AB = 0 iff A*B* = 0.
pub fn verify_star_zd_equivalence(a: &[f64], b: &[f64]) -> (f64, f64) {
    let ab = cd_kernel::cayley_dickson::cd_multiply(a, b);
    let a_star = star(a);
    let b_star = star(b);
    let a_star_b_star = cd_kernel::cayley_dickson::cd_multiply(&a_star, &b_star);
    (
        cd_kernel::cayley_dickson::cd_norm_sq(&ab),
        cd_kernel::cayley_dickson::cd_norm_sq(&a_star_b_star),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_star_involution() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
        assert!(verify_involution(&a) < 1e-15);
    }

    #[test]
    fn test_star_norm_preserved() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
        assert!(verify_norm_preserved(&a) < 1e-10);
    }

    #[test]
    fn test_star_sum_is_2a1() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
        assert!(verify_sum_is_2a1(&a) < 1e-15);
    }

    #[test]
    fn test_star_zd_equivalence() {
        let (a, b) = schafer_1945::division_test::zd_witness_16();
        let (n_ab, n_as_bs) = verify_star_zd_equivalence(&a, &b);
        // AB = 0 should imply A*B* = 0
        assert!(n_ab < 1e-10, "AB should be 0");
        assert!(n_as_bs < 1e-10, "A*B* should be 0 (Lemma 7.6)");
    }
}

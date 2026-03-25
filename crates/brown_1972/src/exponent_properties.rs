//! Exponent properties (Brown Chapter V, Theorems 5.6-5.17).
//!
//! The remarkable results:
//! - a^m * a^n = a^{m+n} for ALL integers m, n (Thm 5.11)
//! - (a^m)^n = a^{mn} for ALL integers m, n (Thm 5.12)
//! - (a^m, b, a^n) = 0 for ALL integers m, n (Thm 5.17)
//! - No nilpotent elements: a^n = 0 implies a = 0 (Cor 5.5)
//! - Only idempotents are 0 and 1 (Cor 5.3)
//! - (a,b,c)^2 = -N((a,b,c)) -- associator squares to neg norm (Lemma 5.14)

use cd_kernel::cayley_dickson::{cd_conjugate, cd_multiply, cd_norm_sq};

/// Compute a^n for non-negative integer n.
///
/// a^0 = e_0 (identity), a^1 = a, a^{n+1} = a * a^n.
///
/// Mirrors: Brown (1972) Definition 5.1.
pub fn cd_power(a: &[f64], n: i32) -> Vec<f64> {
    let dim = a.len();
    if n == 0 {
        let mut result = vec![0.0; dim];
        result[0] = 1.0;
        return result;
    }
    if n > 0 {
        let mut result = a.to_vec();
        for _ in 1..n {
            result = cd_multiply(&result, a);
        }
        result
    } else {
        // Negative: a^{-n} = (a^{-1})^n
        let norm = cd_norm_sq(a);
        if norm.abs() < 1e-15 {
            return vec![f64::NAN; dim];
        }
        let conj = cd_conjugate(a);
        let inv: Vec<f64> = conj.iter().map(|x| x / norm).collect();
        cd_power(&inv, -n)
    }
}

/// Verify Theorem 5.11: a^m * a^n = a^{m+n} for integers m, n.
pub fn verify_exponent_addition(a: &[f64], m: i32, n: i32) -> f64 {
    let am = cd_power(a, m);
    let an = cd_power(a, n);
    let am_an = cd_multiply(&am, &an);
    let amn = cd_power(a, m + n);
    am_an.iter()
        .zip(amn.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// Verify Theorem 5.12: (a^m)^n = a^{mn} for integers m, n.
pub fn verify_power_of_power(a: &[f64], m: i32, n: i32) -> f64 {
    let am = cd_power(a, m);
    let am_n = cd_power(&am, n);
    let amn = cd_power(a, m * n);
    am_n.iter()
        .zip(amn.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// Verify Theorem 5.17: (a^m, b, a^n) = 0 for all integers m, n.
/// The generalized flexibility: powers of a single element always
/// associate with any other element.
pub fn verify_generalized_flexibility(a: &[f64], b: &[f64], m: i32, n: i32) -> f64 {
    let am = cd_power(a, m);
    let an = cd_power(a, n);
    let am_b = cd_multiply(&am, b);
    let lhs = cd_multiply(&am_b, &an);
    let b_an = cd_multiply(b, &an);
    let rhs = cd_multiply(&am, &b_an);
    // Associator (a^m, b, a^n) = (a^m * b) * a^n - a^m * (b * a^n)
    lhs.iter()
        .zip(rhs.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// Verify Corollary 5.5: no nilpotent elements.
/// a^n should be nonzero for all n and all nonzero a.
pub fn verify_no_nilpotent(a: &[f64], max_n: i32) -> bool {
    let norm_a = cd_norm_sq(a);
    if norm_a < 1e-15 {
        return true; // a = 0 is trivially "nilpotent" but not a counterexample
    }
    for n in 1..=max_n {
        let an = cd_power(a, n);
        if cd_norm_sq(&an) < 1e-10 {
            return false; // nilpotent found!
        }
    }
    true
}

/// Verify Lemma 5.14: (a,b,c)^2 = -N((a,b,c)).
/// The associator of any triple squares to negative its norm.
/// This is because T((a,b,c)) = 0 (Thm 3.1), so by the quadratic
/// identity x^2 = T(x)*x - N(x)*1 with T = 0: x^2 = -N(x).
pub fn verify_associator_square(a: &[f64], b: &[f64], c: &[f64]) -> f64 {
    let ab = cd_multiply(a, b);
    let ab_c = cd_multiply(&ab, c);
    let bc = cd_multiply(b, c);
    let a_bc = cd_multiply(a, &bc);
    // associator = (ab)c - a(bc)
    let assoc: Vec<f64> = ab_c.iter().zip(a_bc.iter()).map(|(x, y)| x - y).collect();
    let assoc_sq = cd_multiply(&assoc, &assoc);
    let neg_norm = -cd_norm_sq(&assoc);
    let mut max_err = 0.0_f64;
    for (i, &val) in assoc_sq.iter().enumerate() {
        let expected = if i == 0 { neg_norm } else { 0.0 };
        max_err = max_err.max((val - expected).abs());
    }
    max_err
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_sedenion(seed: u64) -> Vec<f64> {
        // Deterministic pseudo-random sedenion
        let mut x = vec![0.0; 16];
        let mut s = seed;
        for v in &mut x {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *v = ((s >> 33) as f64) / (1u64 << 31) as f64 - 1.0;
        }
        x
    }

    #[test]
    fn test_exponent_addition_dim16() {
        let a = random_sedenion(42);
        for m in -3..=3 {
            for n in -3..=3 {
                let err = verify_exponent_addition(&a, m, n);
                assert!(err < 1e-6, "a^{m}*a^{n} != a^{}: err={err}", m + n);
            }
        }
    }

    #[test]
    fn test_power_of_power_dim16() {
        let a = random_sedenion(137);
        for m in -2..=2 {
            for n in -2..=2 {
                let err = verify_power_of_power(&a, m, n);
                assert!(err < 1e-5, "(a^{m})^{n} != a^{}: err={err}", m * n);
            }
        }
    }

    #[test]
    fn test_generalized_flexibility_dim16() {
        let a = random_sedenion(99);
        let b = random_sedenion(100);
        for m in 0..=3 {
            for n in 0..=3 {
                let err = verify_generalized_flexibility(&a, &b, m, n);
                assert!(err < 1e-6, "(a^{m},b,a^{n}) != 0: err={err}");
            }
        }
    }

    #[test]
    fn test_no_nilpotent_dim16() {
        let a = random_sedenion(42);
        assert!(verify_no_nilpotent(&a, 10));
    }

    #[test]
    fn test_associator_square_dim8() {
        // Octonion triple: (e1, e2, e4)
        let a = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let c = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let err = verify_associator_square(&a, &b, &c);
        assert!(err < 1e-10, "associator^2 should be -N(assoc): err={err}");
    }

    #[test]
    fn test_associator_square_dim16() {
        let a = random_sedenion(1);
        let b = random_sedenion(2);
        let c = random_sedenion(3);
        let err = verify_associator_square(&a, &b, &c);
        assert!(err < 1e-6, "sedenion associator^2 should be -N(assoc): err={err}");
    }
}

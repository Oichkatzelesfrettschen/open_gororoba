//! Hurwitz-Radon function (Part VI, eq.14-15).
//!
//! The generalized composition problem: for dimension n, what is the
//! maximum number rho(n) of linearly independent vector fields on S^{n-1}?
//!
//! Write n = 2^{4a+b} * c where c is odd and 0 <= b <= 3.
//! Then rho(n) = 2^b + 8*a.
//!
//! The Hurwitz composition theorem is the special case: rho(n) = n iff n in {1,2,4,8}.
//!
//! Mirrors: HurwitzTheorem.v Part VI, hurwitz_radon function.

/// Compute the Hurwitz-Radon number rho(n).
///
/// rho(n) is the maximum number of real n x n matrices A_1, ..., A_rho
/// that are orthogonal (A_i * A_j' = 0 for i != j) and satisfy A_i * A_i' = I.
///
/// Write n = 2^{4a+b} * c with c odd, 0 <= b <= 3.
/// Then rho(n) = 2^b + 8*a.
///
/// Mirrors: HurwitzTheorem.v hurwitz_radon.
pub fn hurwitz_radon(n: usize) -> usize {
    if n == 0 {
        return 0;
    }

    // Factor out the largest power of 2
    let mut m = n;
    let mut v2 = 0_usize; // 2-adic valuation
    while m.is_multiple_of(2) {
        m /= 2;
        v2 += 1;
    }
    // n = 2^v2 * m, m odd
    // v2 = 4*a + b where 0 <= b <= 3
    let a = v2 / 4;
    let b = v2 % 4;

    (1 << b) + 8 * a
}

/// Check whether square composition (rho(n) = n) is possible.
///
/// This is Hurwitz's theorem: only n = 1, 2, 4, 8.
///
/// Mirrors: HurwitzTheorem.v summary: "Hurwitz theorem COMPLETE".
pub fn is_composition_algebra_dim(n: usize) -> bool {
    hurwitz_radon(n) >= n
}

/// Classify dimension n by the Hurwitz-Radon theory.
///
/// Returns (n, rho(n), is_composition, deficit) where
/// deficit = n - rho(n) measures "how far" from composition.
pub fn classify(n: usize) -> (usize, usize, bool, usize) {
    let rho = hurwitz_radon(n);
    let is_comp = rho >= n;
    let deficit = if is_comp { 0 } else { n - rho };
    (n, rho, is_comp, deficit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hurwitz_radon_composition_dims() {
        assert_eq!(hurwitz_radon(1), 1); // R
        assert_eq!(hurwitz_radon(2), 2); // C
        assert_eq!(hurwitz_radon(4), 4); // H
        assert_eq!(hurwitz_radon(8), 8); // O
    }

    #[test]
    fn test_hurwitz_radon_non_composition() {
        assert_eq!(hurwitz_radon(16), 9);
        assert_eq!(hurwitz_radon(32), 10);
        assert_eq!(hurwitz_radon(64), 12);
        assert_eq!(hurwitz_radon(128), 16);
        assert_eq!(hurwitz_radon(256), 17);
    }

    #[test]
    fn test_hurwitz_radon_odd() {
        // Odd n: v2 = 0, a = 0, b = 0, rho = 1
        assert_eq!(hurwitz_radon(1), 1);
        assert_eq!(hurwitz_radon(3), 1);
        assert_eq!(hurwitz_radon(5), 1);
        assert_eq!(hurwitz_radon(7), 1);
    }

    #[test]
    fn test_is_composition_algebra_dim() {
        for n in 1..=64 {
            let expected = matches!(n, 1 | 2 | 4 | 8);
            assert_eq!(
                is_composition_algebra_dim(n),
                expected,
                "n={n}: expected {expected}"
            );
        }
    }

    #[test]
    fn test_classify_cd_tower() {
        let dims = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
        for &n in &dims {
            let (_, rho, is_comp, deficit) = classify(n);
            if n <= 8 {
                assert!(is_comp, "n={n} should be composition");
                assert_eq!(deficit, 0);
            } else {
                assert!(!is_comp, "n={n} should NOT be composition");
                assert!(deficit > 0);
                assert!(rho < n, "rho({n})={rho} should be < {n}");
            }
        }
    }
}

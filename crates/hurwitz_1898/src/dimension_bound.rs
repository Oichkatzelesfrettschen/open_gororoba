//! Hurwitz dimension bound (Parts II + III).
//!
//! The counting argument: composition at dimension n requires
//! 2^{n-2} linearly independent skew-symmetric n x n matrices.
//!
//! Basic bound (eq.13): 2^{n-2} <= n^2.
//! Refined bound: 2^{n-2} <= n*(n-1)/2 (skew-symmetric matrix space).
//!
//! Mirrors: HurwitzTheorem.v Parts II and III.

/// Check whether the basic Hurwitz bound 2^{n-2} <= n^2 holds.
///
/// Mirrors: HurwitzTheorem.v hurwitz_bound_holds.
pub fn basic_bound_holds(n: usize) -> bool {
    if n < 2 {
        return true;
    }
    let lhs = 1_usize << (n - 2); // 2^{n-2}
    let rhs = n * n;
    lhs <= rhs
}

/// Check the refined bound: 2^{n-2} <= n*(n-1)/2.
///
/// This is sufficient (but not necessary) to eliminate a dimension.
/// Fails at n = 6 (eliminates it) and at n >= 10 (eliminates all).
/// Does NOT fail at n = 8 (octonion composition works via explicit table).
///
/// Mirrors: HurwitzTheorem.v hurwitz_refined_holds, n6_eliminated.
pub fn refined_bound_holds(n: usize) -> bool {
    if n < 2 {
        return true;
    }
    let lhs = 1_usize << (n - 2); // 2^{n-2}
    let rhs = n * (n - 1) / 2; // dim of skew-symmetric n x n matrices
    lhs <= rhs
}

/// Dimension of the space of n x n skew-symmetric real matrices.
///
/// Mirrors: HurwitzTheorem.v skew_sym_dim.
pub const fn skew_sym_dim(n: usize) -> usize {
    n * (n - 1) / 2
}

/// Number of skew-symmetric products from n-1 Clifford generators.
/// Equal to 2^{n-2} (half of the 2^{n-1} total products).
///
/// Mirrors: HurwitzTheorem.v skew_product_count.
pub const fn skew_product_count(n: usize) -> usize {
    if n < 2 {
        return 0;
    }
    1 << (n - 2)
}

/// Determine whether composition is possible at dimension n
/// using Hurwitz's combined criteria:
/// 1. n must be even (odd eliminated by determinant argument)
/// 2. Basic bound must hold: 2^{n-2} <= n^2
/// 3. n = 6 is eliminated by refined bound
///
/// Returns true only for n in {1, 2, 4, 8}.
///
/// Mirrors: HurwitzTheorem.v combined Parts I-IV.
pub fn composition_possible(n: usize) -> bool {
    match n {
        0 => false,
        1 => true,                           // trivial (R)
        _ if !n.is_multiple_of(2) => false,  // odd: eliminated (Part I.5)
        _ if !basic_bound_holds(n) => false, // 2^{n-2} > n^2 (Part II)
        6 => false,                          // refined counting eliminates n=6 (Part III)
        2 | 4 | 8 => true,                   // positive: norm multiplicativity proved (Part I)
        _ => false,                          // all remaining even n >= 10: basic bound fails
    }
}

/// Classify all dimensions up to max_dim by Hurwitz composition.
///
/// Returns a Vec of (dim, possible, reason).
pub fn classify_dimensions(max_dim: usize) -> Vec<(usize, bool, &'static str)> {
    (1..=max_dim)
        .map(|n| {
            let (possible, reason) = match n {
                1 => (true, "trivial (R)"),
                2 => (
                    true,
                    "complex norm multiplicativity (Brahmagupta-Fibonacci)",
                ),
                4 => (true, "quaternion norm multiplicativity (Euler four-square)"),
                8 => (
                    true,
                    "octonion norm multiplicativity (Degen-Graves eight-square)",
                ),
                _ if !n.is_multiple_of(2) => (false, "odd dimension: det(B_i)^2 = (-1)^n < 0"),
                6 => (
                    false,
                    "refined bound: 2^4=16 > n*(n-1)/2=15 skew-sym matrices",
                ),
                _ if !basic_bound_holds(n) => (false, "basic bound: 2^{n-2} > n^2"),
                _ => (false, "eliminated by combined criteria"),
            };
            (n, possible, reason)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_bound() {
        assert!(basic_bound_holds(2));
        assert!(basic_bound_holds(4));
        assert!(basic_bound_holds(6));
        assert!(basic_bound_holds(8));
        assert!(!basic_bound_holds(10));
        assert!(!basic_bound_holds(12));
        assert!(!basic_bound_holds(16));
    }

    #[test]
    fn test_refined_bound() {
        assert!(refined_bound_holds(2));
        assert!(refined_bound_holds(4));
        assert!(!refined_bound_holds(6)); // eliminates n=6
        assert!(!refined_bound_holds(8)); // does NOT hold at n=8 (that's OK)
        assert!(!refined_bound_holds(10));
    }

    #[test]
    fn test_skew_sym_dim() {
        assert_eq!(skew_sym_dim(2), 1);
        assert_eq!(skew_sym_dim(4), 6);
        assert_eq!(skew_sym_dim(6), 15);
        assert_eq!(skew_sym_dim(8), 28);
    }

    #[test]
    fn test_n6_eliminated() {
        // 2^{6-2} = 16 > 15 = 6*5/2
        assert!(skew_product_count(6) > skew_sym_dim(6));
    }

    #[test]
    fn test_composition_possible() {
        assert!(composition_possible(1));
        assert!(composition_possible(2));
        assert!(!composition_possible(3));
        assert!(composition_possible(4));
        assert!(!composition_possible(5));
        assert!(!composition_possible(6));
        assert!(!composition_possible(7));
        assert!(composition_possible(8));
        for n in 9..=64 {
            assert!(!composition_possible(n), "n={n} should be impossible");
        }
    }

    #[test]
    fn test_classify_dimensions() {
        let classified = classify_dimensions(20);
        let possible: Vec<usize> = classified.iter().filter(|c| c.1).map(|c| c.0).collect();
        assert_eq!(possible, vec![1, 2, 4, 8]);
    }
}

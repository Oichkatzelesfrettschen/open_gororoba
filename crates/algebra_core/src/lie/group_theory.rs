//! Group theory: finite groups, orders, and basic operations.
//!
//! Provides utilities for finite group computations relevant to physics:
//! - PSL(2,q) projective special linear groups
//! - Symmetric group Sn
//! - Alternating group An
//! - Group order formulas
//!
//! # Literature
//! - Artin, M. (2011). Algebra (2nd ed.), Chapters 6-7
//! - Rotman, J. (2010). Advanced Modern Algebra (2nd ed.)
//! - Wilson, R.A. (2009). The Finite Simple Groups
//!
//! # Physics Context
//! Finite groups appear in:
//! - PSL(2,7) acts on sedenion box-kites (de Marrais)
//! - Exceptional groups G2, F4, E6, E7, E8 in particle physics
//! - Symmetry groups of crystal lattices
//! - Gauge group subgroups in GUT models

/// Compute gcd using Euclidean algorithm.
fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Order of the projective special linear group PSL(2,q).
///
/// For q a prime power:
/// |PSL(2,q)| = q(q^2 - 1) / gcd(2, q-1)
///
/// # Arguments
/// * `q` - Prime power > 1
///
/// # Panics
/// If q <= 1
///
/// # Example
/// ```
/// use algebra_core::order_psl2_q;
/// assert_eq!(order_psl2_q(2), 6);   // PSL(2,2) = S3
/// assert_eq!(order_psl2_q(3), 12);  // PSL(2,3) = A4
/// assert_eq!(order_psl2_q(5), 60);  // PSL(2,5) = A5
/// assert_eq!(order_psl2_q(7), 168); // PSL(2,7) = GL(3,2)
/// ```
pub fn order_psl2_q(q: u64) -> u64 {
    if q <= 1 {
        panic!("q must be > 1");
    }
    let g = gcd(2, q - 1);
    q * (q * q - 1) / g
}

/// Order of the symmetric group Sn.
///
/// |Sn| = n!
///
/// # Panics
/// If n > 20 (factorial overflow)
pub fn order_symmetric(n: u64) -> u64 {
    if n > 20 {
        panic!("Factorial overflow for n > 20");
    }
    (1..=n).product()
}

/// Order of the alternating group An.
///
/// |An| = n!/2 for n >= 2
pub fn order_alternating(n: u64) -> u64 {
    if n < 2 {
        return 1;
    }
    order_symmetric(n) / 2
}

/// Order of the general linear group GL(n,q) over F_q.
///
/// |GL(n,q)| = prod_{k=0}^{n-1} (q^n - q^k)
pub fn order_gl(n: u64, q: u64) -> u128 {
    if n == 0 {
        return 1;
    }
    let qn = (q as u128).pow(n as u32);
    let mut order: u128 = 1;
    for k in 0..n {
        let qk = (q as u128).pow(k as u32);
        order *= qn - qk;
    }
    order
}

/// Order of the special linear group SL(n,q) over F_q.
///
/// |SL(n,q)| = |GL(n,q)| / (q-1)
pub fn order_sl(n: u64, q: u64) -> u128 {
    if q <= 1 {
        panic!("q must be > 1");
    }
    order_gl(n, q) / (q - 1) as u128
}

/// Check if n is a prime (simple trial division).
pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 {
        return true;
    }
    if n.is_multiple_of(2) {
        return false;
    }
    let mut i = 3;
    while i * i <= n {
        if n.is_multiple_of(i) {
            return false;
        }
        i += 2;
    }
    true
}

/// Check if n is a prime power.
///
/// Returns Some((p, k)) if n = p^k for prime p and k >= 1, None otherwise.
pub fn prime_power(n: u64) -> Option<(u64, u32)> {
    if n < 2 {
        return None;
    }

    // Find smallest prime factor
    let mut p = 2u64;
    while p * p <= n {
        if n.is_multiple_of(p) {
            // p is the smallest prime factor
            let mut k = 0u32;
            let mut m = n;
            while m.is_multiple_of(p) {
                m /= p;
                k += 1;
            }
            // If m == 1, then n = p^k
            return if m == 1 { Some((p, k)) } else { None };
        }
        p += if p == 2 { 1 } else { 2 };
    }

    // n is prime
    Some((n, 1))
}

/// PSL(2,7) - the simple group of order 168.
///
/// This group acts on sedenion box-kites (de Marrais structure).
/// It is isomorphic to GL(3,2), the automorphism group of the Fano plane.
pub const PSL_2_7_ORDER: u64 = 168;

/// Orders of exceptional simple groups.
pub mod exceptional {
    /// |G2(q)| for G2 over F_q
    pub fn order_g2(q: u64) -> u128 {
        let q2 = (q * q) as u128;
        let q6 = q2 * q2 * q2;
        q6 * (q2 - 1) * (q6 - 1)
    }

    /// Order of the sporadic Mathieu group M11
    pub const M11_ORDER: u64 = 7920;

    /// Order of the sporadic Mathieu group M12
    pub const M12_ORDER: u64 = 95040;

    /// Order of the sporadic Mathieu group M22
    pub const M22_ORDER: u64 = 443520;

    /// Order of the sporadic Mathieu group M23
    pub const M23_ORDER: u64 = 10200960;

    /// Order of the sporadic Mathieu group M24
    pub const M24_ORDER: u64 = 244823040;

    /// Order of the Monster group (largest sporadic simple group)
    /// Approximately 8 * 10^53
    pub const MONSTER_ORDER_APPROX: f64 = 8.08e53;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_psl2_q() {
        // PSL(2,2) = S3
        assert_eq!(order_psl2_q(2), 6);

        // PSL(2,3) = A4
        assert_eq!(order_psl2_q(3), 12);

        // PSL(2,4) = PSL(2,2^2) = A5
        assert_eq!(order_psl2_q(4), 60);

        // PSL(2,5) = A5
        assert_eq!(order_psl2_q(5), 60);

        // PSL(2,7) = GL(3,2), order 168
        assert_eq!(order_psl2_q(7), 168);

        // PSL(2,8) = PSL(2,2^3)
        assert_eq!(order_psl2_q(8), 504);

        // PSL(2,9) = A6
        assert_eq!(order_psl2_q(9), 360);
    }

    #[test]
    fn test_order_symmetric() {
        assert_eq!(order_symmetric(1), 1);
        assert_eq!(order_symmetric(2), 2);
        assert_eq!(order_symmetric(3), 6);
        assert_eq!(order_symmetric(4), 24);
        assert_eq!(order_symmetric(5), 120);
        assert_eq!(order_symmetric(10), 3628800);
    }

    #[test]
    fn test_order_alternating() {
        assert_eq!(order_alternating(1), 1);
        assert_eq!(order_alternating(2), 1);
        assert_eq!(order_alternating(3), 3);
        assert_eq!(order_alternating(4), 12);
        assert_eq!(order_alternating(5), 60);
    }

    #[test]
    fn test_order_gl() {
        // GL(1,q) = F_q^* has order q-1
        assert_eq!(order_gl(1, 2), 1);
        assert_eq!(order_gl(1, 3), 2);
        assert_eq!(order_gl(1, 5), 4);

        // GL(2,2) has order (4-1)(4-2) = 3*2 = 6
        assert_eq!(order_gl(2, 2), 6);

        // GL(3,2) has order (8-1)(8-2)(8-4) = 7*6*4 = 168
        assert_eq!(order_gl(3, 2), 168);
    }

    #[test]
    fn test_is_prime() {
        assert!(!is_prime(0));
        assert!(!is_prime(1));
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(!is_prime(4));
        assert!(is_prime(5));
        assert!(is_prime(7));
        assert!(!is_prime(9));
        assert!(is_prime(11));
        assert!(is_prime(97));
        assert!(!is_prime(100));
    }

    #[test]
    fn test_prime_power() {
        assert_eq!(prime_power(1), None);
        assert_eq!(prime_power(2), Some((2, 1)));
        assert_eq!(prime_power(4), Some((2, 2)));
        assert_eq!(prime_power(8), Some((2, 3)));
        assert_eq!(prime_power(9), Some((3, 2)));
        assert_eq!(prime_power(27), Some((3, 3)));
        assert_eq!(prime_power(6), None); // 2 * 3
        assert_eq!(prime_power(12), None); // 2^2 * 3
    }

    #[test]
    fn test_exceptional_orders() {
        // G2(2) order
        let g2_2 = exceptional::order_g2(2);
        assert_eq!(g2_2, 12096);

        // Mathieu groups
        assert_eq!(exceptional::M11_ORDER, 7920);
        assert_eq!(exceptional::M12_ORDER, 95040);
    }
}

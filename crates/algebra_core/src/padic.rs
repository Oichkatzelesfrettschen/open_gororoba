//! P-adic analysis: valuations, absolute values, and Cantor set functions.
//!
//! Implements p-adic number theory utilities for ultrametric analysis.
//!
//! # Literature
//!
//! ## Foundational
//! - Koblitz, N. (1984). P-adic Numbers, P-adic Analysis, and Zeta Functions
//! - Gouvea, F. (1997). P-adic Numbers: An Introduction
//!
//! ## Physics Applications
//! - Dragovich, B. et al. (2017). p-Adic mathematical physics: the first 30 years.
//!   p-Adic Numbers, Ultrametric Analysis and Applications, 9(2), 87-121.
//! - Volovich, I.V. (1987). p-Adic string. Class. Quantum Grav. 4, L83.
//! - Aref'eva, I. & Volovich, I. (2024). p-Adic strings and cosmology (arXiv:2401.xxxxx)
//!
//! ## Recent Advances (2023-2024)
//! - IAS Special Year on p-adic Arithmetic Geometry (2023-2024)
//! - Perfectoid geometry and prismatic cohomology advances
//! - p-adic Langlands program developments
//! - Bhatt et al. (2024). Perfectoid pure singularities (arXiv:2409.17965)
//!
//! # Physics Context in This Repo
//! P-adic analysis appears in:
//! - **String theory**: Veneziano amplitude generalizes to p-adic integrals
//! - **Quantum cosmology**: Discreteness at Planck scale via p-adic effects
//! - **Dark energy**: Non-local p-adic effective actions from string field theory
//! - **Hierarchical structures**: FRB dispersion measures, fractal cosmology
//! - **Ultrametric spaces**: Natural for tree-like causal structures
//!
//! # Key Concepts
//! - p-adic valuation v_p(n): largest k such that p^k divides n
//! - p-adic absolute value |x|_p = p^{-v_p(x)}
//! - Ultrametric inequality: |x + y|_p <= max(|x|_p, |y|_p)

use std::fmt;

/// P-adic valuation of an integer.
///
/// Returns the largest k >= 0 such that p^k divides n.
///
/// # Panics
/// - If p < 2 (not a valid prime base)
/// - If n == 0 (valuation is infinite)
///
/// # Example
/// ```
/// use algebra_core::padic::vp_int;
/// assert_eq!(vp_int(12, 2), 2);  // 12 = 2^2 * 3
/// assert_eq!(vp_int(12, 3), 1);  // 12 = 4 * 3^1
/// assert_eq!(vp_int(7, 2), 0);   // 7 is odd
/// ```
pub fn vp_int(n: i64, p: u64) -> i32 {
    if p < 2 {
        panic!("p must be >= 2");
    }
    if n == 0 {
        panic!("v_p(0) is undefined (infinite)");
    }

    let mut n_abs = n.unsigned_abs();
    let mut k = 0i32;
    while n_abs.is_multiple_of(p) {
        n_abs /= p;
        k += 1;
    }
    k
}

/// Rational number for p-adic computations.
///
/// Stored in lowest terms with positive denominator.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rational {
    pub num: i64,
    pub den: u64,
}

impl Rational {
    /// Create a new rational, reducing to lowest terms.
    pub fn new(num: i64, den: i64) -> Self {
        if den == 0 {
            panic!("Denominator cannot be zero");
        }

        let sign = if den < 0 { -1 } else { 1 };
        let num = sign * num;
        let den = den.unsigned_abs();

        let g = gcd(num.unsigned_abs(), den);
        Rational {
            num: num / g as i64,
            den: den / g,
        }
    }

    /// Create from integer.
    pub fn from_int(n: i64) -> Self {
        Rational { num: n, den: 1 }
    }

    /// Zero.
    pub fn zero() -> Self {
        Rational { num: 0, den: 1 }
    }

    /// One.
    pub fn one() -> Self {
        Rational { num: 1, den: 1 }
    }

    /// Check if zero.
    pub fn is_zero(&self) -> bool {
        self.num == 0
    }

    /// Check if in [0, 1].
    pub fn in_unit_interval(&self) -> bool {
        self.num >= 0 && (self.num as u64) <= self.den
    }
}

impl fmt::Debug for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

/// GCD using Euclidean algorithm.
fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    if a == 0 { 1 } else { a }
}

/// P-adic valuation of a rational.
///
/// For q = a/b in lowest terms: v_p(q) = v_p(a) - v_p(b)
///
/// # Panics
/// - If p < 2
/// - If q == 0
pub fn vp(q: Rational, p: u64) -> i32 {
    if p < 2 {
        panic!("p must be >= 2");
    }
    if q.is_zero() {
        panic!("v_p(0) is undefined (infinite)");
    }

    vp_int(q.num, p) - vp_int(q.den as i64, p)
}

/// P-adic absolute value |q|_p.
///
/// By definition:
/// - |0|_p = 0
/// - |q|_p = p^{-v_p(q)} for q != 0
pub fn abs_p(q: Rational, p: u64) -> f64 {
    if q.is_zero() {
        return 0.0;
    }
    let v = vp(q, p);
    (p as f64).powi(-v)
}

/// Check if n is a power of two.
pub fn is_power_of_two(n: u64) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

/// Check if a rational is dyadic (denominator is power of 2).
pub fn is_dyadic(q: Rational) -> bool {
    is_power_of_two(q.den)
}

/// Ternary (base-3) digit expansion for Cantor set analysis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CantorDigits {
    pub digits: Vec<u8>,
}

impl CantorDigits {
    /// Check if this represents a Cantor set point (digits are only 0 or 2).
    pub fn is_cantor(&self) -> bool {
        self.digits.iter().all(|&d| d == 0 || d == 2)
    }
}

/// Compute the first `n_digits` base-3 digits of q in [0,1].
///
/// Exact for rationals with denominator a power of 3.
///
/// # Panics
/// - If q is not in [0, 1]
pub fn ternary_digits_power3(q: Rational, n_digits: usize) -> CantorDigits {
    if !q.in_unit_interval() {
        panic!("q must be in [0, 1]");
    }

    let mut digits = Vec::with_capacity(n_digits);

    // Work with exact rational arithmetic
    let mut num = q.num as u64;
    let den = q.den;

    for _ in 0..n_digits {
        // x *= 3
        num *= 3;

        // d = floor(x)
        let d = (num / den) as u8;
        let d = d.min(2); // Clamp to valid ternary digit

        digits.push(d);

        // x -= d
        num -= d as u64 * den;
    }

    CantorDigits { digits }
}

/// Cantor (Devil's staircase) function restricted to Cantor set points.
///
/// For a ternary expansion using only digits {0, 2}, map:
/// - 0 -> 0 (binary digit)
/// - 2 -> 1 (binary digit)
///
/// Returns the value as a rational.
///
/// # Panics
/// - If q is not in [0, 1]
/// - If q's ternary expansion contains 1s (not a Cantor set point)
pub fn cantor_function_on_cantor(q: Rational, n_digits: usize) -> Rational {
    let digs = ternary_digits_power3(q, n_digits);

    if !digs.is_cantor() {
        panic!("q is not a Cantor set point (ternary expansion contains 1)");
    }

    // Convert ternary {0, 2} to binary {0, 1}
    let mut num: u64 = 0;
    let mut den: u64 = 1;

    for &d in &digs.digits {
        num *= 2;
        den *= 2;
        if d == 2 {
            num += 1;
        }
    }

    Rational::new(num as i64, den as i64)
}

/// Compute p-adic distance between two rationals.
///
/// d_p(x, y) = |x - y|_p
pub fn padic_distance(x: Rational, y: Rational, p: u64) -> f64 {
    // x - y = (x.num * y.den - y.num * x.den) / (x.den * y.den)
    let num = x.num * (y.den as i64) - y.num * (x.den as i64);
    let den = (x.den * y.den) as i64;
    let diff = Rational::new(num, den);
    abs_p(diff, p)
}

/// Check if three points satisfy the ultrametric inequality.
///
/// In a p-adic metric: d(x, z) <= max(d(x, y), d(y, z))
/// (stronger than triangle inequality)
pub fn check_ultrametric(x: Rational, y: Rational, z: Rational, p: u64) -> bool {
    let d_xy = padic_distance(x, y, p);
    let d_yz = padic_distance(y, z, p);
    let d_xz = padic_distance(x, z, p);

    d_xz <= d_xy.max(d_yz) + 1e-14 // Small epsilon for floating point
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vp_int_basic() {
        assert_eq!(vp_int(12, 2), 2); // 12 = 2^2 * 3
        assert_eq!(vp_int(12, 3), 1); // 12 = 4 * 3
        assert_eq!(vp_int(7, 2), 0);  // 7 is odd
        assert_eq!(vp_int(8, 2), 3);  // 8 = 2^3
        assert_eq!(vp_int(27, 3), 3); // 27 = 3^3
    }

    #[test]
    fn test_vp_int_negative() {
        assert_eq!(vp_int(-12, 2), 2);
        assert_eq!(vp_int(-27, 3), 3);
    }

    #[test]
    #[should_panic]
    fn test_vp_int_zero_panics() {
        vp_int(0, 2);
    }

    #[test]
    fn test_vp_rational() {
        let q = Rational::new(12, 9); // 4/3 in lowest terms
        assert_eq!(vp(q, 2), 2);  // v_2(4) - v_2(3) = 2 - 0 = 2
        assert_eq!(vp(q, 3), -1); // v_3(4) - v_3(3) = 0 - 1 = -1
    }

    #[test]
    fn test_abs_p() {
        let q = Rational::new(1, 4); // 1/4 = 2^{-2}
        let abs_2 = abs_p(q, 2);
        assert!((abs_2 - 4.0).abs() < 1e-10); // |1/4|_2 = 2^{-(-2)} = 4

        assert_eq!(abs_p(Rational::zero(), 2), 0.0);
    }

    #[test]
    fn test_is_power_of_two() {
        assert!(is_power_of_two(1));
        assert!(is_power_of_two(2));
        assert!(is_power_of_two(4));
        assert!(is_power_of_two(1024));
        assert!(!is_power_of_two(0));
        assert!(!is_power_of_two(3));
        assert!(!is_power_of_two(6));
    }

    #[test]
    fn test_is_dyadic() {
        assert!(is_dyadic(Rational::new(1, 2)));
        assert!(is_dyadic(Rational::new(3, 4)));
        assert!(is_dyadic(Rational::new(5, 8)));
        assert!(!is_dyadic(Rational::new(1, 3)));
        assert!(!is_dyadic(Rational::new(1, 6)));
    }

    #[test]
    fn test_ternary_digits() {
        // 1/3 in base 3 is 0.1
        let digs = ternary_digits_power3(Rational::new(1, 3), 5);
        assert_eq!(digs.digits[0], 1);
        assert!(!digs.is_cantor()); // Contains 1

        // 2/3 in base 3 is 0.2
        let digs2 = ternary_digits_power3(Rational::new(2, 3), 5);
        assert_eq!(digs2.digits[0], 2);
    }

    #[test]
    fn test_cantor_digits() {
        // 0 is in Cantor set
        let digs = ternary_digits_power3(Rational::zero(), 10);
        assert!(digs.is_cantor());
        assert!(digs.digits.iter().all(|&d| d == 0));

        // 2/9 = 0.02 in base 3 - Cantor set point
        let digs2 = ternary_digits_power3(Rational::new(2, 9), 10);
        assert_eq!(digs2.digits[0], 0);
        assert_eq!(digs2.digits[1], 2);
        assert!(digs2.is_cantor());
    }

    #[test]
    fn test_cantor_function() {
        // 0 -> 0
        let result = cantor_function_on_cantor(Rational::zero(), 10);
        assert_eq!(result, Rational::zero());

        // 2/3 = 0.2 in base 3 -> 0.1 in binary = 1/2
        let result2 = cantor_function_on_cantor(Rational::new(2, 3), 10);
        assert_eq!(result2.num, 1);
        // den = 2^10 = 1024, but 1/1024 reduces differently
    }

    #[test]
    fn test_ultrametric_inequality() {
        let x = Rational::from_int(1);
        let y = Rational::from_int(5);
        let z = Rational::from_int(9);

        // Check ultrametric for p=2
        assert!(check_ultrametric(x, y, z, 2));
    }

    #[test]
    fn test_padic_distance() {
        let x = Rational::from_int(0);
        let y = Rational::from_int(4); // 4 = 2^2

        let d = padic_distance(x, y, 2);
        // |4|_2 = 2^{-2} = 0.25
        assert!((d - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_rational_reduction() {
        let q = Rational::new(12, 8);
        assert_eq!(q.num, 3);
        assert_eq!(q.den, 2);

        let q2 = Rational::new(-6, -4);
        assert_eq!(q2.num, 3);
        assert_eq!(q2.den, 2);

        let q3 = Rational::new(6, -4);
        assert_eq!(q3.num, -3);
        assert_eq!(q3.den, 2);
    }
}

//! Wheel algebra: algebraic structure with total division (including by zero).
//!
//! Implements Carlstrom's wheel axioms, providing a mathematically consistent
//! way to handle division by zero using the (0/0) absorbing element.
//!
//! # Literature
//! - Carlstrom, J. (2004). Wheels - On Division by Zero
//! - Setzer, A. (1997). Wheels
//!
//! # Key Properties
//! - Division is total: x/0 = infinity, 0/0 = NaN (bottom element)
//! - NaN absorbs under addition: x + NaN = NaN
//! - Infinity: 1/0, reciprocal of 0
//! - All 8 Carlstrom axioms are satisfied

use num_traits::{One, Zero};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg};

/// Wheel element over rationals.
///
/// Representation:
/// - Finite rationals: (value, 1) with normalized value
/// - Infinity: (1, 0)
/// - NaN (0/0): (0, 0) - absorbing element
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct WheelQ {
    /// Numerator (normalized)
    pub num: i64,
    /// Denominator (normalized: 1 for finite, 0 for inf/nan)
    pub den: i64,
}

impl WheelQ {
    /// Create a wheel element from numerator and denominator.
    ///
    /// Normalizes the representation:
    /// - (n, 0) where n != 0 -> infinity (1, 0)
    /// - (0, 0) -> NaN (0, 0)
    /// - (n, d) where d != 0 -> normalized rational
    pub fn new(num: i64, den: i64) -> Self {
        if den == 0 {
            if num == 0 {
                // 0/0 = NaN
                WheelQ { num: 0, den: 0 }
            } else {
                // n/0 = infinity (normalized to 1/0)
                WheelQ { num: 1, den: 0 }
            }
        } else {
            // Normalize the rational
            let g = gcd(num.abs(), den.abs());
            let sign = if den < 0 { -1 } else { 1 };
            WheelQ {
                num: sign * num / g,
                den: den.abs() / g,
            }
        }
    }

    /// Create from integer (n/1).
    pub fn from_int(n: i64) -> Self {
        WheelQ { num: n, den: 1 }
    }

    /// Zero element (0/1).
    pub fn zero() -> Self {
        WheelQ { num: 0, den: 1 }
    }

    /// One element (1/1).
    pub fn one() -> Self {
        WheelQ { num: 1, den: 1 }
    }

    /// Infinity (1/0).
    pub fn inf() -> Self {
        WheelQ { num: 1, den: 0 }
    }

    /// NaN / bottom element (0/0).
    pub fn nan() -> Self {
        WheelQ { num: 0, den: 0 }
    }

    /// Check if this is NaN (0/0).
    pub fn is_nan(&self) -> bool {
        self.num == 0 && self.den == 0
    }

    /// Check if this is infinity (1/0).
    pub fn is_inf(&self) -> bool {
        self.num == 1 && self.den == 0
    }

    /// Check if this is a finite value.
    pub fn is_finite(&self) -> bool {
        self.den != 0
    }

    /// Wheel addition: (a/b) + (c/d) = (ad + bc) / (bd)
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: Self) -> Self {
        // If either is NaN, result is NaN
        if self.is_nan() || other.is_nan() {
            return Self::nan();
        }

        let new_num = self.num * other.den + self.den * other.num;
        let new_den = self.den * other.den;
        Self::new(new_num, new_den)
    }

    /// Wheel multiplication: (a/b) * (c/d) = (ac) / (bd)
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, other: Self) -> Self {
        Self::new(self.num * other.num, self.den * other.den)
    }

    /// Wheel reciprocal/involution: /(a/b) = b/a
    ///
    /// Key wheel property: /0 = infinity, /infinity = 0, /NaN = NaN
    pub fn inv(self) -> Self {
        Self::new(self.den, self.num)
    }

    /// Wheel division: x/y = x * /y
    #[allow(clippy::should_implement_trait)]
    pub fn div(self, other: Self) -> Self {
        self.mul(other.inv())
    }

    /// Negation: -(a/b) = (-a)/b
    #[allow(clippy::should_implement_trait)]
    pub fn neg(self) -> Self {
        WheelQ {
            num: -self.num,
            den: self.den,
        }
    }

    /// Convert to f64 (NaN and infinity map to f64 equivalents).
    pub fn to_f64(self) -> f64 {
        if self.is_nan() {
            f64::NAN
        } else if self.is_inf() {
            f64::INFINITY
        } else if self.den == 0 {
            // Shouldn't happen with proper normalization
            f64::NAN
        } else {
            self.num as f64 / self.den as f64
        }
    }
}

impl fmt::Debug for WheelQ {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_nan() {
            write!(f, "NaN")
        } else if self.is_inf() {
            write!(f, "Inf")
        } else if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

impl fmt::Display for WheelQ {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl Default for WheelQ {
    fn default() -> Self {
        Self::zero()
    }
}

impl Add for WheelQ {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        WheelQ::add(self, rhs)
    }
}

impl Mul for WheelQ {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        WheelQ::mul(self, rhs)
    }
}

impl Div for WheelQ {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        WheelQ::div(self, rhs)
    }
}

impl Neg for WheelQ {
    type Output = Self;
    fn neg(self) -> Self::Output {
        WheelQ::neg(self)
    }
}

impl Zero for WheelQ {
    fn zero() -> Self {
        WheelQ::zero()
    }
    fn is_zero(&self) -> bool {
        self.num == 0 && self.den == 1
    }
}

impl One for WheelQ {
    fn one() -> Self {
        WheelQ::one()
    }
}

/// GCD using Euclidean algorithm.
fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    if a == 0 { 1 } else { a }
}

/// Verify Carlstrom's 8 wheel axioms on a set of elements.
///
/// Returns Ok(()) if all axioms hold, Err with description if not.
pub fn verify_carlstrom_axioms(elements: &[WheelQ]) -> Result<(), String> {
    let zero = WheelQ::zero();
    let one = WheelQ::one();

    // Axiom 1: <H, 0, +> is a commutative monoid
    for &x in elements {
        if x.add(zero) != x {
            return Err(format!("Axiom 1 failed: {} + 0 != {}", x, x));
        }
    }
    for &x in elements {
        for &y in elements {
            if x.add(y) != y.add(x) {
                return Err(format!("Axiom 1 failed: {} + {} != {} + {}", x, y, y, x));
            }
            for &z in elements {
                if x.add(y.add(z)) != x.add(y).add(z) {
                    return Err("Axiom 1 failed: addition not associative".to_string());
                }
            }
        }
    }

    // Axiom 2: <H, 1, *, /> is a commutative monoid with involution
    for &x in elements {
        if x.mul(one) != x {
            return Err(format!("Axiom 2 failed: {} * 1 != {}", x, x));
        }
        if x.inv().inv() != x {
            return Err(format!("Axiom 2 failed: //({}) != {}", x, x));
        }
    }
    for &x in elements {
        for &y in elements {
            if x.mul(y) != y.mul(x) {
                return Err("Axiom 2 failed: multiplication not commutative".to_string());
            }
            if x.mul(y).inv() != y.inv().mul(x.inv()) {
                return Err("Axiom 2 failed: /(xy) != /y * /x".to_string());
            }
            for &z in elements {
                if x.mul(y.mul(z)) != x.mul(y).mul(z) {
                    return Err("Axiom 2 failed: multiplication not associative".to_string());
                }
            }
        }
    }

    // Axiom 3: (x + y)*z + 0*z = x*z + y*z
    for &x in elements {
        for &y in elements {
            for &z in elements {
                let left = x.add(y).mul(z).add(zero.mul(z));
                let right = x.mul(z).add(y.mul(z));
                if left != right {
                    return Err(format!(
                        "Axiom 3 failed: ({}+{})*{} + 0*{} != {}*{} + {}*{}",
                        x, y, z, z, x, z, y, z
                    ));
                }
            }
        }
    }

    // Axiom 4: x/y + z + 0*y = (x + y*z)/y
    for &x in elements {
        for &y in elements {
            for &z in elements {
                let left = x.div(y).add(z).add(zero.mul(y));
                let right = x.add(y.mul(z)).div(y);
                if left != right {
                    return Err(format!(
                        "Axiom 4 failed: {}/{} + {} + 0*{} != ({} + {}*{})/{}",
                        x, y, z, y, x, y, z, y
                    ));
                }
            }
        }
    }

    // Axiom 5: 0*0 = 0
    if zero.mul(zero) != zero {
        return Err("Axiom 5 failed: 0*0 != 0".to_string());
    }

    // Axiom 6: (x + 0*y)*z = x*z + 0*y
    for &x in elements {
        for &y in elements {
            for &z in elements {
                let left = x.add(zero.mul(y)).mul(z);
                let right = x.mul(z).add(zero.mul(y));
                if left != right {
                    return Err("Axiom 6 failed".to_string());
                }
            }
        }
    }

    // Axiom 7: /(x + 0*y) = /x + 0*y
    for &x in elements {
        for &y in elements {
            let left = x.add(zero.mul(y)).inv();
            let right = x.inv().add(zero.mul(y));
            if left != right {
                return Err("Axiom 7 failed".to_string());
            }
        }
    }

    // Axiom 8: x + 0/0 = 0/0 (NaN absorbs)
    let nan = zero.div(zero);
    for &x in elements {
        if x.add(nan) != nan {
            return Err(format!("Axiom 8 failed: {} + NaN != NaN", x));
        }
    }

    Ok(())
}

/// Generate a canonical test set for wheel axiom verification.
pub fn canonical_test_set() -> Vec<WheelQ> {
    vec![
        WheelQ::zero(),
        WheelQ::one(),
        WheelQ::from_int(-1),
        WheelQ::from_int(2),
        WheelQ::new(1, 2),  // 1/2
        WheelQ::new(-1, 2), // -1/2
        WheelQ::inf(),
        WheelQ::nan(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_construction() {
        assert_eq!(WheelQ::zero(), WheelQ::new(0, 1));
        assert_eq!(WheelQ::one(), WheelQ::new(1, 1));
        assert_eq!(WheelQ::inf(), WheelQ::new(1, 0));
        assert_eq!(WheelQ::nan(), WheelQ::new(0, 0));
    }

    #[test]
    fn test_normalization() {
        assert_eq!(WheelQ::new(2, 4), WheelQ::new(1, 2));
        assert_eq!(WheelQ::new(-2, 4), WheelQ::new(-1, 2));
        assert_eq!(WheelQ::new(2, -4), WheelQ::new(-1, 2));
        assert_eq!(WheelQ::new(5, 0), WheelQ::inf());
        assert_eq!(WheelQ::new(-3, 0), WheelQ::inf()); // All n/0 normalize to inf
    }

    #[test]
    fn test_is_predicates() {
        assert!(WheelQ::nan().is_nan());
        assert!(!WheelQ::inf().is_nan());
        assert!(!WheelQ::zero().is_nan());

        assert!(WheelQ::inf().is_inf());
        assert!(!WheelQ::nan().is_inf());
        assert!(!WheelQ::one().is_inf());

        assert!(WheelQ::one().is_finite());
        assert!(!WheelQ::inf().is_finite());
        assert!(!WheelQ::nan().is_finite());
    }

    #[test]
    fn test_addition() {
        let half = WheelQ::new(1, 2);
        let one = WheelQ::one();
        let three_halves = WheelQ::new(3, 2);

        assert_eq!(half.add(one), three_halves);
        assert_eq!(one.add(half), three_halves);
    }

    #[test]
    fn test_nan_absorbs() {
        let nan = WheelQ::nan();
        let one = WheelQ::one();
        let inf = WheelQ::inf();

        assert_eq!(one.add(nan), nan);
        assert_eq!(nan.add(one), nan);
        assert_eq!(inf.add(nan), nan);
    }

    #[test]
    fn test_infinity_arithmetic() {
        let inf = WheelQ::inf();
        let zero = WheelQ::zero();
        let one = WheelQ::one();

        assert_eq!(one.div(zero), inf);
        assert_eq!(inf.inv(), zero);
        assert_eq!(zero.inv(), inf);
    }

    #[test]
    fn test_division_by_zero() {
        let zero = WheelQ::zero();

        // 0/0 = NaN
        assert!(zero.div(zero).is_nan());

        // 1/0 = inf
        assert!(WheelQ::one().div(zero).is_inf());
    }

    #[test]
    fn test_reciprocal_involution() {
        let half = WheelQ::new(1, 2);
        let two = WheelQ::from_int(2);

        assert_eq!(half.inv(), two);
        assert_eq!(half.inv().inv(), half);
    }

    #[test]
    fn test_carlstrom_axioms() {
        let test_set = canonical_test_set();
        let result = verify_carlstrom_axioms(&test_set);
        assert!(result.is_ok(), "Carlstrom axioms failed: {:?}", result);
    }

    #[test]
    fn test_to_f64() {
        assert_eq!(WheelQ::one().to_f64(), 1.0);
        assert_eq!(WheelQ::new(1, 2).to_f64(), 0.5);
        assert!(WheelQ::nan().to_f64().is_nan());
        assert!(WheelQ::inf().to_f64().is_infinite());
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", WheelQ::nan()), "NaN");
        assert_eq!(format!("{}", WheelQ::inf()), "Inf");
        assert_eq!(format!("{}", WheelQ::one()), "1");
        assert_eq!(format!("{}", WheelQ::new(1, 2)), "1/2");
    }
}

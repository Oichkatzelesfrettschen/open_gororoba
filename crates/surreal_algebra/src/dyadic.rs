//! Dyadic surreal numbers: exact arithmetic for surreal coefficients.
//!
//! A dyadic rational is k/2^n for integer k and non-negative integer n.
//! All surreal numbers born on day <= N are dyadic rationals.
//! This representation gives exact arithmetic (no floating-point error)
//! and is closed under addition, subtraction, and multiplication.
//!
//! # Representation
//!
//! We store (numerator: i128, shift: u32) where the value is
//! numerator / 2^shift.  The shift is always non-negative.
//! We normalize by removing trailing factors of 2 from the numerator
//! and decrementing the shift accordingly.
//!
//! # Birthday
//!
//! The "birthday" of a surreal number is the earliest day on which it
//! can be constructed.  For dyadic rationals:
//! - 0 has birthday 0
//! - +/-1 have birthday 1
//! - +/-1/2, +/-2 have birthday 2
//! - In general, k/2^n has birthday max(|k|, n) approximately
//!
//! The birthday filtration is used in D4a (surreal valuation).

/// A dyadic rational number: exact representation as numerator / 2^shift.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurrealDyadic {
    /// Numerator (can be negative).
    pub num: i128,
    /// Shift: the value is num / 2^shift.
    pub shift: u32,
}

impl SurrealDyadic {
    /// Create a new dyadic rational, normalized.
    pub fn new(num: i128, shift: u32) -> Self {
        let mut s = Self { num, shift };
        s.normalize();
        s
    }

    /// Create from an integer (shift = 0).
    pub fn from_int(n: i64) -> Self {
        Self { num: n as i128, shift: 0 }
    }

    /// The surreal number 0.
    pub fn zero() -> Self { Self { num: 0, shift: 0 } }

    /// The surreal number 1.
    pub fn one() -> Self { Self { num: 1, shift: 0 } }

    /// Check if this is zero.
    pub fn is_zero(&self) -> bool { self.num == 0 }

    /// Convert to f64 (lossy for large values).
    pub fn to_f64(&self) -> f64 {
        self.num as f64 / (1_u128 << self.shift) as f64
    }

    /// Normalize: remove trailing factors of 2 from numerator.
    fn normalize(&mut self) {
        if self.num == 0 {
            self.shift = 0;
            return;
        }
        while self.num % 2 == 0 && self.shift > 0 {
            self.num /= 2;
            self.shift -= 1;
        }
    }

    /// Align two dyadic rationals to the same shift (for addition).
    fn align(a: &Self, b: &Self) -> (i128, i128, u32) {
        if a.shift >= b.shift {
            let diff = a.shift - b.shift;
            (a.num, b.num << diff, a.shift)
        } else {
            let diff = b.shift - a.shift;
            (a.num << diff, b.num, b.shift)
        }
    }

    /// Approximate birthday: max(bit_length(|num|), shift).
    pub fn birthday(&self) -> u32 {
        let bit_len = if self.num == 0 { 0 } else {
            128 - self.num.unsigned_abs().leading_zeros()
        };
        bit_len.max(self.shift)
    }
}

impl std::ops::Add for SurrealDyadic {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let (a, b, shift) = Self::align(&self, &rhs);
        Self::new(a + b, shift)
    }
}

impl std::ops::Sub for SurrealDyadic {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let (a, b, shift) = Self::align(&self, &rhs);
        Self::new(a - b, shift)
    }
}

impl std::ops::Mul for SurrealDyadic {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::new(self.num * rhs.num, self.shift + rhs.shift)
    }
}

impl std::ops::Neg for SurrealDyadic {
    type Output = Self;
    fn neg(self) -> Self {
        Self { num: -self.num, shift: self.shift }
    }
}

impl std::fmt::Display for SurrealDyadic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.shift == 0 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, 1_u128 << self.shift)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_arithmetic() {
        let one = SurrealDyadic::one();
        let two = one + one;
        assert_eq!(two, SurrealDyadic::from_int(2));

        let half = SurrealDyadic::new(1, 1);
        assert_eq!(half + half, one);
        assert_eq!(half.to_f64(), 0.5);

        let quarter = half * half;
        assert_eq!(quarter, SurrealDyadic::new(1, 2));
        assert_eq!(quarter.to_f64(), 0.25);
    }

    #[test]
    fn test_birthday() {
        assert_eq!(SurrealDyadic::zero().birthday(), 0);
        assert_eq!(SurrealDyadic::one().birthday(), 1);
        assert_eq!(SurrealDyadic::from_int(-1).birthday(), 1);
        assert_eq!(SurrealDyadic::new(1, 1).birthday(), 1); // 1/2
        assert_eq!(SurrealDyadic::from_int(2).birthday(), 2);
        assert_eq!(SurrealDyadic::new(1, 2).birthday(), 2); // 1/4
    }

    #[test]
    fn test_normalization() {
        let a = SurrealDyadic::new(4, 2); // 4/4 = 1
        assert_eq!(a, SurrealDyadic::one());

        let b = SurrealDyadic::new(6, 1); // 6/2 = 3
        assert_eq!(b, SurrealDyadic::from_int(3));
    }
}

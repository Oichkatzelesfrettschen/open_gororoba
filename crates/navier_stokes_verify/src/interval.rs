// SPDX-License-Identifier: MIT

//! Exact rational arithmetic and outward transcendental enclosures.
//!
//! `num-bigint`, `num-rational`, `num-integer`, their dependencies, and the
//! compiled Rust implementation belong to the arithmetic trust boundary.
//! No floating-point arithmetic enters a bound. Binary64 input conversion
//! preserves the represented number exactly; it cannot recover error made
//! before that conversion. Resource limits return errors without an enclosure.

use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::{error::Error, fmt};

pub type Rational = BigRational;

pub const MAX_INPUT_BYTES: usize = 4096;
pub const MAX_RATIONAL_BITS: u64 = 32_768;
pub const MAX_PRECISION_BITS: u32 = 512;
pub const MAX_EXP_ARGUMENT: u32 = 1024;
const MAX_TAYLOR_TERMS: u32 = 4096;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArithmeticError {
    InvalidInput(&'static str),
    ResourceLimit(&'static str),
    ImplementationFailure(&'static str),
}

impl ArithmeticError {
    pub fn is_invalid_input(&self) -> bool {
        matches!(self, Self::InvalidInput(_))
    }

    pub fn is_implementation_failure(&self) -> bool {
        matches!(self, Self::ImplementationFailure(_))
    }
}

impl fmt::Display for ArithmeticError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(formatter, "invalid arithmetic input: {message}"),
            Self::ResourceLimit(message) => {
                write!(formatter, "arithmetic resource limit: {message}")
            }
            Self::ImplementationFailure(message) => {
                write!(formatter, "arithmetic implementation failure: {message}")
            }
        }
    }
}

impl Error for ArithmeticError {}

/// Reject malformed raw ratios as well as integers exceeding the bit budget.
pub fn check_rational(value: &Rational) -> Result<(), ArithmeticError> {
    if value.numer().bits() > MAX_RATIONAL_BITS || value.denom().bits() > MAX_RATIONAL_BITS {
        return Err(ArithmeticError::ResourceLimit(
            "rational exceeds 32768 bits",
        ));
    }
    if value.denom() <= &BigInt::zero() {
        return Err(ArithmeticError::InvalidInput(
            "denominator must be positive",
        ));
    }
    if value.numer().gcd(value.denom()) != BigInt::one() {
        return Err(ArithmeticError::InvalidInput("rational must be reduced"));
    }
    Ok(())
}

fn checked(value: Rational) -> Result<Rational, ArithmeticError> {
    check_rational(&value)?;
    Ok(value)
}

pub fn int(value: i64) -> Rational {
    Rational::from_integer(BigInt::from(value))
}

pub fn rational(numerator: i64, denominator: i64) -> Result<Rational, ArithmeticError> {
    if denominator == 0 {
        return Err(ArithmeticError::InvalidInput("division by zero"));
    }
    checked(Rational::new(
        BigInt::from(numerator),
        BigInt::from(denominator),
    ))
}

fn canonical_integer(text: &str, signed: bool) -> bool {
    let digits = if signed {
        text.strip_prefix('-').unwrap_or(text)
    } else {
        text
    };
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return false;
    }
    if digits == "0" {
        return text == "0";
    }
    !digits.starts_with('0')
}

/// Parse reduced rational text such as `0`, `-7`, or `11/13`.
///
/// Decimal points, exponent notation, whitespace, leading zeroes, a leading
/// plus sign, and noncanonical fractions such as `2/4` or `1/1` are rejected.
pub fn parse_rational(text: &str) -> Result<Rational, ArithmeticError> {
    if text.len() > MAX_INPUT_BYTES {
        return Err(ArithmeticError::ResourceLimit(
            "rational text exceeds 4096 bytes",
        ));
    }
    let (numerator, denominator) = match text.split_once('/') {
        Some((numerator, denominator)) => {
            if !canonical_integer(numerator, true) || !canonical_integer(denominator, false) {
                return Err(ArithmeticError::InvalidInput("noncanonical rational text"));
            }
            (numerator, denominator)
        }
        None => {
            if !canonical_integer(text, true) {
                return Err(ArithmeticError::InvalidInput("noncanonical integer text"));
            }
            (text, "1")
        }
    };
    let numerator = BigInt::parse_bytes(numerator.as_bytes(), 10)
        .ok_or(ArithmeticError::InvalidInput("invalid numerator"))?;
    let denominator = BigInt::parse_bytes(denominator.as_bytes(), 10)
        .ok_or(ArithmeticError::InvalidInput("invalid denominator"))?;
    if denominator.is_zero() {
        return Err(ArithmeticError::InvalidInput("division by zero"));
    }
    let value = checked(Rational::new(numerator, denominator))?;
    if to_string(&value) != text {
        return Err(ArithmeticError::InvalidInput(
            "rational text must be reduced",
        ));
    }
    Ok(value)
}

pub fn to_string(value: &Rational) -> String {
    value.to_string()
}

// Checked operands bound exact multiplication temporaries by 65536 bits and
// addition temporaries by 65537 bits. Reduced results must meet the public cap.
pub fn checked_add(left: &Rational, right: &Rational) -> Result<Rational, ArithmeticError> {
    check_rational(left)?;
    check_rational(right)?;
    checked(left + right)
}

pub fn checked_sub(left: &Rational, right: &Rational) -> Result<Rational, ArithmeticError> {
    check_rational(left)?;
    check_rational(right)?;
    checked(left - right)
}

pub fn checked_mul(left: &Rational, right: &Rational) -> Result<Rational, ArithmeticError> {
    check_rational(left)?;
    check_rational(right)?;
    checked(left * right)
}

pub fn checked_div(left: &Rational, right: &Rational) -> Result<Rational, ArithmeticError> {
    check_rational(left)?;
    check_rational(right)?;
    if right.is_zero() {
        return Err(ArithmeticError::InvalidInput("division by zero"));
    }
    checked(left / right)
}

/// Convert the represented finite IEEE 754 binary64 number exactly.
pub fn from_f64_exact(value: f64) -> Result<Rational, ArithmeticError> {
    let bits = value.to_bits();
    let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    if exponent_bits == 0x7ff {
        return Err(ArithmeticError::InvalidInput(
            "binary64 input must be finite",
        ));
    }
    let (mantissa, exponent) = if exponent_bits == 0 {
        (fraction, -1074)
    } else {
        (fraction | (1_u64 << 52), exponent_bits - 1023 - 52)
    };
    let mut numerator = BigInt::from(mantissa);
    if bits >> 63 != 0 {
        numerator = -numerator;
    }
    let denominator = if exponent < 0 {
        BigInt::one() << (-exponent as usize)
    } else {
        numerator <<= exponent as usize;
        BigInt::one()
    };
    checked(Rational::new(numerator, denominator))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Interval {
    pub lower: Rational,
    pub upper: Rational,
}

impl Interval {
    pub fn new(lower: Rational, upper: Rational) -> Result<Self, ArithmeticError> {
        let interval = Self { lower, upper };
        interval.validate()?;
        Ok(interval)
    }

    pub fn point(value: Rational) -> Result<Self, ArithmeticError> {
        Self::new(value.clone(), value)
    }

    pub fn validate(&self) -> Result<(), ArithmeticError> {
        check_rational(&self.lower)?;
        check_rational(&self.upper)?;
        if self.lower > self.upper {
            return Err(ArithmeticError::InvalidInput(
                "interval endpoints are reversed",
            ));
        }
        Ok(())
    }

    pub fn width(&self) -> Result<Rational, ArithmeticError> {
        self.validate()?;
        checked_sub(&self.upper, &self.lower)
    }

    pub fn contains(&self, value: &Rational) -> Result<bool, ArithmeticError> {
        self.validate()?;
        check_rational(value)?;
        Ok(self.lower <= *value && *value <= self.upper)
    }
}

fn check_precision(precision_bits: u32) -> Result<(), ArithmeticError> {
    if !(1..=MAX_PRECISION_BITS).contains(&precision_bits) {
        return Err(ArithmeticError::InvalidInput(
            "precision must be between 1 and 512 bits",
        ));
    }
    Ok(())
}

fn dyadic_unit(precision_bits: u32) -> Rational {
    Rational::new(BigInt::one(), BigInt::one() << precision_bits as usize)
}

/// Enclose a nonnegative square root with width at most `2^-precision_bits`.
///
/// Integer square root gives `q^2 <= floor(x*2^(2p)) < (q+1)^2`.
/// An exact rational square check determines whether both endpoints coincide.
pub fn sqrt_enclosure(value: &Rational, precision_bits: u32) -> Result<Interval, ArithmeticError> {
    check_rational(value)?;
    check_precision(precision_bits)?;
    if value.is_negative() {
        return Err(ArithmeticError::InvalidInput(
            "square root requires nonnegative input",
        ));
    }
    let scaled_numerator = value.numer() << (2 * precision_bits) as usize;
    let quotient = &scaled_numerator / value.denom();
    let root = quotient.sqrt();
    let exact = &root * &root * value.denom() == scaled_numerator;
    let denominator = BigInt::one() << precision_bits as usize;
    let lower = Rational::new(root.clone(), denominator.clone());
    let upper = if exact {
        lower.clone()
    } else {
        Rational::new(root + 1, denominator)
    };
    let interval = Interval::new(lower, upper)?;
    if checked_mul(&interval.lower, &interval.lower)? > *value
        || checked_mul(&interval.upper, &interval.upper)? < *value
    {
        return Err(ArithmeticError::ImplementationFailure(
            "square-root enclosure invariant failed",
        ));
    }
    Ok(interval)
}

fn dyadic_round(
    value: &Rational,
    precision_bits: u32,
    upward: bool,
) -> Result<Rational, ArithmeticError> {
    check_rational(value)?;
    if precision_bits > 8192 {
        return Err(ArithmeticError::ResourceLimit(
            "internal dyadic precision exceeds 8192 bits",
        ));
    }
    let denominator = BigInt::one() << precision_bits as usize;
    let scaled_numerator = value.numer() * &denominator;
    let quotient = if upward {
        scaled_numerator.div_ceil(value.denom())
    } else {
        scaled_numerator.div_floor(value.denom())
    };
    checked(Rational::new(quotient, denominator))
}

/// Round upward to a multiple of `2^-precision_bits` using exact division.
pub fn round_up_dyadic(value: &Rational, precision_bits: u32) -> Result<Rational, ArithmeticError> {
    check_precision(precision_bits)?;
    dyadic_round(value, precision_bits, true)
}

fn positive_exp(value: &Rational, precision_bits: u32) -> Result<Interval, ArithmeticError> {
    if value.is_zero() {
        return Interval::point(int(1));
    }
    let mut reduced = value.clone();
    let half = rational(1, 2)?;
    let mut squarings = 0_u32;
    while reduced > half {
        reduced = checked_div(&reduced, &int(2))?;
        squarings += 1;
        if squarings > 12 {
            return Err(ArithmeticError::ResourceLimit(
                "exponential range reduction limit",
            ));
        }
    }
    let magnitude_ceiling =
        value
            .numer()
            .div_ceil(value.denom())
            .to_u32()
            .ok_or(ArithmeticError::ResourceLimit(
                "exponential argument magnitude",
            ))?;
    let work_bits = precision_bits + 4 * magnitude_ceiling + 2 * squarings + 32;
    let reduced = Interval::new(
        dyadic_round(&reduced, work_bits, false)?,
        dyadic_round(&reduced, work_bits, true)?,
    )?;
    let mut term = Interval::point(int(1))?;
    let mut sum = term.clone();
    let tail_tolerance = dyadic_unit(work_bits);
    let mut enclosure = None;
    for index in 1..=MAX_TAYLOR_TERMS {
        let divisor = int(i64::from(index));
        term = Interval::new(
            dyadic_round(
                &checked_div(&checked_mul(&term.lower, &reduced.lower)?, &divisor)?,
                work_bits,
                false,
            )?,
            dyadic_round(
                &checked_div(&checked_mul(&term.upper, &reduced.upper)?, &divisor)?,
                work_bits,
                true,
            )?,
        )?;
        sum = Interval::new(
            checked_add(&sum.lower, &term.lower)?,
            checked_add(&sum.upper, &term.upper)?,
        )?;
        let next = checked_div(
            &checked_mul(&term.upper, &reduced.upper)?,
            &int(i64::from(index + 1)),
        )?;
        let ratio = checked_div(&reduced.upper, &int(i64::from(index + 2)))?;
        // Every remaining positive term ratio is at most y/(index+2).
        let tail = checked_div(&next, &checked_sub(&int(1), &ratio)?)?;
        if tail <= tail_tolerance {
            enclosure = Some(Interval::new(
                sum.lower,
                dyadic_round(&checked_add(&sum.upper, &tail)?, work_bits, true)?,
            )?);
            break;
        }
    }
    let mut enclosure = enclosure.ok_or(ArithmeticError::ResourceLimit(
        "exponential Taylor term limit",
    ))?;
    for _ in 0..squarings {
        enclosure = Interval::new(
            dyadic_round(
                &checked_mul(&enclosure.lower, &enclosure.lower)?,
                work_bits,
                false,
            )?,
            dyadic_round(
                &checked_mul(&enclosure.upper, &enclosure.upper)?,
                work_bits,
                true,
            )?,
        )?;
    }
    if enclosure.width()? > dyadic_unit(precision_bits) {
        return Err(ArithmeticError::ResourceLimit(
            "exponential enclosure did not reach requested width",
        ));
    }
    Ok(enclosure)
}

/// Enclose `exp(value)` with absolute width at most `2^-precision_bits`.
///
/// Range reduction gives `0 <= y <= 1/2`. Positive Taylor terms and a
/// geometric remainder enclose `exp(y)`; exact dyadic rounding bounds every
/// quantization. Repeated interval squaring reconstructs the positive
/// exponential, and reciprocal endpoints handle negative inputs. Requests
/// with `abs(value) > 1024` exceed the declared resource budget.
pub fn exp_enclosure(value: &Rational, precision_bits: u32) -> Result<Interval, ArithmeticError> {
    check_rational(value)?;
    check_precision(precision_bits)?;
    let magnitude = value.abs();
    if magnitude > int(i64::from(MAX_EXP_ARGUMENT)) {
        return Err(ArithmeticError::ResourceLimit(
            "exponential argument exceeds 1024",
        ));
    }
    let enclosure = positive_exp(&magnitude, precision_bits)?;
    if value.is_negative() {
        let reciprocal = Interval::new(
            checked_div(&int(1), &enclosure.upper)?,
            checked_div(&int(1), &enclosure.lower)?,
        )?;
        if reciprocal.width()? > dyadic_unit(precision_bits) {
            return Err(ArithmeticError::ResourceLimit(
                "reciprocal enclosure did not reach requested width",
            ));
        }
        Ok(reciprocal)
    } else {
        Ok(enclosure)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rational_text_has_one_spelling() {
        for text in ["0", "1", "-19", "1/2", "-27/8"] {
            assert_eq!(to_string(&parse_rational(text).unwrap()), text);
        }
        for text in [
            "", " ", "1 ", "+1", "-0", "00", "01", "-01", "1.0", "1e2", "1/0", "1/-2", "1/+2",
            "1/02", "2/4", "1/1", "0/2", "1/2/3", "--1", "1/", "/2",
        ] {
            assert!(parse_rational(text).is_err(), "accepted {text:?}");
        }
        assert!(matches!(
            parse_rational(&"1".repeat(MAX_INPUT_BYTES + 1)),
            Err(ArithmeticError::ResourceLimit(_))
        ));
        assert!(parse_rational(&"9".repeat(MAX_INPUT_BYTES)).is_ok());
    }

    #[test]
    fn arithmetic_limits_and_raw_ratios_fail_closed() {
        let largest_power =
            Rational::from_integer(BigInt::one() << (MAX_RATIONAL_BITS - 1) as usize);
        assert!(check_rational(&largest_power).is_ok());
        assert!(matches!(
            checked_mul(&largest_power, &int(2)),
            Err(ArithmeticError::ResourceLimit(_))
        ));
        assert_eq!(checked_sub(&largest_power, &largest_power).unwrap(), int(0));
        assert!(
            checked_div(&int(1), &int(0))
                .unwrap_err()
                .is_invalid_input()
        );
        for raw in [
            Rational::new_raw(BigInt::one(), BigInt::zero()),
            Rational::new_raw(BigInt::one(), BigInt::from(-1)),
            Rational::new_raw(BigInt::from(2), BigInt::from(4)),
        ] {
            assert!(checked_add(&raw, &int(1)).is_err());
        }
        assert!(Interval::new(int(1), int(0)).is_err());
        assert_eq!(rational(1, -2).unwrap(), rational(-1, 2).unwrap());
    }

    #[test]
    fn binary64_conversion_preserves_subnormals_and_signed_zero() {
        assert_eq!(from_f64_exact(0.0).unwrap(), int(0));
        assert_eq!(from_f64_exact(-0.0).unwrap(), int(0));
        assert_eq!(from_f64_exact(0.5).unwrap(), rational(1, 2).unwrap());
        assert_eq!(
            from_f64_exact(0.1).unwrap(),
            rational(3_602_879_701_896_397, 36_028_797_018_963_968).unwrap()
        );
        let smallest = Rational::new(BigInt::one(), BigInt::one() << 1074_usize);
        assert_eq!(from_f64_exact(f64::from_bits(1)).unwrap(), smallest);
        assert_eq!(
            from_f64_exact(f64::from_bits((1_u64 << 63) | 1)).unwrap(),
            -smallest
        );
        let largest = Rational::from_integer(BigInt::from((1_u64 << 53) - 1) << 971_usize);
        assert_eq!(from_f64_exact(f64::MAX).unwrap(), largest);
        assert_eq!(
            from_f64_exact(f64::MIN_POSITIVE).unwrap(),
            Rational::new(BigInt::one(), BigInt::one() << 1022_usize)
        );
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(from_f64_exact(invalid).is_err());
        }
    }

    #[test]
    fn upward_dyadic_rounding_handles_signed_values() {
        for input in [
            rational(1, 3).unwrap(),
            rational(-1, 3).unwrap(),
            int(0),
            int(7),
        ] {
            for precision in [1, 64, MAX_PRECISION_BITS] {
                let rounded = round_up_dyadic(&input, precision).unwrap();
                assert!(rounded >= input);
                assert!(&rounded - &input < dyadic_unit(precision));
            }
        }
        assert_eq!(
            round_up_dyadic(&rational(-1, 3).unwrap(), 1).unwrap(),
            int(0)
        );
        assert_eq!(
            round_up_dyadic(&rational(1, 3).unwrap(), 1).unwrap(),
            rational(1, 2).unwrap()
        );
        assert!(round_up_dyadic(&int(0), 0).is_err());
        assert!(round_up_dyadic(&int(0), MAX_PRECISION_BITS + 1).is_err());
    }

    #[test]
    fn square_root_endpoints_satisfy_exact_square_inequalities() {
        assert_eq!(
            sqrt_enclosure(&int(0), 64).unwrap(),
            Interval::point(int(0)).unwrap()
        );
        assert_eq!(
            sqrt_enclosure(&rational(9, 16).unwrap(), 64).unwrap(),
            Interval::point(rational(3, 4).unwrap()).unwrap()
        );
        for input in [
            int(2),
            rational(1, 3).unwrap(),
            rational(999_983, 997).unwrap(),
        ] {
            for precision in [1, 64, MAX_PRECISION_BITS] {
                let bounds = sqrt_enclosure(&input, precision).unwrap();
                assert!(&bounds.lower * &bounds.lower <= input);
                assert!(&bounds.upper * &bounds.upper >= input);
                assert!(bounds.width().unwrap() <= dyadic_unit(precision));
            }
        }
        assert!(sqrt_enclosure(&int(-1), 64).is_err());
        assert!(sqrt_enclosure(&int(1), 0).is_err());
        assert!(sqrt_enclosure(&int(1), MAX_PRECISION_BITS + 1).is_err());
    }

    // This oracle keeps exact unrounded Taylor terms without range reduction.
    fn independent_positive_exp(input: &Rational) -> Interval {
        let mut sum = int(1);
        let mut term = int(1);
        for index in 1..=180 {
            term = term * input / int(index);
            sum += &term;
        }
        let next = term * input / int(181);
        let ratio = input / int(182);
        assert!(ratio < int(1));
        let upper = &sum + next / (int(1) - ratio);
        Interval::new(sum, upper).unwrap()
    }

    #[test]
    fn exponential_encloses_independent_rational_series() {
        assert_eq!(
            exp_enclosure(&int(0), 128).unwrap(),
            Interval::point(int(1)).unwrap()
        );
        for input in [
            rational(1, 3).unwrap(),
            int(1),
            rational(3, 2).unwrap(),
            rational(9, 2).unwrap(),
        ] {
            let oracle = independent_positive_exp(&input);
            for precision in [1, 64, 128, MAX_PRECISION_BITS] {
                let bounds = exp_enclosure(&input, precision).unwrap();
                assert!(bounds.lower <= oracle.lower);
                assert!(bounds.upper >= oracle.upper);
                assert!(bounds.width().unwrap() <= dyadic_unit(precision));
                let negative = exp_enclosure(&(-&input), precision).unwrap();
                assert!(negative.lower <= int(1) / &oracle.upper);
                assert!(negative.upper >= int(1) / &oracle.lower);
                assert!(negative.width().unwrap() <= dyadic_unit(precision));
            }
        }
    }

    #[test]
    fn exponential_quantization_and_resource_limits_are_explicit() {
        let tiny = Rational::new(BigInt::one(), BigInt::one() << 2048_usize);
        let bounds = exp_enclosure(&tiny, 64).unwrap();
        assert!(bounds.lower <= int(1) + &tiny);
        assert!(bounds.upper > int(1) + &tiny);
        assert!(bounds.width().unwrap() <= dyadic_unit(64));
        assert!(matches!(
            exp_enclosure(&int(1025), 64),
            Err(ArithmeticError::ResourceLimit(_))
        ));
        assert!(exp_enclosure(&int(1), 0).unwrap_err().is_invalid_input());
        let bounds = exp_enclosure(&int(16), 128).unwrap();
        assert!(bounds.lower > int(8_000_000));
        assert!(bounds.upper < int(9_000_000));
        assert!(bounds.width().unwrap() <= dyadic_unit(128));
    }
}

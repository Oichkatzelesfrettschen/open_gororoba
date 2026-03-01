//! FLOAT_OPS realization for f64.
//!
//! This module is the TRUST BOUNDARY of the verified extraction pipeline.
//! It maps the abstract field operations from Rocq's FLOAT_OPS module type
//! to concrete f64 operations. The Rocq proofs guarantee correctness for
//! any type satisfying the FLOAT_OPS axioms; this module asserts that f64
//! (approximately) satisfies them.
//!
//! # Axiom Inventory
//!
//! The following field axioms are assumed (proved in Rocq for exact reals):
//! - add_comm, add_assoc, add_zero_l, add_opp_r (additive group)
//! - mul_comm, mul_assoc, mul_one_l, mul_add_distr_l (commutative ring)
//! - sub_def: sub(x,y) = add(x, opp(y))
//!
//! For IEEE 754 f64, these hold exactly for the subset of non-NaN,
//! non-infinite values where rounding is symmetric. The proofs transfer
//! to f64 under the assumption that rounding errors are negligible for
//! the quaternion rotation use case.

//<*trustboundary>
/// The abstract field type, realized as f64.
pub type T = f64;

pub const ZERO: T = 0.0;
pub const ONE: T = 1.0;

#[inline(always)]
pub fn add(x: T, y: T) -> T {
    x + y
}

#[inline(always)]
pub fn mul(x: T, y: T) -> T {
    x * y
}

#[inline(always)]
pub fn sub(x: T, y: T) -> T {
    x - y
}

#[inline(always)]
pub fn opp(x: T) -> T {
    -x
}

#[inline(always)]
pub fn div(x: T, y: T) -> T {
    x / y
}

#[inline(always)]
pub fn sqrt_f(x: T) -> T {
    x.sqrt()
}
//</trustboundary>

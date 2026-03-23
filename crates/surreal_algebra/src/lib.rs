//! Surreal number coefficients for Cayley-Dickson algebras.
//!
//! # Purpose
//!
//! This crate provides a minimal surreal number type suitable for use as
//! coefficients in Cayley-Dickson algebras.  The scalar extension theorem
//! (C-1504, CDScalarExtension.v) proves that CD linearity requires only
//! ring axioms, so A_n(K) is well-defined for any commutative ring K.
//! Surreal numbers (No) form a real-closed ordered field extending R,
//! making A_n(No) a valid CD algebra.
//!
//! # Design
//!
//! We use a **dyadic surreal** representation: surreal numbers born on
//! day <= N are dyadic rationals (fractions with power-of-2 denominators).
//! This gives exact arithmetic without floating-point error, making it
//! suitable for verifying that zero-divisor identities hold exactly.
//!
//! ```text
//! Day 0: { | } = 0
//! Day 1: { 0 | } = 1,  { | 0 } = -1
//! Day 2: { 0 | 1 } = 1/2,  { 1 | } = 2,  { -1 | 0 } = -1/2,  { | -1 } = -2
//! Day n: all dyadic rationals k/2^n for integer k
//! ```
//!
//! # Relationship to other crates
//!
//! - Uses `cd_kernel` for the sign table (`cd_basis_mul_sign_iter`)
//! - The surreal sedenion is `[SurrealDyadic; 16]` with multiplication
//!   using the same sign table as `[f64; 16]`
//! - Verifies zero-divisor persistence: if a*b = 0 over R, then
//!   (alpha*a)*(beta*b) = 0 over No for any surreal alpha, beta
//!
//! # References
//!
//! - Conway, "On Numbers and Games" (1976)
//! - surreal_cayley_dickson_harmonized.md (project docs)
//! - CDScalarExtension.v (formal proof that ring axioms suffice)

mod dyadic;
pub mod finite_field_cd;
mod surreal_cd;

pub use dyadic::SurrealDyadic;
pub use surreal_cd::{surreal_cd_multiply, surreal_sedenion_zd_check};

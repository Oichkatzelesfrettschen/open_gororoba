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
//! Within the repo's current architecture, this crate is a control and
//! falsification lane. It remains first-class because scalar-extension and
//! persistence checks are valuable regression infrastructure, not because the
//! surreal tower is the primary forward bridge architecture.
//!
//! # Architecture
//!
//! ```text
//! surreal_algebra/
//!   dyadic.rs         -- SurrealDyadic: exact i128/2^shift arithmetic
//!   surreal_cd.rs     -- CD multiply with surreal coefficients (22 tests)
//!   finite_field_cd.rs -- CD multiply over F_p (3 tests)
//!
//! Three coefficient fields:
//!   R (real)       -- cd_kernel (f64, existing)
//!   No (surreal)   -- surreal_cd (SurrealDyadic, exact)
//!   F_p (finite)   -- finite_field_cd (u64 mod p)
//! ```
//!
//! # Key results (claims C-1520..C-1530)
//!
//! | Claim | Result |
//! |-------|--------|
//! | C-1520 | ZD persists over F_p for all primes |
//! | C-1521 | ZD variety stratifies by Archimedean class over No |
//! | C-1523 | Friction is field-independent (integer associator) |
//! | C-1524 | Class ratios sector-dependent (lepton 1.53, quark 1.77) |
//! | C-1525 | Sign profiles give 21 doublets (generation-blind) |
//! | C-1526 | Cross-class coupling at 10% shifts masses by 1% |
//! | C-1528 | Subalgebra membership creates 3 generations |
//! | C-1529 | Intra-generation friction = ZERO |
//! | C-1530 | 3 generations persist at dim=32 (topological) |
//!
//! # Why a separate crate (not in cd_kernel)
//!
//! cd_kernel operates on `f64` coefficients with SIMD/x87 optimizations.
//! Surreal arithmetic uses `i128` exact rationals -- a fundamentally
//! different numerical regime. Keeping them separate prevents:
//! - Contaminating cd_kernel's zero-dependency-on-bigint design
//! - Mixing f64 and exact arithmetic in the same compilation unit
//! - Conflating the precision tiers (x87 oracle vs exact surreal)
//!
//! # Design: dyadic surreal representation
//!
//! ```text
//! Day 0: { | } = 0
//! Day 1: { 0 | } = 1,  { | 0 } = -1
//! Day 2: { 0 | 1 } = 1/2,  { 1 | } = 2,  ...
//! Day n: all dyadic rationals k/2^n for integer k
//! ```
//!
//! [`SurrealDyadic`] stores (numerator: i128, shift: u32) where the value
//! is numerator / 2^shift.  Normalized: trailing factors of 2 removed.
//! The `birthday()` method returns the Archimedean complexity proxy.
//!
//! # Relationship to other crates
//!
//! - Uses [`cd_kernel::cayley_dickson::cd_basis_mul_sign_iter`] for sign lookup
//! - The surreal sedenion is `[SurrealDyadic; 16]` with multiplication
//!   using the same sign table as `[f64; 16]`
//! - Verifies zero-divisor persistence across R, No, and F_p
//! - Complements the Rocq proofs: CDScalarExtension.v (C-1504) and
//!   ArchimedeanStratification.v (C-1527)
//!
//! # References
//!
//! - Conway, "On Numbers and Games" (1976)
//! - surreal_cayley_dickson_harmonized.md (project docs, Section 8)
//! - CDScalarExtension.v (ring axioms suffice)
//! - ArchimedeanStratification.v (ZD requires alpha^2 = beta^2)

mod dyadic;
pub mod finite_field_cd;
mod surreal_cd;

pub use dyadic::SurrealDyadic;
pub use surreal_cd::{surreal_cd_multiply, surreal_sedenion_zd_check};

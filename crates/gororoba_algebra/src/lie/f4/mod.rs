//! `F_4` exceptional Lie algebra (rank 4, dim 52, 48 roots).
//!
//! - [`root_system`] -- 24 positive roots, 4 simple roots (2 long + 2 short),
//!   Cartan matrix with the double bond, Weyl vector, and Weyl group order
//!   1152.
//! - [`casimir`] -- Quadratic Casimir of the 26-dim fundamental representation.
//!   Verifies Theorem 12.6 of the project monograph: under physics
//!   normalization (long-root squared length 1),
//!   `epsilon = C_2(26)/|Delta+(F_4)| = 1/4` exactly.
//!
//! # Consolidation note (2026-05-08)
//!
//! Promoted from `algebra_experimental::f4_casimir` (cross-crate move). `F_4`
//! is pure Lie-algebra mathematics with no experimental scaffolding, and
//! belongs alongside the other exceptional algebras under [`crate::lie`].
//! Subsequently split (2026-05-08) into a `root_system + casimir` pair to
//! match the structure of [`super::e6`].

pub mod casimir;
pub mod root_system;

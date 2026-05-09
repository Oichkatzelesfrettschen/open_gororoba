//! `E_6` exceptional Lie algebra (rank 6, dim 78, 72 roots).
//!
//! - [`root_system`] -- 72 E6 roots, 6 simple roots, Cartan matrix, Weyl
//!   group order. Realized as the rank-6 sub-root-system of `E_8` orthogonal
//!   to a chosen pair of E8 simple roots, preserving the branch-at-node-4
//!   numbering used by [`crate::lie::e8::root_system`].
//! - [`casimir`] -- Casimir invariant of the 27-dim fundamental representation,
//!   verifying the textbook ratio against the 36 positive roots.
//!
//! # Magic-square placement
//!
//! `E_6 = L(C, O)` (complex tensor octonion) in the Freudenthal-Tits magic
//! square. Its 27-dim fundamental representation is the (complexified)
//! exceptional Jordan algebra `J_3(O)`. See
//! [`crate::lie::e8::magic_square::FreudenthalTitsMagicSquare`].
//!
//! # Consolidation note (2026-05-08)
//!
//! New module: built from scratch for completeness. Previously the repo had
//! no standalone E6 code (only `MagicSquareLieAlgebra::E6` enum entries and
//! references in `algebra_experimental::cd_external` audits).

pub mod casimir;
pub mod root_system;

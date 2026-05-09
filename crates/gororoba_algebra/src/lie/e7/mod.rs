//! `E_7` exceptional Lie algebra (rank 7, dim 133, 126 roots).
//!
//! - [`geometry`] -- `E_7` root vectors as the rank-7 subset of `E_8` roots
//!   orthogonal to the affine extension; triad enumeration; planar projection.
//! - [`structure`] -- structure constants `N(k, p)` for `E_7` and the
//!   Chevalley-Tits extraspecial cocycle used by [`super::lyndon_basis`].
//!
//! # Consolidation note (2026-05-08)
//!
//! Promoted from sibling files `e7_geometry.rs` and `e7_structure.rs` directly
//! under `lie/`. Hard-renamed; no compatibility shim.

pub mod geometry;
pub mod structure;

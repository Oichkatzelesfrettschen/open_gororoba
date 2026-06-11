//! E8 exceptional Lie algebra: lattice, root system, magic square, atlas bridge,
//! heterotic-string facade.
//!
//! # Layout
//!
//! - [`root_system`] -- `E8Root`, `E8Lattice`, root enumeration, simple roots,
//!   Cartan matrix, Weyl-group order, fundamental weights, theta function,
//!   sphere-packing density.
//! - [`magic_square`] -- `DivisionAlgebra`, `MagicSquareLieAlgebra`,
//!   `FreudenthalTitsMagicSquare`. The 4x4 R/C/H/O lookup whose octonionic
//!   row produces F4, E6, E7, E8.
//! - [`atlas_bridge`] -- cross-validation against the external `atlas-embeddings`
//!   crate (96-vertex Atlas of Resonance Classes -> E8).
//! - [`heterotic`] -- E8xE8 vs SO(32) heterotic-string anomaly-cancellation
//!   facade and the canonical SU(3)xE6 standard-model embedding.
//!
pub mod atlas_bridge;
pub mod heterotic;
pub mod magic_square;
pub mod root_system;

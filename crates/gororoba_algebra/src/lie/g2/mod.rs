//! `G_2` exceptional Lie group.
//!
//! - [`stabilizer`] -- complex-structure stabilizer machinery (`G_2 = Aut(O)`
//!   acting on `Im(O) = R^7` and on the 6D space `H^perp` orthogonal to a
//!   chosen quaternion basis), including structure constants and
//!   stabilizer-decomposition helpers.
//! - [`su3_representation`] -- `G_2 -> SU(3) -> SU(3)/U(1)` chain, the
//!   embedding into 7-dimensional and 6-dimensional representations used
//!   downstream by the neutrino-sector and Fano-line analyses.
//!
//! # Consolidation note (2026-05-08)
//!
//! These two files lived directly under `lie/` as `g2_stabilizer.rs` and
//! `g2_su3_representation.rs`. Promoted into a sibling subdirectory because
//! together they exceed 1800 lines and form a coherent G2-specific cluster.
//! All external imports were hard-renamed; no compatibility shim.

pub mod stabilizer;
pub mod su3_representation;

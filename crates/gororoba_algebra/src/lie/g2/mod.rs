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
pub mod stabilizer;
pub mod su3_representation;

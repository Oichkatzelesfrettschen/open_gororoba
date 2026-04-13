//! NANOGrav pulsar timing data structures and analysis engines.
//!
//! - `timing`: release loader, residuals, DM/DMX parsers, sky vectors.
//! - `timing_model`: structured `.par` file parser (astrometry, dispersion, noise).
//! - `refit`: phase-1 wideband refit solver (nalgebra least-squares).
//! - `engine`: independent timing engine (faer/anise barycentric correction).
//!
//! The `engine` submodule requires the `nanograv-engine` feature because it
//! depends on `faer` (dense linear algebra) and `anise` (NAIF/SPICE ephemeris).

pub mod refit;
pub mod timing;
pub mod timing_model;

#[cfg(feature = "nanograv-engine")]
pub mod engine;

//! Cayley-Dickson algebra operations.
//!
//! This module re-exports the `cd_kernel` crate, which contains the
//! pure arithmetic kernel: recursive multiplication, conjugation, norm,
//! zero-divisor search, associator computation, and SIMD variants.
//!
//! All public items are re-exported verbatim so that existing import
//! paths (`gororoba_algebra::construction::cayley_dickson::*`) remain valid.

pub use cd_kernel::cayley_dickson::*;

// Re-export tests: cd_kernel carries its own #[cfg(test)] block.
// gororoba_algebra's integration tests exercise the re-export paths.

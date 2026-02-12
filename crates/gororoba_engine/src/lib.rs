//! Gororoba Engine: The Mathematical Universe
//!
//! The 6-layer trait architecture: Bit -> Parity -> Topology -> Dynamics -> Correction -> Verification
//! transforming bit-level Cayley-Dickson algebra into testable physics.

pub mod adaptive_gpu;
pub mod bit_source;
pub mod correction_layer;
pub mod dynamics_field;
pub mod parity_filter;
pub mod pipeline;
pub mod topology_geometry;
pub mod traits;
pub mod verification_layer;

pub use adaptive_gpu::{choose_backend, ComputeBackend};
pub use pipeline::GororobaEngine;
pub use traits::{PipelineState, VerificationReport};

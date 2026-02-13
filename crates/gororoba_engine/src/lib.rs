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
pub mod thesis_pipelines;
pub mod topology_geometry;
pub mod traits;
pub mod verification_layer;

pub use adaptive_gpu::{choose_backend, ComputeBackend};
pub use pipeline::GororobaEngine;
pub use thesis_pipelines::{
    Thesis1Pipeline, Thesis2Pipeline, Thesis3Pipeline, Thesis4Pipeline,
};
pub use traits::{PipelineState, ThesisEvidence, ThesisPipeline, VerificationReport};

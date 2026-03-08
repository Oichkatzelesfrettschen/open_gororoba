//! Gororoba Engine: The Mathematical Universe
//!
//! The 6-layer trait architecture: Bit -> Parity -> Topology -> Dynamics -> Correction -> Verification
//! transforming bit-level Cayley-Dickson algebra into testable physics.
//!
//! Pipeline traits and pure-logic layers are defined in `gororoba_pipeline`.
//! GPU backend selection lives in `gororoba_gpu_bridge`.
//! This crate provides domain-specific layer implementations, simulation state,
//! and thesis pipeline orchestration, re-exporting the sub-crate APIs for
//! backward compatibility.

// Domain-specific layer implementations (depend on physics crates)
pub mod correction_layer;
pub mod dynamics_field;
pub mod fluid_dynamics_bridge;
pub mod gate;
pub mod pipeline;
pub mod pipelines;
pub mod provenance;
pub mod simulation;
pub mod singularitarian;
pub mod thesis_pipelines;
pub mod verification_layer;

// Re-export pipeline infrastructure from gororoba_pipeline
pub use gororoba_pipeline::{
    AntiDiagonalParityFilter, BitSourceLayer, CorrectionLayer, DynamicsLayer, FibonacciBitSource,
    ParityLayer, PipelineState, SlidingTriadTopology, ThesisEvidence, ThesisPipeline,
    TopologyLayer, VerificationLayer, VerificationReport,
};

// Re-export pipeline sub-modules for crate::traits paths used by local modules
pub mod traits {
    //! Re-export of pipeline traits for internal use.
    pub use gororoba_pipeline::*;
}

pub mod bit_source {
    //! Re-export of bit source layer.
    pub use gororoba_pipeline::bit_source::*;
}

pub mod parity_filter {
    //! Re-export of parity filter layer.
    pub use gororoba_pipeline::parity_filter::*;
}

pub mod topology_geometry {
    //! Re-export of topology layer.
    pub use gororoba_pipeline::topology_geometry::*;
}

// Re-export GPU bridge from gororoba_gpu_bridge
pub mod adaptive_gpu {
    //! Re-export of GPU backend selection.
    pub use gororoba_gpu_bridge::*;
}

pub use gororoba_gpu_bridge::{
    ComputeBackend, HardwareCaps, SimdCaps, choose_backend, detect_best_backend, probe_simd,
};
pub use pipeline::GororobaEngine;
pub use pipelines::warp_ring::WarpRingPipeline;
pub use simulation::{SimulationConfig, SimulationConfig3D, SimulationState, SimulationState3D};
pub use thesis_pipelines::{Thesis1Pipeline, Thesis2Pipeline, Thesis3Pipeline, Thesis4Pipeline};

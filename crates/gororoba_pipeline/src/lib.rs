//! Pipeline trait definitions and pure-logic layer implementations.
//!
//! This crate defines the 6-layer engine pipeline architecture:
//!
//!   Bit -> Parity -> Topology -> Dynamics -> Correction -> Verification
//!
//! It provides the trait definitions that all layers implement and includes
//! the pure-logic implementations (no domain-physics dependencies) for the
//! first three layers.  Domain-specific layer implementations (dynamics,
//! correction, verification) live in `gororoba_engine` where they can
//! import physics crates.

pub mod traits;

pub mod bit_source;
pub mod parity_filter;
pub mod topology_geometry;

pub use traits::{
    BitSourceLayer, CorrectionLayer, DynamicsLayer, ParityLayer, PipelineState, ThesisEvidence,
    ThesisPipeline, TopologyLayer, VerificationLayer, VerificationReport,
};

pub use bit_source::FibonacciBitSource;
pub use parity_filter::AntiDiagonalParityFilter;
pub use topology_geometry::SlidingTriadTopology;

//! Data, registry, and governance CLI binaries for gororoba.
//!
//! This crate contains 40 binaries covering claims management, data governance,
//! dataset fetching, registry operations, publishing, and quality gates.

pub mod acquisition_dossier;
pub mod nanograv_refit;
pub mod nanograv_timing;
pub mod nanograv_timing_engine;
pub mod nanograv_timing_model;
pub mod project_api_contract;

pub use provenance_ops::source_provenance;

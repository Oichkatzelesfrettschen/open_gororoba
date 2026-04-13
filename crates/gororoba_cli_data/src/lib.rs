//! Data, registry, and governance CLI binaries for gororoba.
//!
//! This crate contains 40 binaries covering claims management, data governance,
//! dataset fetching, registry operations, publishing, and quality gates.

pub mod acquisition_dossier;
pub mod project_api_contract;

// Re-export nanograv modules from data_core for backward compatibility.
pub use data_core::nanograv::refit as nanograv_refit;
pub use data_core::nanograv::timing as nanograv_timing;
pub use data_core::nanograv::engine as nanograv_timing_engine;
pub use data_core::nanograv::timing_model as nanograv_timing_model;

pub use provenance_ops::source_provenance;

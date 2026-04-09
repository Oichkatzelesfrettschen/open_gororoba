//! Artifact-only helpers extracted from `data_core`.
//!
//! This crate intentionally excludes dataset providers, download stacks, and
//! HTTP transfer code so analysis/test crates can depend on artifact helpers
//! without inheriting the full network surface.

pub mod quality;

#[cfg(feature = "hdf5-export")]
pub mod hdf5_export;

pub use quality::{
    RhoQualityError, RhoQualityThresholds, RhoTraceQuality, ScalarTraceError, ScalarTraceQuality,
    ScalarTraceThresholds, assess_rho_trace, assess_scalar_trace, validate_rho_trace,
    validate_scalar_trace_signal,
};

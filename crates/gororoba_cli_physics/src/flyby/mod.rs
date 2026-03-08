//! Flyby anomaly analysis pipeline.
//!
//! Provides reusable library modules for spacecraft flyby trajectory integration,
//! environmental models (NFW dark matter, gravitational focusing wake, heliospheric
//! tensor), and residual analysis. Binaries in `src/bin/` are thin CLI wrappers.

pub mod config;
pub mod environment;
pub mod geometry;
pub mod integrator;
pub mod report;

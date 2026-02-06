//! Bridges between control_core and physics domain crates.
//!
//! This module provides `Plant` trait implementations for various physics
//! simulation crates, enabling feedback control of physical systems.
//!
//! # Feature Flags
//!
//! - `optics`: Enables TCMT cavity control via `optics_core`
//! - `casimir`: Enables Casimir mechanical DOF control via `quantum_core`
//! - `all-bridges`: Enables all physics bridges

#[cfg(feature = "optics")]
pub mod tcmt;

#[cfg(feature = "casimir")]
pub mod casimir_mems;

#[cfg(feature = "optics")]
pub use tcmt::*;

#[cfg(feature = "casimir")]
pub use casimir_mems::*;

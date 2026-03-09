//! Cayley-Dickson Spin Bridge
//!
//! Connects Cayley-Dickson algebraic observables (imbalance density,
//! associator norms) to quantum channel parameters for two-qubit spin
//! tomography. Bridges three crates:
//!
//! - `sign_imbalance`: provides the 3/8 imbalance attractor and imbalance metrics
//! - `spin_tomography_core`: provides `TwoQubitState` density matrix representation
//! - `gororoba_algebra`: provides Pauli matrices and Clifford algebra primitives
//!
//! The bridge models how Cayley-Dickson algebraic structure induces
//! decoherence in spin-correlated particle pairs, with applications
//! to QGP Lambda-AntiLambda polarization measurements.

pub mod decoherence_map;
pub mod depolarizing_channel;
pub mod qgp_model;
pub mod star_channel;

pub use decoherence_map::DecoherenceMap;
pub use depolarizing_channel::apply_depolarizing_channel;
pub use qgp_model::{QGPImbalanceBridge, QGPState};
pub use star_channel::StarChannel;

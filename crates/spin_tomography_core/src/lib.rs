//! Spin Tomography Core
//!
//! Implements two-qubit spin density matrix tomography for Lambda-AntiLambda
//! pairs in heavy-ion collisions. Provides:
//!
//! - `TwoQubitState`: 4x4 complex density matrix with Bloch decomposition,
//!   partial transpose, and negativity (PPT entanglement criterion)
//! - `TomographyMoments`: weighted moment accumulator for spin analyzer events
//! - `SpinEvent`: single analyzer measurement with directions and analyzing powers
//! - `AlgebraicTriad`: right-handed basis mapping CD indices to Pauli matrices
//! - `albert_bridge`: embedding into the Albert algebra J_3(O) for spectral analysis
//! - `event_gen`: synthetic event generator (rejection sampling on S^2 x S^2)
//! - `ndjson`: NDJSON event reader for cross-language validation

pub mod albert_bridge;
pub mod algebraic_triad;
pub mod event_gen;
pub mod ndjson;
pub mod spin_event;
pub mod state;
pub mod tomography;

#[cfg(test)]
mod albert_tests;

pub use albert_bridge::embed_to_albert;
pub use algebraic_triad::AlgebraicTriad;
pub use event_gen::{estimate_t_direct, generate_events, joint_pdf, sample_uniform_s2};
pub use ndjson::read_events_ndjson;
pub use spin_event::SpinEvent;
pub use state::TwoQubitState;
pub use tomography::TomographyMoments;

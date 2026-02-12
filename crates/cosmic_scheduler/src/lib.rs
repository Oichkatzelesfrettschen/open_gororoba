//! Cosmic Scheduler: Generic two-phase clock orchestration
//!
//! Provides a pure Rust abstraction for systems using two-phase timing coordination,
//! inspired by the Intel 4004's phi1/phi2 clock.
//!
//! The key insight: many computational and physical systems naturally decompose into
//! two phases - a preparation phase and an execution phase. By extracting this
//! abstraction, we can coordinate diverse systems (LBM fluid dynamics, quantum evolution,
//! etc.) with unified timing semantics.
//!
//! # Two-Phase Clock Model
//!
//! - **Phi1**: Preparation, collision, compute, precharge
//! - **Phi2**: Execution, streaming, evaluation, transfer
//!
//! Both phases must execute in order, with well-defined timing constraints from the
//! Intel 4004 datasheet adapted for generic use.
//!
//! # Example: LBM Lattice Boltzmann Method
//!
//! ```text
//! Phi1 (Collision Phase)
//!   - Load particle populations f_i
//!   - Compute macroscopic variables (rho, u)
//!   - Compute equilibrium distribution f_i^eq
//!   - Apply collision operator: f_i' = f_i - (f_i - f_i^eq) / tau
//!
//! Phi2 (Streaming Phase)
//!   - Redistribute populations to neighbor nodes
//!   - Enforce boundary conditions
//!   - Advance lattice state
//! ```
//!
//! # Cosmic Engine Integration
//!
//! This scheduler enables **CE-002**: "Two-phase clock (phi1/phi2) isomorphic to
//! LBM collision/streaming split", unifying multiple physical and computational domains.

pub mod phase_scheduler;
pub mod timing_constants;

pub use phase_scheduler::{
    Phase, ScheduleError, ScheduleResult, TwoPhaseClockScheduler, TwoPhaseSystem,
};
pub use timing_constants::{
    clock_spec, format_time, gate_delay, Time, MICROSECOND, MILLISECOND, NANOSECOND, PICOSECOND,
};

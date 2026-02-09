//! control_core: Control systems abstractions for physical plant modeling.
//!
//! This crate provides:
//! - `Plant` trait for physical systems with state, input, and output
//! - `Controller` trait for feedback control algorithms
//! - `StateEstimator` trait for Kalman-style filtering
//! - PID controller implementation
//! - Linear state-space systems
//!
//! # Design Philosophy
//!
//! Control systems bridge the gap between physics simulation and real-world
//! actuation. The key abstractions are:
//!
//! 1. **Plant**: A physical system with state x, input u, output y
//!    - `step(u) -> y`: Advance state by dt given input
//!    - `state() -> x`: Current internal state
//!    - `output() -> y`: Observable output
//!
//! 2. **Controller**: Computes control input from error/state
//!    - `control(reference, measurement) -> u`
//!
//! 3. **StateEstimator**: Estimates hidden state from noisy measurements
//!    - `predict(u)`: Propagate state estimate forward
//!    - `update(y)`: Incorporate measurement
//!
//! # Literature
//! - Astrom & Murray (2008): Feedback Systems
//! - Ogata (2010): Modern Control Engineering
//! - Kalman (1960): A New Approach to Linear Filtering

pub mod feedback;
pub mod filtering;
pub mod pid;
pub mod plant;
pub mod state_space;

// Physics domain bridges (feature-gated)
#[cfg(any(feature = "optics", feature = "casimir"))]
pub mod bridges;

pub use plant::{ContinuousPlant, DiscretePlant, LinearPlant, Plant, PlantDynamics};

pub use feedback::{ControlError, ControlOutput, Controller, FeedbackLoop, ReferenceSignal};

pub use filtering::{
    ExtendedKalmanFilter, FilterState, KalmanFilter, PredictionResult, StateEstimator, UpdateResult,
};

pub use pid::{anti_windup_clamp, derivative_filter, PidController, PidGains, PidState};

pub use state_space::{
    controllability_matrix, is_controllable, is_observable, observability_matrix, StateSpaceModel,
    TransferFunction,
};

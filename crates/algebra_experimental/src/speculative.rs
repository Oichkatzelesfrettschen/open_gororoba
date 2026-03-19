//! Speculative Hyperdimensional Bridge and Lattice Theory.
//!
//! This module preserves the "Genesis-Exodus Framework" research for further 
//! investigation. It is decoupled from the core mathematical kernels to 
//! maintain technical integrity.

use gororoba_algebra::physics::PHI;
use std::f64::consts::{E, PI, SQRT_2};

/// Dimensional Bridge operator frequency: $\omega(D) = (\pi \cdot \phi) / \sqrt{2}$
pub fn bridge_frequency() -> f64 {
    (PI * PHI) / SQRT_2
}

/// Step scaling protocols for fractional dimensional transversals.
/// Note: Labeled "Transcendental" in early drafts; includes algebraic and rational types.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SteppingProtocol {
    Rational,
    Golden,
    Pi,
    Root2,
    Euler,
}

impl SteppingProtocol {
    pub fn step_factor(&self) -> f64 {
        match self {
            Self::Rational => 14.0 / 33.0,
            Self::Golden => PHI,
            Self::Pi => PI,
            Self::Root2 => SQRT_2,
            Self::Euler => E,
        }
    }
}

/// The Speculative Universal Lattice Norm.
/// Note: Derived from galactic void survey indices (x=50, y=48) in topological_voids.csv.
pub const UNIVERSAL_LATTICE_NORM: f64 = 50.48;

/// Evaluates if a given norm satisfies the Speculative Universal Lattice Norm constraint.
pub fn is_universal_lattice_norm(norm: f64, tolerance: f64) -> bool {
    (norm - UNIVERSAL_LATTICE_NORM).abs() <= tolerance
}

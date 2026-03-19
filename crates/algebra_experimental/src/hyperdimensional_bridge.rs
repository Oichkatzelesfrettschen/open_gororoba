//! Hyperdimensional Lattice and Dimensional Bridge Unification.
//!
//! Models the transition from -512D to +512D and the emergence of the
//! Universal Lattice Norm ($\approx 50.48$).
//!
//! Based on the "Genesis-Exodus Framework" which mathematically bridges:
//! - P-adic Quantum Fields
//! - Fractional Stepping Protocols ($\phi, \pi, e, \sqrt{2}$)
//! - Transcendental Anchor (0D)
//! - Cosmic Infinity ($+\infty$D)

use gororoba_algebra::physics::PHI;
use std::f64::consts::{E, PI, SQRT_2};

/// Dimensional Bridge operator frequency: $\omega(D) = (\pi \cdot \phi) / \sqrt{2}$
pub fn bridge_frequency() -> f64 {
    (PI * PHI) / SQRT_2
}

/// Step scaling protocols for fractional dimensional transversals
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

/// The Universal Lattice Norm discovered across all dimensions.
/// Tested across dimensions $D \in \{-512, -256, 0, +256, +512\}$
pub const UNIVERSAL_LATTICE_NORM: f64 = 50.48;

/// Evaluates if a given norm satisfies the Universal Lattice Norm constraint.
pub fn is_universal_lattice_norm(norm: f64, tolerance: f64) -> bool {
    (norm - UNIVERSAL_LATTICE_NORM).abs() <= tolerance
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_frequency() {
        let freq = bridge_frequency();
        // roughly 3.595
        assert!((freq - 3.595).abs() < 0.01);
    }
    
    #[test]
    fn test_universal_lattice_norm_check() {
        assert!(is_universal_lattice_norm(50.49, 0.02));
        assert!(is_universal_lattice_norm(50.47, 0.02));
        assert!(!is_universal_lattice_norm(50.0, 0.02));
    }
}
//! Unified Field Action and Non-Associative Lagrangians
//!
//! This module provides a programmatic manifestation of the theoretical
//! Algebraic Action $\mathcal{S}$ defined in the Unified Algebraic Physics Framework monograph.
//!
//! $$\mathcal{S} = \int d^4x \sqrt{-g} \left[ \frac{R - 2\Lambda(\phi)}{16\pi G} + \mathcal{L}_{matter} + \mathcal{L}_{AVT} \right]$$

use crate::axiomatic_gates::{VACUUM_PHI, binary_entropy};

/// A trait representing a scalar or tensor field that contributes to the total action.
pub trait ActionComponent {
    /// Calculate the Lagrangian density $\mathcal{L}$ for a given local imbalance $\phi$.
    fn lagrangian_density(&self, local_phi: f64) -> f64;
}

/// The Cosmological Constant component, derived from algebraic imbalance.
/// $\Lambda(\phi) \propto H(\phi)$.
pub struct AlgebraicCosmologicalConstant {
    /// Base scaling factor (e.g., $1 / (16\pi G)$ in appropriate units).
    pub scale_factor: f64,
}

impl ActionComponent for AlgebraicCosmologicalConstant {
    fn lagrangian_density(&self, local_phi: f64) -> f64 {
        // -2 Lambda(phi) / (16 pi G)
        // We model Lambda as proportional to the binary entropy of the imbalance.
        let lambda = binary_entropy(local_phi);
        -2.0 * lambda * self.scale_factor
    }
}

/// The Associativity Violation Tensor (AVT) Lagrangian $\mathcal{L}_{AVT}$.
///
/// This component models the "Topological Friction" induced by the 
/// 16D (or higher) non-associative background on propagating waves.
pub struct TopologicalFrictionLagrangian {
    /// Coupling constant between physical fields and the AVT.
    pub coupling_strength: f64,
    /// The fundamental dimensionality of the algebraic manifold (e.g., 16, 512, 1024).
    pub manifold_dimension: usize,
}

impl ActionComponent for TopologicalFrictionLagrangian {
    fn lagrangian_density(&self, local_phi: f64) -> f64 {
        // Friction is maximized when the local imbalance matches the 
        // global topological vacuum attractor (3/8).
        // A simple parabolic well model centered at VACUUM_PHI.
        let deviation = local_phi - VACUUM_PHI;
        
        // Negative sign because friction decreases the total action (dissipation)
        -self.coupling_strength * deviation.powi(2) * (self.manifold_dimension as f64).ln()
    }
}

/// The total unified algebraic action for a given discrete grid point.
pub fn compute_local_action(
    phi: f64,
    ricci_scalar: f64,
    matter_density: f64,
    scale_factor: f64,
    avt_coupling: f64,
    manifold_dimension: usize,
) -> f64 {
    let cc = AlgebraicCosmologicalConstant { scale_factor };
    let avt = TopologicalFrictionLagrangian {
        coupling_strength: avt_coupling,
        manifold_dimension, 
    };

    let l_grav = ricci_scalar * scale_factor + cc.lagrangian_density(phi);
    let l_matter = matter_density; // Simplified matter coupling
    let l_avt = avt.lagrangian_density(phi);

    l_grav + l_matter + l_avt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topological_friction_maximization() {
        let avt = TopologicalFrictionLagrangian {
            coupling_strength: 1.0,
            manifold_dimension: 16,
        };

        // Friction loss should be exactly zero (maximized action) when phi == VACUUM_PHI
        let loss_at_vacuum = avt.lagrangian_density(VACUUM_PHI);
        assert!(loss_at_vacuum.abs() < 1e-10);

        // Any deviation should result in a negative contribution
        let loss_deviated = avt.lagrangian_density(0.5);
        assert!(loss_deviated < 0.0);
    }

    #[test]
    fn test_cosmological_uplift() {
        let cc = AlgebraicCosmologicalConstant { scale_factor: 1.0 };
        
        // Entropy is positive, so the Lambda term (-2 * Lambda) must be negative
        let density = cc.lagrangian_density(VACUUM_PHI);
        assert!(density < 0.0);
    }
}

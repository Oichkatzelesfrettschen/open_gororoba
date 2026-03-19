//! Hydrodynamic medium response and wake formation.
//!
//! Models the energy-momentum deposition of a hard parton traversing
//! the Quark-Gluon Plasma, solving for the source term $J^\nu$ in the 
//! hydrodynamic conservation equations: $\partial_\mu T^{\mu\nu} = J^\nu$.

/// Energy-momentum tensor $T^{\mu\nu}$
#[derive(Debug, Clone, Copy)]
pub struct EnergyMomentumTensor {
    pub t00: f64, // Energy density
    pub t0i: [f64; 3], // Momentum density
    pub tij: [[f64; 3]; 3], // Stress tensor
}

/// The external source term $J^\nu$ encoding energy-momentum deposition
/// along the parton trajectory.
#[derive(Debug, Clone, Copy)]
pub struct WakeSourceTerm {
    pub energy_deposition: f64,   // J^0
    pub momentum_deposition: [f64; 3], // J^i
}

impl WakeSourceTerm {
    /// Creates a localized Gaussian deposition profile (simplified discrete map).
    pub fn new_gaussian(
        amplitude: f64,
        direction: [f64; 3],
        velocity: f64,
    ) -> Self {
        // Normalizing direction
        let norm = (direction[0].powi(2) + direction[1].powi(2) + direction[2].powi(2)).sqrt();
        let nx = direction[0] / norm;
        let ny = direction[1] / norm;
        let nz = direction[2] / norm;

        Self {
            energy_deposition: amplitude,
            momentum_deposition: [
                amplitude * velocity * nx,
                amplitude * velocity * ny,
                amplitude * velocity * nz,
            ],
        }
    }
}

/// Calculates the first-order response perturbation to $T^{\mu\nu}$ given the wake source $J^\nu$
pub fn compute_linear_wake_response(
    background_t00: f64,
    source: &WakeSourceTerm,
    dt: f64,
) -> EnergyMomentumTensor {
    // Highly schematic linear response for validation purposes.
    // Represents $\delta T^{00} \approx J^0 \cdot \delta t$
    let delta_e = source.energy_deposition * dt;
    let mut t = EnergyMomentumTensor {
        t00: background_t00 + delta_e,
        t0i: [0.0, 0.0, 0.0],
        tij: [[0.0; 3]; 3],
    };
    
    for i in 0..3 {
        t.t0i[i] = source.momentum_deposition[i] * dt;
    }
    
    t
}

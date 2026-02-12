//! Dynamics layer: frustration -> viscosity mapping.

use crate::traits::DynamicsLayer;
use vacuum_frustration::FrustrationViscosityBridge;

/// Runtime parameters for the dynamics bridge.
#[derive(Debug, Clone)]
pub struct VacuumDynamicsLayer {
    bridge: FrustrationViscosityBridge,
    pub nu_base: f64,
    pub lambda: f64,
}

impl Default for VacuumDynamicsLayer {
    fn default() -> Self {
        Self {
            bridge: FrustrationViscosityBridge::new(16),
            nu_base: 1.0 / 3.0,
            lambda: 1.0,
        }
    }
}

impl DynamicsLayer for VacuumDynamicsLayer {
    fn viscosity_field(&self, frustration: &[f64]) -> Vec<f64> {
        self.bridge
            .frustration_to_viscosity(frustration, self.nu_base, self.lambda)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::DynamicsLayer;

    #[test]
    fn test_dynamics_viscosity_positive() {
        let dyns = VacuumDynamicsLayer::default();
        let v = dyns.viscosity_field(&[0.2, 0.375, 0.8]);
        assert_eq!(v.len(), 3);
        assert!(v.iter().all(|x| *x > 0.0));
    }
}

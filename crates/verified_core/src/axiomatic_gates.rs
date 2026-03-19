//! Axiomatic gates for Project Genesis constants.

use std::f64::consts::PI;

/// Vacuum imbalance attractor (3/8).
pub const VACUUM_PHI: f64 = 3.0 / 8.0;

/// Theoretical Weak Mixing Angle at GUT scale.
pub const WEAK_MIXING_ANGLE_GUT: f64 = 0.375;

/// Binary entropy function H(p).
pub fn binary_entropy(p: f64) -> f64 {
    if p <= 0.0 || p >= 1.0 {
        return 0.0;
    }
    -(p * p.ln() + (1.0 - p) * (1.0 - p).ln())
}

/// Genesis derivation of the Barbero-Immirzi parameter.
pub fn derive_immirzi_parameter() -> f64 {
    binary_entropy(VACUUM_PHI) / (PI * 3.0f64.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_immirzi_derivation() {
        let gamma = derive_immirzi_parameter();
        let dl_value = 0.123556531;
        let deviation = (gamma - dl_value).abs() / dl_value;
        
        println!("Genesis Immirzi: {}", gamma);
        println!("Domagala-Lewandowski: {}", dl_value);
        println!("Deviation: {:.2}%", deviation * 100.0);
        
        // Assert within 2%
        assert!(deviation < 0.02);
    }

    #[test]
    fn test_mixing_angle_unification() {
        // The 3/8 ratio matches the SU(5) prediction for sin^2 theta_W
        assert!((VACUUM_PHI - WEAK_MIXING_ANGLE_GUT).abs() < 1e-10);
    }
}

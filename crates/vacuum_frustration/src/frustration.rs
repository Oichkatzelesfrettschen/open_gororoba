//! Scalar-frustration bridge utilities.
//!
//! This module provides a falsifiable mapping from signed-graph frustration to
//! scalar-tensor observables:
//! - Local scalar field map: phi(x) = exp(-lambda * F(x))
//! - Effective Brans-Dicke parameter estimate omega_eff(phi)
//! - Optional gravastar/TOV solve to probe "frustration star" candidates

use crate::balance::compute_frustration_index;
use cosmology_core::gravastar::{GravastarSolution, PolytropicEos};
use cosmology_core::homotopy_bridge::solve_gravastar_homotopy;

/// Cassini lower bound used as thesis falsifier.
pub const CASSINI_OMEGA_BD_LOWER_BOUND: f64 = 40_000.0;

/// Scalar map phi = exp(-lambda * F).
#[derive(Clone, Copy, Debug)]
pub struct ScalarFrustrationMap {
    /// Positive coupling constant in the exponential map.
    pub lambda: f64,
}

impl ScalarFrustrationMap {
    /// Construct a new scalar map.
    pub fn new(lambda: f64) -> Self {
        assert!(
            lambda.is_finite() && lambda >= 0.0,
            "lambda must be finite and non-negative"
        );
        Self { lambda }
    }

    /// Map frustration density F in [0,1] to scalar field phi in (0,1].
    pub fn phi(self, frustration: f64) -> f64 {
        let f = frustration.clamp(0.0, 1.0);
        (-self.lambda * f).exp()
    }

    /// Map a full frustration field to scalar field values.
    pub fn phi_field(self, frustration_field: &[f64]) -> Vec<f64> {
        frustration_field.iter().map(|&f| self.phi(f)).collect()
    }

    /// Mean scalar value for a frustration field.
    pub fn mean_phi(self, frustration_field: &[f64]) -> f64 {
        if frustration_field.is_empty() {
            return 1.0;
        }
        let phi = self.phi_field(frustration_field);
        phi.iter().sum::<f64>() / (phi.len() as f64)
    }
}

/// Compute frustration density directly from signed edges.
pub fn frustration_density_from_edges(edges: &[(usize, usize, i32)], num_nodes: usize) -> f64 {
    compute_frustration_index(edges, num_nodes).frustration_density
}

/// Map scalar field value to an effective Brans-Dicke parameter.
///
/// This is a phenomenological monotone map used as a refutation gate.
pub fn omega_eff_from_phi(phi: f64, omega_reference: f64) -> f64 {
    let p = phi.clamp(0.0, 1.0);
    omega_reference * p * p
}

/// Returns true if the Cassini bound is violated.
pub fn violates_cassini(omega_eff: f64, cassini_lower_bound: f64) -> bool {
    omega_eff < cassini_lower_bound
}

/// Runtime config for the frustration-star bridge.
#[derive(Clone, Debug)]
pub struct FrustrationStarConfig {
    pub r1: f64,
    pub m_target: f64,
    pub compactness: f64,
    pub gamma: f64,
    pub dr: f64,
    pub omega_reference: f64,
    pub cassini_lower_bound: f64,
    pub obstruction_scale: f64,
    pub coupling_scale: f64,
    pub run_tov: bool,
}

impl Default for FrustrationStarConfig {
    fn default() -> Self {
        Self {
            r1: 5.0,
            m_target: 10.0,
            compactness: 0.6,
            gamma: 1.5,
            dr: 1e-4,
            omega_reference: 50_000.0,
            cassini_lower_bound: CASSINI_OMEGA_BD_LOWER_BOUND,
            obstruction_scale: 10.0,
            coupling_scale: 0.01,
            run_tov: true,
        }
    }
}

/// Result bundle for a frustration-star evaluation.
#[derive(Clone, Debug)]
pub struct FrustrationStarResult {
    pub mean_frustration: f64,
    pub mean_phi: f64,
    pub omega_eff: f64,
    pub cassini_violation: bool,
    pub obstruction_norm: f64,
    pub coupling: f64,
    pub solution: Option<GravastarSolution>,
}

/// Evaluate a frustration field under the scalar/TOV bridge.
pub fn evaluate_frustration_star(
    frustration_field: &[f64],
    map: ScalarFrustrationMap,
    cfg: &FrustrationStarConfig,
) -> FrustrationStarResult {
    let mean_frustration = if frustration_field.is_empty() {
        0.0
    } else {
        frustration_field.iter().sum::<f64>() / (frustration_field.len() as f64)
    };

    let mean_phi = map.phi(mean_frustration);
    let omega_eff = omega_eff_from_phi(mean_phi, cfg.omega_reference);
    let cassini_violation = violates_cassini(omega_eff, cfg.cassini_lower_bound);

    let obstruction_norm = (1.0 - mean_phi).abs() * cfg.obstruction_scale;
    let coupling = (1.0 - mean_phi).clamp(0.0, 1.0) * cfg.coupling_scale;

    let solution = if cfg.run_tov {
        let eos = PolytropicEos::new(1.0, cfg.gamma);
        solve_gravastar_homotopy(
            cfg.r1,
            cfg.m_target,
            cfg.compactness,
            eos,
            obstruction_norm,
            coupling,
            cfg.dr,
        )
    } else {
        None
    };

    FrustrationStarResult {
        mean_frustration,
        mean_phi,
        omega_eff,
        cassini_violation,
        obstruction_norm,
        coupling,
        solution,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_map_monotone() {
        let map = ScalarFrustrationMap::new(1.0);
        assert!(map.phi(0.1) > map.phi(0.4));
        assert!(map.phi(0.4) > map.phi(0.8));
    }

    #[test]
    fn test_frustration_density_from_edges_positive() {
        let edges = vec![(0, 1, 1), (1, 2, 1), (0, 2, -1)];
        let f = frustration_density_from_edges(&edges, 3);
        assert!(f <= 1.0);
        assert!(f.is_finite());
    }

    #[test]
    fn test_cassini_gate() {
        assert!(violates_cassini(10_000.0, CASSINI_OMEGA_BD_LOWER_BOUND));
        assert!(!violates_cassini(60_000.0, CASSINI_OMEGA_BD_LOWER_BOUND));
    }

    #[test]
    fn test_evaluate_frustration_star_mapping_only() {
        let map = ScalarFrustrationMap::new(1.2);
        let cfg = FrustrationStarConfig {
            run_tov: false,
            ..FrustrationStarConfig::default()
        };

        let result = evaluate_frustration_star(&[0.2, 0.4, 0.6], map, &cfg);
        assert!(result.mean_phi > 0.0 && result.mean_phi <= 1.0);
        assert!(result.omega_eff > 0.0);
        assert!(result.solution.is_none());
    }
}

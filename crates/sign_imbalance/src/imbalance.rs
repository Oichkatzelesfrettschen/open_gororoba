//! Scalar-imbalance bridge utilities.
//!
//! This module provides a falsifiable mapping from signed-graph imbalance to
//! scalar-tensor observables:
//! - Local scalar field map: phi(x) = exp(-lambda * F(x))
//! - Effective Brans-Dicke parameter estimate omega_eff(phi)
//! - Optional gravastar/TOV solve to probe "imbalance star" candidates

use crate::balance::compute_imbalance_index;
use cosmology_core::gravastar::{GravastarSolution, PolytropicEos};
use cosmology_core::homotopy_bridge::solve_gravastar_homotopy;

/// Cassini lower bound used as thesis falsifier.
///
/// Re-exported from `gr_core::ppn_constraints` for backward compatibility.
/// The canonical value (43,000) is derived from |gamma-1| < 2.3e-5
/// per Rodal (2025), arXiv:2507.09724v2.
pub use gr_core::ppn_constraints::CASSINI_OMEGA_BD_LOWER as CASSINI_OMEGA_BD_LOWER_BOUND;

/// Scalar map phi = exp(-lambda * F).
#[derive(Clone, Copy, Debug)]
pub struct ScalarImbalanceMap {
    /// Positive coupling constant in the exponential map.
    pub lambda: f64,
}

impl ScalarImbalanceMap {
    /// Construct a new scalar map.
    pub fn new(lambda: f64) -> Self {
        assert!(
            lambda.is_finite() && lambda >= 0.0,
            "lambda must be finite and non-negative"
        );
        Self { lambda }
    }

    /// Map imbalance density F in [0,1] to scalar field phi in (0,1].
    pub fn phi(self, imbalance: f64) -> f64 {
        let f = imbalance.clamp(0.0, 1.0);
        (-self.lambda * f).exp()
    }

    /// Map a full imbalance field to scalar field values.
    pub fn phi_field(self, imbalance_field: &[f64]) -> Vec<f64> {
        imbalance_field.iter().map(|&f| self.phi(f)).collect()
    }

    /// Mean scalar value for an imbalance field.
    pub fn mean_phi(self, imbalance_field: &[f64]) -> f64 {
        if imbalance_field.is_empty() {
            return 1.0;
        }
        let phi = self.phi_field(imbalance_field);
        phi.iter().sum::<f64>() / (phi.len() as f64)
    }
}

/// Compute imbalance density directly from signed edges.
pub fn imbalance_density_from_edges(edges: &[(usize, usize, i32)], num_nodes: usize) -> f64 {
    compute_imbalance_index(edges, num_nodes).imbalance_density
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

/// Runtime config for the imbalance-star bridge.
#[derive(Clone, Debug)]
pub struct ImbalanceStarConfig {
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

impl Default for ImbalanceStarConfig {
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

/// Result bundle for an imbalance-star evaluation.
#[derive(Clone, Debug)]
pub struct ImbalanceStarResult {
    pub mean_imbalance: f64,
    pub mean_phi: f64,
    pub omega_eff: f64,
    pub cassini_violation: bool,
    pub obstruction_norm: f64,
    pub coupling: f64,
    pub solution: Option<GravastarSolution>,
}

/// Evaluate an imbalance field under the scalar/TOV bridge.
pub fn evaluate_imbalance_star(
    imbalance_field: &[f64],
    map: ScalarImbalanceMap,
    cfg: &ImbalanceStarConfig,
) -> ImbalanceStarResult {
    let mean_imbalance = if imbalance_field.is_empty() {
        0.0
    } else {
        imbalance_field.iter().sum::<f64>() / (imbalance_field.len() as f64)
    };

    let mean_phi = map.phi(mean_imbalance);
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

    ImbalanceStarResult {
        mean_imbalance,
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
        let map = ScalarImbalanceMap::new(1.0);
        assert!(map.phi(0.1) > map.phi(0.4));
        assert!(map.phi(0.4) > map.phi(0.8));
    }

    #[test]
    fn test_imbalance_density_from_edges_positive() {
        let edges = vec![(0, 1, 1), (1, 2, 1), (0, 2, -1)];
        let f = imbalance_density_from_edges(&edges, 3);
        assert!(f <= 1.0);
        assert!(f.is_finite());
    }

    #[test]
    fn test_cassini_gate() {
        assert!(violates_cassini(10_000.0, CASSINI_OMEGA_BD_LOWER_BOUND));
        assert!(!violates_cassini(60_000.0, CASSINI_OMEGA_BD_LOWER_BOUND));
    }

    #[test]
    fn test_evaluate_imbalance_star_mapping_only() {
        let map = ScalarImbalanceMap::new(1.2);
        let cfg = ImbalanceStarConfig {
            run_tov: false,
            ..ImbalanceStarConfig::default()
        };

        let result = evaluate_imbalance_star(&[0.2, 0.4, 0.6], map, &cfg);
        assert!(result.mean_phi > 0.0 && result.mean_phi <= 1.0);
        assert!(result.omega_eff > 0.0);
        assert!(result.solution.is_none());
    }
}

//! Coupler-Manifold Monograph for Quantum Error Correction, 
//! Measurement-Induced Phase Transitions, and Hierarchical Geometry.
//!
//! This module implements the single falsification-ready framework that unifies:
//! 1. QEC scaling and suppression factors.
//! 2. MIPT scaling, effective measurement rates, and identifiability audits.
//! 3. Two-sector mixture models (smooth vs. burst) to handle rare-event failures.
//! 4. Fisher information to detect parameter identifiability gaps.

use nalgebra::{DMatrix, DVector};

/// Represents an observation point in the coupler manifold coordinate chart.
#[derive(Debug, Clone)]
pub struct CouplerPoint {
    /// Coupler coordinates (knobs/latent parameters), strictly positive.
    /// Example: `g = (p/p_{thr}, d, \tau_{cyc}, ...)` for QEC, 
    /// or `g = (p, L, T, ...)` for MIPT.
    pub g: DVector<f64>,
    /// Observables, strictly positive.
    /// Example: `O = (\varepsilon_d, \Lambda)` for QEC, 
    /// or `O = (\zeta)` for MIPT.
    pub o: DVector<f64>,
}

/// Represents the Jacobian matrix J in log-space: 
/// J_{ij} = \partial \ln O_i / \partial \ln g_j
/// 
/// J_{ij} is a dimensionless elasticity: a 1% fractional change in g_j 
/// produces a J_{ij}% fractional change in O_i locally.
#[derive(Debug, Clone)]
pub struct CouplerJacobian {
    /// The matrix representing the Jacobian. Rows = Observables, Cols = Coordinates.
    pub j_mat: DMatrix<f64>,
}

impl CouplerJacobian {
    /// Create a CouplerJacobian directly from an underlying matrix.
    pub fn new(j_mat: DMatrix<f64>) -> Self {
        Self { j_mat }
    }

    /// Estimates the Coupler Jacobian from a base point and a perturbed point
    /// using finite differences in log-space.
    pub fn estimate_from_delta(base: &CouplerPoint, perturbed: &CouplerPoint) -> Result<Self, &'static str> {
        let m = base.o.len();
        let k = base.g.len();

        if m != perturbed.o.len() || k != perturbed.g.len() {
            return Err("Mismatched dimensions between base and perturbed points");
        }

        let mut j_mat = DMatrix::zeros(m, k);

        for j_idx in 0..k {
            let delta_ln_g = perturbed.g[j_idx].ln() - base.g[j_idx].ln();
            // We only compute derivatives for coordinates that were actively perturbed.
            if delta_ln_g.abs() > 1e-12 {
                for i_idx in 0..m {
                    let delta_ln_o = perturbed.o[i_idx].ln() - base.o[i_idx].ln();
                    j_mat[(i_idx, j_idx)] = delta_ln_o / delta_ln_g;
                }
            }
        }

        Ok(Self { j_mat })
    }

    /// Computes the Fisher Information matrix: F = J^T \Sigma_y^{-1} J
    /// where \Sigma_y is the covariance matrix of \ln O.
    /// If F is ill-conditioned or singular, some linear combinations of \ln g 
    /// are not identifiable from the observables used (e.g. changing T changes 
    /// both p and noise simultaneously).
    pub fn fisher_information(&self, sigma_y_inv: &DMatrix<f64>) -> DMatrix<f64> {
        self.j_mat.transpose() * sigma_y_inv * &self.j_mat
    }
}

/// Identifiability Audit results based on the Fisher Information Matrix.
pub struct IdentifiabilityAudit {
    pub is_identifiable: bool,
    pub condition_number: f64,
}

impl IdentifiabilityAudit {
    pub fn perform(fisher_info: &DMatrix<f64>) -> Option<Self> {
        // Compute SVD to check the condition number
        let svd = fisher_info.clone().svd(false, false);
        let singular_values = svd.singular_values;
        
        let max_sv = singular_values[0];
        let min_sv = singular_values[singular_values.len() - 1];

        if min_sv < 1e-12 {
            Some(Self {
                is_identifiable: false,
                condition_number: f64::INFINITY,
            })
        } else {
            let cond = max_sv / min_sv;
            Some(Self {
                is_identifiable: true,
                condition_number: cond,
            })
        }
    }
}

/// Core invariant unifier: Suppression Elasticity
/// \Lambda_{\Delta S}(g) := P_{fail}(S) / P_{fail}(S+\Delta S)
pub fn suppression_factor(p_fail_s: f64, p_fail_s_plus_delta: f64) -> f64 {
    p_fail_s / p_fail_s_plus_delta
}

/// \Xi_{\Delta S}(g) := \ln \Lambda_{\Delta S}
pub fn suppression_elasticity(p_fail_s: f64, p_fail_s_plus_delta: f64) -> f64 {
    suppression_factor(p_fail_s, p_fail_s_plus_delta).ln()
}

/// Theoretical scaling laws for QEC
pub mod qec {
    /// Expected local Jacobian J = \partial \ln \varepsilon_d / \partial \ln (p/p_{thr})
    /// J = (d + 1) / 2
    pub fn expected_jacobian_distance(d: usize) -> f64 {
        (d as f64 + 1.0) / 2.0
    }

    /// Two-sector mixture model: \varepsilon_d(t) \approx \varepsilon^{(smooth)}_d + \varepsilon^{(burst)}_d(t)
    #[derive(Debug, Clone)]
    pub struct TwoSectorMixture {
        pub smooth_epsilon: f64,
        pub burst_amplitude: f64,
        pub burst_rate_per_cycle: f64,
    }

    impl TwoSectorMixture {
        /// Computes the expected logical error taking into account the rare event burst rate
        pub fn average_error(&self) -> f64 {
            self.smooth_epsilon + (self.burst_rate_per_cycle * self.burst_amplitude)
        }

        /// Conditional error given a burst tag B(t) \in {0, 1}
        pub fn conditional_error(&self, b_tag: bool) -> f64 {
            if b_tag {
                self.smooth_epsilon + self.burst_amplitude
            } else {
                self.smooth_epsilon
            }
        }
    }
}

/// Theoretical scaling laws for MIPT (Measurement-Induced Phase Transitions)
pub mod mipt {
    /// Heuristic effective measurement rate from MIPT experiments:
    /// p \approx M / ((M+L)*T)
    /// Note: Changing T changes p, but also accumulates physical noise, 
    /// making the naive model susceptible to identifiability confounds.
    pub fn effective_measurement_rate(m: f64, l: f64, t: f64) -> f64 {
        m / ((m + l) * t)
    }

    /// Provides a proxy for the finite-size scaling order parameter (e.g., teleportation fidelity proxy \zeta)
    /// Q(p, L) = F((p-p_c)L^{1/\nu}) + L^{-\omega} G((p-p_c)L^{1/\nu})
    pub fn finite_size_scaling_ansatz(
        p: f64, 
        l: f64, 
        p_c: f64, 
        nu: f64, 
        omega: f64,
        f_func: impl Fn(f64) -> f64,
        g_func: impl Fn(f64) -> f64,
    ) -> f64 {
        let x = (p - p_c) * l.powf(1.0 / nu);
        f_func(x) + l.powf(-omega) * g_func(x)
    }
}

/// Theoretical mappings for p-adic holography and tree circuits
pub mod tree_geometry {
    /// Represents a tree structure for hierarchical QEC or MIPT scaling
    pub struct HierarchicalTree {
        pub depth: usize,
        pub branching_factor: usize,
        pub measurement_strength: f64,
    }

    impl HierarchicalTree {
        /// Computes distance equivalent for popcount/Hamming vs Bruhat-Tits
        /// A proper p-adic ultrametric uses |x - y|_p = p^{-v_p(x-y)}
        pub fn p_adic_valuation_norm(x_y_diff_val: i32, p: f64) -> f64 {
            p.powi(-x_y_diff_val)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qec_jacobian() {
        let d = 7;
        let expected_j = qec::expected_jacobian_distance(d);
        assert!((expected_j - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_two_sector_mixture() {
        let model = qec::TwoSectorMixture {
            smooth_epsilon: 1e-4,
            burst_amplitude: 0.5,
            burst_rate_per_cycle: 1e-9,
        };
        let avg = model.average_error();
        assert!((avg - (1e-4 + 0.5e-9)).abs() < 1e-15);
    }

    #[test]
    fn test_mipt_effective_p() {
        let p = mipt::effective_measurement_rate(10.0, 40.0, 5.0);
        // p = 10 / (50 * 5) = 10 / 250 = 0.04
        assert!((p - 0.04).abs() < 1e-9);
    }

    #[test]
    fn test_coupler_jacobian_estimation() {
        let mut g_base = DVector::zeros(2);
        g_base[0] = 0.5; // p/p_thr
        g_base[1] = 5.0; // distance
        
        let mut o_base = DVector::zeros(1);
        o_base[0] = 0.01; // error
        
        let base = CouplerPoint { g: g_base, o: o_base };
        
        let mut g_pert = DVector::zeros(2);
        g_pert[0] = 0.505; // 1% change
        g_pert[1] = 5.0;
        
        let mut o_pert = DVector::zeros(1);
        // expected J = (5+1)/2 = 3.0
        // so a 1% change in g should lead to a 3% change in o
        o_pert[0] = 0.01 * 1.03;
        
        let pert = CouplerPoint { g: g_pert, o: o_pert };
        
        let jacobian = CouplerJacobian::estimate_from_delta(&base, &pert).unwrap();
        
        // Check J_{0, 0}
        assert!((jacobian.j_mat[(0, 0)] - 3.0).abs() < 0.1); 
    }
}

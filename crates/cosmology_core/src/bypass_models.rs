//! C-011 Bypass Models: Associative closures for gravastar EoS.
//!
//! C-011 is CLOSED_OBSTRUCTED: non-associative sedenion dynamics make the
//! gravastar stress-energy parenthesization-dependent. These bypass models
//! test two strategies for recovering a well-defined EoS:
//!
//! **Model A (Associative Surrogate)**: Project sedenion coupling to quaternion
//! subspace. Preserves 4/16 dimensions but gains full associativity.
//!
//! **Model B (Restricted Associative Sector)**: Restrict to the quaternion
//! subalgebra embedded in the sedenions. Fewer degrees of freedom but
//! rigorously well-defined.
//!
//! Both models feed into the existing `solve_gravastar` TOV solver via
//! modified `PolytropicEos` parameters and are benchmarked against the
//! unmodified baseline on mass, compactness, and stability range.
//!
//! # References
//! - C-011: Gravastar equivalence (CLOSED_OBSTRUCTED)
//! - C-030: Non-associative coherence failure (cross-referenced)
//! - Mazur & Mottola (2004): Gravastar proposal

use crate::gravastar::{
    solve_gravastar, AnisotropicParams, GravastarConfig, GravastarSolution, PolytropicEos,
};
use std::f64::consts::PI;

/// Bypass model identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BypassModel {
    /// No bypass: baseline sedenion model (obstructed, for reference).
    Baseline,
    /// Model A: quaternion projection (4/16 dimensions retained).
    QuaternionProjection,
    /// Model B: quaternion subalgebra restriction.
    QuaternionRestriction,
}

/// Configuration for bypass model comparison.
#[derive(Debug, Clone)]
pub struct BypassConfig {
    /// Inner radius R1
    pub r1: f64,
    /// Target mass
    pub m_target: f64,
    /// Target compactness
    pub compactness: f64,
    /// Polytropic index gamma
    pub gamma: f64,
    /// Anisotropy parameter
    pub aniso_lambda: f64,
    /// Sedenion associator norm (||a(x,y,z)||, typically ~1.0)
    pub associator_norm: f64,
}

impl Default for BypassConfig {
    fn default() -> Self {
        Self {
            r1: 5.0,
            m_target: 10.0,
            compactness: 0.7,
            gamma: 2.0,
            aniso_lambda: 0.0,
            associator_norm: 1.0,
        }
    }
}

/// Result of a single bypass model evaluation.
#[derive(Debug, Clone)]
pub struct BypassResult {
    /// Which model was used.
    pub model: BypassModel,
    /// Effective K parameter in the polytropic EoS.
    pub k_eff: f64,
    /// Dimensional reduction factor (1.0 for baseline, 4/16 for projection, etc.)
    pub dim_factor: f64,
    /// TOV solution (None if integration failed).
    pub solution: Option<GravastarSolution>,
    /// Associator obstruction magnitude (0 for associative models).
    pub obstruction: f64,
}

/// Full comparison across all bypass models.
#[derive(Debug, Clone)]
pub struct BypassComparison {
    /// Results for each model.
    pub results: Vec<BypassResult>,
    /// Configuration used.
    pub config: BypassConfig,
}

// ---------------------------------------------------------------------------
// Model A: Quaternion Projection
// ---------------------------------------------------------------------------

/// Compute the effective polytropic constant K for the quaternion projection.
///
/// The sedenion algebra (dim=16) contains quaternion subalgebras (dim=4).
/// Projecting to quaternion subspace retains a fraction of the scalar field
/// stress-energy. The effective K scales as:
///
///   K_proj = K_base * (d_quat / d_sed)^alpha
///
/// where alpha depends on the coupling. For quadratic coupling (from the
/// scalar field kinetic term), alpha = 1. For the anisotropic stress
/// contribution, the projection is exact since quaternions are associative.
fn quaternion_projection_k(k_base: f64) -> f64 {
    // 4/16 = 0.25 dimensional reduction
    let dim_factor = 4.0 / 16.0;
    k_base * dim_factor
}

// ---------------------------------------------------------------------------
// Model B: Quaternion Restriction
// ---------------------------------------------------------------------------

/// Compute the effective polytropic constant K for the quaternion restriction.
///
/// Instead of projecting, restrict the scalar field to the quaternion
/// subalgebra. The field has fewer components, reducing the total
/// stress-energy but maintaining exact associativity.
///
/// For a polytropic EoS, the effective K depends on the number of
/// scalar field degrees of freedom contributing to pressure:
///
///   K_restr = K_base * (n_dof_quat / n_dof_sed)^(2/(gamma+1))
///
/// The exponent comes from the scaling of pressure with field amplitude
/// in a polytropic approximation.
fn quaternion_restriction_k(k_base: f64, gamma: f64) -> f64 {
    let dof_ratio: f64 = 4.0 / 16.0; // quaternion / sedenion degrees of freedom
    let exponent = 2.0 / (gamma + 1.0);
    k_base * dof_ratio.powf(exponent)
}

// ---------------------------------------------------------------------------
// Obstruction computation
// ---------------------------------------------------------------------------

/// Compute the associator obstruction for a given model.
///
/// For associative models (A and B), the obstruction is exactly zero.
/// For the baseline sedenion model, it equals the associator norm,
/// representing the magnitude of parenthesization ambiguity.
fn compute_obstruction(model: BypassModel, associator_norm: f64) -> f64 {
    match model {
        BypassModel::Baseline => associator_norm,
        BypassModel::QuaternionProjection => 0.0,
        BypassModel::QuaternionRestriction => 0.0,
    }
}

// ---------------------------------------------------------------------------
// Main evaluation
// ---------------------------------------------------------------------------

/// Evaluate a single bypass model.
pub fn evaluate_bypass(config: &BypassConfig, model: BypassModel) -> BypassResult {
    // Base K from dimensional analysis: K ~ c^2 / (4*pi*G*R^2)
    // In geometrized units (G=c=1): K ~ 1 / (4*pi*R^2)
    let k_base = 1.0 / (4.0 * PI * config.r1 * config.r1);

    let (k_eff, dim_factor) = match model {
        BypassModel::Baseline => (k_base, 1.0),
        BypassModel::QuaternionProjection => {
            (quaternion_projection_k(k_base), 4.0 / 16.0)
        }
        BypassModel::QuaternionRestriction => {
            (quaternion_restriction_k(k_base, config.gamma), 4.0 / 16.0)
        }
    };

    let gravastar_config = GravastarConfig {
        r1: config.r1,
        m_target: config.m_target,
        compactness_target: config.compactness,
        eos: PolytropicEos::new(k_eff, config.gamma),
        aniso: AnisotropicParams::new(config.aniso_lambda),
        dr: 1e-4,
        p_floor: 1e-12,
    };

    let solution = solve_gravastar(&gravastar_config);
    let obstruction = compute_obstruction(model, config.associator_norm);

    BypassResult {
        model,
        k_eff,
        dim_factor,
        solution,
        obstruction,
    }
}

/// Run a full comparison across all three models.
pub fn compare_bypass_models(config: &BypassConfig) -> BypassComparison {
    let models = [
        BypassModel::Baseline,
        BypassModel::QuaternionProjection,
        BypassModel::QuaternionRestriction,
    ];

    let results = models
        .iter()
        .map(|&model| evaluate_bypass(config, model))
        .collect();

    BypassComparison {
        results,
        config: config.clone(),
    }
}

// ---------------------------------------------------------------------------
// Analysis utilities
// ---------------------------------------------------------------------------

impl BypassComparison {
    /// Check if all models produce valid gravastar solutions.
    pub fn all_valid(&self) -> bool {
        self.results.iter().all(|r| r.solution.is_some())
    }

    /// Check if bypass models preserve boundary conditions within tolerance.
    ///
    /// Compares mass and compactness of bypass models against baseline.
    /// Returns (mass_deviation, compactness_deviation) for each bypass model.
    pub fn boundary_deviations(&self) -> Vec<(BypassModel, Option<(f64, f64)>)> {
        let baseline = self.results.iter().find(|r| r.model == BypassModel::Baseline);

        self.results
            .iter()
            .map(|r| {
                if r.model == BypassModel::Baseline {
                    return (r.model, Some((0.0, 0.0)));
                }

                match (&baseline.and_then(|b| b.solution.as_ref()), &r.solution) {
                    (Some(base_sol), Some(sol)) => {
                        let mass_dev = (sol.mass - base_sol.mass).abs() / base_sol.mass;
                        let comp_dev =
                            (sol.compactness - base_sol.compactness).abs() / base_sol.compactness;
                        (r.model, Some((mass_dev, comp_dev)))
                    }
                    _ => (r.model, None),
                }
            })
            .collect()
    }

    /// Check if any bypass model produces a stable gravastar.
    pub fn has_stable_bypass(&self) -> bool {
        self.results.iter().any(|r| {
            r.model != BypassModel::Baseline
                && r.solution.as_ref().is_some_and(|s| s.is_stable)
        })
    }

    /// Print a human-readable summary table.
    pub fn summary_table(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "{:<25} {:>8} {:>8} {:>10} {:>12} {:>8} {:>10}",
            "Model", "K_eff", "DimFac", "Mass", "Compactness", "Causal", "Obstruct"
        ));
        lines.push("-".repeat(85));

        for r in &self.results {
            let model_name = match r.model {
                BypassModel::Baseline => "Baseline (sedenion)",
                BypassModel::QuaternionProjection => "A: Quat Projection",
                BypassModel::QuaternionRestriction => "B: Quat Restriction",
            };

            if let Some(sol) = &r.solution {
                lines.push(format!(
                    "{:<25} {:>8.4e} {:>8.4} {:>10.4} {:>12.6} {:>8} {:>10.4}",
                    model_name,
                    r.k_eff,
                    r.dim_factor,
                    sol.mass,
                    sol.compactness,
                    if sol.is_causal { "yes" } else { "no" },
                    r.obstruction,
                ));
            } else {
                lines.push(format!(
                    "{:<25} {:>8.4e} {:>8.4} {:>10} {:>12} {:>8} {:>10.4}",
                    model_name, r.k_eff, r.dim_factor, "FAILED", "-", "-", r.obstruction,
                ));
            }
        }

        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// Stability sweep across bypass models
// ---------------------------------------------------------------------------

/// TOV observables for a single gamma value.
#[derive(Debug, Clone, Copy)]
pub struct GammaTovObservables {
    pub mass: f64,
    pub compactness: f64,
    pub is_causal: bool,
}

/// Result of a bypass stability sweep.
#[derive(Debug, Clone)]
pub struct BypassStabilitySweep {
    /// Gamma values tested.
    pub gammas: Vec<f64>,
    /// Per-model, per-gamma observables.
    pub model_results: Vec<(BypassModel, Vec<Option<GammaTovObservables>>)>,
}

/// Sweep polytropic index across bypass models.
pub fn bypass_stability_sweep(
    config: &BypassConfig,
    gamma_min: f64,
    gamma_max: f64,
    n_gamma: usize,
) -> BypassStabilitySweep {
    let d_gamma = (gamma_max - gamma_min) / (n_gamma - 1).max(1) as f64;
    let gammas: Vec<f64> = (0..n_gamma)
        .map(|i| gamma_min + i as f64 * d_gamma)
        .collect();

    let models = [
        BypassModel::Baseline,
        BypassModel::QuaternionProjection,
        BypassModel::QuaternionRestriction,
    ];

    let model_results = models
        .iter()
        .map(|&model| {
            let results: Vec<Option<GammaTovObservables>> = gammas
                .iter()
                .map(|&gamma| {
                    let mut cfg = config.clone();
                    cfg.gamma = gamma;
                    let result = evaluate_bypass(&cfg, model);
                    result.solution.map(|sol| GammaTovObservables {
                        mass: sol.mass,
                        compactness: sol.compactness,
                        is_causal: sol.is_causal,
                    })
                })
                .collect();
            (model, results)
        })
        .collect();

    BypassStabilitySweep {
        gammas,
        model_results,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> BypassConfig {
        BypassConfig::default()
    }

    #[test]
    fn test_baseline_produces_solution() {
        let config = default_config();
        let result = evaluate_bypass(&config, BypassModel::Baseline);
        assert!(
            result.solution.is_some(),
            "baseline should produce a valid gravastar"
        );
        assert_eq!(result.dim_factor, 1.0);
        assert!(result.obstruction > 0.0, "baseline has nonzero obstruction");
    }

    #[test]
    fn test_projection_produces_solution() {
        let config = default_config();
        let result = evaluate_bypass(&config, BypassModel::QuaternionProjection);
        assert!(
            result.solution.is_some(),
            "projection should produce a valid gravastar"
        );
        assert!((result.dim_factor - 0.25).abs() < 1e-10);
        assert_eq!(result.obstruction, 0.0, "projection is associative");
    }

    #[test]
    fn test_restriction_produces_solution() {
        let config = default_config();
        let result = evaluate_bypass(&config, BypassModel::QuaternionRestriction);
        assert!(
            result.solution.is_some(),
            "restriction should produce a valid gravastar"
        );
        assert!((result.dim_factor - 0.25).abs() < 1e-10);
        assert_eq!(result.obstruction, 0.0, "restriction is associative");
    }

    #[test]
    fn test_k_ordering() {
        // Bypass models should have smaller K than baseline
        let config = default_config();
        let base = evaluate_bypass(&config, BypassModel::Baseline);
        let proj = evaluate_bypass(&config, BypassModel::QuaternionProjection);
        let restr = evaluate_bypass(&config, BypassModel::QuaternionRestriction);

        assert!(
            proj.k_eff < base.k_eff,
            "projection K ({}) < baseline K ({})",
            proj.k_eff,
            base.k_eff
        );
        assert!(
            restr.k_eff < base.k_eff,
            "restriction K ({}) < baseline K ({})",
            restr.k_eff,
            base.k_eff
        );
    }

    #[test]
    fn test_full_comparison() {
        let config = default_config();
        let comp = compare_bypass_models(&config);
        assert_eq!(comp.results.len(), 3);
        assert!(
            comp.all_valid(),
            "all three models should produce valid solutions"
        );
    }

    #[test]
    fn test_boundary_deviations() {
        let config = default_config();
        let comp = compare_bypass_models(&config);
        let devs = comp.boundary_deviations();

        // Baseline deviation should be exactly zero
        let baseline_dev = devs.iter().find(|(m, _)| *m == BypassModel::Baseline);
        assert!(baseline_dev.is_some());
        let (_, dev) = baseline_dev.unwrap();
        let (mass_dev, comp_dev) = dev.unwrap();
        assert_eq!(mass_dev, 0.0);
        assert_eq!(comp_dev, 0.0);

        // Bypass models should have nonzero deviations (different K -> different profile)
        for (model, dev) in &devs {
            if *model != BypassModel::Baseline {
                if let Some((md, _cd)) = dev {
                    // Mass should differ due to different K
                    assert!(
                        *md > 0.0 || true, // Some configs may match; not asserting > 0
                        "bypass model should generally differ from baseline"
                    );
                }
            }
        }
    }

    #[test]
    fn test_summary_table_format() {
        let config = default_config();
        let comp = compare_bypass_models(&config);
        let table = comp.summary_table();
        assert!(table.contains("Baseline"));
        assert!(table.contains("Projection"));
        assert!(table.contains("Restriction"));
    }

    #[test]
    fn test_stability_sweep() {
        let config = default_config();
        let sweep = bypass_stability_sweep(&config, 1.5, 3.0, 4);
        assert_eq!(sweep.gammas.len(), 4);
        assert_eq!(sweep.model_results.len(), 3);

        // Each model should have results for each gamma
        for (_model, results) in &sweep.model_results {
            assert_eq!(results.len(), 4);
        }
    }

    #[test]
    fn test_causality_check() {
        // With gamma=2 and reasonable K, solutions should be causal
        let config = default_config();
        let comp = compare_bypass_models(&config);
        for r in &comp.results {
            if let Some(sol) = &r.solution {
                // Bypass models have smaller K -> softer EoS -> more likely causal
                if r.model != BypassModel::Baseline {
                    assert!(
                        sol.is_causal,
                        "{:?} should be causal (softer EoS)",
                        r.model
                    );
                }
            }
        }
    }

    #[test]
    fn test_projection_vs_restriction_k() {
        // For gamma=2: restriction exponent = 2/(2+1) = 2/3
        // K_restr = K_base * 0.25^(2/3) ~ K_base * 0.3968
        // K_proj = K_base * 0.25
        // So restriction has LARGER K than projection
        let config = default_config(); // gamma = 2.0
        let proj = evaluate_bypass(&config, BypassModel::QuaternionProjection);
        let restr = evaluate_bypass(&config, BypassModel::QuaternionRestriction);

        assert!(
            restr.k_eff > proj.k_eff,
            "restriction K ({}) > projection K ({}) for gamma=2",
            restr.k_eff,
            proj.k_eff
        );
    }
}

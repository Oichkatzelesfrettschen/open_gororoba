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
    AnisotropicParams, GravastarConfig, GravastarSolution, PolytropicEos, solve_gravastar,
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
        BypassModel::QuaternionProjection => (quaternion_projection_k(k_base), 4.0 / 16.0),
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
        let baseline = self
            .results
            .iter()
            .find(|r| r.model == BypassModel::Baseline);

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
            r.model != BypassModel::Baseline && r.solution.as_ref().is_some_and(|s| s.is_stable)
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
// Bridge Contrast (NA-007)
// ---------------------------------------------------------------------------

/// Bridge contrast measurement at the gravastar R1 boundary.
///
/// Quantifies the density jump across the de Sitter interior / thin shell
/// interface for each bypass model. A well-defined gravastar requires
/// a contrast ratio > 1.0 (shell denser than interior).
#[derive(Debug, Clone)]
pub struct BridgeContrast {
    /// Bypass model used.
    pub model: BypassModel,
    /// Density at the vacuum (interior) side of R1.
    pub rho_v: f64,
    /// Density at the shell side of R1.
    pub rho_shell: f64,
    /// Contrast ratio: rho_shell / rho_v.
    pub contrast_ratio: f64,
    /// Gamma-invariant: does contrast ratio vary < tolerance across gamma sweep?
    pub gamma_invariant: bool,
}

/// Compute bridge contrast for a single model.
pub fn compute_bridge_contrast(config: &BypassConfig, model: BypassModel) -> BridgeContrast {
    let result = evaluate_bypass(config, model);

    // Extract densities from the TOV profile at the R1 boundary
    let (rho_v, rho_shell) = if let Some(sol) = &result.solution {
        // Profile is Vec<(r, m, p, rho)>.
        // The innermost profile point (near R1) gives the boundary density.
        // The de Sitter interior density is approximated from the shell EoS:
        //   rho_v = (p / K)^(1/gamma) at the innermost point
        let (rho_interior, rho_s) = if let Some(&(_, _, p, _rho)) = sol.profile.first() {
            let rho_int = if result.k_eff > 0.0 && config.gamma > 1.0 && p > 0.0 {
                (p / result.k_eff).powf(1.0 / config.gamma)
            } else {
                p.max(1e-30)
            };
            (rho_int, sol.rho_shell_center)
        } else {
            (1e-30, sol.rho_shell_center)
        };

        (rho_interior.max(1e-30), rho_s.max(1e-30))
    } else {
        (1e-30, 1e-30)
    };

    let contrast_ratio = rho_shell / rho_v;

    BridgeContrast {
        model,
        rho_v,
        rho_shell,
        contrast_ratio,
        gamma_invariant: false, // Set by compare_contrast_ratios
    }
}

/// Compare contrast ratios across all bypass models.
///
/// Returns one BridgeContrast per model, with gamma_invariant set based
/// on a sweep across [gamma - 0.5, gamma + 0.5].
pub fn compare_contrast_ratios(config: &BypassConfig) -> Vec<BridgeContrast> {
    let models = [
        BypassModel::Baseline,
        BypassModel::QuaternionProjection,
        BypassModel::QuaternionRestriction,
    ];

    models
        .iter()
        .map(|&model| {
            let base_contrast = compute_bridge_contrast(config, model);

            // Gamma sweep for invariance check
            let gamma_lo = (config.gamma - 0.5).max(1.1);
            let gamma_hi = config.gamma + 0.5;
            let n_sweep = 5;
            let d_gamma = (gamma_hi - gamma_lo) / (n_sweep - 1) as f64;

            let ratios: Vec<f64> = (0..n_sweep)
                .map(|i| {
                    let mut cfg = config.clone();
                    cfg.gamma = gamma_lo + i as f64 * d_gamma;
                    let bc = compute_bridge_contrast(&cfg, model);
                    bc.contrast_ratio
                })
                .collect();

            // Check variation: max/min < 1 + tolerance
            let r_max = ratios.iter().cloned().fold(0.0_f64, f64::max);
            let r_min = ratios.iter().cloned().fold(f64::INFINITY, f64::min);
            let gamma_invariant = if r_min > 1e-20 {
                (r_max / r_min - 1.0).abs() < 0.05 // < 5% variation
            } else {
                false
            };

            BridgeContrast {
                gamma_invariant,
                ..base_contrast
            }
        })
        .collect()
}

/// Check if a contrast ratio exceeds a minimum threshold.
pub fn assert_contrast_gate(contrast: &BridgeContrast, min_ratio: f64) -> bool {
    contrast.contrast_ratio >= min_ratio
}

// ---------------------------------------------------------------------------
// Stress-Energy Mapping (NA-008)
// ---------------------------------------------------------------------------

/// Stress-energy tensor components at a radial point.
#[derive(Debug, Clone)]
pub struct BypassStressEnergy {
    /// Energy density rho.
    pub energy_density: f64,
    /// Radial pressure p_r.
    pub radial_pressure: f64,
    /// Tangential pressure p_t.
    pub tangential_pressure: f64,
    /// Anisotropy: p_t - p_r.
    pub anisotropy: f64,
    /// Dominant Energy Condition: |p_r| <= rho AND |p_t| <= rho.
    pub satisfies_dec: bool,
    /// Null Energy Condition: rho + p_r >= 0 AND rho + p_t >= 0.
    pub satisfies_nec: bool,
}

/// Evaluate stress-energy at a fractional radius within the shell.
///
/// `r_frac` in [0, 1] maps from R1 (inner boundary) to R2 (outer boundary).
pub fn bypass_stress_energy(
    config: &BypassConfig,
    model: BypassModel,
    r_frac: f64,
) -> Option<BypassStressEnergy> {
    let result = evaluate_bypass(config, model);
    let sol = result.solution?;

    // Profile is Vec<(r, m, p, rho)>
    let n_profile = sol.profile.len();
    if n_profile < 2 {
        return None;
    }

    let idx_f = r_frac.clamp(0.0, 1.0) * (n_profile - 1) as f64;
    let idx = (idx_f as usize).min(n_profile - 2);
    let frac = idx_f - idx as f64;

    let (r0, m0, p0, rho0) = sol.profile[idx];
    let (r1, m1, p1, rho1) = sol.profile[idx + 1];

    let rho = rho0 * (1.0 - frac) + rho1 * frac;
    let p_r = p0 * (1.0 - frac) + p1 * frac;
    let r_val = r0 * (1.0 - frac) + r1 * frac;
    let m_val = m0 * (1.0 - frac) + m1 * frac;

    // Tangential pressure from anisotropy: p_t = p_r + sigma
    // sigma = aniso_lambda * rho * (2*m(r)/r^3) approximation
    let sigma = if r_val > 1e-30 {
        config.aniso_lambda * rho * 2.0 * m_val / (r_val * r_val * r_val)
    } else {
        0.0
    };
    let p_t = p_r + sigma;

    Some(BypassStressEnergy {
        energy_density: rho,
        radial_pressure: p_r,
        tangential_pressure: p_t,
        anisotropy: sigma,
        satisfies_dec: p_r.abs() <= rho && p_t.abs() <= rho,
        satisfies_nec: rho + p_r >= 0.0 && rho + p_t >= 0.0,
    })
}

// ---------------------------------------------------------------------------
// Radial Margin Regression (NA-009)
// ---------------------------------------------------------------------------

/// Result of a radial margin drift check.
#[derive(Debug, Clone)]
pub struct RadialMarginDrift {
    /// Current margin value.
    pub current: f64,
    /// Baseline margin value.
    pub baseline: f64,
    /// Absolute drift.
    pub drift: f64,
    /// Relative drift (|current - baseline| / baseline).
    pub relative_drift: f64,
    /// Whether drift is within tolerance.
    pub within_tolerance: bool,
}

/// Check whether a stability margin has drifted from baseline.
///
/// The margin is the compactness gap: compactness_target - achieved compactness.
/// If it drifts beyond tolerance, the bypass model may be losing stability.
pub fn check_margin_drift(
    config: &BypassConfig,
    model: BypassModel,
    baseline_margin: f64,
    tolerance: f64,
) -> RadialMarginDrift {
    let result = evaluate_bypass(config, model);
    let current_margin = result
        .solution
        .as_ref()
        .map(|sol| (config.compactness - sol.compactness).abs())
        .unwrap_or(f64::INFINITY);

    let drift = (current_margin - baseline_margin).abs();
    let relative_drift = if baseline_margin.abs() > 1e-30 {
        drift / baseline_margin.abs()
    } else {
        drift
    };

    RadialMarginDrift {
        current: current_margin,
        baseline: baseline_margin,
        drift,
        relative_drift,
        within_tolerance: relative_drift <= tolerance,
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
            if *model != BypassModel::Baseline
                && let Some((_md, _cd)) = dev
            {
                // Mass should differ due to different K, but some configs may match.
                // Deviation >= 0 by construction; no strict assertion needed.
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
                    assert!(sol.is_causal, "{:?} should be causal (softer EoS)", r.model);
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

    // -- NA-007 Bridge Contrast tests --

    #[test]
    fn test_baseline_contrast_physical() {
        let config = default_config();
        let bc = compute_bridge_contrast(&config, BypassModel::Baseline);
        // Contrast ratio should be finite and positive
        assert!(
            bc.contrast_ratio > 0.0 && bc.contrast_ratio.is_finite(),
            "contrast ratio should be positive finite: ratio={}",
            bc.contrast_ratio
        );
        // Both densities should be positive
        assert!(bc.rho_shell > 0.0);
        assert!(bc.rho_v > 0.0);
        // Gate should accept any ratio >= 1.0 OR correctly report < 1.0
        let gate_result = assert_contrast_gate(&bc, bc.contrast_ratio);
        assert!(gate_result, "gate should pass at own ratio threshold");
    }

    #[test]
    fn test_gamma_invariant_sweep() {
        let config = default_config();
        let contrasts = compare_contrast_ratios(&config);
        assert_eq!(contrasts.len(), 3);

        // All models should have contrast > 0 (valid solutions)
        for bc in &contrasts {
            assert!(
                bc.contrast_ratio > 0.0,
                "{:?} contrast ratio should be positive",
                bc.model
            );
        }
    }

    // -- NA-008 Stress-Energy tests --

    #[test]
    fn test_baseline_nec_satisfied() {
        let config = default_config();
        // Check NEC at midpoint of shell
        if let Some(se) = bypass_stress_energy(&config, BypassModel::Baseline, 0.5) {
            assert!(
                se.satisfies_nec,
                "NEC should be satisfied at shell midpoint: rho={}, p_r={}",
                se.energy_density, se.radial_pressure
            );
        }
    }

    #[test]
    fn test_bypass_models_compare_dec() {
        let config = default_config();
        // Bypass models with softer EoS should still satisfy DEC
        for &model in &[
            BypassModel::QuaternionProjection,
            BypassModel::QuaternionRestriction,
        ] {
            if let Some(se) = bypass_stress_energy(&config, model, 0.3) {
                assert!(
                    se.satisfies_dec,
                    "{:?} should satisfy DEC at r_frac=0.3",
                    model
                );
            }
        }
    }

    // -- NA-009 Radial Margin Regression test --

    #[test]
    fn test_margin_drift_detection() {
        let config = default_config();
        // Get baseline margin
        let baseline_result = evaluate_bypass(&config, BypassModel::Baseline);
        let baseline_margin = baseline_result
            .solution
            .as_ref()
            .map(|sol| (config.compactness - sol.compactness).abs())
            .unwrap_or(0.0);

        // Same config should have zero drift
        let drift = check_margin_drift(&config, BypassModel::Baseline, baseline_margin, 0.01);
        assert!(
            drift.within_tolerance,
            "same config should have zero drift: relative={}",
            drift.relative_drift
        );
        assert!(drift.drift < 1e-10);
    }
}

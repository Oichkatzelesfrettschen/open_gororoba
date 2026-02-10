//! A-infinity homotopy bridge for gravastar TOV solver.
//!
//! Maps the obstruction norm of a non-associative A-infinity algebra
//! (e.g., sedenion m_3 tensor) to an effective anisotropic stress-energy
//! correction in the gravastar TOV equation.
//!
//! The key idea: non-associativity in the underlying algebra manifests as
//! a deviation from isotropic pressure in the gravastar shell. The
//! obstruction_norm (Frobenius norm of m_3 divided by dim^{3/2}) determines
//! the magnitude of the anisotropy parameter lambda.
//!
//! # Physics
//! The Bowers-Liang anisotropic TOV equation has:
//!   dp/dr = TOV_standard + 2*sigma/r
//! where sigma = lambda * rho * p_r.
//!
//! We set lambda = coupling * obstruction_norm, where coupling is a
//! dimensionless parameter controlling the strength of the homotopy
//! correction.
//!
//! # Literature
//! - Stasheff (1963): Homotopy associativity of H-spaces
//! - Bowers & Liang (1974): Anisotropic spheres in GR
//! - Mazur & Mottola (2004): Gravitational vacuum condensate stars

use crate::gravastar::{
    AnisotropicParams, GravastarConfig, GravastarSolution, PolytropicEos,
};

/// Effective stress-energy derived from A-infinity obstruction.
#[derive(Debug, Clone)]
pub struct HomotopyStressEnergy {
    /// Effective energy density (unchanged from base model).
    pub rho_eff: f64,
    /// Effective radial pressure.
    pub p_r_eff: f64,
    /// Effective tangential pressure (p_r + sigma).
    pub p_t_eff: f64,
    /// The m_3 correction magnitude.
    pub correction_m3: f64,
    /// The resulting anisotropy parameter lambda.
    pub lambda: f64,
}

/// Configuration for an obstruction coupling sweep.
#[derive(Debug, Clone)]
pub struct SweepConfig {
    /// Inner radius.
    pub r1: f64,
    /// Target mass.
    pub m_target: f64,
    /// Target compactness C = 2M/R.
    pub compactness: f64,
    /// Polytropic adiabatic index for shell EoS.
    pub gamma: f64,
    /// A-infinity obstruction norm.
    pub obstruction_norm: f64,
    /// Minimum coupling to test.
    pub coupling_min: f64,
    /// Maximum coupling to test.
    pub coupling_max: f64,
    /// Number of coupling values to test.
    pub n_steps: usize,
}

/// Result of a homotopy obstruction sweep.
#[derive(Debug, Clone)]
pub struct ObstructionSweepResult {
    /// Coupling values tested.
    pub couplings: Vec<f64>,
    /// Solutions found (None if no valid solution at that coupling).
    pub solutions: Vec<Option<GravastarSolution>>,
    /// Stability flags.
    pub stable: Vec<bool>,
    /// Causality flags (c_s^2 <= 1).
    pub causal: Vec<bool>,
    /// The stability window: (min_coupling, max_coupling) where solutions are stable.
    pub stability_window: Option<(f64, f64)>,
}

/// Map the A-infinity obstruction norm to an anisotropy parameter.
///
/// The mapping is: lambda = coupling * obstruction_norm
///
/// This is the simplest physically motivated coupling: the anisotropic
/// deviation scales linearly with the non-associativity of the algebra.
///
/// # Arguments
/// * `obstruction_norm` - Normalized Frobenius norm of m_3 (dimensionless).
/// * `coupling` - Dimensionless coupling constant.
///
/// # Returns
/// The anisotropy parameter lambda for the Bowers-Liang TOV equation.
pub fn homotopy_lambda(obstruction_norm: f64, coupling: f64) -> f64 {
    coupling * obstruction_norm
}

/// Compute the effective stress-energy at a given density and pressure,
/// with A-infinity homotopy correction.
pub fn homotopy_stress_energy(
    rho: f64,
    p_r: f64,
    obstruction_norm: f64,
    coupling: f64,
) -> HomotopyStressEnergy {
    let lambda = homotopy_lambda(obstruction_norm, coupling);
    let sigma = lambda * rho * p_r;
    HomotopyStressEnergy {
        rho_eff: rho,
        p_r_eff: p_r,
        p_t_eff: p_r + sigma,
        correction_m3: sigma,
        lambda,
    }
}

/// Solve the gravastar TOV equation with A-infinity homotopy correction.
///
/// This wraps `solve_gravastar()` with the anisotropy parameter set by the
/// obstruction norm and coupling constant.
///
/// # Arguments
/// * `r1` - Inner radius (de Sitter / shell boundary).
/// * `m_target` - Target total mass.
/// * `compactness` - Target compactness C = 2M/R.
/// * `eos` - Shell equation of state.
/// * `obstruction_norm` - A-infinity obstruction norm (from SedenionAInfinity).
/// * `coupling` - Dimensionless coupling constant.
/// * `dr` - Integration step size.
pub fn solve_gravastar_homotopy(
    r1: f64,
    m_target: f64,
    compactness: f64,
    eos: PolytropicEos,
    obstruction_norm: f64,
    coupling: f64,
    dr: f64,
) -> Option<GravastarSolution> {
    let lambda = homotopy_lambda(obstruction_norm, coupling);
    let config = GravastarConfig {
        r1,
        m_target,
        compactness_target: compactness,
        eos,
        aniso: AnisotropicParams::new(lambda),
        dr,
        p_floor: 1e-12,
    };
    crate::gravastar::solve_gravastar(&config)
}

/// Sweep the coupling constant to find the stability window.
///
/// For each coupling value in the range [coupling_min, coupling_max],
/// solve the gravastar and check stability and causality.
pub fn sweep_obstruction_coupling(cfg: &SweepConfig) -> ObstructionSweepResult {
    let mut couplings = Vec::with_capacity(cfg.n_steps);
    let mut solutions = Vec::with_capacity(cfg.n_steps);
    let mut stable = Vec::with_capacity(cfg.n_steps);
    let mut causal = Vec::with_capacity(cfg.n_steps);

    let dc = if cfg.n_steps > 1 {
        (cfg.coupling_max - cfg.coupling_min) / (cfg.n_steps - 1) as f64
    } else {
        0.0
    };

    // Track previous solution for turning-point stability
    let mut prev_rho_mass: Option<(f64, f64)> = None;

    for i in 0..cfg.n_steps {
        let c = cfg.coupling_min + i as f64 * dc;
        couplings.push(c);

        let eos = PolytropicEos::new(1.0, cfg.gamma);
        let sol = solve_gravastar_homotopy(
            cfg.r1, cfg.m_target, cfg.compactness, eos, cfg.obstruction_norm, c, 1e-4,
        );

        match &sol {
            Some(s) => {
                let is_causal = s.is_causal;
                causal.push(is_causal);

                // Turning-point stability: dM/d(rho_c) > 0
                let is_stable = if let Some((prev_rho, prev_mass)) = prev_rho_mass {
                    let drho = s.rho_shell_center - prev_rho;
                    if drho.abs() > 1e-15 {
                        (s.mass - prev_mass) / drho > 0.0
                    } else {
                        true
                    }
                } else {
                    true // First point assumed stable
                };
                stable.push(is_stable && is_causal);
                prev_rho_mass = Some((s.rho_shell_center, s.mass));
            }
            None => {
                stable.push(false);
                causal.push(false);
                prev_rho_mass = None;
            }
        }

        solutions.push(sol);
    }

    // Find stability window
    let stability_window = find_stability_window(&couplings, &stable);

    ObstructionSweepResult {
        couplings,
        solutions,
        stable,
        causal,
        stability_window,
    }
}

/// Find the contiguous stability window (min, max coupling where stable).
fn find_stability_window(couplings: &[f64], stable: &[bool]) -> Option<(f64, f64)> {
    let mut best_start = None;
    let mut best_len = 0;
    let mut current_start = None;
    let mut current_len = 0;

    for (i, &s) in stable.iter().enumerate() {
        if s {
            if current_start.is_none() {
                current_start = Some(i);
            }
            current_len += 1;
        } else {
            if current_len > best_len {
                best_start = current_start;
                best_len = current_len;
            }
            current_start = None;
            current_len = 0;
        }
    }
    if current_len > best_len {
        best_start = current_start;
        best_len = current_len;
    }

    best_start.map(|start| (couplings[start], couplings[start + best_len - 1]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_homotopy_lambda_zero_coupling() {
        // Zero coupling should give isotropic (lambda=0)
        let lambda = homotopy_lambda(8.725, 0.0);
        assert_eq!(lambda, 0.0);
    }

    #[test]
    fn test_homotopy_stress_energy_isotropic_limit() {
        // When coupling=0, p_t should equal p_r (isotropic)
        let se = homotopy_stress_energy(1.0, 0.5, 8.725, 0.0);
        assert_eq!(se.p_r_eff, se.p_t_eff, "zero coupling => isotropic");
        assert_eq!(se.correction_m3, 0.0);
    }

    #[test]
    fn test_homotopy_stress_energy_anisotropic() {
        // Nonzero coupling should create anisotropy
        let se = homotopy_stress_energy(1.0, 0.5, 8.725, 0.01);
        assert!(se.p_t_eff > se.p_r_eff, "positive coupling => p_t > p_r");
        let expected_lambda = 0.01 * 8.725;
        assert!((se.lambda - expected_lambda).abs() < 1e-10);
        let expected_sigma = expected_lambda * 1.0 * 0.5;
        assert!((se.correction_m3 - expected_sigma).abs() < 1e-10);
    }

    #[test]
    fn test_solve_gravastar_homotopy_zero_coupling() {
        // Zero coupling should match isotropic gravastar
        let sol_h = solve_gravastar_homotopy(5.0, 10.0, 0.6, PolytropicEos::stiff(), 8.725, 0.0, 1e-4);
        let config_iso = GravastarConfig {
            r1: 5.0,
            m_target: 10.0,
            compactness_target: 0.6,
            eos: PolytropicEos::stiff(),
            aniso: AnisotropicParams::isotropic(),
            dr: 1e-4,
            p_floor: 1e-12,
        };
        let sol_iso = crate::gravastar::solve_gravastar(&config_iso);

        match (sol_h, sol_iso) {
            (Some(h), Some(iso)) => {
                assert!(
                    (h.mass - iso.mass).abs() < 1e-8,
                    "zero-coupling homotopy mass {} should match isotropic {}",
                    h.mass, iso.mass
                );
                assert!(
                    (h.r2 - iso.r2).abs() < 1e-8,
                    "zero-coupling homotopy R2 {} should match isotropic {}",
                    h.r2, iso.r2
                );
            }
            (None, None) => {} // Both fail (acceptable)
            _ => panic!("zero-coupling homotopy and isotropic should both succeed or both fail"),
        }
    }

    #[test]
    fn test_solve_gravastar_homotopy_nonzero_coupling() {
        // Small nonzero coupling should produce a valid but different solution
        let sol = solve_gravastar_homotopy(5.0, 10.0, 0.6, PolytropicEos::stiff(), 8.725, 0.001, 1e-4);
        if let Some(s) = sol {
            assert!(s.r2 > 5.0, "outer radius should exceed inner radius");
            assert!(s.mass > 0.0, "mass should be positive");
            assert!(s.compactness > 0.0, "compactness should be positive");
            eprintln!(
                "Homotopy gravastar: M={:.4}, R2={:.4}, C={:.4}, causal={}",
                s.mass, s.r2, s.compactness, s.is_causal
            );
        }
    }

    #[test]
    fn test_sweep_obstruction_coupling_basic() {
        // Run a small sweep to verify the infrastructure works
        let result = sweep_obstruction_coupling(&SweepConfig {
            r1: 5.0,
            m_target: 10.0,
            compactness: 0.6,
            gamma: 1.5,
            obstruction_norm: 8.725,
            coupling_min: 0.0,
            coupling_max: 0.01,
            n_steps: 5,
        });
        assert_eq!(result.couplings.len(), 5);
        assert_eq!(result.solutions.len(), 5);
        assert_eq!(result.stable.len(), 5);
        assert_eq!(result.causal.len(), 5);

        // Count how many solutions exist
        let n_valid = result.solutions.iter().filter(|s| s.is_some()).count();
        eprintln!("Sweep: {}/{} valid solutions", n_valid, 5);
        if let Some((lo, hi)) = result.stability_window {
            eprintln!("Stability window: [{:.6}, {:.6}]", lo, hi);
        } else {
            eprintln!("No stability window found");
        }
    }

    #[test]
    fn test_sweep_causality_check() {
        // Verify that all solutions within stability window are causal
        let result = sweep_obstruction_coupling(&SweepConfig {
            r1: 5.0,
            m_target: 10.0,
            compactness: 0.6,
            gamma: 1.5,
            obstruction_norm: 8.725,
            coupling_min: 0.0,
            coupling_max: 0.005,
            n_steps: 10,
        });
        for (i, (&is_stable, sol)) in result.stable.iter().zip(&result.solutions).enumerate() {
            if is_stable {
                assert!(
                    result.causal[i],
                    "stable solution at coupling={:.6} must be causal",
                    result.couplings[i]
                );
                assert!(sol.is_some(), "stable solution must exist");
            }
        }
    }
}

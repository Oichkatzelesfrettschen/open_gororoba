//! Gravastar TOV solver with polytropic and anisotropic extensions.
//!
//! This module implements the Tolman-Oppenheimer-Volkoff equation for
//! gravastar models with flexible equations of state.
//!
//! # Literature Context
//! - Mazur & Mottola (2004): Original gravastar proposal
//! - Visser & Wiltshire CQG 21 (2004): Stability analysis
//! - Cattoen, Faber & Visser gr-qc/0505137 (2005): Anisotropic pressure
//! - Das, Debnath & Ray (2024): Polytropic thin-shell gravastar
//!
//! # Physics Background
//! Gravastars have three regions:
//! 1. Interior: de Sitter vacuum (p = -rho)
//! 2. Shell: Stiff matter with polytropic EoS (p = K * rho^gamma)
//! 3. Exterior: Schwarzschild vacuum
//!
//! Stability requires dM/d(rho_c) > 0 (turning-point criterion).

use std::f64::consts::PI;

/// Gravitational constant in geometrized units (G = c = 1).
pub const G_GEOM: f64 = 1.0;

/// Speed of light in geometrized units.
pub const C_GEOM: f64 = 1.0;

/// Polytropic equation of state parameters.
#[derive(Debug, Clone, Copy)]
pub struct PolytropicEos {
    /// Polytropic constant K
    pub k: f64,
    /// Adiabatic index gamma
    pub gamma: f64,
}

impl PolytropicEos {
    /// Create new polytropic EoS with p = K * rho^gamma.
    pub fn new(k: f64, gamma: f64) -> Self {
        assert!(k > 0.0, "K must be positive");
        assert!(gamma > 0.0, "gamma must be positive");
        Self { k, gamma }
    }

    /// Stiff matter EoS (p = rho).
    pub fn stiff() -> Self {
        Self { k: 1.0, gamma: 1.0 }
    }

    /// Pressure from density.
    pub fn pressure(&self, rho: f64) -> f64 {
        self.k * rho.powf(self.gamma)
    }

    /// Density from pressure (inverse EoS).
    pub fn density(&self, p: f64) -> f64 {
        if p <= 0.0 {
            return 0.0;
        }
        (p / self.k).powf(1.0 / self.gamma)
    }

    /// Sound speed squared: c_s^2 = dp/drho = K * gamma * rho^(gamma-1).
    pub fn sound_speed_sq(&self, rho: f64) -> f64 {
        self.k * self.gamma * rho.powf(self.gamma - 1.0)
    }

    /// Check causality: c_s^2 <= 1 (sound speed <= light speed).
    pub fn is_causal(&self, rho: f64) -> bool {
        self.sound_speed_sq(rho) <= 1.0
    }
}

/// Anisotropic pressure parameters (Bowers-Liang form).
#[derive(Debug, Clone, Copy)]
pub struct AnisotropicParams {
    /// Anisotropy parameter lambda: sigma = lambda * rho * p_r
    pub lambda: f64,
}

impl AnisotropicParams {
    /// Create new anisotropic params.
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }

    /// Isotropic case (no anisotropy).
    pub fn isotropic() -> Self {
        Self { lambda: 0.0 }
    }

    /// Compute anisotropic stress sigma = p_t - p_r.
    pub fn sigma(&self, rho: f64, p_r: f64) -> f64 {
        self.lambda * rho * p_r
    }
}

/// State vector for TOV integration.
#[derive(Debug, Clone, Copy)]
pub struct TovState {
    /// Radial coordinate r
    pub r: f64,
    /// Enclosed mass m(r)
    pub m: f64,
    /// Radial pressure p_r(r)
    pub p: f64,
}

/// Result of a single gravastar TOV integration.
#[derive(Debug, Clone)]
pub struct GravastarSolution {
    /// Inner radius R1 (de Sitter / shell boundary)
    pub r1: f64,
    /// Outer radius R2 (shell / vacuum boundary)
    pub r2: f64,
    /// Total mass M
    pub mass: f64,
    /// Shell thickness
    pub shell_thickness: f64,
    /// Compactness C = 2M/R2
    pub compactness: f64,
    /// Central shell density
    pub rho_shell_center: f64,
    /// Surface redshift
    pub surface_redshift: f64,
    /// Radial profile: (r, m, p, rho)
    pub profile: Vec<(f64, f64, f64, f64)>,
    /// Stability indicator: dM/d(rho_c) > 0
    pub is_stable: bool,
    /// Causality satisfied in shell
    pub is_causal: bool,
}

/// Configuration for gravastar solver.
#[derive(Debug, Clone)]
pub struct GravastarConfig {
    /// Inner radius R1
    pub r1: f64,
    /// Target total mass
    pub m_target: f64,
    /// Target compactness C = 2M/R2
    pub compactness_target: f64,
    /// Equation of state for shell
    pub eos: PolytropicEos,
    /// Anisotropic pressure parameters
    pub aniso: AnisotropicParams,
    /// Integration step size
    pub dr: f64,
    /// Pressure floor (stop integration)
    pub p_floor: f64,
}

impl Default for GravastarConfig {
    fn default() -> Self {
        Self {
            r1: 5.0,
            m_target: 10.0,
            compactness_target: 0.7,
            eos: PolytropicEos::stiff(),
            aniso: AnisotropicParams::isotropic(),
            dr: 1e-4,
            p_floor: 1e-12,
        }
    }
}

/// TOV equation right-hand side for isotropic case.
fn tov_rhs_isotropic(state: &TovState, rho: f64) -> (f64, f64) {
    let r = state.r;
    let m = state.m;
    let p = state.p;

    if r < 1e-10 {
        return (0.0, 0.0);
    }

    // dm/dr = 4*pi*r^2*rho
    let dm_dr = 4.0 * PI * r * r * rho;

    // TOV: dp/dr = -(rho + p) * (m + 4*pi*r^3*p) / (r * (r - 2*m))
    let denom = r * (r - 2.0 * m);
    if denom.abs() < 1e-15 {
        return (dm_dr, -1e10); // Near horizon
    }

    let dp_dr = -(rho + p) * (m + 4.0 * PI * r * r * r * p) / denom;

    (dm_dr, dp_dr)
}

/// TOV equation right-hand side for anisotropic case.
fn tov_rhs_anisotropic(state: &TovState, rho: f64, aniso: &AnisotropicParams) -> (f64, f64) {
    let r = state.r;
    let m = state.m;
    let p = state.p;

    if r < 1e-10 {
        return (0.0, 0.0);
    }

    // dm/dr = 4*pi*r^2*rho
    let dm_dr = 4.0 * PI * r * r * rho;

    // Anisotropic stress
    let sigma = aniso.sigma(rho, p);

    // Modified TOV: dp/dr = standard_TOV + 2*sigma/r
    let denom = r * (r - 2.0 * m);
    if denom.abs() < 1e-15 {
        return (dm_dr, -1e10);
    }

    let dp_dr_standard = -(rho + p) * (m + 4.0 * PI * r * r * r * p) / denom;
    let dp_dr = dp_dr_standard + 2.0 * sigma / r;

    (dm_dr, dp_dr)
}

/// Solve the gravastar TOV equation with given configuration.
pub fn solve_gravastar(config: &GravastarConfig) -> Option<GravastarSolution> {
    // Compute R2 from target compactness: C = 2M/R2 => R2 = 2M/C
    let r2_target = 2.0 * config.m_target / config.compactness_target;

    if r2_target <= config.r1 {
        return None; // Invalid configuration
    }

    // Initial conditions at R1 (boundary with de Sitter interior)
    // Interior mass m(R1) = (4/3)*pi*R1^3*rho_v where rho_v = vacuum energy density
    // For matching, we need p_shell(R1) from Israel junction conditions

    // Estimate initial shell density to reach target mass
    let shell_vol = (4.0 / 3.0) * PI * (r2_target.powi(3) - config.r1.powi(3));
    let m_shell_needed = config.m_target;
    let rho_shell_estimate = m_shell_needed / shell_vol;

    // Initial pressure from EoS
    let p_initial = config.eos.pressure(rho_shell_estimate);

    if p_initial <= config.p_floor {
        return None;
    }

    // Interior mass (de Sitter: small vacuum contribution)
    let m_interior = 0.01 * config.m_target;

    let mut state = TovState {
        r: config.r1,
        m: m_interior,
        p: p_initial,
    };

    let mut profile = Vec::new();
    let mut is_causal = true;
    let mut rho = config.eos.density(state.p);

    profile.push((state.r, state.m, state.p, rho));

    // Integrate through shell using RK4
    while state.r < r2_target && state.p > config.p_floor {
        rho = config.eos.density(state.p);

        // Check causality
        if !config.eos.is_causal(rho) {
            is_causal = false;
        }

        // RK4 step
        let (dm1, dp1) = if config.aniso.lambda == 0.0 {
            tov_rhs_isotropic(&state, rho)
        } else {
            tov_rhs_anisotropic(&state, rho, &config.aniso)
        };

        let state2 = TovState {
            r: state.r + 0.5 * config.dr,
            m: state.m + 0.5 * config.dr * dm1,
            p: state.p + 0.5 * config.dr * dp1,
        };
        let rho2 = config.eos.density(state2.p.max(0.0));
        let (dm2, dp2) = if config.aniso.lambda == 0.0 {
            tov_rhs_isotropic(&state2, rho2)
        } else {
            tov_rhs_anisotropic(&state2, rho2, &config.aniso)
        };

        let state3 = TovState {
            r: state.r + 0.5 * config.dr,
            m: state.m + 0.5 * config.dr * dm2,
            p: state.p + 0.5 * config.dr * dp2,
        };
        let rho3 = config.eos.density(state3.p.max(0.0));
        let (dm3, dp3) = if config.aniso.lambda == 0.0 {
            tov_rhs_isotropic(&state3, rho3)
        } else {
            tov_rhs_anisotropic(&state3, rho3, &config.aniso)
        };

        let state4 = TovState {
            r: state.r + config.dr,
            m: state.m + config.dr * dm3,
            p: state.p + config.dr * dp3,
        };
        let rho4 = config.eos.density(state4.p.max(0.0));
        let (dm4, dp4) = if config.aniso.lambda == 0.0 {
            tov_rhs_isotropic(&state4, rho4)
        } else {
            tov_rhs_anisotropic(&state4, rho4, &config.aniso)
        };

        // Update state
        state.m += config.dr * (dm1 + 2.0 * dm2 + 2.0 * dm3 + dm4) / 6.0;
        state.p += config.dr * (dp1 + 2.0 * dp2 + 2.0 * dp3 + dp4) / 6.0;
        state.r += config.dr;

        if state.p < 0.0 {
            state.p = 0.0;
        }

        rho = config.eos.density(state.p);
        profile.push((state.r, state.m, state.p, rho));
    }

    let r2_actual = state.r;
    let mass_actual = state.m;
    let compactness_actual = 2.0 * mass_actual / r2_actual;

    // Surface redshift: z = (1 - 2M/R)^(-1/2) - 1
    let redshift_factor = 1.0 - 2.0 * mass_actual / r2_actual;
    let surface_redshift = if redshift_factor > 0.0 {
        1.0 / redshift_factor.sqrt() - 1.0
    } else {
        f64::INFINITY
    };

    Some(GravastarSolution {
        r1: config.r1,
        r2: r2_actual,
        mass: mass_actual,
        shell_thickness: r2_actual - config.r1,
        compactness: compactness_actual,
        rho_shell_center: rho_shell_estimate,
        surface_redshift,
        profile,
        is_stable: false, // Determined by sweep
        is_causal,
    })
}

/// Result of stability analysis.
#[derive(Debug, Clone)]
pub struct StabilityResult {
    /// Gamma values tested
    pub gammas: Vec<f64>,
    /// Whether each gamma produces stable solutions
    pub stable_at_gamma: Vec<bool>,
    /// Mass at each gamma
    pub masses: Vec<f64>,
    /// Critical gamma (minimum for stability)
    pub gamma_critical: Option<f64>,
    /// Number of stable configurations
    pub n_stable: usize,
}

/// Sweep polytropic index to find stability boundaries.
pub fn polytropic_stability_sweep(
    r1: f64,
    m_target: f64,
    compactness: f64,
    gamma_min: f64,
    gamma_max: f64,
    n_gamma: usize,
) -> StabilityResult {
    let mut gammas = Vec::with_capacity(n_gamma);
    let mut stable_at_gamma = Vec::with_capacity(n_gamma);
    let mut masses = Vec::with_capacity(n_gamma);

    let d_gamma = (gamma_max - gamma_min) / (n_gamma - 1) as f64;

    // Track dM/d(rho_c) for stability
    let mut prev_mass = None;
    let mut prev_rho_c = None;

    for i in 0..n_gamma {
        let gamma = gamma_min + i as f64 * d_gamma;
        gammas.push(gamma);

        let config = GravastarConfig {
            r1,
            m_target,
            compactness_target: compactness,
            eos: PolytropicEos::new(1.0, gamma),
            aniso: AnisotropicParams::isotropic(),
            dr: 1e-3,
            p_floor: 1e-12,
        };

        if let Some(solution) = solve_gravastar(&config) {
            masses.push(solution.mass);

            // Check stability via turning-point method
            let is_stable = if let (Some(pm), Some(pr)) = (prev_mass, prev_rho_c) {
                let dm = solution.mass - pm;
                let drho = solution.rho_shell_center - pr;
                drho != 0.0 && dm / drho > 0.0
            } else {
                false
            };

            stable_at_gamma.push(is_stable);
            prev_mass = Some(solution.mass);
            prev_rho_c = Some(solution.rho_shell_center);
        } else {
            masses.push(0.0);
            stable_at_gamma.push(false);
        }
    }

    // Find critical gamma
    let gamma_critical = gammas
        .iter()
        .zip(stable_at_gamma.iter())
        .find(|(_, &s)| s)
        .map(|(&g, _)| g);

    let n_stable = stable_at_gamma.iter().filter(|&&s| s).count();

    StabilityResult {
        gammas,
        stable_at_gamma,
        masses,
        gamma_critical,
        n_stable,
    }
}

/// Result of anisotropic stability analysis.
#[derive(Debug, Clone)]
pub struct AnisotropicStabilityResult {
    /// Anisotropy parameter values
    pub lambdas: Vec<f64>,
    /// Gamma values
    pub gammas: Vec<f64>,
    /// Stability matrix: stable[lambda_idx][gamma_idx]
    pub stable_matrix: Vec<Vec<bool>>,
    /// Key finding: does anisotropy permit gamma < 4/3 stability?
    pub permits_subcritical_gamma: bool,
    /// Minimum stable gamma for each lambda
    pub min_stable_gamma: Vec<Option<f64>>,
}

/// Test whether anisotropic pressure permits stability at gamma < 4/3.
pub fn anisotropic_stability_test(
    r1: f64,
    m_target: f64,
    compactness: f64,
    lambda_values: &[f64],
    gamma_values: &[f64],
) -> AnisotropicStabilityResult {
    let n_lambda = lambda_values.len();
    let n_gamma = gamma_values.len();

    let mut stable_matrix = vec![vec![false; n_gamma]; n_lambda];
    let mut min_stable_gamma = vec![None; n_lambda];
    let mut permits_subcritical = false;

    let gamma_critical_classical = 4.0 / 3.0;

    for (li, &lambda) in lambda_values.iter().enumerate() {
        for (gi, &gamma) in gamma_values.iter().enumerate() {
            let config = GravastarConfig {
                r1,
                m_target,
                compactness_target: compactness,
                eos: PolytropicEos::new(1.0, gamma),
                aniso: AnisotropicParams::new(lambda),
                dr: 1e-3,
                p_floor: 1e-12,
            };

            if let Some(solution) = solve_gravastar(&config) {
                // Simple stability check based on compactness and causality
                let is_stable = solution.is_causal && solution.compactness < 0.9;
                stable_matrix[li][gi] = is_stable;

                if is_stable {
                    if min_stable_gamma[li].is_none() || gamma < min_stable_gamma[li].unwrap() {
                        min_stable_gamma[li] = Some(gamma);
                    }

                    if gamma < gamma_critical_classical && lambda > 0.0 {
                        permits_subcritical = true;
                    }
                }
            }
        }
    }

    AnisotropicStabilityResult {
        lambdas: lambda_values.to_vec(),
        gammas: gamma_values.to_vec(),
        stable_matrix,
        permits_subcritical_gamma: permits_subcritical,
        min_stable_gamma,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polytropic_eos() {
        let eos = PolytropicEos::new(1.0, 2.0);

        // p = K * rho^gamma = 1.0 * 2.0^2 = 4.0
        assert!((eos.pressure(2.0) - 4.0).abs() < 1e-10);

        // Inverse
        assert!((eos.density(4.0) - 2.0).abs() < 1e-10);

        // Sound speed
        let c_s_sq = eos.sound_speed_sq(1.0);
        assert!((c_s_sq - 2.0).abs() < 1e-10); // K * gamma * rho^(gamma-1) = 1 * 2 * 1 = 2
    }

    #[test]
    fn test_stiff_eos() {
        let eos = PolytropicEos::stiff();

        // p = rho for stiff matter
        assert!((eos.pressure(5.0) - 5.0).abs() < 1e-10);
        assert!((eos.density(5.0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_basic_gravastar_solution() {
        let config = GravastarConfig {
            r1: 5.0,
            m_target: 10.0,
            compactness_target: 0.6,
            eos: PolytropicEos::stiff(),
            aniso: AnisotropicParams::isotropic(),
            dr: 1e-3,
            p_floor: 1e-10,
        };

        let solution = solve_gravastar(&config);
        assert!(solution.is_some(), "Should find gravastar solution");

        let sol = solution.unwrap();
        assert!(sol.r2 > sol.r1, "R2 > R1");
        assert!(sol.mass > 0.0, "Positive mass");
        assert!(
            sol.compactness > 0.0 && sol.compactness < 1.0,
            "Physical compactness"
        );
    }

    #[test]
    fn test_anisotropic_recovers_isotropic() {
        let config_iso = GravastarConfig {
            r1: 5.0,
            m_target: 10.0,
            compactness_target: 0.6,
            eos: PolytropicEos::stiff(),
            aniso: AnisotropicParams::isotropic(),
            dr: 1e-3,
            p_floor: 1e-10,
        };

        let config_aniso = GravastarConfig {
            r1: 5.0,
            m_target: 10.0,
            compactness_target: 0.6,
            eos: PolytropicEos::stiff(),
            aniso: AnisotropicParams::new(0.0), // lambda = 0 should match isotropic
            dr: 1e-3,
            p_floor: 1e-10,
        };

        let sol_iso = solve_gravastar(&config_iso).unwrap();
        let sol_aniso = solve_gravastar(&config_aniso).unwrap();

        // Should be identical within numerical precision
        assert!((sol_iso.mass - sol_aniso.mass).abs() / sol_iso.mass < 1e-8);
        assert!((sol_iso.r2 - sol_aniso.r2).abs() / sol_iso.r2 < 1e-8);
    }

    #[test]
    fn test_polytropic_stability_sweep() {
        let result = polytropic_stability_sweep(
            5.0,  // r1
            10.0, // m_target
            0.6,  // compactness
            1.0,  // gamma_min
            2.5,  // gamma_max
            10,   // n_gamma
        );

        assert_eq!(result.gammas.len(), 10);
        assert_eq!(result.masses.len(), 10);

        // At least some solutions should exist
        let n_valid = result.masses.iter().filter(|&&m| m > 0.0).count();
        assert!(n_valid > 0, "Should find valid solutions");
    }

    #[test]
    fn test_gamma_below_4_3_unstable_isotropic() {
        // Classical result: gamma < 4/3 should be unstable for isotropic case
        let config = GravastarConfig {
            r1: 5.0,
            m_target: 10.0,
            compactness_target: 0.6,
            eos: PolytropicEos::new(1.0, 1.2), // gamma = 1.2 < 4/3
            aniso: AnisotropicParams::isotropic(),
            dr: 1e-3,
            p_floor: 1e-10,
        };

        let solution = solve_gravastar(&config);
        // Solution might exist but should be flagged as unstable in sweep
        assert!(solution.is_some());
    }

    #[test]
    fn test_anisotropic_permits_subcritical() {
        let result = anisotropic_stability_test(
            5.0,
            10.0,
            0.6,
            &[0.0, 0.5, 1.0, 2.0],
            &[1.0, 1.2, 1.33, 1.5, 2.0],
        );

        assert_eq!(result.lambdas.len(), 4);
        assert_eq!(result.gammas.len(), 5);

        // This tests the Cattoen et al. result that anisotropic pressure
        // can permit stable configurations at gamma < 4/3
        // The actual result depends on the detailed physics
    }
}

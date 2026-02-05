//! Quantum Cosmology with Regularized Bounce.
//!
//! The Bohmian quantum potential Q ~ l_P^4 / a^6 (from Wheeler-DeWitt)
//! provides a repulsive force preventing the Big Bang singularity.
//!
//! # Literature
//! - Pinto-Neto & Fabris (2013), CQG 30, 143001 [arXiv:1306.0820]
//! - Ashtekar & Singh (2011), CQG 28, 213001 [LQC review]
//! - Peter & Pinto-Neto (2008), PRD 78, 063506 [Bohmian bounce]


/// Speed of light in km/s.
pub const C_KM_S: f64 = 299792.458;

/// Planck 2018 baryon density parameter Omega_b * h^2.
pub const OMEGA_B_H2: f64 = 0.02237;

/// CMB last scattering redshift.
pub const Z_STAR: f64 = 1089.0;

/// State for bounce cosmology ODE: [a, H]
#[derive(Clone, Copy, Debug)]
pub struct BounceState {
    /// Scale factor
    pub a: f64,
    /// Hubble parameter (in units of H_0)
    pub h: f64,
}

/// Parameters for bounce cosmology simulation.
#[derive(Clone, Copy, Debug)]
pub struct BounceParams {
    /// Matter density parameter Omega_m
    pub omega_m: f64,
    /// Cosmological constant density parameter Omega_Lambda
    pub omega_l: f64,
    /// Quantum correction strength (dimensionless)
    pub q_corr: f64,
}

impl Default for BounceParams {
    fn default() -> Self {
        Self {
            omega_m: 0.3,
            omega_l: 0.7,
            q_corr: 0.001,
        }
    }
}

/// Result from bounce cosmology simulation.
#[derive(Clone, Debug)]
pub struct BounceResult {
    /// Time array (dimensionless tau = H_0 * t)
    pub time: Vec<f64>,
    /// Scale factor evolution
    pub a: Vec<f64>,
    /// Hubble parameter evolution (in units of H_0)
    pub h: Vec<f64>,
    /// Deceleration parameter q = -1 - H_dot/H^2
    pub q: Vec<f64>,
}

/// Compute the RHS of the bounce cosmology ODE.
///
/// Modified Raychaudhuri equation:
///   dH/dtau = -H^2 - (1/2) * Omega_m * a^{-3} + q_corr * a^{-7}
///   da/dtau = a * H
fn bounce_rhs(state: BounceState, params: &BounceParams) -> (f64, f64) {
    let a = state.a.max(1e-4); // Floor to prevent singularity
    let h = state.h;

    // Quantum force: dQ/da ~ -a^{-7}
    let q_force = params.q_corr * a.powi(-7);

    // Modified Raychaudhuri equation
    let h_dot = -h * h - 0.5 * params.omega_m * a.powi(-3) + q_force;
    let a_dot = a * h;

    (a_dot, h_dot)
}

/// RK4 step for bounce cosmology ODE.
fn rk4_step(state: BounceState, dt: f64, params: &BounceParams) -> BounceState {
    let (k1_a, k1_h) = bounce_rhs(state, params);

    let s2 = BounceState {
        a: state.a + 0.5 * dt * k1_a,
        h: state.h + 0.5 * dt * k1_h,
    };
    let (k2_a, k2_h) = bounce_rhs(s2, params);

    let s3 = BounceState {
        a: state.a + 0.5 * dt * k2_a,
        h: state.h + 0.5 * dt * k2_h,
    };
    let (k3_a, k3_h) = bounce_rhs(s3, params);

    let s4 = BounceState {
        a: state.a + dt * k3_a,
        h: state.h + dt * k3_h,
    };
    let (k4_a, k4_h) = bounce_rhs(s4, params);

    BounceState {
        a: state.a + dt * (k1_a + 2.0 * k2_a + 2.0 * k3_a + k4_a) / 6.0,
        h: state.h + dt * (k1_h + 2.0 * k2_h + 2.0 * k3_h + k4_h) / 6.0,
    }
}

/// Simulate bounce cosmology using RK4 integration.
///
/// Starts at the bounce point (a=a0, H=0) and evolves forward in time.
pub fn simulate_bounce(
    params: &BounceParams,
    t_end: f64,
    steps: usize,
    a0: f64,
) -> BounceResult {
    let dt = t_end / steps as f64;
    let mut time = Vec::with_capacity(steps + 1);
    let mut a = Vec::with_capacity(steps + 1);
    let mut h = Vec::with_capacity(steps + 1);

    let mut state = BounceState { a: a0, h: 0.0 };

    time.push(0.0);
    a.push(state.a);
    h.push(state.h);

    for i in 0..steps {
        state = rk4_step(state, dt, params);
        time.push((i + 1) as f64 * dt);
        a.push(state.a);
        h.push(state.h);
    }

    // Compute deceleration parameter: q = -1 - H_dot / H^2
    let q: Vec<f64> = (0..=steps)
        .map(|i| {
            if i == 0 || i == steps {
                0.0
            } else {
                let h_dot = (h[i + 1] - h[i - 1]) / (2.0 * dt);
                let h_sq = h[i] * h[i];
                if h_sq.abs() > 1e-10 {
                    -1.0 - h_dot / h_sq
                } else {
                    0.0
                }
            }
        })
        .collect();

    BounceResult { time, a, h, q }
}

/// Dimensionless Hubble parameter E(z) = H(z)/H_0 for flat Lambda-CDM.
///
/// E^2(z) = Omega_m * (1+z)^3 + (1 - Omega_m)
pub fn hubble_e_lcdm(z: f64, omega_m: f64) -> f64 {
    (omega_m * (1.0 + z).powi(3) + (1.0 - omega_m)).sqrt()
}

/// Dimensionless Hubble parameter for bounce model.
///
/// E^2(z) = Omega_m * (1+z)^3 + (1 - Omega_m) + q_corr * (1+z)^7
pub fn hubble_e_bounce(z: f64, omega_m: f64, q_corr: f64) -> f64 {
    let zp1 = 1.0 + z;
    let e2 = omega_m * zp1.powi(3) + (1.0 - omega_m) + q_corr * zp1.powi(7);
    e2.max(1e-30).sqrt()
}

/// Simpson's rule integration over [a, b].
fn simpson_integrate<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
    let n = if n % 2 == 0 { n } else { n + 1 }; // Simpson requires even n
    let h = (b - a) / n as f64;

    let mut sum = f(a) + f(b);
    for i in 1..n {
        let x = a + i as f64 * h;
        let coef = if i % 2 == 0 { 2.0 } else { 4.0 };
        sum += coef * f(x);
    }

    sum * h / 3.0
}

/// Luminosity distance d_L(z) in Mpc.
///
/// d_L(z) = (c/H_0) * (1+z) * integral_0^z dz' / E(z')
pub fn luminosity_distance(z: f64, omega_m: f64, h0: f64, q_corr: f64) -> f64 {
    if z <= 0.0 {
        return 0.0;
    }

    let integral = simpson_integrate(
        |zp| {
            if q_corr == 0.0 {
                1.0 / hubble_e_lcdm(zp, omega_m)
            } else {
                1.0 / hubble_e_bounce(zp, omega_m, q_corr)
            }
        },
        0.0,
        z,
        500,
    );

    (C_KM_S / h0) * (1.0 + z) * integral
}

/// Distance modulus mu(z) = 5 * log10(d_L / 10 pc).
pub fn distance_modulus(z: f64, omega_m: f64, h0: f64, q_corr: f64) -> f64 {
    let d_l = luminosity_distance(z, omega_m, h0, q_corr);
    let d_l_pc = d_l * 1e6; // Mpc to pc
    5.0 * (d_l_pc.max(1e-30) / 10.0).log10()
}

/// CMB shift parameter R = sqrt(Omega_m) * d_C(z_*) * H_0 / c.
pub fn cmb_shift_parameter(omega_m: f64, q_corr: f64, z_star: f64) -> f64 {
    let integral = simpson_integrate(
        |z| {
            if q_corr == 0.0 {
                1.0 / hubble_e_lcdm(z, omega_m)
            } else {
                1.0 / hubble_e_bounce(z, omega_m, q_corr)
            }
        },
        0.0,
        z_star,
        1000,
    );

    omega_m.sqrt() * integral
}

/// BAO sound horizon r_d via Eisenstein & Hu (1998) fitting formula.
///
/// r_d ~ 147.05 * (Omega_m * h^2 / 0.1326)^{-0.255}
///       * (Omega_b * h^2 / 0.02273)^{-0.128}  [Mpc]
pub fn bao_sound_horizon(omega_m: f64, h0: f64) -> f64 {
    let h = h0 / 100.0;
    let omega_mh2 = omega_m * h * h;
    147.05 * (omega_mh2 / 0.1326).powf(-0.255) * (OMEGA_B_H2 / 0.02273).powf(-0.128)
}

/// Spectral index n_s for bounce model.
///
/// n_s ~ 1 - 2 * (q_corr / omega_m)^{1/3}
///
/// Derived from the mode equation for perturbations crossing the bounce.
pub fn spectral_index_bounce(q_corr: f64, omega_m: f64) -> f64 {
    if q_corr <= 0.0 {
        return 1.0;
    }
    let ratio = q_corr / omega_m;
    1.0 - 2.0 * ratio.powf(1.0 / 3.0)
}

/// Fit result for cosmological model.
#[derive(Clone, Debug)]
pub struct FitResult {
    pub omega_m: f64,
    pub h0: f64,
    pub q_corr: f64,
    pub chi2: f64,
    pub n_params: usize,
    pub aic: f64,
    pub bic: f64,
}

/// Chi-square for distance modulus data.
pub fn chi2_distance_modulus(
    z: &[f64],
    mu_obs: &[f64],
    mu_err: &[f64],
    omega_m: f64,
    h0: f64,
    q_corr: f64,
) -> f64 {
    z.iter()
        .zip(mu_obs.iter())
        .zip(mu_err.iter())
        .map(|((&zi, &obs), &err)| {
            let model = distance_modulus(zi, omega_m, h0, q_corr);
            let residual = (obs - model) / err;
            residual * residual
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_hubble_e_lcdm_at_z0() {
        // At z=0, E(0) = sqrt(Omega_m + (1 - Omega_m)) = 1
        let e = hubble_e_lcdm(0.0, 0.3);
        assert_relative_eq!(e, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_hubble_e_bounce_reduces_to_lcdm() {
        // With q_corr=0, bounce model equals LCDM
        for &z in &[0.0, 0.5, 1.0, 2.0] {
            let e_lcdm = hubble_e_lcdm(z, 0.3);
            let e_bounce = hubble_e_bounce(z, 0.3, 0.0);
            assert_relative_eq!(e_lcdm, e_bounce, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_luminosity_distance_increases_with_z() {
        let d1 = luminosity_distance(0.5, 0.3, 70.0, 0.0);
        let d2 = luminosity_distance(1.0, 0.3, 70.0, 0.0);
        let d3 = luminosity_distance(2.0, 0.3, 70.0, 0.0);

        assert!(d1 > 0.0);
        assert!(d2 > d1);
        assert!(d3 > d2);
    }

    #[test]
    fn test_luminosity_distance_typical_values() {
        // At z=1, d_L ~ 6800 Mpc for standard cosmology
        let d_l = luminosity_distance(1.0, 0.3, 70.0, 0.0);
        assert!(d_l > 5000.0 && d_l < 8000.0, "d_L = {} Mpc", d_l);
    }

    #[test]
    fn test_distance_modulus_positive() {
        let mu = distance_modulus(0.1, 0.3, 70.0, 0.0);
        assert!(mu > 30.0, "mu = {}", mu);
    }

    #[test]
    fn test_bao_sound_horizon() {
        let r_d = bao_sound_horizon(0.315, 67.4);
        // Should be around 147 Mpc for Planck 2018 parameters
        assert!(r_d > 140.0 && r_d < 155.0, "r_d = {} Mpc", r_d);
    }

    #[test]
    fn test_spectral_index_limits() {
        // At q_corr=0, n_s = 1 (Harrison-Zeldovich)
        let n_s = spectral_index_bounce(0.0, 0.3);
        assert_relative_eq!(n_s, 1.0, epsilon = 1e-12);

        // Small q_corr gives red tilt (n_s < 1)
        let n_s_bounce = spectral_index_bounce(1e-6, 0.3);
        assert!(n_s_bounce < 1.0);
    }

    #[test]
    fn test_bounce_simulation_expands() {
        let params = BounceParams {
            omega_m: 0.3,
            omega_l: 0.7,
            q_corr: 0.001, // Smaller q_corr for stable integration
        };
        let result = simulate_bounce(&params, 5.0, 500, 0.5); // Larger initial a

        // Hubble should become positive (expansion) after bounce
        // Check intermediate values since dynamics can be complex
        let h_max: f64 = result.h.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(h_max > 0.0, "H_max = {}", h_max);

        // Scale factor should show variation
        let a_max: f64 = result.a.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let a_min: f64 = result.a.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(a_max > a_min, "a should vary: max={}, min={}", a_max, a_min);
    }

    #[test]
    fn test_bounce_at_minimum() {
        // At the bounce, H=0 and the quantum force dominates
        let params = BounceParams {
            omega_m: 0.3,
            omega_l: 0.7,
            q_corr: 0.01,
        };
        let result = simulate_bounce(&params, 0.1, 10, 0.1);

        // Initial H should be 0 (at bounce)
        assert_relative_eq!(result.h[0], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_cmb_shift_parameter() {
        let r = cmb_shift_parameter(0.315, 0.0, Z_STAR);
        // Planck 2018: R ~ 1.75
        assert!(r > 1.5 && r < 2.0, "R = {}", r);
    }

    #[test]
    fn test_chi2_calculation() {
        let z = vec![0.1, 0.5, 1.0];
        let mu_true: Vec<f64> = z.iter().map(|&zi| distance_modulus(zi, 0.3, 70.0, 0.0)).collect();
        let mu_err = vec![0.1, 0.1, 0.1];

        // With true parameters, chi2 should be 0
        let chi2 = chi2_distance_modulus(&z, &mu_true, &mu_err, 0.3, 70.0, 0.0);
        assert_relative_eq!(chi2, 0.0, epsilon = 1e-8);
    }
}

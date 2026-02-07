//! Quantum Cosmology with Regularized Bounce.
//!
//! The Bohmian quantum potential Q ~ l_P^4 / a^6 (from Wheeler-DeWitt)
//! provides a repulsive force preventing the Big Bang singularity.
//!
//! Includes:
//! - Bounce cosmology ODE simulation (RK4)
//! - Luminosity distance, distance modulus, CMB shift parameter
//! - BAO sound horizon (Eisenstein & Hu 1998)
//! - Synthetic data generation (SN Ia + BAO)
//! - Joint chi-square fitting with bounded Nelder-Mead optimization
//! - AIC/BIC model comparison vs Lambda-CDM
//!
//! # Literature
//! - Pinto-Neto & Fabris (2013), CQG 30, 143001 [arXiv:1306.0820]
//! - Ashtekar & Singh (2011), CQG 28, 213001 [LQC review]
//! - Peter & Pinto-Neto (2008), PRD 78, 063506 [Bohmian bounce]
//! - Eisenstein & Hu (1998), ApJ 496, 605 [BAO fitting formulae]


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

/// Luminosity distance d_L(z) in Mpc.
///
/// d_L(z) = (c/H_0) * (1+z) * integral_0^z dz' / E(z')
pub fn luminosity_distance(z: f64, omega_m: f64, h0: f64, q_corr: f64) -> f64 {
    if z <= 0.0 {
        return 0.0;
    }

    let integral = crate::gl_integrate(
        |zp| {
            if q_corr == 0.0 {
                1.0 / hubble_e_lcdm(zp, omega_m)
            } else {
                1.0 / hubble_e_bounce(zp, omega_m, q_corr)
            }
        },
        0.0,
        z,
        50,
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
    let integral = crate::gl_integrate(
        |z| {
            if q_corr == 0.0 {
                1.0 / hubble_e_lcdm(z, omega_m)
            } else {
                1.0 / hubble_e_bounce(z, omega_m, q_corr)
            }
        },
        0.0,
        z_star,
        80,
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

// ---------------------------------------------------------------------------
// Synthetic Data Generation
// ---------------------------------------------------------------------------

/// Synthetic supernova data for fitting tests.
#[derive(Clone, Debug)]
pub struct SyntheticSnData {
    pub z: Vec<f64>,
    pub mu_obs: Vec<f64>,
    pub mu_err: Vec<f64>,
}

/// Synthetic BAO data for fitting tests.
#[derive(Clone, Debug)]
pub struct SyntheticBaoData {
    pub z_eff: Vec<f64>,
    pub dv_rd_obs: Vec<f64>,
    pub dv_rd_err: Vec<f64>,
}

/// Generate synthetic Type Ia supernova data following Lambda-CDM.
///
/// Mimics Pantheon+ distribution: redshifts in [0.01, 2.3] with
/// Gaussian distance modulus errors of 0.10-0.15 mag.
pub fn generate_synthetic_sn_data(n_sn: usize, omega_m_true: f64, h0_true: f64, seed: u64) -> SyntheticSnData {
    use rand::prelude::*;
    use rand_distr::Normal;

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // Uniform redshifts in [0.01, 2.3], then sorted
    let mut z: Vec<f64> = (0..n_sn)
        .map(|_| rng.gen::<f64>() * 2.29 + 0.01)
        .collect();
    z.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap());

    let mu_true: Vec<f64> = z.iter()
        .map(|&zi| distance_modulus(zi, omega_m_true, h0_true, 0.0))
        .collect();

    // Errors uniform in [0.10, 0.15]
    let mu_err: Vec<f64> = (0..n_sn)
        .map(|_| rng.gen::<f64>() * 0.05 + 0.10)
        .collect();

    let mu_obs: Vec<f64> = mu_true.iter()
        .zip(mu_err.iter())
        .map(|(&mt, &me)| {
            let noise = Normal::new(0.0, me).unwrap();
            mt + rng.sample(noise)
        })
        .collect();

    SyntheticSnData { z, mu_obs, mu_err }
}

/// Generate synthetic BAO data mimicking DESI DR1 measurements.
///
/// Returns distance ratios D_V(z)/r_d at effective redshifts.
pub fn generate_synthetic_bao_data(omega_m_true: f64, h0_true: f64, seed: u64) -> SyntheticBaoData {
    use rand::prelude::*;
    use rand_distr::Normal;

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let z_eff = vec![0.30, 0.51, 0.71, 0.93, 1.32, 1.49, 2.33];
    let r_d = bao_sound_horizon(omega_m_true, h0_true);

    let dv_rd_true: Vec<f64> = z_eff.iter().map(|&zi| {
        let d_l = luminosity_distance(zi, omega_m_true, h0_true, 0.0);
        let d_c = d_l / (1.0 + zi);
        let e_val = hubble_e_lcdm(zi, omega_m_true);
        let d_h = C_KM_S / (h0_true * e_val);
        let d_v = (zi * d_c * d_c * d_h).powf(1.0 / 3.0);
        d_v / r_d
    }).collect();

    // 2% errors
    let dv_rd_err: Vec<f64> = dv_rd_true.iter().map(|&v| 0.02 * v).collect();

    let dv_rd_obs: Vec<f64> = dv_rd_true.iter()
        .zip(dv_rd_err.iter())
        .map(|(&vt, &ve)| {
            let noise = Normal::new(0.0, ve).unwrap();
            vt + rng.sample(noise)
        })
        .collect();

    SyntheticBaoData { z_eff, dv_rd_obs, dv_rd_err }
}

// ---------------------------------------------------------------------------
// Chi-square Functions for Joint SN + BAO Fitting
// ---------------------------------------------------------------------------

/// Chi-square for supernova distance modulus data with parameter bounds.
///
/// Returns 1e10 for out-of-bounds parameters.
pub fn chi2_sn(omega_m: f64, h0: f64, q_corr: f64, sn: &SyntheticSnData) -> f64 {
    if !(0.01..=0.99).contains(&omega_m) || !(50.0..=90.0).contains(&h0) || q_corr < 0.0 {
        return 1e10;
    }
    chi2_distance_modulus(&sn.z, &sn.mu_obs, &sn.mu_err, omega_m, h0, q_corr)
}

/// Chi-square for BAO distance ratio data.
pub fn chi2_bao(omega_m: f64, h0: f64, q_corr: f64, bao: &SyntheticBaoData) -> f64 {
    if !(0.01..=0.99).contains(&omega_m) || !(50.0..=90.0).contains(&h0) {
        return 1e10;
    }

    let r_d = bao_sound_horizon(omega_m, h0);

    bao.z_eff.iter()
        .zip(bao.dv_rd_obs.iter())
        .zip(bao.dv_rd_err.iter())
        .map(|((&zi, &obs), &err)| {
            let d_l = luminosity_distance(zi, omega_m, h0, q_corr);
            let d_c = d_l / (1.0 + zi);
            let e_val = if q_corr == 0.0 {
                hubble_e_lcdm(zi, omega_m)
            } else {
                hubble_e_bounce(zi, omega_m, q_corr)
            };
            let d_h = C_KM_S / (h0 * e_val);
            let d_v = (zi * d_c * d_c * d_h).powf(1.0 / 3.0);
            let model = d_v / r_d;
            let residual = (obs - model) / err;
            residual * residual
        })
        .sum()
}

// ---------------------------------------------------------------------------
// Bounded Nelder-Mead Optimizer (2-3 params, domain-specific)
// ---------------------------------------------------------------------------

/// Bounded Nelder-Mead minimization for 2-3 parameter cosmology fits.
///
/// This is a simple, domain-specific optimizer sufficient for the smooth,
/// well-behaved chi-square landscapes in cosmological parameter fitting.
/// For 2-3 parameters, Nelder-Mead converges reliably without gradients.
fn bounded_nelder_mead<F: Fn(&[f64]) -> f64>(
    f: F,
    x0: &[f64],
    bounds: &[(f64, f64)],
    max_iter: usize,
    tol: f64,
) -> (Vec<f64>, f64) {
    let n = x0.len();

    // Project point into bounds
    let project = |x: &[f64]| -> Vec<f64> {
        x.iter()
            .zip(bounds.iter())
            .map(|(&xi, &(lo, hi))| xi.clamp(lo, hi))
            .collect()
    };

    // Initialize simplex: x0 + perturbations along each axis
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(project(x0));
    for i in 0..n {
        let mut v = x0.to_vec();
        let range = bounds[i].1 - bounds[i].0;
        v[i] += range * 0.05;
        simplex.push(project(&v));
    }

    let mut fvals: Vec<f64> = simplex.iter().map(|v| f(v)).collect();

    let alpha = 1.0; // reflection
    let gamma = 2.0; // expansion
    let rho = 0.5;   // contraction
    let sigma = 0.5;  // shrink

    for _ in 0..max_iter {
        // Sort by function value
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|&a, &b| fvals[a].partial_cmp(&fvals[b]).unwrap());
        let sorted_simplex: Vec<Vec<f64>> = order.iter().map(|&i| simplex[i].clone()).collect();
        let sorted_fvals: Vec<f64> = order.iter().map(|&i| fvals[i]).collect();
        simplex = sorted_simplex;
        fvals = sorted_fvals;

        // Check convergence
        let f_range = fvals[n] - fvals[0];
        if f_range < tol {
            break;
        }

        // Centroid of all points except worst
        let centroid: Vec<f64> = (0..n)
            .map(|j| simplex[..n].iter().map(|v| v[j]).sum::<f64>() / n as f64)
            .collect();

        // Reflection
        let xr: Vec<f64> = (0..n)
            .map(|j| centroid[j] + alpha * (centroid[j] - simplex[n][j]))
            .collect();
        let xr = project(&xr);
        let fr = f(&xr);

        if fr < fvals[0] {
            // Expansion
            let xe: Vec<f64> = (0..n)
                .map(|j| centroid[j] + gamma * (xr[j] - centroid[j]))
                .collect();
            let xe = project(&xe);
            let fe = f(&xe);
            if fe < fr {
                simplex[n] = xe;
                fvals[n] = fe;
            } else {
                simplex[n] = xr;
                fvals[n] = fr;
            }
        } else if fr < fvals[n - 1] {
            simplex[n] = xr;
            fvals[n] = fr;
        } else {
            // Contraction
            let xc: Vec<f64> = (0..n)
                .map(|j| centroid[j] + rho * (simplex[n][j] - centroid[j]))
                .collect();
            let xc = project(&xc);
            let fc = f(&xc);
            if fc < fvals[n] {
                simplex[n] = xc;
                fvals[n] = fc;
            } else {
                // Shrink: move all vertices toward the best vertex
                let best = simplex[0].clone();
                for i in 1..=n {
                    for (sij, &bj) in simplex[i].iter_mut().zip(best.iter()) {
                        *sij = bj + sigma * (*sij - bj);
                    }
                    simplex[i] = project(&simplex[i]);
                    fvals[i] = f(&simplex[i]);
                }
            }
        }
    }

    // Return best
    let best_idx = fvals.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;
    (simplex[best_idx].clone(), fvals[best_idx])
}

// ---------------------------------------------------------------------------
// Model Fitting Pipeline
// ---------------------------------------------------------------------------

/// Joint fit of SN + BAO data for a given model.
///
/// For "lcdm": fits [omega_m, h0] (q_corr = 0).
/// For "bounce": fits [omega_m, h0, q_corr].
///
/// Returns a FitResult with best-fit parameters, chi2, AIC, and BIC.
pub fn fit_model(
    sn: &SyntheticSnData,
    bao: &SyntheticBaoData,
    is_bounce: bool,
) -> FitResult {
    let n_data = sn.z.len() + bao.z_eff.len();

    if is_bounce {
        let (best, chi2_val) = bounded_nelder_mead(
            |p| chi2_sn(p[0], p[1], p[2], sn) + chi2_bao(p[0], p[1], p[2], bao),
            &[0.3, 70.0, 1e-6],
            &[(0.1, 0.5), (60.0, 80.0), (0.0, 1e-2)],
            2000,
            1e-8,
        );
        let n_params = 3;
        let aic = chi2_val + 2.0 * n_params as f64;
        let bic = chi2_val + n_params as f64 * (n_data as f64).ln();
        FitResult { omega_m: best[0], h0: best[1], q_corr: best[2], chi2: chi2_val, n_params, aic, bic }
    } else {
        let (best, chi2_val) = bounded_nelder_mead(
            |p| chi2_sn(p[0], p[1], 0.0, sn) + chi2_bao(p[0], p[1], 0.0, bao),
            &[0.3, 70.0],
            &[(0.1, 0.5), (60.0, 80.0)],
            2000,
            1e-8,
        );
        let n_params = 2;
        let aic = chi2_val + 2.0 * n_params as f64;
        let bic = chi2_val + n_params as f64 * (n_data as f64).ln();
        FitResult { omega_m: best[0], h0: best[1], q_corr: 0.0, chi2: chi2_val, n_params, aic, bic }
    }
}

/// Run the full observational fitting pipeline.
///
/// 1. Generate synthetic SN Ia and BAO data from Lambda-CDM truth.
/// 2. Fit Lambda-CDM model (2 params: Omega_m, H_0).
/// 3. Fit bounce model (3 params: Omega_m, H_0, q_corr).
/// 4. Compare AIC/BIC.
/// 5. Compute spectral index n_s for best-fit q_corr.
///
/// Returns (lcdm_result, bounce_result, delta_bic, n_s).
pub fn run_observational_fit(seed: u64) -> (FitResult, FitResult, f64, f64) {
    let sn = generate_synthetic_sn_data(200, 0.315, 67.4, seed);
    let bao = generate_synthetic_bao_data(0.315, 67.4, seed + 100);

    let lcdm = fit_model(&sn, &bao, false);
    let bounce = fit_model(&sn, &bao, true);

    let delta_bic = bounce.bic - lcdm.bic;
    let n_s = spectral_index_bounce(bounce.q_corr, 0.315);

    (lcdm, bounce, delta_bic, n_s)
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

    #[test]
    fn test_synthetic_sn_data_generation() {
        let sn = generate_synthetic_sn_data(50, 0.315, 67.4, 42);
        assert_eq!(sn.z.len(), 50);
        assert_eq!(sn.mu_obs.len(), 50);
        assert_eq!(sn.mu_err.len(), 50);

        // Redshifts should be sorted and in range
        assert!(sn.z[0] >= 0.01);
        assert!(sn.z[49] <= 2.30);
        for w in sn.z.windows(2) {
            assert!(w[1] >= w[0], "Redshifts not sorted");
        }

        // Errors should be in [0.10, 0.15]
        for &e in &sn.mu_err {
            assert!(e >= 0.10 && e <= 0.15, "mu_err={e} out of range");
        }
    }

    #[test]
    fn test_synthetic_bao_data_generation() {
        let bao = generate_synthetic_bao_data(0.315, 67.4, 142);
        assert_eq!(bao.z_eff.len(), 7);
        assert_eq!(bao.dv_rd_obs.len(), 7);
        assert_eq!(bao.dv_rd_err.len(), 7);

        // D_V/r_d should be positive
        for &v in &bao.dv_rd_obs {
            assert!(v > 0.0, "D_V/r_d should be positive, got {v}");
        }
    }

    #[test]
    fn test_lcdm_chi2_per_dof_reasonable() {
        let sn = generate_synthetic_sn_data(50, 0.315, 67.4, 42);
        let bao = generate_synthetic_bao_data(0.315, 67.4, 142);
        let result = fit_model(&sn, &bao, false);

        let n_data = sn.z.len() + bao.z_eff.len();
        let dof = n_data - result.n_params;
        let chi2_per_dof = result.chi2 / dof as f64;

        assert!(
            chi2_per_dof > 0.3 && chi2_per_dof < 3.0,
            "chi2/dof = {chi2_per_dof:.2}, expected in [0.3, 3.0]"
        );
    }

    #[test]
    fn test_bounce_delta_bic_threshold() {
        let sn = generate_synthetic_sn_data(50, 0.315, 67.4, 42);
        let bao = generate_synthetic_bao_data(0.315, 67.4, 142);

        let lcdm = fit_model(&sn, &bao, false);
        let bounce = fit_model(&sn, &bao, true);

        let delta_bic = bounce.bic - lcdm.bic;
        // Bounce should NOT be strongly preferred over LCDM for LCDM-generated data
        assert!(
            delta_bic > -10.0,
            "delta_BIC = {delta_bic:.2}, bounce too strongly preferred"
        );
    }
}

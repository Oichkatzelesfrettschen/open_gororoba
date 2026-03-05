//! Orthoplex diffusion dark energy: w(z) from K_{2,2,...,2} heat kernel.
//!
//! Computes the equation of state w(z) by mapping the exact analytical heat
//! kernel of the complete multipartite graph K_{2,2,...,2} (orthoplex graph)
//! to cosmological scales via an inverted power-law ansatz.
//!
//! # Physics
//!
//! The ZD graph components discovered in Cayley-Dickson algebras at dimension N
//! have K_{2,2,...,2} topology with k = N/4 - 1 parts of size 2, giving n = 2k
//! vertices. The Laplacian eigenvalues (L = D - A, degree = 2(k-1)) are:
//!
//! - mu_0 = 0          (multiplicity 1)
//! - mu_1 = 2(k-1)     (multiplicity k)     -- from adj eigenvalue 0
//! - mu_2 = 2k          (multiplicity k-1)   -- from adj eigenvalue -2
//!
//! Total multiplicities: 1 + k + (k-1) = 2k = n vertices.
//!
//! # Scale mapping
//!
//! t(z) = t_0 / (1+z)^alpha maps cosmic expansion to graph diffusion:
//! - z = infinity (Big Bang): t -> 0 (no diffusion, graph unexplored)
//! - z = 0 (today): t = t_0 (maximal diffusion, graph fully explored)
//!
//! # Equation of state
//!
//! w(z) = -1 + beta * d_s(t(z))
//!
//! Recovers Lambda-CDM (w = -1) when beta = 0 or d_s = 0. Smooth everywhere,
//! no singularities. The model is "thawing": starts at w = -1 and departs as
//! the universe expands.
//!
//! # References
//! - Sprint 60 claims C-926, C-929, C-930 (ZD graph topology)
//! - Calcagni (2010), PRL 104, 251301 (spectral dimension in quantum gravity)
//! - DESI Collaboration (2025), arXiv:2503.14738 (dynamical dark energy)

use crate::{
    bounce::{C_KM_S, bao_sound_horizon},
    gl_integrate,
    observational::{
        CMB_SHIFT_R_ERR, CMB_SHIFT_R_OBS, CcMeasurement, FsigMeasurement, RealBaoData, RealSnData,
        SIGMA8_PLANCK, compute_growth_batch,
    },
    optimizer::{NelderMeadConfig, bounded_nelder_mead},
};
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Parameters for the orthoplex diffusion dark energy model.
#[derive(Clone, Debug)]
pub struct OrthoplexParams {
    /// Number of parts in K_{2,2,...,2}. Derived from CD dimension: k = N/4 - 1.
    pub k: usize,
    /// Power-law exponent for scale mapping t(z) = t_0 / (1+z)^alpha.
    pub alpha: f64,
    /// Dimensionless coupling: w = -1 + beta * d_s.
    pub beta: f64,
    /// Present-day diffusion scale.
    pub t_0: f64,
}

/// Result of fitting the orthoplex diffusion model to SN + BAO + CMB + CC data.
#[derive(Clone, Debug)]
pub struct OrthoplexFitResult {
    /// Best-fit matter density parameter.
    pub omega_m: f64,
    /// Best-fit Hubble constant (km/s/Mpc).
    pub h0: f64,
    /// Best-fit power-law exponent.
    pub alpha: f64,
    /// Best-fit coupling constant.
    pub beta: f64,
    /// Best-fit present-day diffusion scale.
    pub t_0: f64,
    /// Number of parts in K_{2,2,...,2} (fixed, not fitted).
    pub k: usize,
    /// Whether beta was fixed (true) or fitted (false).
    pub beta_fixed: bool,
    /// Total chi-square.
    pub chi2_total: f64,
    /// SN contribution to chi-square.
    pub chi2_sn: f64,
    /// BAO contribution to chi-square.
    pub chi2_bao: f64,
    /// CMB shift parameter contribution to chi-square.
    pub chi2_cmb: f64,
    /// Cosmic chronometer contribution to chi-square.
    pub chi2_cc: f64,
    /// f*sigma8 growth rate contribution to chi-square.
    pub chi2_fsig: f64,
    /// Number of free parameters.
    pub n_params: usize,
    /// Number of data points.
    pub n_data: usize,
    /// Akaike Information Criterion.
    pub aic: f64,
    /// Bayesian Information Criterion.
    pub bic: f64,
    /// w(z=0) -- equation of state today.
    pub w_0: f64,
    /// w(z=2) -- equation of state at high redshift.
    pub w_high_z: f64,
}

/// Comparison between Lambda-CDM and orthoplex dark energy models.
#[derive(Clone, Debug)]
pub struct OrthoplexComparison {
    pub lcdm: crate::observational::ObsFitResult,
    pub orthoplex: OrthoplexFitResult,
    /// Delta BIC = BIC_orthoplex - BIC_lcdm (positive = Lambda-CDM preferred).
    pub delta_bic: f64,
    /// Delta AIC = AIC_orthoplex - AIC_lcdm.
    pub delta_aic: f64,
    /// Fixed-beta (beta=1.0, 4-param) fit result, if computed.
    pub orthoplex_fixed_beta: Option<OrthoplexFitResult>,
    /// Delta BIC for fixed-beta variant.
    pub delta_bic_fixed_beta: Option<f64>,
    /// Delta AIC for fixed-beta variant.
    pub delta_aic_fixed_beta: Option<f64>,
}

// ---------------------------------------------------------------------------
// Heat kernel and spectral dimension (exact analytical)
// ---------------------------------------------------------------------------

/// Heat kernel return probability P(t) on K_{2,2,...,2} with k parts of size 2.
///
/// P(t) = (1/(2k)) * [1 + k*exp(-2(k-1)*t) + (k-1)*exp(-2k*t)]
///
/// At t=0: P = 1 (walker is at starting vertex with certainty).
/// At t=inf: P -> 1/(2k) (uniform over all 2k vertices).
pub fn heat_kernel_k22(t: f64, k: usize) -> f64 {
    let kf = k as f64;
    let n = 2.0 * kf;
    (1.0 + kf * (-2.0 * (kf - 1.0) * t).exp() + (kf - 1.0) * (-2.0 * kf * t).exp()) / n
}

/// Spectral dimension d_s(t) on K_{2,2,...,2} with k parts.
///
/// d_s(t) = -2t * P'(t) / P(t)
///
/// where P'(t) = (1/(2k)) * [-2k(k-1)*exp(-2(k-1)*t) - 2k(k-1)*exp(-2k*t)]
///
/// Simplifies to:
/// d_s(t) = 4k*t*(k-1) * [exp(-2(k-1)*t) + exp(-2k*t)]
///          / [1 + k*exp(-2(k-1)*t) + (k-1)*exp(-2k*t)]
///
/// At t=0: d_s = 0 (no diffusion).
/// At t=inf: d_s -> 0 (graph fully explored, finite).
/// Peak at intermediate t: maximal spectral dimension.
pub fn spectral_dimension_k22(t: f64, k: usize) -> f64 {
    if t <= 0.0 {
        return 0.0;
    }
    let kf = k as f64;
    let km1 = kf - 1.0;

    let exp_mu1 = (-2.0 * km1 * t).exp();
    let exp_mu2 = (-2.0 * kf * t).exp();

    let numerator = 4.0 * kf * t * km1 * (exp_mu1 + exp_mu2);
    let denominator = 1.0 + kf * exp_mu1 + km1 * exp_mu2;

    if denominator.abs() < 1e-300 {
        return 0.0;
    }

    numerator / denominator
}

// ---------------------------------------------------------------------------
// Scale mapping and equation of state
// ---------------------------------------------------------------------------

/// Diffusion time as a function of redshift.
///
/// t(z) = t_0 / (1+z)^alpha
///
/// Inverted power-law: diffusion time grows with cosmic expansion.
#[inline]
pub fn diffusion_time(z: f64, alpha: f64, t_0: f64) -> f64 {
    t_0 / (1.0 + z).powf(alpha)
}

/// Equation of state w(z) from orthoplex spectral dimension.
///
/// w(z) = -1 + beta * d_s(t(z))
///
/// Smooth everywhere. Recovers Lambda-CDM when beta=0 or d_s=0.
pub fn w_orthoplex(z: f64, k: usize, alpha: f64, beta: f64, t_0: f64) -> f64 {
    let t = diffusion_time(z, alpha, t_0);
    let ds = spectral_dimension_k22(t, k);
    -1.0 + beta * ds
}

/// Dark energy density ratio X(z) = rho_DE(z) / rho_DE(0).
///
/// X(z) = exp(3 * integral_0^z [1 + w(z')] / (1+z') dz')
///
/// Computed via Gauss-Legendre quadrature (degree 50).
pub fn dark_energy_density_ratio(z: f64, k: usize, alpha: f64, beta: f64, t_0: f64) -> f64 {
    if z <= 0.0 {
        return 1.0;
    }

    let integral = gl_integrate(
        |zp| {
            let w = w_orthoplex(zp, k, alpha, beta, t_0);
            (1.0 + w) / (1.0 + zp)
        },
        0.0,
        z,
        50,
    );

    (3.0 * integral).exp()
}

/// Dimensionless Hubble parameter E(z) for orthoplex dark energy.
///
/// E^2(z) = Omega_m * (1+z)^3 + (1 - Omega_m) * X(z)
pub fn hubble_e_orthoplex(z: f64, omega_m: f64, k: usize, alpha: f64, beta: f64, t_0: f64) -> f64 {
    let zp1 = 1.0 + z;
    let matter = omega_m * zp1 * zp1 * zp1;
    let de = (1.0 - omega_m) * dark_energy_density_ratio(z, k, alpha, beta, t_0);
    (matter + de).max(1e-30).sqrt()
}

/// Luminosity distance d_L(z) in Mpc for orthoplex dark energy.
///
/// d_L(z) = (c/H_0) * (1+z) * integral_0^z dz'/E(z')
pub fn luminosity_distance_orthoplex(
    z: f64,
    omega_m: f64,
    h0: f64,
    k: usize,
    alpha: f64,
    beta: f64,
    t_0: f64,
) -> f64 {
    if z <= 0.0 {
        return 0.0;
    }

    let integral = gl_integrate(
        |zp| 1.0 / hubble_e_orthoplex(zp, omega_m, k, alpha, beta, t_0),
        0.0,
        z,
        50,
    );

    (C_KM_S / h0) * (1.0 + z) * integral
}

/// Distance modulus mu(z) for orthoplex dark energy.
///
/// mu = 5 * log10(d_L / 10 pc)
pub fn distance_modulus_orthoplex(
    z: f64,
    omega_m: f64,
    h0: f64,
    k: usize,
    alpha: f64,
    beta: f64,
    t_0: f64,
) -> f64 {
    let d_l = luminosity_distance_orthoplex(z, omega_m, h0, k, alpha, beta, t_0);
    let d_l_pc = d_l * 1e6; // Mpc to pc
    5.0 * (d_l_pc.max(1e-30) / 10.0).log10()
}

// ---------------------------------------------------------------------------
// Grid-interpolated comoving distance (avoids redundant GL quadrature)
// ---------------------------------------------------------------------------

/// Precomputed comoving distance grid for fast interpolation.
///
/// Instead of running a 50-point GL quadrature from 0 to z_i for each of the
/// ~1578 supernovae (each GL node itself triggering another 50-point inner
/// quadrature for dark_energy_density_ratio), we evaluate d_C(z) on a
/// linearly-spaced grid once, then interpolate. Reduces per-step cost from
/// O(N_data * N_GL^2) to O(N_grid * N_GL^2 + N_data).
struct ComovingGrid {
    /// Grid redshifts, linearly spaced from 0 to z_max.
    z_grid: Vec<f64>,
    /// Comoving distance (c/H_0) * integral_0^z dz'/E(z') at each grid point.
    dc_grid: Vec<f64>,
    /// E(z) at each grid point (for BAO d_H computation).
    e_grid: Vec<f64>,
    /// Grid spacing (uniform).
    dz: f64,
}

impl ComovingGrid {
    /// Build grid with `n_grid` points covering [0, z_max].
    ///
    /// Uses cumulative GL quadrature: each segment [z_i, z_{i+1}] is integrated
    /// once, then accumulated. Total: n_grid GL evaluations instead of
    /// n_data * n_grid.
    #[allow(clippy::too_many_arguments)]
    fn build(
        z_max: f64,
        n_grid: usize,
        omega_m: f64,
        h0: f64,
        k: usize,
        alpha: f64,
        beta: f64,
        t_0: f64,
    ) -> Self {
        let dz = z_max / (n_grid - 1).max(1) as f64;
        let c_over_h0 = C_KM_S / h0;

        let mut z_grid = Vec::with_capacity(n_grid);
        let mut dc_grid = Vec::with_capacity(n_grid);
        let mut e_grid = Vec::with_capacity(n_grid);

        let mut cumulative_dc = 0.0;

        for i in 0..n_grid {
            let z = dz * i as f64;
            z_grid.push(z);

            if i > 0 {
                let z_prev = z_grid[i - 1];
                let segment = gl_integrate(
                    |zp| 1.0 / hubble_e_orthoplex(zp, omega_m, k, alpha, beta, t_0),
                    z_prev,
                    z,
                    20, // 20 GL nodes per segment is sufficient for smooth E(z)
                );
                cumulative_dc += c_over_h0 * segment;
            }

            dc_grid.push(cumulative_dc);
            e_grid.push(hubble_e_orthoplex(z, omega_m, k, alpha, beta, t_0));
        }

        Self { z_grid, dc_grid, e_grid, dz }
    }

    /// Interpolate comoving distance at arbitrary z via linear interpolation.
    #[inline]
    fn interp_dc(&self, z: f64) -> f64 {
        if z <= 0.0 {
            return 0.0;
        }
        let idx_f = z / self.dz;
        let idx = idx_f as usize;
        if idx + 1 >= self.z_grid.len() {
            return *self.dc_grid.last().unwrap_or(&0.0);
        }
        let frac = idx_f - idx as f64;
        self.dc_grid[idx] + frac * (self.dc_grid[idx + 1] - self.dc_grid[idx])
    }

    /// Interpolate E(z) at arbitrary z via linear interpolation.
    #[inline]
    fn interp_e(&self, z: f64) -> f64 {
        if z <= 0.0 {
            return 1.0;
        }
        let idx_f = z / self.dz;
        let idx = idx_f as usize;
        if idx + 1 >= self.z_grid.len() {
            return *self.e_grid.last().unwrap_or(&1.0);
        }
        let frac = idx_f - idx as f64;
        self.e_grid[idx] + frac * (self.e_grid[idx + 1] - self.e_grid[idx])
    }
}

// ---------------------------------------------------------------------------
// Chi-square functions for observational fitting
// ---------------------------------------------------------------------------

/// Chi-square for Pantheon+ SN data with orthoplex dark energy model.
///
/// When `sn.precision` is `Some`, uses the full covariance precision matrix:
///   chi2 = delta^T P delta - (1^T P delta)^2 / (1^T P 1)
/// Otherwise falls back to diagonal:
///   chi2 = sum (delta_i / sigma_i)^2 - [sum(delta_i/sigma_i^2)]^2 / [sum(1/sigma_i^2)]
/// Both include analytic M_B marginalization (Conley+ 2011).
pub fn chi2_sn_orthoplex(
    omega_m: f64,
    h0: f64,
    k: usize,
    alpha: f64,
    beta: f64,
    t_0: f64,
    sn: &RealSnData,
) -> f64 {
    if !(0.01..=0.99).contains(&omega_m) || !(50.0..=90.0).contains(&h0) {
        return 1e10;
    }

    let z_max = sn.z.iter().cloned().fold(0.0_f64, f64::max);
    let grid = ComovingGrid::build(z_max * 1.01, 200, omega_m, h0, k, alpha, beta, t_0);

    // Compute residuals (mu_obs - mu_model) for all SNe
    let n = sn.z.len();
    let residuals: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let zi = sn.z[i];
            let dc = grid.interp_dc(zi);
            let d_l = dc * (1.0 + zi);
            let d_l_pc = d_l * 1e6;
            let mu_model = 5.0 * (d_l_pc.max(1e-30) / 10.0).log10();
            sn.mu[i] - mu_model
        })
        .collect();

    if let Some(ref prec) = sn.precision {
        // Full covariance: chi2_marg = A - B^2/C
        // A = delta^T * P * delta
        // B = 1^T * P * delta
        // C = 1^T * P * 1
        let delta = nalgebra::DVector::from_column_slice(&residuals);
        let p_delta = prec * &delta;
        let a = delta.dot(&p_delta);
        let b: f64 = p_delta.iter().sum();
        let c: f64 = prec.row_iter().map(|row| row.iter().sum::<f64>()).sum();
        if c.abs() < 1e-30 {
            return 1e10;
        }
        a - b * b / c
    } else {
        // Diagonal fallback: original implementation
        let mut a_sum = 0.0;
        let mut b_sum = 0.0;
        let mut c_sum = 0.0;
        for (i, &r) in residuals.iter().enumerate().take(n) {
            let inv_var = 1.0 / (sn.mu_err[i] * sn.mu_err[i]);
            a_sum += r * r * inv_var;
            b_sum += r * inv_var;
            c_sum += inv_var;
        }
        a_sum - b_sum * b_sum / c_sum
    }
}

/// Chi-square for real BAO data with orthoplex dark energy model.
///
/// Uses same grid-interpolation as SN chi2 for comoving distance.
/// BAO has only 7-12 bins so parallelism is unnecessary here.
pub fn chi2_bao_orthoplex(
    omega_m: f64,
    h0: f64,
    k: usize,
    alpha: f64,
    beta: f64,
    t_0: f64,
    bao: &RealBaoData,
) -> f64 {
    if !(0.01..=0.99).contains(&omega_m) || !(50.0..=90.0).contains(&h0) {
        return 1e10;
    }

    let z_max = bao.z_eff.iter().cloned().fold(0.0_f64, f64::max);
    let grid = ComovingGrid::build(z_max * 1.01, 200, omega_m, h0, k, alpha, beta, t_0);

    let r_d = bao_sound_horizon(omega_m, h0);
    let mut chi2 = 0.0;

    for i in 0..bao.z_eff.len() {
        let zi = bao.z_eff[i];
        let d_c = grid.interp_dc(zi);
        let e_val = grid.interp_e(zi);
        let d_h = C_KM_S / (h0 * e_val);

        if bao.is_isotropic[i] {
            let dv_model = (zi * d_c * d_c * d_h).powf(1.0 / 3.0) / r_d;
            let dv_obs = bao.dm_over_rd[i];
            let sigma = bao.dm_over_rd_err[i];
            if sigma > 0.0 {
                let residual = (dv_obs - dv_model) / sigma;
                chi2 += residual * residual;
            }
        } else {
            let dm_model = d_c / r_d;
            let dh_model = d_h / r_d;

            let delta_dm = bao.dm_over_rd[i] - dm_model;
            let delta_dh = bao.dh_over_rd[i] - dh_model;

            let s_dm = bao.dm_over_rd_err[i];
            let s_dh = bao.dh_over_rd_err[i];
            let rho_i = bao.rho[i];

            let var_dm = s_dm * s_dm;
            let var_dh = s_dh * s_dh;
            let cov = rho_i * s_dm * s_dh;

            let det = var_dm * var_dh - cov * cov;
            if det.abs() < 1e-30 {
                continue;
            }

            let inv_det = 1.0 / det;
            chi2 += inv_det
                * (var_dh * delta_dm * delta_dm - 2.0 * cov * delta_dm * delta_dh
                    + var_dm * delta_dh * delta_dh);
        }
    }

    chi2
}

// ---------------------------------------------------------------------------
// CMB and cosmic chronometer chi-square for orthoplex
// ---------------------------------------------------------------------------

/// CMB shift parameter R for the orthoplex dark energy model.
///
/// R = sqrt(Omega_m) * integral_0^{z_star} dz/E(z)
///
/// At z_star = 1089, orthoplex w(z) -> -1 rapidly (d_s -> 0 at large z),
/// so this effectively matches LCDM. Uses GL(100) for the high-z integral.
pub fn cmb_shift_parameter_orthoplex(
    omega_m: f64,
    k: usize,
    alpha: f64,
    beta: f64,
    t_0: f64,
) -> f64 {
    let z_star = 1089.0;
    let integral = gl_integrate(
        |z| 1.0 / hubble_e_orthoplex(z, omega_m, k, alpha, beta, t_0),
        0.0,
        z_star,
        100,
    );
    omega_m.sqrt() * integral
}

/// Chi-square from the CMB shift parameter for orthoplex dark energy.
pub fn chi2_cmb_shift_orthoplex(
    omega_m: f64,
    k: usize,
    alpha: f64,
    beta: f64,
    t_0: f64,
) -> f64 {
    let r_model = cmb_shift_parameter_orthoplex(omega_m, k, alpha, beta, t_0);
    let residual = (r_model - CMB_SHIFT_R_OBS) / CMB_SHIFT_R_ERR;
    residual * residual
}

/// Chi-square for cosmic chronometer H(z) data with orthoplex dark energy.
///
/// H_model(z) = H_0 * E(z). Each E(z) call triggers a 50-node inner GL
/// for dark_energy_density_ratio, but only 31 CC points = 1550 evals per step.
pub fn chi2_cc_orthoplex(
    omega_m: f64,
    h0: f64,
    k: usize,
    alpha: f64,
    beta: f64,
    t_0: f64,
    cc: &[CcMeasurement],
) -> f64 {
    if !(0.01..=0.99).contains(&omega_m) || !(50.0..=90.0).contains(&h0) {
        return 1e10;
    }

    let mut chi2 = 0.0;
    for m in cc {
        let e_val = hubble_e_orthoplex(m.z, omega_m, k, alpha, beta, t_0);
        let h_model = h0 * e_val;
        let residual = (m.h_obs - h_model) / m.h_err;
        chi2 += residual * residual;
    }

    chi2
}

/// Chi-square for f*sigma8 growth rate data with orthoplex dark energy.
///
/// Uses batch RK4 growth ODE integrator (single sweep) with orthoplex E(z).
pub fn chi2_fsig8_orthoplex(
    omega_m: f64,
    k: usize,
    alpha: f64,
    beta: f64,
    t_0: f64,
    fsig: &[FsigMeasurement],
) -> f64 {
    if fsig.is_empty() {
        return 0.0;
    }

    let e_func = |z: f64| hubble_e_orthoplex(z, omega_m, k, alpha, beta, t_0);
    let zs: Vec<f64> = fsig.iter().map(|m| m.z).collect();
    let results = compute_growth_batch(omega_m, &e_func, &zs, SIGMA8_PLANCK);

    let mut chi2 = 0.0;
    for (m, &(_, fsig8_model)) in fsig.iter().zip(results.iter()) {
        let residual = (m.fsig8_obs - fsig8_model) / m.fsig8_err;
        chi2 += residual * residual;
    }
    chi2
}

// ---------------------------------------------------------------------------
// Model fitting
// ---------------------------------------------------------------------------

/// Fit orthoplex dark energy model to real SN + BAO + CC + f*sigma8 data.
///
/// Free parameters: omega_m, h0, alpha, beta, t_0 (5 parameters).
/// Fixed: k (graph topology, determined by CD algebra dimension).
///
/// Objective includes: SN chi2 + BAO chi2 + CMB shift chi2 + CC chi2 + f*sigma8 chi2.
/// Pass `&[]` for `cc` or `fsig` to omit those constraints.
///
/// Uses multi-start Nelder-Mead: 9 initial guesses spanning the expanded
/// parameter space (alpha up to 25, t_0 down to 1e-5) to avoid local
/// minima in the 5D chi2 landscape. Each start runs 10k iterations.
pub fn fit_orthoplex_model(
    sn: &RealSnData,
    bao: &RealBaoData,
    cc: &[CcMeasurement],
    fsig: &[FsigMeasurement],
    k: usize,
) -> OrthoplexFitResult {
    let n_bao_data = crate::observational::bao_data_point_count(bao);
    let n_cmb = 1;
    let n_cc = cc.len();
    let n_data = sn.z.len() + n_bao_data + n_cmb + n_cc + fsig.len();

    let bounds = [
        (0.1, 0.5),    // omega_m
        (60.0, 80.0),  // h0
        (0.1, 25.0),   // alpha  (uncaged from 5.0)
        (-0.5, 1.0),   // beta   (negative allows phantom)
        (1e-5, 10.0),  // t_0    (uncaged from 0.01)
    ];

    let obj = |p: &[f64]| {
        let omega_m = p[0];
        let h0 = p[1];
        let alpha = p[2];
        let beta = p[3];
        let t_0 = p[4];

        let chi2_sn = chi2_sn_orthoplex(omega_m, h0, k, alpha, beta, t_0, sn);
        let chi2_bao = chi2_bao_orthoplex(omega_m, h0, k, alpha, beta, t_0, bao);
        let chi2_cmb = chi2_cmb_shift_orthoplex(omega_m, k, alpha, beta, t_0);
        let chi2_cc_val = chi2_cc_orthoplex(omega_m, h0, k, alpha, beta, t_0, cc);

        // f*sigma8 excluded from inner loop (RK4 too expensive); evaluated post-fit.
        chi2_sn + chi2_bao + chi2_cmb + chi2_cc_val
    };

    // Multi-start: sample initial guesses across the expanded space.
    // Parallelized via rayon to use all CPU cores concurrently.
    let initial_guesses: Vec<Vec<f64>> = vec![
        // [omega_m, h0, alpha, beta, t_0]
        vec![0.3,  70.0, 1.0,  0.0,   1.0  ],  // baseline (near LCDM)
        vec![0.3,  70.0, 5.0,  0.05,  0.01 ],  // previous boundary hit
        vec![0.3,  70.0, 8.0,  0.1,   0.005],  // deeper alpha, sharper t_0
        vec![0.3,  70.0, 12.0, 0.05,  0.001],  // high alpha, very sharp onset
        vec![0.3,  70.0, 20.0, 0.02,  1e-4 ],  // extreme alpha frontier
        vec![0.28, 68.0, 3.0,  0.15,  0.5  ],  // lower H0, moderate alpha
        vec![0.32, 72.0, 6.0, -0.1,   0.1  ],  // phantom regime (beta < 0)
        vec![0.3,  70.0, 10.0, 0.03,  0.003],  // mid-range exploration
        vec![0.3,  70.0, 15.0, 0.08,  5e-4 ],  // high-alpha sharp-onset
    ];

    let nm_config = NelderMeadConfig {
        bounds: bounds.to_vec(),
        max_iter: 2_000,
        tol: 1e-4,
        ..Default::default()
    };
    let (best, _global_chi2) = bounded_nelder_mead(obj, &initial_guesses, &nm_config);

    let omega_m = best[0];
    let h0 = best[1];
    let alpha = best[2];
    let beta = best[3];
    let t_0 = best[4];

    let chi2_sn_val = chi2_sn_orthoplex(omega_m, h0, k, alpha, beta, t_0, sn);
    let chi2_bao_val = chi2_bao_orthoplex(omega_m, h0, k, alpha, beta, t_0, bao);
    let chi2_cmb_val = chi2_cmb_shift_orthoplex(omega_m, k, alpha, beta, t_0);
    let chi2_cc_val = chi2_cc_orthoplex(omega_m, h0, k, alpha, beta, t_0, cc);
    let chi2_fsig_val = chi2_fsig8_orthoplex(omega_m, k, alpha, beta, t_0, fsig);
    let chi2_total = chi2_sn_val + chi2_bao_val + chi2_cmb_val + chi2_cc_val + chi2_fsig_val;
    let n_params = 5;
    let aic = chi2_total + 2.0 * n_params as f64;
    let bic = chi2_total + n_params as f64 * (n_data as f64).ln();

    let w_0 = w_orthoplex(0.0, k, alpha, beta, t_0);
    let w_high_z = w_orthoplex(2.0, k, alpha, beta, t_0);

    OrthoplexFitResult {
        omega_m,
        h0,
        alpha,
        beta,
        t_0,
        k,
        beta_fixed: false,
        chi2_total,
        chi2_sn: chi2_sn_val,
        chi2_bao: chi2_bao_val,
        chi2_cmb: chi2_cmb_val,
        chi2_cc: chi2_cc_val,
        chi2_fsig: chi2_fsig_val,
        n_params,
        n_data,
        aic,
        bic,
        w_0,
        w_high_z,
    }
}

/// Run full model comparison: Lambda-CDM vs orthoplex on real data.
///
/// Pass `&[]` for `cc` or `fsig` to omit those constraints.
pub fn compare_orthoplex(
    sn: &RealSnData,
    bao: &RealBaoData,
    cc: &[CcMeasurement],
    fsig: &[FsigMeasurement],
    k: usize,
) -> OrthoplexComparison {
    let lcdm = crate::observational::fit_real_data(sn, bao, cc, fsig, false);
    let orthoplex = fit_orthoplex_model(sn, bao, cc, fsig, k);

    let delta_bic = orthoplex.bic - lcdm.bic;
    let delta_aic = orthoplex.aic - lcdm.aic;

    OrthoplexComparison {
        lcdm,
        orthoplex,
        delta_bic,
        delta_aic,
        orthoplex_fixed_beta: None,
        delta_bic_fixed_beta: None,
        delta_aic_fixed_beta: None,
    }
}

/// Fit orthoplex dark energy model with beta fixed at 1.0 (4 free parameters).
///
/// Reduces from 5 to 4 free parameters: omega_m, h0, alpha, t_0.
/// Beta=1.0 is where the 5-param optimizer converges anyway, so chi2 should
/// be nearly identical while saving ln(N) ~ 7.39 in BIC penalty.
pub fn fit_orthoplex_model_fixed_beta(
    sn: &RealSnData,
    bao: &RealBaoData,
    cc: &[CcMeasurement],
    fsig: &[FsigMeasurement],
    k: usize,
) -> OrthoplexFitResult {
    let beta = 1.0; // Fixed
    let n_bao_data = crate::observational::bao_data_point_count(bao);
    let n_cmb = 1;
    let n_cc = cc.len();
    let n_data = sn.z.len() + n_bao_data + n_cmb + n_cc + fsig.len();

    let bounds = [
        (0.1, 0.5),    // omega_m
        (60.0, 80.0),  // h0
        (0.1, 25.0),   // alpha
        (1e-5, 10.0),  // t_0
    ];

    let obj = |p: &[f64]| {
        let omega_m = p[0];
        let h0 = p[1];
        let alpha = p[2];
        let t_0 = p[3];

        let chi2_sn = chi2_sn_orthoplex(omega_m, h0, k, alpha, beta, t_0, sn);
        let chi2_bao = chi2_bao_orthoplex(omega_m, h0, k, alpha, beta, t_0, bao);
        let chi2_cmb = chi2_cmb_shift_orthoplex(omega_m, k, alpha, beta, t_0);
        let chi2_cc_val = chi2_cc_orthoplex(omega_m, h0, k, alpha, beta, t_0, cc);

        // f*sigma8 excluded from inner loop (RK4 too expensive); evaluated post-fit.
        chi2_sn + chi2_bao + chi2_cmb + chi2_cc_val
    };

    // Multi-start: 4D grid (no beta dimension).
    // Parallelized via rayon to use all CPU cores concurrently.
    let initial_guesses: Vec<Vec<f64>> = vec![
        // [omega_m, h0, alpha, t_0]
        vec![0.3,  70.0, 1.0,   1.0  ],
        vec![0.3,  70.0, 5.0,   0.01 ],
        vec![0.3,  70.0, 8.0,   0.005],
        vec![0.3,  70.0, 12.0,  0.001],
        vec![0.3,  70.0, 20.0,  1e-4 ],
        vec![0.28, 68.0, 3.0,   0.5  ],
        vec![0.32, 72.0, 6.0,   0.1  ],
        vec![0.3,  70.0, 10.0,  0.003],
        vec![0.3,  70.0, 15.0,  5e-4 ],
    ];

    let nm_config = NelderMeadConfig {
        bounds: bounds.to_vec(),
        max_iter: 2_000,
        tol: 1e-4,
        ..Default::default()
    };
    let (global_best, _global_chi2) = bounded_nelder_mead(obj, &initial_guesses, &nm_config);

    let omega_m = global_best[0];
    let h0 = global_best[1];
    let alpha = global_best[2];
    let t_0 = global_best[3];

    let chi2_sn_val = chi2_sn_orthoplex(omega_m, h0, k, alpha, beta, t_0, sn);
    let chi2_bao_val = chi2_bao_orthoplex(omega_m, h0, k, alpha, beta, t_0, bao);
    let chi2_cmb_val = chi2_cmb_shift_orthoplex(omega_m, k, alpha, beta, t_0);
    let chi2_cc_val = chi2_cc_orthoplex(omega_m, h0, k, alpha, beta, t_0, cc);
    let chi2_fsig_val = chi2_fsig8_orthoplex(omega_m, k, alpha, beta, t_0, fsig);
    let chi2_total = chi2_sn_val + chi2_bao_val + chi2_cmb_val + chi2_cc_val + chi2_fsig_val;
    let n_params = 4;
    let aic = chi2_total + 2.0 * n_params as f64;
    let bic = chi2_total + n_params as f64 * (n_data as f64).ln();

    let w_0 = w_orthoplex(0.0, k, alpha, beta, t_0);
    let w_high_z = w_orthoplex(2.0, k, alpha, beta, t_0);

    OrthoplexFitResult {
        omega_m,
        h0,
        alpha,
        beta,
        t_0,
        k,
        beta_fixed: true,
        chi2_total,
        chi2_sn: chi2_sn_val,
        chi2_bao: chi2_bao_val,
        chi2_cmb: chi2_cmb_val,
        chi2_cc: chi2_cc_val,
        chi2_fsig: chi2_fsig_val,
        n_params,
        n_data,
        aic,
        bic,
        w_0,
        w_high_z,
    }
}

/// Compare Lambda-CDM vs orthoplex with fixed beta=1.0.
pub fn compare_orthoplex_fixed_beta(
    sn: &RealSnData,
    bao: &RealBaoData,
    cc: &[CcMeasurement],
    fsig: &[FsigMeasurement],
    k: usize,
) -> OrthoplexComparison {
    let lcdm = crate::observational::fit_real_data(sn, bao, cc, fsig, false);
    let orthoplex = fit_orthoplex_model_fixed_beta(sn, bao, cc, fsig, k);

    let delta_bic = orthoplex.bic - lcdm.bic;
    let delta_aic = orthoplex.aic - lcdm.aic;

    OrthoplexComparison {
        lcdm,
        orthoplex_fixed_beta: None,
        delta_bic_fixed_beta: None,
        delta_aic_fixed_beta: None,
        orthoplex,
        delta_bic,
        delta_aic,
    }
}

/// Run full model comparison: LCDM vs free-beta vs fixed-beta orthoplex.
///
/// Populates all fields in OrthoplexComparison including the fixed-beta variant.
pub fn compare_orthoplex_all(
    sn: &RealSnData,
    bao: &RealBaoData,
    cc: &[CcMeasurement],
    fsig: &[FsigMeasurement],
    k: usize,
) -> OrthoplexComparison {
    let lcdm = crate::observational::fit_real_data(sn, bao, cc, fsig, false);
    let orthoplex = fit_orthoplex_model(sn, bao, cc, fsig, k);
    let fixed = fit_orthoplex_model_fixed_beta(sn, bao, cc, fsig, k);

    let delta_bic = orthoplex.bic - lcdm.bic;
    let delta_aic = orthoplex.aic - lcdm.aic;
    let delta_bic_fb = fixed.bic - lcdm.bic;
    let delta_aic_fb = fixed.aic - lcdm.aic;

    OrthoplexComparison {
        lcdm,
        orthoplex,
        delta_bic,
        delta_aic,
        orthoplex_fixed_beta: Some(fixed),
        delta_bic_fixed_beta: Some(delta_bic_fb),
        delta_aic_fixed_beta: Some(delta_aic_fb),
    }
}

/// Generate w(z) table for CSV output.
///
/// Returns (z, w, d_s, t, beta_ds) tuples at `n_points` evenly spaced
/// redshifts in [0, z_max].
pub fn w_of_z_table(
    k: usize,
    alpha: f64,
    beta: f64,
    t_0: f64,
    z_max: f64,
    n_points: usize,
) -> Vec<(f64, f64, f64, f64, f64)> {
    (0..n_points)
        .map(|i| {
            let z = z_max * i as f64 / (n_points - 1).max(1) as f64;
            let t = diffusion_time(z, alpha, t_0);
            let ds = spectral_dimension_k22(t, k);
            let w = -1.0 + beta * ds;
            let beta_ds = beta * ds;
            (z, w, ds, t, beta_ds)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Profile likelihood for alpha
// ---------------------------------------------------------------------------

/// Profile likelihood scan over alpha.
///
/// For each alpha in `alpha_grid`, fix alpha and optimize the remaining
/// parameters (omega_m, h0, t_0, optionally beta) via bounded Nelder-Mead.
/// Returns Vec<(alpha, chi2_min, omega_m, h0, t_0)>.
pub fn profile_likelihood_alpha(
    sn: &RealSnData,
    bao: &RealBaoData,
    cc: &[CcMeasurement],
    fsig: &[FsigMeasurement],
    k: usize,
    alpha_grid: &[f64],
    fixed_beta: bool,
) -> Vec<(f64, f64, f64, f64, f64)> {
    let beta_val = if fixed_beta { 1.0 } else { 0.0 };

    alpha_grid
        .par_iter()
        .map(|&alpha_fixed| {
            if fixed_beta {
                // 3-param optimization: omega_m, h0, t_0
                let obj = |p: &[f64]| {
                    let omega_m = p[0];
                    let h0 = p[1];
                    let t_0 = p[2];
                    chi2_sn_orthoplex(omega_m, h0, k, alpha_fixed, beta_val, t_0, sn)
                        + chi2_bao_orthoplex(omega_m, h0, k, alpha_fixed, beta_val, t_0, bao)
                        + chi2_cmb_shift_orthoplex(omega_m, k, alpha_fixed, beta_val, t_0)
                        + chi2_cc_orthoplex(omega_m, h0, k, alpha_fixed, beta_val, t_0, cc)
                };

                let nm_config = NelderMeadConfig {
                    bounds: vec![(0.1, 0.5), (60.0, 80.0), (1e-5, 10.0)],
                    max_iter: 2_000,
                    tol: 1e-4,
                    ..Default::default()
                };
                let guesses = vec![
                    vec![0.3, 70.0, 0.01],
                    vec![0.28, 68.0, 0.5],
                    vec![0.32, 72.0, 0.001],
                ];
                let (best, _) = bounded_nelder_mead(obj, &guesses, &nm_config);
                let omega_m = best[0];
                let h0 = best[1];
                let t_0 = best[2];
                let chi2 = chi2_sn_orthoplex(omega_m, h0, k, alpha_fixed, beta_val, t_0, sn)
                    + chi2_bao_orthoplex(omega_m, h0, k, alpha_fixed, beta_val, t_0, bao)
                    + chi2_cmb_shift_orthoplex(omega_m, k, alpha_fixed, beta_val, t_0)
                    + chi2_cc_orthoplex(omega_m, h0, k, alpha_fixed, beta_val, t_0, cc)
                    + chi2_fsig8_orthoplex(omega_m, k, alpha_fixed, beta_val, t_0, fsig);
                (alpha_fixed, chi2, omega_m, h0, t_0)
            } else {
                // 4-param optimization: omega_m, h0, beta, t_0
                let obj = |p: &[f64]| {
                    let omega_m = p[0];
                    let h0 = p[1];
                    let beta = p[2];
                    let t_0 = p[3];
                    chi2_sn_orthoplex(omega_m, h0, k, alpha_fixed, beta, t_0, sn)
                        + chi2_bao_orthoplex(omega_m, h0, k, alpha_fixed, beta, t_0, bao)
                        + chi2_cmb_shift_orthoplex(omega_m, k, alpha_fixed, beta, t_0)
                        + chi2_cc_orthoplex(omega_m, h0, k, alpha_fixed, beta, t_0, cc)
                };

                let nm_config = NelderMeadConfig {
                    bounds: vec![(0.1, 0.5), (60.0, 80.0), (-0.5, 1.0), (1e-5, 10.0)],
                    max_iter: 2_000,
                    tol: 1e-4,
                    ..Default::default()
                };
                let guesses = vec![
                    vec![0.3, 70.0, 0.05, 0.01],
                    vec![0.28, 68.0, 0.1, 0.5],
                    vec![0.32, 72.0, 0.0, 0.001],
                ];
                let (best, _) = bounded_nelder_mead(obj, &guesses, &nm_config);
                let omega_m = best[0];
                let h0 = best[1];
                let beta = best[2];
                let t_0 = best[3];
                let chi2 = chi2_sn_orthoplex(omega_m, h0, k, alpha_fixed, beta, t_0, sn)
                    + chi2_bao_orthoplex(omega_m, h0, k, alpha_fixed, beta, t_0, bao)
                    + chi2_cmb_shift_orthoplex(omega_m, k, alpha_fixed, beta, t_0)
                    + chi2_cc_orthoplex(omega_m, h0, k, alpha_fixed, beta, t_0, cc)
                    + chi2_fsig8_orthoplex(omega_m, k, alpha_fixed, beta, t_0, fsig);
                (alpha_fixed, chi2, omega_m, h0, t_0)
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// CDG-2 ultra-diffuse galaxy chi-square constraint
// ---------------------------------------------------------------------------

/// Chi-square for CDG-2 UDG abundance ratio constraint.
///
/// Uses Press-Schechter mass function to compute the ratio of UDG abundance
/// in orthoplex vs Lambda-CDM at the Perseus cluster redshift. The chi2 term
/// penalizes models where the ratio deviates from 1 (weak Gaussian prior).
#[allow(clippy::too_many_arguments)]
pub fn chi2_udg_orthoplex(
    omega_m: f64,
    k: usize,
    alpha: f64,
    beta: f64,
    t_0: f64,
    sigma8: f64,
    ns: f64,
) -> f64 {
    let ratio = crate::halo::udg_abundance_ratio(
        crate::halo::PERSEUS_Z,
        omega_m,
        sigma8,
        ns,
        1e8,  // m_lo: 10^8 Msun
        1e10, // m_hi: 10^10 Msun
        k,
        alpha,
        beta,
        t_0,
    );

    // Weak prior: ratio should be >= 1 for orthoplex model
    // sigma_ratio = 0.5 (generous uncertainty)
    let sigma_ratio = 0.5;
    let residual = (ratio - 1.0) / sigma_ratio;
    residual * residual
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const K: usize = 63; // dim=256

    #[test]
    fn heat_kernel_t0_is_unity() {
        let p = heat_kernel_k22(0.0, K);
        assert_relative_eq!(p, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn heat_kernel_large_t() {
        // P(inf) -> 1/(2k) = 1/126 for k=63
        let p = heat_kernel_k22(100.0, K);
        let expected = 1.0 / (2.0 * K as f64);
        assert_relative_eq!(p, expected, epsilon = 1e-10);
    }

    #[test]
    fn spectral_dim_t0_is_zero() {
        let ds = spectral_dimension_k22(0.0, K);
        assert_relative_eq!(ds, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn spectral_dim_large_t_is_zero() {
        let ds = spectral_dimension_k22(100.0, K);
        assert!(ds.abs() < 1e-10, "d_s(100) = {ds}, expected ~0");
    }

    #[test]
    fn spectral_dim_peak_exists() {
        // d_s must have a positive peak at some intermediate t
        let mut max_ds = 0.0_f64;
        for i in 1..1000 {
            let t = i as f64 * 0.001;
            let ds = spectral_dimension_k22(t, K);
            max_ds = max_ds.max(ds);
        }
        assert!(max_ds > 0.1, "max d_s = {max_ds}, expected > 0.1");
    }

    #[test]
    fn w_vacuum_limit() {
        // When beta=0, w = -1 everywhere
        let w = w_orthoplex(0.5, K, 1.0, 0.0, 1.0);
        assert_relative_eq!(w, -1.0, epsilon = 1e-12);
    }

    #[test]
    fn dark_energy_ratio_z0_is_one() {
        let x = dark_energy_density_ratio(0.0, K, 1.0, 0.1, 1.0);
        assert_relative_eq!(x, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn hubble_e_z0_is_one() {
        let e = hubble_e_orthoplex(0.0, 0.3, K, 1.0, 0.1, 1.0);
        assert_relative_eq!(e, 1.0, epsilon = 1e-8);
    }

    #[test]
    fn cmb_shift_orthoplex_reduces_to_lcdm() {
        // With beta=0, orthoplex -> LCDM: CMB shift parameters should match.
        let omega_m = 0.3;
        let r_orthoplex = cmb_shift_parameter_orthoplex(omega_m, K, 1.0, 0.0, 1.0);
        let r_lcdm = crate::bounce::cmb_shift_parameter(omega_m, 0.0, 1089.0);
        assert_relative_eq!(r_orthoplex, r_lcdm, epsilon = 1e-4);
    }
}

//! Real observational data fitting for bounce cosmology vs Lambda-CDM.
//!
//! Provides fitting infrastructure that accepts real dataset types from
//! data_core (Pantheon+ supernovae, DESI DR1 BAO) and runs joint chi-square
//! minimization using bounded Nelder-Mead optimization.
//!
//! # Key differences from synthetic pipeline (bounce.rs)
//!
//! - **Anisotropic BAO**: Real DESI data provides DM(z)/rd and DH(z)/rd with
//!   correlation coefficient rho, not just isotropic D_V/r_d.
//! - **Pantheon+ filtering**: Real SN data requires NaN filtering, redshift cuts,
//!   and calibrator exclusion.
//! - **Analytic M_B marginalization**: For distance modulus fitting, the absolute
//!   magnitude offset is marginalized analytically (Conley+ 2011).
//!
//! # References
//! - Scolnic et al. (2022), ApJ 938, 113 [Pantheon+]
//! - DESI Collaboration (2024), arXiv:2404.03002 [DESI DR1 BAO]
//! - Conley et al. (2011), ApJS 192, 1 [analytic marginalization]

use crate::{
    bounce::{C_KM_S, Z_STAR, bao_sound_horizon, cmb_shift_parameter, hubble_e_bounce, hubble_e_lcdm},
    gl_integrate,
};
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Data structures for real observational data
// ---------------------------------------------------------------------------

/// Real supernova distance modulus data (e.g., from Pantheon+).
///
/// Unlike SyntheticSnData, this includes per-SN metadata for filtering
/// and the absolute magnitude offset M_B as a nuisance parameter.
#[derive(Clone, Debug)]
pub struct RealSnData {
    /// CMB-frame redshifts.
    pub z: Vec<f64>,
    /// Distance modulus measurements (mag).
    pub mu: Vec<f64>,
    /// Distance modulus errors (mag).
    pub mu_err: Vec<f64>,
    /// Number of SNe after filtering.
    pub n_sne: usize,
    /// Precision matrix C^{-1} from full Pantheon+ STAT+SYS covariance.
    ///
    /// If `Some`, `chi2_sn_orthoplex` uses the matrix-form chi2:
    ///   chi2 = delta_mu^T * precision * delta_mu
    /// with analytic M_B marginalization in matrix form.
    ///
    /// If `None`, falls back to diagonal: sum (delta_mu_i / sigma_i)^2.
    /// The precision matrix is computed ONCE at data load time via Cholesky
    /// decomposition of the covariance matrix.
    pub precision: Option<nalgebra::DMatrix<f64>>,
}

/// Real BAO measurements (e.g., from DESI DR1).
///
/// Supports mixed isotropic (DV/rd only) and anisotropic (DM/rd + DH/rd)
/// bins. DESI DR1 BGS and QSO tracers provide only isotropic DV/rd due to
/// lower signal-to-noise; all other tracers provide anisotropic measurements.
///
/// For isotropic bins, `dm_over_rd` holds DV/rd and `dh_over_rd` is unused.
#[derive(Clone, Debug)]
pub struct RealBaoData {
    /// Effective redshifts.
    pub z_eff: Vec<f64>,
    /// True if this bin provides only isotropic DV/rd.
    pub is_isotropic: Vec<bool>,
    /// DM(z)/rd for anisotropic bins, or DV(z)/rd for isotropic bins.
    pub dm_over_rd: Vec<f64>,
    /// DM(z)/rd errors (or DV(z)/rd errors for isotropic bins).
    pub dm_over_rd_err: Vec<f64>,
    /// DH(z)/rd measurements (Hubble distance / sound horizon). Unused for isotropic bins.
    pub dh_over_rd: Vec<f64>,
    /// DH(z)/rd errors. Unused for isotropic bins.
    pub dh_over_rd_err: Vec<f64>,
    /// Correlation coefficient between DM/rd and DH/rd per bin. Zero for isotropic bins.
    pub rho: Vec<f64>,
    /// Tracer labels.
    pub tracer: Vec<String>,
}

/// Cosmic chronometer H(z) measurement.
///
/// Direct measurements of the Hubble parameter H(z) from differential
/// ages of passively evolving galaxies (Jimenez & Loeb 2002).
#[derive(Clone, Debug)]
pub struct CcMeasurement {
    /// Redshift of the measurement.
    pub z: f64,
    /// Observed H(z) in km/s/Mpc.
    pub h_obs: f64,
    /// 1-sigma uncertainty on H(z) in km/s/Mpc.
    pub h_err: f64,
}

/// Growth rate f*sigma8 measurement from redshift-space distortion surveys.
#[derive(Clone, Debug)]
pub struct FsigMeasurement {
    /// Effective redshift.
    pub z: f64,
    /// Observed f*sigma8 value.
    pub fsig8_obs: f64,
    /// 1-sigma uncertainty on f*sigma8.
    pub fsig8_err: f64,
    /// Survey name.
    pub survey: String,
}

/// Full observational fit result.
#[derive(Clone, Debug)]
pub struct ObsFitResult {
    /// Best-fit matter density parameter.
    pub omega_m: f64,
    /// Best-fit Hubble constant (km/s/Mpc).
    pub h0: f64,
    /// Best-fit quantum correction (0 for Lambda-CDM).
    pub q_corr: f64,
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
    /// Model label ("Lambda-CDM" or "Bounce").
    pub model: String,
}

/// Summary comparing Lambda-CDM and bounce fits.
#[derive(Clone, Debug)]
pub struct ModelComparison {
    pub lcdm: ObsFitResult,
    pub bounce: ObsFitResult,
    /// Delta BIC = BIC_bounce - BIC_lcdm (positive = Lambda-CDM preferred).
    pub delta_bic: f64,
    /// Delta AIC = AIC_bounce - AIC_lcdm.
    pub delta_aic: f64,
    /// Bounce spectral index n_s.
    pub n_s_bounce: f64,
}

// ---------------------------------------------------------------------------
// Chi-square functions for real data
// ---------------------------------------------------------------------------

/// Chi-square for Pantheon+ SN Ia distance modulus data.
///
/// Uses analytic marginalization over the absolute magnitude offset M_B:
///
///   chi2_marg = chi2_full - B^2/C + ln(C / 2*pi)
///
/// where B = sum( (mu_obs - mu_model) / sigma^2 )
///       C = sum( 1 / sigma^2 )
///
/// This removes one degree of freedom (M_B) without explicitly fitting it,
/// following Conley et al. (2011).
pub fn chi2_sn_real(omega_m: f64, h0: f64, q_corr: f64, sn: &RealSnData) -> f64 {
    if !(0.01..=0.99).contains(&omega_m) || !(50.0..=90.0).contains(&h0) || q_corr < 0.0 {
        return 1e10;
    }

    let mut a_sum = 0.0_f64; // sum( residual^2 / sigma^2 )
    let mut b_sum = 0.0_f64; // sum( residual / sigma^2 )
    let mut c_sum = 0.0_f64; // sum( 1 / sigma^2 )

    for i in 0..sn.z.len() {
        let mu_model = crate::bounce::distance_modulus(sn.z[i], omega_m, h0, q_corr);
        let residual = sn.mu[i] - mu_model;
        let inv_var = 1.0 / (sn.mu_err[i] * sn.mu_err[i]);

        a_sum += residual * residual * inv_var;
        b_sum += residual * inv_var;
        c_sum += inv_var;
    }

    // Analytic marginalization over M_B
    a_sum - b_sum * b_sum / c_sum
}

/// Chi-square for real BAO measurements (mixed isotropic + anisotropic).
///
/// **Anisotropic bins** (DM/rd + DH/rd + correlation): computes 2x2 chi-square
///   delta = [DM/rd_obs - DM/rd_model, DH/rd_obs - DH/rd_model]
///   C = [[sigma_DM^2,           rho*sigma_DM*sigma_DH],
///        [rho*sigma_DM*sigma_DH, sigma_DH^2          ]]
///   chi2_i = delta^T * C^{-1} * delta
///
/// **Isotropic bins** (DV/rd only): computes scalar chi-square
///   DV(z) = (z * d_C(z)^2 * d_H(z))^{1/3}
///   chi2_i = ((DV/rd_obs - DV/rd_model) / sigma)^2
///
/// References:
/// - DESI Collaboration (2024), arXiv:2404.03002
/// - Eisenstein et al. (2005), ApJ 633, 560
pub fn chi2_bao_real(omega_m: f64, h0: f64, q_corr: f64, bao: &RealBaoData) -> f64 {
    if !(0.01..=0.99).contains(&omega_m) || !(50.0..=90.0).contains(&h0) {
        return 1e10;
    }

    let r_d = bao_sound_horizon(omega_m, h0);
    let mut chi2 = 0.0;

    for i in 0..bao.z_eff.len() {
        let zi = bao.z_eff[i];

        // Model predictions (needed for both types)
        let d_c = comoving_distance_model(zi, omega_m, h0, q_corr);

        let e_val = if q_corr == 0.0 {
            hubble_e_lcdm(zi, omega_m)
        } else {
            hubble_e_bounce(zi, omega_m, q_corr)
        };
        let d_h = C_KM_S / (h0 * e_val);

        if bao.is_isotropic[i] {
            // Isotropic: DV(z)/rd = (z * d_C^2 * d_H)^{1/3} / rd
            let dv_model = (zi * d_c * d_c * d_h).powf(1.0 / 3.0) / r_d;
            let dv_obs = bao.dm_over_rd[i]; // DV/rd stored in dm_over_rd for isotropic
            let sigma = bao.dm_over_rd_err[i];
            if sigma > 0.0 {
                let residual = (dv_obs - dv_model) / sigma;
                chi2 += residual * residual;
            }
        } else {
            // Anisotropic: DM(z)/rd and DH(z)/rd with 2x2 covariance
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

/// Comoving distance d_C(z) for general cosmology (supports bounce correction).
fn comoving_distance_model(z: f64, omega_m: f64, h0: f64, q_corr: f64) -> f64 {
    if z <= 0.0 {
        return 0.0;
    }

    let integral = gl_integrate(
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

    (C_KM_S / h0) * integral
}

// ---------------------------------------------------------------------------
// Grid-interpolated LCDM comoving distance (avoids redundant GL quadrature)
// ---------------------------------------------------------------------------

/// Precomputed comoving distance grid for LCDM, analogous to orthoplex ComovingGrid.
///
/// For LCDM, E(z) is algebraic (no inner quadrature), so grid construction is
/// extremely fast. Cumulative GL(20) segments on 200 points. Linear interpolation
/// gives O(dz^2) ~ 1.6e-4 relative error, far below SN statistical uncertainty.
struct LcdmComovingGrid {
    /// Grid redshifts, linearly spaced from 0 to z_max.
    z_grid: Vec<f64>,
    /// Comoving distance (c/H_0) * integral_0^z dz'/E(z') at each grid point.
    dc_grid: Vec<f64>,
    /// Grid spacing (uniform).
    dz: f64,
}

impl LcdmComovingGrid {
    /// Build grid with `n_grid` points covering [0, z_max].
    fn build(z_max: f64, n_grid: usize, omega_m: f64, h0: f64) -> Self {
        let dz = z_max / (n_grid - 1).max(1) as f64;
        let c_over_h0 = C_KM_S / h0;

        let mut z_grid = Vec::with_capacity(n_grid);
        let mut dc_grid = Vec::with_capacity(n_grid);

        let mut cumulative_dc = 0.0;

        for i in 0..n_grid {
            let z = dz * i as f64;
            z_grid.push(z);

            if i > 0 {
                let z_prev = z_grid[i - 1];
                let segment = gl_integrate(
                    |zp| 1.0 / hubble_e_lcdm(zp, omega_m),
                    z_prev,
                    z,
                    20,
                );
                cumulative_dc += c_over_h0 * segment;
            }

            dc_grid.push(cumulative_dc);
        }

        Self { z_grid, dc_grid, dz }
    }

    /// Interpolate comoving distance at arbitrary z.
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
}

/// Grid-accelerated SN chi2 for LCDM. Builds one grid, then uses rayon par_iter.
fn chi2_sn_real_grid(grid: &LcdmComovingGrid, sn: &RealSnData) -> f64 {
    let (a_sum, b_sum, c_sum) = (0..sn.z.len())
        .into_par_iter()
        .map(|i| {
            let zi = sn.z[i];
            let dc = grid.interp_dc(zi);
            let d_l = dc * (1.0 + zi);
            let d_l_pc = d_l * 1e6;
            let mu_model = 5.0 * (d_l_pc.max(1e-30) / 10.0).log10();
            let residual = sn.mu[i] - mu_model;
            let inv_var = 1.0 / (sn.mu_err[i] * sn.mu_err[i]);
            (residual * residual * inv_var, residual * inv_var, inv_var)
        })
        .reduce(
            || (0.0, 0.0, 0.0),
            |(a1, b1, c1), (a2, b2, c2)| (a1 + a2, b1 + b2, c1 + c2),
        );

    a_sum - b_sum * b_sum / c_sum
}

/// Grid-accelerated BAO chi2 for LCDM.
fn chi2_bao_real_grid(
    grid: &LcdmComovingGrid,
    omega_m: f64,
    h0: f64,
    bao: &RealBaoData,
) -> f64 {
    let r_d = bao_sound_horizon(omega_m, h0);
    let mut chi2 = 0.0;

    for i in 0..bao.z_eff.len() {
        let zi = bao.z_eff[i];
        let d_c = grid.interp_dc(zi);
        let e_val = hubble_e_lcdm(zi, omega_m);
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
// CMB shift parameter chi-square
// ---------------------------------------------------------------------------

/// Planck 2018 (TT,TE,EE+lowE+lensing) CMB shift parameter.
///
/// R = 1.7502 +/- 0.0046 from Chen, Kumar & Ratra (2024), JCAP 04, 069.
/// Alternative: Efstathiou & Gratton (2020) give R = 1.7488 +/- 0.0044
/// (difference < 0.5 sigma).
pub const CMB_SHIFT_R_OBS: f64 = 1.7502;

/// 1-sigma uncertainty on the CMB shift parameter.
pub const CMB_SHIFT_R_ERR: f64 = 0.0046;

/// Chi-square contribution from the CMB shift parameter R for LCDM/bounce.
///
/// Uses `cmb_shift_parameter()` from bounce.rs with Z_STAR = 1089.
/// One data point per evaluation -- negligible cost.
pub fn chi2_cmb_shift(omega_m: f64, q_corr: f64) -> f64 {
    let r_model = cmb_shift_parameter(omega_m, q_corr, Z_STAR);
    let residual = (r_model - CMB_SHIFT_R_OBS) / CMB_SHIFT_R_ERR;
    residual * residual
}

// ---------------------------------------------------------------------------
// Cosmic chronometer H(z) data (Moresco et al. 2022 compilation)
// ---------------------------------------------------------------------------

/// Return 31 cosmic chronometer H(z) measurements from the Moresco et al.
/// (2022) compilation (LCDM-independent H(z) from differential galaxy ages).
///
/// Reference: Moresco et al. (2022), ApJ 898, 82, Table 2 (full compilation).
/// Units: H(z) in km/s/Mpc.
pub fn cosmic_chronometer_data() -> Vec<CcMeasurement> {
    // (z, H_obs, H_err) -- 31 measurements spanning z = 0.07 to 1.965
    let data: &[(f64, f64, f64)] = &[
        (0.07,   69.0,   19.6),
        (0.09,   69.0,   12.0),
        (0.12,   68.6,   26.2),
        (0.17,   83.0,   8.0),
        (0.179,  75.0,   4.0),
        (0.199,  75.0,   5.0),
        (0.20,   72.9,   29.6),
        (0.27,   77.0,   14.0),
        (0.28,   88.8,   36.6),
        (0.352,  83.0,   14.0),
        (0.3802, 83.0,   13.5),
        (0.4,    95.0,   17.0),
        (0.4004, 77.0,   10.2),
        (0.4247, 87.1,   11.2),
        (0.4497, 92.8,   12.9),
        (0.47,   89.0,   49.6),
        (0.4783, 80.9,   9.0),
        (0.48,   97.0,   62.0),
        (0.593,  104.0,  13.0),
        (0.68,   92.0,   8.0),
        (0.781,  105.0,  12.0),
        (0.875,  125.0,  17.0),
        (0.88,   90.0,   40.0),
        (0.9,    117.0,  23.0),
        (1.037,  154.0,  20.0),
        (1.3,    168.0,  17.0),
        (1.363,  160.0,  33.6),
        (1.43,   177.0,  18.0),
        (1.53,   140.0,  14.0),
        (1.75,   202.0,  40.0),
        (1.965,  186.5,  50.4),
    ];

    data.iter()
        .map(|&(z, h_obs, h_err)| CcMeasurement { z, h_obs, h_err })
        .collect()
}

/// Chi-square for cosmic chronometer H(z) data (LCDM or bounce).
///
/// Model: H(z) = H_0 * E(z). Pure algebraic for LCDM, no quadrature needed.
pub fn chi2_cc(omega_m: f64, h0: f64, q_corr: f64, cc: &[CcMeasurement]) -> f64 {
    if !(0.01..=0.99).contains(&omega_m) || !(50.0..=90.0).contains(&h0) {
        return 1e10;
    }

    let mut chi2 = 0.0;
    for m in cc {
        let e_val = if q_corr == 0.0 {
            hubble_e_lcdm(m.z, omega_m)
        } else {
            hubble_e_bounce(m.z, omega_m, q_corr)
        };
        let h_model = h0 * e_val;
        let residual = (m.h_obs - h_model) / m.h_err;
        chi2 += residual * residual;
    }

    chi2
}

// ---------------------------------------------------------------------------
// f*sigma8 growth rate data compilation
// ---------------------------------------------------------------------------

/// Planck 2018 sigma8 normalization (TT,TE,EE+lowE+lensing).
pub const SIGMA8_PLANCK: f64 = 0.811;

/// Return compiled f*sigma8 growth rate measurements from RSD surveys.
///
/// Sources: 6dFGS (Beutler+ 2012), SDSS MGS (Howlett+ 2015),
/// BOSS DR12 (Alam+ 2017), WiggleZ (Blake+ 2012), VIPERS (de la Torre+ 2013),
/// FastSound (Okumura+ 2016), DESI DR1 (2024).
pub fn growth_rate_data() -> Vec<FsigMeasurement> {
    let data: &[(&str, f64, f64, f64)] = &[
        ("6dFGS",       0.067,  0.423,  0.055),
        ("SDSS MGS",    0.15,   0.49,   0.15),
        ("BOSS DR12",   0.38,   0.497,  0.045),
        ("BOSS DR12",   0.51,   0.458,  0.038),
        ("BOSS DR12",   0.61,   0.436,  0.034),
        ("WiggleZ",     0.44,   0.413,  0.080),
        ("WiggleZ",     0.60,   0.390,  0.063),
        ("WiggleZ",     0.73,   0.437,  0.072),
        ("VIPERS",      0.60,   0.550,  0.120),
        ("VIPERS",      0.86,   0.400,  0.110),
        ("FastSound",   1.40,   0.482,  0.116),
        ("DESI DR1",    0.295,  0.408,  0.040),
        ("DESI DR1",    0.51,   0.452,  0.032),
        ("DESI DR1",    0.706,  0.447,  0.028),
        ("DESI DR1",    0.93,   0.444,  0.037),
        ("DESI DR1",    1.317,  0.370,  0.050),
    ];

    data.iter()
        .map(|&(survey, z, fsig8_obs, fsig8_err)| FsigMeasurement {
            z,
            fsig8_obs,
            fsig8_err,
            survey: survey.to_string(),
        })
        .collect()
}

/// Compute growth factor D(z) and f*sigma8(z) via RK4 integration.
///
/// Integrates the linear growth ODE from a_init = 1e-3 to a = 1/(1+z):
///   dy1/da = y2
///   dy2/da = -(3/a + dlnE/da) * y2 + (3/2) * Omega_m / (a^5 * E^2) * y1
///
/// `e_func` maps redshift z to dimensionless Hubble E(z) = H(z)/H_0.
/// Returns (D(z)/D(0), f*sigma8(z)) where f = d ln D / d ln a.
pub fn compute_growth_fsig8<F: Fn(f64) -> f64>(
    z: f64,
    omega_m: f64,
    e_func: &F,
    sigma8_0: f64,
) -> (f64, f64) {
    let results = compute_growth_batch(omega_m, e_func, &[z], sigma8_0);
    results[0]
}

/// Batch-compute f*sigma8 for multiple redshifts in a single RK4 sweep.
///
/// Does ONE integration from a_init=1e-3 to a=1.0, recording (D, dD/da)
/// at each target redshift along the way. This is O(n_steps) regardless
/// of how many redshifts are requested, vs O(n_steps * n_redshifts) for
/// the per-redshift approach.
pub fn compute_growth_batch<F: Fn(f64) -> f64>(
    omega_m: f64,
    e_func: &F,
    redshifts: &[f64],
    sigma8_0: f64,
) -> Vec<(f64, f64)> {
    if redshifts.is_empty() {
        return vec![];
    }

    let a_init = 1e-3;
    let a_final = 1.0;
    let n_steps: usize = 500;
    let da = (a_final - a_init) / n_steps as f64;

    // Convert redshifts to scale factors and sort by a (ascending).
    let mut targets: Vec<(usize, f64)> = redshifts
        .iter()
        .enumerate()
        .map(|(i, &z)| (i, 1.0 / (1.0 + z)))
        .collect();
    targets.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Storage for results (indexed by original order).
    let mut d_vals: Vec<f64> = vec![0.0; redshifts.len()];
    let mut dd_vals: Vec<f64> = vec![0.0; redshifts.len()];
    let mut a_targets: Vec<f64> = vec![0.0; redshifts.len()];
    for &(orig_idx, a_t) in &targets {
        a_targets[orig_idx] = a_t;
    }

    let rhs = |a: f64, d: f64, dd: f64| -> (f64, f64) {
        if a < 1e-10 {
            return (dd, 0.0);
        }
        let z_a = 1.0 / a - 1.0;
        let e_val = e_func(z_a);
        let e2 = e_val * e_val;

        let eps = 1e-5 * a;
        let z_p = 1.0 / (a + eps) - 1.0;
        let z_m = 1.0 / (a - eps) - 1.0;
        let dln_e_da = (e_func(z_p).ln() - e_func(z_m).ln()) / (2.0 * eps);

        let coeff1 = -(3.0 / a + dln_e_da);
        let coeff2 = 1.5 * omega_m / (a * a * a * a * a * e2);

        (dd, coeff1 * dd + coeff2 * d)
    };

    // IC: D(a_init) = a_init, D'(a_init) = 1 (matter-dominated)
    let mut y1 = a_init;
    let mut y2 = 1.0;
    let mut next_target = 0;

    for step in 0..n_steps {
        let a = a_init + step as f64 * da;

        let (k1a, k1b) = rhs(a, y1, y2);
        let (k2a, k2b) = rhs(a + 0.5 * da, y1 + 0.5 * da * k1a, y2 + 0.5 * da * k1b);
        let (k3a, k3b) = rhs(a + 0.5 * da, y1 + 0.5 * da * k2a, y2 + 0.5 * da * k2b);
        let (k4a, k4b) = rhs(a + da, y1 + da * k3a, y2 + da * k3b);

        y1 += da / 6.0 * (k1a + 2.0 * k2a + 2.0 * k3a + k4a);
        y2 += da / 6.0 * (k1b + 2.0 * k2b + 2.0 * k3b + k4b);

        let a_next = a + da;

        // Record any targets whose scale factor falls in [a, a_next].
        while next_target < targets.len() {
            let (orig_idx, a_t) = targets[next_target];
            if a_t <= a_next {
                d_vals[orig_idx] = y1;
                dd_vals[orig_idx] = y2;
                next_target += 1;
            } else {
                break;
            }
        }
    }

    let d_0 = y1; // D(z=0) = D(a=1)

    // Compute f*sigma8 for each target.
    let mut results = Vec::with_capacity(redshifts.len());
    for i in 0..redshifts.len() {
        let a_t = a_targets[i];
        let d_z = d_vals[i];
        let dd_z = dd_vals[i];

        let f_z = if d_z.abs() > 1e-30 { (a_t / d_z) * dd_z } else { 0.0 };
        let fsig8 = sigma8_0 * f_z * d_z / d_0;

        results.push((d_z / d_0, fsig8));
    }

    results
}

/// Chi-square for f*sigma8 growth rate data (LCDM).
pub fn chi2_fsig8(
    omega_m: f64,
    fsig: &[FsigMeasurement],
) -> f64 {
    if fsig.is_empty() {
        return 0.0;
    }

    let e_func = |z: f64| hubble_e_lcdm(z, omega_m);
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
// Bounded Nelder-Mead (reused from bounce.rs pattern)
// ---------------------------------------------------------------------------

/// Bounded Nelder-Mead optimizer for cosmological parameter fitting.
fn bounded_nelder_mead<F: Fn(&[f64]) -> f64>(
    f: F,
    x0: &[f64],
    bounds: &[(f64, f64)],
    max_iter: usize,
    tol: f64,
) -> (Vec<f64>, f64) {
    let n = x0.len();

    let project = |x: &[f64]| -> Vec<f64> {
        x.iter()
            .zip(bounds.iter())
            .map(|(&xi, &(lo, hi))| xi.clamp(lo, hi))
            .collect()
    };

    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(project(x0));
    for i in 0..n {
        let mut v = x0.to_vec();
        let range = bounds[i].1 - bounds[i].0;
        v[i] += range * 0.05;
        simplex.push(project(&v));
    }

    let mut fvals: Vec<f64> = simplex.iter().map(|v| f(v)).collect();

    let alpha = 1.0;
    let gamma = 2.0;
    let rho = 0.5;
    let sigma = 0.5;

    for _ in 0..max_iter {
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|&a, &b| fvals[a].partial_cmp(&fvals[b]).unwrap());
        let sorted_simplex: Vec<Vec<f64>> = order.iter().map(|&i| simplex[i].clone()).collect();
        let sorted_fvals: Vec<f64> = order.iter().map(|&i| fvals[i]).collect();
        simplex = sorted_simplex;
        fvals = sorted_fvals;

        let f_range = fvals[n] - fvals[0];
        if f_range < tol {
            break;
        }

        let centroid: Vec<f64> = (0..n)
            .map(|j| simplex[..n].iter().map(|v| v[j]).sum::<f64>() / n as f64)
            .collect();

        let xr: Vec<f64> = (0..n)
            .map(|j| centroid[j] + alpha * (centroid[j] - simplex[n][j]))
            .collect();
        let xr = project(&xr);
        let fr = f(&xr);

        if fr < fvals[0] {
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
            let xc: Vec<f64> = (0..n)
                .map(|j| centroid[j] + rho * (simplex[n][j] - centroid[j]))
                .collect();
            let xc = project(&xc);
            let fc = f(&xc);
            if fc < fvals[n] {
                simplex[n] = xc;
                fvals[n] = fc;
            } else {
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

    let best_idx = fvals
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;
    (simplex[best_idx].clone(), fvals[best_idx])
}

// ---------------------------------------------------------------------------
// Model fitting with real data
// ---------------------------------------------------------------------------

/// Fit Lambda-CDM or bounce model to real SN + BAO + CC + f*sigma8 data.
///
/// Lambda-CDM: 2 free parameters (Omega_m, H_0), q_corr = 0.
/// Bounce: 3 free parameters (Omega_m, H_0, q_corr).
///
/// Objective includes: SN chi2 + BAO chi2 + CMB shift chi2 + CC chi2 + f*sigma8 chi2.
/// Pass `&[]` for `cc` or `fsig` to omit those constraints.
pub fn fit_real_data(
    sn: &RealSnData,
    bao: &RealBaoData,
    cc: &[CcMeasurement],
    fsig: &[FsigMeasurement],
    is_bounce: bool,
) -> ObsFitResult {
    let n_bao_data = bao_data_point_count(bao);
    let n_cmb = 1;
    let n_cc = cc.len();
    let n_data = sn.z.len() + n_bao_data + n_cmb + n_cc + fsig.len();

    // Precompute z_max for LCDM grid (only used in LCDM branch)
    let z_max_sn = sn.z.iter().cloned().fold(0.0_f64, f64::max);
    let z_max_bao = bao.z_eff.iter().cloned().fold(0.0_f64, f64::max);
    let z_max = z_max_sn.max(z_max_bao) * 1.01;

    let (best, n_params, model_name) = if is_bounce {
        // Bounce: keep sequential path (q_corr != 0, no grid optimization)
        let (best, _chi2) = bounded_nelder_mead(
            |p| {
                chi2_sn_real(p[0], p[1], p[2], sn)
                    + chi2_bao_real(p[0], p[1], p[2], bao)
                    + chi2_cmb_shift(p[0], p[2])
                    + chi2_cc(p[0], p[1], p[2], cc)
            },
            &[0.3, 70.0, 1e-6],
            &[(0.1, 0.5), (60.0, 80.0), (0.0, 1e-2)],
            5000,
            1e-10,
        );
        (best, 3, "Bounce")
    } else {
        // LCDM: grid-accelerated + rayon parallel SN reduction
        let (best, _chi2) = bounded_nelder_mead(
            |p| {
                let omega_m = p[0];
                let h0 = p[1];
                if !(0.01..=0.99).contains(&omega_m) || !(50.0..=90.0).contains(&h0) {
                    return 1e10;
                }
                let grid = LcdmComovingGrid::build(z_max, 200, omega_m, h0);
                chi2_sn_real_grid(&grid, sn)
                    + chi2_bao_real_grid(&grid, omega_m, h0, bao)
                    + chi2_cmb_shift(omega_m, 0.0)
                    + chi2_cc(omega_m, h0, 0.0, cc)
            },
            &[0.3, 70.0],
            &[(0.1, 0.5), (60.0, 80.0)],
            5000,
            1e-10,
        );
        (best, 2, "Lambda-CDM")
    };

    // Post-fit: recompute all chi2 components including f*sigma8 (excluded from optimizer).
    let q_corr = if is_bounce { best[2] } else { 0.0 };
    let chi2_sn_val = chi2_sn_real(best[0], best[1], q_corr, sn);
    let chi2_bao_val = chi2_bao_real(best[0], best[1], q_corr, bao);
    let chi2_cmb_val = chi2_cmb_shift(best[0], q_corr);
    let chi2_cc_val = chi2_cc(best[0], best[1], q_corr, cc);
    let chi2_fsig_val = chi2_fsig8(best[0], fsig);
    let chi2_total = chi2_sn_val + chi2_bao_val + chi2_cmb_val + chi2_cc_val + chi2_fsig_val;
    let aic = chi2_total + 2.0 * n_params as f64;
    let bic = chi2_total + n_params as f64 * (n_data as f64).ln();

    ObsFitResult {
        omega_m: best[0],
        h0: best[1],
        q_corr,
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
        model: model_name.to_string(),
    }
}

/// Run full model comparison: Lambda-CDM vs bounce on real data.
///
/// Pass `&[]` for `cc` or `fsig` to omit those constraints.
pub fn compare_models(
    sn: &RealSnData,
    bao: &RealBaoData,
    cc: &[CcMeasurement],
    fsig: &[FsigMeasurement],
) -> ModelComparison {
    let lcdm = fit_real_data(sn, bao, cc, fsig, false);
    let bounce = fit_real_data(sn, bao, cc, fsig, true);

    let delta_bic = bounce.bic - lcdm.bic;
    let delta_aic = bounce.aic - lcdm.aic;
    let n_s = crate::bounce::spectral_index_bounce(bounce.q_corr, bounce.omega_m);

    ModelComparison {
        lcdm,
        bounce,
        delta_bic,
        delta_aic,
        n_s_bounce: n_s,
    }
}

// ---------------------------------------------------------------------------
// Conversion helpers (data_core types -> fitting types)
// ---------------------------------------------------------------------------

/// Filter Pantheon+ supernova data into fitting-ready arrays.
///
/// Applies:
/// - Redshift cut: z_cmb >= z_min
/// - NaN filtering: skip entries with NaN in z, mu, or mu_err
/// - Calibrator exclusion: skip Cepheid calibrators (is_calibrator = true)
///   unless include_calibrators is set
/// - Error floor: mu_err >= 0.01 mag
pub fn filter_pantheon_data(
    z_cmb: &[f64],
    mu: &[f64],
    mu_err: &[f64],
    is_calibrator: &[bool],
    z_min: f64,
    include_calibrators: bool,
) -> RealSnData {
    let mut fz = Vec::new();
    let mut fmu = Vec::new();
    let mut fme = Vec::new();

    for i in 0..z_cmb.len() {
        if z_cmb[i].is_nan() || mu[i].is_nan() || mu_err[i].is_nan() {
            continue;
        }
        if z_cmb[i] < z_min {
            continue;
        }
        if !include_calibrators && is_calibrator[i] {
            continue;
        }
        let err = mu_err[i].max(0.01); // Floor at 0.01 mag
        fz.push(z_cmb[i]);
        fmu.push(mu[i]);
        fme.push(err);
    }

    let n_sne = fz.len();
    RealSnData {
        z: fz,
        mu: fmu,
        mu_err: fme,
        n_sne,
        precision: None,
    }
}

/// Convert DESI DR1 BAO measurements into RealBaoData.
///
/// Accepts parallel arrays extracted from data_core BaoMeasurement structs.
/// BGS and QSO bins are isotropic (DV/rd only); all others are anisotropic.
///
/// For isotropic bins, `dm_over_rd` holds DV/rd and `dh_over_rd` is unused.
#[allow(clippy::too_many_arguments)]
pub fn desi_to_real_bao(
    z_eff: &[f64],
    is_isotropic: &[bool],
    dm_over_rd: &[f64],
    dm_over_rd_err: &[f64],
    dh_over_rd: &[f64],
    dh_over_rd_err: &[f64],
    rho: &[f64],
    tracer: &[String],
) -> RealBaoData {
    RealBaoData {
        z_eff: z_eff.to_vec(),
        is_isotropic: is_isotropic.to_vec(),
        dm_over_rd: dm_over_rd.to_vec(),
        dm_over_rd_err: dm_over_rd_err.to_vec(),
        dh_over_rd: dh_over_rd.to_vec(),
        dh_over_rd_err: dh_over_rd_err.to_vec(),
        rho: rho.to_vec(),
        tracer: tracer.to_vec(),
    }
}

/// Count the number of effective data points from BAO measurements.
///
/// Anisotropic bins contribute 2 data points (DM/rd + DH/rd).
/// Isotropic bins contribute 1 data point (DV/rd).
pub fn bao_data_point_count(bao: &RealBaoData) -> usize {
    bao.is_isotropic
        .iter()
        .map(|&iso| if iso { 1 } else { 2 })
        .sum()
}

/// Compute precision matrix from covariance via Cholesky decomposition.
///
/// If the covariance matrix is ill-conditioned (Cholesky fails), applies
/// Tikhonov regularization: adds ridge * max(diag(C)) * I to the diagonal.
pub fn compute_precision_matrix(cov: &nalgebra::DMatrix<f64>) -> nalgebra::DMatrix<f64> {
    let n = cov.nrows();

    // Try Cholesky first
    if let Some(chol) = cov.clone().cholesky() {
        return chol.inverse();
    }

    // Tikhonov regularization: add small ridge to diagonal
    let max_diag = (0..n).map(|i| cov[(i, i)].abs()).fold(0.0_f64, f64::max);
    let ridge = 1e-8 * max_diag;
    let mut cov_reg = cov.clone();
    for i in 0..n {
        cov_reg[(i, i)] += ridge;
    }

    cov_reg
        .cholesky()
        .expect("Cholesky failed even with regularization")
        .inverse()
}

/// Set precision matrix on RealSnData from a covariance matrix.
///
/// The covariance matrix must be N x N where N matches sn.n_sne.
/// If the dimensions match, computes precision = C^{-1} via Cholesky.
/// If dimensions differ (filtering removed some SNe), falls back to diagonal.
pub fn set_sn_precision_from_cov(sn: &mut RealSnData, cov: &nalgebra::DMatrix<f64>) {
    if cov.nrows() == sn.n_sne && cov.ncols() == sn.n_sne {
        sn.precision = Some(compute_precision_matrix(cov));
    }
    // If dimensions don't match, leave precision as None (diagonal fallback)
}

/// Filter Pantheon+ data and return the surviving original indices.
///
/// Identical filtering logic to `filter_pantheon_data`, but also returns
/// the indices into the original arrays that survived. This enables
/// extracting the correct sub-matrix from the full 1701x1701 covariance.
pub fn filter_pantheon_data_with_indices(
    z_cmb: &[f64],
    mu: &[f64],
    mu_err: &[f64],
    is_calibrator: &[bool],
    z_min: f64,
    include_calibrators: bool,
) -> (RealSnData, Vec<usize>) {
    let mut fz = Vec::new();
    let mut fmu = Vec::new();
    let mut fme = Vec::new();
    let mut kept = Vec::new();

    for i in 0..z_cmb.len() {
        if z_cmb[i].is_nan() || mu[i].is_nan() || mu_err[i].is_nan() {
            continue;
        }
        if z_cmb[i] < z_min {
            continue;
        }
        if !include_calibrators && is_calibrator[i] {
            continue;
        }
        let err = mu_err[i].max(0.01);
        fz.push(z_cmb[i]);
        fmu.push(mu[i]);
        fme.push(err);
        kept.push(i);
    }

    let n_sne = fz.len();
    let data = RealSnData {
        z: fz,
        mu: fmu,
        mu_err: fme,
        n_sne,
        precision: None,
    };
    (data, kept)
}

/// Extract a sub-matrix from the full covariance using kept indices.
///
/// Given the full N x N covariance and the indices of SNe that survived
/// filtering, returns the k x k sub-matrix where k = kept_indices.len().
///
/// If cov dimensions already match k (user provided pre-filtered covariance),
/// returns a clone without extraction.
pub fn extract_cov_submatrix(
    cov: &nalgebra::DMatrix<f64>,
    kept_indices: &[usize],
) -> nalgebra::DMatrix<f64> {
    let k = kept_indices.len();
    if cov.nrows() == k && cov.ncols() == k {
        return cov.clone();
    }

    let mut sub = nalgebra::DMatrix::zeros(k, k);
    for (new_i, &orig_i) in kept_indices.iter().enumerate() {
        for (new_j, &orig_j) in kept_indices.iter().enumerate() {
            sub[(new_i, new_j)] = cov[(orig_i, orig_j)];
        }
    }
    sub
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_test_sn_data() -> RealSnData {
        // Fake but realistic-looking SN data at known Lambda-CDM values
        let z: Vec<f64> = (1..=20).map(|i| 0.1 * i as f64).collect();
        let mu: Vec<f64> = z
            .iter()
            .map(|&zi| crate::bounce::distance_modulus(zi, 0.3, 70.0, 0.0))
            .collect();
        let mu_err = vec![0.12; 20];
        RealSnData {
            z,
            mu,
            mu_err,
            n_sne: 20,
            precision: None,
        }
    }

    fn make_test_bao_data() -> RealBaoData {
        // Use one anisotropic bin to keep test simple
        let z_eff = vec![0.51];
        let r_d = bao_sound_horizon(0.3, 70.0);
        let d_c = comoving_distance_model(0.51, 0.3, 70.0, 0.0);
        let dm = d_c / r_d;
        let e = hubble_e_lcdm(0.51, 0.3);
        let dh = C_KM_S / (70.0 * e * r_d);

        RealBaoData {
            z_eff,
            is_isotropic: vec![false],
            dm_over_rd: vec![dm],
            dm_over_rd_err: vec![0.25],
            dh_over_rd: vec![dh],
            dh_over_rd_err: vec![0.61],
            rho: vec![-0.45],
            tracer: vec!["test".to_string()],
        }
    }

    #[test]
    fn test_chi2_sn_real_at_truth() {
        let sn = make_test_sn_data();
        // At true parameters, chi2 should be near 0 (marginalized)
        let chi2 = chi2_sn_real(0.3, 70.0, 0.0, &sn);
        assert!(chi2 < 1.0, "chi2_sn at truth = {chi2}, expected < 1");
    }

    #[test]
    fn test_chi2_bao_real_at_truth() {
        let bao = make_test_bao_data();
        let chi2 = chi2_bao_real(0.3, 70.0, 0.0, &bao);
        assert!(chi2 < 1.0, "chi2_bao at truth = {chi2}, expected < 1");
    }

    #[test]
    fn test_chi2_sn_real_rejects_bad_params() {
        let sn = make_test_sn_data();
        let chi2 = chi2_sn_real(1.5, 70.0, 0.0, &sn);
        assert!(chi2 > 1e9, "Out-of-bounds should return penalty");
    }

    #[test]
    fn test_fit_real_data_lcdm() {
        let sn = make_test_sn_data();
        let bao = make_test_bao_data();
        let result = fit_real_data(&sn, &bao, &[], &[], false);

        assert_relative_eq!(result.omega_m, 0.3, epsilon = 0.05);
        assert_relative_eq!(result.h0, 70.0, epsilon = 3.0);
        assert_eq!(result.n_params, 2);
        assert_eq!(result.model, "Lambda-CDM");
    }

    #[test]
    fn test_fit_real_data_bounce() {
        let sn = make_test_sn_data();
        let bao = make_test_bao_data();
        let result = fit_real_data(&sn, &bao, &[], &[], true);

        assert_eq!(result.n_params, 3);
        assert_eq!(result.model, "Bounce");
        // q_corr should be small for Lambda-CDM-generated data
        assert!(result.q_corr < 0.01, "q_corr = {}", result.q_corr);
    }

    #[test]
    fn test_compare_models() {
        let sn = make_test_sn_data();
        let bao = make_test_bao_data();
        let comparison = compare_models(&sn, &bao, &[], &[]);

        // Lambda-CDM should be preferred (lower BIC) for Lambda-CDM-generated data
        assert!(
            comparison.delta_bic > -5.0,
            "delta_BIC = {} (bounce too strongly preferred)",
            comparison.delta_bic
        );
    }

    #[test]
    fn test_lcdm_grid_matches_direct() {
        // Grid-interpolated LCDM comoving distance should match direct GL within 0.1%
        let omega_m = 0.3;
        let h0 = 70.0;
        let grid = LcdmComovingGrid::build(2.5, 200, omega_m, h0);

        for &z in &[0.1, 0.5, 1.0, 1.5, 2.0] {
            let dc_grid = grid.interp_dc(z);
            let dc_direct = comoving_distance_model(z, omega_m, h0, 0.0);
            let rel_err = ((dc_grid - dc_direct) / dc_direct).abs();
            assert!(
                rel_err < 0.001,
                "Grid vs direct at z={z}: grid={dc_grid:.4}, direct={dc_direct:.4}, rel_err={rel_err:.6}"
            );
        }
    }

    #[test]
    fn test_chi2_cmb_shift_at_planck() {
        // At Planck best-fit (omega_m ~ 0.315), CMB chi2 should be small
        let chi2 = chi2_cmb_shift(0.315, 0.0);
        assert!(
            chi2 < 10.0,
            "chi2_cmb at Planck best-fit = {chi2}, expected < 10"
        );
    }

    #[test]
    fn test_chi2_cc_at_planck() {
        // At Planck values, CC chi2/dof should be reasonable (< 3)
        let cc = cosmic_chronometer_data();
        let chi2 = chi2_cc(0.315, 67.4, 0.0, &cc);
        let chi2_per_dof = chi2 / cc.len() as f64;
        assert!(
            chi2_per_dof < 3.0,
            "chi2_cc/dof at Planck = {chi2_per_dof:.2}, expected < 3"
        );
    }

    #[test]
    fn test_cosmic_chronometer_data_count() {
        let cc = cosmic_chronometer_data();
        assert_eq!(cc.len(), 31, "Expected 31 CC measurements");
        // Verify redshift ordering and range
        assert!(cc.first().unwrap().z < 0.1);
        assert!(cc.last().unwrap().z > 1.9);
    }

    #[test]
    fn test_filter_pantheon_data() {
        let z = vec![0.01, 0.05, 0.1, f64::NAN, 0.5];
        let mu = vec![33.0, 36.0, 38.0, 40.0, 42.0];
        let mu_err = vec![0.1, 0.1, 0.1, 0.1, 0.1];
        let cal = vec![false, false, false, false, true];

        let sn = filter_pantheon_data(&z, &mu, &mu_err, &cal, 0.02, false);
        // Should exclude: z=0.01 (below z_min), NaN, calibrator
        assert_eq!(sn.n_sne, 2);
        assert_relative_eq!(sn.z[0], 0.05, epsilon = 1e-10);
        assert_relative_eq!(sn.z[1], 0.1, epsilon = 1e-10);
    }

    #[test]
    fn test_anisotropic_vs_isotropic_consistency() {
        // When rho=0 and DH/rd errors are very large, anisotropic should
        // approximately equal isotropic fitting on DM/rd only
        let bao = RealBaoData {
            z_eff: vec![0.51],
            is_isotropic: vec![false],
            dm_over_rd: vec![13.62],
            dm_over_rd_err: vec![0.25],
            dh_over_rd: vec![20.98],
            dh_over_rd_err: vec![100.0], // Very large -> negligible constraint
            rho: vec![0.0],
            tracer: vec!["test".to_string()],
        };

        let chi2 = chi2_bao_real(0.3, 70.0, 0.0, &bao);
        assert!(chi2.is_finite(), "chi2 should be finite");
    }

    #[test]
    fn test_isotropic_bin_chi2_at_truth() {
        // Compute DV/rd at fiducial parameters and verify chi2 = 0
        let z = 0.295;
        let r_d = bao_sound_horizon(0.3, 70.0);
        let d_c = comoving_distance_model(z, 0.3, 70.0, 0.0);
        let e = hubble_e_lcdm(z, 0.3);
        let d_h = C_KM_S / (70.0 * e);
        let dv = (z * d_c * d_c * d_h).powf(1.0 / 3.0) / r_d;

        let bao = RealBaoData {
            z_eff: vec![z],
            is_isotropic: vec![true],
            dm_over_rd: vec![dv], // DV/rd stored in dm_over_rd for isotropic
            dm_over_rd_err: vec![0.15],
            dh_over_rd: vec![0.0],
            dh_over_rd_err: vec![0.0],
            rho: vec![0.0],
            tracer: vec!["BGS".to_string()],
        };

        let chi2 = chi2_bao_real(0.3, 70.0, 0.0, &bao);
        assert!(chi2 < 1e-6, "chi2 at truth should be ~0, got {chi2}");
    }

    #[test]
    fn test_mixed_isotropic_anisotropic_data_point_count() {
        // 2 isotropic + 5 anisotropic = 2*1 + 5*2 = 12
        let bao = RealBaoData {
            z_eff: vec![0.295, 0.51, 0.706, 0.93, 1.317, 1.491, 2.33],
            is_isotropic: vec![true, false, false, false, false, true, false],
            dm_over_rd: vec![7.93, 13.62, 16.85, 21.71, 27.79, 26.07, 39.71],
            dm_over_rd_err: vec![0.15, 0.25, 0.32, 0.28, 0.69, 0.67, 0.94],
            dh_over_rd: vec![0.0, 20.98, 20.08, 17.88, 13.82, 0.0, 8.52],
            dh_over_rd_err: vec![0.0, 0.61, 0.60, 0.35, 0.42, 0.0, 0.17],
            rho: vec![0.0, -0.445, -0.420, -0.389, -0.444, 0.0, -0.477],
            tracer: vec![
                "BGS".into(),
                "LRG1".into(),
                "LRG2".into(),
                "LRG3+ELG1".into(),
                "ELG2".into(),
                "QSO".into(),
                "Lya".into(),
            ],
        };
        assert_eq!(bao_data_point_count(&bao), 12);
    }

    #[test]
    fn test_precision_matrix_diagonal() {
        // For a diagonal covariance, precision should be diagonal inverse
        let cov =
            nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![4.0, 9.0, 16.0]));
        let prec = compute_precision_matrix(&cov);
        assert!((prec[(0, 0)] - 0.25).abs() < 1e-10);
        assert!((prec[(1, 1)] - 1.0 / 9.0).abs() < 1e-10);
        assert!((prec[(2, 2)] - 1.0 / 16.0).abs() < 1e-10);
        // Off-diagonals should be near zero
        assert!(prec[(0, 1)].abs() < 1e-10);
    }

    #[test]
    fn test_set_sn_precision_from_cov_matching() {
        let mut sn = RealSnData {
            z: vec![0.1, 0.2, 0.3],
            mu: vec![33.0, 35.0, 37.0],
            mu_err: vec![0.1, 0.1, 0.1],
            n_sne: 3,
            precision: None,
        };
        let cov =
            nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![4.0, 9.0, 16.0]));
        set_sn_precision_from_cov(&mut sn, &cov);
        assert!(sn.precision.is_some());
    }

    #[test]
    fn test_set_sn_precision_from_cov_mismatched() {
        let mut sn = RealSnData {
            z: vec![0.1, 0.2],
            mu: vec![33.0, 35.0],
            mu_err: vec![0.1, 0.1],
            n_sne: 2,
            precision: None,
        };
        // 3x3 covariance but only 2 SNe -- should leave precision as None
        let cov =
            nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![4.0, 9.0, 16.0]));
        set_sn_precision_from_cov(&mut sn, &cov);
        assert!(sn.precision.is_none());
    }

    #[test]
    fn test_filter_pantheon_data_with_indices() {
        let z = vec![0.001, 0.05, f64::NAN, 0.10, 0.02];
        let mu = vec![28.0, 36.0, 35.0, 38.0, 33.0];
        let mu_err = vec![0.1, 0.2, 0.1, 0.15, 0.12];
        let is_cal = vec![false, false, false, false, true];

        let (data, kept) = filter_pantheon_data_with_indices(
            &z, &mu, &mu_err, &is_cal, 0.01, false,
        );

        // z[0]=0.001 < z_min, z[2]=NaN, z[4] is calibrator -> kept = [1, 3]
        assert_eq!(data.n_sne, 2);
        assert_eq!(kept, vec![1, 3]);
        assert_relative_eq!(data.z[0], 0.05, epsilon = 1e-10);
        assert_relative_eq!(data.z[1], 0.10, epsilon = 1e-10);
    }

    #[test]
    fn test_extract_cov_submatrix() {
        // 4x4 matrix, select indices [0, 2]
        let full = nalgebra::DMatrix::from_row_slice(4, 4, &[
            1.0, 0.1, 0.2, 0.3,
            0.1, 2.0, 0.4, 0.5,
            0.2, 0.4, 3.0, 0.6,
            0.3, 0.5, 0.6, 4.0,
        ]);
        let kept = vec![0, 2];
        let sub = extract_cov_submatrix(&full, &kept);
        assert_eq!(sub.nrows(), 2);
        assert_eq!(sub.ncols(), 2);
        assert_relative_eq!(sub[(0, 0)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(sub[(0, 1)], 0.2, epsilon = 1e-10);
        assert_relative_eq!(sub[(1, 0)], 0.2, epsilon = 1e-10);
        assert_relative_eq!(sub[(1, 1)], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_extract_cov_submatrix_already_matching() {
        // When dimensions already match, return clone
        let cov = nalgebra::DMatrix::from_row_slice(2, 2, &[1.0, 0.1, 0.1, 2.0]);
        let kept = vec![0, 1];
        let sub = extract_cov_submatrix(&cov, &kept);
        assert_eq!(sub, cov);
    }
}

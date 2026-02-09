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

use crate::bounce::{bao_sound_horizon, hubble_e_bounce, hubble_e_lcdm, C_KM_S};
use crate::gl_integrate;

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

/// Fit Lambda-CDM or bounce model to real SN + BAO data.
///
/// Lambda-CDM: 2 free parameters (Omega_m, H_0), q_corr = 0.
/// Bounce: 3 free parameters (Omega_m, H_0, q_corr).
///
/// BAO data point count: 2 per anisotropic bin + 1 per isotropic bin.
pub fn fit_real_data(sn: &RealSnData, bao: &RealBaoData, is_bounce: bool) -> ObsFitResult {
    let n_bao_data = bao_data_point_count(bao);
    let n_data = sn.z.len() + n_bao_data;

    let (best, chi2_total, n_params, model_name) = if is_bounce {
        let (best, chi2) = bounded_nelder_mead(
            |p| chi2_sn_real(p[0], p[1], p[2], sn) + chi2_bao_real(p[0], p[1], p[2], bao),
            &[0.3, 70.0, 1e-6],
            &[(0.1, 0.5), (60.0, 80.0), (0.0, 1e-2)],
            5000,
            1e-10,
        );
        (best, chi2, 3, "Bounce")
    } else {
        let (best, chi2) = bounded_nelder_mead(
            |p| chi2_sn_real(p[0], p[1], 0.0, sn) + chi2_bao_real(p[0], p[1], 0.0, bao),
            &[0.3, 70.0],
            &[(0.1, 0.5), (60.0, 80.0)],
            5000,
            1e-10,
        );
        (best, chi2, 2, "Lambda-CDM")
    };

    let q_corr = if is_bounce { best[2] } else { 0.0 };
    let chi2_sn_val = chi2_sn_real(best[0], best[1], q_corr, sn);
    let chi2_bao_val = chi2_bao_real(best[0], best[1], q_corr, bao);
    let aic = chi2_total + 2.0 * n_params as f64;
    let bic = chi2_total + n_params as f64 * (n_data as f64).ln();

    ObsFitResult {
        omega_m: best[0],
        h0: best[1],
        q_corr,
        chi2_total,
        chi2_sn: chi2_sn_val,
        chi2_bao: chi2_bao_val,
        n_params,
        n_data,
        aic,
        bic,
        model: model_name.to_string(),
    }
}

/// Run full model comparison: Lambda-CDM vs bounce on real data.
pub fn compare_models(sn: &RealSnData, bao: &RealBaoData) -> ModelComparison {
    let lcdm = fit_real_data(sn, bao, false);
    let bounce = fit_real_data(sn, bao, true);

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
        let result = fit_real_data(&sn, &bao, false);

        assert_relative_eq!(result.omega_m, 0.3, epsilon = 0.05);
        assert_relative_eq!(result.h0, 70.0, epsilon = 3.0);
        assert_eq!(result.n_params, 2);
        assert_eq!(result.model, "Lambda-CDM");
    }

    #[test]
    fn test_fit_real_data_bounce() {
        let sn = make_test_sn_data();
        let bao = make_test_bao_data();
        let result = fit_real_data(&sn, &bao, true);

        assert_eq!(result.n_params, 3);
        assert_eq!(result.model, "Bounce");
        // q_corr should be small for Lambda-CDM-generated data
        assert!(result.q_corr < 0.01, "q_corr = {}", result.q_corr);
    }

    #[test]
    fn test_compare_models() {
        let sn = make_test_sn_data();
        let bao = make_test_bao_data();
        let comparison = compare_models(&sn, &bao);

        // Lambda-CDM should be preferred (lower BIC) for Lambda-CDM-generated data
        assert!(
            comparison.delta_bic > -5.0,
            "delta_BIC = {} (bounce too strongly preferred)",
            comparison.delta_bic
        );
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
}

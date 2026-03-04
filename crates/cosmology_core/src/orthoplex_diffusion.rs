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
    observational::{RealBaoData, RealSnData},
};

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

/// Result of fitting the orthoplex diffusion model to SN + BAO data.
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
    /// Total chi-square.
    pub chi2_total: f64,
    /// SN contribution to chi-square.
    pub chi2_sn: f64,
    /// BAO contribution to chi-square.
    pub chi2_bao: f64,
    /// Number of free parameters (5).
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
// Chi-square functions for observational fitting
// ---------------------------------------------------------------------------

/// Chi-square for Pantheon+ SN data with orthoplex dark energy model.
///
/// Uses analytic M_B marginalization (Conley+ 2011).
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

    let mut a_sum = 0.0_f64;
    let mut b_sum = 0.0_f64;
    let mut c_sum = 0.0_f64;

    for i in 0..sn.z.len() {
        let mu_model = distance_modulus_orthoplex(sn.z[i], omega_m, h0, k, alpha, beta, t_0);
        let residual = sn.mu[i] - mu_model;
        let inv_var = 1.0 / (sn.mu_err[i] * sn.mu_err[i]);

        a_sum += residual * residual * inv_var;
        b_sum += residual * inv_var;
        c_sum += inv_var;
    }

    a_sum - b_sum * b_sum / c_sum
}

/// Chi-square for real BAO data with orthoplex dark energy model.
///
/// Supports mixed isotropic + anisotropic bins (same format as observational.rs).
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

    let r_d = bao_sound_horizon(omega_m, h0);
    let mut chi2 = 0.0;

    for i in 0..bao.z_eff.len() {
        let zi = bao.z_eff[i];

        // Comoving distance via GL quadrature
        let d_c = if zi <= 0.0 {
            0.0
        } else {
            let integral = gl_integrate(
                |zp| 1.0 / hubble_e_orthoplex(zp, omega_m, k, alpha, beta, t_0),
                0.0,
                zi,
                50,
            );
            (C_KM_S / h0) * integral
        };

        let e_val = hubble_e_orthoplex(zi, omega_m, k, alpha, beta, t_0);
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
// Model fitting
// ---------------------------------------------------------------------------

/// Fit orthoplex dark energy model to real SN + BAO data.
///
/// Free parameters: omega_m, h0, alpha, beta, t_0 (5 parameters).
/// Fixed: k (graph topology, determined by CD algebra dimension).
///
/// Uses multi-start Nelder-Mead: 9 initial guesses spanning the expanded
/// parameter space (alpha up to 25, t_0 down to 1e-5) to avoid local
/// minima in the 5D chi2 landscape. Each start runs 10k iterations.
pub fn fit_orthoplex_model(sn: &RealSnData, bao: &RealBaoData, k: usize) -> OrthoplexFitResult {
    let n_bao_data = crate::observational::bao_data_point_count(bao);
    let n_data = sn.z.len() + n_bao_data;

    let bounds = [
        (0.1, 0.5),    // omega_m
        (60.0, 80.0),  // h0
        (0.1, 25.0),   // alpha  (uncaged from 5.0)
        (-0.5, 1.0),   // beta   (negative allows phantom)
        (1e-5, 10.0),  // t_0    (uncaged from 0.01)
    ];

    let obj = |p: &[f64]| {
        chi2_sn_orthoplex(p[0], p[1], k, p[2], p[3], p[4], sn)
            + chi2_bao_orthoplex(p[0], p[1], k, p[2], p[3], p[4], bao)
    };

    // Multi-start: sample initial guesses across the expanded space.
    // Grid covers low/mid/high alpha x low/mid/high t_0, with omega_m/h0/beta
    // seeded near their physical values.
    let initial_guesses: &[&[f64]] = &[
        // [omega_m, h0, alpha, beta, t_0]
        &[0.3,  70.0, 1.0,  0.0,   1.0  ],  // baseline (near LCDM)
        &[0.3,  70.0, 5.0,  0.05,  0.01 ],  // previous boundary hit
        &[0.3,  70.0, 8.0,  0.1,   0.005],  // deeper alpha, sharper t_0
        &[0.3,  70.0, 12.0, 0.05,  0.001],  // high alpha, very sharp onset
        &[0.3,  70.0, 20.0, 0.02,  1e-4 ],  // extreme alpha frontier
        &[0.28, 68.0, 3.0,  0.15,  0.5  ],  // lower H0, moderate alpha
        &[0.32, 72.0, 6.0, -0.1,   0.1  ],  // phantom regime (beta < 0)
        &[0.3,  70.0, 10.0, 0.03,  0.003],  // mid-range exploration
        &[0.3,  70.0, 15.0, 0.08,  5e-4 ],  // high-alpha sharp-onset
    ];

    let mut global_best: Vec<f64> = initial_guesses[0].to_vec();
    let mut global_chi2 = f64::INFINITY;

    for &x0 in initial_guesses {
        let (candidate, chi2) = bounded_nelder_mead(obj, x0, &bounds, 10_000, 1e-10);
        if chi2 < global_chi2 {
            global_best = candidate;
            global_chi2 = chi2;
        }
    }

    let (best, chi2_total) = (global_best, global_chi2);

    let omega_m = best[0];
    let h0 = best[1];
    let alpha = best[2];
    let beta = best[3];
    let t_0 = best[4];

    let chi2_sn_val = chi2_sn_orthoplex(omega_m, h0, k, alpha, beta, t_0, sn);
    let chi2_bao_val = chi2_bao_orthoplex(omega_m, h0, k, alpha, beta, t_0, bao);
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
        chi2_total,
        chi2_sn: chi2_sn_val,
        chi2_bao: chi2_bao_val,
        n_params,
        n_data,
        aic,
        bic,
        w_0,
        w_high_z,
    }
}

/// Run full model comparison: Lambda-CDM vs orthoplex on real data.
pub fn compare_orthoplex(sn: &RealSnData, bao: &RealBaoData, k: usize) -> OrthoplexComparison {
    let lcdm = crate::observational::fit_real_data(sn, bao, false);
    let orthoplex = fit_orthoplex_model(sn, bao, k);

    let delta_bic = orthoplex.bic - lcdm.bic;
    let delta_aic = orthoplex.aic - lcdm.aic;

    OrthoplexComparison {
        lcdm,
        orthoplex,
        delta_bic,
        delta_aic,
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
// Bounded Nelder-Mead optimizer
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

    let nm_alpha = 1.0;
    let nm_gamma = 2.0;
    let nm_rho = 0.5;
    let nm_sigma = 0.5;

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
            .map(|j| centroid[j] + nm_alpha * (centroid[j] - simplex[n][j]))
            .collect();
        let xr = project(&xr);
        let fr = f(&xr);

        if fr < fvals[0] {
            let xe: Vec<f64> = (0..n)
                .map(|j| centroid[j] + nm_gamma * (xr[j] - centroid[j]))
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
                .map(|j| centroid[j] + nm_rho * (simplex[n][j] - centroid[j]))
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
                        *sij = bj + nm_sigma * (*sij - bj);
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
}

//! Multi-system density scaling fit: epsilon_bar = K * (dNch/dy / A_perp) * L^beta.
//!
//! Combines per-centrality epsilon_bar extractions from multiple collision
//! systems (Pb-Pb 5.02 TeV, Pb-Pb 2.76 TeV, Au-Au 200 GeV, Xe-Xe 5.44 TeV)
//! to determine the path-length exponent beta and proportionality constant K.
//!
//! Reference: Arleo & Falmagne arXiv:2411.13258, Sec. IV (Eq. 5).

/// Input data point for the density scaling fit.
#[derive(Debug, Clone)]
pub struct DensityScalingPoint {
    /// Average energy loss epsilon_bar (GeV).
    pub epsilon_bar: f64,
    /// Uncertainty on epsilon_bar (GeV).
    pub epsilon_bar_err: f64,
    /// Charged-particle multiplicity density dNch/dy at midrapidity.
    pub dnch_dy: f64,
    /// Transverse overlap area A_perp (fm^2).
    pub a_perp: f64,
    /// Average path length L (fm).
    pub l_avg: f64,
    /// Collision system label (for bookkeeping).
    pub system: String,
    /// Centrality bin label.
    pub centrality: String,
}

/// Result of the density scaling fit.
#[derive(Debug, Clone)]
pub struct DensityScalingResult {
    /// Best-fit path-length exponent.
    pub beta: f64,
    /// Upper error on beta (delta-chi2 = 1).
    pub beta_err_up: f64,
    /// Lower error on beta (delta-chi2 = 1).
    pub beta_err_down: f64,
    /// Best-fit proportionality constant K (fm^{1-beta}).
    pub k_constant: f64,
    /// Error on K.
    pub k_err: f64,
    /// Chi2 at best fit.
    pub chi2_min: f64,
    /// Number of degrees of freedom.
    pub ndf: usize,
    /// Chi2/ndf (reduced chi2).
    pub chi2_per_ndf: f64,
}

/// Perform the density scaling fit over a grid of beta values.
///
/// For each beta, computes X_i = (dNch/dy_i / A_perp_i) * L_i^beta,
/// then fits epsilon_bar_i = K * X_i via weighted least-squares (1 free parameter).
///
/// The chi2(beta) profile is scanned to find beta_min and delta-chi2=1 error bands.
///
/// # Arguments
/// * `data` - Per-centrality measurements from all collision systems
/// * `beta_lo` - Lower bound of beta scan
/// * `beta_hi` - Upper bound of beta scan
/// * `beta_step` - Step size for beta grid
pub fn fit_density_scaling(
    data: &[DensityScalingPoint],
    beta_lo: f64,
    beta_hi: f64,
    beta_step: f64,
) -> DensityScalingResult {
    let n_steps = ((beta_hi - beta_lo) / beta_step).ceil() as usize + 1;

    let mut best_beta = beta_lo;
    let mut best_chi2 = f64::INFINITY;
    let mut best_k = 0.0;
    let mut chi2_profile = Vec::with_capacity(n_steps);

    for step in 0..n_steps {
        let beta = beta_lo + step as f64 * beta_step;
        let (k, c2) = chi2_at_beta(data, beta);
        chi2_profile.push((beta, c2));

        if c2 < best_chi2 {
            best_chi2 = c2;
            best_beta = beta;
            best_k = k;
        }
    }

    // Refine with golden-section search around the minimum
    let refine_lo = (best_beta - 2.0 * beta_step).max(beta_lo);
    let refine_hi = (best_beta + 2.0 * beta_step).min(beta_hi);
    let (refined_beta, refined_chi2) =
        golden_section_minimize(|b| chi2_at_beta(data, b).1, refine_lo, refine_hi, 1e-6);
    if refined_chi2 < best_chi2 {
        best_beta = refined_beta;
        best_chi2 = refined_chi2;
        best_k = chi2_at_beta(data, refined_beta).0;
    }

    // NDF = n_data - 2 (fitting K and beta)
    let ndf = if data.len() > 2 { data.len() - 2 } else { 1 };

    // Delta-chi2 = 1 error scan for beta
    let target = best_chi2 + 1.0;
    let beta_err_up = scan_for_crossing(&chi2_profile, best_beta, true, target) - best_beta;
    let beta_err_down = best_beta - scan_for_crossing(&chi2_profile, best_beta, false, target);

    // K error from the weighted fit at best beta
    let k_err = k_error(data, best_beta);

    DensityScalingResult {
        beta: best_beta,
        beta_err_up: beta_err_up.max(0.0),
        beta_err_down: beta_err_down.max(0.0),
        k_constant: best_k,
        k_err,
        chi2_min: best_chi2,
        ndf,
        chi2_per_ndf: best_chi2 / ndf as f64,
    }
}

/// Compute chi2 and K for a given beta via weighted least-squares.
///
/// Model: epsilon_bar_i = K * X_i where X_i = (dNch/dy_i / A_perp_i) * L_i^beta.
/// Weighted least-squares: K = sum(w_i * eps_i * X_i) / sum(w_i * X_i^2)
/// where w_i = 1/sigma_i^2.
fn chi2_at_beta(data: &[DensityScalingPoint], beta: f64) -> (f64, f64) {
    let mut sum_wx_eps = 0.0;
    let mut sum_wx2 = 0.0;

    for d in data {
        let x = (d.dnch_dy / d.a_perp) * d.l_avg.powf(beta);
        let w = 1.0 / (d.epsilon_bar_err * d.epsilon_bar_err);
        sum_wx_eps += w * d.epsilon_bar * x;
        sum_wx2 += w * x * x;
    }

    if sum_wx2.abs() < 1e-30 {
        return (0.0, f64::INFINITY);
    }

    let k = sum_wx_eps / sum_wx2;

    // Chi2
    let chi2: f64 = data
        .iter()
        .map(|d| {
            let x = (d.dnch_dy / d.a_perp) * d.l_avg.powf(beta);
            let model = k * x;
            let diff = d.epsilon_bar - model;
            diff * diff / (d.epsilon_bar_err * d.epsilon_bar_err)
        })
        .sum();

    (k, chi2)
}

/// Error on K from the weighted fit: sigma_K = 1/sqrt(sum(w_i * X_i^2)).
fn k_error(data: &[DensityScalingPoint], beta: f64) -> f64 {
    let sum_wx2: f64 = data
        .iter()
        .map(|d| {
            let x = (d.dnch_dy / d.a_perp) * d.l_avg.powf(beta);
            let w = 1.0 / (d.epsilon_bar_err * d.epsilon_bar_err);
            w * x * x
        })
        .sum();

    if sum_wx2 > 0.0 {
        1.0 / sum_wx2.sqrt()
    } else {
        f64::INFINITY
    }
}

/// Scan chi2 profile to find where chi2 crosses the target value.
fn scan_for_crossing(profile: &[(f64, f64)], beta_min: f64, forward: bool, target: f64) -> f64 {
    if forward {
        for w in profile.windows(2) {
            if w[0].0 >= beta_min && w[0].1 < target && w[1].1 >= target {
                let frac = (target - w[0].1) / (w[1].1 - w[0].1);
                return w[0].0 + frac * (w[1].0 - w[0].0);
            }
        }
        // If no crossing found, return last point
        profile.last().map(|p| p.0).unwrap_or(beta_min)
    } else {
        for w in profile.windows(2).rev() {
            if w[1].0 <= beta_min && w[1].1 < target && w[0].1 >= target {
                let frac = (target - w[1].1) / (w[0].1 - w[1].1);
                return w[1].0 - frac * (w[1].0 - w[0].0);
            }
        }
        profile.first().map(|p| p.0).unwrap_or(beta_min)
    }
}

/// Golden-section search for 1D minimization.
fn golden_section_minimize<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, tol: f64) -> (f64, f64) {
    let phi = (5.0_f64.sqrt() - 1.0) / 2.0; // ~0.618
    let (mut a, mut b) = (a, b);

    let mut c = b - phi * (b - a);
    let mut d = a + phi * (b - a);
    let mut fc = f(c);
    let mut fd = f(d);

    for _ in 0..200 {
        if (b - a).abs() < tol {
            break;
        }
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - phi * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + phi * (b - a);
            fd = f(d);
        }
    }

    let x = 0.5 * (a + b);
    (x, f(x))
}

/// Result of a multi-system density scaling fit with per-system K constants.
#[derive(Debug, Clone)]
pub struct MultiSystemScalingResult {
    /// Best-fit path-length exponent (shared across systems).
    pub beta: f64,
    /// Upper error on beta (delta-chi2 = 1).
    pub beta_err_up: f64,
    /// Lower error on beta (delta-chi2 = 1).
    pub beta_err_down: f64,
    /// Per-system K constants: (system_label, K, K_err).
    pub k_per_system: Vec<(String, f64, f64)>,
    /// Total chi2 at best fit.
    pub chi2_min: f64,
    /// Number of degrees of freedom.
    pub ndf: usize,
    /// Chi2/ndf (reduced chi2).
    pub chi2_per_ndf: f64,
}

/// Multi-system density scaling fit with per-system K constants and shared beta.
///
/// For each beta, groups data by `system` label, fits a separate K for each
/// system via weighted least-squares, and sums the chi2 values.
pub fn fit_density_scaling_multi_k(
    data: &[DensityScalingPoint],
    beta_lo: f64,
    beta_hi: f64,
    beta_step: f64,
) -> MultiSystemScalingResult {
    let mut systems: Vec<String> = data.iter().map(|d| d.system.clone()).collect();
    systems.sort();
    systems.dedup();
    let n_systems = systems.len();

    let n_steps = ((beta_hi - beta_lo) / beta_step).ceil() as usize + 1;
    let mut best_beta = beta_lo;
    let mut best_chi2 = f64::INFINITY;
    let mut chi2_profile = Vec::with_capacity(n_steps);

    for step in 0..n_steps {
        let beta = beta_lo + step as f64 * beta_step;
        let c2 = chi2_at_beta_multi_k(data, &systems, beta);
        chi2_profile.push((beta, c2));
        if c2 < best_chi2 {
            best_chi2 = c2;
            best_beta = beta;
        }
    }

    // Refine with golden-section search
    let refine_lo = (best_beta - 2.0 * beta_step).max(beta_lo);
    let refine_hi = (best_beta + 2.0 * beta_step).min(beta_hi);
    let (refined_beta, refined_chi2) = golden_section_minimize(
        |b| chi2_at_beta_multi_k(data, &systems, b),
        refine_lo,
        refine_hi,
        1e-6,
    );
    if refined_chi2 < best_chi2 {
        best_beta = refined_beta;
        best_chi2 = refined_chi2;
    }

    // Extract per-system K at best beta
    let k_per_system: Vec<(String, f64, f64)> = systems
        .iter()
        .map(|sys| {
            let sub: Vec<&DensityScalingPoint> =
                data.iter().filter(|d| &d.system == sys).collect();
            let (k, _) = chi2_at_beta(&sub_to_owned(&sub), best_beta);
            let ke = k_error(&sub_to_owned(&sub), best_beta);
            (sys.clone(), k, ke)
        })
        .collect();

    let ndf = if data.len() > n_systems + 1 {
        data.len() - n_systems - 1
    } else {
        1
    };

    let target = best_chi2 + 1.0;
    let beta_err_up = scan_for_crossing(&chi2_profile, best_beta, true, target) - best_beta;
    let beta_err_down = best_beta - scan_for_crossing(&chi2_profile, best_beta, false, target);

    MultiSystemScalingResult {
        beta: best_beta,
        beta_err_up: beta_err_up.max(0.0),
        beta_err_down: beta_err_down.max(0.0),
        k_per_system,
        chi2_min: best_chi2,
        ndf,
        chi2_per_ndf: best_chi2 / ndf as f64,
    }
}

/// Chi2 for multi-K model: sum of per-system chi2 with independent K fits.
fn chi2_at_beta_multi_k(data: &[DensityScalingPoint], systems: &[String], beta: f64) -> f64 {
    systems
        .iter()
        .map(|sys| {
            let sub: Vec<DensityScalingPoint> = data
                .iter()
                .filter(|d| &d.system == sys)
                .cloned()
                .collect();
            if sub.is_empty() {
                return 0.0;
            }
            chi2_at_beta(&sub, beta).1
        })
        .sum()
}

/// Helper: convert slice of refs to owned vec for chi2_at_beta.
fn sub_to_owned(sub: &[&DensityScalingPoint]) -> Vec<DensityScalingPoint> {
    sub.iter().map(|d| (*d).clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_synthetic_data(beta_true: f64, k_true: f64) -> Vec<DensityScalingPoint> {
        // Simulate several centrality bins with known parameters
        let centralities = [
            (1800.0, 120.0, 6.5, "0-5%"),
            (1400.0, 110.0, 6.0, "5-10%"),
            (1000.0, 95.0, 5.4, "10-20%"),
            (650.0, 78.0, 4.7, "20-30%"),
            (420.0, 62.0, 4.0, "30-40%"),
            (260.0, 48.0, 3.3, "40-50%"),
        ];

        centralities
            .iter()
            .map(|&(dnch_dy, a_perp, l_avg, cent): &(f64, f64, f64, &str)| {
                let x = (dnch_dy / a_perp) * l_avg.powf(beta_true);
                let eps = k_true * x;
                DensityScalingPoint {
                    epsilon_bar: eps,
                    epsilon_bar_err: eps * 0.1, // 10% relative error
                    dnch_dy,
                    a_perp,
                    l_avg,
                    system: "Pb-Pb 5.02 TeV".to_string(),
                    centrality: cent.to_string(),
                }
            })
            .collect()
    }

    #[test]
    fn test_density_scaling_recovery() {
        let beta_true = 1.02;
        let k_true = 0.33;
        let data = make_synthetic_data(beta_true, k_true);

        let result = fit_density_scaling(&data, 0.0, 3.0, 0.01);

        assert!(
            (result.beta - beta_true).abs() < 0.05,
            "beta = {} +{}/- {} (expected {})",
            result.beta,
            result.beta_err_up,
            result.beta_err_down,
            beta_true
        );
        assert!(
            (result.k_constant - k_true).abs() < 0.05,
            "K = {} (expected {})",
            result.k_constant,
            k_true
        );
        assert!(
            result.chi2_per_ndf < 0.01,
            "chi2/ndf = {} (expected ~0 for perfect data)",
            result.chi2_per_ndf
        );
    }

    #[test]
    fn test_golden_section_parabola() {
        let (x, _f) = golden_section_minimize(|x| (x - 2.5) * (x - 2.5), 0.0, 5.0, 1e-8);
        assert!((x - 2.5).abs() < 1e-5, "minimum at {} (expected 2.5)", x);
    }
}

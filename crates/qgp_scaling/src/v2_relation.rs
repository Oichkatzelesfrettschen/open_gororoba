//! v2/eccentricity vs d(ln R_AA)/d(ln pT) linear relation.
//!
//! Arleo and Falmagne derive a first-order relation between azimuthal
//! anisotropy divided by directional-path eccentricity and the logarithmic
//! slope of R_AA, under their angular-path and harmonic approximations:
//!
//!   v2 / e = (beta / 2) * d(ln R_AA) / d(ln pT)
//!
//! The regression requires matched v2 and R_AA populations and the source
//! path estimator. Substitution of spatial moments or unidentified cumulant
//! tables defines a separate comparison rather than source conformance.
//!
//! Reference: Arleo and Falmagne arXiv:2212.01324, Eqs. (12), (16) and (17).

/// Data point for the v2/eccentricity vs slope relation.
#[derive(Debug, Clone)]
pub struct V2SlopePoint {
    /// pT bin center (GeV).
    pub pt: f64,
    /// v2 / eccentricity.
    pub v2_over_ecc: f64,
    /// Error on v2/eccentricity.
    pub v2_over_ecc_err: f64,
    /// d(ln R_AA)/d(ln pT) at this pT.
    pub dln_raa_dln_pt: f64,
    /// Error on the logarithmic derivative.
    pub slope_err: f64,
    /// Centrality label (for bookkeeping).
    pub centrality: String,
}

/// Result of the v2/eccentricity linear fit.
#[derive(Debug, Clone)]
pub struct V2RelationResult {
    /// Extracted beta from v2/e = (beta/2) * slope.
    pub beta: f64,
    /// Error on beta.
    pub beta_err: f64,
    /// Intercept of the linear fit (should be ~0).
    pub intercept: f64,
    /// Error on intercept.
    pub intercept_err: f64,
    /// Pearson correlation coefficient.
    pub r_squared: f64,
    /// Chi2 of the fit.
    pub chi2: f64,
    /// Number of degrees of freedom.
    pub ndf: usize,
}

/// Fit v2/e = (beta/2) * d(ln R_AA)/d(ln pT) using weighted linear regression.
///
/// Fits y = a*x + b where:
///   y = v2/eccentricity
///   x = d(ln R_AA)/d(ln pT)
///   a = beta/2
///
/// # Arguments
/// * `data` - v2/eccentricity and slope measurements at various pT and centralities
/// * `force_zero_intercept` - If true, fit y = a*x (no intercept)
pub fn fit_v2_relation(data: &[V2SlopePoint], force_zero_intercept: bool) -> V2RelationResult {
    if force_zero_intercept {
        fit_through_origin(data)
    } else {
        fit_with_intercept(data)
    }
}

/// Weighted linear regression through the origin: y = a*x.
/// a = sum(w_i * x_i * y_i) / sum(w_i * x_i^2)
fn fit_through_origin(data: &[V2SlopePoint]) -> V2RelationResult {
    let mut sum_wxy = 0.0;
    let mut sum_wx2 = 0.0;
    let mut sum_wy = 0.0;
    let mut sum_w = 0.0;

    for d in data {
        let w = 1.0 / (d.v2_over_ecc_err * d.v2_over_ecc_err);
        sum_wxy += w * d.dln_raa_dln_pt * d.v2_over_ecc;
        sum_wx2 += w * d.dln_raa_dln_pt * d.dln_raa_dln_pt;
        sum_wy += w * d.v2_over_ecc;
        sum_w += w;
    }

    let slope = if sum_wx2 > 0.0 {
        sum_wxy / sum_wx2
    } else {
        0.0
    };
    let slope_err = if sum_wx2 > 0.0 {
        1.0 / sum_wx2.sqrt()
    } else {
        f64::INFINITY
    };

    let beta = 2.0 * slope;
    let beta_err = 2.0 * slope_err;

    // Chi2
    let chi2: f64 = data
        .iter()
        .map(|d| {
            let model = slope * d.dln_raa_dln_pt;
            let diff = d.v2_over_ecc - model;
            diff * diff / (d.v2_over_ecc_err * d.v2_over_ecc_err)
        })
        .sum();

    // R^2
    let y_mean = if sum_w > 0.0 { sum_wy / sum_w } else { 0.0 };
    let ss_tot: f64 = data
        .iter()
        .map(|d| {
            let w = 1.0 / (d.v2_over_ecc_err * d.v2_over_ecc_err);
            w * (d.v2_over_ecc - y_mean).powi(2)
        })
        .sum();
    let r_squared = if ss_tot > 0.0 {
        1.0 - chi2 / ss_tot
    } else {
        0.0
    };

    let ndf = if data.len() > 1 { data.len() - 1 } else { 1 };

    V2RelationResult {
        beta,
        beta_err,
        intercept: 0.0,
        intercept_err: 0.0,
        r_squared,
        chi2,
        ndf,
    }
}

/// Weighted linear regression with intercept: y = a*x + b.
fn fit_with_intercept(data: &[V2SlopePoint]) -> V2RelationResult {
    let mut sum_w = 0.0;
    let mut sum_wx = 0.0;
    let mut sum_wy = 0.0;
    let mut sum_wxx = 0.0;
    let mut sum_wxy = 0.0;

    for d in data {
        let w = 1.0 / (d.v2_over_ecc_err * d.v2_over_ecc_err);
        sum_w += w;
        sum_wx += w * d.dln_raa_dln_pt;
        sum_wy += w * d.v2_over_ecc;
        sum_wxx += w * d.dln_raa_dln_pt * d.dln_raa_dln_pt;
        sum_wxy += w * d.dln_raa_dln_pt * d.v2_over_ecc;
    }

    let det = sum_w * sum_wxx - sum_wx * sum_wx;
    let (slope, intercept) = if det.abs() > 1e-30 {
        let a = (sum_w * sum_wxy - sum_wx * sum_wy) / det;
        let b = (sum_wxx * sum_wy - sum_wx * sum_wxy) / det;
        (a, b)
    } else {
        (0.0, 0.0)
    };

    let slope_err = if det.abs() > 1e-30 {
        (sum_w / det).sqrt()
    } else {
        f64::INFINITY
    };
    let intercept_err = if det.abs() > 1e-30 {
        (sum_wxx / det).sqrt()
    } else {
        f64::INFINITY
    };

    let beta = 2.0 * slope;
    let beta_err = 2.0 * slope_err;

    // Chi2
    let chi2: f64 = data
        .iter()
        .map(|d| {
            let model = slope * d.dln_raa_dln_pt + intercept;
            let diff = d.v2_over_ecc - model;
            diff * diff / (d.v2_over_ecc_err * d.v2_over_ecc_err)
        })
        .sum();

    // R^2
    let y_mean = if sum_w > 0.0 { sum_wy / sum_w } else { 0.0 };
    let ss_tot: f64 = data
        .iter()
        .map(|d| {
            let w = 1.0 / (d.v2_over_ecc_err * d.v2_over_ecc_err);
            w * (d.v2_over_ecc - y_mean).powi(2)
        })
        .sum();
    let r_squared = if ss_tot > 0.0 {
        1.0 - chi2 / ss_tot
    } else {
        0.0
    };

    let ndf = if data.len() > 2 { data.len() - 2 } else { 1 };

    V2RelationResult {
        beta,
        beta_err,
        intercept,
        intercept_err,
        r_squared,
        chi2,
        ndf,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_linear_data(beta_true: f64, n_pts: usize) -> Vec<V2SlopePoint> {
        // v2/e = (beta/2) * slope
        let slope_coeff = beta_true / 2.0;
        (0..n_pts)
            .map(|i| {
                let x = 0.1 + i as f64 * 0.05; // d(ln R_AA)/d(ln pT) values
                let y = slope_coeff * x;
                V2SlopePoint {
                    pt: 5.0 + i as f64 * 2.0,
                    v2_over_ecc: y,
                    v2_over_ecc_err: 0.01,
                    dln_raa_dln_pt: x,
                    slope_err: 0.005,
                    centrality: format!("{}-{}%", i * 10, (i + 1) * 10),
                }
            })
            .collect()
    }

    #[test]
    fn test_v2_relation_recovery_origin() {
        let beta_true = 1.02;
        let data = make_linear_data(beta_true, 10);
        let result = fit_v2_relation(&data, true);

        assert!(
            (result.beta - beta_true).abs() < 0.01,
            "beta = {} +/- {} (expected {})",
            result.beta,
            result.beta_err,
            beta_true
        );
        assert!(
            result.r_squared > 0.99,
            "R^2 = {} (expected ~1)",
            result.r_squared
        );
    }

    #[test]
    fn test_v2_relation_recovery_intercept() {
        let beta_true = 0.94;
        let data = make_linear_data(beta_true, 10);
        let result = fit_v2_relation(&data, false);

        assert!(
            (result.beta - beta_true).abs() < 0.02,
            "beta = {} +/- {} (expected {})",
            result.beta,
            result.beta_err,
            beta_true
        );
        assert!(
            result.intercept.abs() < 0.01,
            "intercept = {} (expected ~0)",
            result.intercept
        );
    }

    #[test]
    fn test_v2_relation_linearity() {
        // Check that chi2/ndf is small for perfect linear data
        let data = make_linear_data(1.0, 20);
        let result = fit_v2_relation(&data, false);
        assert!(
            result.chi2 / (result.ndf as f64) < 0.01,
            "chi2/ndf = {} (expected ~0 for perfect data)",
            result.chi2 / (result.ndf as f64)
        );
    }
}

//! R_AA model and scaling function for parton energy loss.
//!
//! Implements the scaling-collapse approach of Arleo & Falmagne
//! (arXiv:2411.13258): R_AA(pT) = ((pT - epsilon_bar) / pT)^{n-1}
//! where epsilon_bar is the average energy loss and n is the spectral
//! index of the pp production spectrum (dN/dpT ~ pT^{-n}).
//!
//! Also provides the Savitzky-Golay logarithmic derivative
//! d(ln R_AA)/d(ln pT) needed for the v2/eccentricity relation.

/// Compute R_AA from the discrete energy loss model.
///
/// R_AA(pT) = ((pT - epsilon_bar) / pT)^{n-1} for pT > epsilon_bar
///          = 0                                  for pT <= epsilon_bar
///
/// This is the large-pT limit of the medium-modified spectrum ratio
/// when the pp spectrum follows a power law dN/dpT ~ pT^{-n} and
/// each parton loses a fixed energy epsilon_bar.
///
/// Reference: Arleo, JHEP 0211 (2002) 044; Arleo & Falmagne Eq.(1).
#[must_use]
pub fn r_aa_model(pt: f64, epsilon_bar: f64, n: f64) -> f64 {
    if pt <= epsilon_bar || epsilon_bar < 0.0 {
        return 0.0;
    }
    ((pt - epsilon_bar) / pt).powf(n - 1.0)
}

/// Universal scaling function f(u, n) = ((u-1)/u)^{n-1} where u = pT / epsilon_bar.
///
/// R_AA(pT) = f(pT / epsilon_bar, n) after rescaling the pT axis.
/// This collapses R_AA curves from different centralities onto a single curve.
#[must_use]
pub fn scaling_function(u: f64, n: f64) -> f64 {
    if u <= 1.0 {
        return 0.0;
    }
    ((u - 1.0) / u).powf(n - 1.0)
}

/// Compute d(ln R_AA)/d(ln pT) using 5-point Savitzky-Golay finite differences.
///
/// For the analytic model R_AA = ((pT - eps)/pT)^{n-1}, the derivative is:
///   d(ln R_AA)/d(ln pT) = (n-1) * epsilon_bar / (pT - epsilon_bar)
///
/// For data, we use the Savitzky-Golay smoothed derivative on log-log grid.
/// The 5-point stencil coefficients for the first derivative are:
///   f'(x_i) ~ (-2*f_{i-2} - f_{i-1} + f_{i+1} + 2*f_{i+2}) / (10*h)
///
/// # Arguments
/// * `pt` - pT values (must be sorted ascending, length >= 5)
/// * `raa` - R_AA values (same length as pt)
///
/// # Returns
/// Vector of d(ln R_AA)/d(ln pT) at each interior point.
/// First two and last two points use forward/backward differences.
pub fn log_derivative_raa(pt: &[f64], raa: &[f64]) -> Vec<f64> {
    assert_eq!(pt.len(), raa.len(), "pt and raa must have same length");
    let n = pt.len();
    if n < 3 {
        return vec![0.0; n];
    }

    // Work in log space
    let log_pt: Vec<f64> = pt.iter().map(|&x| x.ln()).collect();
    let log_raa: Vec<f64> = raa.iter().map(|&r| {
        if r > 0.0 { r.ln() } else { f64::NEG_INFINITY }
    }).collect();

    let mut deriv = vec![0.0; n];

    for i in 0..n {
        if log_raa[i].is_infinite() {
            deriv[i] = 0.0;
            continue;
        }

        if n >= 5 && i >= 2 && i + 2 < n {
            // 5-point Savitzky-Golay stencil (assuming roughly uniform spacing in log-pT)
            let h = (log_pt[i + 1] - log_pt[i - 1]) / 2.0;
            if h.abs() < 1e-30 {
                deriv[i] = 0.0;
                continue;
            }
            // Check all points are finite
            if log_raa[i - 2].is_infinite() || log_raa[i - 1].is_infinite()
                || log_raa[i + 1].is_infinite() || log_raa[i + 2].is_infinite()
            {
                // Fall back to central difference
                let h2 = log_pt[i + 1] - log_pt[i - 1];
                if h2.abs() > 1e-30 && log_raa[i + 1].is_finite() && log_raa[i - 1].is_finite() {
                    deriv[i] = (log_raa[i + 1] - log_raa[i - 1]) / h2;
                }
                continue;
            }
            deriv[i] = (-2.0 * log_raa[i - 2] - log_raa[i - 1]
                + log_raa[i + 1] + 2.0 * log_raa[i + 2])
                / (10.0 * h);
        } else if i > 0 && i + 1 < n {
            // Central difference for boundary-adjacent points
            let h = log_pt[i + 1] - log_pt[i - 1];
            if h.abs() > 1e-30 && log_raa[i + 1].is_finite() && log_raa[i - 1].is_finite() {
                deriv[i] = (log_raa[i + 1] - log_raa[i - 1]) / h;
            }
        } else if i == 0 && n > 1 {
            // Forward difference
            let h = log_pt[1] - log_pt[0];
            if h.abs() > 1e-30 && log_raa[1].is_finite() {
                deriv[i] = (log_raa[1] - log_raa[0]) / h;
            }
        } else if i == n - 1 && n > 1 {
            // Backward difference
            let h = log_pt[n - 1] - log_pt[n - 2];
            if h.abs() > 1e-30 && log_raa[n - 2].is_finite() {
                deriv[i] = (log_raa[n - 1] - log_raa[n - 2]) / h;
            }
        }
    }

    deriv
}

/// Analytic logarithmic derivative of the R_AA model.
///
/// d(ln R_AA)/d(ln pT) = (n-1) * epsilon_bar / (pT - epsilon_bar)
///
/// This is positive (R_AA increases with pT) and decreases as pT grows.
#[must_use]
pub fn analytic_log_derivative(pt: f64, epsilon_bar: f64, n: f64) -> f64 {
    if pt <= epsilon_bar {
        return 0.0;
    }
    (n - 1.0) * epsilon_bar / (pt - epsilon_bar)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raa_model_basic() {
        // At pT >> epsilon_bar, R_AA -> 1
        let raa = r_aa_model(100.0, 5.0, 6.0);
        assert!(raa > 0.7 && raa < 1.0, "R_AA(100, 5, 6) = {}", raa);

        // At pT = 2*epsilon_bar, R_AA = (1/2)^{n-1} = (1/2)^5 = 0.03125
        let raa2 = r_aa_model(10.0, 5.0, 6.0);
        assert!((raa2 - 0.03125).abs() < 1e-10, "R_AA(10, 5, 6) = {}", raa2);
    }

    #[test]
    fn test_raa_model_zero_below_epsilon() {
        assert_eq!(r_aa_model(3.0, 5.0, 6.0), 0.0);
        assert_eq!(r_aa_model(5.0, 5.0, 6.0), 0.0);
    }

    #[test]
    fn test_scaling_function_matches_raa() {
        let eps = 3.5;
        let n = 6.2;
        let pt = 15.0;
        let u = pt / eps;
        let raa = r_aa_model(pt, eps, n);
        let sf = scaling_function(u, n);
        assert!((raa - sf).abs() < 1e-12,
            "R_AA = {}, f(u) = {} (should match)", raa, sf);
    }

    #[test]
    fn test_log_derivative_smooth() {
        // For R_AA = pT^alpha (power law), d(ln R_AA)/d(ln pT) = alpha
        let alpha = 2.5;
        let pt: Vec<f64> = (1..=50).map(|i| i as f64).collect();
        let raa: Vec<f64> = pt.iter().map(|&p| p.powf(alpha)).collect();

        let deriv = log_derivative_raa(&pt, &raa);

        // Check interior points (skip boundary; widen tolerance near edges)
        for i in 5..45 {
            assert!((deriv[i] - alpha).abs() < 0.2,
                "d(ln R_AA)/d(ln pT)[{}] = {} (expected {})", i, deriv[i], alpha);
        }
    }

    #[test]
    fn test_analytic_log_derivative() {
        let eps = 4.0;
        let n = 6.0;
        let pt = 20.0;
        let expected = (n - 1.0) * eps / (pt - eps); // 5*4/16 = 1.25
        let actual = analytic_log_derivative(pt, eps, n);
        assert!((actual - expected).abs() < 1e-10,
            "analytic deriv = {} (expected {})", actual, expected);
    }

    #[test]
    fn test_log_derivative_matches_analytic() {
        // Generate R_AA from the model and check that numerical derivative
        // matches the analytic formula
        let eps = 4.0;
        let n = 6.0;
        let pt: Vec<f64> = (0..=100).map(|i| 5.0 + i as f64 * 0.5).collect();
        let raa: Vec<f64> = pt.iter().map(|&p| r_aa_model(p, eps, n)).collect();

        let num_deriv = log_derivative_raa(&pt, &raa);

        // Check a range of interior points
        for i in 10..90 {
            let expected = analytic_log_derivative(pt[i], eps, n);
            if expected > 0.01 {
                let rel_err = (num_deriv[i] - expected).abs() / expected;
                assert!(rel_err < 0.05,
                    "pT={}: numerical={}, analytic={}, rel_err={}",
                    pt[i], num_deriv[i], expected, rel_err);
            }
        }
    }
}

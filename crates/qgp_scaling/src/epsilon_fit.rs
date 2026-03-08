//! Chi-square epsilon_bar extraction per centrality bin.
//!
//! For each centrality bin, fits the R_AA model to data by minimizing
//! chi2 over the single parameter epsilon_bar (average parton energy loss).
//! Uses Brent's method for 1D minimization and delta-chi2=1 scan for
//! asymmetric error bars.
//!
//! Reference: Arleo & Falmagne arXiv:2411.13258, Sec. III.

use crate::{quenching::r_aa_model, straggling::StragglingGrid};

/// Result of epsilon_bar extraction for a single centrality bin.
#[derive(Debug, Clone)]
pub struct EpsilonFitResult {
    /// Best-fit average energy loss (GeV).
    pub epsilon_bar: f64,
    /// Upper error (delta-chi2 = 1).
    pub err_up: f64,
    /// Lower error (delta-chi2 = 1).
    pub err_down: f64,
    /// Minimum chi2 value.
    pub chi2_min: f64,
    /// Number of degrees of freedom (n_data - 1).
    pub ndf: usize,
    /// Spectral index used.
    pub n_spectral: f64,
}

/// Data point for R_AA measurement.
#[derive(Debug, Clone)]
pub struct RaaDataPoint {
    /// Transverse momentum (GeV).
    pub pt: f64,
    /// R_AA value.
    pub raa: f64,
    /// Statistical error.
    pub stat_err: f64,
    /// Systematic error.
    pub syst_err: f64,
}

impl RaaDataPoint {
    /// Total error (stat and syst added in quadrature).
    #[must_use]
    pub fn total_err(&self) -> f64 {
        (self.stat_err * self.stat_err + self.syst_err * self.syst_err).sqrt()
    }
}

/// Compute chi2 for a given epsilon_bar against R_AA data.
fn chi2(data: &[RaaDataPoint], epsilon_bar: f64, n: f64) -> f64 {
    let mut sum = 0.0;
    for pt in data {
        let model = r_aa_model(pt.pt, epsilon_bar, n);
        let err = pt.total_err();
        if err > 0.0 {
            let diff = pt.raa - model;
            sum += diff * diff / (err * err);
        }
    }
    sum
}

/// Extract epsilon_bar from R_AA data using Brent's method.
///
/// Searches for the epsilon_bar in [eps_lo, eps_hi] that minimizes chi2.
/// Uses the golden-section search with parabolic interpolation (Brent's method).
///
/// # Arguments
/// * `data` - R_AA data points (already filtered to pT > pT_min)
/// * `n` - Spectral index of the pp spectrum
/// * `eps_lo` - Lower bound for epsilon_bar search (GeV)
/// * `eps_hi` - Upper bound for epsilon_bar search (GeV)
/// * `tol` - Convergence tolerance on epsilon_bar
pub fn extract_epsilon(
    data: &[RaaDataPoint],
    n: f64,
    eps_lo: f64,
    eps_hi: f64,
    tol: f64,
) -> EpsilonFitResult {
    // Brent's method for 1D minimization
    let (best_eps, best_chi2) = brent_minimize(|eps| chi2(data, eps, n), eps_lo, eps_hi, tol);

    let ndf = if data.len() > 1 { data.len() - 1 } else { 1 };

    // Delta-chi2 = 1 scan for error bars
    let target = best_chi2 + 1.0;

    // Upper error: find eps > best_eps where chi2 = target
    let err_up = find_delta_chi2_crossing(|eps| chi2(data, eps, n), best_eps, eps_hi, target, tol)
        - best_eps;

    // Lower error: find eps < best_eps where chi2 = target
    let err_down = best_eps
        - find_delta_chi2_crossing(|eps| chi2(data, eps, n), eps_lo, best_eps, target, tol);

    EpsilonFitResult {
        epsilon_bar: best_eps,
        err_up,
        err_down,
        chi2_min: best_chi2,
        ndf,
        n_spectral: n,
    }
}

/// Compute chi2 using the straggling-smeared R_AA from a precomputed grid.
fn chi2_straggling(data: &[RaaDataPoint], epsilon_bar: f64, grid: &StragglingGrid) -> f64 {
    let mut sum = 0.0;
    for pt in data {
        let model = grid.lookup(pt.pt, epsilon_bar);
        let err = pt.total_err();
        if err > 0.0 {
            let diff = pt.raa - model;
            sum += diff * diff / (err * err);
        }
    }
    sum
}

/// Extract epsilon_bar from R_AA data using the straggling-smeared model.
///
/// Identical to [`extract_epsilon`] except that the chi2 function calls
/// `grid.lookup(pt, epsilon_bar)` (O(log N) bilinear interpolation) rather than
/// `r_aa_model(pt, epsilon_bar, n)`.  The spectral index is read from
/// `grid.n_spectral` so there is no risk of a silent mismatch.
///
/// # Arguments
/// * `data`    -- R_AA data points (already filtered to the desired pT range)
/// * `eps_lo`  -- Lower bound for epsilon_bar search (GeV)
/// * `eps_hi`  -- Upper bound for epsilon_bar search (GeV)
/// * `tol`     -- Convergence tolerance on epsilon_bar
/// * `grid`    -- Precomputed straggling lookup table
pub fn extract_epsilon_straggling(
    data: &[RaaDataPoint],
    eps_lo: f64,
    eps_hi: f64,
    tol: f64,
    grid: &StragglingGrid,
) -> EpsilonFitResult {
    let (best_eps, best_chi2) =
        brent_minimize(|eps| chi2_straggling(data, eps, grid), eps_lo, eps_hi, tol);

    let ndf = if data.len() > 1 { data.len() - 1 } else { 1 };

    let target = best_chi2 + 1.0;

    let err_up = find_delta_chi2_crossing(
        |eps| chi2_straggling(data, eps, grid),
        best_eps,
        eps_hi,
        target,
        tol,
    ) - best_eps;

    let err_down = best_eps
        - find_delta_chi2_crossing(
            |eps| chi2_straggling(data, eps, grid),
            eps_lo,
            best_eps,
            target,
            tol,
        );

    EpsilonFitResult {
        epsilon_bar: best_eps,
        err_up,
        err_down,
        chi2_min: best_chi2,
        ndf,
        n_spectral: grid.n_spectral,
    }
}

/// Brent's method for 1D minimization on [a, b].
/// Returns (x_min, f_min).
fn brent_minimize<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, tol: f64) -> (f64, f64) {
    let golden = 0.381966011250105; // (3 - sqrt(5)) / 2
    let max_iter = 200;

    let (mut a, mut b) = if a < b { (a, b) } else { (b, a) };
    let mut x = a + golden * (b - a);
    let mut w = x;
    let mut v = x;
    let mut fx = f(x);
    let mut fw = fx;
    let mut fv = fx;
    let mut d = 0.0_f64;
    let mut e = 0.0_f64;

    for _ in 0..max_iter {
        let midpoint = 0.5 * (a + b);
        let tol1 = tol * x.abs() + 1e-10;
        let tol2 = 2.0 * tol1;

        if (x - midpoint).abs() <= tol2 - 0.5 * (b - a) {
            return (x, fx);
        }

        // Try parabolic interpolation
        let mut use_golden = true;
        if e.abs() > tol1 {
            let r = (x - w) * (fx - fv);
            let q = (x - v) * (fx - fw);
            let mut p = (x - v) * q - (x - w) * r;
            let mut q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            } else {
                q = -q;
            }
            let prev_e = e;

            if p.abs() < (0.5 * q * prev_e).abs() && p > q * (a - x) && p < q * (b - x) {
                d = p / q;
                let u = x + d;
                if (u - a) < tol2 || (b - u) < tol2 {
                    d = if x < midpoint { tol1 } else { -tol1 };
                }
                use_golden = false;
            }
        }

        if use_golden {
            e = if x < midpoint { b - x } else { a - x };
            d = golden * e;
        } else {
            e = d;
        }

        let u = if d.abs() >= tol1 {
            x + d
        } else {
            x + tol1 * d.signum()
        };
        let fu = f(u);

        if fu <= fx {
            if u < x {
                b = x;
            } else {
                a = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || (w - x).abs() < 1e-30 {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || (v - x).abs() < 1e-30 || (v - w).abs() < 1e-30 {
                v = u;
                fv = fu;
            }
        }
    }

    (x, fx)
}

/// Find the x in [a, b] where f(x) = target using bisection.
/// Assumes f(a) and f(b) bracket the target (one above, one below).
fn find_delta_chi2_crossing<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, target: f64, tol: f64) -> f64 {
    let fa = f(a) - target;
    let fb = f(b) - target;

    // If both on same side, return the endpoint closer to target
    if fa * fb > 0.0 {
        return if fa.abs() < fb.abs() { a } else { b };
    }

    let mut lo = a;
    let mut hi = b;
    let mut f_lo = fa;

    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if (hi - lo) < tol {
            return mid;
        }
        let f_mid = f(mid) - target;
        if f_mid * f_lo < 0.0 {
            hi = mid;
        } else {
            lo = mid;
            f_lo = f_mid;
        }
    }

    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate synthetic R_AA data from the model with known epsilon_bar.
    fn synthetic_raa_data(
        epsilon_bar: f64,
        n: f64,
        pt_min: f64,
        pt_max: f64,
        n_pts: usize,
    ) -> Vec<RaaDataPoint> {
        let dpt = (pt_max - pt_min) / (n_pts - 1) as f64;
        (0..n_pts)
            .map(|i| {
                let pt = pt_min + i as f64 * dpt;
                let raa = r_aa_model(pt, epsilon_bar, n);
                RaaDataPoint {
                    pt,
                    raa,
                    stat_err: 0.02,
                    syst_err: 0.03,
                }
            })
            .collect()
    }

    #[test]
    fn test_epsilon_extraction_synthetic() {
        let true_eps = 5.0;
        let n = 6.0;
        let data = synthetic_raa_data(true_eps, n, 6.0, 50.0, 30);

        let result = extract_epsilon(&data, n, 0.1, 20.0, 1e-6);

        assert!(
            (result.epsilon_bar - true_eps).abs() < 0.1,
            "extracted epsilon = {} (expected {})",
            result.epsilon_bar,
            true_eps
        );
        assert!(
            result.chi2_min < 0.01,
            "chi2_min = {} (expected ~0 for perfect data)",
            result.chi2_min
        );
    }

    #[test]
    fn test_epsilon_extraction_noisy() {
        // Add some noise-like scatter to the data
        let true_eps = 3.5;
        let n = 6.2;
        let mut data = synthetic_raa_data(true_eps, n, 5.0, 40.0, 25);

        // Perturb every other point slightly
        for (i, d) in data.iter_mut().enumerate() {
            if i % 2 == 0 {
                d.raa += 0.01;
            } else {
                d.raa -= 0.01;
            }
        }

        let result = extract_epsilon(&data, n, 0.1, 15.0, 1e-6);

        // Should still recover epsilon within error bars
        assert!(
            (result.epsilon_bar - true_eps).abs() < 0.5,
            "extracted epsilon = {} (expected {} +/- 0.5)",
            result.epsilon_bar,
            true_eps
        );
        assert!(result.err_up > 0.0, "err_up should be positive");
        assert!(result.err_down > 0.0, "err_down should be positive");
    }

    #[test]
    fn test_brent_minimize_parabola() {
        // Minimize (x - 3)^2
        let (x_min, f_min) = brent_minimize(|x| (x - 3.0) * (x - 3.0), 0.0, 10.0, 1e-8);
        assert!((x_min - 3.0).abs() < 1e-6, "x_min = {} (expected 3)", x_min);
        assert!(f_min < 1e-10, "f_min = {} (expected 0)", f_min);
    }

    #[test]
    fn test_delta_chi2_crossing() {
        // f(x) = (x-5)^2 + 10, target = 11 => crossing at x=4 and x=6
        let crossing =
            find_delta_chi2_crossing(|x| (x - 5.0) * (x - 5.0) + 10.0, 5.0, 10.0, 11.0, 1e-8);
        assert!(
            (crossing - 6.0).abs() < 1e-6,
            "crossing = {} (expected 6)",
            crossing
        );
    }

    /// Generate synthetic R_AA data from the straggling model with known epsilon_bar.
    fn synthetic_straggling_data(
        epsilon_bar: f64,
        n: f64,
        kappa: f64,
        pt_min: f64,
        pt_max: f64,
        n_pts: usize,
    ) -> Vec<RaaDataPoint> {
        use crate::straggling::{r_aa_straggling, straggling_sigma};
        let dpt = (pt_max - pt_min) / (n_pts - 1) as f64;
        (0..n_pts)
            .map(|i| {
                let pt = pt_min + i as f64 * dpt;
                let sigma = straggling_sigma(epsilon_bar, kappa);
                let raa = r_aa_straggling(pt, epsilon_bar, n, sigma);
                RaaDataPoint {
                    pt,
                    raa,
                    stat_err: 0.02,
                    syst_err: 0.03,
                }
            })
            .collect()
    }

    #[test]
    fn test_epsilon_straggling_extraction_synthetic() {
        use crate::straggling::{DEFAULT_KAPPA, StragglingGrid};

        let true_eps = 5.0;
        let n = 6.0;
        let kappa = DEFAULT_KAPPA;
        // Generate data from the straggling model at pT well above epsilon_bar
        let data = synthetic_straggling_data(true_eps, n, kappa, 6.0, 50.0, 30);

        let grid = StragglingGrid::new((1.0, 100.0), 100, (0.1, 20.0), 80, n, kappa);
        let result = extract_epsilon_straggling(&data, 0.1, 20.0, 1e-6, &grid);

        assert!(
            (result.epsilon_bar - true_eps).abs() < 0.2,
            "extracted epsilon = {} (expected {})",
            result.epsilon_bar,
            true_eps
        );
        assert!(
            result.chi2_min < 0.5,
            "chi2_min = {} (expected ~0 for self-consistent data)",
            result.chi2_min
        );
        // n_spectral should come from the grid
        assert!(
            (result.n_spectral - n).abs() < 1e-12,
            "n_spectral mismatch: got {}, expected {}",
            result.n_spectral,
            n
        );
    }

    #[test]
    fn test_epsilon_straggling_extraction_noisy() {
        use crate::straggling::{DEFAULT_KAPPA, StragglingGrid};

        let true_eps = 3.5;
        let n = 6.2;
        let kappa = DEFAULT_KAPPA;
        let mut data = synthetic_straggling_data(true_eps, n, kappa, 4.0, 40.0, 25);

        // Perturb every other point slightly
        for (i, d) in data.iter_mut().enumerate() {
            if i % 2 == 0 {
                d.raa += 0.01;
            } else {
                d.raa -= 0.01;
            }
        }

        let grid = StragglingGrid::new((1.0, 80.0), 100, (0.1, 15.0), 80, n, kappa);
        let result = extract_epsilon_straggling(&data, 0.1, 15.0, 1e-6, &grid);

        assert!(
            (result.epsilon_bar - true_eps).abs() < 0.5,
            "extracted epsilon = {} (expected {} +/- 0.5)",
            result.epsilon_bar,
            true_eps
        );
        assert!(result.err_up > 0.0, "err_up should be positive");
        assert!(result.err_down > 0.0, "err_down should be positive");
    }
}

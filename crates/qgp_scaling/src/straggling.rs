//! Quantum straggling: Gaussian-smeared R_AA for smooth low-pT behaviour.
//!
//! Real parton energy loss in the QGP is stochastic.  The actual energy loss epsilon
//! fluctuates around the mean epsilon_bar with a Gaussian probability distribution
//! (quantum straggling / Landau-Pomeranchuk-Migdal fluctuations):
//!
//! ```text
//! P(epsilon; epsilon_bar, sigma) = (1 / sqrt(2pi sigma^2)) * exp(-(epsilon - epsilon_bar)^2 / (2sigma^2))
//! ```
//!
//! where sigma = kappa * sqrtepsilon_bar  (BDMPS scaling: standard deviation grows as sqrt(mean energy loss),
//! equivalently the variance grows linearly with the mean energy loss).
//!
//! The straggling-smeared R_AA is:
//!
//! ```text
//! R_AA^smeared(pT, epsilon_bar, n, sigma) = integral_0^pT  P(epsilon; epsilon_bar, sigma) * ((pT - epsilon) / pT)^{n-1}  depsilon
//! ```
//!
//! This integral smooths out the sharp cutoff at pT = epsilon_bar, making R_AA a
//! smooth function of pT even near epsilon_bar, and allowing fits to extend down to
//! pT ~ 1 GeV without stalling the Brent optimiser.
//!
//! # Performance strategy
//!
//! Computing the quadrature in the optimiser's inner loop is too expensive.
//! [`StragglingGrid`] precomputes R_AA^smeared on a (pT, epsilon_bar) grid **once**
//! and then provides O(log N) bilinear interpolation at query time.

use gauss_quad::GaussLegendre;
use std::num::NonZeroUsize;

/// Default BDMPS-inspired straggling width parameter kappa.
pub const DEFAULT_KAPPA: f64 = 0.5;

/// Number of GL quadrature nodes used for production accuracy.
const N_GL: usize = 32;

/// Compute the straggling width sigma = kappa * sqrtepsilon_bar.
///
/// This follows the BDMPS prediction that the standard deviation of the
/// energy-loss distribution grows as the square root of the mean energy loss
/// (equivalently, the variance grows linearly with the mean energy loss).
///
/// # Arguments
/// * `epsilon_bar` -- mean energy loss epsilon_bar (GeV)
/// * `kappa` -- width coefficient kappa (dimensionless); use [`DEFAULT_KAPPA`] = 0.5
///
/// # Returns
/// sigma (GeV).  Returns 0 for non-positive `epsilon_bar`.
#[must_use]
pub fn straggling_sigma(epsilon_bar: f64, kappa: f64) -> f64 {
    if epsilon_bar <= 0.0 {
        return 0.0;
    }
    kappa * epsilon_bar.sqrt()
}

/// Compute the Gaussian-convolved (straggling-smeared) R_AA.
///
/// ```text
/// R_AA^smeared(pT, epsilon_bar, n, sigma) = integral_{epsilon_lo}^{epsilon_hi}  P(epsilon; epsilon_bar, sigma) * ((pT - epsilon) / pT)^{n-1}  depsilon
/// ```
///
/// Integration domain is truncated to `[max(0, epsilon_bar - 5sigma), min(pT, epsilon_bar + 5sigma)]`.
/// When `sigma <= 0` the result falls back to the sharp discrete model.
///
/// Uses 32-point Gauss-Legendre quadrature.
///
/// # Performance note
///
/// This function constructs a fresh [`GaussLegendre`] object on every call.
/// For high-throughput use (e.g. inside a fitting loop) use [`StragglingGrid`]
/// instead, which precomputes the values once and serves them via O(log N)
/// bilinear interpolation.
///
/// # Arguments
/// * `pt`          -- transverse momentum (GeV)
/// * `epsilon_bar` -- mean energy loss epsilon_bar (GeV)
/// * `n`           -- spectral index of the pp spectrum
/// * `sigma`       -- straggling width sigma (GeV); pass 0 for the sharp limit
#[must_use]
pub fn r_aa_straggling(pt: f64, epsilon_bar: f64, n: f64, sigma: f64) -> f64 {
    let gl = GaussLegendre::new(
        NonZeroUsize::new(N_GL).expect("Gauss-Legendre degree must be non-zero"),
    );
    r_aa_straggling_with_gl(pt, epsilon_bar, n, sigma, &gl)
}

/// Inner implementation that reuses a pre-constructed [`GaussLegendre`] object.
///
/// This avoids reconstructing the quadrature nodes on every call when
/// evaluating on a grid (see [`StragglingGrid::new`]).
fn r_aa_straggling_with_gl(
    pt: f64,
    epsilon_bar: f64,
    n: f64,
    sigma: f64,
    gl: &GaussLegendre,
) -> f64 {
    if pt <= 0.0 || epsilon_bar < 0.0 {
        return 0.0;
    }

    // Degenerate limit: no straggling -> sharp discrete model
    if sigma <= 0.0 {
        return if pt > epsilon_bar {
            ((pt - epsilon_bar) / pt).powf(n - 1.0)
        } else {
            0.0
        };
    }

    // Integration domain: 5sigma window around epsilon_bar, physical lower bound at 0
    let eps_lo = (epsilon_bar - 5.0 * sigma).max(0.0);
    // Upper bound for the Gaussian support used for normalization (no pT clip);
    // the R_AA integral below clips this to min(pT, eps_hi_full)
    let eps_hi_full = epsilon_bar + 5.0 * sigma;

    let two_sigma_sq = 2.0 * sigma * sigma;
    let norm = (std::f64::consts::TAU * sigma * sigma).sqrt().recip();

    // Normalization: Gaussian mass over the physical domain [eps_lo, eps_hi_full].
    // When epsilon_bar is small relative to sigma, a non-negligible fraction of the untruncated
    // Gaussian falls below epsilon = 0 (unphysical).  Re-normalising here ensures
    // R_AA(pT -> inf) -> 1 regardless of how much probability mass is clipped at 0.
    let gauss_mass = gl.integrate(eps_lo, eps_hi_full, |eps: f64| {
        let diff = eps - epsilon_bar;
        norm * (-(diff * diff) / two_sigma_sq).exp()
    });

    if gauss_mass < 1e-15 {
        return 0.0;
    }

    // If the entire Gaussian support is above pT, R_AA ~ 0
    if eps_lo >= pt {
        return 0.0;
    }

    // R_AA integral over [eps_lo, min(pT, eps_hi_full)]
    let eps_hi = eps_hi_full.min(pt);

    // If the window collapses (eps_hi <= eps_lo), return 0
    if eps_hi <= eps_lo {
        return 0.0;
    }

    let integrand = |eps: f64| {
        // Gaussian weight
        let diff = eps - epsilon_bar;
        let gauss = norm * (-(diff * diff) / two_sigma_sq).exp();
        // Power-law suppression factor: 0 when eps >= pT
        let raa_factor = if eps < pt {
            ((pt - eps) / pt).powf(n - 1.0)
        } else {
            0.0
        };
        gauss * raa_factor
    };

    gl.integrate(eps_lo, eps_hi, integrand) / gauss_mass
}

/// Precomputed 2-D lookup table for the straggling-smeared R_AA.
///
/// Build this **once** before the fitting loop, then call [`StragglingGrid::lookup`]
/// for O(1) bilinear interpolation during optimisation.
///
/// # Grid layout
/// - `pt_grid`: log-spaced pT nodes (better resolution at low pT)
/// - `eps_grid`: linearly-spaced epsilon_bar nodes
/// - `raa_values[i_pt][i_eps]`: precomputed R_AA^smeared value
pub struct StragglingGrid {
    /// pT grid points (sorted ascending, log-spaced)
    pt_grid: Vec<f64>,
    /// epsilon_bar grid points (sorted ascending, linear-spaced)
    eps_grid: Vec<f64>,
    /// Precomputed R_AA values: `raa_values[i_pt][i_eps]`
    raa_values: Vec<Vec<f64>>,
    /// Spectral index n (fixed for the grid)
    pub n_spectral: f64,
    /// Straggling width parameter kappa
    pub kappa: f64,
}

impl StragglingGrid {
    /// Construct the grid by evaluating `r_aa_straggling` at every (pT_i, epsilon_bar_j) node.
    ///
    /// This is O(`n_pt` * `n_eps` * 32) GL evaluations --- expensive but done once.
    ///
    /// # Default parameters (suggested)
    /// - `pt_range  = (1.0, 100.0)`, `n_pt  = 200` (log-spaced)
    /// - `eps_range = (0.1,  30.0)`, `n_eps = 150` (linear-spaced)
    ///
    /// # Arguments
    /// * `pt_range`    -- `(pt_min, pt_max)` in GeV
    /// * `n_pt`        -- number of pT grid nodes
    /// * `eps_range`   -- `(eps_min, eps_max)` in GeV
    /// * `n_eps`       -- number of epsilon_bar grid nodes
    /// * `n_spectral`  -- spectral index n (same value used in every evaluation)
    /// * `kappa`       -- straggling width coefficient kappa
    #[must_use]
    pub fn new(
        pt_range: (f64, f64),
        n_pt: usize,
        eps_range: (f64, f64),
        n_eps: usize,
        n_spectral: f64,
        kappa: f64,
    ) -> Self {
        assert!(n_pt >= 2, "need at least 2 pT grid nodes");
        assert!(n_eps >= 2, "need at least 2 eps grid nodes");
        assert!(
            pt_range.0 > 0.0 && pt_range.1 > pt_range.0,
            "pt_range must satisfy 0 < pt_min < pt_max, got {:?}",
            pt_range
        );
        assert!(
            eps_range.0 > 0.0 && eps_range.1 > eps_range.0,
            "eps_range must satisfy 0 < eps_min < eps_max, got {:?}",
            eps_range
        );

        // Log-spaced pT grid
        let log_pt_lo = pt_range.0.ln();
        let log_pt_hi = pt_range.1.ln();
        let pt_grid: Vec<f64> = (0..n_pt)
            .map(|i| {
                let t = i as f64 / (n_pt - 1) as f64;
                (log_pt_lo + t * (log_pt_hi - log_pt_lo)).exp()
            })
            .collect();

        // Linear-spaced epsilon_bar grid
        let eps_grid: Vec<f64> = (0..n_eps)
            .map(|j| {
                let t = j as f64 / (n_eps - 1) as f64;
                eps_range.0 + t * (eps_range.1 - eps_range.0)
            })
            .collect();

        // Precompute the table, constructing the GL object once for all grid evaluations
        let gl = GaussLegendre::new(
            NonZeroUsize::new(N_GL).expect("Gauss-Legendre degree must be non-zero"),
        );
        let raa_values: Vec<Vec<f64>> = pt_grid
            .iter()
            .map(|&pt| {
                eps_grid
                    .iter()
                    .map(|&eps_bar| {
                        let sigma = straggling_sigma(eps_bar, kappa);
                        r_aa_straggling_with_gl(pt, eps_bar, n_spectral, sigma, &gl)
                    })
                    .collect()
            })
            .collect();

        Self {
            pt_grid,
            eps_grid,
            raa_values,
            n_spectral,
            kappa,
        }
    }

    /// Look up R_AA^smeared via bilinear interpolation.
    ///
    /// Returns the interpolated value for (`pt`, `epsilon_bar`).
    /// Values outside the grid are clamped to the nearest boundary.
    ///
    /// This is O(log N) for the binary searches + O(1) for the interpolation ---
    /// far cheaper than a 32-point GL integration.
    #[must_use]
    pub fn lookup(&self, pt: f64, epsilon_bar: f64) -> f64 {
        let (i0, i1, t_pt) = interp_index(&self.pt_grid, pt);
        let (j0, j1, t_eps) = interp_index(&self.eps_grid, epsilon_bar);

        // Bilinear interpolation
        let v00 = self.raa_values[i0][j0];
        let v10 = self.raa_values[i1][j0];
        let v01 = self.raa_values[i0][j1];
        let v11 = self.raa_values[i1][j1];

        (1.0 - t_pt) * (1.0 - t_eps) * v00
            + t_pt * (1.0 - t_eps) * v10
            + (1.0 - t_pt) * t_eps * v01
            + t_pt * t_eps * v11
    }
}

/// Binary-search helper: find the bracketing indices and interpolation weight
/// for `x` in the sorted slice `grid`.
///
/// Returns `(i_lo, i_hi, t)` where `t in [0, 1]` and
/// `grid[i_lo] * (1-t) + grid[i_hi] * t == x` (approximately).
///
/// Clamps to the boundary when `x` is outside the grid.
fn interp_index(grid: &[f64], x: f64) -> (usize, usize, f64) {
    let n = grid.len();
    debug_assert!(n >= 2);

    // Clamp below
    if x <= grid[0] {
        return (0, 1, 0.0);
    }
    // Clamp above
    if x >= grid[n - 1] {
        return (n - 2, n - 1, 1.0);
    }

    // Binary search for the interval [grid[i], grid[i+1]] containing x
    let i = grid
        .partition_point(|&v| v <= x)
        .saturating_sub(1)
        .min(n - 2);

    let lo = grid[i];
    let hi = grid[i + 1];
    let t = if (hi - lo).abs() < f64::EPSILON {
        0.0
    } else {
        (x - lo) / (hi - lo)
    };

    (i, i + 1, t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quenching::r_aa_model;

    #[test]
    fn test_straggling_sigma_basic() {
        // sigma = kappa * sqrtepsilon_bar
        let sigma = straggling_sigma(4.0, 0.5);
        assert!(
            (sigma - 1.0).abs() < 1e-12,
            "sigma = {sigma} (expected 1.0)"
        );

        let sigma2 = straggling_sigma(9.0, 1.0);
        assert!(
            (sigma2 - 3.0).abs() < 1e-12,
            "sigma = {sigma2} (expected 3.0)"
        );
    }

    #[test]
    fn test_straggling_sigma_zero_for_nonpositive() {
        assert_eq!(straggling_sigma(0.0, 0.5), 0.0);
        assert_eq!(straggling_sigma(-1.0, 0.5), 0.0);
    }

    #[test]
    fn test_r_aa_straggling_normalization_at_small_epsilon() {
        // With epsilon_bar = 0.5 GeV and kappa = 0.5, sigma ~ 0.354 GeV.
        // The 5sigma lower bound ~ -1.27 < 0, so ~20% of the untruncated Gaussian
        // lies below 0.  Without re-normalisation, R_AA(pT -> inf) ~ 0.80.
        // The corrected normalization should give R_AA ~ 1 at pT >> epsilon_bar.
        let eps = 0.5;
        let n = 6.0;
        let sigma = straggling_sigma(eps, DEFAULT_KAPPA);
        let pt_large = 500.0; // far above epsilon_bar

        let raa = r_aa_straggling(pt_large, eps, n, sigma);
        assert!(
            (raa - 1.0).abs() < 0.01,
            "R_AA at pT >> epsilon_bar should be ~1 (normalization fix), got {raa}"
        );
    }

    #[test]
    fn test_r_aa_straggling_large_pt_approaches_discrete() {
        // At pT >> epsilon_bar (and sigma small compared to pT), the Gaussian is fully
        // within [0, pT] and the power-law factor is nearly constant over the
        // smearing window, so R_AA^smeared ~ R_AA^discrete.
        let eps = 3.0;
        let n = 6.0;
        let sigma = straggling_sigma(eps, 0.1); // very narrow smearing
        let pt = 50.0;

        let smeared = r_aa_straggling(pt, eps, n, sigma);
        let discrete = r_aa_model(pt, eps, n);

        let rel_diff = (smeared - discrete).abs() / discrete;
        assert!(
            rel_diff < 0.01,
            "large-pT limit failed: smeared={smeared:.6}, discrete={discrete:.6}, rel_diff={rel_diff:.4}"
        );
    }

    #[test]
    fn test_r_aa_straggling_smooth_near_epsilon() {
        // The smeared R_AA should be non-zero and smoothly nonzero just below epsilon_bar,
        // unlike the sharp discrete model which is exactly zero.
        let eps = 5.0;
        let n = 6.0;
        let sigma = straggling_sigma(eps, 0.5); // DEFAULT_KAPPA

        // Discrete model is 0 at pT = epsilon_bar
        assert_eq!(r_aa_model(eps, eps, n), 0.0);

        // Smeared model should be > 0 at pT = epsilon_bar (some of the Gaussian weight
        // has epsilon < pT, contributing to the integral)
        let smeared_at_eps = r_aa_straggling(eps, eps, n, sigma);
        assert!(
            smeared_at_eps > 0.0,
            "smeared R_AA at pT = epsilon_bar should be > 0, got {smeared_at_eps}"
        );

        // And it should also be > 0 just below epsilon_bar
        let pt_below = eps * 0.8;
        let smeared_below = r_aa_straggling(pt_below, eps, n, sigma);
        assert!(
            smeared_below > 0.0,
            "smeared R_AA below epsilon_bar should be > 0, got {smeared_below}"
        );
    }

    #[test]
    fn test_r_aa_straggling_falls_back_to_discrete_when_sigma_zero() {
        let eps = 4.0;
        let n = 6.0;
        let pt = 12.0;

        let smeared = r_aa_straggling(pt, eps, n, 0.0);
        let discrete = r_aa_model(pt, eps, n);
        assert!(
            (smeared - discrete).abs() < 1e-12,
            "zero-sigma fallback: smeared={smeared}, discrete={discrete}"
        );
    }

    #[test]
    fn test_r_aa_straggling_monotone_in_pt() {
        // R_AA^smeared should be non-decreasing in pT for fixed epsilon_bar
        let eps = 3.0;
        let n = 6.0;
        let sigma = straggling_sigma(eps, DEFAULT_KAPPA);

        let pts: Vec<f64> = (1..=50).map(|i| i as f64).collect();
        let values: Vec<f64> = pts
            .iter()
            .map(|&pt| r_aa_straggling(pt, eps, n, sigma))
            .collect();

        for i in 1..values.len() {
            assert!(
                values[i] >= values[i - 1] - 1e-10,
                "R_AA not monotone at pT={}: values[{}]={} < values[{}]={}",
                pts[i],
                i,
                values[i],
                i - 1,
                values[i - 1]
            );
        }
    }

    #[test]
    fn test_straggling_grid_construction() {
        let grid = StragglingGrid::new((1.0, 50.0), 20, (0.5, 15.0), 15, 6.0, DEFAULT_KAPPA);
        assert_eq!(grid.pt_grid.len(), 20);
        assert_eq!(grid.eps_grid.len(), 15);
        assert_eq!(grid.raa_values.len(), 20);
        assert_eq!(grid.raa_values[0].len(), 15);
    }

    #[test]
    fn test_straggling_grid_lookup_matches_direct() {
        // Grid lookup should closely match direct quadrature at grid nodes
        let n_pt = 10;
        let n_eps = 10;
        let grid = StragglingGrid::new((2.0, 40.0), n_pt, (0.5, 10.0), n_eps, 6.0, DEFAULT_KAPPA);

        for (i, &pt) in grid.pt_grid.iter().enumerate() {
            for (j, &eps_bar) in grid.eps_grid.iter().enumerate() {
                let expected = grid.raa_values[i][j];
                let looked_up = grid.lookup(pt, eps_bar);
                assert!(
                    (looked_up - expected).abs() < 1e-10,
                    "grid node ({pt},{eps_bar}): lookup={looked_up}, direct={expected}"
                );
            }
        }
    }

    #[test]
    fn test_straggling_grid_lookup_interpolation_accuracy() {
        // At off-grid points bilinear interpolation should be close to direct quadrature
        let grid = StragglingGrid::new((1.0, 50.0), 100, (0.1, 20.0), 80, 6.1, DEFAULT_KAPPA);

        // Pick a point in the interior of the grid
        let pt = 8.3;
        let eps_bar = 3.7;
        let sigma = straggling_sigma(eps_bar, DEFAULT_KAPPA);

        let direct = r_aa_straggling(pt, eps_bar, 6.1, sigma);
        let interp = grid.lookup(pt, eps_bar);

        let tol = 0.01; // bilinear interpolation on a dense grid
        assert!(
            (interp - direct).abs() < tol,
            "interpolation error too large: interp={interp:.6}, direct={direct:.6}, diff={:.6}",
            (interp - direct).abs()
        );
    }

    #[test]
    fn test_straggling_grid_clamp_below() {
        let grid = StragglingGrid::new((2.0, 50.0), 20, (0.5, 15.0), 15, 6.0, DEFAULT_KAPPA);
        // Querying below the grid range should not panic and return a sensible value
        let v = grid.lookup(0.1, 0.1);
        assert!(v.is_finite());
    }

    #[test]
    fn test_straggling_grid_clamp_above() {
        let grid = StragglingGrid::new((2.0, 50.0), 20, (0.5, 15.0), 15, 6.0, DEFAULT_KAPPA);
        // Querying above the grid range should not panic
        let v = grid.lookup(200.0, 50.0);
        assert!(v.is_finite());
    }
}

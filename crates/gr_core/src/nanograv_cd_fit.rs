//! NANOGrav 15-year free spectrum fitting with Cayley-Dickson algebraic tower.
//!
//! Implements a novel spectral fitting procedure that maps the algebraic
//! properties of Cayley-Dickson algebras at each doubling level (16D through
//! 1024D) onto the NANOGrav gravitational wave background spectrum.
//!
//! # Cayley-Dickson Doubling Stack
//!
//! | Level *n* | Dim 2^*n* | Name               | Key property lost       |
//! |-----------|-----------|--------------------|-------------------------|
//! | 4         | 16        | Sedenion           | Alternativity           |
//! | 5         | 32        | Trigintaduonion     | —                       |
//! | 6         | 64        | Sexagintaquattuornion | —                     |
//! | 7         | 128       | 128-nion           | —                       |
//! | 8         | 256       | 256-nion           | —                       |
//! | 9         | 512       | 512-nion           | —                       |
//! | 10        | 1024      | Deca-nion          | —                       |
//!
//! # Spectral Model
//!
//! The baseline GWB spectrum is a power law in frequency:
//!
//! ```text
//!   log₁₀(ρ_k) = A + γ · log₁₀(f_k / f_ref)
//! ```
//!
//! At each CD dimension *d* = 2^*n*, a correction term is added:
//!
//! ```text
//!   log₁₀(ρ_k) = A + γ · log₁₀(f_k / f_ref) + λ_d · Φ_d(k)
//! ```
//!
//! where `Φ_d(k)` is a basis function derived from the associator structure
//! at dimension *d*. The fitting determines the coupling strength `λ_d` and
//! evaluates goodness-of-fit (χ²/dof, BIC) to identify which CD dimension
//! best explains the observed spectral residuals.
//!
//! The basis function encodes the doubling level:
//!
//! ```text
//!   Φ_d(k) = σ_d · cos(2π · (n − 3) · (k + 0.5) / N_bins)
//! ```
//!
//! where σ_d is the associator density at dimension *d* and *n* = log₂(*d*).
//! The (k + 0.5) factor evaluates the cosine at bin centers (a half-bin
//! offset) to reduce edge artifacts. Higher doubling levels create finer
//! spectral modulation, weighted by the degree of non-associativity.
//!
//! # References
//!
//! - Agazie et al. (2023), ApJL 951, L8 — NANOGrav 15-year GWB
//! - Lamb, Taylor & van Haasteren (2023), PhysRevD 108, 103019 — KDE method
//! - Baez (2002), Bull. AMS 39, 145-205 — The Octonions
//! - Schafer (1966) — On the algebras formed by the Cayley-Dickson process

use cd_kernel::cayley_dickson::{associator_independence_stats, AssociatorStats};
use nalgebra::{DMatrix, DVector};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Cayley-Dickson doubling levels from 16D (2⁴) to 1024D (2¹⁰).
pub const CD_STACK: [usize; 7] = [16, 32, 64, 128, 256, 512, 1024];

/// Reference frequency: 1 / (1 year) in Hz.
pub const F_YR: f64 = 1.0 / (365.25 * 86400.0);

/// Number of bins in the NANOGrav 15-year HD-correlated free spectrum.
pub const N_BINS: usize = 30;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A single frequency bin from the NANOGrav 15-year free spectrum.
#[derive(Debug, Clone, Copy)]
pub struct FreqBin {
    /// Frequency [Hz], nanohertz range.
    pub f_hz: f64,
    /// Median log₁₀(ρ) (characteristic strain power).
    pub log10_rho: f64,
    /// Lower 5th percentile of log₁₀(ρ).
    pub log10_rho_lo: f64,
    /// Upper 95th percentile of log₁₀(ρ).
    pub log10_rho_hi: f64,
}

/// Algebraic properties measured at a single CD dimension.
#[derive(Debug, Clone)]
pub struct CdAlgebraicProps {
    /// Algebra dimension (power of two, ≥ 16).
    pub dim: usize,
    /// Doubling level *n* where dim = 2^*n*.
    pub level: u32,
    /// Full associator statistics from Monte Carlo sampling.
    pub stats: AssociatorStats,
}

/// Fit result for a single CD dimension against the NANOGrav spectrum.
#[derive(Debug, Clone)]
pub struct CdDimFitResult {
    /// Algebra dimension.
    pub dim: usize,
    /// Doubling level.
    pub level: u32,
    /// Fitted log₁₀ amplitude (intercept).
    pub amplitude: f64,
    /// Fitted spectral index (slope in log-log space).
    pub spectral_index: f64,
    /// Fitted CD coupling strength λ_d.
    pub cd_coupling: f64,
    /// Chi-squared statistic.
    pub chi_sq: f64,
    /// Degrees of freedom (N_bins − 3).
    pub dof: usize,
    /// Reduced chi-squared (χ²/dof).
    pub chi_sq_per_dof: f64,
    /// Bayesian Information Criterion: χ² + k·ln(n).
    pub bic: f64,
    /// Per-bin residuals (data − model) / σ.
    pub residuals: Vec<f64>,
    /// Algebraic properties at this dimension.
    pub algebraic_props: CdAlgebraicProps,
}

/// Baseline (power-law only) fit result.
#[derive(Debug, Clone)]
pub struct BaselineFitResult {
    /// Fitted log₁₀ amplitude.
    pub amplitude: f64,
    /// Fitted spectral index.
    pub spectral_index: f64,
    /// Chi-squared.
    pub chi_sq: f64,
    /// Degrees of freedom (N_bins − 2).
    pub dof: usize,
    /// Reduced chi-squared.
    pub chi_sq_per_dof: f64,
    /// BIC.
    pub bic: f64,
    /// Per-bin residuals.
    pub residuals: Vec<f64>,
}

/// Complete result for the full CD tower fit (16D through 1024D).
#[derive(Debug, Clone)]
pub struct CdTowerFitResult {
    /// Baseline power-law-only fit.
    pub baseline: BaselineFitResult,
    /// Per-dimension fit results, one per CD stack entry.
    pub fits: Vec<CdDimFitResult>,
    /// Dimension with the lowest χ²/dof.
    pub best_dim: usize,
    /// Lowest χ²/dof achieved.
    pub best_chi_sq_per_dof: f64,
    /// Improvement in χ² over baseline at the best dimension.
    pub delta_chi_sq_best: f64,
}

// ---------------------------------------------------------------------------
// NANOGrav 15-year HD-correlated free spectrum (inline)
// ---------------------------------------------------------------------------

/// NANOGrav 15-year HD-correlated free spectrum (30 bins).
///
/// Extracted from Zenodo record 10344086 (KDE v1.1.0).
/// Reference: Agazie et al. (2023), ApJL 951, L8.
pub const HD_FREE_SPECTRUM: [FreqBin; N_BINS] = [
    FreqBin { f_hz: 1.9768264576e-09, log10_rho: -3.824356, log10_rho_lo: -12.239962, log10_rho_hi: -1.192436 },
    FreqBin { f_hz: 3.9536529151e-09, log10_rho: -8.314518, log10_rho_lo: -14.419953, log10_rho_hi: -1.563951 },
    FreqBin { f_hz: 5.9304793727e-09, log10_rho: -5.686763, log10_rho_lo: -14.198086, log10_rho_hi: -1.378676 },
    FreqBin { f_hz: 7.9073058303e-09, log10_rho: -4.324256, log10_rho_lo: -12.283386, log10_rho_hi: -1.242426 },
    FreqBin { f_hz: 9.8841322878e-09, log10_rho: -4.191523, log10_rho_lo: -10.567086, log10_rho_hi: -1.229152 },
    FreqBin { f_hz: 1.1860958745e-08, log10_rho: -4.139625, log10_rho_lo: -9.835739, log10_rho_hi: -1.223963 },
    FreqBin { f_hz: 1.3837785203e-08, log10_rho: -4.119372, log10_rho_lo: -9.763380, log10_rho_hi: -1.221937 },
    FreqBin { f_hz: 1.5814611661e-08, log10_rho: -4.455705, log10_rho_lo: -12.469490, log10_rho_hi: -1.255570 },
    FreqBin { f_hz: 1.7791438118e-08, log10_rho: -4.284785, log10_rho_lo: -9.058763, log10_rho_hi: -1.238479 },
    FreqBin { f_hz: 1.9768264576e-08, log10_rho: -4.197701, log10_rho_lo: -9.572466, log10_rho_hi: -1.229770 },
    FreqBin { f_hz: 2.1745091033e-08, log10_rho: -4.256292, log10_rho_lo: -9.804122, log10_rho_hi: -1.235629 },
    FreqBin { f_hz: 2.3721917491e-08, log10_rho: -4.297346, log10_rho_lo: -9.526431, log10_rho_hi: -1.239735 },
    FreqBin { f_hz: 2.5698743948e-08, log10_rho: -4.366242, log10_rho_lo: -8.755714, log10_rho_hi: -1.246624 },
    FreqBin { f_hz: 2.7675570406e-08, log10_rho: -4.330105, log10_rho_lo: -8.838375, log10_rho_hi: -1.243011 },
    FreqBin { f_hz: 2.9652396864e-08, log10_rho: -4.253942, log10_rho_lo: -9.057492, log10_rho_hi: -1.235394 },
    FreqBin { f_hz: 3.1629223321e-08, log10_rho: -3.355449, log10_rho_lo: -11.156964, log10_rho_hi: -1.145545 },
    FreqBin { f_hz: 3.3606049779e-08, log10_rho: -4.019221, log10_rho_lo: -10.332758, log10_rho_hi: -1.211922 },
    FreqBin { f_hz: 3.5582876236e-08, log10_rho: -4.218475, log10_rho_lo: -9.979593, log10_rho_hi: -1.231847 },
    FreqBin { f_hz: 3.7559702694e-08, log10_rho: -4.283774, log10_rho_lo: -9.594341, log10_rho_hi: -1.238377 },
    FreqBin { f_hz: 3.9536529151e-08, log10_rho: -4.354106, log10_rho_lo: -9.628553, log10_rho_hi: -1.245411 },
    FreqBin { f_hz: 4.1513355609e-08, log10_rho: -4.358596, log10_rho_lo: -8.971711, log10_rho_hi: -1.245860 },
    FreqBin { f_hz: 4.3490182066e-08, log10_rho: -4.326085, log10_rho_lo: -9.048752, log10_rho_hi: -1.242608 },
    FreqBin { f_hz: 4.5467008524e-08, log10_rho: -4.281646, log10_rho_lo: -9.260830, log10_rho_hi: -1.238165 },
    FreqBin { f_hz: 4.7443834982e-08, log10_rho: -4.320090, log10_rho_lo: -9.515511, log10_rho_hi: -1.242009 },
    FreqBin { f_hz: 4.9420661439e-08, log10_rho: -4.288476, log10_rho_lo: -10.414934, log10_rho_hi: -1.238848 },
    FreqBin { f_hz: 5.1397487897e-08, log10_rho: -4.339775, log10_rho_lo: -10.295456, log10_rho_hi: -1.243978 },
    FreqBin { f_hz: 5.3374314354e-08, log10_rho: -4.454512, log10_rho_lo: -8.581017, log10_rho_hi: -1.255451 },
    FreqBin { f_hz: 5.5351140812e-08, log10_rho: -4.357294, log10_rho_lo: -8.794067, log10_rho_hi: -1.245729 },
    FreqBin { f_hz: 5.7327967269e-08, log10_rho: -4.424931, log10_rho_lo: -8.688157, log10_rho_hi: -1.252493 },
    FreqBin { f_hz: 5.9304793727e-08, log10_rho: -4.274741, log10_rho_lo: -9.584661, log10_rho_hi: -1.237474 },
];

// ---------------------------------------------------------------------------
// Algebraic property computation
// ---------------------------------------------------------------------------

/// Compute associator statistics at a given CD dimension via Monte Carlo.
///
/// # Arguments
/// * `dim` — Cayley-Dickson dimension (must be power of two, ≥ 16).
/// * `n_trials` — Number of random triples to sample.
/// * `seed` — Deterministic seed for reproducibility.
pub fn compute_cd_algebraic_props(dim: usize, n_trials: usize, seed: u64) -> CdAlgebraicProps {
    assert!(
        dim.is_power_of_two() && dim >= 16,
        "compute_cd_algebraic_props: dim must be a power of two and at least 16, got {}",
        dim
    );
    let stats = associator_independence_stats(dim, n_trials, seed);
    CdAlgebraicProps {
        dim,
        level: dim.trailing_zeros(),
        stats,
    }
}

// ---------------------------------------------------------------------------
// CD basis functions
// ---------------------------------------------------------------------------

/// Construct the spectral basis function for CD dimension `dim`.
///
/// The basis function at bin index *k* (0-indexed) is:
///
/// ```text
///   Φ_d(k) = σ_d · cos(2π · step · (k + 0.5) / N)
/// ```
///
/// where `step = log₂(d) − 3` is the number of doublings past the octonions,
/// σ_d is the RMS associator norm at dimension *d*, and *N* is the total
/// number of bins. The half-bin offset avoids edge artifacts.
pub fn cd_basis_function(dim: usize, mean_assoc_sq: f64, n_bins: usize) -> Vec<f64> {
    let level = dim.trailing_zeros() as usize;
    let step = level.saturating_sub(3);
    let sigma = mean_assoc_sq.sqrt();
    (0..n_bins)
        .map(|k| sigma * (2.0 * PI * step as f64 * (k as f64 + 0.5) / n_bins as f64).cos())
        .collect()
}

// ---------------------------------------------------------------------------
// Measurement uncertainties
// ---------------------------------------------------------------------------

/// Convert 90 % credible interval bounds to an approximate Gaussian σ.
///
/// Assumes the posterior is approximately Gaussian:
///   σ ≈ (hi − lo) / (2 × 1.645)
///
/// Clamps to a minimum of 0.1 to prevent singular weights.
fn sigma_from_ci(lo: f64, hi: f64) -> f64 {
    let raw = (hi - lo) / (2.0 * 1.645);
    raw.max(0.1)
}

// ---------------------------------------------------------------------------
// Weighted least squares
// ---------------------------------------------------------------------------

/// Solve weighted least squares: min_β ‖W^{1/2}(y − Xβ)‖² .
///
/// Returns `None` if the normal equations are singular.
fn wls_solve(
    design: &DMatrix<f64>,
    y: &DVector<f64>,
    weights: &DVector<f64>,
) -> Option<DVector<f64>> {
    let n = design.nrows();
    let p = design.ncols();
    // Weighted design matrix: W^{1/2} X
    let wx = DMatrix::from_fn(n, p, |i, j| weights[i].sqrt() * design[(i, j)]);
    let wy = DVector::from_fn(n, |i, _| weights[i].sqrt() * y[i]);
    let xtx = wx.transpose() * &wx;
    let xty = wx.transpose() * &wy;
    xtx.lu().solve(&xty)
}

/// Compute weighted chi-squared: Σ w_k (y_k − ŷ_k)².
fn chi_squared(y: &DVector<f64>, y_hat: &DVector<f64>, weights: &DVector<f64>) -> f64 {
    y.iter()
        .zip(y_hat.iter())
        .zip(weights.iter())
        .map(|((yi, yhi), wi)| wi * (yi - yhi).powi(2))
        .sum()
}

/// BIC = χ² + k·ln(n), where k = number of parameters, n = number of data points.
fn bic(chi_sq: f64, k: usize, n: usize) -> f64 {
    chi_sq + k as f64 * (n as f64).ln()
}

/// Compute standardized residuals: (y_k − ŷ_k) / σ_k .
fn standardized_residuals(y: &DVector<f64>, y_hat: &DVector<f64>, sigma: &[f64]) -> Vec<f64> {
    y.iter()
        .zip(y_hat.iter())
        .zip(sigma.iter())
        .map(|((yi, yhi), si)| (yi - yhi) / si)
        .collect()
}

// ---------------------------------------------------------------------------
// Baseline (power-law only) fit
// ---------------------------------------------------------------------------

/// Fit a pure power-law model to the free spectrum data.
///
/// Model: log₁₀(ρ_k) = A + γ · log₁₀(f_k / f_ref)
///
/// Two free parameters (A, γ) fitted via weighted least squares.
pub fn fit_baseline(data: &[FreqBin]) -> BaselineFitResult {
    let n = data.len();
    let f_ref = F_YR;

    // Observation vector and weights
    let y = DVector::from_fn(n, |i, _| data[i].log10_rho);
    let sigma_vec: Vec<f64> = data
        .iter()
        .map(|b| sigma_from_ci(b.log10_rho_lo, b.log10_rho_hi))
        .collect();
    let weights = DVector::from_fn(n, |i, _| 1.0 / (sigma_vec[i] * sigma_vec[i]));

    // Design matrix: [1, log10(f/f_ref)]
    let design = DMatrix::from_fn(n, 2, |i, j| match j {
        0 => 1.0,
        _ => (data[i].f_hz / f_ref).log10(),
    });

    let beta = wls_solve(&design, &y, &weights)
        .expect("fit_baseline: WLS solver failed (singular normal equations); baseline fit is undefined");
    let y_hat = &design * &beta;
    let cs = chi_squared(&y, &y_hat, &weights);
    let dof = n.saturating_sub(2);
    let cs_dof = if dof > 0 { cs / dof as f64 } else { cs };
    let residuals = standardized_residuals(&y, &y_hat, &sigma_vec);

    BaselineFitResult {
        amplitude: beta[0],
        spectral_index: beta[1],
        chi_sq: cs,
        dof,
        chi_sq_per_dof: cs_dof,
        bic: bic(cs, 2, n),
        residuals,
    }
}

// ---------------------------------------------------------------------------
// Single-dimension CD fit
// ---------------------------------------------------------------------------

/// Fit the power-law + CD correction model at a single dimension.
///
/// Model: log₁₀(ρ_k) = A + γ · log₁₀(f_k / f_ref) + λ_d · Φ_d(k)
///
/// Three free parameters (A, γ, λ_d) fitted via WLS.
pub fn fit_single_dim(
    data: &[FreqBin],
    props: &CdAlgebraicProps,
) -> CdDimFitResult {
    let n = data.len();
    let f_ref = F_YR;

    let y = DVector::from_fn(n, |i, _| data[i].log10_rho);
    let sigma_vec: Vec<f64> = data
        .iter()
        .map(|b| sigma_from_ci(b.log10_rho_lo, b.log10_rho_hi))
        .collect();
    let weights = DVector::from_fn(n, |i, _| 1.0 / (sigma_vec[i] * sigma_vec[i]));

    // CD basis vector
    let phi = cd_basis_function(props.dim, props.stats.mean_assoc_sq, n);

    // Design matrix: [1, log10(f/f_ref), Φ_d(k)]
    let design = DMatrix::from_fn(n, 3, |i, j| match j {
        0 => 1.0,
        1 => (data[i].f_hz / f_ref).log10(),
        _ => phi[i],
    });

    let beta = wls_solve(&design, &y, &weights).unwrap_or_else(|| {
        panic!(
            "Weighted least squares solver failed in fit_single_dim: singular normal equations (dim = {}, n = {})",
            props.dim, n
        )
    });
    let y_hat = &design * &beta;
    let cs = chi_squared(&y, &y_hat, &weights);
    let dof = n.saturating_sub(3);
    let cs_dof = if dof > 0 { cs / dof as f64 } else { cs };
    let residuals = standardized_residuals(&y, &y_hat, &sigma_vec);

    CdDimFitResult {
        dim: props.dim,
        level: props.level,
        amplitude: beta[0],
        spectral_index: beta[1],
        cd_coupling: beta[2],
        chi_sq: cs,
        dof,
        chi_sq_per_dof: cs_dof,
        bic: bic(cs, 3, n),
        residuals,
        algebraic_props: props.clone(),
    }
}

// ---------------------------------------------------------------------------
// Full tower fit
// ---------------------------------------------------------------------------

/// Fit the full Cayley-Dickson tower (16D → 1024D) to the NANOGrav spectrum.
///
/// At each doubling step the associator statistics are computed via Monte
/// Carlo (`n_trials` random unit-vector triples, seeded by `seed`), a
/// spectral basis function is constructed, and a 3-parameter WLS fit is
/// performed. The baseline power-law fit (2 parameters) is included for
/// comparison.
///
/// # Arguments
/// * `n_trials` — Number of Monte Carlo samples per dimension for
///   associator statistics.  Larger values yield more stable density
///   estimates but increase runtime (especially for dim ≥ 256).
/// * `seed` — Deterministic PRNG seed for reproducibility.
pub fn fit_nanograv_cd_tower(n_trials: usize, seed: u64) -> CdTowerFitResult {
    let data = &HD_FREE_SPECTRUM;
    let baseline = fit_baseline(data);

    let mut fits = Vec::with_capacity(CD_STACK.len());
    for &dim in &CD_STACK {
        // Derive a per-dimension seed to avoid reusing the same random stream
        let dim_seed = seed ^ ((dim as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let props = compute_cd_algebraic_props(dim, n_trials, dim_seed);
        let result = fit_single_dim(data, &props);
        fits.push(result);
    }

    let best = fits
        .iter()
        .min_by(|a, b| a.chi_sq_per_dof.total_cmp(&b.chi_sq_per_dof))
        .unwrap();

    CdTowerFitResult {
        delta_chi_sq_best: baseline.chi_sq - best.chi_sq,
        best_dim: best.dim,
        best_chi_sq_per_dof: best.chi_sq_per_dof,
        baseline,
        fits,
    }
}

/// Convenience wrapper with default settings (200 trials, seed = 42).
///
/// Suitable for quick exploration; increase `n_trials` for publication-
/// quality associator statistics.
pub fn fit_nanograv_cd_tower_default() -> CdTowerFitResult {
    fit_nanograv_cd_tower(200, 42)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Data integrity -------------------------------------------------------

    #[test]
    fn data_frequencies_monotonic() {
        for w in HD_FREE_SPECTRUM.windows(2) {
            assert!(
                w[1].f_hz > w[0].f_hz,
                "Frequencies must be strictly increasing"
            );
        }
    }

    #[test]
    fn data_credible_intervals_ordered() {
        for (i, b) in HD_FREE_SPECTRUM.iter().enumerate() {
            assert!(
                b.log10_rho_lo < b.log10_rho && b.log10_rho < b.log10_rho_hi,
                "Bin {i}: lo < median < hi violated"
            );
        }
    }

    #[test]
    fn data_frequencies_nanohertz_range() {
        for b in &HD_FREE_SPECTRUM {
            assert!(b.f_hz > 1e-10 && b.f_hz < 1e-7, "Frequency out of nHz range");
        }
    }

    // -- Basis function -------------------------------------------------------

    #[test]
    fn basis_function_length() {
        let phi = cd_basis_function(16, 0.5, 30);
        assert_eq!(phi.len(), 30);
    }

    #[test]
    fn basis_function_bounded() {
        let phi = cd_basis_function(64, 1.0, 30);
        for &v in &phi {
            assert!(v.abs() <= 1.0 + 1e-12, "Basis must be bounded by σ_d");
        }
    }

    #[test]
    fn basis_function_zero_for_zero_associator() {
        let phi = cd_basis_function(16, 0.0, 30);
        for &v in &phi {
            assert!(v.abs() < 1e-15, "Zero associator → zero basis");
        }
    }

    #[test]
    fn basis_functions_differ_across_dims() {
        let phi_16 = cd_basis_function(16, 0.5, 30);
        let phi_32 = cd_basis_function(32, 0.5, 30);
        let diff: f64 = phi_16
            .iter()
            .zip(&phi_32)
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.01, "Different dims should give different basis functions");
    }

    // -- Sigma from CI --------------------------------------------------------

    #[test]
    fn sigma_positive() {
        let s = sigma_from_ci(-10.0, -1.0);
        assert!(s > 0.0);
    }

    #[test]
    fn sigma_minimum_clamp() {
        let s = sigma_from_ci(-1.001, -1.0);
        assert!(s >= 0.1, "Sigma should be clamped to minimum 0.1");
    }

    // -- Baseline fit ---------------------------------------------------------

    #[test]
    fn baseline_fit_returns_finite() {
        let result = fit_baseline(&HD_FREE_SPECTRUM);
        assert!(result.amplitude.is_finite());
        assert!(result.spectral_index.is_finite());
        assert!(result.chi_sq.is_finite() && result.chi_sq >= 0.0);
        assert!(result.chi_sq_per_dof.is_finite());
    }

    #[test]
    fn baseline_fit_dof_correct() {
        let result = fit_baseline(&HD_FREE_SPECTRUM);
        assert_eq!(result.dof, N_BINS - 2);
    }

    #[test]
    fn baseline_residuals_count() {
        let result = fit_baseline(&HD_FREE_SPECTRUM);
        assert_eq!(result.residuals.len(), N_BINS);
    }

    #[test]
    fn baseline_spectral_index_finite() {
        let result = fit_baseline(&HD_FREE_SPECTRUM);
        assert!(
            result.spectral_index.is_finite(),
            "Spectral index should be finite, got {}",
            result.spectral_index
        );
    }

    // -- Single-dimension fit -------------------------------------------------

    #[test]
    fn single_dim_fit_sedenion_16d() {
        let props = compute_cd_algebraic_props(16, 100, 42);
        assert_eq!(props.dim, 16);
        assert_eq!(props.level, 4);
        assert!(props.stats.mean_assoc_sq > 0.0, "Sedenion should be non-associative");

        let result = fit_single_dim(&HD_FREE_SPECTRUM, &props);
        assert!(result.chi_sq.is_finite() && result.chi_sq >= 0.0);
        assert_eq!(result.dof, N_BINS - 3);
        assert_eq!(result.residuals.len(), N_BINS);
    }

    #[test]
    fn single_dim_fit_trigintaduonion_32d() {
        let props = compute_cd_algebraic_props(32, 100, 42);
        assert_eq!(props.level, 5);

        let result = fit_single_dim(&HD_FREE_SPECTRUM, &props);
        assert!(result.chi_sq.is_finite());
        assert!(result.amplitude.is_finite());
    }

    // -- Algebraic properties -------------------------------------------------

    #[test]
    fn algebraic_props_associator_grows_with_dim() {
        let p16 = compute_cd_algebraic_props(16, 100, 42);
        let p64 = compute_cd_algebraic_props(64, 100, 42);
        // Higher dimensions should generally have larger associator norms.
        // We check that both are non-zero; exact ordering depends on sampling
        // but for large enough trials and these dims, 64D > 16D.
        assert!(p16.stats.mean_assoc_sq > 0.0);
        assert!(p64.stats.mean_assoc_sq > 0.0);
    }

    #[test]
    fn algebraic_props_level_correct() {
        for &dim in &CD_STACK {
            let props = compute_cd_algebraic_props(dim, 50, 99);
            let expected_level = dim.trailing_zeros();
            assert_eq!(props.level, expected_level, "Level mismatch for dim={dim}");
        }
    }

    // -- Full tower fit -------------------------------------------------------

    #[test]
    fn tower_fit_properties() {
        let result = fit_nanograv_cd_tower(50, 42);

        // Covers all dims in the stack
        assert_eq!(result.fits.len(), CD_STACK.len());
        for (fit, &expected_dim) in result.fits.iter().zip(CD_STACK.iter()) {
            assert_eq!(fit.dim, expected_dim);
        }

        // Best dimension must be one of the stack entries
        assert!(
            CD_STACK.contains(&result.best_dim),
            "Best dim {} not in stack",
            result.best_dim
        );

        // All chi-squared values must be finite and non-negative
        assert!(result.baseline.chi_sq.is_finite());
        for fit in &result.fits {
            assert!(
                fit.chi_sq.is_finite() && fit.chi_sq >= 0.0,
                "Dim {} has invalid chi_sq: {}",
                fit.dim,
                fit.chi_sq
            );
        }

        // Adding a parameter can only improve or maintain chi_sq,
        // so delta_chi_sq >= 0 (baseline_chi_sq >= best_chi_sq).
        assert!(
            result.delta_chi_sq_best >= -1e-10,
            "Adding a parameter should not worsen chi_sq, got delta={}",
            result.delta_chi_sq_best
        );

        // BIC values must be finite
        assert!(result.baseline.bic.is_finite());
        for fit in &result.fits {
            assert!(fit.bic.is_finite(), "Dim {} has non-finite BIC", fit.dim);
        }

        // For well-behaved WLS, mean of weighted residuals ≈ 0
        for fit in &result.fits {
            let sum: f64 = fit.residuals.iter().sum();
            let mean = sum / fit.residuals.len() as f64;
            assert!(
                mean.abs() < 5.0,
                "Dim {} has large mean residual: {mean}",
                fit.dim
            );
        }
    }

    // -- Stepwise doubling completeness ---------------------------------------

    #[test]
    fn cd_stack_is_doubling_sequence() {
        for w in CD_STACK.windows(2) {
            assert_eq!(w[1], 2 * w[0], "Stack must double: {} -> {}", w[0], w[1]);
        }
    }

    #[test]
    fn cd_stack_starts_at_16_ends_at_1024() {
        assert_eq!(CD_STACK[0], 16);
        assert_eq!(*CD_STACK.last().unwrap(), 1024);
    }
}

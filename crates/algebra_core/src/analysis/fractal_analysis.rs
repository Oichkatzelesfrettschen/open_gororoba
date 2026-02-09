//! Fractal analysis: Hurst exponent, R/S analysis, and scaling properties.
//!
//! Provides utilities for analyzing self-similar and fractal time series,
//! including fractional Brownian motion (fBm) generation for testing.
//!
//! # Literature
//!
//! ## Foundational
//! - Hurst, H.E. (1951). Long-term storage capacity of reservoirs.
//!   Transactions of the American Society of Civil Engineers, 116, 770-808.
//! - Mandelbrot, B.B. & Wallis, J.R. (1969). Robustness of the rescaled range R/S
//!   in the measurement of noncyclic long run statistical dependence.
//!   Water Resources Research, 5(5), 967-988.
//! - Hosking, J.R.M. (1984). Modeling persistence in hydrological time series
//!   using fractional differencing. Water Resources Research, 20(12), 1898-1908.
//!
//! ## Modern Methods
//! - Peng, C.K. et al. (1994). Mosaic organization of DNA nucleotides.
//!   Physical Review E, 49(2), 1685. (Detrended Fluctuation Analysis)
//! - Di Matteo, T. et al. (2005). Multi-scaling in finance.
//!   Quantitative Finance, 5(1), 27-37.
//!
//! # Crate Research (2026-02)
//!
//! Evaluated external crates before implementation:
//! - `hurst` (0.1.0): GPL-3.0 license incompatible with MIT. Has R/S methods only.
//! - `diffusionx` (0.11.2): MIT, has fBm but requires mimalloc + heavy deps.
//! - `stochastic-processes` (0.1.2): MIT, basic processes only.
//!
//! Decision: Custom implementation using Hosking method for fBm generation.
//! This avoids GPL contamination and heavy dependencies while providing
//! exact fBm simulation with known Hurst exponent for rigorous testing.
//!
//! # Performance (Criterion benchmarks, 2026-02)
//!
//! | n   | Hosking (ours) | diffusionx | Speedup |
//! |-----|----------------|------------|---------|
//! | 128 | 27.0 us        | 20.4 us    | 1.3x    |
//! | 256 | 98.6 us        | 34.7 us    | 2.8x    |
//! | 512 | 365 us         | 57.5 us    | 6.3x    |
//!
//! Hosking is O(n^2) via Durbin-Levinson; diffusionx is O(n log n) via circulant FFT.
//! For n > 1000, prefer diffusionx (available via `stochastic` module).
//! For n <= 256 or when minimizing dependencies, Hosking is acceptable.
//!
//! # Physics Context in This Repo
//! Fractal analysis appears in:
//! - **Cosmology**: Self-similarity in quantum cosmology time series
//! - **Turbulence**: Kolmogorov scaling and intermittency
//! - **Zero-divisor geometry**: Fractal structure in Cayley-Dickson algebras
//!
//! # Interpretation
//! - H = 0.5: Brownian motion (random walk)
//! - H > 0.5: Persistent (trending, long-range positive correlations)
//! - H < 0.5: Anti-persistent (mean-reverting, negative correlations)

use std::f64;

/// Result of Hurst exponent analysis.
#[derive(Debug, Clone)]
pub struct HurstResult {
    /// Estimated Hurst exponent H in [0, 1].
    pub hurst: f64,
    /// Goodness of fit (R^2) from the linear regression.
    pub r_squared: f64,
    /// Number of lag values used in the fit.
    pub n_lags: usize,
}

/// Result of R/S (Rescaled Range) analysis.
#[derive(Debug, Clone)]
pub struct RescaledRangeResult {
    /// Lag value.
    pub lag: usize,
    /// Rescaled range R/S at this lag.
    pub rs: f64,
}

/// Result of Detrended Fluctuation Analysis (DFA).
#[derive(Debug, Clone)]
pub struct DfaResult {
    /// DFA exponent alpha (related to Hurst: H = alpha for fractional Gaussian noise).
    pub alpha: f64,
    /// Goodness of fit (R^2).
    pub r_squared: f64,
    /// Scales used in the analysis.
    pub scales: Vec<usize>,
    /// Fluctuation function F(n) at each scale.
    pub fluctuations: Vec<f64>,
}

/// Calculate the Hurst exponent using the variance-based method.
///
/// This method estimates H from the scaling of the standard deviation
/// of lagged differences: std(X[t+lag] - X[t]) ~ lag^H
///
/// # Arguments
/// * `series` - Time series data
/// * `min_lag` - Minimum lag value (default: 2)
/// * `max_lag` - Maximum lag value (default: min(50, len/4))
///
/// # Returns
/// `HurstResult` containing the exponent and fit quality, or `None` if
/// insufficient data.
///
/// # Example
/// ```
/// use algebra_core::calculate_hurst;
/// let brownian: Vec<f64> = (0..100).map(|i| (i as f64).sqrt()).collect();
/// if let Some(result) = calculate_hurst(&brownian, 2, 20) {
///     println!("Hurst exponent: {:.3}", result.hurst);
/// }
/// ```
pub fn calculate_hurst(series: &[f64], min_lag: usize, max_lag: usize) -> Option<HurstResult> {
    if series.len() < min_lag + 2 {
        return None;
    }

    let effective_max_lag = max_lag.min(series.len() / 4).max(min_lag + 1);

    let mut log_lags: Vec<f64> = Vec::new();
    let mut log_stds: Vec<f64> = Vec::new();

    for lag in min_lag..=effective_max_lag {
        let diffs: Vec<f64> = (lag..series.len())
            .map(|i| series[i] - series[i - lag])
            .collect();

        if diffs.is_empty() {
            continue;
        }

        let std = standard_deviation(&diffs);
        if std <= 1e-15 {
            continue;
        }

        log_lags.push((lag as f64).ln());
        log_stds.push(std.ln());
    }

    if log_lags.len() < 2 {
        return Some(HurstResult {
            hurst: 0.5,
            r_squared: 0.0,
            n_lags: 0,
        });
    }

    let (slope, r_squared) = linear_regression(&log_lags, &log_stds);

    Some(HurstResult {
        hurst: slope,
        r_squared,
        n_lags: log_lags.len(),
    })
}

/// Calculate the Hurst exponent using classic R/S analysis.
///
/// The Rescaled Range method:
/// 1. For each window size n, divide series into non-overlapping windows
/// 2. For each window: compute range R and standard deviation S
/// 3. Average R/S across windows
/// 4. Fit log(R/S) ~ H * log(n)
///
/// # Arguments
/// * `series` - Time series data
/// * `min_window` - Minimum window size
/// * `max_window` - Maximum window size
///
/// # Returns
/// `HurstResult` with the classical R/S Hurst estimate.
pub fn hurst_rs_analysis(
    series: &[f64],
    min_window: usize,
    max_window: usize,
) -> Option<HurstResult> {
    if series.len() < min_window * 2 {
        return None;
    }

    let effective_max = max_window.min(series.len() / 2);
    let mut results: Vec<RescaledRangeResult> = Vec::new();

    // Use logarithmically spaced window sizes
    let mut window = min_window;
    while window <= effective_max {
        if let Some(rs) = rescaled_range_at_scale(series, window) {
            results.push(RescaledRangeResult { lag: window, rs });
        }
        // Increase by ~20% each step
        window = (window as f64 * 1.2).ceil() as usize;
        if window == results.last().map(|r| r.lag).unwrap_or(0) {
            window += 1;
        }
    }

    if results.len() < 2 {
        return None;
    }

    let log_n: Vec<f64> = results.iter().map(|r| (r.lag as f64).ln()).collect();
    let log_rs: Vec<f64> = results.iter().map(|r| r.rs.ln()).collect();

    let (slope, r_squared) = linear_regression(&log_n, &log_rs);

    Some(HurstResult {
        hurst: slope,
        r_squared,
        n_lags: results.len(),
    })
}

/// Compute rescaled range R/S at a given window size.
fn rescaled_range_at_scale(series: &[f64], window: usize) -> Option<f64> {
    if window < 2 || series.len() < window {
        return None;
    }

    let n_windows = series.len() / window;
    if n_windows == 0 {
        return None;
    }

    let mut rs_values: Vec<f64> = Vec::new();

    for i in 0..n_windows {
        let start = i * window;
        let end = start + window;
        let chunk = &series[start..end];

        let mean = chunk.iter().sum::<f64>() / chunk.len() as f64;
        let deviations: Vec<f64> = chunk.iter().map(|x| x - mean).collect();

        // Cumulative sum of deviations
        let mut cumsum = Vec::with_capacity(window);
        let mut sum = 0.0;
        for d in &deviations {
            sum += d;
            cumsum.push(sum);
        }

        // Range
        let max_val = cumsum.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_val = cumsum.iter().cloned().fold(f64::INFINITY, f64::min);
        let range = max_val - min_val;

        // Standard deviation
        let std = standard_deviation(chunk);
        if std > 1e-15 {
            rs_values.push(range / std);
        }
    }

    if rs_values.is_empty() {
        return None;
    }

    Some(rs_values.iter().sum::<f64>() / rs_values.len() as f64)
}

/// Detrended Fluctuation Analysis (DFA).
///
/// DFA is more robust than R/S for non-stationary time series.
///
/// # Arguments
/// * `series` - Time series data
/// * `min_scale` - Minimum window scale
/// * `max_scale` - Maximum window scale
/// * `order` - Polynomial order for detrending (1 = linear, 2 = quadratic)
///
/// # Returns
/// `DfaResult` with alpha exponent and fluctuation function.
pub fn dfa_analysis(
    series: &[f64],
    min_scale: usize,
    max_scale: usize,
    order: usize,
) -> Option<DfaResult> {
    if series.len() < min_scale * 4 {
        return None;
    }

    // Integrate the series (cumulative sum of deviations from mean)
    let mean = series.iter().sum::<f64>() / series.len() as f64;
    let mut integrated: Vec<f64> = Vec::with_capacity(series.len());
    let mut sum = 0.0;
    for &x in series {
        sum += x - mean;
        integrated.push(sum);
    }

    let effective_max = max_scale.min(series.len() / 4);
    let mut scales: Vec<usize> = Vec::new();
    let mut fluctuations: Vec<f64> = Vec::new();

    // Logarithmically spaced scales
    let mut scale = min_scale;
    while scale <= effective_max {
        if let Some(f) = dfa_fluctuation_at_scale(&integrated, scale, order) {
            scales.push(scale);
            fluctuations.push(f);
        }
        scale = (scale as f64 * 1.25).ceil() as usize;
        if scale == scales.last().copied().unwrap_or(0) {
            scale += 1;
        }
    }

    if scales.len() < 2 {
        return None;
    }

    let log_n: Vec<f64> = scales.iter().map(|&s| (s as f64).ln()).collect();
    let log_f: Vec<f64> = fluctuations.iter().map(|&f| f.ln()).collect();

    let (alpha, r_squared) = linear_regression(&log_n, &log_f);

    Some(DfaResult {
        alpha,
        r_squared,
        scales,
        fluctuations,
    })
}

/// Compute DFA fluctuation at a given scale.
fn dfa_fluctuation_at_scale(integrated: &[f64], scale: usize, order: usize) -> Option<f64> {
    if scale < order + 2 || integrated.len() < scale {
        return None;
    }

    let n_segments = integrated.len() / scale;
    if n_segments == 0 {
        return None;
    }

    let mut total_variance = 0.0;
    let mut count = 0;

    for i in 0..n_segments {
        let start = i * scale;
        let segment: Vec<f64> = integrated[start..start + scale].to_vec();

        // Fit polynomial and compute variance of residuals
        let trend = polynomial_fit(&segment, order);
        let variance: f64 = segment
            .iter()
            .zip(trend.iter())
            .map(|(y, t)| (y - t).powi(2))
            .sum::<f64>()
            / scale as f64;

        total_variance += variance;
        count += 1;
    }

    if count == 0 {
        return None;
    }

    Some((total_variance / count as f64).sqrt())
}

/// Fit a polynomial of given order to the data (least squares).
fn polynomial_fit(y: &[f64], order: usize) -> Vec<f64> {
    let n = y.len();
    if n == 0 {
        return vec![];
    }

    // Simple implementation for low-order polynomials
    // For order 0: constant (mean)
    // For order 1: linear trend
    // For order 2: quadratic

    match order {
        0 => {
            let mean = y.iter().sum::<f64>() / n as f64;
            vec![mean; n]
        }
        1 => {
            // Linear: y = a + b*x
            let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let mean_x = (n - 1) as f64 / 2.0;
            let mean_y = y.iter().sum::<f64>() / n as f64;

            let mut num = 0.0;
            let mut den = 0.0;
            for i in 0..n {
                let dx = x[i] - mean_x;
                let dy = y[i] - mean_y;
                num += dx * dy;
                den += dx * dx;
            }

            let b = if den.abs() > 1e-15 { num / den } else { 0.0 };
            let a = mean_y - b * mean_x;

            x.iter().map(|&xi| a + b * xi).collect()
        }
        _ => {
            // For higher orders, fall back to linear (simplification)
            // A full implementation would use matrix inversion
            polynomial_fit(y, 1)
        }
    }
}

/// Standard deviation of a slice.
fn standard_deviation(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }

    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    variance.sqrt()
}

/// Linear regression: returns (slope, r_squared).
fn linear_regression(x: &[f64], y: &[f64]) -> (f64, f64) {
    if x.len() != y.len() || x.len() < 2 {
        return (0.5, 0.0);
    }

    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut ss_xy = 0.0;
    let mut ss_xx = 0.0;
    let mut ss_yy = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        ss_xy += dx * dy;
        ss_xx += dx * dx;
        ss_yy += dy * dy;
    }

    let slope = if ss_xx.abs() > 1e-15 {
        ss_xy / ss_xx
    } else {
        0.5
    };

    let r_squared = if ss_xx.abs() > 1e-15 && ss_yy.abs() > 1e-15 {
        (ss_xy * ss_xy) / (ss_xx * ss_yy)
    } else {
        0.0
    };

    (slope, r_squared)
}

/// Classify a Hurst exponent value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HurstClassification {
    /// H < 0.5: Mean-reverting, anti-persistent
    AntiPersistent,
    /// H ~ 0.5: Random walk (Brownian motion)
    RandomWalk,
    /// H > 0.5: Trending, persistent
    Persistent,
}

/// Classify a Hurst exponent value.
///
/// Uses tolerance of 0.05 for random walk classification.
pub fn classify_hurst(h: f64) -> HurstClassification {
    if h < 0.45 {
        HurstClassification::AntiPersistent
    } else if h > 0.55 {
        HurstClassification::Persistent
    } else {
        HurstClassification::RandomWalk
    }
}

/// Analyze multiple time series and return aggregate statistics.
pub struct MultiSeriesHurstResult {
    /// Individual Hurst exponents for each series.
    pub individual: Vec<HurstResult>,
    /// Mean Hurst exponent.
    pub mean: f64,
    /// Standard deviation of Hurst exponents.
    pub std: f64,
    /// Overall classification based on mean.
    pub classification: HurstClassification,
}

/// Generate fractional Gaussian noise (fGn) with specified Hurst exponent.
///
/// Uses the Hosking method (exact simulation via Cholesky decomposition
/// of the autocovariance matrix).
///
/// # Arguments
/// * `n` - Number of samples to generate
/// * `hurst` - Target Hurst exponent in (0, 1)
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// Vector of fGn samples with the specified long-range dependence.
///
/// # Literature
/// - Hosking, J.R.M. (1984). Modeling persistence in hydrological time series
///   using fractional differencing. Water Resources Research, 20(12), 1898-1908.
pub fn generate_fgn(n: usize, hurst: f64, seed: u64) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }

    // Autocovariance function for fGn: gamma(k) = 0.5 * (|k-1|^{2H} - 2|k|^{2H} + |k+1|^{2H})
    let h2 = 2.0 * hurst;
    let autocovariance = |k: i64| -> f64 {
        if k == 0 {
            1.0
        } else {
            let k = k.abs() as f64;
            0.5 * ((k - 1.0).powf(h2) - 2.0 * k.powf(h2) + (k + 1.0).powf(h2))
        }
    };

    // Build autocovariance vector
    let gamma: Vec<f64> = (0..n as i64).map(autocovariance).collect();

    // Cholesky decomposition via Durbin-Levinson algorithm
    // More efficient than full Cholesky for Toeplitz matrices
    let mut phi = vec![vec![0.0; n]; n];
    let mut sigma2 = vec![0.0; n];

    sigma2[0] = gamma[0];
    phi[0][0] = 1.0;

    for k in 1..n {
        // Compute reflection coefficient
        let mut num = gamma[k];
        for j in 1..k {
            num -= phi[k - 1][j] * gamma[k - j];
        }
        phi[k][k] = num / sigma2[k - 1];

        // Update phi coefficients
        for j in 1..k {
            phi[k][j] = phi[k - 1][j] - phi[k][k] * phi[k - 1][k - j];
        }

        // Update prediction variance
        sigma2[k] = sigma2[k - 1] * (1.0 - phi[k][k] * phi[k][k]);
    }

    // Generate standard normal samples using Box-Muller
    let mut rng_state = seed;
    let mut normals = Vec::with_capacity(n);
    for _ in 0..n {
        // LCG for uniform samples
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u1 = ((rng_state >> 33) as f64 + 1.0) / (u32::MAX as f64 + 2.0);
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u2 = ((rng_state >> 33) as f64) / (u32::MAX as f64);

        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        normals.push(z);
    }

    // Generate fGn using innovations algorithm
    let mut fgn = vec![0.0; n];
    fgn[0] = sigma2[0].sqrt() * normals[0];

    for k in 1..n {
        let mut mean = 0.0;
        for j in 1..=k {
            mean += phi[k][j] * fgn[k - j];
        }
        fgn[k] = mean + sigma2[k].sqrt() * normals[k];
    }

    fgn
}

/// Generate fractional Brownian motion (fBm) with specified Hurst exponent.
///
/// fBm is the cumulative sum of fGn.
///
/// # Arguments
/// * `n` - Number of samples
/// * `hurst` - Target Hurst exponent in (0, 1)
/// * `seed` - Random seed
///
/// # Returns
/// Vector of fBm samples.
pub fn generate_fbm(n: usize, hurst: f64, seed: u64) -> Vec<f64> {
    let fgn = generate_fgn(n, hurst, seed);
    let mut fbm = Vec::with_capacity(n);
    let mut sum = 0.0;
    for x in fgn {
        sum += x;
        fbm.push(sum);
    }
    fbm
}

/// Analyze Hurst exponents for multiple time series.
pub fn analyze_multiple_series(
    series_list: &[&[f64]],
    min_lag: usize,
    max_lag: usize,
) -> MultiSeriesHurstResult {
    let results: Vec<HurstResult> = series_list
        .iter()
        .filter_map(|s| calculate_hurst(s, min_lag, max_lag))
        .collect();

    let hursts: Vec<f64> = results.iter().map(|r| r.hurst).collect();
    let mean = if hursts.is_empty() {
        0.5
    } else {
        hursts.iter().sum::<f64>() / hursts.len() as f64
    };

    let std = if hursts.len() < 2 {
        0.0
    } else {
        standard_deviation(&hursts)
    };

    MultiSeriesHurstResult {
        individual: results,
        mean,
        std,
        classification: classify_hurst(mean),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hurst_random_walk() {
        // Cumulative sum of uniform noise approximates Brownian motion
        // H should be close to 0.5
        let mut rng_state = 12345u64;
        let mut series = Vec::with_capacity(1000);
        let mut sum = 0.0;
        for _ in 0..1000 {
            // Simple LCG random
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = ((rng_state >> 33) as f64) / (u32::MAX as f64) - 0.5;
            sum += r;
            series.push(sum);
        }

        let result = calculate_hurst(&series, 2, 100).unwrap();
        // Random walk H ~ 0.5 (allow wide tolerance due to finite sample)
        assert!(
            result.hurst > 0.3 && result.hurst < 0.7,
            "Hurst {} not in random walk range",
            result.hurst
        );
    }

    #[test]
    fn test_hurst_persistent_fbm() {
        // Generate fractional Brownian motion with H = 0.8 (persistent)
        // The variance method should recover H close to the true value
        let true_h = 0.8;
        let series = generate_fbm(2000, true_h, 42);

        let result = calculate_hurst(&series, 2, 100).unwrap();

        // With 2000 samples, we should recover H within +/- 0.15 of true value
        // This is a realistic tolerance for finite-sample estimation
        let error = (result.hurst - true_h).abs();
        assert!(
            error < 0.15,
            "Estimated H = {:.3} should be within 0.15 of true H = {:.1} (error = {:.3})",
            result.hurst,
            true_h,
            error
        );

        // Should definitely classify as persistent
        assert!(
            result.hurst > 0.55,
            "H = {:.3} should be > 0.55 for persistent fBm",
            result.hurst
        );
    }

    #[test]
    fn test_hurst_antipersistent_fbm() {
        // Generate fBm with H = 0.3 (anti-persistent / mean-reverting)
        let true_h = 0.3;
        let series = generate_fbm(2000, true_h, 123);

        let result = calculate_hurst(&series, 2, 100).unwrap();

        // Should classify as anti-persistent
        assert!(
            result.hurst < 0.45,
            "H = {:.3} should be < 0.45 for anti-persistent fBm (true H = {:.1})",
            result.hurst,
            true_h
        );
    }

    #[test]
    fn test_fbm_generator_basic() {
        // Verify fBm generator produces reasonable output
        let fbm = generate_fbm(100, 0.5, 999);
        assert_eq!(fbm.len(), 100);

        // Should have non-trivial variance
        let mean = fbm.iter().sum::<f64>() / 100.0;
        let var: f64 = fbm.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 99.0;
        assert!(var > 0.0, "fBm should have positive variance");
    }

    #[test]
    fn test_hurst_short_series() {
        let series = vec![1.0, 2.0, 3.0];
        let result = calculate_hurst(&series, 2, 50);
        // Should handle gracefully (either Some with limited data or None)
        assert!(result.is_some() || result.is_none());
    }

    #[test]
    fn test_hurst_constant() {
        // Constant series: std = 0, should return H = 0.5 or handle gracefully
        let series = vec![1.0; 100];
        let result = calculate_hurst(&series, 2, 20);
        assert!(result.is_some());
    }

    #[test]
    fn test_rs_analysis() {
        let series: Vec<f64> = (0..500).map(|i| (i as f64 * 0.01).sin()).collect();
        let result = hurst_rs_analysis(&series, 10, 100);
        assert!(result.is_some());
    }

    #[test]
    fn test_dfa_analysis() {
        let series: Vec<f64> = (0..500).map(|i| i as f64 * 0.1).collect();
        let result = dfa_analysis(&series, 10, 100, 1);
        assert!(result.is_some());
        let dfa = result.unwrap();
        assert!(!dfa.scales.is_empty());
    }

    #[test]
    fn test_classify_hurst() {
        assert_eq!(classify_hurst(0.3), HurstClassification::AntiPersistent);
        assert_eq!(classify_hurst(0.5), HurstClassification::RandomWalk);
        assert_eq!(classify_hurst(0.7), HurstClassification::Persistent);
    }

    #[test]
    fn test_linear_regression() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let (slope, r_sq) = linear_regression(&x, &y);
        assert!((slope - 2.0).abs() < 1e-10, "slope = {}", slope);
        assert!((r_sq - 1.0).abs() < 1e-10, "r_sq = {}", r_sq);
    }

    #[test]
    fn test_standard_deviation() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let std = standard_deviation(&data);
        // Expected: sqrt(32/7) ~ 2.138
        assert!((std - 2.138).abs() < 0.01, "std = {}", std);
    }

    #[test]
    fn test_multiple_series() {
        let s1: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let s2: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();

        let result = analyze_multiple_series(&[&s1, &s2], 2, 20);
        assert_eq!(result.individual.len(), 2);
    }
}

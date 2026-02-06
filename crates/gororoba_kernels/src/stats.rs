//! Statistical methodology for physics claims verification (R3).
//!
//! Implements:
//! - Frechet distance between normalized spectra (C-070)
//! - Bootstrap confidence intervals (C-074)
//! - Haar-distributed random unitaries for null tests (C-077)
//!
//! References:
//! - Alt & Godau (1995): Computing the Frechet distance
//! - Efron & Tibshirani (1993): An Introduction to the Bootstrap
//! - Stewart (1980): Efficient generation of random orthogonal matrices
//! - Mezzadri (2007): How to generate random matrices from compact groups

use nalgebra::DMatrix;
use num_complex::Complex64;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use statrs::distribution::{ContinuousCDF, Normal};

/// Discrete Frechet distance between two sequences.
///
/// Measures similarity between curves that accounts for location and ordering.
/// More appropriate than point-wise metrics for spectrum comparison.
pub fn frechet_distance(p: &[f64], q: &[f64]) -> f64 {
    let n = p.len();
    let m = q.len();

    if n == 0 || m == 0 {
        return f64::INFINITY;
    }

    // DP table for coupling distance
    let mut ca = vec![vec![f64::NEG_INFINITY; m]; n];

    // Distance function
    let d = |i: usize, j: usize| (p[i] - q[j]).abs();

    // Recursive function with memoization via iteration
    ca[0][0] = d(0, 0);

    // First row
    for j in 1..m {
        ca[0][j] = ca[0][j - 1].max(d(0, j));
    }

    // First column
    for i in 1..n {
        ca[i][0] = ca[i - 1][0].max(d(i, 0));
    }

    // Fill DP table
    for i in 1..n {
        for j in 1..m {
            let min_prev = ca[i - 1][j].min(ca[i][j - 1]).min(ca[i - 1][j - 1]);
            ca[i][j] = min_prev.max(d(i, j));
        }
    }

    ca[n - 1][m - 1]
}

/// Normalize a spectrum to [0, 1] range for comparison.
pub fn normalize_spectrum(spectrum: &[f64]) -> Vec<f64> {
    if spectrum.is_empty() {
        return vec![];
    }

    let min = spectrum.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = spectrum.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    if (max - min).abs() < 1e-15 {
        return vec![0.5; spectrum.len()];
    }

    spectrum.iter().map(|x| (x - min) / (max - min)).collect()
}

/// Generate random monotonic spectrum for null test.
pub fn random_monotonic_spectrum(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut spectrum: Vec<f64> = (0..n).map(|_| rng.gen::<f64>()).collect();
    spectrum.sort_by(|a, b| a.partial_cmp(b).unwrap());
    spectrum
}

/// Frechet distance null test result.
#[derive(Debug, Clone)]
pub struct FrechetNullTestResult {
    pub observed_distance: f64,
    pub null_distances: Vec<f64>,
    pub p_value: f64,
    pub significant_at_005: bool,
    pub mean_null: f64,
    pub std_null: f64,
}

/// Null test: compare observed spectrum distance against random monotonic spectra.
///
/// For C-070: If p > 0.05, claim should be closed as Refuted.
pub fn frechet_null_test(
    observed: &[f64],
    reference: &[f64],
    n_permutations: usize,
    seed: u64,
) -> FrechetNullTestResult {
    let obs_norm = normalize_spectrum(observed);
    let ref_norm = normalize_spectrum(reference);
    let observed_distance = frechet_distance(&obs_norm, &ref_norm);

    // Generate null distribution
    let mut null_distances = Vec::with_capacity(n_permutations);
    for i in 0..n_permutations {
        let random_spec = random_monotonic_spectrum(observed.len(), seed + i as u64);
        let random_norm = normalize_spectrum(&random_spec);
        null_distances.push(frechet_distance(&random_norm, &ref_norm));
    }

    // Compute p-value (proportion of null >= observed)
    let n_extreme = null_distances
        .iter()
        .filter(|&&d| d <= observed_distance)
        .count();
    let p_value = n_extreme as f64 / n_permutations as f64;

    // Statistics
    let mean_null: f64 = null_distances.iter().sum::<f64>() / n_permutations as f64;
    let var_null: f64 = null_distances
        .iter()
        .map(|d| (d - mean_null).powi(2))
        .sum::<f64>()
        / n_permutations as f64;
    let std_null = var_null.sqrt();

    FrechetNullTestResult {
        observed_distance,
        null_distances,
        p_value,
        significant_at_005: p_value < 0.05,
        mean_null,
        std_null,
    }
}

/// Bootstrap confidence interval result.
#[derive(Debug, Clone)]
pub struct BootstrapCIResult {
    pub point_estimate: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub ci_width: f64,
    pub ci_width_relative: f64, // width / |point_estimate|
    pub standard_error: f64,
    pub n_bootstrap: usize,
}

/// Compute bootstrap confidence interval for a statistic.
///
/// Generic over any function that computes a statistic from a sample.
pub fn bootstrap_ci<F>(
    data: &[f64],
    statistic_fn: F,
    n_bootstrap: usize,
    confidence: f64,
    seed: u64,
) -> BootstrapCIResult
where
    F: Fn(&[f64]) -> f64,
{
    let point_estimate = statistic_fn(data);
    let n = data.len();

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut bootstrap_stats = Vec::with_capacity(n_bootstrap);

    for _ in 0..n_bootstrap {
        // Resample with replacement
        let resample: Vec<f64> = (0..n).map(|_| data[rng.gen_range(0..n)]).collect();
        bootstrap_stats.push(statistic_fn(&resample));
    }

    bootstrap_stats.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Percentile method
    let alpha = 1.0 - confidence;
    let lower_idx = ((alpha / 2.0) * n_bootstrap as f64) as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * n_bootstrap as f64) as usize;

    let ci_lower = bootstrap_stats[lower_idx];
    let ci_upper = bootstrap_stats[upper_idx.min(n_bootstrap - 1)];
    let ci_width = ci_upper - ci_lower;

    // Standard error
    let mean: f64 = bootstrap_stats.iter().sum::<f64>() / n_bootstrap as f64;
    let var: f64 = bootstrap_stats
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>()
        / n_bootstrap as f64;
    let standard_error = var.sqrt();

    let ci_width_relative = if point_estimate.abs() > 1e-15 {
        ci_width / point_estimate.abs()
    } else {
        f64::INFINITY
    };

    BootstrapCIResult {
        point_estimate,
        ci_lower,
        ci_upper,
        ci_width,
        ci_width_relative,
        standard_error,
        n_bootstrap,
    }
}

/// Generate Haar-distributed random unitary matrix.
///
/// Uses QR decomposition of random Gaussian matrix (Stewart 1980, Mezzadri 2007).
/// The resulting matrix is uniform with respect to Haar measure on U(n).
pub fn haar_random_unitary(dim: usize, seed: u64) -> DMatrix<Complex64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Generate random complex Gaussian matrix
    let mut z = DMatrix::zeros(dim, dim);
    for i in 0..dim {
        for j in 0..dim {
            let re = normal.inverse_cdf(rng.gen());
            let im = normal.inverse_cdf(rng.gen());
            z[(i, j)] = Complex64::new(re, im);
        }
    }

    // QR decomposition via Gram-Schmidt
    let q = gram_schmidt_qr(&z);

    // Ensure Haar measure by correcting phase (Mezzadri's trick)
    let mut result = q.clone();
    for j in 0..dim {
        let diag = q[(j, j)];
        if diag.norm() > 1e-15 {
            let phase = diag / Complex64::new(diag.norm(), 0.0);
            for i in 0..dim {
                result[(i, j)] /= phase;
            }
        }
    }

    result
}

/// QR decomposition via modified Gram-Schmidt (for complex matrices).
fn gram_schmidt_qr(a: &DMatrix<Complex64>) -> DMatrix<Complex64> {
    let n = a.nrows();
    let mut q = a.clone();

    for j in 0..n {
        // Normalize column j
        let mut norm_sq = 0.0;
        for i in 0..n {
            norm_sq += q[(i, j)].norm_sqr();
        }
        let norm = norm_sq.sqrt();

        if norm > 1e-15 {
            for i in 0..n {
                q[(i, j)] /= norm;
            }
        }

        // Orthogonalize remaining columns against j
        for k in (j + 1)..n {
            let mut proj = Complex64::new(0.0, 0.0);
            for i in 0..n {
                proj += q[(i, j)].conj() * q[(i, k)];
            }
            for i in 0..n {
                q[(i, k)] = q[(i, k)] - proj * q[(i, j)];
            }
        }
    }

    q
}

/// Frobenius distance between two complex matrices.
pub fn frobenius_distance(a: &DMatrix<Complex64>, b: &DMatrix<Complex64>) -> f64 {
    let diff = a - b;
    let mut sum = 0.0;
    for i in 0..diff.nrows() {
        for j in 0..diff.ncols() {
            sum += diff[(i, j)].norm_sqr();
        }
    }
    sum.sqrt()
}

/// Haar-measure null test for matrix comparison (C-077).
///
/// Tests whether observed matrix is closer to a reference than random unitaries.
#[derive(Debug, Clone)]
pub struct HaarNullTestResult {
    pub observed_distance: f64,
    pub null_distances: Vec<f64>,
    pub p_value: f64,
    pub significant_at_005: bool,
    pub mean_null: f64,
    pub std_null: f64,
}

/// Null test comparing a matrix to random Haar-distributed unitaries.
pub fn haar_null_test(
    observed: &DMatrix<Complex64>,
    reference: &DMatrix<Complex64>,
    n_permutations: usize,
    seed: u64,
) -> HaarNullTestResult {
    let dim = observed.nrows();
    let observed_distance = frobenius_distance(observed, reference);

    // Generate null distribution
    let mut null_distances = Vec::with_capacity(n_permutations);
    for i in 0..n_permutations {
        let random_unitary = haar_random_unitary(dim, seed + i as u64);
        null_distances.push(frobenius_distance(&random_unitary, reference));
    }

    // Compute p-value (proportion of null <= observed)
    let n_extreme = null_distances
        .iter()
        .filter(|&&d| d <= observed_distance)
        .count();
    let p_value = n_extreme as f64 / n_permutations as f64;

    // Statistics
    let mean_null: f64 = null_distances.iter().sum::<f64>() / n_permutations as f64;
    let var_null: f64 = null_distances
        .iter()
        .map(|d| (d - mean_null).powi(2))
        .sum::<f64>()
        / n_permutations as f64;
    let std_null = var_null.sqrt();

    HaarNullTestResult {
        observed_distance,
        null_distances,
        p_value,
        significant_at_005: p_value < 0.05,
        mean_null,
        std_null,
    }
}

/// PMNS matrix (Pontecorvo-Maki-Nakagawa-Sakata) for neutrino mixing.
///
/// Using latest PDG values (2024).
pub fn pmns_matrix() -> DMatrix<Complex64> {
    // Mixing angles from PDG 2024
    let theta_12 = 33.41_f64.to_radians();
    let theta_23 = 42.2_f64.to_radians();
    let theta_13 = 8.54_f64.to_radians();
    let delta_cp = 232.0_f64.to_radians();

    let c12 = theta_12.cos();
    let s12 = theta_12.sin();
    let c23 = theta_23.cos();
    let s23 = theta_23.sin();
    let c13 = theta_13.cos();
    let s13 = theta_13.sin();

    let exp_delta = Complex64::new(delta_cp.cos(), -delta_cp.sin());

    // Standard parametrization
    DMatrix::from_row_slice(
        3,
        3,
        &[
            Complex64::new(c12 * c13, 0.0),
            Complex64::new(s12 * c13, 0.0),
            s13 * exp_delta.conj(),
            -s12 * c23 - c12 * s23 * s13 * exp_delta,
            c12 * c23 - s12 * s23 * s13 * exp_delta,
            Complex64::new(s23 * c13, 0.0),
            s12 * s23 - c12 * c23 * s13 * exp_delta,
            -c12 * s23 - s12 * c23 * s13 * exp_delta,
            Complex64::new(c23 * c13, 0.0),
        ],
    )
}

/// Result of PMNS comparison test (C-077).
#[derive(Debug, Clone)]
pub struct PmnsComparisonResult {
    pub frobenius_distance: f64,
    pub haar_null_test: HaarNullTestResult,
    pub closer_than_random: bool,
}

/// Test if a predicted mixing matrix is closer to PMNS than random unitaries.
pub fn test_pmns_prediction(
    predicted: &DMatrix<Complex64>,
    n_permutations: usize,
    seed: u64,
) -> PmnsComparisonResult {
    let pmns = pmns_matrix();
    let frobenius_dist = frobenius_distance(predicted, &pmns);
    let null_test = haar_null_test(predicted, &pmns, n_permutations, seed);

    PmnsComparisonResult {
        frobenius_distance: frobenius_dist,
        haar_null_test: null_test.clone(),
        closer_than_random: null_test.p_value < 0.05,
    }
}

/// Power-law fit result with bootstrap CI.
#[derive(Debug, Clone)]
pub struct PowerLawFitResult {
    pub amplitude: BootstrapCIResult,
    pub exponent: BootstrapCIResult,
    pub r_squared: f64,
    pub uncertain: bool, // CI width > 50% of point estimate
}

/// Fit A * x^alpha to data with bootstrap confidence intervals.
pub fn fit_power_law_with_ci(
    x: &[f64],
    y: &[f64],
    n_bootstrap: usize,
    seed: u64,
) -> PowerLawFitResult {
    // Transform to linear: log(y) = log(A) + alpha * log(x)
    let log_x: Vec<f64> = x.iter().map(|xi| xi.ln()).collect();
    let log_y: Vec<f64> = y.iter().map(|yi| yi.ln()).collect();

    // Combine for resampling
    let pairs: Vec<f64> = log_x
        .iter()
        .zip(log_y.iter())
        .flat_map(|(&lx, &ly)| vec![lx, ly])
        .collect();

    // Function to compute exponent from paired data
    let exponent_fn = |data: &[f64]| {
        let n = data.len() / 2;
        let lx: Vec<f64> = (0..n).map(|i| data[2 * i]).collect();
        let ly: Vec<f64> = (0..n).map(|i| data[2 * i + 1]).collect();

        let sum_x: f64 = lx.iter().sum();
        let sum_y: f64 = ly.iter().sum();
        let sum_xx: f64 = lx.iter().map(|x| x * x).sum();
        let sum_xy: f64 = lx.iter().zip(ly.iter()).map(|(x, y)| x * y).sum();

        let n_f = n as f64;
        (n_f * sum_xy - sum_x * sum_y) / (n_f * sum_xx - sum_x * sum_x)
    };

    // Function to compute log(A) from paired data
    let log_amplitude_fn = |data: &[f64]| {
        let n = data.len() / 2;
        let lx: Vec<f64> = (0..n).map(|i| data[2 * i]).collect();
        let ly: Vec<f64> = (0..n).map(|i| data[2 * i + 1]).collect();

        let sum_x: f64 = lx.iter().sum();
        let sum_y: f64 = ly.iter().sum();
        let sum_xx: f64 = lx.iter().map(|x| x * x).sum();
        let sum_xy: f64 = lx.iter().zip(ly.iter()).map(|(x, y)| x * y).sum();

        let n_f = n as f64;
        let slope = (n_f * sum_xy - sum_x * sum_y) / (n_f * sum_xx - sum_x * sum_x);
        (sum_y - slope * sum_x) / n_f
    };

    let exponent_ci = bootstrap_ci(&pairs, exponent_fn, n_bootstrap, 0.95, seed);

    let log_amp_ci = bootstrap_ci(&pairs, log_amplitude_fn, n_bootstrap, 0.95, seed + 1);
    let amplitude_ci = BootstrapCIResult {
        point_estimate: log_amp_ci.point_estimate.exp(),
        ci_lower: log_amp_ci.ci_lower.exp(),
        ci_upper: log_amp_ci.ci_upper.exp(),
        ci_width: log_amp_ci.ci_upper.exp() - log_amp_ci.ci_lower.exp(),
        ci_width_relative: (log_amp_ci.ci_upper.exp() - log_amp_ci.ci_lower.exp())
            / log_amp_ci.point_estimate.exp().abs(),
        standard_error: log_amp_ci.standard_error * log_amp_ci.point_estimate.exp(),
        n_bootstrap,
    };

    // R-squared
    let predicted: Vec<f64> = log_x
        .iter()
        .map(|lx| log_amp_ci.point_estimate + exponent_ci.point_estimate * lx)
        .collect();
    let mean_y: f64 = log_y.iter().sum::<f64>() / log_y.len() as f64;
    let ss_tot: f64 = log_y.iter().map(|y| (y - mean_y).powi(2)).sum();
    let ss_res: f64 = log_y
        .iter()
        .zip(predicted.iter())
        .map(|(y, p)| (y - p).powi(2))
        .sum();
    let r_squared = 1.0 - ss_res / ss_tot;

    // Uncertain if CI width > 50% of estimate
    let uncertain =
        exponent_ci.ci_width_relative > 0.5 || amplitude_ci.ci_width_relative > 0.5;

    PowerLawFitResult {
        amplitude: amplitude_ci,
        exponent: exponent_ci,
        r_squared,
        uncertain,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frechet_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        assert!((frechet_distance(&a, &a) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_frechet_different() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.5, 2.5, 3.5];
        let d = frechet_distance(&a, &b);
        assert!(d > 0.0);
        assert!(d <= 0.5);
    }

    #[test]
    fn test_frechet_symmetric() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.1, 2.2, 2.9, 4.1];
        let d_ab = frechet_distance(&a, &b);
        let d_ba = frechet_distance(&b, &a);
        assert!((d_ab - d_ba).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_spectrum() {
        let s = vec![10.0, 20.0, 30.0, 40.0];
        let n = normalize_spectrum(&s);
        assert!((n[0] - 0.0).abs() < 1e-10);
        assert!((n[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_frechet_null_test_random() {
        let observed = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let reference = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = frechet_null_test(&observed, &reference, 100, 42);

        // Identical should have very low distance
        assert!(result.observed_distance < 0.01);
    }

    #[test]
    fn test_bootstrap_ci_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean_fn = |d: &[f64]| d.iter().sum::<f64>() / d.len() as f64;

        let result = bootstrap_ci(&data, mean_fn, 1000, 0.95, 42);

        // Point estimate should be 3.0
        assert!((result.point_estimate - 3.0).abs() < 1e-10);
        // CI should contain the true mean
        assert!(result.ci_lower <= 3.0);
        assert!(result.ci_upper >= 3.0);
    }

    #[test]
    fn test_haar_unitary_is_unitary() {
        let u = haar_random_unitary(4, 42);

        // Check U * U^dagger = I
        let mut u_dagger = u.clone();
        for i in 0..4 {
            for j in 0..4 {
                u_dagger[(i, j)] = u[(j, i)].conj();
            }
        }

        let product = &u * &u_dagger;

        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                assert!(
                    (product[(i, j)] - expected).norm() < 1e-10,
                    "U * U^dag != I at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_haar_null_test_random_not_close() {
        // Generate two random unitaries - they should not be particularly close
        let u1 = haar_random_unitary(3, 42);
        let u2 = haar_random_unitary(3, 123);

        let result = haar_null_test(&u1, &u2, 100, 456);

        // Just verify the test runs and produces valid output
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert!(result.mean_null > 0.0);
        assert!(result.std_null >= 0.0);
    }

    #[test]
    fn test_pmns_is_unitary() {
        let pmns = pmns_matrix();

        // Check unitarity: U * U^dagger = I
        let mut pmns_dagger = pmns.clone();
        for i in 0..3 {
            for j in 0..3 {
                pmns_dagger[(i, j)] = pmns[(j, i)].conj();
            }
        }

        let product = &pmns * &pmns_dagger;

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                assert!(
                    (product[(i, j)] - expected).norm() < 1e-10,
                    "PMNS not unitary at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_pmns_frobenius_from_random() {
        let pmns = pmns_matrix();
        let random_u = haar_random_unitary(3, 42);

        let dist = frobenius_distance(&pmns, &random_u);

        // PMNS is a specific matrix, random should be at some distance
        assert!(dist > 0.1);
        assert!(dist < 4.0); // Maximum possible ~sqrt(18) for 3x3 unitary
    }

    #[test]
    fn test_power_law_fit() {
        // Generate perfect power law: y = 2 * x^0.5
        let x: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|xi| 2.0 * xi.powf(0.5)).collect();

        let result = fit_power_law_with_ci(&x, &y, 100, 42);

        // Should recover amplitude ~2 and exponent ~0.5
        assert!((result.amplitude.point_estimate - 2.0).abs() < 0.1);
        assert!((result.exponent.point_estimate - 0.5).abs() < 0.05);
        assert!(result.r_squared > 0.99);
        // CI should contain the true values
        assert!(result.exponent.ci_lower <= 0.5);
        assert!(result.exponent.ci_upper >= 0.5);
    }

    #[test]
    fn test_power_law_noisy() {
        // Generate noisy power law
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|xi| 3.0 * xi.powf(-1.5) * (1.0 + 0.1 * (rng.gen::<f64>() - 0.5)))
            .collect();

        let result = fit_power_law_with_ci(&x, &y, 200, 42);

        // Should approximately recover the parameters
        assert!((result.exponent.point_estimate - (-1.5)).abs() < 0.3);
    }
}

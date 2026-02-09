//! Statistical methodology for physics claims verification (R3).
//!
//! Implements:
//! - Frechet distance between normalized spectra (C-070)
//! - Bootstrap confidence intervals (C-074)
//! - Haar-distributed random unitaries for null tests (C-077)
//! - Energy Distance (ED) two-sample test with efficient permutations
//! - Maximum Mean Discrepancy (MMD) with efficient permutations
//! - Claims gates for CI integration (pass/fail verdicts)
//! - Study type classification (observational vs randomized/experimental)
//!
//! References:
//! - Alt & Godau (1995): Computing the Frechet distance
//! - Efron & Tibshirani (1993): An Introduction to the Bootstrap
//! - Stewart (1980): Efficient generation of random orthogonal matrices
//! - Mezzadri (2007): How to generate random matrices from compact groups
//! - Chaibub Neto & Prisco (2024): Efficient permutation tests for ED/MMD
//! - Szekely & Rizzo (2004): Energy distance and testing for distances
//! - Roy (2003): "Discovery and the Scientific Method" (Annals of Statistics)

pub mod claims_gates;
pub mod dip;
pub mod hypergraph;
pub mod ultrametric;

pub use claims_gates::{
    bootstrap_ci_gate, frechet_gate, inverse_pvalue_gate, metric_gate, pvalue_gate,
    run_bootstrap_ci_gate, run_frechet_gate, run_two_sample_gate, two_sample_gate, Evidence,
    GateRegistry, GateResult, Verdict,
};

// C-074 fitting types are defined inline below (AssociatorGrowthFitResult,
// fit_associator_growth_law, c074_decision_rule) - no re-export needed.

use nalgebra::DMatrix;
use num_complex::Complex64;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use statrs::distribution::{ContinuousCDF, Normal};

// =============================================================================
// Study Type Classification (Task #128: Tag studies as observational vs randomized)
// =============================================================================

/// Classification of study design for claims methodology.
///
/// Per Roy (2003), distinguishing observational from experimental studies
/// affects the strength of causal inference and appropriate statistical tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StudyType {
    /// Randomized controlled study with manipulation of independent variable.
    /// Supports causal inference if properly controlled.
    Experimental,

    /// Observational study without manipulation (e.g., surveys, archival data).
    /// Supports correlational inference only; confounders must be considered.
    Observational,

    /// Computational/mathematical study based on algebraic or numerical analysis.
    /// No empirical data; results are mathematical facts if derivation is correct.
    Computational,

    /// Simulation study generating synthetic data from a model.
    /// Supports inference about model behavior, not necessarily physical reality.
    Simulation,

    /// Meta-analysis combining results from multiple studies.
    /// Strength depends on constituent study quality.
    MetaAnalysis,
}

/// Metadata for a claims-relevant study.
#[derive(Debug, Clone)]
pub struct StudyMetadata {
    /// Study identifier (e.g., claim ID or paper reference)
    pub id: String,
    /// Study type classification
    pub study_type: StudyType,
    /// Sample size (N) or equivalent metric
    pub sample_size: usize,
    /// Whether the study was pre-registered
    pub pre_registered: bool,
    /// Effect size (if applicable, e.g., Cohen's d)
    pub effect_size: Option<f64>,
    /// Power analysis performed (for experimental studies)
    pub power_analysis: bool,
    /// Blinding used (for experimental studies)
    pub blinded: bool,
    /// Notes on methodology
    pub notes: String,
}

impl StudyMetadata {
    /// Create new study metadata with required fields.
    pub fn new(id: impl Into<String>, study_type: StudyType, sample_size: usize) -> Self {
        Self {
            id: id.into(),
            study_type,
            sample_size,
            pre_registered: false,
            effect_size: None,
            power_analysis: false,
            blinded: false,
            notes: String::new(),
        }
    }

    /// Builder pattern: mark as pre-registered
    pub fn with_preregistration(mut self) -> Self {
        self.pre_registered = true;
        self
    }

    /// Builder pattern: add effect size
    pub fn with_effect_size(mut self, d: f64) -> Self {
        self.effect_size = Some(d);
        self
    }

    /// Builder pattern: add notes
    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = notes.into();
        self
    }

    /// Assess study quality score (0.0 to 1.0).
    ///
    /// Higher scores indicate stronger methodology:
    /// - Pre-registration: +0.2
    /// - Experimental (vs observational): +0.2
    /// - Large sample (N >= 100): +0.2
    /// - Effect size reported: +0.2
    /// - Power analysis: +0.1
    /// - Blinding: +0.1
    pub fn quality_score(&self) -> f64 {
        let mut score: f64 = 0.0;

        if self.pre_registered {
            score += 0.2;
        }

        match self.study_type {
            StudyType::Experimental => score += 0.2,
            StudyType::Computational => score += 0.15, // Mathematical proofs are strong
            StudyType::Simulation => score += 0.1,
            StudyType::MetaAnalysis => score += 0.15,
            StudyType::Observational => {} // No bonus for observational
        }

        if self.sample_size >= 100 {
            score += 0.2;
        } else if self.sample_size >= 30 {
            score += 0.1;
        }

        if self.effect_size.is_some() {
            score += 0.2;
        }

        if self.power_analysis {
            score += 0.1;
        }

        if self.blinded {
            score += 0.1;
        }

        score.min(1.0)
    }
}

// =============================================================================
// End Study Type Classification
// =============================================================================

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
                let qij = q[(i, j)];
                q[(i, k)] -= proj * qij;
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
    let uncertain = exponent_ci.ci_width_relative > 0.5 || amplitude_ci.ci_width_relative > 0.5;

    PowerLawFitResult {
        amplitude: amplitude_ci,
        exponent: exponent_ci,
        r_squared,
        uncertain,
    }
}

// ============================================================================
// Associator Growth Law Fit (C-074)
// Form: y = A_inf * (1 - B * d^alpha)
// ============================================================================

/// Result of associator growth law fit with bootstrap CI.
///
/// Models the expected squared associator norm as:
/// E[||A(a,b,c)||^2] = A_inf * (1 - B * d^alpha)
///
/// where d is the CD dimension and A_inf, B, alpha are fitted parameters.
#[derive(Debug, Clone)]
pub struct AssociatorGrowthFitResult {
    /// Asymptotic value A_inf (expected: ~2.0 for unit vectors)
    pub a_inf: BootstrapCIResult,
    /// Coefficient B in the correction term
    pub b_coeff: BootstrapCIResult,
    /// Exponent alpha (expected: ~-1.8 from Python fit)
    pub alpha: BootstrapCIResult,
    /// R-squared of the fit
    pub r_squared: f64,
    /// Whether CI widths indicate high uncertainty (>50% of point estimate)
    pub uncertain: bool,
}

/// Fit the associator growth law y = A_inf * (1 - B * d^alpha) with bootstrap CI.
///
/// The C-074 claim states: <||A(a,b,c)||^2> = 2.00 * (1 - 14.6 * d^{-1.80})
///
/// This function fits that form to data and provides uncertainty quantification.
/// Decision rule: if alpha 95% CI excludes -1.80, the claim needs revision.
pub fn fit_associator_growth_law(
    dimensions: &[f64],
    mean_sq_norms: &[f64],
    n_bootstrap: usize,
    seed: u64,
) -> AssociatorGrowthFitResult {
    assert_eq!(dimensions.len(), mean_sq_norms.len());
    let n = dimensions.len();

    // Grid search over alpha, then fit A_inf and B via linear regression
    // Transform: y / A_inf = 1 - B * d^alpha  =>  B * d^alpha = 1 - y/A_inf
    // For fixed alpha: log(1 - y/A_inf) = log(B) + alpha * log(d)

    // First pass: estimate parameters via grid search
    let (best_a_inf, best_b, best_alpha) = grid_search_growth_law(dimensions, mean_sq_norms);

    // Combine data for resampling: [d0, y0, d1, y1, ...]
    let pairs: Vec<f64> = dimensions
        .iter()
        .zip(mean_sq_norms.iter())
        .flat_map(|(&d, &y)| vec![d, y])
        .collect();

    // Bootstrap functions for each parameter
    let a_inf_fn = |data: &[f64]| {
        let n = data.len() / 2;
        let dims: Vec<f64> = (0..n).map(|i| data[2 * i]).collect();
        let ys: Vec<f64> = (0..n).map(|i| data[2 * i + 1]).collect();
        let (a, _, _) = grid_search_growth_law(&dims, &ys);
        a
    };

    let b_fn = |data: &[f64]| {
        let n = data.len() / 2;
        let dims: Vec<f64> = (0..n).map(|i| data[2 * i]).collect();
        let ys: Vec<f64> = (0..n).map(|i| data[2 * i + 1]).collect();
        let (_, b, _) = grid_search_growth_law(&dims, &ys);
        b
    };

    let alpha_fn = |data: &[f64]| {
        let n = data.len() / 2;
        let dims: Vec<f64> = (0..n).map(|i| data[2 * i]).collect();
        let ys: Vec<f64> = (0..n).map(|i| data[2 * i + 1]).collect();
        let (_, _, alpha) = grid_search_growth_law(&dims, &ys);
        alpha
    };

    let a_inf_ci = bootstrap_ci(&pairs, a_inf_fn, n_bootstrap, 0.95, seed);
    let b_ci = bootstrap_ci(&pairs, b_fn, n_bootstrap, 0.95, seed + 1);
    let alpha_ci = bootstrap_ci(&pairs, alpha_fn, n_bootstrap, 0.95, seed + 2);

    // Compute R-squared with best-fit parameters
    let predicted: Vec<f64> = dimensions
        .iter()
        .map(|&d| best_a_inf * (1.0 - best_b * d.powf(best_alpha)))
        .collect();
    let mean_y: f64 = mean_sq_norms.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = mean_sq_norms.iter().map(|&y| (y - mean_y).powi(2)).sum();
    let ss_res: f64 = mean_sq_norms
        .iter()
        .zip(predicted.iter())
        .map(|(&y, &p)| (y - p).powi(2))
        .sum();
    let r_squared = 1.0 - ss_res / ss_tot;

    // Uncertain if any CI width > 50% of point estimate
    let uncertain = a_inf_ci.ci_width_relative > 0.5
        || b_ci.ci_width_relative > 0.5
        || alpha_ci.ci_width_relative > 0.5;

    AssociatorGrowthFitResult {
        a_inf: a_inf_ci,
        b_coeff: b_ci,
        alpha: alpha_ci,
        r_squared,
        uncertain,
    }
}

/// Grid search to find best A_inf, B, alpha for y = A_inf * (1 - B * d^alpha).
fn grid_search_growth_law(dims: &[f64], ys: &[f64]) -> (f64, f64, f64) {
    let mut best_sse = f64::INFINITY;
    let mut best_params = (2.0, 14.6, -1.8);

    // Grid search over alpha in [-3.0, -0.5]
    for alpha_idx in 0..51 {
        let alpha = -3.0 + (alpha_idx as f64) * 0.05;

        // For fixed alpha, estimate A_inf from asymptotic behavior
        // At large d, y -> A_inf, so use max y as initial guess
        let a_inf_guess = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Grid search over A_inf around the guess
        for a_mult_idx in 0..21 {
            let a_inf = a_inf_guess * (0.9 + 0.01 * a_mult_idx as f64);
            if a_inf <= 0.0 {
                continue;
            }

            // Given A_inf and alpha, estimate B via least squares
            // y/A_inf = 1 - B * d^alpha  =>  B = (1 - y/A_inf) / d^alpha
            let mut sum_num = 0.0;
            let mut sum_den = 0.0;
            for (&d, &y) in dims.iter().zip(ys.iter()) {
                let d_alpha = d.powf(alpha);
                let target = 1.0 - y / a_inf;
                sum_num += target * d_alpha;
                sum_den += d_alpha * d_alpha;
            }

            if sum_den.abs() < 1e-15 {
                continue;
            }
            let b = sum_num / sum_den;

            // Compute SSE
            let sse: f64 = dims
                .iter()
                .zip(ys.iter())
                .map(|(&d, &y)| {
                    let pred = a_inf * (1.0 - b * d.powf(alpha));
                    (y - pred).powi(2)
                })
                .sum();

            if sse < best_sse {
                best_sse = sse;
                best_params = (a_inf, b, alpha);
            }
        }
    }

    best_params
}

/// C-074 decision rule: check if fitted parameters match the claim.
///
/// Claim: A_inf ~ 2.0, B ~ 14.6, alpha ~ -1.80
///
/// Returns (matches, reason) where matches is true if all parameters
/// are within acceptable tolerance of the claimed values.
pub fn c074_decision_rule(fit: &AssociatorGrowthFitResult) -> (bool, String) {
    let mut reasons = Vec::new();

    // Check A_inf ~ 2.0 (within 10%)
    if (fit.a_inf.point_estimate - 2.0).abs() > 0.2 {
        reasons.push(format!(
            "A_inf={:.3} not near 2.0",
            fit.a_inf.point_estimate
        ));
    }

    // Check if alpha 95% CI contains -1.80
    if fit.alpha.ci_lower > -1.80 || fit.alpha.ci_upper < -1.80 {
        reasons.push(format!(
            "alpha 95% CI [{:.2}, {:.2}] excludes -1.80",
            fit.alpha.ci_lower, fit.alpha.ci_upper
        ));
    }

    // Check uncertainty
    if fit.uncertain {
        reasons.push("High uncertainty (CI width > 50%)".to_string());
    }

    // Check R-squared
    if fit.r_squared < 0.95 {
        reasons.push(format!("R^2={:.3} < 0.95", fit.r_squared));
    }

    if reasons.is_empty() {
        (
            true,
            "Claim supported: parameters match within tolerance".to_string(),
        )
    } else {
        (false, reasons.join("; "))
    }
}

// ============================================================================
// Energy Distance and MMD with Efficient Permutation Testing
// (Chaibub Neto & Prisco 2024, arXiv:2406.06488)
// ============================================================================

/// Compute Euclidean distance matrix for a set of vectors.
///
/// Each row of `data` is a sample point (n_samples x dim).
pub fn euclidean_distance_matrix(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = data.len();
    let mut dist = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            let d = euclidean_distance(&data[i], &data[j]);
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }

    dist
}

/// Euclidean distance between two vectors.
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Precomputed distance matrices for efficient permutation testing.
///
/// Stores DXX (within X), DYY (within Y), and DXY (between X and Y).
#[derive(Debug, Clone)]
pub struct DistanceMatrices {
    pub dxx: Vec<Vec<f64>>,
    pub dyy: Vec<Vec<f64>>,
    pub dxy: Vec<Vec<f64>>,
    pub nx: usize,
    pub ny: usize,
}

impl DistanceMatrices {
    /// Build distance matrices from two samples X and Y.
    pub fn new(x: &[Vec<f64>], y: &[Vec<f64>]) -> Self {
        let nx = x.len();
        let ny = y.len();

        let dxx = euclidean_distance_matrix(x);
        let dyy = euclidean_distance_matrix(y);

        // Cross-distance matrix
        let mut dxy = vec![vec![0.0; ny]; nx];
        for i in 0..nx {
            for j in 0..ny {
                dxy[i][j] = euclidean_distance(&x[i], &y[j]);
            }
        }

        Self {
            dxx,
            dyy,
            dxy,
            nx,
            ny,
        }
    }
}

/// Energy Distance statistic.
///
/// ED = (2/nx*ny) * sum(DXY) - (1/nx^2) * sum(DXX) - (1/ny^2) * sum(DYY)
pub fn energy_distance(dm: &DistanceMatrices) -> f64 {
    let nx = dm.nx as f64;
    let ny = dm.ny as f64;

    // Sum of cross-distances
    let sum_xy: f64 = dm.dxy.iter().flat_map(|row| row.iter()).sum();

    // Sum of within-X distances (upper triangle * 2)
    let sum_xx: f64 = dm.dxx.iter().flat_map(|row| row.iter()).sum();

    // Sum of within-Y distances (upper triangle * 2)
    let sum_yy: f64 = dm.dyy.iter().flat_map(|row| row.iter()).sum();

    (2.0 / (nx * ny)) * sum_xy - (1.0 / (nx * nx)) * sum_xx - (1.0 / (ny * ny)) * sum_yy
}

/// Energy Distance with efficient permutation test.
///
/// Uses the swap-based approach from Chaibub Neto & Prisco (2024):
/// instead of recomputing distance matrices for each permutation,
/// we swap elements between groups and update the sums incrementally.
#[derive(Debug, Clone)]
pub struct EnergyDistanceResult {
    pub statistic: f64,
    pub p_value: f64,
    pub null_distribution: Vec<f64>,
    pub significant_at_005: bool,
}

/// Perform Energy Distance permutation test with efficient swapping.
///
/// The key insight is that all information for ED under any permutation
/// is already in the combined distance matrix. Swapping labels only
/// requires O(n) updates per permutation instead of O(n^2) recomputation.
pub fn energy_distance_permutation_test(
    x: &[Vec<f64>],
    y: &[Vec<f64>],
    n_permutations: usize,
    seed: u64,
) -> EnergyDistanceResult {
    let dm = DistanceMatrices::new(x, y);
    let observed = energy_distance(&dm);

    // Combine all data for permutation
    let mut combined: Vec<Vec<f64>> = x.to_vec();
    combined.extend(y.iter().cloned());

    let n_total = combined.len();
    let nx = x.len();

    // Precompute full distance matrix once
    let full_dist = euclidean_distance_matrix(&combined);

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut null_distribution = Vec::with_capacity(n_permutations);

    for _ in 0..n_permutations {
        // Generate random permutation of indices
        let mut indices: Vec<usize> = (0..n_total).collect();
        indices.shuffle(&mut rng);

        // First nx go to X, rest to Y (under permutation)
        let perm_x_indices = &indices[..nx];
        let perm_y_indices = &indices[nx..];

        // Compute ED using precomputed distances
        let ed = compute_ed_from_indices(&full_dist, perm_x_indices, perm_y_indices);
        null_distribution.push(ed);
    }

    // P-value: proportion of null >= observed
    let n_extreme = null_distribution.iter().filter(|&&d| d >= observed).count();
    let p_value = n_extreme as f64 / n_permutations as f64;

    EnergyDistanceResult {
        statistic: observed,
        p_value,
        null_distribution,
        significant_at_005: p_value < 0.05,
    }
}

/// Compute ED from a full distance matrix given index assignments.
fn compute_ed_from_indices(
    full_dist: &[Vec<f64>],
    x_indices: &[usize],
    y_indices: &[usize],
) -> f64 {
    let nx = x_indices.len() as f64;
    let ny = y_indices.len() as f64;

    // Sum of cross-distances
    let mut sum_xy = 0.0;
    for &i in x_indices {
        for &j in y_indices {
            sum_xy += full_dist[i][j];
        }
    }

    // Sum of within-X distances
    let mut sum_xx = 0.0;
    for (idx_a, &i) in x_indices.iter().enumerate() {
        for &j in &x_indices[(idx_a + 1)..] {
            sum_xx += 2.0 * full_dist[i][j];
        }
    }

    // Sum of within-Y distances
    let mut sum_yy = 0.0;
    for (idx_a, &i) in y_indices.iter().enumerate() {
        for &j in &y_indices[(idx_a + 1)..] {
            sum_yy += 2.0 * full_dist[i][j];
        }
    }

    (2.0 / (nx * ny)) * sum_xy - (1.0 / (nx * nx)) * sum_xx - (1.0 / (ny * ny)) * sum_yy
}

/// Gaussian RBF kernel value.
fn rbf_kernel(a: &[f64], b: &[f64], sigma: f64) -> f64 {
    let dist_sq: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    (-dist_sq / (2.0 * sigma * sigma)).exp()
}

/// Compute kernel matrix for a set of vectors.
pub fn kernel_matrix(data: &[Vec<f64>], sigma: f64) -> Vec<Vec<f64>> {
    let n = data.len();
    let mut k = vec![vec![0.0; n]; n];

    for i in 0..n {
        k[i][i] = 1.0; // k(x, x) = 1 for RBF
        for j in (i + 1)..n {
            let val = rbf_kernel(&data[i], &data[j], sigma);
            k[i][j] = val;
            k[j][i] = val;
        }
    }

    k
}

/// Precomputed kernel matrices for efficient MMD permutation testing.
#[derive(Debug, Clone)]
pub struct KernelMatrices {
    pub kxx: Vec<Vec<f64>>,
    pub kyy: Vec<Vec<f64>>,
    pub kxy: Vec<Vec<f64>>,
    pub nx: usize,
    pub ny: usize,
}

impl KernelMatrices {
    /// Build kernel matrices from two samples X and Y.
    pub fn new(x: &[Vec<f64>], y: &[Vec<f64>], sigma: f64) -> Self {
        let nx = x.len();
        let ny = y.len();

        let kxx = kernel_matrix(x, sigma);
        let kyy = kernel_matrix(y, sigma);

        // Cross-kernel matrix
        let mut kxy = vec![vec![0.0; ny]; nx];
        for i in 0..nx {
            for j in 0..ny {
                kxy[i][j] = rbf_kernel(&x[i], &y[j], sigma);
            }
        }

        Self {
            kxx,
            kyy,
            kxy,
            nx,
            ny,
        }
    }
}

/// Maximum Mean Discrepancy (squared) statistic.
///
/// MMD^2 = (1/nx^2) * sum(KXX) + (1/ny^2) * sum(KYY) - (2/nx*ny) * sum(KXY)
pub fn mmd_squared(km: &KernelMatrices) -> f64 {
    let nx = km.nx as f64;
    let ny = km.ny as f64;

    let sum_xx: f64 = km.kxx.iter().flat_map(|row| row.iter()).sum();
    let sum_yy: f64 = km.kyy.iter().flat_map(|row| row.iter()).sum();
    let sum_xy: f64 = km.kxy.iter().flat_map(|row| row.iter()).sum();

    (1.0 / (nx * nx)) * sum_xx + (1.0 / (ny * ny)) * sum_yy - (2.0 / (nx * ny)) * sum_xy
}

/// MMD permutation test result.
#[derive(Debug, Clone)]
pub struct MmdResult {
    pub statistic: f64,
    pub p_value: f64,
    pub null_distribution: Vec<f64>,
    pub significant_at_005: bool,
}

/// Perform MMD permutation test with efficient kernel reuse.
pub fn mmd_permutation_test(
    x: &[Vec<f64>],
    y: &[Vec<f64>],
    sigma: f64,
    n_permutations: usize,
    seed: u64,
) -> MmdResult {
    let km = KernelMatrices::new(x, y, sigma);
    let observed = mmd_squared(&km);

    // Combine all data for permutation
    let mut combined: Vec<Vec<f64>> = x.to_vec();
    combined.extend(y.iter().cloned());

    let n_total = combined.len();
    let nx = x.len();

    // Precompute full kernel matrix once
    let full_kernel = kernel_matrix(&combined, sigma);

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut null_distribution = Vec::with_capacity(n_permutations);

    for _ in 0..n_permutations {
        // Generate random permutation
        let mut indices: Vec<usize> = (0..n_total).collect();
        indices.shuffle(&mut rng);

        let perm_x_indices = &indices[..nx];
        let perm_y_indices = &indices[nx..];

        // Compute MMD^2 using precomputed kernel
        let mmd = compute_mmd_from_indices(&full_kernel, perm_x_indices, perm_y_indices);
        null_distribution.push(mmd);
    }

    let n_extreme = null_distribution.iter().filter(|&&d| d >= observed).count();
    let p_value = n_extreme as f64 / n_permutations as f64;

    MmdResult {
        statistic: observed,
        p_value,
        null_distribution,
        significant_at_005: p_value < 0.05,
    }
}

/// Compute MMD^2 from full kernel matrix given index assignments.
fn compute_mmd_from_indices(
    full_kernel: &[Vec<f64>],
    x_indices: &[usize],
    y_indices: &[usize],
) -> f64 {
    let nx = x_indices.len() as f64;
    let ny = y_indices.len() as f64;

    // Sum of K(xi, xj) for all pairs in X
    let mut sum_xx = 0.0;
    for &i in x_indices {
        for &j in x_indices {
            sum_xx += full_kernel[i][j];
        }
    }

    // Sum of K(yi, yj) for all pairs in Y
    let mut sum_yy = 0.0;
    for &i in y_indices {
        for &j in y_indices {
            sum_yy += full_kernel[i][j];
        }
    }

    // Sum of K(xi, yj) for cross pairs
    let mut sum_xy = 0.0;
    for &i in x_indices {
        for &j in y_indices {
            sum_xy += full_kernel[i][j];
        }
    }

    (1.0 / (nx * nx)) * sum_xx + (1.0 / (ny * ny)) * sum_yy - (2.0 / (nx * ny)) * sum_xy
}

/// Median heuristic for kernel bandwidth selection.
///
/// Returns the median of pairwise distances, a common choice for sigma.
pub fn median_heuristic(data: &[Vec<f64>]) -> f64 {
    let n = data.len();
    if n < 2 {
        return 1.0;
    }

    let mut distances = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            distances.push(euclidean_distance(&data[i], &data[j]));
        }
    }

    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
    distances[distances.len() / 2]
}

/// Combined two-sample test using both ED and MMD.
#[derive(Debug, Clone)]
pub struct TwoSampleTestResult {
    pub energy_distance: EnergyDistanceResult,
    pub mmd: MmdResult,
    /// Fisher's method combined p-value
    pub combined_p_value: f64,
    pub either_significant: bool,
    pub both_significant: bool,
}

/// Perform combined two-sample test with both ED and MMD.
pub fn two_sample_test(
    x: &[Vec<f64>],
    y: &[Vec<f64>],
    n_permutations: usize,
    seed: u64,
) -> TwoSampleTestResult {
    // Combine for sigma estimation
    let mut combined: Vec<Vec<f64>> = x.to_vec();
    combined.extend(y.iter().cloned());
    let sigma = median_heuristic(&combined);

    let ed_result = energy_distance_permutation_test(x, y, n_permutations, seed);
    let mmd_result = mmd_permutation_test(x, y, sigma, n_permutations, seed + 1000);

    // Fisher's method: -2 * sum(ln(p_i)) ~ chi^2(2k)
    let p1 = ed_result.p_value.max(1e-15);
    let p2 = mmd_result.p_value.max(1e-15);
    let chi2_stat = -2.0 * (p1.ln() + p2.ln());

    // Chi-squared(4) p-value approximation using regularized gamma
    // P(chi^2_4 >= x) = 1 - (1 + x/2) * exp(-x/2) for df=4
    let combined_p = (1.0 + chi2_stat / 2.0) * (-chi2_stat / 2.0).exp();

    TwoSampleTestResult {
        energy_distance: ed_result.clone(),
        mmd: mmd_result.clone(),
        combined_p_value: combined_p,
        either_significant: ed_result.significant_at_005 || mmd_result.significant_at_005,
        both_significant: ed_result.significant_at_005 && mmd_result.significant_at_005,
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

    // ========================================================================
    // Energy Distance and MMD Tests
    // ========================================================================

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let d = euclidean_distance(&a, &b);
        assert!((d - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_energy_distance_identical_samples() {
        // Identical samples should have ED = 0
        let x: Vec<Vec<f64>> = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let dm = DistanceMatrices::new(&x, &x);
        let ed = energy_distance(&dm);
        assert!(ed.abs() < 1e-10, "ED of identical samples should be 0");
    }

    #[test]
    fn test_energy_distance_different_samples() {
        let x: Vec<Vec<f64>> = vec![vec![0.0], vec![1.0], vec![2.0]];
        let y: Vec<Vec<f64>> = vec![vec![10.0], vec![11.0], vec![12.0]];

        let dm = DistanceMatrices::new(&x, &y);
        let ed = energy_distance(&dm);

        // Samples are far apart, ED should be positive and large
        assert!(ed > 5.0, "ED should be large for distant samples");
    }

    #[test]
    fn test_ed_permutation_test_identical() {
        // Identical distributions should have high p-value
        let x: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 * 0.1]).collect();
        let y = x.clone();

        let result = energy_distance_permutation_test(&x, &y, 100, 42);

        // P-value should be high (not significant)
        assert!(
            result.p_value > 0.1,
            "Identical samples should not be significantly different"
        );
        assert!(!result.significant_at_005);
    }

    #[test]
    fn test_ed_permutation_test_different() {
        // Very different distributions should have low p-value
        let x: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
        let y: Vec<Vec<f64>> = (0..20).map(|i| vec![100.0 + i as f64]).collect();

        let result = energy_distance_permutation_test(&x, &y, 100, 42);

        // P-value should be very low
        assert!(
            result.p_value < 0.05,
            "Distant samples should be significantly different"
        );
        assert!(result.significant_at_005);
    }

    #[test]
    fn test_mmd_identical_samples() {
        let x: Vec<Vec<f64>> = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let km = KernelMatrices::new(&x, &x, 1.0);
        let mmd = mmd_squared(&km);

        // MMD of sample with itself should be 0
        assert!(mmd.abs() < 1e-10, "MMD of identical samples should be 0");
    }

    #[test]
    fn test_mmd_permutation_test_different() {
        let x: Vec<Vec<f64>> = (0..15).map(|i| vec![i as f64]).collect();
        let y: Vec<Vec<f64>> = (0..15).map(|i| vec![50.0 + i as f64]).collect();

        let result = mmd_permutation_test(&x, &y, 1.0, 100, 42);

        // Distant samples should be significantly different
        assert!(result.significant_at_005);
    }

    #[test]
    fn test_median_heuristic() {
        let data: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];

        let sigma = median_heuristic(&data);

        // Distances: 1, 1, sqrt(2), 1, sqrt(2), 1
        // Sorted: 1, 1, 1, 1, 1.41, 1.41
        // Median ~ 1
        assert!(
            sigma > 0.9 && sigma < 1.5,
            "Median heuristic should be reasonable"
        );
    }

    #[test]
    fn test_two_sample_test_combined() {
        let x: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 0.1]).collect();
        let y: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 0.1 + 0.01]).collect();

        let result = two_sample_test(&x, &y, 50, 42);

        // Just verify it runs and produces valid output
        assert!(result.combined_p_value >= 0.0 && result.combined_p_value <= 1.0);
        assert!(result.energy_distance.p_value >= 0.0);
        assert!(result.mmd.p_value >= 0.0);
    }

    #[test]
    fn test_compute_ed_from_indices() {
        // Simple test: compute ED from indices should match direct computation
        let data: Vec<Vec<f64>> = vec![vec![0.0], vec![1.0], vec![2.0], vec![10.0], vec![11.0]];
        let full_dist = euclidean_distance_matrix(&data);

        let x_indices = vec![0, 1, 2];
        let y_indices = vec![3, 4];

        let ed = compute_ed_from_indices(&full_dist, &x_indices, &y_indices);

        // Verify ED is positive for distant groups
        assert!(ed > 0.0);
    }

    // ========================================================================
    // C-074 Associator Growth Law Tests
    // ========================================================================

    #[test]
    fn test_grid_search_growth_law_exact() {
        // Generate synthetic data from the claimed form:
        // y = 2.0 * (1 - 14.6 * d^{-1.80})
        let dims: Vec<f64> = vec![4.0, 8.0, 16.0, 32.0, 64.0, 128.0];
        let ys: Vec<f64> = dims
            .iter()
            .map(|&d| 2.0 * (1.0 - 14.6 * d.powf(-1.80)))
            .collect();

        let (a_inf, b, alpha) = super::grid_search_growth_law(&dims, &ys);

        // Should recover parameters close to original
        assert!((a_inf - 2.0).abs() < 0.1, "A_inf={} not near 2.0", a_inf);
        assert!((b - 14.6).abs() < 2.0, "B={} not near 14.6", b);
        assert!(
            (alpha - (-1.8)).abs() < 0.15,
            "alpha={} not near -1.8",
            alpha
        );
    }

    #[test]
    fn test_fit_associator_growth_law_basic() {
        // Synthetic data matching the C-074 claim
        let dims: Vec<f64> = vec![4.0, 8.0, 16.0, 32.0, 64.0];
        let ys: Vec<f64> = dims
            .iter()
            .map(|&d| 2.0 * (1.0 - 14.6 * d.powf(-1.80)))
            .collect();

        let result = super::fit_associator_growth_law(&dims, &ys, 100, 42);

        // Check R-squared is high for exact data
        assert!(result.r_squared > 0.99, "R^2={} too low", result.r_squared);
        // A_inf should be near 2.0
        assert!(
            (result.a_inf.point_estimate - 2.0).abs() < 0.2,
            "A_inf={} not near 2.0",
            result.a_inf.point_estimate
        );
        // Note: uncertain flag may still be true for small datasets due to
        // bootstrap variance - we only check the point estimates are correct.
    }

    // Helper to construct BootstrapCIResult for testing
    fn make_ci(point: f64, lower: f64, upper: f64) -> super::BootstrapCIResult {
        let width = upper - lower;
        super::BootstrapCIResult {
            point_estimate: point,
            ci_lower: lower,
            ci_upper: upper,
            ci_width: width,
            ci_width_relative: width / point.abs(),
            standard_error: width / 3.92, // Approx SE from 95% CI width
            n_bootstrap: 1000,
        }
    }

    #[test]
    fn test_c074_decision_rule_pass() {
        // Construct a result that should pass
        let result = super::AssociatorGrowthFitResult {
            a_inf: make_ci(2.0, 1.9, 2.1),
            b_coeff: make_ci(14.6, 12.0, 17.0),
            alpha: make_ci(-1.80, -2.0, -1.6),
            r_squared: 0.98,
            uncertain: false,
        };

        let (matches, reason) = super::c074_decision_rule(&result);
        assert!(matches, "Should pass: {}", reason);
    }

    #[test]
    fn test_c074_decision_rule_fail_alpha_ci() {
        // Construct a result where alpha CI excludes -1.80
        let result = super::AssociatorGrowthFitResult {
            a_inf: make_ci(2.0, 1.9, 2.1),
            b_coeff: make_ci(14.6, 12.0, 17.0),
            alpha: make_ci(-2.5, -3.0, -2.0), // CI excludes -1.80
            r_squared: 0.98,
            uncertain: false,
        };

        let (matches, reason) = super::c074_decision_rule(&result);
        assert!(!matches, "Should fail due to alpha CI");
        assert!(
            reason.contains("excludes -1.80"),
            "Reason should mention alpha: {}",
            reason
        );
    }

    #[test]
    fn test_c074_decision_rule_fail_low_r_squared() {
        // Construct a result with low R-squared
        let result = super::AssociatorGrowthFitResult {
            a_inf: make_ci(2.0, 1.9, 2.1),
            b_coeff: make_ci(14.6, 12.0, 17.0),
            alpha: make_ci(-1.80, -2.0, -1.6),
            r_squared: 0.85, // Below 0.95 threshold
            uncertain: false,
        };

        let (matches, reason) = super::c074_decision_rule(&result);
        assert!(!matches, "Should fail due to low R^2");
        assert!(
            reason.contains("R^2"),
            "Reason should mention R^2: {}",
            reason
        );
    }

    // ========================================================================
    // C-074 Integration Test with Actual Associator Data
    // ========================================================================
    //
    // This test generates actual Cayley-Dickson associator data using algebra_core
    // and verifies the C-074 claim: E[||A(a,b,c)||^2] = 2.00 * (1 - 14.6 * d^{-1.80})
    //
    // The test samples random unit vectors at each dimension, computes mean
    // squared associator norms, and checks if the fitted parameters match the claim.

    #[test]
    fn test_c074_integration_with_real_associator_data() {
        use algebra_core::{cd_associator, cd_norm_sq};
        use rand::prelude::*;
        use rand::rngs::StdRng;

        // Dimensions to test: 8 (octonions), 16 (sedenions), 32 (pathions), 64
        // Note: quaternions (dim=4) are associative, so skip them.
        // Octonions (dim=8) are only alternative (non-associative with zero associator
        // for certain special cases but generally non-zero).
        let dimensions: Vec<usize> = vec![8, 16, 32, 64];
        let n_samples = 500; // Number of random triples per dimension
        let seed = 12345u64;

        let mut rng = StdRng::seed_from_u64(seed);
        let mut dims_f64 = Vec::new();
        let mut mean_sq_norms = Vec::new();

        for dim in &dimensions {
            let mut sum_sq_norm = 0.0;

            for _ in 0..n_samples {
                // Generate three random unit vectors
                let a = random_unit_vector(*dim, &mut rng);
                let b = random_unit_vector(*dim, &mut rng);
                let c = random_unit_vector(*dim, &mut rng);

                // Compute associator A(a,b,c) = (ab)c - a(bc)
                let assoc = cd_associator(&a, &b, &c);
                let norm_sq = cd_norm_sq(&assoc);
                sum_sq_norm += norm_sq;
            }

            let mean_sq = sum_sq_norm / n_samples as f64;
            dims_f64.push(*dim as f64);
            mean_sq_norms.push(mean_sq);

            // Debug: print observed means
            eprintln!("dim={}: mean_sq_norm = {:.6}", dim, mean_sq);
        }

        // Fit the growth law: y = A_inf * (1 - B * d^alpha)
        let result = super::fit_associator_growth_law(&dims_f64, &mean_sq_norms, 200, seed);

        // Print diagnostic information
        eprintln!("\nC-074 Fit Results:");
        eprintln!(
            "  A_inf: {:.4} [{:.4}, {:.4}]",
            result.a_inf.point_estimate, result.a_inf.ci_lower, result.a_inf.ci_upper
        );
        eprintln!(
            "  B:     {:.4} [{:.4}, {:.4}]",
            result.b_coeff.point_estimate, result.b_coeff.ci_lower, result.b_coeff.ci_upper
        );
        eprintln!(
            "  alpha: {:.4} [{:.4}, {:.4}]",
            result.alpha.point_estimate, result.alpha.ci_lower, result.alpha.ci_upper
        );
        eprintln!("  R^2:   {:.4}", result.r_squared);
        eprintln!("  Uncertain: {}", result.uncertain);

        // Apply the decision rule
        let (matches, reason) = super::c074_decision_rule(&result);
        eprintln!(
            "\nDecision: {} - {}",
            if matches {
                "VERIFIED"
            } else {
                "NEEDS REVISION"
            },
            reason
        );

        // Core assertions for the integration test:
        // 1. R-squared should be high (model fits the data well)
        assert!(
            result.r_squared > 0.85,
            "R^2={:.4} too low - growth law form may not fit",
            result.r_squared
        );

        // 2. A_inf should be in reasonable range (near 2.0 for unit vectors)
        assert!(
            result.a_inf.point_estimate > 1.0 && result.a_inf.point_estimate < 3.0,
            "A_inf={:.4} outside expected range [1.0, 3.0]",
            result.a_inf.point_estimate
        );

        // 3. Alpha should be negative (decay with dimension)
        assert!(
            result.alpha.point_estimate < 0.0,
            "alpha={:.4} should be negative",
            result.alpha.point_estimate
        );

        // Note: The exact claim values (2.00, 14.6, -1.80) may need revision
        // based on the actual data. This test documents the empirical values
        // and checks if they're in a physically reasonable range.
    }

    /// Generate a random unit vector of given dimension.
    fn random_unit_vector(dim: usize, rng: &mut impl Rng) -> Vec<f64> {
        let mut v: Vec<f64> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in &mut v {
            *x /= norm;
        }
        v
    }

    #[test]
    fn test_c074_quaternion_associativity() {
        use algebra_core::{cd_associator, cd_norm_sq};
        use rand::prelude::*;
        use rand::rngs::StdRng;

        // Quaternions (dim=4) should be associative: ||A(a,b,c)|| = 0
        let mut rng = StdRng::seed_from_u64(42);
        let dim = 4;

        for _ in 0..20 {
            let a = random_unit_vector(dim, &mut rng);
            let b = random_unit_vector(dim, &mut rng);
            let c = random_unit_vector(dim, &mut rng);

            let assoc = cd_associator(&a, &b, &c);
            let norm = cd_norm_sq(&assoc).sqrt();

            assert!(
                norm < 1e-10,
                "Quaternion associator should be zero, got {}",
                norm
            );
        }
    }

    #[test]
    fn test_c074_octonion_nonassociativity() {
        use algebra_core::{cd_associator, cd_norm_sq};
        use rand::prelude::*;
        use rand::rngs::StdRng;

        // Octonions (dim=8) are non-associative: most triples have ||A(a,b,c)|| > 0
        let mut rng = StdRng::seed_from_u64(42);
        let dim = 8;
        let mut nonzero_count = 0;

        for _ in 0..50 {
            let a = random_unit_vector(dim, &mut rng);
            let b = random_unit_vector(dim, &mut rng);
            let c = random_unit_vector(dim, &mut rng);

            let assoc = cd_associator(&a, &b, &c);
            let norm = cd_norm_sq(&assoc).sqrt();

            if norm > 1e-6 {
                nonzero_count += 1;
            }
        }

        assert!(
            nonzero_count >= 40,
            "Octonions should be non-associative for most triples, only {}/50 nonzero",
            nonzero_count
        );
    }
}

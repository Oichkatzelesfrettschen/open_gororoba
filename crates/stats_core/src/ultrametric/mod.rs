//! Ultrametric structure analysis for real-valued datasets.
//!
//! Tests whether a set of values exhibits ultrametric (tree-like/hierarchical)
//! structure beyond what is expected from random data.
//!
//! # Methods
//!
//! 1. **Ultrametric fraction test**: For random triples (a,b,c), check
//!    d(a,c) <= max(d(a,b), d(b,c)). Random data yields ~20% by chance.
//!
//! 2. **Graded defect test**: Measures how badly the ultrametric inequality
//!    is violated. Hierarchical data has lower defect.
//!
//! 3. **P-adic clustering test**: Computes p-adic distances between integer-
//!    rounded values and checks ultrametric fraction under each prime's metric.
//!    P-adic metrics are inherently ultrametric, so the question is whether
//!    the data's p-adic structure exceeds what shuffled data produces.
//!
//! 4. **Local ultrametricity** (Bradley arXiv:2408.07174): For each point,
//!    find its epsilon-neighborhood and test triples within that neighborhood.
//!
//! 5. **Baire metric** (Murtagh arXiv:1104.4063): Encode multi-attribute
//!    tuples as p-adic digit sequences. Baire distance = p^(-k) where k is
//!    the longest common prefix length.
//!
//! 6. **Temporal cascade analysis**: For repeating transient sources, test
//!    whether waiting-time hierarchies exhibit ultrametric structure.
//!
//! 7. **Dendrogram / cophenetic correlation**: Build hierarchical clustering
//!    and measure how well the tree structure preserves pairwise distances.
//!
//! # Statistical methodology
//!
//! All tests use permutation nulls (shuffled values) with Bonferroni correction
//! across the full test battery.
//!
//! # References
//!
//! - Rammal, Toulouse, Virasoro (1986): Ultrametricity for physicists
//! - Mezard, Parisi, Virasoro (1987): Spin Glass Theory and Beyond
//! - Dragovich et al. (2017): p-Adic mathematical physics
//! - Bradley (2025): arXiv:2408.07174 (local ultrametricity)
//! - Murtagh (2004): arXiv:1104.4063 (Baire metric hierarchical clustering)

pub mod local;
pub mod baire;
pub mod temporal;
pub mod dendrogram;
pub mod null_models;
pub mod adaptive;
pub mod subset_search;
#[cfg(feature = "gpu")]
pub mod gpu;

use algebra_core::padic::{Rational, padic_distance};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use crate::claims_gates::{Evidence, GateResult, Verdict};

// Re-export submodule public types
pub use local::{LocalUltrametricResult, local_ultrametricity_test};
pub use baire::{
    BaireEncoder, AttributeSpec, baire_distance_matrix,
    euclidean_distance_matrix, euclidean_ultrametric_test, BaireTestResult,
    normalize_data_column_major, matrix_free_fraction,
    matrix_free_ultrametric_test, matrix_free_ultrametric_test_with_null,
    matrix_free_tolerance_curve, matrix_free_tolerance_curve_with_null,
};
pub use temporal::{CascadeAnalysis, WaitingTimeStats, analyze_temporal_cascade};
pub use dendrogram::{
    DendrogramResult, MultiLinkageResult, cophenetic_distance_matrix,
    cophenetic_correlation, hierarchical_ultrametric_test,
    hierarchical_ultrametric_test_with_method, multi_linkage_test,
};
pub use null_models::{NullModel, apply_null_column_major};
pub use adaptive::{AdaptiveConfig, AdaptiveResult, StopReason, adaptive_permutation_test};
pub use subset_search::{
    SubsetTestResult, SubsetSearchResult, SubsetSearchConfig,
    attribute_subsets, project_data, subset_search,
};

/// Configuration for ultrametric analysis.
#[derive(Debug, Clone)]
pub struct UltrametricConfig {
    /// Number of random triples to sample.
    pub n_triples: usize,
    /// Number of permutations for null distribution.
    pub n_permutations: usize,
    /// Primes for p-adic clustering tests.
    pub primes: Vec<u64>,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl Default for UltrametricConfig {
    fn default() -> Self {
        Self {
            n_triples: 100_000,
            n_permutations: 1_000,
            primes: vec![2, 3, 5, 7, 11, 13],
            seed: 42,
        }
    }
}

/// Result of ultrametric fraction test.
#[derive(Debug, Clone)]
pub struct UltrametricFractionResult {
    pub n_values: usize,
    pub n_triples: usize,
    pub ultrametric_fraction: f64,
    pub null_fraction_mean: f64,
    pub null_fraction_std: f64,
    pub p_value: f64,
    pub bootstrap_ci: (f64, f64),
}

/// Result of graded ultrametric defect test.
#[derive(Debug, Clone)]
pub struct UltrametricDefectResult {
    pub mean_defect: f64,
    pub median_defect: f64,
    pub null_mean_defect: f64,
    pub defect_p_value: f64,
}

/// Result of p-adic clustering test for a single prime.
#[derive(Debug, Clone)]
pub struct PadicClusterResult {
    pub prime: u64,
    pub mean_padic_distance: f64,
    pub padic_ultrametric_fraction: f64,
    pub null_ultrametric_fraction: f64,
    pub p_value: f64,
}

/// Complete ultrametric analysis result.
#[derive(Debug, Clone)]
pub struct UltrametricAnalysis {
    pub fraction_result: UltrametricFractionResult,
    pub defect_result: UltrametricDefectResult,
    pub padic_results: Vec<PadicClusterResult>,
    pub bonferroni_threshold: f64,
    pub any_significant: bool,
    pub verdict: Verdict,
}

// ---------------------------------------------------------------------------
// Core test functions
// ---------------------------------------------------------------------------

/// Test ultrametric fraction on Euclidean distances.
///
/// Samples `n_triples` random triples from `values`, computes pairwise
/// Euclidean distances, and checks the strong ultrametric inequality:
///   d(a,c) <= max(d(a,b), d(b,c))
///
/// Builds a null distribution by shuffling values `n_permutations` times.
pub fn ultrametric_fraction_test(
    values: &[f64],
    n_triples: usize,
    n_permutations: usize,
    seed: u64,
) -> UltrametricFractionResult {
    let n = values.len();
    assert!(n >= 3, "Need at least 3 values for triple test");

    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Compute observed ultrametric fraction
    let observed_frac = compute_ultrametric_fraction(values, n_triples, &mut rng);

    // Bootstrap CI for the observed fraction
    // Use a smaller triple count for each bootstrap replicate to keep cost manageable
    let n_bootstrap = 200;
    let boot_triple_count = n_triples.min(10_000);
    let mut boot_fracs = Vec::with_capacity(n_bootstrap);
    for _ in 0..n_bootstrap {
        let frac = compute_ultrametric_fraction(values, boot_triple_count, &mut rng);
        boot_fracs.push(frac);
    }
    boot_fracs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ci_lower = boot_fracs[(0.025 * n_bootstrap as f64) as usize];
    let ci_upper = boot_fracs[(0.975 * n_bootstrap as f64) as usize];

    // Null distribution: shuffle values and recompute
    let mut null_fracs = Vec::with_capacity(n_permutations);
    let mut shuffled = values.to_vec();
    for _ in 0..n_permutations {
        shuffled.shuffle(&mut rng);
        let frac = compute_ultrametric_fraction(&shuffled, n_triples, &mut rng);
        null_fracs.push(frac);
    }

    let null_mean = null_fracs.iter().sum::<f64>() / n_permutations as f64;
    let null_var = null_fracs
        .iter()
        .map(|f| (f - null_mean).powi(2))
        .sum::<f64>()
        / n_permutations as f64;
    let null_std = null_var.sqrt();

    // Two-sided p-value: fraction of null at least as extreme as observed
    // (testing whether observed fraction is significantly different from null)
    let n_extreme = null_fracs
        .iter()
        .filter(|&&f| (f - null_mean).abs() >= (observed_frac - null_mean).abs())
        .count();
    let p_value = (n_extreme as f64 + 1.0) / (n_permutations as f64 + 1.0);

    UltrametricFractionResult {
        n_values: n,
        n_triples,
        ultrametric_fraction: observed_frac,
        null_fraction_mean: null_mean,
        null_fraction_std: null_std,
        p_value,
        bootstrap_ci: (ci_lower, ci_upper),
    }
}

/// Compute ultrametric fraction for a set of values.
fn compute_ultrametric_fraction(values: &[f64], n_triples: usize, rng: &mut ChaCha8Rng) -> f64 {
    let n = values.len();
    let mut count = 0usize;

    for _ in 0..n_triples {
        let i = rng.gen_range(0..n);
        let mut j = rng.gen_range(0..n - 1);
        if j >= i {
            j += 1;
        }
        let mut k = rng.gen_range(0..n - 2);
        if k >= i.min(j) {
            k += 1;
        }
        if k >= i.max(j) {
            k += 1;
        }

        let d_ij = (values[i] - values[j]).abs();
        let d_jk = (values[j] - values[k]).abs();
        let d_ik = (values[i] - values[k]).abs();

        // Sort distances: dists[0] <= dists[1] <= dists[2]
        let mut dists = [d_ij, d_jk, d_ik];
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Ultrametric property: all triangles are isosceles with the
        // unequal side being shortest. Equivalently, the two largest
        // pairwise distances are approximately equal.
        //
        // We use a relative tolerance: the two largest distances must
        // differ by less than 5% of the largest distance.
        // This matches the methodology in Rammal et al. (1986).
        if dists[2] > 1e-15 {
            let relative_diff = (dists[2] - dists[1]) / dists[2];
            if relative_diff < 0.05 {
                count += 1;
            }
        } else {
            // All distances zero (all values equal) -- trivially ultrametric
            count += 1;
        }
    }

    count as f64 / n_triples as f64
}

/// Compute ultrametric fraction on a precomputed distance matrix.
///
/// This generalized version works with any distance matrix (Euclidean,
/// Baire, comoving, etc.) rather than computing Euclidean distances
/// from scalar values.
///
/// `dist_matrix` is a flat upper-triangle stored row-major:
/// index(i,j) = i*n - i*(i+1)/2 + j - i - 1 for i < j.
pub fn ultrametric_fraction_from_matrix(
    dist_matrix: &[f64],
    n_points: usize,
    n_triples: usize,
    seed: u64,
) -> f64 {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut count = 0usize;

    let idx = |i: usize, j: usize| -> usize {
        let (a, b) = if i < j { (i, j) } else { (j, i) };
        a * n_points - a * (a + 1) / 2 + b - a - 1
    };

    for _ in 0..n_triples {
        let i = rng.gen_range(0..n_points);
        let mut j = rng.gen_range(0..n_points - 1);
        if j >= i { j += 1; }
        let mut k = rng.gen_range(0..n_points - 2);
        if k >= i.min(j) { k += 1; }
        if k >= i.max(j) { k += 1; }

        let d_ij = dist_matrix[idx(i, j)];
        let d_jk = dist_matrix[idx(j, k)];
        let d_ik = dist_matrix[idx(i, k)];

        let mut dists = [d_ij, d_jk, d_ik];
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap());

        if dists[2] > 1e-15 {
            let relative_diff = (dists[2] - dists[1]) / dists[2];
            if relative_diff < 0.05 {
                count += 1;
            }
        } else {
            count += 1;
        }
    }

    count as f64 / n_triples as f64
}

/// Compute ultrametric fraction on a distance matrix with configurable tolerance.
///
/// Like `ultrametric_fraction_from_matrix` but with an explicit `epsilon`
/// parameter instead of the hardcoded 0.05. The two largest pairwise
/// distances in a triple must differ by less than `epsilon * d_max` to
/// count as ultrametric.
pub fn ultrametric_fraction_from_matrix_eps(
    dist_matrix: &[f64],
    n_points: usize,
    n_triples: usize,
    seed: u64,
    epsilon: f64,
) -> f64 {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut count = 0usize;

    let idx = |i: usize, j: usize| -> usize {
        let (a, b) = if i < j { (i, j) } else { (j, i) };
        a * n_points - a * (a + 1) / 2 + b - a - 1
    };

    for _ in 0..n_triples {
        let i = rng.gen_range(0..n_points);
        let mut j = rng.gen_range(0..n_points - 1);
        if j >= i { j += 1; }
        let mut k = rng.gen_range(0..n_points - 2);
        if k >= i.min(j) { k += 1; }
        if k >= i.max(j) { k += 1; }

        let d_ij = dist_matrix[idx(i, j)];
        let d_jk = dist_matrix[idx(j, k)];
        let d_ik = dist_matrix[idx(i, k)];

        let mut dists = [d_ij, d_jk, d_ik];
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap());

        if dists[2] > 1e-15 {
            let relative_diff = (dists[2] - dists[1]) / dists[2];
            if relative_diff < epsilon {
                count += 1;
            }
        } else {
            count += 1;
        }
    }

    count as f64 / n_triples as f64
}

/// One point on the tolerance curve.
#[derive(Debug, Clone)]
pub struct ToleranceCurvePoint {
    /// The epsilon (relative tolerance) value.
    pub epsilon: f64,
    /// Observed ultrametric fraction at this epsilon.
    pub observed: f64,
    /// Mean null fraction at this epsilon.
    pub null_mean: f64,
    /// Excess = observed - null_mean. Positive means more ultrametric than null.
    pub excess: f64,
}

/// Result of a tolerance curve sweep.
#[derive(Debug, Clone)]
pub struct ToleranceCurveResult {
    /// Points along the curve (one per epsilon value).
    pub points: Vec<ToleranceCurvePoint>,
    /// Area under the excess curve (trapezoidal rule).
    /// Positive AUC means the data is consistently more ultrametric than null.
    pub auc_excess: f64,
    /// Maximum excess observed at any epsilon.
    pub max_excess: f64,
    /// Epsilon at which maximum excess occurs.
    pub best_epsilon: f64,
}

/// Compute the tolerance curve: ultrametric fraction as a function of epsilon.
///
/// Instead of a single threshold, sweeps epsilon from 0.01 to 0.20 in steps
/// of 0.01. At each epsilon, computes observed fraction and null mean (from
/// column-shuffled permutations). Returns the full curve plus summary
/// statistics (AUC of excess, max excess, best epsilon).
///
/// The RNG state is shared across epsilons for each permutation so the same
/// triple samples and shuffles are reused, ensuring the curve is smooth.
pub fn tolerance_curve(
    dist_matrix: &[f64],
    n_points: usize,
    n_triples: usize,
    null_dist_matrices: &[Vec<f64>],
    seed: u64,
) -> ToleranceCurveResult {
    let epsilons: Vec<f64> = (1..=20).map(|i| i as f64 * 0.01).collect();

    let mut points = Vec::with_capacity(epsilons.len());

    for &eps in &epsilons {
        let obs = ultrametric_fraction_from_matrix_eps(
            dist_matrix, n_points, n_triples, seed, eps,
        );

        let null_mean = if null_dist_matrices.is_empty() {
            0.0
        } else {
            let sum: f64 = null_dist_matrices
                .iter()
                .map(|null_dm| {
                    ultrametric_fraction_from_matrix_eps(
                        null_dm, n_points, n_triples, seed, eps,
                    )
                })
                .sum();
            sum / null_dist_matrices.len() as f64
        };

        points.push(ToleranceCurvePoint {
            epsilon: eps,
            observed: obs,
            null_mean,
            excess: obs - null_mean,
        });
    }

    // AUC of excess curve (trapezoidal rule)
    let mut auc = 0.0;
    for w in points.windows(2) {
        let h = w[1].epsilon - w[0].epsilon;
        auc += 0.5 * h * (w[0].excess + w[1].excess);
    }

    let (max_excess, best_epsilon) = points
        .iter()
        .max_by(|a, b| a.excess.partial_cmp(&b.excess).unwrap())
        .map(|p| (p.excess, p.epsilon))
        .unwrap_or((0.0, 0.05));

    ToleranceCurveResult {
        points,
        auc_excess: auc,
        max_excess,
        best_epsilon,
    }
}

/// Test graded ultrametric defect.
///
/// For each triple, the defect measures how badly the ultrametric inequality
/// is violated: defect = max(0, d_max - d_second) / d_second
///
/// Hierarchical data has lower mean defect than random.
pub fn ultrametric_defect_test(
    values: &[f64],
    n_triples: usize,
    n_permutations: usize,
    seed: u64,
) -> UltrametricDefectResult {
    let n = values.len();
    assert!(n >= 3, "Need at least 3 values for triple test");

    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Compute observed defects
    let defects = compute_defects(values, n_triples, &mut rng);
    let mean_defect = defects.iter().sum::<f64>() / defects.len() as f64;

    let mut sorted_defects = defects.clone();
    sorted_defects.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_defect = sorted_defects[sorted_defects.len() / 2];

    // Null distribution
    let mut null_means = Vec::with_capacity(n_permutations);
    let mut shuffled = values.to_vec();
    for _ in 0..n_permutations {
        shuffled.shuffle(&mut rng);
        let null_defects = compute_defects(&shuffled, n_triples, &mut rng);
        let null_mean = null_defects.iter().sum::<f64>() / null_defects.len() as f64;
        null_means.push(null_mean);
    }

    let null_mean_defect = null_means.iter().sum::<f64>() / n_permutations as f64;

    // One-sided p-value: fraction of null with mean defect <= observed
    // (lower defect = more ultrametric)
    let n_lower = null_means.iter().filter(|&&m| m <= mean_defect).count();
    let defect_p_value = (n_lower as f64 + 1.0) / (n_permutations as f64 + 1.0);

    UltrametricDefectResult {
        mean_defect,
        median_defect,
        null_mean_defect,
        defect_p_value,
    }
}

/// Compute defects for random triples.
fn compute_defects(values: &[f64], n_triples: usize, rng: &mut ChaCha8Rng) -> Vec<f64> {
    let n = values.len();
    let mut defects = Vec::with_capacity(n_triples);

    for _ in 0..n_triples {
        let i = rng.gen_range(0..n);
        let mut j = rng.gen_range(0..n - 1);
        if j >= i {
            j += 1;
        }
        let mut k = rng.gen_range(0..n - 2);
        if k >= i.min(j) {
            k += 1;
        }
        if k >= i.max(j) {
            k += 1;
        }

        let d_ij = (values[i] - values[j]).abs();
        let d_jk = (values[j] - values[k]).abs();
        let d_ik = (values[i] - values[k]).abs();

        let mut dists = [d_ij, d_jk, d_ik];
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // defect = (d_max - d_second) / d_second when d_second > 0
        if dists[1] > 1e-15 {
            let defect = (dists[2] - dists[1]) / dists[1];
            defects.push(defect.max(0.0));
        } else {
            defects.push(0.0);
        }
    }

    defects
}

/// Test p-adic clustering for a single prime.
///
/// Converts float values to integers (multiply by scale factor, round),
/// computes p-adic distances, and checks ultrametric fraction.
/// Since p-adic distances are inherently ultrametric, the test checks
/// whether the *fraction* exceeds what shuffled data produces.
pub fn padic_clustering_test(
    values: &[f64],
    prime: u64,
    n_triples: usize,
    n_permutations: usize,
    seed: u64,
) -> PadicClusterResult {
    let n = values.len();
    assert!(n >= 3, "Need at least 3 values for triple test");

    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Convert to integers: multiply by 10 for 0.1 precision
    let int_values: Vec<i64> = values.iter().map(|&v| (v * 10.0).round() as i64).collect();

    // Compute observed p-adic ultrametric fraction and mean distance
    let (obs_frac, obs_mean_dist) =
        compute_padic_fraction(&int_values, prime, n_triples, &mut rng);

    // Null distribution
    let mut null_fracs = Vec::with_capacity(n_permutations);
    let mut shuffled = int_values.clone();
    for _ in 0..n_permutations {
        shuffled.shuffle(&mut rng);
        let (frac, _) = compute_padic_fraction(&shuffled, prime, n_triples, &mut rng);
        null_fracs.push(frac);
    }

    let null_mean_frac = null_fracs.iter().sum::<f64>() / n_permutations as f64;

    // Two-sided p-value
    let n_extreme = null_fracs
        .iter()
        .filter(|&&f| (f - null_mean_frac).abs() >= (obs_frac - null_mean_frac).abs())
        .count();
    let p_value = (n_extreme as f64 + 1.0) / (n_permutations as f64 + 1.0);

    PadicClusterResult {
        prime,
        mean_padic_distance: obs_mean_dist,
        padic_ultrametric_fraction: obs_frac,
        null_ultrametric_fraction: null_mean_frac,
        p_value,
    }
}

/// Compute p-adic ultrametric fraction and mean distance for integer values.
fn compute_padic_fraction(
    int_values: &[i64],
    prime: u64,
    n_triples: usize,
    rng: &mut ChaCha8Rng,
) -> (f64, f64) {
    let n = int_values.len();
    let mut ultra_count = 0usize;
    let mut total_dist = 0.0;
    let mut dist_count = 0usize;

    for _ in 0..n_triples {
        let i = rng.gen_range(0..n);
        let mut j = rng.gen_range(0..n - 1);
        if j >= i {
            j += 1;
        }
        let mut k = rng.gen_range(0..n - 2);
        if k >= i.min(j) {
            k += 1;
        }
        if k >= i.max(j) {
            k += 1;
        }

        let a = Rational::from_int(int_values[i]);
        let b = Rational::from_int(int_values[j]);
        let c = Rational::from_int(int_values[k]);

        let d_ab = padic_distance(a, b, prime);
        let d_bc = padic_distance(b, c, prime);
        let d_ac = padic_distance(a, c, prime);

        total_dist += d_ab + d_bc + d_ac;
        dist_count += 3;

        // Check ultrametric: two largest p-adic distances approximately equal.
        // P-adic metrics are inherently ultrametric, so this should hold exactly,
        // but we use the same relative tolerance for consistency.
        let mut dists = [d_ab, d_bc, d_ac];
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap());

        if dists[2] > 1e-15 {
            let relative_diff = (dists[2] - dists[1]) / dists[2];
            if relative_diff < 0.05 {
                ultra_count += 1;
            }
        } else {
            ultra_count += 1;
        }
    }

    let frac = ultra_count as f64 / n_triples as f64;
    let mean_dist = if dist_count > 0 {
        total_dist / dist_count as f64
    } else {
        0.0
    };

    (frac, mean_dist)
}

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

/// Run the complete ultrametric analysis battery.
///
/// Runs:
/// 1. Euclidean ultrametric fraction test
/// 2. Graded defect test
/// 3. P-adic clustering test for each prime in config
///
/// Applies Bonferroni correction across all tests.
pub fn run_ultrametric_analysis(
    values: &[f64],
    config: &UltrametricConfig,
) -> UltrametricAnalysis {
    let n_tests = 2 + config.primes.len();
    let bonferroni_threshold = 0.05 / n_tests as f64;

    // 1. Fraction test
    let fraction_result = ultrametric_fraction_test(
        values,
        config.n_triples,
        config.n_permutations,
        config.seed,
    );

    // 2. Defect test
    let defect_result = ultrametric_defect_test(
        values,
        config.n_triples,
        config.n_permutations,
        config.seed + 1_000_000,
    );

    // 3. P-adic tests
    let padic_results: Vec<PadicClusterResult> = config
        .primes
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            padic_clustering_test(
                values,
                p,
                config.n_triples,
                config.n_permutations,
                config.seed + 2_000_000 + i as u64 * 100_000,
            )
        })
        .collect();

    // Check significance with Bonferroni correction
    let fraction_sig = fraction_result.p_value < bonferroni_threshold;
    let defect_sig = defect_result.defect_p_value < bonferroni_threshold;
    let padic_sig: Vec<bool> = padic_results
        .iter()
        .map(|r| r.p_value < bonferroni_threshold)
        .collect();

    let any_significant = fraction_sig || defect_sig || padic_sig.iter().any(|&s| s);

    let verdict = if any_significant {
        Verdict::Pass
    } else {
        Verdict::Fail
    };

    UltrametricAnalysis {
        fraction_result,
        defect_result,
        padic_results,
        bonferroni_threshold,
        any_significant,
        verdict,
    }
}

// ---------------------------------------------------------------------------
// Claims gate
// ---------------------------------------------------------------------------

/// Gate for ultrametric structure claims.
///
/// - Pass: ultrametric signal detected (any test significant after Bonferroni)
/// - Fail: no ultrametric signal (consistent with random data)
pub fn ultrametric_gate(
    claim_id: &str,
    analysis: &UltrametricAnalysis,
    description: &str,
) -> GateResult {
    let evidence = Evidence::Custom {
        metric_name: "ultrametric_fraction".to_string(),
        value: analysis.fraction_result.ultrametric_fraction,
        threshold: analysis.bonferroni_threshold,
        pass_if_above: true,
    };

    if analysis.any_significant {
        GateResult::pass(
            claim_id,
            evidence,
            &format!(
                "{}: ultrametric signal detected (frac={:.4}, null={:.4}, p={:.4})",
                description,
                analysis.fraction_result.ultrametric_fraction,
                analysis.fraction_result.null_fraction_mean,
                analysis.fraction_result.p_value,
            ),
        )
    } else {
        GateResult::fail(
            claim_id,
            evidence,
            &format!(
                "{}: no ultrametric signal (frac={:.4}, null={:.4}, p={:.4}, Bonferroni threshold={:.4})",
                description,
                analysis.fraction_result.ultrametric_fraction,
                analysis.fraction_result.null_fraction_mean,
                analysis.fraction_result.p_value,
                analysis.bonferroni_threshold,
            ),
        )
    }
}

// ---------------------------------------------------------------------------
// Multiple testing correction
// ---------------------------------------------------------------------------

/// Result of Benjamini-Hochberg FDR correction.
#[derive(Debug, Clone)]
pub struct FdrResult {
    /// Adjusted p-values (in original order).
    pub adjusted_p_values: Vec<f64>,
    /// Which tests are significant at the given FDR level (in original order).
    pub significant: Vec<bool>,
    /// Number of significant tests.
    pub n_significant: usize,
    /// FDR level used.
    pub fdr_level: f64,
}

/// Benjamini-Hochberg FDR correction for multiple testing.
///
/// Given a set of p-values, returns adjusted p-values and a boolean mask
/// indicating which tests are significant at the specified FDR level.
///
/// Less conservative than Bonferroni for exploration with many tests.
///
/// # Algorithm
/// 1. Sort p-values by rank
/// 2. Adjusted p_i = min(p_i * m / rank_i, 1.0)
/// 3. Enforce monotonicity from bottom up
/// 4. Compare adjusted p-values to FDR level
pub fn benjamini_hochberg(p_values: &[f64], fdr_level: f64) -> FdrResult {
    let m = p_values.len();
    if m == 0 {
        return FdrResult {
            adjusted_p_values: vec![],
            significant: vec![],
            n_significant: 0,
            fdr_level,
        };
    }

    // Sort indices by p-value
    let mut indices: Vec<usize> = (0..m).collect();
    indices.sort_by(|&a, &b| {
        p_values[a]
            .partial_cmp(&p_values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Compute adjusted p-values
    let mut adjusted = vec![0.0; m];
    for (rank_minus_1, &orig_idx) in indices.iter().enumerate() {
        let rank = rank_minus_1 + 1;
        adjusted[orig_idx] = (p_values[orig_idx] * m as f64 / rank as f64).min(1.0);
    }

    // Enforce monotonicity: walk from largest rank to smallest,
    // ensuring adjusted[rank_i] <= adjusted[rank_{i+1}]
    let mut running_min = 1.0;
    for &orig_idx in indices.iter().rev() {
        adjusted[orig_idx] = adjusted[orig_idx].min(running_min);
        running_min = adjusted[orig_idx];
    }

    let significant: Vec<bool> = adjusted.iter().map(|&p| p < fdr_level).collect();
    let n_significant = significant.iter().filter(|&&s| s).count();

    FdrResult {
        adjusted_p_values: adjusted,
        significant,
        n_significant,
        fdr_level,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ultrametric_fraction_powers_of_two() {
        // Powers of 2 have strong 2-adic structure and should show
        // high ultrametric fraction under Euclidean distance too,
        // since they form a geometric sequence.
        let values: Vec<f64> = (0..10).map(|i| 2.0_f64.powi(i)).collect();

        let result = ultrametric_fraction_test(&values, 5_000, 100, 42);

        // The fraction should be well-defined
        assert!(
            result.ultrametric_fraction >= 0.0 && result.ultrametric_fraction <= 1.0,
            "Fraction out of range: {}",
            result.ultrametric_fraction
        );
        // For geometric sequences, the fraction is typically high (~0.33)
        assert!(
            result.ultrametric_fraction > 0.15,
            "Expected high fraction for geometric sequence, got {}",
            result.ultrametric_fraction
        );
    }

    #[test]
    fn test_ultrametric_fraction_uniform_random() {
        // Uniform random data baseline calibration.
        // With 5% relative tolerance on the isosceles condition,
        // 1D uniform random data yields ~10% ultrametric fraction.
        // (The often-cited ~20% figure uses a looser absolute tolerance.)
        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let values: Vec<f64> = (0..200).map(|_| rng.gen_range(0.0..1000.0)).collect();

        let result = ultrametric_fraction_test(&values, 50_000, 100, 456);

        // Should be near 0.10 (within reasonable margin for 5% tolerance)
        assert!(
            result.ultrametric_fraction > 0.05 && result.ultrametric_fraction < 0.20,
            "Expected ~0.10 for uniform random with 5% tolerance, got {}",
            result.ultrametric_fraction
        );
        // P-value should be high (not significant -- data is random)
        assert!(
            result.p_value > 0.01,
            "Random data should not be significant, p={}",
            result.p_value
        );
    }

    #[test]
    fn test_defect_zero_for_trivial_case() {
        // All equal values: every distance is 0, defect should be 0
        let values = vec![5.0; 20];
        let result = ultrametric_defect_test(&values, 1_000, 50, 42);

        assert!(
            result.mean_defect < 1e-10,
            "Identical values should have zero defect, got {}",
            result.mean_defect
        );
    }

    #[test]
    fn test_defect_positive_for_random() {
        // Random data should have positive defect
        let mut rng = ChaCha8Rng::seed_from_u64(789);
        let values: Vec<f64> = (0..100).map(|_| rng.gen_range(0.0..1000.0)).collect();

        let result = ultrametric_defect_test(&values, 10_000, 100, 42);

        assert!(
            result.mean_defect > 0.0,
            "Random data should have positive defect, got {}",
            result.mean_defect
        );
        assert!(
            result.median_defect > 0.0,
            "Median defect should be positive, got {}",
            result.median_defect
        );
    }

    #[test]
    fn test_padic_distance_handles_equal_values() {
        // When two values are equal, p-adic distance should be 0
        let a = Rational::from_int(42);
        let b = Rational::from_int(42);

        let d = padic_distance(a, b, 2);
        assert!(
            d.abs() < 1e-15,
            "Distance between equal values should be 0, got {}",
            d
        );
    }

    #[test]
    fn test_padic_clustering_basic() {
        // Basic smoke test for p-adic clustering
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let values: Vec<f64> = (0..50).map(|_| rng.gen_range(1.0..500.0)).collect();

        let result = padic_clustering_test(&values, 2, 5_000, 100, 42);

        assert!(
            result.padic_ultrametric_fraction >= 0.0
                && result.padic_ultrametric_fraction <= 1.0,
            "Fraction out of range: {}",
            result.padic_ultrametric_fraction
        );
        assert!(
            result.p_value >= 0.0 && result.p_value <= 1.0,
            "P-value out of range: {}",
            result.p_value
        );
    }

    #[test]
    fn test_run_analysis_smoke() {
        // End-to-end test with synthetic data (small for speed)
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let values: Vec<f64> = (0..30).map(|_| rng.gen_range(100.0..1000.0)).collect();

        let config = UltrametricConfig {
            n_triples: 1_000,
            n_permutations: 50,
            primes: vec![2, 3, 5],
            seed: 42,
        };

        let analysis = run_ultrametric_analysis(&values, &config);

        // Verify structure
        assert_eq!(analysis.padic_results.len(), 3);
        assert!(analysis.bonferroni_threshold > 0.0);
        assert!(analysis.bonferroni_threshold < 0.05);

        // Random data should not be significant
        // (this is probabilistic but should hold for reasonable seeds)
        assert_eq!(
            analysis.verdict,
            Verdict::Fail,
            "Random data should not show ultrametric signal"
        );
    }

    #[test]
    fn test_ultrametric_gate_integration() {
        // Test the gate function with synthetic analysis
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let values: Vec<f64> = (0..30).map(|_| rng.gen_range(100.0..1000.0)).collect();

        let config = UltrametricConfig {
            n_triples: 1_000,
            n_permutations: 50,
            primes: vec![2, 3],
            seed: 42,
        };

        let analysis = run_ultrametric_analysis(&values, &config);
        let gate = ultrametric_gate("C-071", &analysis, "FRB ultrametric test");

        assert_eq!(gate.claim_id, "C-071");
        // Random data: gate should fail (no signal)
        assert_eq!(gate.verdict, Verdict::Fail);
    }

    #[test]
    fn test_ultrametric_fraction_from_matrix() {
        // Test the matrix-based fraction computation with a small example
        // 4 points forming a perfect ultrametric tree:
        //       *
        //      / \
        //     *   *
        //    / \ / \
        //   0  1 2  3
        // d(0,1) = 1, d(2,3) = 1, d(0,2) = d(0,3) = d(1,2) = d(1,3) = 2
        // Upper triangle (row-major): d01, d02, d03, d12, d13, d23
        let dist_matrix = vec![1.0, 2.0, 2.0, 2.0, 2.0, 1.0];

        let frac = ultrametric_fraction_from_matrix(&dist_matrix, 4, 10_000, 42);

        // Perfect ultrametric: fraction should be 1.0
        assert!(
            frac > 0.95,
            "Perfect ultrametric tree should yield fraction ~1.0, got {}",
            frac
        );
    }

    #[test]
    fn test_fraction_eps_monotonic() {
        // Ultrametric fraction must increase with epsilon (monotonic).
        // Larger tolerance accepts more triples.
        let dist_matrix = vec![1.0, 2.0, 2.0, 2.0, 2.0, 1.0];
        let n = 4;

        let mut prev = 0.0;
        for i in 1..=20 {
            let eps = i as f64 * 0.01;
            let frac = ultrametric_fraction_from_matrix_eps(
                &dist_matrix, n, 10_000, 42, eps,
            );
            assert!(
                frac >= prev - 0.01, // Allow tiny noise from sampling
                "Fraction should be non-decreasing: eps={} frac={} < prev={}",
                eps, frac, prev,
            );
            prev = frac;
        }
    }

    #[test]
    fn test_tolerance_curve_structure() {
        // Tolerance curve on perfect ultrametric: excess should be positive
        // at all epsilons since tree distances are always exactly isosceles.
        let dist_matrix = vec![1.0, 2.0, 2.0, 2.0, 2.0, 1.0];
        let n = 4;

        // Generate a "null" by perturbing distances
        let null_dm = vec![1.2, 1.8, 2.1, 1.9, 2.2, 0.9];
        let result = tolerance_curve(&dist_matrix, n, 5_000, &[null_dm], 42);

        assert_eq!(result.points.len(), 20);
        assert!(result.points[0].epsilon > 0.0);
        assert!(result.points[19].epsilon < 0.21);
        // Perfect ultrametric should have positive AUC
        assert!(
            result.auc_excess > 0.0,
            "Perfect ultrametric should have positive AUC, got {}",
            result.auc_excess,
        );
        assert!(result.max_excess > 0.0);
        assert!(result.best_epsilon > 0.0);
    }

    #[test]
    fn test_bh_fdr_no_signal() {
        // All p-values high: nothing should be significant
        let p_values = vec![0.50, 0.60, 0.70, 0.80, 0.90];
        let result = benjamini_hochberg(&p_values, 0.05);

        assert_eq!(result.n_significant, 0);
        assert!(result.significant.iter().all(|&s| !s));
        assert!(result.adjusted_p_values.iter().all(|&p| p >= 0.05));
    }

    #[test]
    fn test_bh_fdr_some_signal() {
        // Mix of strong and weak signals
        let p_values = vec![0.001, 0.004, 0.030, 0.500, 0.800];
        let result = benjamini_hochberg(&p_values, 0.05);

        // First two should survive FDR correction
        assert!(result.significant[0], "p=0.001 should be significant");
        assert!(result.significant[1], "p=0.004 should be significant");
        // Last two definitely not
        assert!(!result.significant[3], "p=0.500 should not be significant");
        assert!(!result.significant[4], "p=0.800 should not be significant");
        assert!(result.n_significant >= 2);
    }

    #[test]
    fn test_bh_fdr_adjusted_monotonic() {
        // Adjusted p-values should be monotonic when sorted by raw p-value
        let p_values = vec![0.01, 0.03, 0.05, 0.10, 0.50];
        let result = benjamini_hochberg(&p_values, 0.05);

        // Already sorted by raw p-value; adjusted should be non-decreasing
        for w in result.adjusted_p_values.windows(2) {
            assert!(
                w[1] >= w[0] - 1e-15,
                "Adjusted p-values should be non-decreasing: {} > {}",
                w[0], w[1],
            );
        }
    }

    #[test]
    fn test_bh_fdr_empty() {
        let result = benjamini_hochberg(&[], 0.05);
        assert_eq!(result.n_significant, 0);
        assert!(result.adjusted_p_values.is_empty());
    }

    #[test]
    fn test_bh_fdr_all_significant() {
        // All p-values very small
        let p_values = vec![0.0001, 0.0002, 0.0003];
        let result = benjamini_hochberg(&p_values, 0.05);
        assert_eq!(result.n_significant, 3);
    }
}

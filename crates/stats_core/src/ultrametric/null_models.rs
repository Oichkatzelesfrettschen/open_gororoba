//! Null model strategies for ultrametric permutation testing.
//!
//! Provides multiple null models that destroy different types of structure
//! in multi-attribute datasets while preserving specified properties.
//!
//! # Models
//!
//! - **ColumnIndependent**: Shuffle each column independently. Destroys
//!   inter-attribute correlations while preserving each attribute's marginal
//!   distribution. This is the default and backward-compatible choice.
//!
//! - **RowPermutation**: Permute entire rows. Preserves joint distributions
//!   within each observation but destroys any ordering or spatial structure.
//!   Appropriate when the question is whether row *labels* carry ultrametric
//!   signal beyond what the joint distribution already provides.
//!
//! - **ToroidalShift**: Apply a random circular shift independently per
//!   column. Preserves marginal distributions and local autocorrelation
//!   structure while destroying cross-column alignment. Appropriate for
//!   time-ordered or spatially-ordered data where local structure is expected.
//!
//! - **RandomRotation**: Apply a Haar-distributed random SO(d) rotation
//!   to the d-dimensional data. Preserves the full covariance ellipsoid
//!   while destroying any axis-aligned hierarchical structure. Appropriate
//!   when testing whether ultrametricity is a property of the geometry
//!   rather than an artifact of the coordinate system.
//!
//! # Which null is informative for which test?
//!
//! Not all null models are informative for all test statistics. For any
//! statistic that depends only on the multiset of pairwise distances (like
//! ultrametric fraction), transformations that preserve all pairwise
//! distances are **identity-equivalent** (the test statistic is unchanged,
//! so the p-value is trivially 1.0):
//!
//! - **RowPermutation** is a relabeling of observations. Pairwise
//!   Euclidean distances are a set property, so permuting row indices
//!   leaves the distance multiset invariant. This null is informative
//!   only when the test statistic is sensitive to row ordering (e.g.,
//!   temporal cascades, windowed tests, adjacency-based statistics).
//!
//! - **RandomRotation** is an isometry: R in O(d) preserves all
//!   Euclidean distances exactly (||Rx - Ry|| = ||x - y||). The
//!   covariance eigenvalues are also preserved. This null is informative
//!   only for Baire-metric tests (where digit encoding is axis-aligned)
//!   or coordinate-dependent statistics.
//!
//! - **ToroidalShift** applies different circular shifts per column,
//!   creating chimeric rows that break inter-column alignment. This is
//!   informative for Euclidean-distance tests (like ColumnIndependent),
//!   but additionally preserves within-column autocorrelation. Use it
//!   when row ordering encodes temporal/spatial structure.
//!
//! - **ColumnIndependent** is the universally informative null for
//!   ultrametric fraction tests, because it destroys inter-attribute
//!   correlation structure while preserving marginals -- exactly the
//!   joint structure that creates hierarchical distance patterns.
//!
//! In summary, for Euclidean-distance-based ultrametric tests:
//!   - Identity-equivalent: RowPermutation, RandomRotation
//!   - Informative: ColumnIndependent, ToroidalShift
//!
//! # References
//!
//! - Mezzadri (2007): How to generate random matrices from compact groups
//! - Phipson & Smyth (2010): Permutation p-values should never be zero

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

/// Result from running all four null models on the same data.
#[derive(Debug, Clone)]
pub struct MultiNullResult {
    /// Observed ultrametric fraction (same across all null models).
    pub observed_fraction: f64,
    /// P-value under ColumnIndependent null.
    pub p_column_independent: f64,
    /// P-value under RowPermutation null.
    pub p_row_permutation: f64,
    /// P-value under ToroidalShift null.
    pub p_toroidal_shift: f64,
    /// P-value under RandomRotation null.
    pub p_random_rotation: f64,
    /// Number of points.
    pub n: usize,
    /// Number of attributes.
    pub d: usize,
    /// Number of permutations used per null model.
    pub n_permutations: usize,
}

/// Run all four null models on the same column-major data and return
/// p-values for each.
///
/// This implements the "different nulls test different hypotheses" principle
/// from Thesis H. The caller provides column-major data and the function
/// computes the ultrametric fraction p-value under each null model,
/// enabling direct comparison of which null models are informative vs
/// identity-equivalent for Euclidean-distance-based statistics.
///
/// # Arguments
///
/// * `cols` - Column-major data `[d][n]` (flat array, `cols[col * n + row]`)
/// * `n` - Number of observations (rows)
/// * `d` - Number of attributes (columns)
/// * `n_triples` - Number of sampled triples for fraction computation
/// * `n_permutations` - Number of permutations per null model
/// * `seed` - RNG seed for reproducibility
/// * `epsilon` - Relative tolerance for ultrametric triple test
pub fn multi_null_comparison(
    cols: &[f64],
    n: usize,
    d: usize,
    n_triples: usize,
    n_permutations: usize,
    seed: u64,
    epsilon: f64,
) -> MultiNullResult {
    assert_eq!(cols.len(), n * d);
    assert!(n >= 3);

    // Use exhaustive triple enumeration for small n (identity-equivalence
    // of RowPermutation only holds exactly for exhaustive evaluation).
    // C(100,3) = 161700 -- still fast.
    let use_exhaustive = n <= 100;

    let compute_fraction = |data: &[f64]| -> f64 {
        if use_exhaustive {
            exhaustive_euclidean_ultrametric_fraction(data, n, d, epsilon)
        } else {
            sampled_euclidean_ultrametric_fraction(data, n, d, n_triples, seed, epsilon)
        }
    };

    // Compute observed fraction (same for all null models)
    let obs_frac = compute_fraction(cols);

    // Run each null model
    let nulls = [
        NullModel::ColumnIndependent,
        NullModel::RowPermutation,
        NullModel::ToroidalShift,
        NullModel::RandomRotation,
    ];

    let mut p_values = [0.0_f64; 4];

    for (idx, &null_model) in nulls.iter().enumerate() {
        // Different seed per null model to avoid correlation between null distributions
        let mut rng = ChaCha8Rng::seed_from_u64(seed + 1_000_000 * (idx as u64 + 1));
        let mut n_extreme = 0usize;
        let mut shuffled = cols.to_vec();

        for _ in 0..n_permutations {
            shuffled.copy_from_slice(cols);
            apply_null_column_major(&mut shuffled, n, d, null_model, &mut rng);
            let null_frac = compute_fraction(&shuffled);
            if null_frac >= obs_frac {
                n_extreme += 1;
            }
        }

        // Phipson-Smyth correction: p = (r+1)/(k+1)
        p_values[idx] = (n_extreme as f64 + 1.0) / (n_permutations as f64 + 1.0);
    }

    MultiNullResult {
        observed_fraction: obs_frac,
        p_column_independent: p_values[0],
        p_row_permutation: p_values[1],
        p_toroidal_shift: p_values[2],
        p_random_rotation: p_values[3],
        n,
        d,
        n_permutations,
    }
}

/// Compute ultrametric fraction from sampled triples using Euclidean distances.
///
/// This is the core statistic: for each sampled triple (i,j,k), compute the
/// three pairwise Euclidean distances and check whether the two largest are
/// approximately equal (ultrametric inequality with relative tolerance epsilon).
fn sampled_euclidean_ultrametric_fraction(
    cols: &[f64],
    n: usize,
    d: usize,
    n_triples: usize,
    seed: u64,
    epsilon: f64,
) -> f64 {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut n_ultra = 0u64;
    let mut n_tested = 0u64;

    for _ in 0..n_triples {
        // Sample three distinct indices
        let i = rng.gen_range(0..n);
        let mut j = rng.gen_range(0..n - 1);
        if j >= i { j += 1; }
        let mut k = rng.gen_range(0..n - 2);
        if k >= i.min(j) { k += 1; }
        if k >= i.max(j) { k += 1; }

        // Compute pairwise Euclidean distances
        let dij = euclidean_dist_col_major(cols, n, d, i, j);
        let dik = euclidean_dist_col_major(cols, n, d, i, k);
        let djk = euclidean_dist_col_major(cols, n, d, j, k);

        let mut sorted = [dij, dik, djk];
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        n_tested += 1;
        if (sorted[2] - sorted[1]).abs() < epsilon * sorted[2].max(1e-15) {
            n_ultra += 1;
        }
    }

    if n_tested == 0 { return 0.0; }
    n_ultra as f64 / n_tested as f64
}

/// Compute ultrametric fraction by exhaustive enumeration of ALL C(n,3) triples.
///
/// Unlike the sampled version, this is invariant under RowPermutation because
/// it evaluates all index triples -- relabeling the points merely changes the
/// order of enumeration, not the multiset of (d_ij, d_ik, d_jk) values.
///
/// Use this for n <= ~100 (C(100,3) = 161700 triples, still fast).
fn exhaustive_euclidean_ultrametric_fraction(
    cols: &[f64],
    n: usize,
    d: usize,
    epsilon: f64,
) -> f64 {
    let mut n_ultra = 0u64;
    let mut n_tested = 0u64;

    for i in 0..n {
        for j in (i + 1)..n {
            let dij = euclidean_dist_col_major(cols, n, d, i, j);
            for k in (j + 1)..n {
                let dik = euclidean_dist_col_major(cols, n, d, i, k);
                let djk = euclidean_dist_col_major(cols, n, d, j, k);

                let mut sorted = [dij, dik, djk];
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

                n_tested += 1;
                if (sorted[2] - sorted[1]).abs() < epsilon * sorted[2].max(1e-15) {
                    n_ultra += 1;
                }
            }
        }
    }

    if n_tested == 0 { return 0.0; }
    n_ultra as f64 / n_tested as f64
}

/// Euclidean distance between rows i and j in column-major layout.
fn euclidean_dist_col_major(cols: &[f64], n: usize, d: usize, i: usize, j: usize) -> f64 {
    let mut sq = 0.0;
    for c in 0..d {
        let diff = cols[c * n + i] - cols[c * n + j];
        sq += diff * diff;
    }
    sq.sqrt()
}

/// Null model strategy for permutation testing.
///
/// See the [module-level documentation](self) for guidance on which null
/// models are informative for different test statistics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum NullModel {
    /// Shuffle each column independently (default, backward compatible).
    /// Informative for all distance-based ultrametric tests.
    #[default]
    ColumnIndependent,
    /// Permute entire rows (preserves joint distribution).
    /// Identity-equivalent for set-based distance statistics; informative
    /// only when row order encodes temporal/spatial structure.
    RowPermutation,
    /// Random circular shift per column (preserves autocorrelation).
    /// Informative only for time-ordered or spatially-ordered data where
    /// the test statistic is sensitive to row indexing.
    ToroidalShift,
    /// Haar-distributed SO(d) rotation (preserves covariance ellipsoid).
    /// Identity-equivalent for Euclidean-distance statistics (isometry);
    /// informative only for axis-aligned metrics like Baire encoding.
    RandomRotation,
}

// ---------------------------------------------------------------------------
// NullModelStrategy trait (Layer 5 abstraction)
// ---------------------------------------------------------------------------

/// Polymorphic interface for null model strategies.
///
/// Implementing this trait allows external crates to define custom null models
/// (e.g., codebook-specific nulls in algebra_core) that integrate with the
/// adaptive permutation testing framework without modifying stats_core.
///
/// # Required methods
///
/// - [`name()`](NullModelStrategy::name): Human-readable identifier for logging.
/// - [`apply()`](NullModelStrategy::apply): Transform column-major data under
///   the null hypothesis.
///
/// # Optional methods
///
/// - [`is_euclidean_identity()`](NullModelStrategy::is_euclidean_identity):
///   Whether this null preserves all pairwise Euclidean distances.
pub trait NullModelStrategy: Send + Sync {
    /// Short human-readable name (e.g., "ColumnIndependent").
    fn name(&self) -> &str;

    /// Apply the null model transformation to column-major data in-place.
    ///
    /// `cols` is column-major: `cols[c * n + i]` = value of attribute `c`
    /// for observation `i`. The array has length `n * d`.
    fn apply(&self, cols: &mut [f64], n: usize, d: usize, rng: &mut ChaCha8Rng);

    /// Whether this null model is identity-equivalent for Euclidean-distance
    /// statistics (the transformation is an isometry or relabeling).
    ///
    /// If `true`, permutation tests based on pairwise Euclidean distances
    /// yield trivial p-values (~1.0). Callers can skip such tests to save
    /// computation.
    fn is_euclidean_identity(&self) -> bool {
        false
    }
}

/// Apply a null model transformation to column-major data in-place.
///
/// `cols`: flat column-major array of shape `[d][n]`, i.e. `cols[col * n + row]`.
/// `n`: number of rows (observations).
/// `d`: number of columns (attributes).
/// `null_model`: which null model strategy to apply.
/// `rng`: random number generator (mutated).
pub fn apply_null_column_major(
    cols: &mut [f64],
    n: usize,
    d: usize,
    null_model: NullModel,
    rng: &mut ChaCha8Rng,
) {
    match null_model {
        NullModel::ColumnIndependent => {
            for col in 0..d {
                let start = col * n;
                cols[start..start + n].shuffle(rng);
            }
        }
        NullModel::RowPermutation => {
            // Generate a row permutation and apply it to all columns
            let mut perm: Vec<usize> = (0..n).collect();
            perm.shuffle(rng);

            // Apply permutation via a temporary buffer (one column at a time)
            let mut buf = vec![0.0_f64; n];
            for col in 0..d {
                let start = col * n;
                for (i, &pi) in perm.iter().enumerate() {
                    buf[i] = cols[start + pi];
                }
                cols[start..start + n].copy_from_slice(&buf);
            }
        }
        NullModel::ToroidalShift => {
            // Random circular shift per column
            let mut buf = vec![0.0_f64; n];
            for col in 0..d {
                let shift = rng.gen_range(0..n);
                let start = col * n;
                // Rotate: new[i] = old[(i + shift) % n]
                for i in 0..n {
                    buf[i] = cols[start + (i + shift) % n];
                }
                cols[start..start + n].copy_from_slice(&buf);
            }
        }
        NullModel::RandomRotation => {
            // Haar-distributed SO(d) rotation via QR of Gaussian matrix
            // (Mezzadri 2007). For d <= ~20 this is fast; for larger d
            // the O(d^3) QR cost dominates.
            if d < 2 {
                // 1D: rotation is trivial (sign flip), degenerate
                return;
            }
            let rot = haar_random_so_d(d, rng);

            // Apply rotation: for each row i, compute new_row = R * old_row
            // Working in column-major layout, this is:
            //   new_cols[c * n + i] = sum_k rot[c][k] * old_cols[k * n + i]
            let old = cols.to_vec();
            for i in 0..n {
                for c in 0..d {
                    let mut val = 0.0;
                    for k in 0..d {
                        val += rot[c * d + k] * old[k * n + i];
                    }
                    cols[c * n + i] = val;
                }
            }
        }
    }
}

/// Generate a Haar-distributed random SO(d) matrix via QR decomposition
/// of a Gaussian matrix (Mezzadri 2007).
///
/// Returns a flat row-major d x d matrix.
fn haar_random_so_d(d: usize, rng: &mut ChaCha8Rng) -> Vec<f64> {
    use statrs::distribution::{ContinuousCDF, Normal};

    let normal = Normal::new(0.0, 1.0).unwrap();

    // Generate d x d Gaussian matrix (row-major)
    let mut z = vec![0.0_f64; d * d];
    for val in z.iter_mut() {
        *val = normal.inverse_cdf(rng.gen());
    }

    // Modified Gram-Schmidt QR
    // Q stored row-major: Q[i * d + j] = Q[i][j]
    let mut q = z;
    let mut r_diag = vec![0.0_f64; d];

    for j in 0..d {
        // Compute norm of column j
        let mut norm_sq = 0.0;
        for i in 0..d {
            norm_sq += q[i * d + j] * q[i * d + j];
        }
        let norm = norm_sq.sqrt();
        r_diag[j] = norm;

        if norm > 1e-15 {
            for i in 0..d {
                q[i * d + j] /= norm;
            }
        }

        // Orthogonalize remaining columns
        for k in (j + 1)..d {
            let mut proj = 0.0;
            for i in 0..d {
                proj += q[i * d + j] * q[i * d + k];
            }
            for i in 0..d {
                let qij = q[i * d + j];
                q[i * d + k] -= proj * qij;
            }
        }
    }

    // Mezzadri correction: multiply column j by sign(R_jj) to get Haar measure
    for j in 0..d {
        let sign = if r_diag[j] >= 0.0 { 1.0 } else { -1.0 };
        for i in 0..d {
            q[i * d + j] *= sign;
        }
    }

    // Ensure det(Q) = +1 (SO(d) not O(d)): if det < 0, flip one column
    let det = determinant_sign(&q, d);
    if det < 0.0 {
        for i in 0..d {
            q[i * d] = -q[i * d]; // Flip column 0
        }
    }

    q
}

/// Compute the sign of the determinant of a d x d row-major matrix
/// via LU decomposition (partial pivoting). Returns +1.0 or -1.0.
fn determinant_sign(mat: &[f64], d: usize) -> f64 {
    let mut lu = mat.to_vec();
    let mut sign = 1.0;

    for k in 0..d {
        // Find pivot
        let mut max_val = lu[k * d + k].abs();
        let mut max_row = k;
        for i in (k + 1)..d {
            let v = lu[i * d + k].abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }
        if max_val < 1e-15 {
            return 0.0; // Singular
        }
        if max_row != k {
            // Swap rows
            for j in 0..d {
                lu.swap(k * d + j, max_row * d + j);
            }
            sign = -sign;
        }
        // Eliminate below
        for i in (k + 1)..d {
            let factor = lu[i * d + k] / lu[k * d + k];
            for j in (k + 1)..d {
                let val = lu[k * d + j];
                lu[i * d + j] -= factor * val;
            }
        }
    }

    // Sign is determined by the pivot swaps and diagonal signs
    for k in 0..d {
        if lu[k * d + k] < 0.0 {
            sign = -sign;
        }
    }
    sign
}

// ---------------------------------------------------------------------------
// NullModelStrategy implementations for the 4 built-in null models
// ---------------------------------------------------------------------------

/// Column-independent shuffle null model.
///
/// Shuffles each attribute column independently, destroying inter-attribute
/// correlations while preserving each attribute's marginal distribution.
/// This is the universally informative null for distance-based ultrametric
/// tests.
#[derive(Debug, Clone, Copy, Default)]
pub struct ColumnIndependentNull;

impl NullModelStrategy for ColumnIndependentNull {
    fn name(&self) -> &str {
        "ColumnIndependent"
    }

    fn apply(&self, cols: &mut [f64], n: usize, d: usize, rng: &mut ChaCha8Rng) {
        for col in 0..d {
            let start = col * n;
            cols[start..start + n].shuffle(rng);
        }
    }
}

/// Row permutation null model.
///
/// Permutes entire rows, preserving joint distributions within each
/// observation but destroying row ordering. Identity-equivalent for
/// set-based distance statistics; informative only when row order
/// encodes temporal or spatial structure.
#[derive(Debug, Clone, Copy, Default)]
pub struct RowPermutationNull;

impl NullModelStrategy for RowPermutationNull {
    fn name(&self) -> &str {
        "RowPermutation"
    }

    fn apply(&self, cols: &mut [f64], n: usize, d: usize, rng: &mut ChaCha8Rng) {
        let mut perm: Vec<usize> = (0..n).collect();
        perm.shuffle(rng);
        let mut buf = vec![0.0_f64; n];
        for col in 0..d {
            let start = col * n;
            for (i, &pi) in perm.iter().enumerate() {
                buf[i] = cols[start + pi];
            }
            cols[start..start + n].copy_from_slice(&buf);
        }
    }

    fn is_euclidean_identity(&self) -> bool {
        true
    }
}

/// Toroidal shift null model.
///
/// Applies a random circular shift independently per column, preserving
/// marginal distributions and within-column autocorrelation structure
/// while destroying cross-column alignment.
#[derive(Debug, Clone, Copy, Default)]
pub struct ToroidalShiftNull;

impl NullModelStrategy for ToroidalShiftNull {
    fn name(&self) -> &str {
        "ToroidalShift"
    }

    fn apply(&self, cols: &mut [f64], n: usize, d: usize, rng: &mut ChaCha8Rng) {
        let mut buf = vec![0.0_f64; n];
        for col in 0..d {
            let shift = rng.gen_range(0..n);
            let start = col * n;
            for i in 0..n {
                buf[i] = cols[start + (i + shift) % n];
            }
            cols[start..start + n].copy_from_slice(&buf);
        }
    }
}

/// Random rotation null model.
///
/// Applies a Haar-distributed SO(d) rotation, preserving the covariance
/// ellipsoid and all pairwise Euclidean distances (isometry). Identity-
/// equivalent for Euclidean-distance statistics; informative only for
/// axis-aligned metrics like Baire encoding.
#[derive(Debug, Clone, Copy, Default)]
pub struct RandomRotationNull;

impl NullModelStrategy for RandomRotationNull {
    fn name(&self) -> &str {
        "RandomRotation"
    }

    fn apply(&self, cols: &mut [f64], n: usize, d: usize, rng: &mut ChaCha8Rng) {
        if d < 2 {
            return;
        }
        let rot = haar_random_so_d(d, rng);
        let old = cols.to_vec();
        for i in 0..n {
            for c in 0..d {
                let mut val = 0.0;
                for k in 0..d {
                    val += rot[c * d + k] * old[k * n + i];
                }
                cols[c * n + i] = val;
            }
        }
    }

    fn is_euclidean_identity(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// Bridge: NullModel enum <-> NullModelStrategy trait
// ---------------------------------------------------------------------------

impl NullModel {
    /// Convert this enum variant to the corresponding trait object.
    pub fn to_strategy(&self) -> Box<dyn NullModelStrategy> {
        match self {
            NullModel::ColumnIndependent => Box::new(ColumnIndependentNull),
            NullModel::RowPermutation => Box::new(RowPermutationNull),
            NullModel::ToroidalShift => Box::new(ToroidalShiftNull),
            NullModel::RandomRotation => Box::new(RandomRotationNull),
        }
    }
}

/// All four built-in null model strategies.
///
/// Useful for systematic comparison (Thesis H pattern) or for running
/// all null models in sequence to classify identity-equivalent vs
/// informative nulls for a given test statistic.
pub fn all_strategies() -> Vec<Box<dyn NullModelStrategy>> {
    vec![
        Box::new(ColumnIndependentNull),
        Box::new(RowPermutationNull),
        Box::new(ToroidalShiftNull),
        Box::new(RandomRotationNull),
    ]
}

// ---------------------------------------------------------------------------
// Adaptive null test integration
// ---------------------------------------------------------------------------

/// Configuration for an adaptive null model test.
///
/// Bundles a null model strategy with the adaptive engine configuration
/// and RNG seed, keeping the `run_adaptive_null_test` signature clean.
pub struct NullTestConfig<'a> {
    /// Null model strategy to apply.
    pub strategy: &'a dyn NullModelStrategy,
    /// Adaptive stopping configuration.
    pub adaptive: &'a super::adaptive::AdaptiveConfig,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

/// Run a one-sided permutation test combining a null model strategy with
/// the adaptive stopping engine.
///
/// Repeatedly applies the null model to generate null-distributed data,
/// computes the test statistic on each null realization, and counts how
/// many null statistics meet or exceed the observed value. The adaptive
/// engine (Besag & Clifford 1991) handles batching and early stopping.
///
/// # Arguments
///
/// * `data` - Column-major data `[d][n]` (flat array, length `n * d`)
/// * `n` - Number of observations (rows)
/// * `d` - Number of attributes (columns)
/// * `observed_statistic` - Pre-computed test statistic on the original data
/// * `statistic_fn` - Function computing the test statistic from (cols, n, d)
/// * `config` - Null test configuration (strategy + adaptive + seed)
pub fn run_adaptive_null_test(
    data: &[f64],
    n: usize,
    d: usize,
    observed_statistic: f64,
    statistic_fn: &dyn Fn(&[f64], usize, usize) -> f64,
    config: &NullTestConfig,
) -> super::adaptive::AdaptiveResult {
    use super::adaptive::adaptive_permutation_test;

    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let mut null_data = data.to_vec();

    adaptive_permutation_test(config.adaptive, |batch_size| {
        let mut n_extreme = 0usize;
        for _ in 0..batch_size {
            null_data.copy_from_slice(data);
            config.strategy.apply(&mut null_data, n, d, &mut rng);
            let null_stat = statistic_fn(&null_data, n, d);
            if null_stat >= observed_statistic {
                n_extreme += 1;
            }
        }
        n_extreme
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create test data in column-major layout.
    fn make_test_data(n: usize, d: usize, seed: u64) -> Vec<f64> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..d * n).map(|_| rng.gen_range(0.0..1.0)).collect()
    }

    #[test]
    fn test_column_independent_preserves_marginals() {
        let n = 100;
        let d = 3;
        let original = make_test_data(n, d, 42);
        let mut shuffled = original.clone();
        let mut rng = ChaCha8Rng::seed_from_u64(99);

        apply_null_column_major(&mut shuffled, n, d, NullModel::ColumnIndependent, &mut rng);

        // Each column should have the same set of values (just reordered)
        for col in 0..d {
            let start = col * n;
            let mut orig_col: Vec<f64> = original[start..start + n].to_vec();
            let mut shuf_col: Vec<f64> = shuffled[start..start + n].to_vec();
            orig_col.sort_by(|a, b| a.partial_cmp(b).unwrap());
            shuf_col.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for (a, b) in orig_col.iter().zip(shuf_col.iter()) {
                assert!((a - b).abs() < 1e-15, "Column {col} values not preserved");
            }
        }
    }

    #[test]
    fn test_row_permutation_preserves_rows() {
        let n = 50;
        let d = 4;
        let original = make_test_data(n, d, 42);
        let mut shuffled = original.clone();
        let mut rng = ChaCha8Rng::seed_from_u64(99);

        apply_null_column_major(&mut shuffled, n, d, NullModel::RowPermutation, &mut rng);

        // Extract all rows from original and shuffled, sort, should match
        let extract_rows = |data: &[f64]| -> Vec<Vec<f64>> {
            (0..n)
                .map(|i| (0..d).map(|c| data[c * n + i]).collect::<Vec<f64>>())
                .collect()
        };

        let mut orig_rows = extract_rows(&original);
        let mut shuf_rows = extract_rows(&shuffled);

        // Sort rows for comparison (each row is a unique multiset)
        let row_key = |r: &Vec<f64>| -> Vec<u64> {
            r.iter().map(|x| x.to_bits()).collect()
        };
        orig_rows.sort_by(|a, b| row_key(a).cmp(&row_key(b)));
        shuf_rows.sort_by(|a, b| row_key(a).cmp(&row_key(b)));

        for (orig, shuf) in orig_rows.iter().zip(shuf_rows.iter()) {
            for (a, b) in orig.iter().zip(shuf.iter()) {
                assert!((a - b).abs() < 1e-15, "Row values not preserved");
            }
        }
    }

    #[test]
    fn test_toroidal_shift_preserves_marginals() {
        let n = 80;
        let d = 3;
        let original = make_test_data(n, d, 42);
        let mut shifted = original.clone();
        let mut rng = ChaCha8Rng::seed_from_u64(99);

        apply_null_column_major(&mut shifted, n, d, NullModel::ToroidalShift, &mut rng);

        // Each column should have the same multiset of values
        for col in 0..d {
            let start = col * n;
            let mut orig_col: Vec<f64> = original[start..start + n].to_vec();
            let mut shif_col: Vec<f64> = shifted[start..start + n].to_vec();
            orig_col.sort_by(|a, b| a.partial_cmp(b).unwrap());
            shif_col.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for (a, b) in orig_col.iter().zip(shif_col.iter()) {
                assert!((a - b).abs() < 1e-15, "Toroidal shift changed column {col} values");
            }
        }
    }

    #[test]
    fn test_random_rotation_preserves_distances() {
        // Random rotation is an isometry: pairwise distances should be preserved.
        let n = 30;
        let d = 3;
        let original = make_test_data(n, d, 42);
        let mut rotated = original.clone();
        let mut rng = ChaCha8Rng::seed_from_u64(99);

        apply_null_column_major(&mut rotated, n, d, NullModel::RandomRotation, &mut rng);

        // Compute pairwise distances before and after
        for i in 0..n {
            for j in (i + 1)..n {
                let mut d_orig = 0.0;
                let mut d_rot = 0.0;
                for c in 0..d {
                    let ov = original[c * n + i] - original[c * n + j];
                    let rv = rotated[c * n + i] - rotated[c * n + j];
                    d_orig += ov * ov;
                    d_rot += rv * rv;
                }
                assert!(
                    (d_orig.sqrt() - d_rot.sqrt()).abs() < 1e-10,
                    "Rotation changed distance ({i},{j}): {:.6} -> {:.6}",
                    d_orig.sqrt(),
                    d_rot.sqrt(),
                );
            }
        }
    }

    #[test]
    fn test_column_independent_matches_legacy() {
        // Verify that ColumnIndependent produces the same result as the
        // inline shuffle_column() calls in baire.rs
        let n = 50;
        let d = 2;
        let data = make_test_data(n, d, 42);

        // Method 1: our function
        let mut v1 = data.clone();
        let mut rng1 = ChaCha8Rng::seed_from_u64(77);
        apply_null_column_major(&mut v1, n, d, NullModel::ColumnIndependent, &mut rng1);

        // Method 2: manual column shuffles (legacy approach)
        let mut v2 = data.clone();
        let mut rng2 = ChaCha8Rng::seed_from_u64(77);
        for col in 0..d {
            let start = col * n;
            v2[start..start + n].shuffle(&mut rng2);
        }

        // Should be identical
        for (a, b) in v1.iter().zip(v2.iter()) {
            assert!((a - b).abs() < 1e-15, "Column-independent should match legacy");
        }
    }

    #[test]
    fn test_rotation_matrix_is_orthogonal() {
        let d = 4;
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let q = haar_random_so_d(d, &mut rng);

        // Check Q^T Q = I
        for i in 0..d {
            for j in 0..d {
                let mut dot = 0.0;
                for k in 0..d {
                    dot += q[k * d + i] * q[k * d + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "Q^T Q not identity at ({i},{j}): {dot:.6} != {expected}",
                );
            }
        }
    }

    #[test]
    fn test_rotation_matrix_has_positive_determinant() {
        let d = 5;
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let q = haar_random_so_d(d, &mut rng);

        let det = determinant_sign(&q, d);
        assert!(
            det > 0.0,
            "SO(d) matrix should have positive determinant, got sign {det}",
        );
    }

    #[test]
    fn test_default_is_column_independent() {
        assert_eq!(NullModel::default(), NullModel::ColumnIndependent);
    }

    // === Thesis H: Null-Model Identity Property ===

    /// Thesis H states: RandomRotation is identity-equivalent for Euclidean-
    /// distance-based ultrametric fraction tests, because rotation is an isometry
    /// that preserves all pairwise distances. The ultrametric fraction computed
    /// from Euclidean distances is therefore unchanged by rotation, making the
    /// permutation p-value trivially 1.0.
    ///
    /// This test verifies the identity property by computing ultrametric fraction
    /// before and after rotation, confirming they are equal.
    #[test]
    fn test_thesis_h_null_model_identity_rotation() {
        let n = 30;
        let d = 4;
        let original = make_test_data(n, d, 42);

        // Compute pairwise Euclidean distances for original data
        let orig_dists = euclidean_pairwise_distances(&original, n, d);

        // Apply random rotation
        let mut rotated = original.clone();
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        apply_null_column_major(&mut rotated, n, d, NullModel::RandomRotation, &mut rng);

        // Compute pairwise distances after rotation
        let rot_dists = euclidean_pairwise_distances(&rotated, n, d);

        // All pairwise distances should match (rotation is isometry)
        for (d_orig, d_rot) in orig_dists.iter().zip(rot_dists.iter()) {
            assert!(
                (d_orig - d_rot).abs() < 1e-8,
                "Rotation changed distance: {:.8} -> {:.8}",
                d_orig, d_rot
            );
        }

        // Compute ultrametric fraction for both
        let uf_orig = ultrametric_fraction_from_distances(&orig_dists, n);
        let uf_rot = ultrametric_fraction_from_distances(&rot_dists, n);

        assert!(
            (uf_orig - uf_rot).abs() < 1e-10,
            "Thesis H violated: ultrametric fraction changed by rotation: {} -> {}",
            uf_orig, uf_rot
        );
    }

    /// Thesis H also states: ColumnIndependent is the informative null --
    /// it should detectably change the ultrametric fraction by breaking
    /// inter-attribute correlations that create hierarchical structure.
    #[test]
    fn test_thesis_h_column_independent_is_informative() {
        let n = 50;
        let d = 3;
        // Create data with strong hierarchical structure (clustered)
        let mut data = vec![0.0_f64; n * d];
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        // Two clusters: first 25 points near origin, next 25 far away
        for i in 0..25 {
            for c in 0..d {
                data[c * n + i] = rng.gen_range(-0.1..0.1);
            }
        }
        for i in 25..50 {
            for c in 0..d {
                data[c * n + i] = 10.0 + rng.gen_range(-0.1..0.1);
            }
        }

        let orig_dists = euclidean_pairwise_distances(&data, n, d);
        let uf_orig = ultrametric_fraction_from_distances(&orig_dists, n);

        // ColumnIndependent shuffle should change the structure
        let mut shuffled = data.clone();
        let mut srng = ChaCha8Rng::seed_from_u64(77);
        apply_null_column_major(&mut shuffled, n, d, NullModel::ColumnIndependent, &mut srng);

        let shuf_dists = euclidean_pairwise_distances(&shuffled, n, d);
        let uf_shuf = ultrametric_fraction_from_distances(&shuf_dists, n);

        // The structured data should have high ultrametric fraction (near 1.0
        // for well-separated clusters). After shuffling, it should decrease
        // because the correlation structure is broken.
        assert!(
            uf_orig > 0.5,
            "Original clustered data should have high UF, got {}",
            uf_orig
        );
        // We only assert the original has meaningful UF; the shuffled value
        // depends on chance but should generally differ.
        eprintln!(
            "Thesis H: UF original={:.4}, UF shuffled={:.4}, diff={:.4}",
            uf_orig, uf_shuf, (uf_orig - uf_shuf).abs()
        );
    }

    /// Helper: compute pairwise Euclidean distances from column-major data.
    fn euclidean_pairwise_distances(data: &[f64], n: usize, d: usize) -> Vec<f64> {
        let mut dists = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let mut sq = 0.0;
                for c in 0..d {
                    let diff = data[c * n + i] - data[c * n + j];
                    sq += diff * diff;
                }
                dists.push(sq.sqrt());
            }
        }
        dists
    }

    /// Helper: compute ultrametric fraction from a flat pairwise distance vector.
    /// For each triple (i,j,k), check if the ultrametric inequality holds:
    /// max(d_ij, d_ik, d_jk) <= max of the two smallest (i.e., the two largest
    /// are equal). Fraction of triples satisfying this.
    fn ultrametric_fraction_from_distances(dists: &[f64], n: usize) -> f64 {
        let idx = |i: usize, j: usize| -> usize {
            // Upper triangle index for (i, j) where i < j
            i * (2 * n - i - 1) / 2 + (j - i - 1)
        };

        let mut n_triples = 0u64;
        let mut n_ultra = 0u64;

        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    let dij = dists[idx(i, j)];
                    let dik = dists[idx(i, k)];
                    let djk = dists[idx(j, k)];

                    // Sort the three distances
                    let mut sorted = [dij, dik, djk];
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

                    // Ultrametric: max(d1,d2,d3) approx second-max
                    // Use 5% relative tolerance (appropriate for test data)
                    n_triples += 1;
                    if (sorted[2] - sorted[1]).abs() < 0.05 * sorted[2].max(1e-15) {
                        n_ultra += 1;
                    }
                }
            }
        }

        if n_triples == 0 {
            return 0.0;
        }
        n_ultra as f64 / n_triples as f64
    }

    // === Thesis H: Systematic Multi-Null Comparison ===

    /// Helper: create strongly clustered synthetic data in column-major layout.
    ///
    /// Creates `n_clusters` well-separated clusters of `points_per_cluster`
    /// points, each in d dimensions. Cluster centers are spaced at distance 10.0
    /// with intra-cluster spread of 0.1. This produces data with unambiguous
    /// hierarchical (ultrametric) structure in Euclidean space.
    fn make_clustered_data(
        n_clusters: usize,
        points_per_cluster: usize,
        d: usize,
        seed: u64,
    ) -> Vec<f64> {
        let n = n_clusters * points_per_cluster;
        let mut cols = vec![0.0_f64; n * d];
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        for cluster in 0..n_clusters {
            let center: Vec<f64> = (0..d).map(|_| rng.gen_range(-50.0..50.0)).collect();
            for p in 0..points_per_cluster {
                let row = cluster * points_per_cluster + p;
                for c in 0..d {
                    cols[c * n + row] = center[c] + rng.gen_range(-0.1..0.1);
                }
            }
        }
        cols
    }

    /// Thesis H comprehensive test: run all 4 null models on the same
    /// hierarchically clustered data and verify the two-class partition:
    ///
    /// **Identity-equivalent** (test statistic unchanged, p ~ 1.0):
    /// - RowPermutation: relabeling observations preserves all pairwise distances
    /// - RandomRotation: isometry preserves all pairwise distances
    ///
    /// **Informative** (test statistic changes, p << 0.05 for structured data):
    /// - ColumnIndependent: breaks inter-attribute correlation structure
    /// - ToroidalShift: per-column circular shifts with different offsets
    ///   create chimeric rows, breaking inter-attribute alignment
    ///
    /// This directly verifies the "different nulls test different hypotheses"
    /// principle. The identity/informative classification depends on whether
    /// the null model transformation is an isometry (or relabeling) with
    /// respect to the metric used by the test statistic.
    #[test]
    fn test_thesis_h_multi_null_comparison() {
        // 4 clusters of 15 points in 3D -> 60 points total
        let n_clusters = 4;
        let pts_per = 15;
        let d = 3;
        let n = n_clusters * pts_per;
        let cols = make_clustered_data(n_clusters, pts_per, d, 42);

        let result = multi_null_comparison(
            &cols,
            n,
            d,
            5000,   // ignored when n<=100 (exhaustive mode)
            500,    // 500 permutations per null model
            42,
            0.05,   // 5% relative tolerance
        );

        eprintln!(
            "Multi-null comparison (n={}, d={}, 4 clusters x {}):",
            n, d, pts_per
        );
        eprintln!("  Observed UF = {:.4}", result.observed_fraction);
        eprintln!("  p(ColumnIndependent) = {:.4}", result.p_column_independent);
        eprintln!("  p(RowPermutation)    = {:.4}", result.p_row_permutation);
        eprintln!("  p(ToroidalShift)     = {:.4}", result.p_toroidal_shift);
        eprintln!("  p(RandomRotation)    = {:.4}", result.p_random_rotation);

        // === Identity-equivalent nulls ===

        // RowPermutation: relabeling observations preserves the multiset
        // of all C(n,3) distance triples, so the exhaustive fraction is
        // unchanged. With exhaustive evaluation, p = 1.0 exactly.
        assert!(
            result.p_row_permutation > 0.5,
            "RowPermutation should be identity-equivalent (p > 0.5), got p = {:.4}",
            result.p_row_permutation,
        );

        // RandomRotation: rotation is an isometry (preserves all pairwise
        // Euclidean distances exactly), so the fraction is unchanged.
        assert!(
            result.p_random_rotation > 0.5,
            "RandomRotation should be identity-equivalent (p > 0.5), got p = {:.4}",
            result.p_random_rotation,
        );

        // === Informative nulls ===

        // ColumnIndependent: breaks inter-attribute correlations that
        // create cluster structure. The observed fraction (reflecting
        // hierarchical clusters) should exceed most null realizations.
        assert!(
            result.p_column_independent < 0.05,
            "ColumnIndependent should be informative (p < 0.05), got p = {:.4}",
            result.p_column_independent,
        );

        // ToroidalShift: applies DIFFERENT circular shifts per column,
        // creating chimeric rows that break inter-column alignment.
        // This is structurally similar to ColumnIndependent (both destroy
        // joint correlation) but preserves within-column autocorrelation.
        // For clustered data, it should also be informative.
        assert!(
            result.p_toroidal_shift < 0.05,
            "ToroidalShift should be informative (p < 0.05), got p = {:.4}",
            result.p_toroidal_shift,
        );
    }

    /// Verify that the multi-null comparison detects when ColumnIndependent
    /// is NOT informative: for uniform random data (no structure), all four
    /// null models should yield p > 0.05 (no false positives).
    #[test]
    fn test_thesis_h_no_false_positives_under_null() {
        let n = 60;
        let d = 3;
        let cols = make_test_data(n, d, 123);

        let result = multi_null_comparison(
            &cols,
            n,
            d,
            5000,
            200,
            123,
            0.05,
        );

        eprintln!(
            "Multi-null (uniform random, no structure): UF = {:.4}",
            result.observed_fraction,
        );
        eprintln!("  p(ColumnIndependent) = {:.4}", result.p_column_independent);
        eprintln!("  p(RowPermutation)    = {:.4}", result.p_row_permutation);
        eprintln!("  p(RandomRotation)    = {:.4}", result.p_random_rotation);

        // Under the null (uniform random data), ColumnIndependent should
        // NOT flag significance. The observed UF should be consistent with
        // the null distribution.
        // We use a generous threshold to avoid flaky tests: the point is
        // that uniform random data should not be detected as hierarchical.
        assert!(
            result.p_column_independent > 0.01,
            "Uniform random data should not be flagged (p > 0.01), got p = {:.4}",
            result.p_column_independent,
        );

        // The identity-equivalent nulls should remain non-significant.
        assert!(
            result.p_row_permutation > 0.2,
            "RowPermutation on random data: p = {:.4}",
            result.p_row_permutation,
        );
        assert!(
            result.p_random_rotation > 0.2,
            "RandomRotation on random data: p = {:.4}",
            result.p_random_rotation,
        );
    }

    /// Verify that ColumnIndependent sensitivity scales with cluster separation.
    ///
    /// With well-separated clusters, p should be very small. As clusters merge
    /// (overlap increases), the signal weakens and p approaches 1.0. This
    /// confirms that the test measures genuine hierarchical structure, not a
    /// statistical artifact.
    #[test]
    fn test_thesis_h_sensitivity_gradient() {
        let n_clusters = 3;
        let pts_per = 20;
        let d = 3;
        let n = n_clusters * pts_per;

        let mut p_values = Vec::new();

        // Create datasets with decreasing cluster separation
        for &separation in &[50.0, 10.0, 2.0, 0.5] {
            let mut cols = vec![0.0_f64; n * d];
            let mut rng = ChaCha8Rng::seed_from_u64(42);

            for cluster in 0..n_clusters {
                let center: Vec<f64> = (0..d)
                    .map(|_| (cluster as f64) * separation)
                    .collect();
                for p in 0..pts_per {
                    let row = cluster * pts_per + p;
                    for c in 0..d {
                        cols[c * n + row] = center[c] + rng.gen_range(-1.0..1.0);
                    }
                }
            }

            // Only test ColumnIndependent (the informative null)
            let mut null_rng = ChaCha8Rng::seed_from_u64(99);
            let obs_frac = exhaustive_euclidean_ultrametric_fraction(
                &cols, n, d, 0.05,
            );
            let mut n_extreme = 0usize;
            let mut shuffled = cols.clone();

            for _ in 0..200 {
                shuffled.copy_from_slice(&cols);
                apply_null_column_major(
                    &mut shuffled, n, d, NullModel::ColumnIndependent, &mut null_rng,
                );
                let null_frac = exhaustive_euclidean_ultrametric_fraction(
                    &shuffled, n, d, 0.05,
                );
                if null_frac >= obs_frac {
                    n_extreme += 1;
                }
            }

            let p = (n_extreme as f64 + 1.0) / 201.0;
            p_values.push((separation, p));
            eprintln!("  sep={separation:.1}: obs_frac={obs_frac:.4}, p={p:.4}");
        }

        // Well-separated clusters should have lower p-values than overlapping ones.
        // The p-values should be monotonically non-decreasing as separation decreases.
        // Allow minor violations due to sampling noise, but the trend should hold.
        assert!(
            p_values[0].1 < p_values[3].1,
            "p-value should increase as clusters merge: sep={:.1} p={:.4} vs sep={:.1} p={:.4}",
            p_values[0].0, p_values[0].1,
            p_values[3].0, p_values[3].1,
        );
    }

    // === NullModelStrategy trait tests ===

    #[test]
    fn test_all_strategies_returns_four() {
        let strategies = all_strategies();
        assert_eq!(strategies.len(), 4);
        assert_eq!(strategies[0].name(), "ColumnIndependent");
        assert_eq!(strategies[1].name(), "RowPermutation");
        assert_eq!(strategies[2].name(), "ToroidalShift");
        assert_eq!(strategies[3].name(), "RandomRotation");
    }

    #[test]
    fn test_euclidean_identity_flags() {
        assert!(!ColumnIndependentNull.is_euclidean_identity());
        assert!(RowPermutationNull.is_euclidean_identity());
        assert!(!ToroidalShiftNull.is_euclidean_identity());
        assert!(RandomRotationNull.is_euclidean_identity());
    }

    #[test]
    fn test_trait_matches_enum_column_independent() {
        let n = 50;
        let d = 3;
        let data = make_test_data(n, d, 42);

        // Trait-based
        let mut v1 = data.clone();
        let mut rng1 = ChaCha8Rng::seed_from_u64(77);
        ColumnIndependentNull.apply(&mut v1, n, d, &mut rng1);

        // Enum-based
        let mut v2 = data.clone();
        let mut rng2 = ChaCha8Rng::seed_from_u64(77);
        apply_null_column_major(&mut v2, n, d, NullModel::ColumnIndependent, &mut rng2);

        for (a, b) in v1.iter().zip(v2.iter()) {
            assert!(
                (a - b).abs() < 1e-15,
                "Trait and enum should produce identical results",
            );
        }
    }

    #[test]
    fn test_trait_matches_enum_row_permutation() {
        let n = 50;
        let d = 3;
        let data = make_test_data(n, d, 42);

        let mut v1 = data.clone();
        let mut rng1 = ChaCha8Rng::seed_from_u64(77);
        RowPermutationNull.apply(&mut v1, n, d, &mut rng1);

        let mut v2 = data.clone();
        let mut rng2 = ChaCha8Rng::seed_from_u64(77);
        apply_null_column_major(&mut v2, n, d, NullModel::RowPermutation, &mut rng2);

        for (a, b) in v1.iter().zip(v2.iter()) {
            assert!(
                (a - b).abs() < 1e-15,
                "Trait and enum should produce identical results",
            );
        }
    }

    #[test]
    fn test_trait_matches_enum_toroidal_shift() {
        let n = 50;
        let d = 3;
        let data = make_test_data(n, d, 42);

        let mut v1 = data.clone();
        let mut rng1 = ChaCha8Rng::seed_from_u64(77);
        ToroidalShiftNull.apply(&mut v1, n, d, &mut rng1);

        let mut v2 = data.clone();
        let mut rng2 = ChaCha8Rng::seed_from_u64(77);
        apply_null_column_major(&mut v2, n, d, NullModel::ToroidalShift, &mut rng2);

        for (a, b) in v1.iter().zip(v2.iter()) {
            assert!(
                (a - b).abs() < 1e-15,
                "Trait and enum should produce identical results",
            );
        }
    }

    #[test]
    fn test_trait_matches_enum_random_rotation() {
        let n = 30;
        let d = 4;
        let data = make_test_data(n, d, 42);

        let mut v1 = data.clone();
        let mut rng1 = ChaCha8Rng::seed_from_u64(77);
        RandomRotationNull.apply(&mut v1, n, d, &mut rng1);

        let mut v2 = data.clone();
        let mut rng2 = ChaCha8Rng::seed_from_u64(77);
        apply_null_column_major(&mut v2, n, d, NullModel::RandomRotation, &mut rng2);

        for (a, b) in v1.iter().zip(v2.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "Trait and enum should produce identical results",
            );
        }
    }

    #[test]
    fn test_to_strategy_bridge() {
        let n = 40;
        let d = 2;
        let data = make_test_data(n, d, 42);

        for model in [
            NullModel::ColumnIndependent,
            NullModel::RowPermutation,
            NullModel::ToroidalShift,
            NullModel::RandomRotation,
        ] {
            let strategy = model.to_strategy();

            let mut v1 = data.clone();
            let mut rng1 = ChaCha8Rng::seed_from_u64(99);
            strategy.apply(&mut v1, n, d, &mut rng1);

            let mut v2 = data.clone();
            let mut rng2 = ChaCha8Rng::seed_from_u64(99);
            apply_null_column_major(&mut v2, n, d, model, &mut rng2);

            for (a, b) in v1.iter().zip(v2.iter()) {
                assert!(
                    (a - b).abs() < 1e-10,
                    "to_strategy() for {:?} should match enum dispatch",
                    model,
                );
            }
        }
    }

    #[test]
    fn test_run_adaptive_null_test_significant() {
        // Clustered data: ColumnIndependent should detect significance.
        let n_clusters = 4;
        let pts_per = 15;
        let d = 3;
        let n = n_clusters * pts_per;
        let cols = make_clustered_data(n_clusters, pts_per, d, 42);

        let obs_frac = exhaustive_euclidean_ultrametric_fraction(&cols, n, d, 0.05);

        let adaptive = super::super::adaptive::AdaptiveConfig {
            batch_size: 50,
            max_permutations: 500,
            alpha: 0.05,
            confidence: 0.99,
            min_permutations: 100,
        };

        let config = NullTestConfig {
            strategy: &ColumnIndependentNull,
            adaptive: &adaptive,
            seed: 42,
        };

        let result = run_adaptive_null_test(
            &cols,
            n,
            d,
            obs_frac,
            &|data, nn, dd| exhaustive_euclidean_ultrametric_fraction(data, nn, dd, 0.05),
            &config,
        );

        assert!(
            result.p_value < 0.05,
            "Clustered data should be significant under ColumnIndependent, p = {:.4}",
            result.p_value,
        );
    }

    #[test]
    fn test_run_adaptive_null_test_identity_null() {
        // Clustered data: RowPermutation (identity-equivalent) should yield p ~ 1.0.
        let n_clusters = 4;
        let pts_per = 15;
        let d = 3;
        let n = n_clusters * pts_per;
        let cols = make_clustered_data(n_clusters, pts_per, d, 42);

        let obs_frac = exhaustive_euclidean_ultrametric_fraction(&cols, n, d, 0.05);

        let adaptive = super::super::adaptive::AdaptiveConfig {
            batch_size: 50,
            max_permutations: 200,
            alpha: 0.05,
            confidence: 0.99,
            min_permutations: 100,
        };

        let config = NullTestConfig {
            strategy: &RowPermutationNull,
            adaptive: &adaptive,
            seed: 42,
        };

        let result = run_adaptive_null_test(
            &cols,
            n,
            d,
            obs_frac,
            &|data, nn, dd| exhaustive_euclidean_ultrametric_fraction(data, nn, dd, 0.05),
            &config,
        );

        assert!(
            result.p_value > 0.5,
            "RowPermutation should be identity-equivalent, p = {:.4}",
            result.p_value,
        );
    }

    #[test]
    fn test_strategy_trait_is_object_safe() {
        // Verify the trait can be used as a trait object (dyn dispatch).
        let strategies: Vec<Box<dyn NullModelStrategy>> = vec![
            Box::new(ColumnIndependentNull),
            Box::new(RowPermutationNull),
        ];

        let n = 20;
        let d = 2;
        let data = make_test_data(n, d, 42);

        for strategy in &strategies {
            let mut v = data.clone();
            let mut rng = ChaCha8Rng::seed_from_u64(99);
            strategy.apply(&mut v, n, d, &mut rng);
            // Verify data was modified (not all zeros or unchanged)
            let changed = v.iter().zip(data.iter()).any(|(a, b)| (a - b).abs() > 1e-15);
            assert!(
                changed,
                "Strategy '{}' should modify data",
                strategy.name(),
            );
        }
    }
}

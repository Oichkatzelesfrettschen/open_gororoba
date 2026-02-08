//! Baire metric for multi-attribute ultrametric analysis.
//!
//! Encodes multi-attribute tuples as digit sequences and computes the
//! Baire distance: d(x,y) = p^(-k) where k is the length of the longest
//! common prefix in the digit representations.
//!
//! This approach (Murtagh arXiv:1104.4063) tests whether multi-dimensional
//! astrophysical catalogs (pulsars, magnetars, FRBs, GW sources) have
//! intrinsic hierarchical structure when viewed through the lens of their
//! combined physical parameters.
//!
//! # Algorithm
//!
//! 1. Normalize each attribute to [0, 1] (linear or log scale)
//! 2. Quantize to `n_digits` digits in base `base`
//! 3. Interleave digits from all attributes into a single sequence
//! 4. Baire distance = base^(-first_differing_position)
//! 5. Apply standard ultrametric fraction test on the Baire distance matrix
//!
//! # References
//!
//! - Murtagh (2004): arXiv:1104.4063 (On ultrametricity, data coding, and computation)
//! - Murtagh & Contreras (2012): Algorithms for hierarchical clustering

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use super::null_models::{NullModel, apply_null_column_major};

/// Specification for a single attribute in the Baire encoding.
#[derive(Debug, Clone)]
pub struct AttributeSpec {
    /// Human-readable name.
    pub name: String,
    /// Minimum value (for normalization).
    pub min: f64,
    /// Maximum value (for normalization).
    pub max: f64,
    /// Whether to apply log10 before normalizing (for quantities
    /// spanning orders of magnitude, e.g., period, luminosity).
    pub log_scale: bool,
}

/// Encoder that converts multi-attribute objects into digit sequences
/// and computes Baire distances.
#[derive(Debug, Clone)]
pub struct BaireEncoder {
    /// Attribute specifications (one per dimension).
    pub attributes: Vec<AttributeSpec>,
    /// Base for digit representation (default: 10).
    pub base: u64,
    /// Number of digits per attribute (precision).
    pub n_digits: usize,
}

impl BaireEncoder {
    /// Create a new Baire encoder.
    pub fn new(attributes: Vec<AttributeSpec>, base: u64, n_digits: usize) -> Self {
        assert!(base >= 2, "Base must be >= 2");
        assert!(n_digits >= 1, "Need at least 1 digit");
        Self {
            attributes,
            base,
            n_digits,
        }
    }

    /// Encode a single object's attributes into a digit sequence.
    ///
    /// `values`: one f64 per attribute, in the same order as `self.attributes`.
    /// Returns: interleaved digit sequence of length n_attributes * n_digits.
    pub fn encode(&self, values: &[f64]) -> Vec<u64> {
        assert_eq!(values.len(), self.attributes.len());

        // Normalize and quantize each attribute
        let mut per_attr_digits: Vec<Vec<u64>> = Vec::with_capacity(self.attributes.len());

        for (spec, &val) in self.attributes.iter().zip(values.iter()) {
            let normalized = self.normalize(val, spec);
            let digits = self.quantize(normalized);
            per_attr_digits.push(digits);
        }

        // Interleave: digit 0 of attr 0, digit 0 of attr 1, ...,
        //             digit 1 of attr 0, digit 1 of attr 1, ...
        let n_attrs = self.attributes.len();
        let total_digits = n_attrs * self.n_digits;
        let mut interleaved = Vec::with_capacity(total_digits);

        for d in 0..self.n_digits {
            for attr_digits in &per_attr_digits {
                interleaved.push(attr_digits[d]);
            }
        }

        interleaved
    }

    /// Compute the Baire distance between two digit sequences.
    ///
    /// d(x,y) = base^(-k) where k is the 1-indexed position of the first
    /// differing digit. If sequences are identical, distance is 0.
    pub fn baire_distance(&self, seq_a: &[u64], seq_b: &[u64]) -> f64 {
        assert_eq!(seq_a.len(), seq_b.len());

        for (k, (&a, &b)) in seq_a.iter().zip(seq_b.iter()).enumerate() {
            if a != b {
                // k is 0-indexed, so first-differing position is k+1
                return (self.base as f64).powi(-((k + 1) as i32));
            }
        }

        0.0 // Identical sequences
    }

    /// Normalize a value to [0, 1] using the attribute specification.
    pub fn normalize(&self, val: f64, spec: &AttributeSpec) -> f64 {
        if val.is_nan() || val.is_infinite() {
            return 0.5; // Map missing/invalid to midpoint
        }

        let v = if spec.log_scale {
            if val <= 0.0 {
                spec.min.log10() // Clamp to min
            } else {
                val.log10()
            }
        } else {
            val
        };

        let min = if spec.log_scale && spec.min > 0.0 {
            spec.min.log10()
        } else {
            spec.min
        };
        let max = if spec.log_scale && spec.max > 0.0 {
            spec.max.log10()
        } else {
            spec.max
        };

        if (max - min).abs() < 1e-30 {
            return 0.5;
        }

        ((v - min) / (max - min)).clamp(0.0, 1.0)
    }

    /// Quantize a [0,1] value into `n_digits` digits in the given base.
    fn quantize(&self, val: f64) -> Vec<u64> {
        let mut digits = Vec::with_capacity(self.n_digits);
        let mut remainder = val.clamp(0.0, 1.0 - 1e-15);

        for _ in 0..self.n_digits {
            let scaled = remainder * self.base as f64;
            let digit = scaled.floor() as u64;
            let digit = digit.min(self.base - 1); // Clamp to valid range
            digits.push(digit);
            remainder = scaled - digit as f64;
        }

        digits
    }
}

/// Compute the full Baire distance matrix for a set of objects.
///
/// `data`: each row is one object, each column is one attribute value.
/// Returns a flat upper-triangle distance matrix.
pub fn baire_distance_matrix(
    encoder: &BaireEncoder,
    data: &[Vec<f64>],
) -> Vec<f64> {
    let n = data.len();
    let n_pairs = n * (n - 1) / 2;

    // Encode all objects
    let encoded: Vec<Vec<u64>> = data.iter().map(|row| encoder.encode(row)).collect();

    // Compute pairwise distances
    let mut dists = Vec::with_capacity(n_pairs);
    for i in 0..n {
        for j in (i + 1)..n {
            dists.push(encoder.baire_distance(&encoded[i], &encoded[j]));
        }
    }

    dists
}

/// Compute normalized Euclidean distance matrix in multi-attribute space.
///
/// Each attribute is normalized to [0,1] using the encoder's attribute specs
/// (respecting log_scale settings), then standard Euclidean distance is computed.
/// Returns a flat upper-triangle distance matrix.
pub fn euclidean_distance_matrix(
    encoder: &BaireEncoder,
    data: &[Vec<f64>],
) -> Vec<f64> {
    let n = data.len();
    let n_pairs = n * (n - 1) / 2;
    let n_attrs = encoder.attributes.len();

    // Normalize all rows
    let normalized: Vec<Vec<f64>> = data.iter().map(|row| {
        (0..n_attrs).map(|a| encoder.normalize(row[a], &encoder.attributes[a]))
            .collect()
    }).collect();

    let mut dists = Vec::with_capacity(n_pairs);
    for i in 0..n {
        for j in (i + 1)..n {
            let d2: f64 = (0..n_attrs)
                .map(|a| (normalized[i][a] - normalized[j][a]).powi(2))
                .sum();
            dists.push(d2.sqrt());
        }
    }

    dists
}

/// Test ultrametric fraction on Euclidean distances in multi-attribute space.
///
/// This is the scientifically meaningful test: does the multi-attribute
/// parameter space exhibit ultrametric structure under the standard metric?
/// The Baire encoding is used only for normalization (log-scale, ranges).
///
/// Null hypothesis: shuffle each column independently to break inter-attribute
/// correlations, recompute Euclidean distances, and measure ultrametric fraction.
///
/// Uses a **one-sided** p-value: counts null fractions >= observed.
/// This tests whether observed ultrametricity is *higher* than chance.
pub fn euclidean_ultrametric_test(
    encoder: &BaireEncoder,
    data: &[Vec<f64>],
    n_triples: usize,
    n_permutations: usize,
    seed: u64,
) -> BaireTestResult {
    let n = data.len();
    assert!(n >= 3, "Need at least 3 objects");

    let dist_matrix = euclidean_distance_matrix(encoder, data);

    // Observed ultrametric fraction
    let obs_frac = super::ultrametric_fraction_from_matrix(
        &dist_matrix, n, n_triples, seed,
    );

    // Null: shuffle each column independently
    let mut rng = ChaCha8Rng::seed_from_u64(seed + 1_000_000);
    let mut null_fracs = Vec::with_capacity(n_permutations);
    let mut shuffled_data = data.to_vec();

    for _ in 0..n_permutations {
        for col in 0..encoder.attributes.len() {
            let mut col_values: Vec<f64> = shuffled_data.iter().map(|row| row[col]).collect();
            col_values.shuffle(&mut rng);
            for (i, &v) in col_values.iter().enumerate() {
                shuffled_data[i][col] = v;
            }
        }

        let null_dists = euclidean_distance_matrix(encoder, &shuffled_data);
        let null_frac = super::ultrametric_fraction_from_matrix(
            &null_dists, n, n_triples, seed + 2_000_000,
        );
        null_fracs.push(null_frac);
    }

    let null_mean = null_fracs.iter().sum::<f64>() / n_permutations as f64;
    let null_var = null_fracs
        .iter()
        .map(|f| (f - null_mean).powi(2))
        .sum::<f64>()
        / n_permutations as f64;
    let null_std = null_var.sqrt();

    let n_extreme = null_fracs
        .iter()
        .filter(|&&f| f >= obs_frac)
        .count();
    let p_value = (n_extreme as f64 + 1.0) / (n_permutations as f64 + 1.0);

    BaireTestResult {
        n_objects: n,
        n_attributes: encoder.attributes.len(),
        base: encoder.base,
        n_digits: encoder.n_digits,
        ultrametric_fraction: obs_frac,
        null_fraction_mean: null_mean,
        null_fraction_std: null_std,
        p_value,
    }
}

/// Run ultrametric fraction test on Baire distances.
///
/// Encodes the data using the Baire metric, computes the distance matrix,
/// and tests the ultrametric fraction against a null distribution
/// (shuffled attribute assignments).
///
/// Uses a **two-sided** p-value: counts null fractions whose deviation
/// from the null mean is >= the observed deviation. This tests whether
/// observed ultrametricity *differs* from chance in either direction.
/// For one-sided testing (recommended for most applications), use
/// [`matrix_free_ultrametric_test_with_null`] instead.
pub fn baire_ultrametric_test(
    encoder: &BaireEncoder,
    data: &[Vec<f64>],
    n_triples: usize,
    n_permutations: usize,
    seed: u64,
) -> BaireTestResult {
    let n = data.len();
    assert!(n >= 3, "Need at least 3 objects");

    let dist_matrix = baire_distance_matrix(encoder, data);

    // Observed ultrametric fraction
    let obs_frac = super::ultrametric_fraction_from_matrix(
        &dist_matrix, n, n_triples, seed,
    );

    // Null: shuffle each column independently to break inter-attribute correlations
    let mut rng = ChaCha8Rng::seed_from_u64(seed + 1_000_000);
    let mut null_fracs = Vec::with_capacity(n_permutations);
    let mut shuffled_data = data.to_vec();

    for _ in 0..n_permutations {
        // Shuffle each column independently (breaks inter-attribute correlations)
        for col in 0..encoder.attributes.len() {
            let mut col_values: Vec<f64> = shuffled_data.iter().map(|row| row[col]).collect();
            col_values.shuffle(&mut rng);
            for (i, &v) in col_values.iter().enumerate() {
                shuffled_data[i][col] = v;
            }
        }

        let null_dists = baire_distance_matrix(encoder, &shuffled_data);
        let null_frac = super::ultrametric_fraction_from_matrix(
            &null_dists, n, n_triples, seed + 2_000_000,
        );
        null_fracs.push(null_frac);
    }

    let null_mean = null_fracs.iter().sum::<f64>() / n_permutations as f64;
    let null_var = null_fracs
        .iter()
        .map(|f| (f - null_mean).powi(2))
        .sum::<f64>()
        / n_permutations as f64;
    let null_std = null_var.sqrt();

    // Two-sided p-value
    let n_extreme = null_fracs
        .iter()
        .filter(|&&f| (f - null_mean).abs() >= (obs_frac - null_mean).abs())
        .count();
    let p_value = (n_extreme as f64 + 1.0) / (n_permutations as f64 + 1.0);

    BaireTestResult {
        n_objects: n,
        n_attributes: encoder.attributes.len(),
        base: encoder.base,
        n_digits: encoder.n_digits,
        ultrametric_fraction: obs_frac,
        null_fraction_mean: null_mean,
        null_fraction_std: null_std,
        p_value,
    }
}

/// Result of Baire metric ultrametric test.
#[derive(Debug, Clone)]
pub struct BaireTestResult {
    /// Number of objects in the dataset.
    pub n_objects: usize,
    /// Number of attributes used in the encoding.
    pub n_attributes: usize,
    /// Base used for digit representation.
    pub base: u64,
    /// Digits per attribute.
    pub n_digits: usize,
    /// Observed ultrametric fraction on Baire distances.
    pub ultrametric_fraction: f64,
    /// Mean ultrametric fraction from null distribution.
    pub null_fraction_mean: f64,
    /// Standard deviation of null fractions.
    pub null_fraction_std: f64,
    /// P-value for the ultrametric fraction test.
    pub p_value: f64,
}

// ---------------------------------------------------------------------------
// Matrix-free ultrametric tests: O(N*d) memory instead of O(N^2)
// ---------------------------------------------------------------------------

/// Normalize a full dataset into [0,1]^d using the encoder's attribute specs.
///
/// Returns a flat column-major array: `normalized[col * n + row]` for SIMD-
/// friendly column access during distance computation.
pub fn normalize_data_column_major(
    encoder: &BaireEncoder,
    data: &[Vec<f64>],
) -> (Vec<f64>, usize, usize) {
    let n = data.len();
    let d = encoder.attributes.len();
    let mut cols = vec![0.0_f64; d * n];
    for (row_i, row) in data.iter().enumerate() {
        for (col_j, spec) in encoder.attributes.iter().enumerate() {
            cols[col_j * n + row_i] = encoder.normalize(row[col_j], spec);
        }
    }
    (cols, n, d)
}

/// Compute squared Euclidean distance between two rows in column-major layout.
#[inline(always)]
fn euclidean_dist_sq_colmajor(cols: &[f64], n: usize, d: usize, i: usize, j: usize) -> f64 {
    let mut sum = 0.0_f64;
    for c in 0..d {
        let diff = cols[c * n + i] - cols[c * n + j];
        sum += diff * diff;
    }
    sum
}

/// Compute ultrametric fraction by sampling triples directly from data.
///
/// No distance matrix is built. Memory: O(N * d) for the normalized data.
/// Computation: O(n_triples * d) for the triple evaluations.
///
/// Uses squared distances internally to avoid 3 sqrt() calls per triple.
/// The relative tolerance epsilon on distances is converted to an equivalent
/// threshold on squared distances:
///
///   (d_max - d_mid) / d_max < epsilon
///   <==> d_mid / d_max > (1 - epsilon)
///   <==> d_mid^2 / d_max^2 > (1 - epsilon)^2   [since sqrt is monotone]
///   <==> (d_max^2 - d_mid^2) / d_max^2 < 1 - (1 - epsilon)^2
///
/// This gives a ~30% speedup on the inner loop (sqrt is expensive in f64).
pub fn matrix_free_fraction(
    cols: &[f64],
    n: usize,
    d: usize,
    n_triples: usize,
    seed: u64,
    epsilon: f64,
) -> f64 {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut count = 0_usize;

    // Convert epsilon on distances to epsilon on squared distances
    let epsilon_sq = 1.0 - (1.0 - epsilon).powi(2);

    for _ in 0..n_triples {
        let i = rng.gen_range(0..n);
        let mut j = rng.gen_range(0..n - 1);
        if j >= i { j += 1; }
        let mut k = rng.gen_range(0..n - 2);
        if k >= i.min(j) { k += 1; }
        if k >= i.max(j) { k += 1; }

        // Work with squared distances (no sqrt needed)
        let d_ij_sq = euclidean_dist_sq_colmajor(cols, n, d, i, j);
        let d_jk_sq = euclidean_dist_sq_colmajor(cols, n, d, j, k);
        let d_ik_sq = euclidean_dist_sq_colmajor(cols, n, d, i, k);

        let mut dists_sq = [d_ij_sq, d_jk_sq, d_ik_sq];
        dists_sq.sort_by(|a, b| a.partial_cmp(b).unwrap());

        if dists_sq[2] > 1e-30 {
            let relative_diff_sq = (dists_sq[2] - dists_sq[1]) / dists_sq[2];
            if relative_diff_sq < epsilon_sq {
                count += 1;
            }
        } else {
            count += 1;
        }
    }

    count as f64 / n_triples as f64
}

/// Full matrix-free ultrametric test with permutation null.
///
/// Memory: O(N * d) for the data -- no O(N^2) distance matrix.
/// For 50K objects with 6 attributes, this uses ~2.4 MB instead of ~10 GB.
///
/// Uses the default null model (ColumnIndependent). For alternative null
/// models, use [`matrix_free_ultrametric_test_with_null`].
pub fn matrix_free_ultrametric_test(
    encoder: &BaireEncoder,
    data: &[Vec<f64>],
    n_triples: usize,
    n_permutations: usize,
    seed: u64,
) -> BaireTestResult {
    matrix_free_ultrametric_test_with_null(
        encoder, data, n_triples, n_permutations, seed, NullModel::default(),
    )
}

/// Full matrix-free ultrametric test with configurable null model.
///
/// Memory: O(N * d) for the data -- no O(N^2) distance matrix.
/// For 50K objects with 6 attributes, this uses ~2.4 MB instead of ~10 GB.
///
/// Uses a **one-sided** p-value: counts null fractions >= observed.
/// This tests whether observed ultrametricity is *higher* than chance,
/// which is the scientifically relevant direction for most applications.
///
/// The `null_model` parameter controls how the data is randomized for
/// the permutation null distribution. See [`NullModel`] for options.
/// Note: `RowPermutation` and `RandomRotation` are identity-equivalent
/// for Euclidean-distance statistics -- see [`NullModel`] docs for details.
pub fn matrix_free_ultrametric_test_with_null(
    encoder: &BaireEncoder,
    data: &[Vec<f64>],
    n_triples: usize,
    n_permutations: usize,
    seed: u64,
    null_model: NullModel,
) -> BaireTestResult {
    let n = data.len();
    assert!(n >= 3, "Need at least 3 objects");
    let d = encoder.attributes.len();

    let (cols, _, _) = normalize_data_column_major(encoder, data);

    // Observed fraction at default epsilon = 0.05
    let obs_frac = matrix_free_fraction(&cols, n, d, n_triples, seed, 0.05);

    // Null distribution via the selected null model
    let mut rng = ChaCha8Rng::seed_from_u64(seed + 1_000_000);
    let mut null_fracs = Vec::with_capacity(n_permutations);
    let mut shuffled_cols = cols.clone();

    for _ in 0..n_permutations {
        shuffled_cols.copy_from_slice(&cols);
        apply_null_column_major(&mut shuffled_cols, n, d, null_model, &mut rng);
        let null_frac = matrix_free_fraction(
            &shuffled_cols, n, d, n_triples, seed + 2_000_000, 0.05,
        );
        null_fracs.push(null_frac);
    }

    let null_mean = null_fracs.iter().sum::<f64>() / n_permutations as f64;
    let null_var = null_fracs
        .iter()
        .map(|f| (f - null_mean).powi(2))
        .sum::<f64>()
        / n_permutations as f64;
    let null_std = null_var.sqrt();

    let n_extreme = null_fracs.iter().filter(|&&f| f >= obs_frac).count();
    let p_value = (n_extreme as f64 + 1.0) / (n_permutations as f64 + 1.0);

    BaireTestResult {
        n_objects: n,
        n_attributes: d,
        base: encoder.base,
        n_digits: encoder.n_digits,
        ultrametric_fraction: obs_frac,
        null_fraction_mean: null_mean,
        null_fraction_std: null_std,
        p_value,
    }
}

/// Matrix-free tolerance curve: sweep epsilon without any O(N^2) matrices.
///
/// Uses the default null model (ColumnIndependent). For alternative null
/// models, use [`matrix_free_tolerance_curve_with_null`].
///
/// Memory: O(N * d) -- just the data.
/// Computation: O(n_epsilons * (1 + n_perms) * n_triples * d)
pub fn matrix_free_tolerance_curve(
    encoder: &BaireEncoder,
    data: &[Vec<f64>],
    n_triples: usize,
    n_permutations: usize,
    seed: u64,
) -> super::ToleranceCurveResult {
    matrix_free_tolerance_curve_with_null(
        encoder, data, n_triples, n_permutations, seed, NullModel::default(),
    )
}

/// Matrix-free tolerance curve with configurable null model.
///
/// For each epsilon, computes the fraction for observed data and for each
/// null permutation. No distance matrices are stored -- everything is
/// computed on-the-fly from the column-major normalized data.
///
/// The `null_model` parameter controls how the data is randomized for
/// the permutation null distribution. See [`NullModel`] for options.
///
/// Memory: O(N * d) -- just the data.
/// Computation: O(n_epsilons * (1 + n_perms) * n_triples * d)
pub fn matrix_free_tolerance_curve_with_null(
    encoder: &BaireEncoder,
    data: &[Vec<f64>],
    n_triples: usize,
    n_permutations: usize,
    seed: u64,
    null_model: NullModel,
) -> super::ToleranceCurveResult {
    let n = data.len();
    let d = encoder.attributes.len();

    let (cols, _, _) = normalize_data_column_major(encoder, data);
    let epsilons: Vec<f64> = (1..=20).map(|i| i as f64 * 0.01).collect();

    // Compute observed fractions at each epsilon
    let obs_fracs: Vec<f64> = epsilons
        .iter()
        .map(|&eps| matrix_free_fraction(&cols, n, d, n_triples, seed, eps))
        .collect();

    // Compute null mean at each epsilon
    let mut null_means = vec![0.0_f64; epsilons.len()];
    let mut rng = ChaCha8Rng::seed_from_u64(seed + 1_000_000);
    let mut shuffled_cols = cols.clone();
    let n_perms = n_permutations.min(50);

    for _ in 0..n_perms {
        shuffled_cols.copy_from_slice(&cols);
        apply_null_column_major(&mut shuffled_cols, n, d, null_model, &mut rng);
        for (ei, &eps) in epsilons.iter().enumerate() {
            let null_frac = matrix_free_fraction(
                &shuffled_cols, n, d, n_triples, seed + 2_000_000, eps,
            );
            null_means[ei] += null_frac;
        }
    }
    for nm in &mut null_means {
        *nm /= n_perms as f64;
    }

    // Build curve points
    let mut points = Vec::with_capacity(epsilons.len());
    for (ei, &eps) in epsilons.iter().enumerate() {
        points.push(super::ToleranceCurvePoint {
            epsilon: eps,
            observed: obs_fracs[ei],
            null_mean: null_means[ei],
            excess: obs_fracs[ei] - null_means[ei],
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

    super::ToleranceCurveResult {
        points,
        auc_excess: auc,
        max_excess,
        best_epsilon,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_baire_encoder_identical_objects() {
        let spec = vec![
            AttributeSpec { name: "x".into(), min: 0.0, max: 100.0, log_scale: false },
            AttributeSpec { name: "y".into(), min: 0.0, max: 100.0, log_scale: false },
        ];
        let encoder = BaireEncoder::new(spec, 10, 4);

        let seq_a = encoder.encode(&[50.0, 50.0]);
        let seq_b = encoder.encode(&[50.0, 50.0]);

        assert_eq!(encoder.baire_distance(&seq_a, &seq_b), 0.0);
    }

    #[test]
    fn test_baire_encoder_different_first_digit() {
        let spec = vec![
            AttributeSpec { name: "x".into(), min: 0.0, max: 100.0, log_scale: false },
        ];
        let encoder = BaireEncoder::new(spec, 10, 4);

        // 10.0/100.0 = 0.1 -> digits [1, 0, 0, 0]
        // 90.0/100.0 = 0.9 -> digits [9, 0, 0, 0]
        let seq_a = encoder.encode(&[10.0]);
        let seq_b = encoder.encode(&[90.0]);

        let d = encoder.baire_distance(&seq_a, &seq_b);
        // First digit differs -> d = 10^(-1) = 0.1
        assert!((d - 0.1).abs() < 1e-10, "Expected 0.1, got {}", d);
    }

    #[test]
    fn test_baire_ultrametric_property() {
        // Baire distances are inherently ultrametric by construction.
        // All triples should satisfy the ultrametric inequality exactly.
        let spec = vec![
            AttributeSpec { name: "x".into(), min: 0.0, max: 100.0, log_scale: false },
            AttributeSpec { name: "y".into(), min: 0.0, max: 100.0, log_scale: false },
        ];
        let encoder = BaireEncoder::new(spec, 10, 4);

        // Create data with clearly different first digits
        let data: Vec<Vec<f64>> = vec![
            vec![10.0, 10.0],
            vec![15.0, 15.0],
            vec![50.0, 50.0],
            vec![55.0, 55.0],
            vec![90.0, 90.0],
        ];

        let dists = baire_distance_matrix(&encoder, &data);

        // Check all triples
        let n = 5;
        let idx = |i: usize, j: usize| -> usize {
            let (a, b) = if i < j { (i, j) } else { (j, i) };
            a * n - a * (a + 1) / 2 + b - a - 1
        };

        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    let d_ij = dists[idx(i, j)];
                    let d_jk = dists[idx(j, k)];
                    let d_ik = dists[idx(i, k)];

                    // Ultrametric: d(i,k) <= max(d(i,j), d(j,k))
                    let max_pair = d_ij.max(d_jk);
                    assert!(
                        d_ik <= max_pair + 1e-15,
                        "Ultrametric violated: d({},{})={}, d({},{})={}, d({},{})={}",
                        i, j, d_ij, j, k, d_jk, i, k, d_ik
                    );
                }
            }
        }
    }

    #[test]
    fn test_baire_log_scale() {
        let spec = vec![
            AttributeSpec { name: "period".into(), min: 0.001, max: 10.0, log_scale: true },
        ];
        let encoder = BaireEncoder::new(spec, 10, 4);

        // 0.001 and 10.0 should map to extremes of the normalized range
        let seq_min = encoder.encode(&[0.001]);
        let seq_max = encoder.encode(&[10.0]);

        let d = encoder.baire_distance(&seq_min, &seq_max);
        // First digit should differ -> d = 10^(-1) = 0.1
        assert!(d > 0.0, "Different extremes should have nonzero distance");
    }

    #[test]
    fn test_baire_distance_matrix_size() {
        let spec = vec![
            AttributeSpec { name: "x".into(), min: 0.0, max: 1.0, log_scale: false },
        ];
        let encoder = BaireEncoder::new(spec, 10, 3);

        let data: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64 * 0.2]).collect();
        let dists = baire_distance_matrix(&encoder, &data);

        assert_eq!(dists.len(), 5 * 4 / 2);
    }

    #[test]
    fn test_matrix_free_fraction_matches_matrix_version() {
        // Verify that matrix-free fraction gives the same result as the
        // matrix-based version (same seed, same data, same epsilon).
        let specs = vec![
            AttributeSpec { name: "x".into(), min: 0.0, max: 10.0, log_scale: false },
            AttributeSpec { name: "y".into(), min: 0.0, max: 10.0, log_scale: false },
        ];
        let encoder = BaireEncoder::new(specs, 10, 4);
        let mut rng = ChaCha8Rng::seed_from_u64(999);
        let data: Vec<Vec<f64>> = (0..200)
            .map(|_| vec![rng.gen_range(0.0..10.0), rng.gen_range(0.0..10.0)])
            .collect();

        // Matrix-based
        let dm = euclidean_distance_matrix(&encoder, &data);
        let frac_matrix = crate::ultrametric::ultrametric_fraction_from_matrix_eps(
            &dm, data.len(), 10_000, 42, 0.05,
        );

        // Matrix-free
        let (cols, n, d) = normalize_data_column_major(&encoder, &data);
        let frac_free = matrix_free_fraction(&cols, n, d, 10_000, 42, 0.05);

        assert!(
            (frac_matrix - frac_free).abs() < 1e-10,
            "Matrix ({frac_matrix}) and matrix-free ({frac_free}) fractions must match",
        );
    }

    #[test]
    fn test_matrix_free_ultrametric_test_runs() {
        let specs = vec![
            AttributeSpec { name: "a".into(), min: 0.0, max: 1.0, log_scale: false },
            AttributeSpec { name: "b".into(), min: 0.0, max: 1.0, log_scale: false },
        ];
        let encoder = BaireEncoder::new(specs, 10, 4);
        let mut rng = ChaCha8Rng::seed_from_u64(77);
        let data: Vec<Vec<f64>> = (0..100)
            .map(|_| vec![rng.gen_range(0.0..1.0), rng.gen_range(0.0..1.0)])
            .collect();

        let result = matrix_free_ultrametric_test(&encoder, &data, 5_000, 50, 42);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert!(result.ultrametric_fraction > 0.0);
        assert_eq!(result.n_objects, 100);
    }

    #[test]
    fn test_matrix_free_tolerance_curve_structure() {
        let specs = vec![
            AttributeSpec { name: "x".into(), min: 0.0, max: 1.0, log_scale: false },
            AttributeSpec { name: "y".into(), min: 0.0, max: 1.0, log_scale: false },
        ];
        let encoder = BaireEncoder::new(specs, 10, 4);
        let mut rng = ChaCha8Rng::seed_from_u64(55);
        let data: Vec<Vec<f64>> = (0..50)
            .map(|_| vec![rng.gen_range(0.0..1.0), rng.gen_range(0.0..1.0)])
            .collect();

        let tc = matrix_free_tolerance_curve(&encoder, &data, 5_000, 20, 42);
        assert_eq!(tc.points.len(), 20);
        // Observed fraction must be monotonically non-decreasing with epsilon
        for w in tc.points.windows(2) {
            assert!(
                w[1].observed >= w[0].observed - 1e-10,
                "Fraction must increase with epsilon: {} -> {}",
                w[0].observed, w[1].observed,
            );
        }
    }

    #[test]
    fn test_column_major_layout() {
        let specs = vec![
            AttributeSpec { name: "x".into(), min: 0.0, max: 10.0, log_scale: false },
            AttributeSpec { name: "y".into(), min: 0.0, max: 10.0, log_scale: false },
        ];
        let encoder = BaireEncoder::new(specs, 10, 4);
        let data = vec![vec![2.0, 8.0], vec![4.0, 6.0], vec![6.0, 4.0]];

        let (cols, n, d) = normalize_data_column_major(&encoder, &data);
        assert_eq!(n, 3);
        assert_eq!(d, 2);
        assert_eq!(cols.len(), 6);
        // Column 0 (x): normalized values of [2, 4, 6] in range [0, 10]
        assert!((cols[0] - 0.2).abs() < 1e-10); // x[0] = 2/10
        assert!((cols[1] - 0.4).abs() < 1e-10); // x[1] = 4/10
        assert!((cols[2] - 0.6).abs() < 1e-10); // x[2] = 6/10
        // Column 1 (y): normalized values of [8, 6, 4] in range [0, 10]
        assert!((cols[3] - 0.8).abs() < 1e-10); // y[0] = 8/10
        assert!((cols[4] - 0.6).abs() < 1e-10); // y[1] = 6/10
        assert!((cols[5] - 0.4).abs() < 1e-10); // y[2] = 4/10
    }
}

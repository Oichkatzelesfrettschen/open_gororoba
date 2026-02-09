//! Attribute subset search for multi-attribute ultrametric analysis.
//!
//! Enumerates all subsets of attributes (of size >= min_k), runs an
//! ultrametric fraction test on each subset, and returns results with
//! BH-FDR adjusted p-values.
//!
//! This extracts the combinatorial search logic from the CLI binary
//! into a reusable library function.

use super::baire::{
    matrix_free_ultrametric_test_with_null, AttributeSpec, BaireEncoder, BaireTestResult,
};
use super::null_models::NullModel;

/// Result for a single attribute subset test.
#[derive(Debug, Clone)]
pub struct SubsetTestResult {
    /// Indices of the attributes in this subset.
    pub attribute_indices: Vec<usize>,
    /// Names of the attributes in this subset.
    pub attribute_names: Vec<String>,
    /// Observed ultrametric fraction.
    pub fraction: f64,
    /// Null mean ultrametric fraction.
    pub null_mean: f64,
    /// Effect size (observed - null_mean).
    pub effect_size: f64,
    /// Raw (unadjusted) p-value.
    pub raw_p: f64,
}

/// Aggregate result of searching over all attribute subsets.
#[derive(Debug, Clone)]
pub struct SubsetSearchResult {
    /// Results for each subset, ordered by raw p-value ascending.
    pub subsets: Vec<SubsetTestResult>,
    /// BH-FDR adjusted p-values (same order as subsets).
    pub adjusted_p_values: Vec<f64>,
    /// Which subsets are significant at the given FDR level.
    pub significant: Vec<bool>,
    /// Total number of subsets tested.
    pub n_subsets: usize,
    /// FDR level used for correction.
    pub fdr_level: f64,
}

/// Generate all subsets of `{0, 1, ..., total-1}` with size >= `min_size`.
///
/// Returns subsets ordered by size then lexicographic order.
pub fn attribute_subsets(total: usize, min_size: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    for size in min_size..=total {
        let mut combo = vec![0usize; size];
        generate_combinations(total, size, 0, 0, &mut combo, &mut result);
    }
    result
}

/// Recursive combination generator.
fn generate_combinations(
    n: usize,
    k: usize,
    start: usize,
    depth: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Vec<usize>>,
) {
    if depth == k {
        result.push(current.clone());
        return;
    }
    for i in start..n {
        current[depth] = i;
        generate_combinations(n, k, i + 1, depth + 1, current, result);
    }
}

/// Project row-major data to a subset of columns.
///
/// `data`: rows of observations, each row has `specs.len()` values.
/// `specs`: attribute specifications (one per column).
/// `indices`: which columns to keep.
///
/// Returns the projected data and projected attribute specs.
pub fn project_data(
    data: &[Vec<f64>],
    specs: &[AttributeSpec],
    indices: &[usize],
) -> (Vec<Vec<f64>>, Vec<AttributeSpec>) {
    let projected: Vec<Vec<f64>> = data
        .iter()
        .map(|row| indices.iter().map(|&i| row[i]).collect())
        .collect();

    let proj_specs: Vec<AttributeSpec> = indices
        .iter()
        .map(|&old_col| {
            let s = &specs[old_col];
            let vals: Vec<f64> = projected
                .iter()
                .map(|row| row[indices.iter().position(|&j| j == old_col).unwrap()])
                .collect();
            let min = vals.iter().copied().fold(f64::INFINITY, f64::min);
            let max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            AttributeSpec {
                name: s.name.clone(),
                min,
                max,
                log_scale: s.log_scale,
            }
        })
        .collect();

    (projected, proj_specs)
}

/// Configuration for attribute subset search.
#[derive(Debug, Clone)]
pub struct SubsetSearchConfig {
    /// Minimum subset size to test (typically 2).
    pub min_k: usize,
    /// Number of random triples per test.
    pub n_triples: usize,
    /// Number of permutations per test.
    pub n_permutations: usize,
    /// Null model strategy for permutation testing.
    pub null_model: NullModel,
    /// FDR level for BH correction (e.g. 0.05).
    pub fdr_level: f64,
    /// Base RNG seed.
    pub seed: u64,
}

impl Default for SubsetSearchConfig {
    fn default() -> Self {
        Self {
            min_k: 2,
            n_triples: 100_000,
            n_permutations: 200,
            null_model: NullModel::default(),
            fdr_level: 0.05,
            seed: 42,
        }
    }
}

/// Search all attribute subsets for ultrametric signal.
///
/// For each subset of size >= `config.min_k`, runs a matrix-free ultrametric
/// test with the specified null model. Results are sorted by raw p-value
/// and BH-FDR corrected.
///
/// # Arguments
///
/// * `data` - Row-major data: `data[i]` is observation i, `data[i][j]` is attribute j
/// * `specs` - Attribute specifications (one per column)
/// * `config` - Search configuration (min_k, n_triples, n_permutations, etc.)
pub fn subset_search(
    data: &[Vec<f64>],
    specs: &[AttributeSpec],
    config: &SubsetSearchConfig,
) -> SubsetSearchResult {
    let SubsetSearchConfig {
        min_k,
        n_triples,
        n_permutations,
        null_model,
        fdr_level,
        seed,
    } = *config;
    let subsets = attribute_subsets(specs.len(), min_k);
    let n_subsets = subsets.len();

    let mut results: Vec<SubsetTestResult> = subsets
        .iter()
        .enumerate()
        .map(|(idx, indices)| {
            let (proj_data, proj_specs) = project_data(data, specs, indices);
            let encoder = BaireEncoder::new(proj_specs.clone(), 10, 4);
            let subset_seed = seed + (idx as u64) * 10_000;

            let baire: BaireTestResult = matrix_free_ultrametric_test_with_null(
                &encoder,
                &proj_data,
                n_triples,
                n_permutations,
                subset_seed,
                null_model,
            );

            SubsetTestResult {
                attribute_indices: indices.clone(),
                attribute_names: proj_specs.iter().map(|s| s.name.clone()).collect(),
                fraction: baire.ultrametric_fraction,
                null_mean: baire.null_fraction_mean,
                effect_size: baire.ultrametric_fraction - baire.null_fraction_mean,
                raw_p: baire.p_value,
            }
        })
        .collect();

    // Sort by raw p-value ascending
    results.sort_by(|a, b| a.raw_p.partial_cmp(&b.raw_p).unwrap());

    // BH-FDR correction
    let raw_ps: Vec<f64> = results.iter().map(|r| r.raw_p).collect();
    let fdr = super::benjamini_hochberg(&raw_ps, fdr_level);

    SubsetSearchResult {
        subsets: results,
        adjusted_p_values: fdr.adjusted_p_values,
        significant: fdr.significant,
        n_subsets,
        fdr_level,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    fn make_random_data(n: usize, d: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<AttributeSpec>) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let data: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..d).map(|_| rng.gen_range(0.0..1.0)).collect())
            .collect();
        let specs: Vec<AttributeSpec> = (0..d)
            .map(|i| AttributeSpec {
                name: format!("attr_{i}"),
                min: 0.0,
                max: 1.0,
                log_scale: false,
            })
            .collect();
        (data, specs)
    }

    #[test]
    fn test_attribute_subsets_count() {
        // 4 attributes, min_size 2: C(4,2)+C(4,3)+C(4,4) = 6+4+1 = 11
        let subs = attribute_subsets(4, 2);
        assert_eq!(subs.len(), 11);

        // 3 attributes, min_size 2: C(3,2)+C(3,3) = 3+1 = 4
        let subs = attribute_subsets(3, 2);
        assert_eq!(subs.len(), 4);

        // 5 attributes, min_size 3: C(5,3)+C(5,4)+C(5,5) = 10+5+1 = 16
        let subs = attribute_subsets(5, 3);
        assert_eq!(subs.len(), 16);
    }

    #[test]
    fn test_attribute_subsets_content() {
        let subs = attribute_subsets(3, 2);
        assert_eq!(
            subs,
            vec![vec![0, 1], vec![0, 2], vec![1, 2], vec![0, 1, 2],]
        );
    }

    #[test]
    fn test_project_data_dimensions() {
        let (data, specs) = make_random_data(20, 5, 42);
        let (proj, proj_specs) = project_data(&data, &specs, &[1, 3]);
        assert_eq!(proj.len(), 20);
        assert_eq!(proj[0].len(), 2);
        assert_eq!(proj_specs.len(), 2);
        assert_eq!(proj_specs[0].name, "attr_1");
        assert_eq!(proj_specs[1].name, "attr_3");
    }

    #[test]
    fn test_project_data_values() {
        let (data, specs) = make_random_data(10, 4, 42);
        let (proj, _) = project_data(&data, &specs, &[2]);
        for (i, row) in proj.iter().enumerate() {
            assert!(
                (row[0] - data[i][2]).abs() < 1e-15,
                "projected value should match original column"
            );
        }
    }

    #[test]
    fn test_subset_search_random_data_fpr() {
        // With random data and alpha=0.05, roughly 5% of tests should be
        // significant by chance. With BH-FDR correction, the false positive
        // rate should be controlled: expect few or no significant subsets.
        let (data, specs) = make_random_data(50, 3, 42);
        let config = SubsetSearchConfig {
            min_k: 2,
            n_triples: 5_000,
            n_permutations: 50,
            null_model: NullModel::ColumnIndependent,
            fdr_level: 0.05,
            seed: 42,
        };
        let result = subset_search(&data, &specs, &config);

        assert_eq!(result.n_subsets, 4); // C(3,2)+C(3,3) = 4
        assert_eq!(result.subsets.len(), 4);
        assert_eq!(result.adjusted_p_values.len(), 4);
        assert_eq!(result.significant.len(), 4);

        // With only 4 tests on random data, most should be non-significant
        // after BH correction. Allow up to 1 false positive.
        let n_sig = result.significant.iter().filter(|&&s| s).count();
        assert!(
            n_sig <= 1,
            "random data should have few BH-significant subsets, got {n_sig}"
        );
    }

    #[test]
    fn test_subset_search_sorted_by_p() {
        let (data, specs) = make_random_data(40, 3, 42);
        let config = SubsetSearchConfig {
            n_triples: 5_000,
            n_permutations: 50,
            seed: 42,
            ..SubsetSearchConfig::default()
        };
        let result = subset_search(&data, &specs, &config);

        for w in result.subsets.windows(2) {
            assert!(
                w[0].raw_p <= w[1].raw_p,
                "results should be sorted by raw_p ascending"
            );
        }
    }

    #[test]
    fn test_subset_search_with_different_null_models() {
        // Verify that all null models produce valid results (no panics)
        let (data, specs) = make_random_data(30, 3, 42);
        for null_model in [
            NullModel::ColumnIndependent,
            NullModel::RowPermutation,
            NullModel::ToroidalShift,
            NullModel::RandomRotation,
        ] {
            let config = SubsetSearchConfig {
                n_triples: 1_000,
                n_permutations: 20,
                null_model,
                seed: 42,
                ..SubsetSearchConfig::default()
            };
            let result = subset_search(&data, &specs, &config);
            assert_eq!(result.n_subsets, 4);
            for sub in &result.subsets {
                assert!(sub.raw_p >= 0.0 && sub.raw_p <= 1.0);
                assert!(sub.fraction >= 0.0 && sub.fraction <= 1.0);
            }
        }
    }
}

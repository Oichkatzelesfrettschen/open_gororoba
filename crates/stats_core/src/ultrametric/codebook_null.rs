//! Codebook null test: permutation test for encoding dictionary assignment.
//!
//! Tests whether the specific basis->lattice assignment in an encoding
//! dictionary creates geometric structure that reflects the algebraic
//! zero-divisor relationships.
//!
//! # Method
//!
//! The null hypothesis is that any random permutation of the basis->lattice
//! assignment would produce the same aggregate distance structure among
//! ZD-connected basis pairs. The test statistic is the mean squared
//! Euclidean distance between lattice vectors of basis elements that
//! participate in the same zero-divisor relationship.
//!
//! Under the null, row permutation randomizes which basis elements map
//! to which lattice points, keeping the point cloud fixed but destroying
//! the algebraic labeling. A small p-value indicates the observed
//! assignment is "special" -- the ZD network is geometrically imprinted
//! in the lattice encoding.
//!
//! # References
//!
//! - Monograph Layer 5: NullModel abstraction with adaptive sequential testing
//! - Uses Besag & Clifford (1991) adaptive stopping via `run_adaptive_null_test`

use std::collections::HashSet;
use algebra_core::analysis::codebook::EncodingDictionary;
use algebra_core::construction::cayley_dickson::find_zero_divisors;
use super::null_models::{NullModelStrategy, RowPermutationNull, NullTestConfig, run_adaptive_null_test};
use super::adaptive::{AdaptiveConfig, AdaptiveResult};

/// Result of a codebook null test.
#[derive(Debug, Clone)]
pub struct CodebookNullResult {
    /// Name of the test statistic used.
    pub statistic_name: String,
    /// Observed value of the test statistic.
    pub observed_value: f64,
    /// Result from the adaptive permutation test.
    pub adaptive_result: AdaptiveResult,
    /// Number of basis elements in the dictionary.
    pub n_basis: usize,
    /// Number of ZD-connected basis pairs used.
    pub n_zd_pairs: usize,
    /// CD algebra dimension.
    pub dim: usize,
}

/// Extract unique basis-element pairs from 2-blade zero-divisor tuples.
///
/// Each ZD tuple (i, j, k, l) represents (e_i + e_j) * (e_k + e_l) = 0.
/// This function extracts all C(4,2) = 6 basis pairs from each tuple,
/// deduplicates across all tuples, and returns sorted pairs (a, b)
/// with a < b.
pub fn extract_zd_basis_pairs(dim: usize) -> Vec<(usize, usize)> {
    let zd_tuples = find_zero_divisors(dim, 1e-10);
    let mut pairs = HashSet::new();
    for &(i, j, k, l, _) in &zd_tuples {
        let mut elems = [i, j, k, l];
        elems.sort_unstable();
        // All C(4,2) pairs from the 4 basis elements in this ZD tuple
        for a_idx in 0..elems.len() {
            for b_idx in (a_idx + 1)..elems.len() {
                if elems[a_idx] != elems[b_idx] {
                    pairs.insert((elems[a_idx], elems[b_idx]));
                }
            }
        }
    }
    let mut sorted: Vec<(usize, usize)> = pairs.into_iter().collect();
    sorted.sort();
    sorted
}

/// Convert an encoding dictionary to column-major f64 data.
///
/// Returns a flat array of size dim * 8, laid out as:
///   data[col * dim + row] = lattice_vec[col] for basis element `row`.
pub fn dictionary_to_column_major(dict: &EncodingDictionary) -> Vec<f64> {
    let n = dict.dim();
    let d = 8;
    let mut data = vec![0.0f64; n * d];
    for (idx, lv) in dict.iter() {
        for col in 0..d {
            data[col * n + idx] = lv[col] as f64;
        }
    }
    data
}

/// Compute mean squared Euclidean distance between specific row pairs.
///
/// `data` is column-major: data[col * n + row].
/// Only computes distances for the given `(i, j)` pairs, not all pairs.
/// Returns 0.0 if `pairs` is empty.
pub fn selective_mean_squared_distance(
    data: &[f64],
    n: usize,
    d: usize,
    pairs: &[(usize, usize)],
) -> f64 {
    if pairs.is_empty() {
        return 0.0;
    }
    let mut total = 0.0;
    for &(i, j) in pairs {
        let mut dist2 = 0.0;
        for col in 0..d {
            let diff = data[col * n + i] - data[col * n + j];
            dist2 += diff * diff;
        }
        total += dist2;
    }
    total / pairs.len() as f64
}

/// Run a codebook null test with a custom null model strategy.
///
/// This is the generalized Layer 5 API: any `NullModelStrategy` can be
/// used to generate the null distribution. The test statistic is the
/// mean squared Euclidean distance between lattice vectors of
/// ZD-connected basis pairs.
///
/// Different null models test different hypotheses:
/// - `RowPermutationNull`: is the basis->lattice *labeling* special?
/// - `ColumnIndependentNull`: is the inter-attribute correlation special?
///
/// A small p-value means the observed assignment is geometrically
/// "special" with respect to the ZD network under the given null.
pub fn run_codebook_null_test_with_strategy(
    dict: &EncodingDictionary,
    zd_basis_pairs: &[(usize, usize)],
    strategy: &dyn NullModelStrategy,
    adaptive_config: &AdaptiveConfig,
    seed: u64,
) -> CodebookNullResult {
    let n = dict.dim();
    let d = 8;
    let data = dictionary_to_column_major(dict);

    let observed = selective_mean_squared_distance(&data, n, d, zd_basis_pairs);

    let null_config = NullTestConfig {
        strategy,
        adaptive: adaptive_config,
        seed,
    };

    let pairs_for_closure = zd_basis_pairs.to_vec();
    let statistic_fn = move |d_slice: &[f64], n_rows: usize, n_cols: usize| -> f64 {
        selective_mean_squared_distance(d_slice, n_rows, n_cols, &pairs_for_closure)
    };

    let adaptive_result = run_adaptive_null_test(
        &data, n, d, observed, &statistic_fn, &null_config,
    );

    CodebookNullResult {
        statistic_name: "mean_zd_pair_squared_distance".to_string(),
        observed_value: observed,
        n_basis: n,
        n_zd_pairs: zd_basis_pairs.len(),
        dim: n,
        adaptive_result,
    }
}

/// Run a codebook null test with pre-computed ZD basis pairs.
///
/// Uses `RowPermutationNull` as the default strategy: tests whether
/// the specific basis->lattice labeling creates shorter (or longer)
/// ZD-pair lattice distances than a random relabeling would.
///
/// For other null models, use `run_codebook_null_test_with_strategy`.
pub fn run_codebook_null_test(
    dict: &EncodingDictionary,
    zd_basis_pairs: &[(usize, usize)],
    adaptive_config: &AdaptiveConfig,
    seed: u64,
) -> CodebookNullResult {
    let strategy = RowPermutationNull;
    run_codebook_null_test_with_strategy(dict, zd_basis_pairs, &strategy, adaptive_config, seed)
}

/// Convenience: extract ZD basis pairs from a CD algebra dimension
/// and run the codebook null test.
///
/// Calls `find_zero_divisors(dim)` internally to obtain ZD tuples,
/// extracts basis pairs, then runs `run_codebook_null_test`.
pub fn codebook_null_test_from_dim(
    dict: &EncodingDictionary,
    adaptive_config: &AdaptiveConfig,
    seed: u64,
) -> CodebookNullResult {
    let zd_pairs = extract_zd_basis_pairs(dict.dim());
    run_codebook_null_test(dict, &zd_pairs, adaptive_config, seed)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use algebra_core::analysis::codebook::{LatticeVector, enumerate_lambda_256};
    use super::super::null_models::ColumnIndependentNull;
    use super::super::adaptive::StopReason;

    /// Build a small (dim=4) dictionary with known lattice vectors.
    fn sample_dictionary_4() -> EncodingDictionary {
        let pairs: Vec<(usize, LatticeVector)> = vec![
            (0, [-1, -1, -1, -1, 0, 0, 0, 0]),
            (1, [-1, -1, 0, 0, -1, -1, 0, 0]),
            (2, [-1, -1, 0, 0, 0, 0, -1, -1]),
            (3, [-1, 0, -1, 0, -1, 0, -1, 0]),
        ];
        EncodingDictionary::try_from_pairs(4, &pairs).unwrap()
    }

    #[test]
    fn test_dictionary_to_column_major_layout() {
        let dict = sample_dictionary_4();
        let data = dictionary_to_column_major(&dict);
        assert_eq!(data.len(), 4 * 8);
        // Row 0, col 0: lattice_vec[0] of basis 0 = -1
        assert_eq!(data[0 * 4 + 0], -1.0);
        // Row 0, col 4: lattice_vec[4] of basis 0 = 0
        assert_eq!(data[4 * 4 + 0], 0.0);
        // Row 1, col 0: lattice_vec[0] of basis 1 = -1
        assert_eq!(data[0 * 4 + 1], -1.0);
        // Row 3, col 2: lattice_vec[2] of basis 3 = -1
        assert_eq!(data[2 * 4 + 3], -1.0);
    }

    #[test]
    fn test_selective_mean_squared_distance_empty_pairs() {
        let data = vec![0.0; 32];
        assert_eq!(selective_mean_squared_distance(&data, 4, 8, &[]), 0.0);
    }

    #[test]
    fn test_selective_mean_squared_distance_single_pair() {
        let dict = sample_dictionary_4();
        let data = dictionary_to_column_major(&dict);
        // Basis 0: [-1,-1,-1,-1,0,0,0,0]
        // Basis 1: [-1,-1,0,0,-1,-1,0,0]
        // Diff:    [0,0,-1,-1,1,1,0,0] -> d^2 = 0+0+1+1+1+1+0+0 = 4
        let msd = selective_mean_squared_distance(&data, 4, 8, &[(0, 1)]);
        assert!((msd - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_selective_mean_squared_distance_multiple_pairs() {
        let dict = sample_dictionary_4();
        let data = dictionary_to_column_major(&dict);
        // Pair (0,1): d^2 = 4 (computed above)
        // Pair (0,2): [-1,-1,-1,-1,0,0,0,0] vs [-1,-1,0,0,0,0,-1,-1]
        //   Diff: [0,0,-1,-1,0,0,1,1] -> d^2 = 4
        // Mean = (4 + 4) / 2 = 4
        let msd = selective_mean_squared_distance(&data, 4, 8, &[(0, 1), (0, 2)]);
        assert!((msd - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_selective_vs_global_mean_distance() {
        // When all pairs are included, selective = global mean distance
        let dict = sample_dictionary_4();
        let data = dictionary_to_column_major(&dict);
        let all_pairs: Vec<(usize, usize)> = (0..4)
            .flat_map(|i| ((i + 1)..4).map(move |j| (i, j)))
            .collect();
        let selective = selective_mean_squared_distance(&data, 4, 8, &all_pairs);

        // Compute global manually
        let mut total = 0.0;
        let mut count = 0;
        for i in 0..4 {
            for j in (i + 1)..4 {
                let mut d2 = 0.0;
                for col in 0..8 {
                    let diff = data[col * 4 + i] - data[col * 4 + j];
                    d2 += diff * diff;
                }
                total += d2;
                count += 1;
            }
        }
        let global = total / count as f64;
        assert!((selective - global).abs() < 1e-10);
    }

    #[test]
    fn test_extract_zd_basis_pairs_dim16() {
        // Sedenions (dim=16) should have ZD pairs starting from basis 1
        // (e_0 is the identity and never participates in 2-blade ZDs)
        let pairs = extract_zd_basis_pairs(16);
        assert!(!pairs.is_empty(), "dim=16 should have ZD basis pairs");
        // All pairs should have a < b
        for &(a, b) in &pairs {
            assert!(a < b, "pairs must be sorted: ({a}, {b})");
        }
        // No pair should involve basis 0 (identity)
        for &(a, _) in &pairs {
            assert!(a >= 1, "basis 0 (identity) should not appear in ZD pairs");
        }
        // At dim=16, there are 15 non-identity basis elements, so max
        // C(15,2) = 105 pairs. The actual count should be substantial.
        assert!(
            pairs.len() >= 20,
            "expected substantial ZD connectivity at dim=16, got {}",
            pairs.len()
        );
    }

    #[test]
    fn test_extract_zd_basis_pairs_sorted_and_unique() {
        let pairs = extract_zd_basis_pairs(16);
        // Check sorted
        for w in pairs.windows(2) {
            assert!(w[0] < w[1], "pairs must be strictly sorted");
        }
        // Check unique (implied by strict sorting, but verify)
        let set: HashSet<(usize, usize)> = pairs.iter().copied().collect();
        assert_eq!(set.len(), pairs.len(), "pairs must be unique");
    }

    #[test]
    fn test_codebook_null_test_smoke_dim4() {
        // Use a small dictionary where we manually specify ZD pairs.
        // This is a smoke test -- dim=4 quaternions have no ZD, so
        // we use synthetic pairs to verify the machinery runs.
        let dict = sample_dictionary_4();
        let synthetic_pairs = vec![(0, 1), (2, 3)];
        let config = AdaptiveConfig {
            batch_size: 50,
            max_permutations: 200,
            alpha: 0.05,
            confidence: 0.95,
            min_permutations: 50,
        };
        let result = run_codebook_null_test(&dict, &synthetic_pairs, &config, 42);

        assert_eq!(result.n_basis, 4);
        assert_eq!(result.n_zd_pairs, 2);
        assert_eq!(result.dim, 4);
        assert!(result.observed_value > 0.0);
        assert!(result.adaptive_result.p_value >= 0.0);
        assert!(result.adaptive_result.p_value <= 1.0);
    }

    #[test]
    fn test_codebook_null_test_trivial_p_value_with_one_pair() {
        // With only 1 pair out of C(4,2)=6 total, a random permutation
        // can produce any pair distance, so p should not be 0.
        let dict = sample_dictionary_4();
        let config = AdaptiveConfig {
            batch_size: 100,
            max_permutations: 500,
            alpha: 0.05,
            confidence: 0.95,
            min_permutations: 100,
        };
        let result = run_codebook_null_test(&dict, &[(0, 1)], &config, 42);
        // p-value should be moderate (not 0) for a single pair
        assert!(
            result.adaptive_result.p_value > 0.0,
            "p=0 is impossible with Phipson-Smyth correction"
        );
    }

    #[test]
    fn test_codebook_null_test_all_pairs_high_p() {
        // When ALL pairs are included, the selective mean distance equals
        // the global mean distance, which is invariant under row permutation.
        // So p-value should be very high (the statistic never changes).
        let dict = sample_dictionary_4();
        let all_pairs: Vec<(usize, usize)> = (0..4)
            .flat_map(|i| ((i + 1)..4).map(move |j| (i, j)))
            .collect();
        let config = AdaptiveConfig {
            batch_size: 50,
            max_permutations: 200,
            alpha: 0.05,
            confidence: 0.95,
            min_permutations: 50,
        };
        let result = run_codebook_null_test(&dict, &all_pairs, &config, 42);
        // When all pairs included, statistic is invariant -> p ~ 1.0
        assert!(
            result.adaptive_result.p_value > 0.5,
            "all-pairs should yield high p-value (identity-like), got {}",
            result.adaptive_result.p_value
        );
    }

    #[test]
    fn test_codebook_null_result_fields() {
        let dict = sample_dictionary_4();
        let config = AdaptiveConfig {
            batch_size: 50,
            max_permutations: 100,
            alpha: 0.05,
            confidence: 0.95,
            min_permutations: 50,
        };
        let result = run_codebook_null_test(&dict, &[(1, 2)], &config, 123);
        assert_eq!(result.statistic_name, "mean_zd_pair_squared_distance");
        assert_eq!(result.n_basis, 4);
        assert_eq!(result.n_zd_pairs, 1);
    }

    // ====================================================================
    // Thesis I: Layer 5 integration validation tests
    // ====================================================================
    //
    // These tests validate that the NullModel abstraction (Layer 5)
    // correctly integrates with the encoding dictionary (Layer 1) and
    // ZD graph structure (Layer 3) through the adaptive sequential
    // testing framework. Each test exercises the full L0-L5 pipeline.

    /// Build a dim=16 (sedenion) dictionary from the first 16 Lambda_256
    /// lattice vectors. Sedenions are the smallest CD algebra with
    /// zero divisors, making them ideal for integration testing.
    fn sample_sedenion_dictionary() -> EncodingDictionary {
        let lambda = enumerate_lambda_256();
        assert!(lambda.len() >= 16, "Lambda_256 must have at least 16 vectors");
        let pairs: Vec<(usize, LatticeVector)> = lambda[..16]
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        EncodingDictionary::try_from_pairs(16, &pairs).unwrap()
    }

    #[test]
    fn test_thesis_i_codebook_null_column_independent() {
        // Thesis I test 4: Run codebook null test with ColumnIndependent
        // null via the generalized strategy API (Layer 5 trait dispatch).
        //
        // ColumnIndependent shuffles each lattice coordinate independently,
        // destroying inter-coordinate correlation. For lattice vectors in
        // {-1,0,1}^8, this creates chimeric vectors that break the
        // specific distance relationships between ZD-connected pairs.
        //
        // This test validates:
        // - L0/L1: EncodingDictionary construction from Lambda_256
        // - L3: ZD basis pair extraction from find_zero_divisors
        // - L5: NullModelStrategy trait dispatch with ColumnIndependentNull
        // - L5: Adaptive stopping engine integration
        let dict = sample_sedenion_dictionary();
        let zd_pairs = extract_zd_basis_pairs(16);
        assert!(
            !zd_pairs.is_empty(),
            "sedenions must have ZD-connected basis pairs"
        );

        let strategy = ColumnIndependentNull;
        let config = AdaptiveConfig {
            batch_size: 50,
            max_permutations: 500,
            alpha: 0.05,
            confidence: 0.95,
            min_permutations: 100,
        };

        let result = run_codebook_null_test_with_strategy(
            &dict, &zd_pairs, &strategy, &config, 42,
        );

        assert_eq!(result.n_basis, 16);
        assert_eq!(result.n_zd_pairs, zd_pairs.len());
        assert_eq!(result.dim, 16);
        assert!(result.observed_value > 0.0, "ZD pairs must have positive distance");
        assert!(
            result.adaptive_result.p_value > 0.0,
            "Phipson-Smyth guarantees p > 0"
        );
        assert!(
            result.adaptive_result.p_value <= 1.0,
            "p-value must be at most 1"
        );
        assert!(
            result.adaptive_result.n_permutations_used >= 100,
            "must run at least min_permutations"
        );
    }

    #[test]
    fn test_thesis_i_multi_null_codebook_comparison() {
        // Thesis I test 5: Compare ColumnIndependent vs RowPermutation
        // nulls on the same codebook data.
        //
        // For the selective_mean_squared_distance statistic on ZD-connected
        // pairs, BOTH nulls are informative (unlike the global fraction
        // statistic where RowPermutation is identity-equivalent):
        //
        // - ColumnIndependent: shuffles coordinates, creating chimeric
        //   lattice vectors. This breaks BOTH the labeling AND the point
        //   cloud structure, testing whether the observed correlation
        //   pattern is special.
        //
        // - RowPermutation: permutes which basis maps to which lattice
        //   point. This keeps the point cloud identical but randomizes
        //   which pair is "ZD-connected", testing whether the algebraic
        //   labeling is geometrically imprinted.
        //
        // Both should give valid p-values, but they test different
        // hypotheses about the encoding dictionary.
        let dict = sample_sedenion_dictionary();
        let zd_pairs = extract_zd_basis_pairs(16);

        let config = AdaptiveConfig {
            batch_size: 50,
            max_permutations: 500,
            alpha: 0.05,
            confidence: 0.95,
            min_permutations: 100,
        };

        // ColumnIndependent null
        let ci_strategy = ColumnIndependentNull;
        let ci_result = run_codebook_null_test_with_strategy(
            &dict, &zd_pairs, &ci_strategy, &config, 42,
        );

        // RowPermutation null (same as default run_codebook_null_test)
        let rp_result = run_codebook_null_test(&dict, &zd_pairs, &config, 42);

        // Both must produce valid results
        assert!(ci_result.adaptive_result.p_value > 0.0);
        assert!(ci_result.adaptive_result.p_value <= 1.0);
        assert!(rp_result.adaptive_result.p_value > 0.0);
        assert!(rp_result.adaptive_result.p_value <= 1.0);

        // Both must have the same observed statistic (same data, same pairs)
        assert!(
            (ci_result.observed_value - rp_result.observed_value).abs() < 1e-15,
            "observed statistic must be identical regardless of null model"
        );

        // Verify the strategy name propagates correctly
        assert_eq!(ci_result.statistic_name, "mean_zd_pair_squared_distance");
        assert_eq!(rp_result.statistic_name, "mean_zd_pair_squared_distance");
    }

    #[test]
    fn test_thesis_i_adaptive_speedup_invariant_statistic() {
        // Thesis I test 6: Verify adaptive engine stops early when the
        // test statistic is invariant under the null model.
        //
        // When ALL C(n,2) pairs are included, the mean squared distance
        // is a symmetric function of the pairwise distance multiset.
        // Under RowPermutation, all pairwise distances are preserved
        // (row permutation is a relabeling), so the statistic is
        // literally unchanged every permutation.
        //
        // The adaptive engine should detect this quickly: every null
        // realization yields null_stat == observed, so r/k ~ 1.0 and
        // p ~ 1.0, causing early stopping as NonSignificantEarly.
        let dict = sample_dictionary_4();
        let all_pairs: Vec<(usize, usize)> = (0..4)
            .flat_map(|i| ((i + 1)..4).map(move |j| (i, j)))
            .collect();

        let config = AdaptiveConfig {
            batch_size: 20,
            max_permutations: 2000,
            alpha: 0.05,
            confidence: 0.95,
            min_permutations: 40,
        };

        let result = run_codebook_null_test(&dict, &all_pairs, &config, 42);

        // p-value should be very high (statistic is invariant)
        assert!(
            result.adaptive_result.p_value > 0.5,
            "invariant statistic must yield high p, got {}",
            result.adaptive_result.p_value
        );

        // Adaptive should stop early (not exhaust all 2000 permutations)
        assert!(
            result.adaptive_result.stopped_early,
            "adaptive engine should stop early on invariant statistic"
        );
        assert!(
            result.adaptive_result.n_permutations_used < config.max_permutations,
            "used {} of {} permutations -- should have stopped early",
            result.adaptive_result.n_permutations_used,
            config.max_permutations
        );
        assert_eq!(
            result.adaptive_result.stop_reason,
            StopReason::NonSignificantEarly,
            "stop reason should be NonSignificantEarly for invariant statistic"
        );

        // Verify substantial speedup: should use far fewer than max
        let speedup_ratio =
            config.max_permutations as f64 / result.adaptive_result.n_permutations_used as f64;
        assert!(
            speedup_ratio > 2.0,
            "expected at least 2x speedup, got {:.1}x ({} of {} perms)",
            speedup_ratio,
            result.adaptive_result.n_permutations_used,
            config.max_permutations
        );
    }
}

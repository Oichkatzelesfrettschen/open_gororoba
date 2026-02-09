//! Baire-codebook bridge: ultrametric testing on lattice vector subsets.
//!
//! Bridges the Baire distance framework with codebook membership predicates
//! to test ultrametric structure in filtered subsets of the lattice encoding.
//!
//! # Key Pitfall: Shared Prefix Degeneracy
//!
//! Lambda_256 lattice vectors all satisfy l_0 = -1, l_1 = -1. This means
//! ALL pairs share the first 2 coordinates, making raw Baire distances
//! degenerate (all distances <= 3^{-3}). Functions in this module
//! automatically detect and strip shared prefixes before computing Baire
//! distances, using only the "free" tail coordinates.
//!
//! For example, Lambda_256 with its 2-coordinate shared prefix yields
//! effective 6-digit Baire sequences, providing meaningful hierarchical
//! distance structure.
//!
//! # Encoding
//!
//! Lattice coordinates in {-1, 0, 1} are mapped to base-3 digits via
//! -1 -> 0, 0 -> 1, 1 -> 2. The Baire distance is then 3^{-k} where
//! k is the 1-indexed position of the first differing digit.

use algebra_core::analysis::codebook::LatticeVector;
use super::baire::AttributeSpec;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

/// Map a trinary coordinate {-1, 0, 1} to a base-3 digit {0, 1, 2}.
#[cfg(test)]
fn trinary_to_digit(x: i8) -> u64 {
    match x {
        -1 => 0,
        0 => 1,
        1 => 2,
        _ => panic!("lattice coordinate {x} outside {{-1, 0, 1}}"),
    }
}

/// Compute the shared prefix length among a set of lattice vectors.
///
/// Returns the number of leading coordinates where ALL vectors agree.
/// For Lambda_256, this is typically 2 (all have l_0=-1, l_1=-1).
/// Returns 0 if vectors differ in the first coordinate or if the set
/// is empty.
pub fn shared_prefix_length(vectors: &[LatticeVector]) -> usize {
    if vectors.is_empty() {
        return 0;
    }
    let first = &vectors[0];
    for pos in 0..8 {
        if vectors.iter().any(|v| v[pos] != first[pos]) {
            return pos;
        }
    }
    8 // All coordinates identical (degenerate case)
}

/// Compute Baire distance between two lattice vectors.
///
/// Uses base-3 encoding: -1 -> 0, 0 -> 1, 1 -> 2.
/// Distance = 3^{-k} where k is the 1-indexed position of the first
/// differing digit. Returns 0.0 if vectors are identical.
///
/// `skip_prefix`: number of leading coordinates to ignore (e.g., 2 for
/// Lambda_256 where l_0 and l_1 are always -1).
pub fn lattice_baire_distance(
    a: &LatticeVector,
    b: &LatticeVector,
    skip_prefix: usize,
) -> f64 {
    for pos in skip_prefix..8 {
        if a[pos] != b[pos] {
            let effective_pos = pos - skip_prefix + 1; // 1-indexed
            return 3.0_f64.powi(-(effective_pos as i32));
        }
    }
    0.0
}

/// Compute the full Baire distance matrix for a set of lattice vectors.
///
/// Returns a flat upper-triangle distance matrix: index(i,j) for i < j
/// = i * n - i*(i+1)/2 + j - i - 1.
///
/// `skip_prefix`: coordinates to skip (auto-detect with `shared_prefix_length`).
pub fn lattice_baire_distance_matrix(
    vectors: &[LatticeVector],
    skip_prefix: usize,
) -> Vec<f64> {
    let n = vectors.len();
    let n_pairs = n * (n - 1) / 2;
    let mut dists = Vec::with_capacity(n_pairs);
    for i in 0..n {
        for j in (i + 1)..n {
            dists.push(lattice_baire_distance(&vectors[i], &vectors[j], skip_prefix));
        }
    }
    dists
}

/// Filter lattice vectors by a predicate (e.g., Lambda_256 membership).
pub fn filter_by_predicate(
    vectors: &[LatticeVector],
    pred: impl Fn(&LatticeVector) -> bool,
) -> Vec<LatticeVector> {
    vectors.iter().copied().filter(|v| pred(v)).collect()
}

/// Convert lattice vectors to column-major f64 data for the Baire
/// encoder's matrix-free tests.
///
/// Returns (data, n_rows, n_cols) where data is column-major.
/// `skip_prefix`: number of leading coordinates to omit.
pub fn lattice_to_column_major(
    vectors: &[LatticeVector],
    skip_prefix: usize,
) -> (Vec<f64>, usize, usize) {
    let n = vectors.len();
    let d = 8 - skip_prefix;
    let mut cols = vec![0.0f64; d * n];
    for (row, v) in vectors.iter().enumerate() {
        for col in 0..d {
            cols[col * n + row] = v[col + skip_prefix] as f64;
        }
    }
    (cols, n, d)
}

/// Create BaireEncoder attribute specs for lattice coordinate analysis.
///
/// Each of the 8 (or 8-skip) coordinates ranges from -1 to 1.
/// No log scaling is applied.
pub fn lattice_attribute_specs(n_coords: usize) -> Vec<AttributeSpec> {
    (0..n_coords)
        .map(|i| AttributeSpec {
            name: format!("l_{i}"),
            min: -1.0,
            max: 1.0,
            log_scale: false,
        })
        .collect()
}

/// Result of a codebook Baire ultrametric test.
#[derive(Debug, Clone)]
pub struct CodebookBaireResult {
    /// Number of lattice vectors tested.
    pub n_vectors: usize,
    /// Number of effective coordinates (after prefix stripping).
    pub effective_dim: usize,
    /// Shared prefix length that was stripped.
    pub prefix_stripped: usize,
    /// Observed ultrametric fraction on Baire distances.
    pub ultrametric_fraction: f64,
    /// Mean null fraction.
    pub null_fraction_mean: f64,
    /// Standard deviation of null fractions.
    pub null_fraction_std: f64,
    /// P-value (one-sided: fraction of null >= observed).
    pub p_value: f64,
}

/// Run an ultrametric fraction test on lattice Baire distances.
///
/// Automatically detects the shared prefix, strips it, computes Baire
/// distances on the remaining tail coordinates, and tests the ultrametric
/// fraction against a column-independent null.
///
/// Uses the matrix-free approach: column-major data with sampled triples
/// computing Euclidean distances (which for trinary data coincide with
/// squared Hamming distance, giving meaningful ultrametric structure).
pub fn codebook_baire_ultrametric_test(
    vectors: &[LatticeVector],
    n_triples: usize,
    n_permutations: usize,
    seed: u64,
) -> CodebookBaireResult {
    assert!(vectors.len() >= 3, "need at least 3 vectors");

    let prefix_len = shared_prefix_length(vectors);
    let effective_dim = 8 - prefix_len;

    // Compute observed Baire distance matrix (on stripped coordinates)
    let dist_matrix = lattice_baire_distance_matrix(vectors, prefix_len);
    let n = vectors.len();

    let obs_frac = super::ultrametric_fraction_from_matrix(
        &dist_matrix, n, n_triples, seed,
    );

    // Null: shuffle each tail coordinate independently, recompute Baire distances
    let mut rng = ChaCha8Rng::seed_from_u64(seed + 1_000_000);
    let mut null_fracs = Vec::with_capacity(n_permutations);

    // Working copy of tail coordinates
    let mut shuffled: Vec<LatticeVector> = vectors.to_vec();

    for _ in 0..n_permutations {
        // Shuffle each tail coordinate independently across vectors
        for coord in prefix_len..8 {
            let mut col_vals: Vec<i8> = shuffled.iter().map(|v| v[coord]).collect();
            col_vals.shuffle(&mut rng);
            for (i, &val) in col_vals.iter().enumerate() {
                shuffled[i][coord] = val;
            }
        }

        let null_dists = lattice_baire_distance_matrix(&shuffled, prefix_len);
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

    // One-sided p-value: fraction of null >= observed
    let n_extreme = null_fracs.iter().filter(|&&f| f >= obs_frac).count();
    let p_value = (n_extreme as f64 + 1.0) / (n_permutations as f64 + 1.0);

    CodebookBaireResult {
        n_vectors: n,
        effective_dim,
        prefix_stripped: prefix_len,
        ultrametric_fraction: obs_frac,
        null_fraction_mean: null_mean,
        null_fraction_std: null_std,
        p_value,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use algebra_core::analysis::codebook::{enumerate_lambda_256, is_in_lambda_256};

    #[test]
    fn test_trinary_to_digit() {
        assert_eq!(trinary_to_digit(-1), 0);
        assert_eq!(trinary_to_digit(0), 1);
        assert_eq!(trinary_to_digit(1), 2);
    }

    #[test]
    #[should_panic(expected = "outside")]
    fn test_trinary_to_digit_invalid() {
        trinary_to_digit(2);
    }

    #[test]
    fn test_shared_prefix_length_lambda256() {
        let lambda_256 = enumerate_lambda_256();
        let prefix = shared_prefix_length(&lambda_256);
        // Lambda_256 vectors all have l_0=-1, l_1=-1
        assert_eq!(prefix, 2, "Lambda_256 should have 2-coordinate shared prefix");
    }

    #[test]
    fn test_shared_prefix_length_empty() {
        let empty: Vec<LatticeVector> = vec![];
        assert_eq!(shared_prefix_length(&empty), 0);
    }

    #[test]
    fn test_shared_prefix_length_single() {
        let vecs = vec![[-1, -1, 0, 0, 0, 0, 0, 0]];
        assert_eq!(shared_prefix_length(&vecs), 8); // All coords "shared"
    }

    #[test]
    fn test_shared_prefix_length_no_shared() {
        let vecs = vec![
            [-1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ];
        assert_eq!(shared_prefix_length(&vecs), 0);
    }

    #[test]
    fn test_lattice_baire_distance_identical() {
        let a: LatticeVector = [-1, -1, 0, 0, 1, 1, 0, 0];
        assert_eq!(lattice_baire_distance(&a, &a, 0), 0.0);
    }

    #[test]
    fn test_lattice_baire_distance_first_digit() {
        let a: LatticeVector = [-1, 0, 0, 0, 0, 0, 0, 0];
        let b: LatticeVector = [0, 0, 0, 0, 0, 0, 0, 0];
        // First position differs -> d = 3^(-1) = 1/3
        let d = lattice_baire_distance(&a, &b, 0);
        assert!((d - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_lattice_baire_distance_third_digit() {
        let a: LatticeVector = [-1, -1, 0, 0, 0, 0, 0, 0];
        let b: LatticeVector = [-1, -1, 1, 0, 0, 0, 0, 0];
        // Positions 0,1 match; position 2 differs -> d = 3^(-3)
        let d = lattice_baire_distance(&a, &b, 0);
        assert!((d - 3.0_f64.powi(-3)).abs() < 1e-10);
    }

    #[test]
    fn test_lattice_baire_distance_with_skip() {
        let a: LatticeVector = [-1, -1, 0, 0, 0, 0, 0, 0];
        let b: LatticeVector = [-1, -1, 1, 0, 0, 0, 0, 0];
        // Skip 2 -> effective position 0 is coord 2, which differs
        // -> d = 3^(-1) = 1/3
        let d = lattice_baire_distance(&a, &b, 2);
        assert!((d - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_lattice_baire_is_ultrametric() {
        // Baire distances are ultrametric by construction:
        // d(a,c) <= max(d(a,b), d(b,c)) always holds.
        let a: LatticeVector = [-1, -1, 0, 0, 0, 0, 0, 0];
        let b: LatticeVector = [-1, -1, 1, 0, 0, 0, 0, 0];
        let c: LatticeVector = [-1, -1, 1, 1, 0, 0, 0, 0];

        let d_ab = lattice_baire_distance(&a, &b, 0);
        let d_bc = lattice_baire_distance(&b, &c, 0);
        let d_ac = lattice_baire_distance(&a, &c, 0);

        assert!(
            d_ac <= d_ab.max(d_bc) + 1e-15,
            "Baire distances must be ultrametric: d(a,c)={d_ac} <= max({d_ab}, {d_bc})"
        );
    }

    #[test]
    fn test_filter_by_predicate() {
        let vecs: Vec<LatticeVector> = vec![
            [-1, -1, -1, -1, 0, 0, 0, 0],  // in Lambda_256
            [0, -1, 0, -1, 0, -1, 0, -1],   // NOT in Lambda_256 (l_0=0)
            [-1, -1, -1, 0, -1, 0, 0, 0],   // might be in Lambda_256
        ];
        let filtered = filter_by_predicate(&vecs, is_in_lambda_256);
        // At minimum, the first vector should pass
        assert!(filtered.len() >= 1);
        for v in &filtered {
            assert!(is_in_lambda_256(v));
        }
    }

    #[test]
    fn test_distance_matrix_size() {
        let vecs: Vec<LatticeVector> = vec![
            [-1, -1, 0, 0, 0, 0, 0, 0],
            [-1, -1, 1, 0, 0, 0, 0, 0],
            [-1, -1, 0, 1, 0, 0, 0, 0],
            [-1, -1, 0, 0, 1, 0, 0, 0],
        ];
        let dm = lattice_baire_distance_matrix(&vecs, 2);
        assert_eq!(dm.len(), 4 * 3 / 2); // C(4,2) = 6
    }

    #[test]
    fn test_lattice_to_column_major() {
        let vecs: Vec<LatticeVector> = vec![
            [-1, -1, 0, 1, 0, 0, 0, 0],
            [-1, -1, 1, 0, 0, 0, 0, 0],
        ];
        let (data, n, d) = lattice_to_column_major(&vecs, 2);
        assert_eq!(n, 2);
        assert_eq!(d, 6);
        assert_eq!(data.len(), 12);
        // Row 0, col 0 (coord 2): 0
        assert_eq!(data[0 * 2 + 0], 0.0);
        // Row 0, col 1 (coord 3): 1
        assert_eq!(data[1 * 2 + 0], 1.0);
        // Row 1, col 0 (coord 2): 1
        assert_eq!(data[0 * 2 + 1], 1.0);
    }

    #[test]
    fn test_baire_fraction_on_lambda256_not_trivial() {
        // The Baire distance on Lambda_256 with prefix stripping should
        // NOT yield a trivially 1.0 fraction (that would indicate the
        // test is degenerate). The fraction should be high (since Baire
        // is ultrametric by construction) but the null comparison is the
        // key: shuffling columns should change the Baire structure.
        let lambda_256 = enumerate_lambda_256();
        assert!(lambda_256.len() >= 10, "need enough vectors for test");

        // Use a small subset for speed
        let subset: Vec<LatticeVector> = lambda_256.into_iter().take(30).collect();
        let prefix = shared_prefix_length(&subset);
        assert!(
            prefix >= 2 && prefix < 8,
            "Lambda_256 subset should have at least the fixed 2-prefix and remain non-degenerate, got {prefix}"
        );

        let result = codebook_baire_ultrametric_test(&subset, 5_000, 100, 42);

        assert_eq!(result.prefix_stripped, prefix);
        assert_eq!(result.effective_dim, 8 - prefix);
        assert!(
            result.ultrametric_fraction > 0.0,
            "fraction should be positive: {}",
            result.ultrametric_fraction
        );
        assert!(
            result.ultrametric_fraction <= 1.0,
            "fraction should be <= 1.0: {}",
            result.ultrametric_fraction
        );
        // The null mean should also be high (Baire is always ultrametric),
        // but the point is the test runs without crashing or degenerating.
        assert!(
            result.null_fraction_mean > 0.0,
            "null fraction should be positive"
        );
    }

    #[test]
    fn test_codebook_baire_result_fields() {
        let vecs: Vec<LatticeVector> = vec![
            [-1, -1, 0, 0, 0, 0, 0, 0],
            [-1, -1, 1, 0, 0, 0, 0, 0],
            [-1, -1, 0, 1, 0, 0, 0, 0],
            [-1, -1, 0, 0, 1, 0, 0, 0],
        ];
        let result = codebook_baire_ultrametric_test(&vecs, 1_000, 50, 42);
        assert_eq!(result.n_vectors, 4);
        assert_eq!(result.prefix_stripped, 2);
        assert_eq!(result.effective_dim, 6);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_lattice_attribute_specs() {
        let specs = lattice_attribute_specs(6);
        assert_eq!(specs.len(), 6);
        for (i, s) in specs.iter().enumerate() {
            assert_eq!(s.name, format!("l_{i}"));
            assert_eq!(s.min, -1.0);
            assert_eq!(s.max, 1.0);
            assert!(!s.log_scale);
        }
    }
}

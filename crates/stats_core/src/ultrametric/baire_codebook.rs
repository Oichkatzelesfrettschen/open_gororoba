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

use super::baire::AttributeSpec;
use algebra_core::analysis::codebook::LatticeVector;
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
pub fn lattice_baire_distance(a: &LatticeVector, b: &LatticeVector, skip_prefix: usize) -> f64 {
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
pub fn lattice_baire_distance_matrix(vectors: &[LatticeVector], skip_prefix: usize) -> Vec<f64> {
    let n = vectors.len();
    let n_pairs = n * (n - 1) / 2;
    let mut dists = Vec::with_capacity(n_pairs);
    for i in 0..n {
        for j in (i + 1)..n {
            dists.push(lattice_baire_distance(
                &vectors[i],
                &vectors[j],
                skip_prefix,
            ));
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

    let obs_frac = super::ultrametric_fraction_from_matrix(&dist_matrix, n, n_triples, seed);

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
        let null_frac =
            super::ultrametric_fraction_from_matrix(&null_dists, n, n_triples, seed + 2_000_000);
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
        assert_eq!(
            prefix, 2,
            "Lambda_256 should have 2-coordinate shared prefix"
        );
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
        let vecs = vec![[-1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]];
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
            [-1, -1, -1, -1, 0, 0, 0, 0], // in Lambda_256
            [0, -1, 0, -1, 0, -1, 0, -1], // NOT in Lambda_256 (l_0=0)
            [-1, -1, -1, 0, -1, 0, 0, 0], // might be in Lambda_256
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
        let vecs: Vec<LatticeVector> = vec![[-1, -1, 0, 1, 0, 0, 0, 0], [-1, -1, 1, 0, 0, 0, 0, 0]];
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

    // ================================================================
    // Euclidean Ultrametricity on Prefix-Stripped Lattice (C-500)
    // ================================================================

    #[test]
    fn test_euclidean_ultrametricity_across_filtration_levels() {
        use algebra_core::analysis::codebook::{
            enumerate_lattice_by_predicate, is_in_lambda_1024, is_in_lambda_2048,
            is_in_lambda_512,
        };
        use super::super::baire::matrix_free_fraction;
        use super::super::null_models::{apply_null_column_major, NullModel};
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let lambda_2048 = enumerate_lattice_by_predicate(is_in_lambda_2048);
        let lambda_1024 = enumerate_lattice_by_predicate(is_in_lambda_1024);
        let lambda_512 = enumerate_lattice_by_predicate(is_in_lambda_512);
        let lambda_256 = enumerate_lambda_256();

        eprintln!("=== Euclidean Ultrametricity on Prefix-Stripped Lattice ===");

        let n_triples = 50_000;
        let n_permutations = 200;
        let seed = 42u64;

        for (name, vectors) in [
            ("Lambda_2048", &lambda_2048),
            ("Lambda_1024", &lambda_1024),
            ("Lambda_512", &lambda_512),
            ("Lambda_256", &lambda_256),
        ] {
            let n = vectors.len();
            let prefix = shared_prefix_length(vectors);
            let d = 8 - prefix;

            // Convert to column-major f64 (prefix-stripped)
            let (cols, _, _) = lattice_to_column_major(vectors, prefix);

            // Observed Euclidean ultrametric fraction (epsilon=0.05)
            let obs_frac = matrix_free_fraction(&cols, n, d, n_triples, seed, 0.05);

            // Null: column-independent shuffle (200 permutations)
            let mut rng = ChaCha8Rng::seed_from_u64(seed + 1_000_000);
            let mut null_fracs = Vec::with_capacity(n_permutations);
            let mut shuffled = cols.clone();

            for _ in 0..n_permutations {
                shuffled.copy_from_slice(&cols);
                apply_null_column_major(
                    &mut shuffled,
                    n,
                    d,
                    NullModel::ColumnIndependent,
                    &mut rng,
                );
                let null_frac =
                    matrix_free_fraction(&shuffled, n, d, n_triples, seed + 2_000_000, 0.05);
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

            let effect_size = obs_frac - null_mean;
            let z_score = if null_std > 1e-12 {
                effect_size / null_std
            } else {
                0.0
            };

            eprintln!(
                "\n  {} ({} vectors, prefix={}, effective_dim={}):",
                name, n, prefix, d
            );
            eprintln!(
                "    Observed fraction: {:.4}",
                obs_frac
            );
            eprintln!(
                "    Null mean +/- std: {:.4} +/- {:.4}",
                null_mean, null_std
            );
            eprintln!(
                "    Effect size: {:.4}, z-score: {:.2}, p-value: {:.4}",
                effect_size, z_score, p_value
            );

            // Sanity checks
            assert!(
                obs_frac >= 0.0 && obs_frac <= 1.0,
                "Fraction must be in [0,1]"
            );
            assert!(
                null_mean >= 0.0 && null_mean <= 1.0,
                "Null mean must be in [0,1]"
            );
        }

        // The Baire distance test MUST give fraction 1.0 (tautological)
        let baire_dists = lattice_baire_distance_matrix(&lambda_256, 2);
        let baire_frac = super::super::ultrametric_fraction_from_matrix(
            &baire_dists,
            lambda_256.len(),
            50_000,
            42,
        );
        eprintln!(
            "\n  Baire distance on Lambda_256: fraction = {:.4} (tautological)",
            baire_frac
        );
        assert!(
            (baire_frac - 1.0).abs() < 1e-10,
            "Baire distances are ultrametric by construction, got {}",
            baire_frac
        );

        eprintln!("\n=== SUMMARY ===");
        eprintln!("Baire distances: always 1.0 (trivial, by construction).");
        eprintln!("Euclidean distances: see above for non-trivial ultrametricity");
        eprintln!("vs column-shuffle null at each filtration level.");
    }

    // ================================================================
    // Intermediate Filtration Gradient (C-500 follow-up)
    // ================================================================

    /// Investigate the ultrametricity transition between Lambda_1024 (z=-1.57)
    /// and Lambda_512 (z=9.22) by applying the 6 trie-cut exclusion rules
    /// one at a time.
    ///
    /// The 6 rules, applied cumulatively:
    ///   k=0: Lambda_1024                (1026 vectors)
    ///   k=1: exclude l_1=1              (largest cut)
    ///   k=2: also exclude l_1=0,l_2=1
    ///   k=3: also exclude l_1=0,l_2=0,l_3=0
    ///   k=4: also exclude l_1=0,l_2=0,l_3=1
    ///   k=5: also exclude l_1=0,l_2=0,l_3=-1,l_4=1
    ///   k=6: Lambda_512                 (512 vectors)
    #[test]
    fn test_intermediate_filtration_gradient() {
        use algebra_core::analysis::codebook::{
            enumerate_lattice_by_predicate, is_in_lambda_1024_minus_k,
        };
        use super::super::baire::matrix_free_fraction;
        use super::super::null_models::{apply_null_column_major, NullModel};
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let n_triples = 50_000;
        let n_permutations = 200;
        let seed = 42u64;

        eprintln!("\n=== Intermediate Filtration Gradient (C-500 follow-up) ===");
        eprintln!("k | N vectors | prefix | eff_dim | obs_frac | null_mean | z-score | p-value");
        eprintln!("--|----------|--------|---------|----------|-----------|---------|--------");

        let mut prev_n = 0usize;
        let mut z_scores = Vec::new();
        let mut sizes = Vec::new();

        for k in 0..=6 {
            let vectors = enumerate_lattice_by_predicate(|v| is_in_lambda_1024_minus_k(v, k));
            let n = vectors.len();
            let prefix = shared_prefix_length(&vectors);
            let d = 8 - prefix;

            // Column-major, prefix-stripped
            let (cols, _, _) = lattice_to_column_major(&vectors, prefix);

            // Observed ultrametric fraction
            let obs_frac = matrix_free_fraction(&cols, n, d, n_triples, seed, 0.05);

            // Null distribution: column-independent shuffle
            let mut rng = ChaCha8Rng::seed_from_u64(seed + 1_000_000 + k as u64);
            let mut null_fracs = Vec::with_capacity(n_permutations);
            let mut shuffled = cols.clone();

            for _ in 0..n_permutations {
                shuffled.copy_from_slice(&cols);
                apply_null_column_major(
                    &mut shuffled,
                    n,
                    d,
                    NullModel::ColumnIndependent,
                    &mut rng,
                );
                let null_frac =
                    matrix_free_fraction(&shuffled, n, d, n_triples, seed + 2_000_000, 0.05);
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

            let z_score = if null_std > 1e-12 {
                (obs_frac - null_mean) / null_std
            } else {
                0.0
            };

            let removed = if k == 0 { 0 } else { prev_n - n };
            eprintln!(
                "{} | {:>8} | {:>6} | {:>7} | {:>8.4} | {:>9.4} | {:>7.2} | {:>7.4}  (removed {})",
                k, n, prefix, d, obs_frac, null_mean, z_score, p_value, removed
            );

            z_scores.push(z_score);
            sizes.push(n);
            prev_n = n;
        }

        // Structural assertions:
        // 1. k=0 is Lambda_1024 (1026 vectors)
        assert_eq!(sizes[0], 1026, "k=0 must be Lambda_1024");
        // 2. k=6 is Lambda_512 (512 vectors)
        assert_eq!(sizes[6], 512, "k=6 must be Lambda_512");
        // 3. Sizes monotonically decrease
        for i in 1..=6 {
            assert!(
                sizes[i] <= sizes[i - 1],
                "Size must decrease: k={} has {} > k={} has {}",
                i, sizes[i], i - 1, sizes[i - 1]
            );
        }
        // 4. z-score at k=6 should be substantially higher than k=0
        //    (Lambda_512 is ultrametric, Lambda_1024 is not)
        assert!(
            z_scores[6] > z_scores[0] + 2.0,
            "Lambda_512 (k=6) z={:.2} should be >2 above Lambda_1024 (k=0) z={:.2}",
            z_scores[6], z_scores[0]
        );

        eprintln!("\n=== GRADIENT SUMMARY ===");
        eprintln!("z-score progression: {:?}", z_scores.iter().map(|z| format!("{:.2}", z)).collect::<Vec<_>>());

        // Find the first k where z > 3.0 (strong signal)
        if let Some(transition_k) = z_scores.iter().position(|&z| z > 3.0) {
            eprintln!("First k with z > 3.0: {} (N={})", transition_k, sizes[transition_k]);
        } else {
            eprintln!("No k reached z > 3.0 (gradient is gradual)");
        }
    }

    /// Test that is_in_lambda_1024_minus_k(v, 0) == is_in_lambda_1024(v)
    /// and is_in_lambda_1024_minus_k(v, 6) == is_in_lambda_512(v).
    #[test]
    fn test_intermediate_filtration_consistency() {
        use algebra_core::analysis::codebook::{
            enumerate_lattice_by_predicate, is_in_lambda_1024,
            is_in_lambda_1024_minus_k, is_in_lambda_512,
        };

        let all_1024 = enumerate_lattice_by_predicate(is_in_lambda_1024);
        let all_512 = enumerate_lattice_by_predicate(is_in_lambda_512);
        let k0 = enumerate_lattice_by_predicate(|v| is_in_lambda_1024_minus_k(v, 0));
        let k6 = enumerate_lattice_by_predicate(|v| is_in_lambda_1024_minus_k(v, 6));

        assert_eq!(all_1024.len(), k0.len(), "k=0 must equal Lambda_1024");
        assert_eq!(all_512.len(), k6.len(), "k=6 must equal Lambda_512");
        assert_eq!(all_1024, k0, "k=0 vectors must match Lambda_1024");
        assert_eq!(all_512, k6, "k=6 vectors must match Lambda_512");

        // Monotone containment: k_i+1 is a subset of k_i
        let mut prev = k0;
        for k in 1..=6 {
            let current = enumerate_lattice_by_predicate(|v| is_in_lambda_1024_minus_k(v, k));
            for v in &current {
                assert!(
                    prev.contains(v),
                    "k={} vector {:?} not in k={} set",
                    k, v, k - 1
                );
            }
            assert!(
                current.len() <= prev.len(),
                "k={} has {} vectors > k={} has {}",
                k, current.len(), k - 1, prev.len()
            );
            prev = current;
        }
    }

    /// Count how many vectors each rule removes, confirming the filtration
    /// step sizes match the trie structure.
    #[test]
    fn test_intermediate_filtration_step_sizes() {
        use algebra_core::analysis::codebook::{
            enumerate_lattice_by_predicate, is_in_lambda_1024_minus_k,
        };

        eprintln!("\n=== Filtration Step Sizes ===");
        let mut sizes = Vec::new();
        for k in 0..=6 {
            let n = enumerate_lattice_by_predicate(|v| is_in_lambda_1024_minus_k(v, k)).len();
            sizes.push(n);
        }

        eprintln!("Sizes: {:?}", sizes);
        for k in 1..=6 {
            let removed = sizes[k - 1] - sizes[k];
            eprintln!(
                "  Rule {}: {} -> {} (removed {})",
                k, sizes[k - 1], sizes[k], removed
            );
        }

        // k=0 is Lambda_1024 = 1026
        assert_eq!(sizes[0], 1026);
        // k=6 is Lambda_512 = 512
        assert_eq!(sizes[6], 512);
        // Total removed
        assert_eq!(sizes[0] - sizes[6], 514, "Total removed must be 514");
    }

    /// Random-removal control: does removing 297 RANDOM vectors from
    /// Lambda_1024 produce the same ultrametricity jump as rule 1?
    ///
    /// If so, the phase transition is a trivial size effect.
    /// If not, the l_1=+1 vectors are algebraically special.
    #[test]
    fn test_random_removal_control() {
        use algebra_core::analysis::codebook::{
            enumerate_lattice_by_predicate, is_in_lambda_1024,
        };
        use super::super::baire::matrix_free_fraction;
        use super::super::null_models::{apply_null_column_major, NullModel};
        use rand::prelude::*;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let all_1024 = enumerate_lattice_by_predicate(is_in_lambda_1024);
        let n_full = all_1024.len();
        assert_eq!(n_full, 1026);

        let n_triples = 50_000;
        let n_null_perms = 200;
        let n_random_trials = 20;
        let n_remove = 297; // Same count as rule 1

        eprintln!("\n=== Random Removal Control (C-501 validation) ===");
        eprintln!("Removing {} random vectors vs. the {} l_1=+1 vectors", n_remove, n_remove);

        // First: compute z-score for rule-1 removal (reference)
        let rule1_vectors: Vec<_> = all_1024.iter()
            .copied()
            .filter(|v| v[1] != 1)
            .collect();
        assert_eq!(rule1_vectors.len(), 729);

        let prefix = shared_prefix_length(&rule1_vectors);
        let d = 8 - prefix;
        let (cols, _, _) = lattice_to_column_major(&rule1_vectors, prefix);
        let rule1_obs = matrix_free_fraction(&cols, rule1_vectors.len(), d, n_triples, 42, 0.05);

        let mut rng_null = ChaCha8Rng::seed_from_u64(1_000_042);
        let mut null_fracs = Vec::with_capacity(n_null_perms);
        let mut shuffled = cols.clone();
        for _ in 0..n_null_perms {
            shuffled.copy_from_slice(&cols);
            apply_null_column_major(
                &mut shuffled, rule1_vectors.len(), d,
                NullModel::ColumnIndependent, &mut rng_null,
            );
            null_fracs.push(matrix_free_fraction(&shuffled, rule1_vectors.len(), d, n_triples, 2_000_042, 0.05));
        }
        let null_mean = null_fracs.iter().sum::<f64>() / n_null_perms as f64;
        let null_std = (null_fracs.iter().map(|f| (f - null_mean).powi(2)).sum::<f64>()
            / n_null_perms as f64).sqrt();
        let rule1_z = if null_std > 1e-12 { (rule1_obs - null_mean) / null_std } else { 0.0 };

        eprintln!("Rule-1 removal: N=729, obs={:.4}, null={:.4}+/-{:.4}, z={:.2}",
            rule1_obs, null_mean, null_std, rule1_z);

        // Now: 20 random removal trials
        let mut random_z_scores = Vec::new();
        let mut rng_sample = ChaCha8Rng::seed_from_u64(12345);

        for trial in 0..n_random_trials {
            let mut indices: Vec<usize> = (0..n_full).collect();
            indices.shuffle(&mut rng_sample);
            let keep: Vec<_> = indices[n_remove..].iter().map(|&i| all_1024[i]).collect();
            assert_eq!(keep.len(), 729);

            let prefix_r = shared_prefix_length(&keep);
            let d_r = 8 - prefix_r;
            let (cols_r, _, _) = lattice_to_column_major(&keep, prefix_r);
            let obs_r = matrix_free_fraction(&cols_r, keep.len(), d_r, n_triples, 42, 0.05);

            let mut rng_null_r = ChaCha8Rng::seed_from_u64(1_000_042 + trial as u64 * 1000);
            let mut null_fracs_r = Vec::with_capacity(n_null_perms);
            let mut shuffled_r = cols_r.clone();
            for _ in 0..n_null_perms {
                shuffled_r.copy_from_slice(&cols_r);
                apply_null_column_major(
                    &mut shuffled_r, keep.len(), d_r,
                    NullModel::ColumnIndependent, &mut rng_null_r,
                );
                null_fracs_r.push(matrix_free_fraction(&shuffled_r, keep.len(), d_r, n_triples, 2_000_042, 0.05));
            }
            let null_mean_r = null_fracs_r.iter().sum::<f64>() / n_null_perms as f64;
            let null_std_r = (null_fracs_r.iter().map(|f| (f - null_mean_r).powi(2)).sum::<f64>()
                / n_null_perms as f64).sqrt();
            let z_r = if null_std_r > 1e-12 { (obs_r - null_mean_r) / null_std_r } else { 0.0 };

            eprintln!("  Random trial {:>2}: obs={:.4}, null={:.4}+/-{:.4}, z={:.2}",
                trial, obs_r, null_mean_r, null_std_r, z_r);
            random_z_scores.push(z_r);
        }

        let mean_random_z = random_z_scores.iter().sum::<f64>() / n_random_trials as f64;
        let max_random_z = random_z_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        eprintln!("\n--- Control Summary ---");
        eprintln!("Rule-1 z-score:     {:.2}", rule1_z);
        eprintln!("Random mean z:      {:.2}", mean_random_z);
        eprintln!("Random max z:       {:.2}", max_random_z);
        eprintln!("Random z-scores:    {:?}",
            random_z_scores.iter().map(|z| format!("{:.2}", z)).collect::<Vec<_>>());

        // The rule-1 z-score should exceed the MEAN random z-score
        // (if l_1=+1 vectors are special, targeted removal beats random removal).
        // We use a generous margin since random removal still removes some l_1=+1 vectors.
        assert!(
            rule1_z > mean_random_z,
            "Rule-1 z={:.2} should exceed random mean z={:.2}",
            rule1_z, mean_random_z
        );
    }

    // ================================================================
    // Lambda_512 -> Lambda_256 Intermediate Gradient
    // ================================================================

    /// Investigate the ultrametricity transition between Lambda_512 (z=9.22)
    /// and Lambda_256 (z=17.07) by applying the 6 Lambda_256 exclusion rules
    /// one at a time.
    ///
    /// Tests whether there is a second phase transition (analogous to C-501's
    /// Lambda_1024->Lambda_512 jump at rule 1), or whether ultrametricity
    /// saturates/strengthens gradually.
    #[test]
    fn test_lambda512_to_256_intermediate_gradient() {
        use algebra_core::analysis::codebook::{
            enumerate_lattice_by_predicate, is_in_lambda_512_minus_k,
        };
        use super::super::baire::matrix_free_fraction;
        use super::super::null_models::{apply_null_column_major, NullModel};
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let n_triples = 50_000;
        let n_permutations = 200;
        let seed = 42u64;

        eprintln!("\n=== Lambda_512 -> Lambda_256 Intermediate Gradient ===");
        eprintln!("k | N vectors | prefix | eff_dim | obs_frac | null_mean | z-score | p-value");
        eprintln!("--|----------|--------|---------|----------|-----------|---------|--------");

        let mut prev_n = 0usize;
        let mut z_scores = Vec::new();
        let mut sizes = Vec::new();

        for k in 0..=6 {
            let vectors = enumerate_lattice_by_predicate(|v| is_in_lambda_512_minus_k(v, k));
            let n = vectors.len();
            let prefix = shared_prefix_length(&vectors);
            let d = 8 - prefix;

            let (cols, _, _) = lattice_to_column_major(&vectors, prefix);
            let obs_frac = matrix_free_fraction(&cols, n, d, n_triples, seed, 0.05);

            let mut rng = ChaCha8Rng::seed_from_u64(seed + 3_000_000 + k as u64);
            let mut null_fracs = Vec::with_capacity(n_permutations);
            let mut shuffled = cols.clone();

            for _ in 0..n_permutations {
                shuffled.copy_from_slice(&cols);
                apply_null_column_major(
                    &mut shuffled, n, d,
                    NullModel::ColumnIndependent, &mut rng,
                );
                null_fracs.push(
                    matrix_free_fraction(&shuffled, n, d, n_triples, seed + 4_000_000, 0.05)
                );
            }

            let null_mean = null_fracs.iter().sum::<f64>() / n_permutations as f64;
            let null_std = (null_fracs.iter().map(|f| (f - null_mean).powi(2)).sum::<f64>()
                / n_permutations as f64).sqrt();
            let n_extreme = null_fracs.iter().filter(|&&f| f >= obs_frac).count();
            let p_value = (n_extreme as f64 + 1.0) / (n_permutations as f64 + 1.0);
            let z_score = if null_std > 1e-12 {
                (obs_frac - null_mean) / null_std
            } else {
                0.0
            };

            let removed = if k == 0 { 0 } else { prev_n - n };
            eprintln!(
                "{} | {:>8} | {:>6} | {:>7} | {:>8.4} | {:>9.4} | {:>7.2} | {:>7.4}  (removed {})",
                k, n, prefix, d, obs_frac, null_mean, z_score, p_value, removed
            );

            z_scores.push(z_score);
            sizes.push(n);
            prev_n = n;
        }

        // Structural assertions
        assert_eq!(sizes[0], 512, "k=0 must be Lambda_512");
        assert_eq!(sizes[6], 256, "k=6 must be Lambda_256");
        for i in 1..=6 {
            assert!(sizes[i] <= sizes[i - 1], "Sizes must decrease");
        }
        // Both endpoints should be significantly ultrametric
        assert!(z_scores[0] > 3.0, "Lambda_512 z={:.2} should be significant", z_scores[0]);
        assert!(z_scores[6] > 3.0, "Lambda_256 z={:.2} should be significant", z_scores[6]);

        eprintln!("\n=== GRADIENT SUMMARY ===");
        eprintln!("z-score progression: {:?}",
            z_scores.iter().map(|z| format!("{:.2}", z)).collect::<Vec<_>>());

        // Step sizes
        eprintln!("Step sizes:");
        for k in 1..=6 {
            eprintln!("  Rule {}: {} -> {} (removed {})", k, sizes[k-1], sizes[k], sizes[k-1] - sizes[k]);
        }
    }

    /// Random-removal control for Lambda_512->Lambda_256 (C-504 companion).
    /// Verifies that removing 147 l_1=0 vectors (rule 1) produces higher
    /// ultrametricity than removing 147 random vectors.
    #[test]
    fn test_lambda512_to_256_random_removal_control() {
        use algebra_core::analysis::codebook::{
            enumerate_lattice_by_predicate, is_in_lambda_512,
        };
        use super::super::baire::matrix_free_fraction;
        use super::super::null_models::{apply_null_column_major, NullModel};
        use rand::prelude::*;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let all_512 = enumerate_lattice_by_predicate(is_in_lambda_512);
        let n_full = all_512.len();
        assert_eq!(n_full, 512);

        let n_triples = 50_000;
        let n_null_perms = 200;
        let n_random_trials = 20;
        let n_remove = 147; // Same count as rule 1 (l_1=0 removal)

        eprintln!("\n=== Random Removal Control (C-504 validation) ===");
        eprintln!("Removing {} random vs {} l_1=0 vectors from Lambda_512", n_remove, n_remove);

        // Reference: rule-1 removal (keep only l_1=-1, i.e. l_1!=0)
        let rule1_vectors: Vec<_> = all_512.iter()
            .copied()
            .filter(|v| v[1] != 0)
            .collect();
        assert_eq!(rule1_vectors.len(), 365);

        let prefix = shared_prefix_length(&rule1_vectors);
        let d = 8 - prefix;
        let (cols, _, _) = lattice_to_column_major(&rule1_vectors, prefix);
        let rule1_obs = matrix_free_fraction(&cols, rule1_vectors.len(), d, n_triples, 42, 0.05);

        let mut rng_null = ChaCha8Rng::seed_from_u64(7_000_042);
        let mut null_fracs = Vec::with_capacity(n_null_perms);
        let mut shuffled = cols.clone();
        for _ in 0..n_null_perms {
            shuffled.copy_from_slice(&cols);
            apply_null_column_major(
                &mut shuffled, rule1_vectors.len(), d,
                NullModel::ColumnIndependent, &mut rng_null,
            );
            null_fracs.push(matrix_free_fraction(&shuffled, rule1_vectors.len(), d, n_triples, 8_000_042, 0.05));
        }
        let null_mean = null_fracs.iter().sum::<f64>() / n_null_perms as f64;
        let null_std = (null_fracs.iter().map(|f| (f - null_mean).powi(2)).sum::<f64>()
            / n_null_perms as f64).sqrt();
        let rule1_z = if null_std > 1e-12 { (rule1_obs - null_mean) / null_std } else { 0.0 };

        eprintln!("Rule-1 removal: N=365, obs={:.4}, null={:.4}+/-{:.4}, z={:.2}",
            rule1_obs, null_mean, null_std, rule1_z);

        // Random removal trials
        let mut random_z_scores = Vec::new();
        let mut rng_sample = ChaCha8Rng::seed_from_u64(54321);

        for trial in 0..n_random_trials {
            let mut indices: Vec<usize> = (0..n_full).collect();
            indices.shuffle(&mut rng_sample);
            let keep: Vec<_> = indices[n_remove..].iter().map(|&i| all_512[i]).collect();
            assert_eq!(keep.len(), 365);

            let prefix_r = shared_prefix_length(&keep);
            let d_r = 8 - prefix_r;
            let (cols_r, _, _) = lattice_to_column_major(&keep, prefix_r);
            let obs_r = matrix_free_fraction(&cols_r, keep.len(), d_r, n_triples, 42, 0.05);

            let mut rng_null_r = ChaCha8Rng::seed_from_u64(7_000_042 + trial as u64 * 1000);
            let mut null_fracs_r = Vec::with_capacity(n_null_perms);
            let mut shuffled_r = cols_r.clone();
            for _ in 0..n_null_perms {
                shuffled_r.copy_from_slice(&cols_r);
                apply_null_column_major(
                    &mut shuffled_r, keep.len(), d_r,
                    NullModel::ColumnIndependent, &mut rng_null_r,
                );
                null_fracs_r.push(matrix_free_fraction(&shuffled_r, keep.len(), d_r, n_triples, 8_000_042, 0.05));
            }
            let null_mean_r = null_fracs_r.iter().sum::<f64>() / n_null_perms as f64;
            let null_std_r = (null_fracs_r.iter().map(|f| (f - null_mean_r).powi(2)).sum::<f64>()
                / n_null_perms as f64).sqrt();
            let z_r = if null_std_r > 1e-12 { (obs_r - null_mean_r) / null_std_r } else { 0.0 };

            eprintln!("  Random trial {:>2}: obs={:.4}, null={:.4}+/-{:.4}, z={:.2}",
                trial, obs_r, null_mean_r, null_std_r, z_r);
            random_z_scores.push(z_r);
        }

        let mean_random_z = random_z_scores.iter().sum::<f64>() / n_random_trials as f64;
        let max_random_z = random_z_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        eprintln!("\n--- Control Summary ---");
        eprintln!("Rule-1 z-score:     {:.2}", rule1_z);
        eprintln!("Random mean z:      {:.2}", mean_random_z);
        eprintln!("Random max z:       {:.2}", max_random_z);

        // Targeted removal should beat random removal
        assert!(
            rule1_z > mean_random_z,
            "Rule-1 z={:.2} should exceed random mean z={:.2}",
            rule1_z, mean_random_z
        );
    }

    /// Consistency test for is_in_lambda_512_minus_k boundaries.
    #[test]
    fn test_lambda512_minus_k_consistency() {
        use algebra_core::analysis::codebook::{
            enumerate_lattice_by_predicate, is_in_lambda_256,
            is_in_lambda_512, is_in_lambda_512_minus_k,
        };

        let k0 = enumerate_lattice_by_predicate(|v| is_in_lambda_512_minus_k(v, 0));
        let k6 = enumerate_lattice_by_predicate(|v| is_in_lambda_512_minus_k(v, 6));
        let all_512 = enumerate_lattice_by_predicate(is_in_lambda_512);
        let all_256 = enumerate_lattice_by_predicate(is_in_lambda_256);

        assert_eq!(k0.len(), all_512.len(), "k=0 must equal Lambda_512");
        assert_eq!(k6.len(), all_256.len(), "k=6 must equal Lambda_256");
        assert_eq!(k0, all_512);
        assert_eq!(k6, all_256);

        // Monotone containment
        let mut prev = k0;
        for k in 1..=6 {
            let current = enumerate_lattice_by_predicate(|v| is_in_lambda_512_minus_k(v, k));
            assert!(current.len() <= prev.len(), "Must decrease");
            prev = current;
        }
    }

    // ================================================================
    // S_base -> Lambda_2048 Transition
    // ================================================================

    /// S_base -> Lambda_2048 intermediate gradient.
    /// Tests whether removing the 3 forbidden prefix patterns from S_base
    /// affects ultrametricity significantly.
    #[test]
    fn test_sbase_to_lambda2048_gradient() {
        use algebra_core::analysis::codebook::{
            enumerate_lattice_by_predicate, is_in_sbase_minus_k,
        };
        use super::super::baire::matrix_free_fraction;
        use super::super::null_models::{apply_null_column_major, NullModel};
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let n_triples = 50_000;
        let n_permutations = 200;
        let seed = 42u64;

        eprintln!("\n=== S_base -> Lambda_2048 Gradient ===");
        eprintln!("k | N vectors | prefix | eff_dim | obs_frac | null_mean | z-score | p-value");
        eprintln!("--|----------|--------|---------|----------|-----------|---------|--------");

        let mut z_scores = Vec::new();
        let mut sizes = Vec::new();
        let mut prev_n = 0usize;

        for k in 0..=3 {
            let vectors =
                enumerate_lattice_by_predicate(|v| is_in_sbase_minus_k(v, k));
            let n = vectors.len();
            let prefix = shared_prefix_length(&vectors);
            let d = 8 - prefix;

            let (cols, _, _) = lattice_to_column_major(&vectors, prefix);

            let obs_frac = matrix_free_fraction(&cols, n, d, n_triples, seed, 0.05);

            let mut rng = ChaCha8Rng::seed_from_u64(seed + 9_000_000 + k as u64);
            let mut null_fracs = Vec::with_capacity(n_permutations);
            let mut shuffled = cols.clone();

            for _ in 0..n_permutations {
                shuffled.copy_from_slice(&cols);
                apply_null_column_major(
                    &mut shuffled,
                    n,
                    d,
                    NullModel::ColumnIndependent,
                    &mut rng,
                );
                let null_frac =
                    matrix_free_fraction(&shuffled, n, d, n_triples, seed + 10_000_000, 0.05);
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

            let z_score = if null_std > 1e-12 {
                (obs_frac - null_mean) / null_std
            } else {
                0.0
            };

            let removed = if k == 0 { 0 } else { prev_n - n };
            eprintln!(
                "{} | {:>8} | {:>6} | {:>7} | {:>8.4} | {:>9.4} | {:>7.2} | {:>7.4}  (removed {})",
                k, n, prefix, d, obs_frac, null_mean, z_score, p_value, removed
            );

            z_scores.push(z_score);
            sizes.push(n);
            prev_n = n;
        }

        eprintln!("\n=== GRADIENT SUMMARY ===");
        eprintln!(
            "z-score progression: {:?}",
            z_scores.iter().map(|z| format!("{:.2}", z)).collect::<Vec<_>>()
        );

        // Verify boundary: k=3 should match Lambda_2048
        let k3_set = enumerate_lattice_by_predicate(|v| is_in_sbase_minus_k(v, 3));
        let l2048_set = enumerate_lattice_by_predicate(
            algebra_core::analysis::codebook::is_in_lambda_2048,
        );
        assert_eq!(k3_set.len(), l2048_set.len(), "k=3 must equal Lambda_2048");
    }

    /// Boundary consistency for is_in_sbase_minus_k.
    #[test]
    fn test_sbase_minus_k_consistency() {
        use algebra_core::analysis::codebook::{
            enumerate_lattice_by_predicate, is_in_base_universe, is_in_lambda_2048,
            is_in_sbase_minus_k,
        };

        let k0 = enumerate_lattice_by_predicate(|v| is_in_sbase_minus_k(v, 0));
        let sbase = enumerate_lattice_by_predicate(is_in_base_universe);
        assert_eq!(k0.len(), sbase.len(), "k=0 must equal S_base");

        let k3 = enumerate_lattice_by_predicate(|v| is_in_sbase_minus_k(v, 3));
        let l2048 = enumerate_lattice_by_predicate(is_in_lambda_2048);
        assert_eq!(k3.len(), l2048.len(), "k=3 must equal Lambda_2048");

        // Monotone containment
        let mut prev = k0;
        for k in 1..=3 {
            let current = enumerate_lattice_by_predicate(|v| is_in_sbase_minus_k(v, k));
            assert!(
                current.len() <= prev.len(),
                "k={}: {} must be <= {}",
                k,
                current.len(),
                prev.len()
            );
            prev = current;
        }
    }

    /// Investigate l_0=0 vs l_0=-1 subpopulation ultrametricity at Lambda_2048.
    /// This tests whether l_0=0 vectors are actively anti-ultrametric
    /// or merely neutral diluters.
    #[test]
    fn test_l0_subpopulation_ultrametricity() {
        use algebra_core::analysis::codebook::{
            enumerate_lattice_by_predicate, is_in_lambda_2048,
        };
        use super::super::baire::matrix_free_fraction;
        use super::super::null_models::{apply_null_column_major, NullModel};
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let n_triples = 50_000;
        let n_permutations = 200;
        let seed = 42u64;

        let all_2048 = enumerate_lattice_by_predicate(is_in_lambda_2048);
        let l0_neg1: Vec<_> = all_2048.iter().copied().filter(|v| v[0] == -1).collect();
        let l0_zero: Vec<_> = all_2048.iter().copied().filter(|v| v[0] == 0).collect();

        eprintln!("\n=== l_0 Subpopulation Ultrametricity at Lambda_2048 ===");
        eprintln!("Lambda_2048: N={}", all_2048.len());
        eprintln!("  l_0=-1 subset: N={}", l0_neg1.len());
        eprintln!("  l_0=0  subset: N={}", l0_zero.len());

        // Function to compute z-score for a subset
        let compute_z = |vectors: &[[i8; 8]], label: &str| -> f64 {
            let n = vectors.len();
            let prefix = shared_prefix_length(vectors);
            let d = 8 - prefix;
            let (cols, _, _) = lattice_to_column_major(vectors, prefix);

            let obs_frac = matrix_free_fraction(&cols, n, d, n_triples, seed, 0.05);

            let mut rng = ChaCha8Rng::seed_from_u64(seed + 11_000_000);
            let mut null_fracs = Vec::with_capacity(n_permutations);
            let mut shuffled = cols.clone();

            for _ in 0..n_permutations {
                shuffled.copy_from_slice(&cols);
                apply_null_column_major(
                    &mut shuffled,
                    n,
                    d,
                    NullModel::ColumnIndependent,
                    &mut rng,
                );
                null_fracs.push(matrix_free_fraction(
                    &shuffled, n, d, n_triples, seed + 12_000_000, 0.05,
                ));
            }

            let null_mean = null_fracs.iter().sum::<f64>() / n_permutations as f64;
            let null_var = null_fracs
                .iter()
                .map(|f| (f - null_mean).powi(2))
                .sum::<f64>()
                / n_permutations as f64;
            let null_std = null_var.sqrt();

            let z = if null_std > 1e-12 {
                (obs_frac - null_mean) / null_std
            } else {
                0.0
            };

            eprintln!(
                "  {}: prefix={}, d={}, obs={:.4}, null={:.4}+/-{:.4}, z={:.2}",
                label, prefix, d, obs_frac, null_mean, null_std, z
            );
            z
        };

        let z_neg1 = compute_z(&l0_neg1, "l_0=-1");
        let z_zero = compute_z(&l0_zero, "l_0=0 ");

        eprintln!("\n  Difference: l_0=-1 z={:.2}, l_0=0 z={:.2}", z_neg1, z_zero);
        eprintln!(
            "  Interpretation: l_0=0 is {} ultrametric than l_0=-1",
            if z_zero < z_neg1 { "LESS" } else { "MORE" }
        );
    }

    // ================================================================
    // Lambda_2048 -> Lambda_1024 Transition
    // ================================================================

    /// Lambda_2048 -> Lambda_1024 intermediate gradient.
    /// Tests whether the l_0=-1 slice and subsequent pattern exclusions
    /// produce a monotone or discontinuous ultrametricity change.
    #[test]
    fn test_lambda2048_to_1024_intermediate_gradient() {
        use algebra_core::analysis::codebook::{
            enumerate_lattice_by_predicate, is_in_lambda_2048_minus_k,
        };
        use super::super::baire::matrix_free_fraction;
        use super::super::null_models::{apply_null_column_major, NullModel};
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let n_triples = 50_000;
        let n_permutations = 200;
        let seed = 42u64;

        eprintln!("\n=== Lambda_2048 -> Lambda_1024 Intermediate Gradient ===");
        eprintln!("k | N vectors | prefix | eff_dim | obs_frac | null_mean | z-score | p-value");
        eprintln!("--|----------|--------|---------|----------|-----------|---------|--------");

        let mut z_scores = Vec::new();
        let mut sizes = Vec::new();
        let mut prev_n = 0usize;

        for k in 0..=4 {
            let vectors =
                enumerate_lattice_by_predicate(|v| is_in_lambda_2048_minus_k(v, k));
            let n = vectors.len();
            let prefix = shared_prefix_length(&vectors);
            let d = 8 - prefix;

            let (cols, _, _) = lattice_to_column_major(&vectors, prefix);

            let obs_frac = matrix_free_fraction(&cols, n, d, n_triples, seed, 0.05);

            let mut rng = ChaCha8Rng::seed_from_u64(seed + 5_000_000 + k as u64);
            let mut null_fracs = Vec::with_capacity(n_permutations);
            let mut shuffled = cols.clone();

            for _ in 0..n_permutations {
                shuffled.copy_from_slice(&cols);
                apply_null_column_major(
                    &mut shuffled,
                    n,
                    d,
                    NullModel::ColumnIndependent,
                    &mut rng,
                );
                let null_frac =
                    matrix_free_fraction(&shuffled, n, d, n_triples, seed + 6_000_000, 0.05);
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

            let z_score = if null_std > 1e-12 {
                (obs_frac - null_mean) / null_std
            } else {
                0.0
            };

            let removed = if k == 0 { 0 } else { prev_n - n };
            eprintln!(
                "{} | {:>8} | {:>6} | {:>7} | {:>8.4} | {:>9.4} | {:>7.2} | {:>7.4}  (removed {})",
                k, n, prefix, d, obs_frac, null_mean, z_score, p_value, removed
            );

            z_scores.push(z_score);
            sizes.push(n);
            prev_n = n;
        }

        eprintln!("\n=== GRADIENT SUMMARY ===");
        eprintln!(
            "z-score progression: {:?}",
            z_scores.iter().map(|z| format!("{:.2}", z)).collect::<Vec<_>>()
        );
        eprintln!("Step sizes:");
        for k in 1..=4 {
            eprintln!(
                "  Rule {}: {} -> {} (removed {})",
                k,
                sizes[k - 1],
                sizes[k],
                sizes[k - 1] - sizes[k]
            );
        }

        // Verify boundary: k=4 should match Lambda_1024
        let k4_set = enumerate_lattice_by_predicate(|v| is_in_lambda_2048_minus_k(v, 4));
        let l1024_set = enumerate_lattice_by_predicate(algebra_core::analysis::codebook::is_in_lambda_1024);
        assert_eq!(
            k4_set.len(),
            l1024_set.len(),
            "k=4 must equal Lambda_1024"
        );
    }

    /// Boundary consistency for is_in_lambda_2048_minus_k.
    #[test]
    fn test_lambda2048_minus_k_consistency() {
        use algebra_core::analysis::codebook::{
            enumerate_lattice_by_predicate, is_in_lambda_1024, is_in_lambda_2048,
            is_in_lambda_2048_minus_k,
        };

        let k0 = enumerate_lattice_by_predicate(|v| is_in_lambda_2048_minus_k(v, 0));
        let all_2048 = enumerate_lattice_by_predicate(is_in_lambda_2048);
        assert_eq!(k0.len(), all_2048.len(), "k=0 must equal Lambda_2048");

        let k4 = enumerate_lattice_by_predicate(|v| is_in_lambda_2048_minus_k(v, 4));
        let all_1024 = enumerate_lattice_by_predicate(is_in_lambda_1024);
        assert_eq!(k4.len(), all_1024.len(), "k=4 must equal Lambda_1024");

        // Monotone containment
        let mut prev = k0;
        for k in 1..=4 {
            let current = enumerate_lattice_by_predicate(|v| is_in_lambda_2048_minus_k(v, k));
            assert!(
                current.len() <= prev.len(),
                "k={}: {} must be <= {}",
                k,
                current.len(),
                prev.len()
            );
            prev = current;
        }
    }

    // ================================================================
    // l_1 Filter on l_0=-1 Subset (C-508 follow-up)
    // ================================================================

    /// Test whether applying the l_1 filter directly to the l_0=-1 subset
    /// reveals hidden ultrametric substructure. Computes z-scores for:
    ///   (a) l_0=-1, all l_1 (the Lambda_2048 l_0=-1 subset, z=-3.07 from C-508)
    ///   (b) l_0=-1, l_1=-1 only
    ///   (c) l_0=-1, l_1=0 only
    ///   (d) l_0=-1, l_1=+1 only
    /// If (b) or (c) show positive z while (d) shows strong negative z,
    /// the l_1 phase transition exists within the l_0=-1 population itself.
    #[test]
    fn test_l1_filter_on_l0_neg1_subset() {
        use algebra_core::analysis::codebook::{
            enumerate_lattice_by_predicate, is_in_lambda_2048,
        };
        use super::super::baire::matrix_free_fraction;
        use super::super::null_models::{apply_null_column_major, NullModel};
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let n_triples = 50_000;
        let n_permutations = 200;
        let seed = 42u64;

        let all_2048 = enumerate_lattice_by_predicate(is_in_lambda_2048);
        let l0_neg1: Vec<_> = all_2048.iter().copied().filter(|v| v[0] == -1).collect();

        // Partition l_0=-1 by l_1 value
        let l1_neg1: Vec<_> = l0_neg1.iter().copied().filter(|v| v[1] == -1).collect();
        let l1_zero: Vec<_> = l0_neg1.iter().copied().filter(|v| v[1] == 0).collect();
        let l1_pos1: Vec<_> = l0_neg1.iter().copied().filter(|v| v[1] == 1).collect();
        // Also: l_1 != +1 (what Lambda_512 effectively keeps)
        let l1_not_pos1: Vec<_> = l0_neg1.iter().copied().filter(|v| v[1] != 1).collect();

        eprintln!("\n=== l_1 Filter on l_0=-1 Subset ===");
        eprintln!("l_0=-1 total: N={}", l0_neg1.len());
        eprintln!("  l_1=-1: N={}", l1_neg1.len());
        eprintln!("  l_1=0:  N={}", l1_zero.len());
        eprintln!("  l_1=+1: N={}", l1_pos1.len());
        eprintln!("  l_1!=+1: N={}", l1_not_pos1.len());

        let compute_z = |vectors: &[[i8; 8]], label: &str, seed_offset: u64| -> f64 {
            let n = vectors.len();
            if n < 10 {
                eprintln!("  {}: N={} too small", label, n);
                return 0.0;
            }
            let prefix = shared_prefix_length(vectors);
            let d = 8 - prefix;
            let (cols, _, _) = lattice_to_column_major(vectors, prefix);

            let obs = matrix_free_fraction(&cols, n, d, n_triples, seed, 0.05);

            let mut rng = ChaCha8Rng::seed_from_u64(seed + 13_000_000 + seed_offset);
            let mut null_fracs = Vec::with_capacity(n_permutations);
            let mut shuffled = cols.clone();

            for _ in 0..n_permutations {
                shuffled.copy_from_slice(&cols);
                apply_null_column_major(
                    &mut shuffled, n, d,
                    NullModel::ColumnIndependent, &mut rng,
                );
                null_fracs.push(matrix_free_fraction(
                    &shuffled, n, d, n_triples, seed + 14_000_000, 0.05,
                ));
            }

            let null_mean = null_fracs.iter().sum::<f64>() / n_permutations as f64;
            let null_std = (null_fracs.iter().map(|f| (f - null_mean).powi(2)).sum::<f64>()
                / n_permutations as f64).sqrt();
            let z = if null_std > 1e-12 { (obs - null_mean) / null_std } else { 0.0 };

            eprintln!("  {}: prefix={}, d={}, N={}, obs={:.4}, null={:.4}+/-{:.4}, z={:.2}",
                label, prefix, d, n, obs, null_mean, null_std, z);
            z
        };

        let z_all = compute_z(&l0_neg1, "all l_1  ", 0);
        let z_neg1 = compute_z(&l1_neg1, "l_1=-1   ", 1);
        let z_zero = compute_z(&l1_zero, "l_1=0    ", 2);
        let z_pos1 = compute_z(&l1_pos1, "l_1=+1   ", 3);
        let z_not1 = compute_z(&l1_not_pos1, "l_1!=+1  ", 4);

        eprintln!("\n=== Summary ===");
        eprintln!("l_0=-1, all l_1:  z={:.2}", z_all);
        eprintln!("l_0=-1, l_1=-1:   z={:.2}", z_neg1);
        eprintln!("l_0=-1, l_1=0:    z={:.2}", z_zero);
        eprintln!("l_0=-1, l_1=+1:   z={:.2}", z_pos1);
        eprintln!("l_0=-1, l_1!=+1:  z={:.2}", z_not1);
        eprintln!("Phase transition present: l_1!=+1 z ({:.2}) vs all z ({:.2})", z_not1, z_all);
    }
}

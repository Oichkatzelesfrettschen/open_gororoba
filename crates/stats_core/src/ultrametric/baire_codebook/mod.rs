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
use gororoba_algebra::analysis::codebook::LatticeVector;
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
// N-dimensional lattice Baire functions (arbitrary D via &[i8] slices)
// ============================================================================

/// Compute the shared prefix length among N-D integer lattice vectors.
///
/// Generalizes `shared_prefix_length` from 8D `LatticeVector` to slices
/// of arbitrary dimensionality. All vectors must have the same length.
pub fn shared_prefix_length_nd(vectors: &[Vec<i8>]) -> usize {
    if vectors.is_empty() {
        return 0;
    }
    let dim = vectors[0].len();
    let first = &vectors[0];
    for pos in 0..dim {
        if vectors.iter().any(|v| v[pos] != first[pos]) {
            return pos;
        }
    }
    dim
}

/// Compute Baire distance between two N-D integer lattice vectors.
///
/// Uses base-3 encoding: -1 -> 0, 0 -> 1, 1 -> 2 (for trinary lattices).
/// Distance = 3^{-k} where k is the 1-indexed position of the first
/// differing digit after `skip_prefix` leading coordinates.
pub fn lattice_baire_distance_nd(a: &[i8], b: &[i8], skip_prefix: usize) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "dimension mismatch");
    let dim = a.len();
    for pos in skip_prefix..dim {
        if a[pos] != b[pos] {
            let effective_pos = pos - skip_prefix + 1; // 1-indexed
            return 3.0_f64.powi(-(effective_pos as i32));
        }
    }
    0.0
}

/// Compute the full Baire distance matrix for N-D integer lattice vectors.
///
/// Returns a flat upper-triangle distance matrix.
pub fn lattice_baire_distance_matrix_nd(vectors: &[Vec<i8>], skip_prefix: usize) -> Vec<f64> {
    let n = vectors.len();
    let n_pairs = n * (n - 1) / 2;
    let mut dists = Vec::with_capacity(n_pairs);
    for i in 0..n {
        for j in (i + 1)..n {
            dists.push(lattice_baire_distance_nd(
                &vectors[i],
                &vectors[j],
                skip_prefix,
            ));
        }
    }
    dists
}

/// Run a codebook Baire ultrametric test on N-D integer lattice vectors.
///
/// Generalizes `codebook_baire_ultrametric_test` from 8D to arbitrary
/// dimensionality. Auto-detects shared prefix, strips it, and runs the
/// ultrametric fraction test with column-shuffled null.
pub fn codebook_baire_ultrametric_test_nd(
    vectors: &[Vec<i8>],
    n_triples: usize,
    n_permutations: usize,
    seed: u64,
) -> CodebookBaireResult {
    assert!(vectors.len() >= 3, "need at least 3 vectors");
    let dim = vectors[0].len();
    debug_assert!(vectors.iter().all(|v| v.len() == dim), "dimension mismatch");

    let prefix_len = shared_prefix_length_nd(vectors);
    let effective_dim = dim - prefix_len;

    let dist_matrix = lattice_baire_distance_matrix_nd(vectors, prefix_len);
    let n = vectors.len();

    let obs_frac = super::ultrametric_fraction_from_matrix(&dist_matrix, n, n_triples, seed);

    // Null: shuffle each tail coordinate independently
    let mut rng = ChaCha8Rng::seed_from_u64(seed + 1_000_000);
    let mut null_fracs = Vec::with_capacity(n_permutations);
    let mut shuffled: Vec<Vec<i8>> = vectors.to_vec();

    for _ in 0..n_permutations {
        for coord in prefix_len..dim {
            let mut col_vals: Vec<i8> = shuffled.iter().map(|v| v[coord]).collect();
            col_vals.shuffle(&mut rng);
            for (i, &val) in col_vals.iter().enumerate() {
                shuffled[i][coord] = val;
            }
        }

        let null_dists = lattice_baire_distance_matrix_nd(&shuffled, prefix_len);
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
mod tests;

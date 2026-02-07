//! Hierarchical clustering and cophenetic correlation analysis.
//!
//! Builds single-linkage dendrograms from distance matrices using kodama,
//! extracts cophenetic distances, and measures how well the tree structure
//! preserves the original pairwise distances (cophenetic correlation).
//!
//! A cophenetic correlation close to 1.0 indicates the data has intrinsic
//! tree-like (ultrametric) structure.
//!
//! # Algorithm
//!
//! 1. Compute pairwise distance matrix from point coordinates
//! 2. Run single-linkage hierarchical clustering (kodama)
//! 3. Extract cophenetic distance matrix from the dendrogram
//! 4. Compute Pearson correlation between original and cophenetic distances
//! 5. Compare to null distribution (shuffled coordinates)
//!
//! # References
//!
//! - Sokal & Rohlf (1962): Cophenetic correlation coefficient
//! - Murtagh & Contreras (2012): Algorithms for hierarchical clustering

use kodama::{linkage, Method};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use crate::claims_gates::Verdict;

/// Result of dendrogram-based ultrametric analysis.
#[derive(Debug, Clone)]
pub struct DendrogramResult {
    /// Number of points clustered.
    pub n_points: usize,
    /// Linkage method used.
    pub linkage_method: String,
    /// Cophenetic correlation coefficient (Pearson r between original
    /// and cophenetic distances). 1.0 = perfect ultrametric.
    pub cophenetic_correlation: f64,
    /// Mean cophenetic correlation from null (shuffled) distribution.
    pub null_cophenetic_mean: f64,
    /// Standard deviation of null cophenetic correlations.
    pub null_cophenetic_std: f64,
    /// P-value: fraction of null correlations >= observed.
    pub p_value: f64,
    /// RMSE between original and cophenetic distances (absolute distortion).
    pub cophenetic_rmse: f64,
    /// Maximum relative error between original and cophenetic distances.
    pub cophenetic_max_rel_error: f64,
    /// Ultrametric fraction computed on the original distance matrix.
    pub ultrametric_fraction: f64,
    /// Verdict based on cophenetic correlation significance.
    pub verdict: Verdict,
}

/// Result of testing across multiple linkage methods.
#[derive(Debug, Clone)]
pub struct MultiLinkageResult {
    /// Results for each linkage method.
    pub results: Vec<DendrogramResult>,
    /// Whether the signal is stable across methods (>= 2 pass at p < 0.05).
    pub linkage_stable: bool,
}

/// Name a kodama Method for display.
fn method_name(method: Method) -> &'static str {
    match method {
        Method::Single => "single",
        Method::Complete => "complete",
        Method::Average => "average",
        Method::Ward => "ward",
        _ => "other",
    }
}

/// Compute RMSE between original and cophenetic distances.
fn cophenetic_rmse(original: &[f64], cophenetic: &[f64]) -> f64 {
    assert_eq!(original.len(), cophenetic.len());
    let n = original.len() as f64;
    let mse: f64 = original
        .iter()
        .zip(cophenetic.iter())
        .map(|(&o, &c)| (o - c).powi(2))
        .sum::<f64>()
        / n;
    mse.sqrt()
}

/// Compute max relative error between original and cophenetic distances.
fn cophenetic_max_rel_error(original: &[f64], cophenetic: &[f64]) -> f64 {
    assert_eq!(original.len(), cophenetic.len());
    original
        .iter()
        .zip(cophenetic.iter())
        .filter(|(&o, _)| o > 1e-15)
        .map(|(&o, &c)| ((o - c) / o).abs())
        .fold(0.0_f64, f64::max)
}

/// Compute the cophenetic distance matrix from a kodama dendrogram.
///
/// The cophenetic distance c(i,j) is the dissimilarity at which points
/// i and j first merge in the hierarchical clustering. For single-linkage,
/// this equals the minimax path distance in the minimum spanning tree.
///
/// Returns a flat upper-triangle matrix (row-major):
/// index(i,j) = i*n - i*(i+1)/2 + j - i - 1 for i < j.
pub fn cophenetic_distance_matrix(
    n_points: usize,
    steps: &[(usize, usize, f64, usize)],
) -> Vec<f64> {
    let n_pairs = n_points * (n_points - 1) / 2;
    let mut coph = vec![0.0; n_pairs];

    // Union-Find to track which original points belong to each cluster.
    // After n_points-1 merges, all points are in one cluster.
    // Clusters 0..n_points are the original points.
    // Cluster n_points+i is the result of step i.
    let mut members: Vec<Vec<usize>> = (0..n_points).map(|i| vec![i]).collect();

    let idx = |i: usize, j: usize| -> usize {
        let (a, b) = if i < j { (i, j) } else { (j, i) };
        a * n_points - a * (a + 1) / 2 + b - a - 1
    };

    for &(c1, c2, dissimilarity, _size) in steps {
        // For every pair (i in c1, j in c2), set cophenetic distance
        let members_c1 = &members[c1];
        let members_c2 = &members[c2];

        for &i in members_c1 {
            for &j in members_c2 {
                coph[idx(i, j)] = dissimilarity;
            }
        }

        // Merge: the new cluster gets all members from both
        let mut merged = members[c1].clone();
        merged.extend_from_slice(&members[c2]);
        members.push(merged);
    }

    coph
}

/// Compute Pearson correlation between two flat upper-triangle matrices.
pub fn cophenetic_correlation(
    original_dists: &[f64],
    cophenetic_dists: &[f64],
) -> f64 {
    assert_eq!(original_dists.len(), cophenetic_dists.len());
    let n = original_dists.len() as f64;

    let mean_o = original_dists.iter().sum::<f64>() / n;
    let mean_c = cophenetic_dists.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_o = 0.0;
    let mut var_c = 0.0;

    for (&o, &c) in original_dists.iter().zip(cophenetic_dists.iter()) {
        let do_ = o - mean_o;
        let dc = c - mean_c;
        cov += do_ * dc;
        var_o += do_ * do_;
        var_c += dc * dc;
    }

    if var_o < 1e-30 || var_c < 1e-30 {
        return 0.0;
    }

    cov / (var_o.sqrt() * var_c.sqrt())
}

/// Run the full dendrogram-based ultrametric test.
///
/// Takes a flat upper-triangle distance matrix and the number of points.
/// Builds a dendrogram using the specified linkage method, computes cophenetic
/// distances, and tests whether the cophenetic correlation exceeds what shuffled
/// data produces.
///
/// `dist_matrix` layout: index(i,j) = i*n - i*(i+1)/2 + j - i - 1 for i < j.
pub fn hierarchical_ultrametric_test(
    dist_matrix: &[f64],
    n_points: usize,
    n_permutations: usize,
    seed: u64,
) -> DendrogramResult {
    hierarchical_ultrametric_test_with_method(
        dist_matrix, n_points, n_permutations, seed, Method::Single,
    )
}

/// Run dendrogram test with a specific linkage method.
pub fn hierarchical_ultrametric_test_with_method(
    dist_matrix: &[f64],
    n_points: usize,
    n_permutations: usize,
    seed: u64,
    method: Method,
) -> DendrogramResult {
    let n_pairs = n_points * (n_points - 1) / 2;
    assert_eq!(dist_matrix.len(), n_pairs, "Distance matrix size mismatch");

    // Build dendrogram
    let mut condensed = dist_matrix.to_vec();
    let dend = linkage(&mut condensed, n_points, method);

    // Extract steps as tuples
    let steps: Vec<(usize, usize, f64, usize)> = dend
        .steps()
        .iter()
        .map(|s| (s.cluster1, s.cluster2, s.dissimilarity, s.size))
        .collect();

    // Compute cophenetic distances
    let coph = cophenetic_distance_matrix(n_points, &steps);

    // Cophenetic correlation and distortion
    let obs_corr = cophenetic_correlation(dist_matrix, &coph);
    let rmse = cophenetic_rmse(dist_matrix, &coph);
    let max_rel = cophenetic_max_rel_error(dist_matrix, &coph);

    // Ultrametric fraction on original distances
    let obs_frac = super::ultrametric_fraction_from_matrix(
        dist_matrix, n_points, 100_000.min(n_pairs * 10), seed,
    );

    // Null distribution: shuffle the distance matrix labels
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut null_corrs = Vec::with_capacity(n_permutations);

    let mut perm: Vec<usize> = (0..n_points).collect();

    let idx = |i: usize, j: usize, n: usize| -> usize {
        let (a, b) = if i < j { (i, j) } else { (j, i) };
        a * n - a * (a + 1) / 2 + b - a - 1
    };

    for _ in 0..n_permutations {
        perm.shuffle(&mut rng);

        let mut perm_dists = vec![0.0; n_pairs];
        for i in 0..n_points {
            for j in (i + 1)..n_points {
                perm_dists[idx(i, j, n_points)] =
                    dist_matrix[idx(perm[i], perm[j], n_points)];
            }
        }

        let mut perm_condensed = perm_dists.clone();
        let perm_dend = linkage(&mut perm_condensed, n_points, method);
        let perm_steps: Vec<(usize, usize, f64, usize)> = perm_dend
            .steps()
            .iter()
            .map(|s| (s.cluster1, s.cluster2, s.dissimilarity, s.size))
            .collect();

        let perm_coph = cophenetic_distance_matrix(n_points, &perm_steps);
        let perm_corr = cophenetic_correlation(&perm_dists, &perm_coph);
        null_corrs.push(perm_corr);
    }

    let null_mean = null_corrs.iter().sum::<f64>() / n_permutations as f64;
    let null_var = null_corrs
        .iter()
        .map(|c| (c - null_mean).powi(2))
        .sum::<f64>()
        / n_permutations as f64;
    let null_std = null_var.sqrt();

    let n_ge = null_corrs.iter().filter(|&&c| c >= obs_corr).count();
    let p_value = (n_ge as f64 + 1.0) / (n_permutations as f64 + 1.0);

    let verdict = if p_value < 0.05 {
        Verdict::Pass
    } else {
        Verdict::Fail
    };

    DendrogramResult {
        n_points,
        linkage_method: method_name(method).to_string(),
        cophenetic_correlation: obs_corr,
        null_cophenetic_mean: null_mean,
        null_cophenetic_std: null_std,
        p_value,
        cophenetic_rmse: rmse,
        cophenetic_max_rel_error: max_rel,
        ultrametric_fraction: obs_frac,
        verdict,
    }
}

/// Test across all four standard linkage methods.
///
/// A real ultrametric signal should be stable across linkage choices.
/// If the signal appears only under single-linkage (which can "chain"),
/// it is likely an artifact.
pub fn multi_linkage_test(
    dist_matrix: &[f64],
    n_points: usize,
    n_permutations: usize,
    seed: u64,
) -> MultiLinkageResult {
    let methods = [Method::Single, Method::Complete, Method::Average, Method::Ward];

    let results: Vec<DendrogramResult> = methods
        .iter()
        .enumerate()
        .map(|(i, &m)| {
            hierarchical_ultrametric_test_with_method(
                dist_matrix, n_points, n_permutations, seed + i as u64 * 100, m,
            )
        })
        .collect();

    let n_pass = results.iter().filter(|r| r.verdict == Verdict::Pass).count();
    let linkage_stable = n_pass >= 2;

    MultiLinkageResult {
        results,
        linkage_stable,
    }
}

/// Compute a flat upper-triangle Euclidean distance matrix from 3D coordinates.
///
/// Input: slice of (x, y, z) tuples.
/// Output: flat upper-triangle, index(i,j) = i*n - i*(i+1)/2 + j - i - 1.
pub fn euclidean_distance_matrix_3d(coords: &[(f64, f64, f64)]) -> Vec<f64> {
    let n = coords.len();
    let n_pairs = n * (n - 1) / 2;
    let mut dists = vec![0.0; n_pairs];

    let mut k = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = coords[i].0 - coords[j].0;
            let dy = coords[i].1 - coords[j].1;
            let dz = coords[i].2 - coords[j].2;
            dists[k] = (dx * dx + dy * dy + dz * dz).sqrt();
            k += 1;
        }
    }

    dists
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cophenetic_perfect_ultrametric() {
        // 4 points forming a perfect ultrametric tree:
        //       *
        //      / \     (merge at d=2)
        //     *   *
        //    / \ / \   (merge at d=1)
        //   0  1 2  3
        //
        // d(0,1)=1, d(2,3)=1, d(0,2)=d(0,3)=d(1,2)=d(1,3)=2
        // Upper triangle: d01, d02, d03, d12, d13, d23
        let dist_matrix = vec![1.0, 2.0, 2.0, 2.0, 2.0, 1.0];
        let n = 4;

        let mut condensed = dist_matrix.clone();
        let dend = linkage(&mut condensed, n, Method::Single);

        let steps: Vec<(usize, usize, f64, usize)> = dend
            .steps()
            .iter()
            .map(|s| (s.cluster1, s.cluster2, s.dissimilarity, s.size))
            .collect();

        let coph = cophenetic_distance_matrix(n, &steps);

        // For a perfect ultrametric, cophenetic distances should equal original
        let corr = cophenetic_correlation(&dist_matrix, &coph);
        assert!(
            corr > 0.99,
            "Perfect ultrametric should have cophenetic correlation ~1.0, got {}",
            corr
        );
    }

    #[test]
    fn test_cophenetic_random_lower_correlation() {
        // Random 3D points should have lower cophenetic correlation
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let n = 20;
        let coords: Vec<(f64, f64, f64)> = (0..n)
            .map(|_| (
                rng.gen_range(0.0..100.0),
                rng.gen_range(0.0..100.0),
                rng.gen_range(0.0..100.0),
            ))
            .collect();

        let dists = euclidean_distance_matrix_3d(&coords);

        let mut condensed = dists.clone();
        let dend = linkage(&mut condensed, n, Method::Single);
        let steps: Vec<(usize, usize, f64, usize)> = dend
            .steps()
            .iter()
            .map(|s| (s.cluster1, s.cluster2, s.dissimilarity, s.size))
            .collect();

        let coph = cophenetic_distance_matrix(n, &steps);
        let corr = cophenetic_correlation(&dists, &coph);

        // Random data typically has cophenetic correlation 0.6-0.9
        assert!(
            corr > 0.0 && corr < 1.0,
            "Random data cophenetic correlation should be between 0 and 1, got {}",
            corr
        );
    }

    #[test]
    fn test_hierarchical_test_smoke() {
        // Smoke test on small random data
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let n = 10;
        let coords: Vec<(f64, f64, f64)> = (0..n)
            .map(|_| (
                rng.gen_range(0.0..100.0),
                rng.gen_range(0.0..100.0),
                rng.gen_range(0.0..100.0),
            ))
            .collect();

        let dists = euclidean_distance_matrix_3d(&coords);
        let result = hierarchical_ultrametric_test(&dists, n, 50, 42);

        assert_eq!(result.n_points, 10);
        assert!(result.cophenetic_correlation > 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_euclidean_distance_matrix_3d_simple() {
        // Two points at known distance
        let coords = vec![(0.0, 0.0, 0.0), (3.0, 4.0, 0.0)];
        let dists = euclidean_distance_matrix_3d(&coords);
        assert_eq!(dists.len(), 1);
        assert!((dists[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_distortion_perfect_ultrametric() {
        // Perfect ultrametric: cophenetic distances = original -> RMSE ~ 0
        let dist_matrix = vec![1.0, 2.0, 2.0, 2.0, 2.0, 1.0];
        let result = hierarchical_ultrametric_test(&dist_matrix, 4, 20, 42);

        assert!(
            result.cophenetic_rmse < 0.01,
            "Perfect ultrametric RMSE should be ~0, got {}",
            result.cophenetic_rmse,
        );
        assert!(result.linkage_method == "single");
    }

    #[test]
    fn test_distortion_random_positive() {
        // Random data should have non-zero distortion
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let n = 15;
        let coords: Vec<(f64, f64, f64)> = (0..n)
            .map(|_| (
                rng.gen_range(0.0..100.0),
                rng.gen_range(0.0..100.0),
                rng.gen_range(0.0..100.0),
            ))
            .collect();

        let dists = euclidean_distance_matrix_3d(&coords);
        let result = hierarchical_ultrametric_test(&dists, n, 20, 42);

        assert!(
            result.cophenetic_rmse > 0.0,
            "Random data should have positive RMSE, got {}",
            result.cophenetic_rmse,
        );
        assert!(
            result.cophenetic_max_rel_error > 0.0,
            "Random data should have positive max relative error, got {}",
            result.cophenetic_max_rel_error,
        );
    }

    #[test]
    fn test_multi_linkage_runs_all_methods() {
        let dist_matrix = vec![1.0, 2.0, 2.0, 2.0, 2.0, 1.0];
        let result = multi_linkage_test(&dist_matrix, 4, 20, 42);

        assert_eq!(result.results.len(), 4);
        let methods: Vec<&str> = result.results.iter().map(|r| r.linkage_method.as_str()).collect();
        assert!(methods.contains(&"single"));
        assert!(methods.contains(&"complete"));
        assert!(methods.contains(&"average"));
        assert!(methods.contains(&"ward"));
    }

    #[test]
    fn test_complete_linkage_on_ultrametric() {
        // Perfect ultrametric should work well under complete linkage too
        let dist_matrix = vec![1.0, 2.0, 2.0, 2.0, 2.0, 1.0];
        let result = hierarchical_ultrametric_test_with_method(
            &dist_matrix, 4, 20, 42, Method::Complete,
        );

        assert!(
            result.cophenetic_correlation > 0.9,
            "Complete linkage on perfect ultrametric should have high correlation, got {}",
            result.cophenetic_correlation,
        );
        assert_eq!(result.linkage_method, "complete");
    }
}

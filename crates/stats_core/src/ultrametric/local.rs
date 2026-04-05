//! Local ultrametricity test (Bradley arXiv:2408.07174).
//!
//! For each point in a dataset, examines its epsilon-neighborhood and tests
//! all triples within that neighborhood for the ultrametric inequality.
//! The local ultrametricity index is the fraction of triples satisfying
//! the inequality. Aggregating across all points gives a distribution of
//! local indices that can be compared to a Poisson null.
//!
//! This approach detects ultrametric structure that is *locally* present
//! even when the global dataset is not ultrametric -- e.g., hierarchical
//! clustering within filaments of the cosmic web.
//!
//! # Algorithm
//!
//! 1. Build a k-d tree from the 3D point coordinates
//! 2. For each point, find all neighbors within radius epsilon
//! 3. If the neighborhood has >= 3 points, sample random triples
//! 4. Compute the local ultrametric fraction for each neighborhood
//! 5. Aggregate: mean, median, and distribution of local indices
//! 6. Compare to null: Poisson point process in the same volume
//!
//! # References
//!
//! - Bradley (2025): arXiv:2408.07174 (local ultrametricity)
//! - Rammal, Toulouse, Virasoro (1986): Ultrametricity for physicists

use kiddo::KdTree;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use crate::claims_gates::Verdict;

/// Result of local ultrametricity analysis.
#[derive(Debug, Clone)]
pub struct LocalUltrametricResult {
    /// Number of points in the dataset.
    pub n_points: usize,
    /// Epsilon radius used for neighborhood queries.
    pub epsilon: f64,
    /// Number of points with neighborhoods of size >= 3.
    pub n_testable: usize,
    /// Mean local ultrametric index across all testable points.
    pub mean_local_index: f64,
    /// Median local ultrametric index.
    pub median_local_index: f64,
    /// Standard deviation of local indices.
    pub std_local_index: f64,
    /// Mean local index from Poisson null distribution.
    pub null_mean_index: f64,
    /// P-value: fraction of null means >= observed mean.
    pub p_value: f64,
    /// Verdict based on significance.
    pub verdict: Verdict,
}

/// Compute the local ultrametric index for a single neighborhood.
///
/// Given a set of 3D points, samples random triples and checks the
/// ultrametric inequality. Returns the fraction of triples satisfying it.
///
/// Uses squared distances internally to avoid 3 sqrt() calls per triple.
fn neighborhood_ultrametric_index(
    points: &[(f64, f64, f64)],
    n_samples: usize,
    rng: &mut ChaCha8Rng,
) -> f64 {
    let n = points.len();
    if n < 3 {
        return f64::NAN;
    }

    let mut count = 0usize;
    // epsilon=0.05 on distances -> epsilon_sq on squared distances
    let epsilon_sq = 1.0 - (1.0 - 0.05_f64).powi(2); // = 0.0975

    for _ in 0..n_samples {
        let i = rng.random_range(0..n);
        let mut j = rng.random_range(0..n - 1);
        if j >= i {
            j += 1;
        }
        let mut k = rng.random_range(0..n - 2);
        if k >= i.min(j) {
            k += 1;
        }
        if k >= i.max(j) {
            k += 1;
        }

        let d_ij_sq = euclidean_3d_sq(&points[i], &points[j]);
        let d_jk_sq = euclidean_3d_sq(&points[j], &points[k]);
        let d_ik_sq = euclidean_3d_sq(&points[i], &points[k]);

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

    count as f64 / n_samples as f64
}

/// Squared Euclidean distance between two 3D points.
///
/// Returns d^2 to avoid the sqrt; callers that need the actual distance
/// should take the square root of the result.
fn euclidean_3d_sq(a: &(f64, f64, f64), b: &(f64, f64, f64)) -> f64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    let dz = a.2 - b.2;
    dx * dx + dy * dy + dz * dz
}

/// Run the local ultrametricity test on 3D point coordinates.
///
/// For each point, finds neighbors within `epsilon` using a k-d tree,
/// computes the local ultrametric index, and compares the aggregate
/// to a Poisson null.
///
/// `coords`: slice of (x, y, z) coordinates (e.g., comoving Mpc).
/// `epsilon`: neighborhood radius in the same units as coords.
/// `n_samples_per_neighborhood`: triples to sample per neighborhood.
/// `n_permutations`: permutations for null distribution.
/// `seed`: RNG seed.
pub fn local_ultrametricity_test(
    coords: &[(f64, f64, f64)],
    epsilon: f64,
    n_samples_per_neighborhood: usize,
    n_permutations: usize,
    seed: u64,
) -> LocalUltrametricResult {
    let n = coords.len();
    assert!(n >= 3, "Need at least 3 points");

    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Build k-d tree
    let mut tree: KdTree<f64, 3> = KdTree::new();
    for (i, &(x, y, z)) in coords.iter().enumerate() {
        tree.add(&[x, y, z], i as u64);
    }

    // For each point, find epsilon-neighborhood and compute local index
    let epsilon_sq = epsilon * epsilon;
    let mut local_indices = Vec::with_capacity(n);

    for &(x, y, z) in coords {
        let neighbors: Vec<_> = tree
            .within::<kiddo::SquaredEuclidean>(&[x, y, z], epsilon_sq)
            .iter()
            .map(|nb| {
                let idx = nb.item as usize;
                coords[idx]
            })
            .collect();

        if neighbors.len() >= 3 {
            let idx =
                neighborhood_ultrametric_index(&neighbors, n_samples_per_neighborhood, &mut rng);
            local_indices.push(idx);
        }
    }

    let n_testable = local_indices.len();
    if n_testable == 0 {
        return LocalUltrametricResult {
            n_points: n,
            epsilon,
            n_testable: 0,
            mean_local_index: f64::NAN,
            median_local_index: f64::NAN,
            std_local_index: f64::NAN,
            null_mean_index: f64::NAN,
            p_value: 1.0,
            verdict: Verdict::Fail,
        };
    }

    let mean_idx = local_indices.iter().sum::<f64>() / n_testable as f64;
    let var_idx = local_indices
        .iter()
        .map(|x| (x - mean_idx).powi(2))
        .sum::<f64>()
        / n_testable as f64;
    let std_idx = var_idx.sqrt();

    let mut sorted = local_indices.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_idx = sorted[n_testable / 2];

    // Null distribution: shuffle coordinate assignments
    let mut null_means = Vec::with_capacity(n_permutations);
    let mut shuffled_coords = coords.to_vec();

    for _ in 0..n_permutations {
        shuffled_coords.shuffle(&mut rng);

        // Rebuild k-d tree for shuffled coordinates
        let mut null_tree: KdTree<f64, 3> = KdTree::new();
        for (i, &(x, y, z)) in shuffled_coords.iter().enumerate() {
            null_tree.add(&[x, y, z], i as u64);
        }

        let mut null_indices = Vec::new();
        for &(x, y, z) in &shuffled_coords {
            let neighbors: Vec<_> = null_tree
                .within::<kiddo::SquaredEuclidean>(&[x, y, z], epsilon_sq)
                .iter()
                .map(|nb| {
                    let idx = nb.item as usize;
                    shuffled_coords[idx]
                })
                .collect();

            if neighbors.len() >= 3 {
                let idx = neighborhood_ultrametric_index(
                    &neighbors,
                    n_samples_per_neighborhood,
                    &mut rng,
                );
                null_indices.push(idx);
            }
        }

        if !null_indices.is_empty() {
            let null_mean = null_indices.iter().sum::<f64>() / null_indices.len() as f64;
            null_means.push(null_mean);
        }
    }

    let null_mean_index = if null_means.is_empty() {
        f64::NAN
    } else {
        null_means.iter().sum::<f64>() / null_means.len() as f64
    };

    // One-sided p-value: fraction of null means >= observed mean
    let n_ge = null_means.iter().filter(|&&m| m >= mean_idx).count();
    let p_value = if null_means.is_empty() {
        1.0
    } else {
        (n_ge as f64 + 1.0) / (null_means.len() as f64 + 1.0)
    };

    let verdict = if p_value < 0.05 {
        Verdict::Pass
    } else {
        Verdict::Fail
    };

    LocalUltrametricResult {
        n_points: n,
        epsilon,
        n_testable,
        mean_local_index: mean_idx,
        median_local_index: median_idx,
        std_local_index: std_idx,
        null_mean_index,
        p_value,
        verdict,
    }
}

// ---------------------------------------------------------------------------
// N-dimensional generalization (brute-force neighbor search)
// ---------------------------------------------------------------------------

/// Squared Euclidean distance between two N-D points given as slices.
///
/// Panics if slices have different lengths.
fn euclidean_nd_sq(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "dimension mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| {
            let d = ai - bi;
            d * d
        })
        .sum()
}

/// Compute the local ultrametric index for a single N-D neighborhood.
///
/// Same algorithm as `neighborhood_ultrametric_index` but works on
/// arbitrary-dimensional points stored as `&[Vec<f64>]`.
fn neighborhood_ultrametric_index_nd(
    points: &[Vec<f64>],
    n_samples: usize,
    rng: &mut ChaCha8Rng,
) -> f64 {
    let n = points.len();
    if n < 3 {
        return f64::NAN;
    }

    let mut count = 0usize;
    let epsilon_sq = 1.0 - (1.0 - 0.05_f64).powi(2); // = 0.0975

    for _ in 0..n_samples {
        let i = rng.random_range(0..n);
        let mut j = rng.random_range(0..n - 1);
        if j >= i {
            j += 1;
        }
        let mut k = rng.random_range(0..n - 2);
        if k >= i.min(j) {
            k += 1;
        }
        if k >= i.max(j) {
            k += 1;
        }

        let d_ij_sq = euclidean_nd_sq(&points[i], &points[j]);
        let d_jk_sq = euclidean_nd_sq(&points[j], &points[k]);
        let d_ik_sq = euclidean_nd_sq(&points[i], &points[k]);

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

    count as f64 / n_samples as f64
}

/// Run the local ultrametricity test on N-dimensional point coordinates.
///
/// Generalizes `local_ultrametricity_test` from 3D tuples to arbitrary
/// dimensionality. Uses brute-force O(n^2) neighbor search instead of a
/// k-d tree, which is practical for datasets up to ~1000 points at any
/// dimension. For large 3D datasets, prefer the original function.
///
/// Each point is a `Vec<f64>` of the same dimensionality. The function
/// panics if points have inconsistent dimensions or if fewer than 3
/// points are provided.
///
/// `coords`: slice of N-D coordinate vectors.
/// `epsilon`: neighborhood radius (Euclidean distance).
/// `n_samples_per_neighborhood`: triples to sample per neighborhood.
/// `n_permutations`: permutations for null distribution.
/// `seed`: RNG seed.
pub fn local_ultrametricity_test_nd(
    coords: &[Vec<f64>],
    epsilon: f64,
    n_samples_per_neighborhood: usize,
    n_permutations: usize,
    seed: u64,
) -> LocalUltrametricResult {
    let n = coords.len();
    assert!(n >= 3, "Need at least 3 points");
    let dim = coords[0].len();
    assert!(dim >= 1, "Points must have at least 1 dimension");
    debug_assert!(
        coords.iter().all(|p| p.len() == dim),
        "All points must have the same dimensionality"
    );

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let epsilon_sq = epsilon * epsilon;

    // Brute-force epsilon-neighborhood search
    let mut local_indices = Vec::with_capacity(n);
    for i in 0..n {
        let neighbors: Vec<Vec<f64>> = coords
            .iter()
            .filter(|p| euclidean_nd_sq(p, &coords[i]) <= epsilon_sq)
            .cloned()
            .collect();

        if neighbors.len() >= 3 {
            let idx =
                neighborhood_ultrametric_index_nd(&neighbors, n_samples_per_neighborhood, &mut rng);
            local_indices.push(idx);
        }
    }

    let n_testable = local_indices.len();
    if n_testable == 0 {
        return LocalUltrametricResult {
            n_points: n,
            epsilon,
            n_testable: 0,
            mean_local_index: f64::NAN,
            median_local_index: f64::NAN,
            std_local_index: f64::NAN,
            null_mean_index: f64::NAN,
            p_value: 1.0,
            verdict: Verdict::Fail,
        };
    }

    let mean_idx = local_indices.iter().sum::<f64>() / n_testable as f64;
    let var_idx = local_indices
        .iter()
        .map(|x| (x - mean_idx).powi(2))
        .sum::<f64>()
        / n_testable as f64;
    let std_idx = var_idx.sqrt();

    let mut sorted = local_indices.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_idx = sorted[n_testable / 2];

    // Null distribution: shuffle coordinate assignments
    let mut null_means = Vec::with_capacity(n_permutations);
    let mut shuffled_coords = coords.to_vec();

    for _ in 0..n_permutations {
        shuffled_coords.shuffle(&mut rng);

        let mut null_indices = Vec::new();
        for i in 0..n {
            let neighbors: Vec<Vec<f64>> = shuffled_coords
                .iter()
                .filter(|p| euclidean_nd_sq(p, &shuffled_coords[i]) <= epsilon_sq)
                .cloned()
                .collect();

            if neighbors.len() >= 3 {
                let idx = neighborhood_ultrametric_index_nd(
                    &neighbors,
                    n_samples_per_neighborhood,
                    &mut rng,
                );
                null_indices.push(idx);
            }
        }

        if !null_indices.is_empty() {
            let null_mean = null_indices.iter().sum::<f64>() / null_indices.len() as f64;
            null_means.push(null_mean);
        }
    }

    let null_mean_index = if null_means.is_empty() {
        f64::NAN
    } else {
        null_means.iter().sum::<f64>() / null_means.len() as f64
    };

    // One-sided p-value: fraction of null means >= observed mean
    let n_ge = null_means.iter().filter(|&&m| m >= mean_idx).count();
    let p_value = if null_means.is_empty() {
        1.0
    } else {
        (n_ge as f64 + 1.0) / (null_means.len() as f64 + 1.0)
    };

    let verdict = if p_value < 0.05 {
        Verdict::Pass
    } else {
        Verdict::Fail
    };

    LocalUltrametricResult {
        n_points: n,
        epsilon,
        n_testable,
        mean_local_index: mean_idx,
        median_local_index: median_idx,
        std_local_index: std_idx,
        null_mean_index,
        p_value,
        verdict,
    }
}

/// Compute a flat upper-triangle Euclidean distance matrix from N-D coordinates.
///
/// Generalizes `euclidean_distance_matrix_3d` in dendrogram.rs to arbitrary
/// dimensionality. Output layout matches the standard upper-triangle format:
/// index(i,j) = i*n - i*(i+1)/2 + j - i - 1 for i < j.
pub fn euclidean_distance_matrix_nd(coords: &[Vec<f64>]) -> Vec<f64> {
    let n = coords.len();
    let n_pairs = n * (n - 1) / 2;
    let mut dists = vec![0.0; n_pairs];

    let mut k = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            dists[k] = euclidean_nd_sq(&coords[i], &coords[j]).sqrt();
            k += 1;
        }
    }

    dists
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neighborhood_index_all_equal() {
        let points = vec![(0.0, 0.0, 0.0); 5];
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let idx = neighborhood_ultrametric_index(&points, 100, &mut rng);
        // All distances zero -> trivially ultrametric
        assert!((idx - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_local_ultrametricity_smoke() {
        // Small random point cloud
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let coords: Vec<(f64, f64, f64)> = (0..30)
            .map(|_| {
                (
                    rng.random_range(0.0..10.0),
                    rng.random_range(0.0..10.0),
                    rng.random_range(0.0..10.0),
                )
            })
            .collect();

        let result = local_ultrametricity_test(
            &coords, 5.0, // epsilon
            500, // samples per neighborhood
            20,  // permutations (small for speed)
            42,
        );

        assert_eq!(result.n_points, 30);
        assert!(result.n_testable > 0);
        assert!(result.mean_local_index >= 0.0 && result.mean_local_index <= 1.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_local_ultrametricity_too_small_epsilon() {
        // Epsilon so small that no neighborhoods have >= 3 points
        let coords: Vec<(f64, f64, f64)> = (0..10).map(|i| (i as f64 * 100.0, 0.0, 0.0)).collect();

        let result = local_ultrametricity_test(&coords, 0.001, 100, 10, 42);

        assert_eq!(result.n_testable, 0);
        assert_eq!(result.verdict, Verdict::Fail);
    }

    // -----------------------------------------------------------------------
    // N-D generalization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_euclidean_nd_sq_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 6.0, 3.0];
        // (3^2 + 4^2 + 0^2) = 25
        assert!((euclidean_nd_sq(&a, &b) - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_nd_sq_16d() {
        // 16D: all components differ by 1 -> d^2 = 16
        let a = vec![0.0; 16];
        let b = vec![1.0; 16];
        assert!((euclidean_nd_sq(&a, &b) - 16.0).abs() < 1e-10);
    }

    #[test]
    fn test_neighborhood_index_nd_all_equal() {
        let points = vec![vec![0.0, 0.0, 0.0, 0.0]; 5];
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let idx = neighborhood_ultrametric_index_nd(&points, 100, &mut rng);
        // All distances zero -> trivially ultrametric
        assert!((idx - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_local_ultrametricity_nd_smoke_3d() {
        // N-D version on 3D data should give similar results to the 3D version
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let coords_3d: Vec<(f64, f64, f64)> = (0..30)
            .map(|_| {
                (
                    rng.random_range(0.0..10.0),
                    rng.random_range(0.0..10.0),
                    rng.random_range(0.0..10.0),
                )
            })
            .collect();

        // Convert to Vec<Vec<f64>>
        let coords_nd: Vec<Vec<f64>> = coords_3d.iter().map(|&(x, y, z)| vec![x, y, z]).collect();

        let result = local_ultrametricity_test_nd(
            &coords_nd, 5.0, // epsilon
            500, // samples per neighborhood
            20,  // permutations (small for speed)
            42,
        );

        assert_eq!(result.n_points, 30);
        assert!(result.n_testable > 0);
        assert!(result.mean_local_index >= 0.0 && result.mean_local_index <= 1.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_local_ultrametricity_nd_8d() {
        // 8D random point cloud (octonion-scale)
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let coords: Vec<Vec<f64>> = (0..40)
            .map(|_| (0..8).map(|_| rng.random_range(0.0..10.0)).collect())
            .collect();

        let result = local_ultrametricity_test_nd(
            &coords, 12.0, // larger epsilon for 8D (distances grow with sqrt(D))
            300, 10, 42,
        );

        assert_eq!(result.n_points, 40);
        assert!(
            result.n_testable > 0,
            "should find some neighborhoods in 8D"
        );
        assert!(result.mean_local_index >= 0.0 && result.mean_local_index <= 1.0);
    }

    #[test]
    fn test_local_ultrametricity_nd_16d() {
        // 16D random point cloud (sedenion-scale)
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let coords: Vec<Vec<f64>> = (0..50)
            .map(|_| (0..16).map(|_| rng.random_range(0.0..10.0)).collect())
            .collect();

        let result = local_ultrametricity_test_nd(
            &coords, 18.0, // sqrt(16)*10/2 ~ 20; use 18 to ensure neighborhoods
            200, 10, 42,
        );

        assert_eq!(result.n_points, 50);
        // In 16D, typical Euclidean distance between uniform [0,10]^16 points
        // is sqrt(16 * 100/12) ~ 11.5, so eps=18 should capture many neighbors
        assert!(
            result.n_testable > 0,
            "should find some neighborhoods in 16D"
        );
    }

    #[test]
    fn test_local_ultrametricity_nd_too_small_epsilon() {
        let coords: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 100.0, 0.0]).collect();

        let result = local_ultrametricity_test_nd(&coords, 0.001, 100, 10, 42);

        assert_eq!(result.n_testable, 0);
        assert_eq!(result.verdict, Verdict::Fail);
    }

    #[test]
    fn test_euclidean_distance_matrix_nd() {
        // 3 points in 2D forming a right triangle
        let coords = vec![vec![0.0, 0.0], vec![3.0, 0.0], vec![0.0, 4.0]];

        let dists = euclidean_distance_matrix_nd(&coords);

        // d(0,1) = 3, d(0,2) = 4, d(1,2) = 5
        assert_eq!(dists.len(), 3);
        assert!((dists[0] - 3.0).abs() < 1e-10, "d(0,1) = {}", dists[0]);
        assert!((dists[1] - 4.0).abs() < 1e-10, "d(0,2) = {}", dists[1]);
        assert!((dists[2] - 5.0).abs() < 1e-10, "d(1,2) = {}", dists[2]);
    }

    #[test]
    fn test_euclidean_distance_matrix_nd_matches_3d() {
        // Compare N-D matrix to the 3D dendrogram version for consistency
        let coords_3d = [
            (1.0, 2.0, 3.0),
            (4.0, 5.0, 6.0),
            (7.0, 8.0, 9.0),
            (0.0, 0.0, 0.0),
        ];
        let coords_nd: Vec<Vec<f64>> = coords_3d.iter().map(|&(x, y, z)| vec![x, y, z]).collect();

        let dists_nd = euclidean_distance_matrix_nd(&coords_nd);

        // Manually compute expected distances
        let expected_01 =
            ((4.0 - 1.0_f64).powi(2) + (5.0 - 2.0_f64).powi(2) + (6.0 - 3.0_f64).powi(2)).sqrt();
        assert!((dists_nd[0] - expected_01).abs() < 1e-10);
        assert_eq!(dists_nd.len(), 6); // C(4,2) = 6
    }
}

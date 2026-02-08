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
//! - **ToroidalShift** is a structured row permutation with the same
//!   caveat: informative only when row index encodes temporal or spatial
//!   ordering and the statistic is sensitive to that ordering.
//!
//! - **ColumnIndependent** is the universally informative null for
//!   ultrametric fraction tests, because it destroys inter-attribute
//!   correlation structure while preserving marginals -- exactly the
//!   joint structure that creates hierarchical distance patterns.
//!
//! # References
//!
//! - Mezzadri (2007): How to generate random matrices from compact groups
//! - Phipson & Smyth (2010): Permutation p-values should never be zero

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

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
}

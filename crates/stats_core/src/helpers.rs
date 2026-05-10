//! Canonical statistical helper functions.
//!
//! These are simple, commonly needed functions that were duplicated across
//! multiple binaries. They live in stats_core as the Tier 1 home for
//! statistical operations.
//!
//! ## Hypothesis-test p-value surface
//!
//! Three functions cover the p-values needed across the workspace.  All wrap
//! statrs distributions so no CLI crate needs a direct statrs dependency:
//!
//! - [`chi_squared_p_value`] -- upper-tail p-value (Fisher combined test, etc.)
//! - [`normal_two_tailed_p_value`] -- symmetric two-tailed p-value from z-score
//! - [`students_t_two_tailed_p_value`] -- symmetric two-tailed p-value from t-score

/// Arithmetic mean of a slice. Returns 0.0 for empty input.
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Sample standard deviation given pre-computed mean.
///
/// Uses Bessel correction (N-1 denominator). Returns 0.0 for fewer than
/// 2 elements.
pub fn std_dev(values: &[f64], mu: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let var = values
        .iter()
        .map(|v| {
            let d = v - mu;
            d * d
        })
        .sum::<f64>()
        / (values.len() - 1) as f64;
    var.sqrt()
}

/// Median of a mutable slice. Sorts in place. Returns 0.0 for empty input.
///
/// For even-length slices, returns the average of the two middle elements.
pub fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) * 0.5
    } else {
        values[mid]
    }
}

/// Chi-squared upper-tail p-value.
///
/// WHY: Chi-squared test statistics are non-negative and large values indicate
/// departure from the null, so the p-value is always the upper tail P(X > x).
/// Fisher's combined test, goodness-of-fit tests, and independence tests all
/// use this convention. Wrapping statrs here removes the direct dep from every
/// CLI crate that runs such a test.
///
/// Panics if `df <= 0`. Returns 1.0 for `statistic <= 0` (entire mass is above
/// a non-positive value for a non-negative random variable).
pub fn chi_squared_p_value(df: f64, statistic: f64) -> f64 {
    use statrs::distribution::{ChiSquared, ContinuousCDF};
    let chi = ChiSquared::new(df).expect("chi_squared_p_value: df must be positive");
    1.0 - chi.cdf(statistic)
}

/// Standard-normal two-tailed p-value from a z-score.
///
/// WHY: Mann-Whitney U and other rank-based tests produce a z-score under the
/// normal approximation and report `2 * P(Z > |z|)`. Callers that directly
/// imported `statrs::Normal` for this single operation no longer need the dep.
///
/// Returns 1.0 for z = 0, approaches 0.0 for large |z|. Always in [0, 1].
pub fn normal_two_tailed_p_value(z: f64) -> f64 {
    use statrs::distribution::{ContinuousCDF, Normal};
    let n = Normal::new(0.0, 1.0).expect("normal_two_tailed_p_value: constant params are valid");
    2.0 * (1.0 - n.cdf(z.abs()))
}

/// Student's t two-tailed p-value from a t-score and degrees of freedom.
///
/// WHY: Pearson correlation significance and Welch/Student t-tests report
/// `2 * P(T > |t|)` under the t distribution. Centralising this in stats_core
/// removes the direct statrs dep from every CLI bin that performs such a test.
///
/// Panics if `dof <= 0`. Returns 1.0 for t = 0, approaches 0.0 for large |t|.
pub fn students_t_two_tailed_p_value(t: f64, dof: f64) -> f64 {
    use statrs::distribution::{ContinuousCDF, StudentsT};
    let dist =
        StudentsT::new(0.0, 1.0, dof).expect("students_t_two_tailed_p_value: dof must be positive");
    2.0 * (1.0 - dist.cdf(t.abs()))
}

/// Compute predicted values from ridge regression: beta = (X^T X + lambda*I)^{-1} X^T y.
///
/// WHY: Ridge regression (Tikhonov regularization) is used by several CLI binaries to
/// residualize covariates from feature vectors.  Centralising the solve here removes the
/// direct nalgebra dep from those binaries; all nalgebra types are hidden behind this API.
///
/// `design_rows` is the design matrix in row-major order: each inner Vec is one sample's
/// covariate vector.  `y` must have the same length as `design_rows`.  `lambda` is the
/// L2 regularization coefficient (ridge penalty).
///
/// Returns `None` if the system is singular (LU decomposition fails) or inputs are empty.
/// Returns `Some(predictions)` where each element is the model-predicted value for that row.
pub fn ridge_predictions(design_rows: &[Vec<f64>], y: &[f64], lambda: f64) -> Option<Vec<f64>> {
    use nalgebra::{DMatrix, DVector};
    let n = design_rows.len();
    if n == 0 || n != y.len() {
        return None;
    }
    let k = design_rows.first()?.len();
    if k == 0 {
        return None;
    }
    let flat: Vec<f64> = design_rows
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect();
    let x = DMatrix::from_row_slice(n, k, &flat);
    let y_vec = DVector::from_column_slice(y);
    let xt = x.transpose();
    let solve_matrix = &xt * &x + DMatrix::<f64>::identity(k, k) * lambda;
    let beta = solve_matrix.lu().solve(&(&xt * &y_vec))?;
    let predictions = &x * &beta;
    Some(predictions.iter().copied().collect())
}

/// Compute the singular values of a matrix given in row-major order.
///
/// WHY: Several CLI bins need only the singular value spectrum (effective rank,
/// condition number, explained variance) without the full SVD decomposition.
/// Centralising this hides DMatrix construction from callers and eliminates
/// their direct nalgebra dependency.
///
/// Returns singular values in the order produced by nalgebra (largest first for
/// full SVD, though in practice order may vary -- callers should sort if required).
/// Returns an empty Vec for empty or zero-width input.
pub fn singular_values(matrix: &[Vec<f64>]) -> Vec<f64> {
    use nalgebra::DMatrix;
    let n_rows = matrix.len();
    if n_rows == 0 {
        return vec![];
    }
    let n_cols = match matrix.first() {
        Some(row) => row.len(),
        None => return vec![],
    };
    if n_cols == 0 {
        return vec![];
    }
    let flat: Vec<f64> = matrix.iter().flat_map(|row| row.iter().copied()).collect();
    let mat = DMatrix::from_row_slice(n_rows, n_cols, &flat);
    mat.svd(false, false)
        .singular_values
        .iter()
        .copied()
        .collect()
}

/// Compute the right singular vectors and singular values of a matrix.
///
/// WHY: PCA projection requires V^T to project data onto principal components.
/// Returning plain `Vec<Vec<f64>>` for the rows of V^T lets callers project
/// without any nalgebra dependency:
/// `pc_k = sum_j(centered_row[j] * v_t_rows[k][j])`.
///
/// Returns `(singular_values, v_t_rows)` where `v_t_rows[k]` is the k-th
/// right singular vector (k-th row of V^T). Both Vecs are empty for empty
/// or zero-width input.
pub fn svd_right_singular_vectors(matrix: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    use nalgebra::DMatrix;
    let n_rows = matrix.len();
    if n_rows == 0 {
        return (vec![], vec![]);
    }
    let n_cols = match matrix.first() {
        Some(row) => row.len(),
        None => return (vec![], vec![]),
    };
    if n_cols == 0 {
        return (vec![], vec![]);
    }
    let flat: Vec<f64> = matrix.iter().flat_map(|row| row.iter().copied()).collect();
    let mat = DMatrix::from_row_slice(n_rows, n_cols, &flat);
    let svd = mat.svd(false, true);
    let svals: Vec<f64> = svd.singular_values.iter().copied().collect();
    let v_t_rows = match svd.v_t {
        Some(v_t) => (0..v_t.nrows())
            .map(|k| v_t.row(k).iter().copied().collect())
            .collect(),
        None => vec![],
    };
    (svals, v_t_rows)
}

/// Compute the R^2 of the least-squares projection of `y` onto the column space of `design`.
///
/// WHY: Tests whether a given basis (design matrix) explains observed residuals.
/// Uses the SVD pseudoinverse to handle rank-deficient design matrices:
/// `y_hat = A * pinv(A) * y = A * V * S^{-1} * U^T * y`.
/// Centralising this here removes the DMatrix dep from every bin that performs
/// such a basis-projection test.
///
/// `design_rows` is the design matrix in row-major order. `y` must have the same
/// length as `design_rows`. Returns 0.0 for degenerate input (empty, length
/// mismatch, or zero total variance in `y`).
pub fn svd_projection_r_squared(design_rows: &[Vec<f64>], y: &[f64]) -> f64 {
    use nalgebra::{DMatrix, DVector};
    let n = design_rows.len();
    if n == 0 || n != y.len() {
        return 0.0;
    }
    let n_cols = match design_rows.first() {
        Some(row) => row.len(),
        None => return 0.0,
    };
    if n_cols == 0 {
        return 0.0;
    }
    let flat: Vec<f64> = design_rows
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect();
    let a = DMatrix::from_row_slice(n, n_cols, &flat);
    let y_vec = DVector::from_column_slice(y);
    let svd = a.clone().svd(true, true);
    let (u, v_t) = match (svd.u.as_ref(), svd.v_t.as_ref()) {
        (Some(u), Some(vt)) => (u, vt),
        _ => return 0.0,
    };
    let sigma = &svd.singular_values;
    let threshold = if sigma[0].abs() > 0.0 {
        1e-12 * sigma[0].abs()
    } else {
        1e-30
    };
    // y_hat = A * V * S^{-1} * U^T * y
    let mut coeffs = DVector::zeros(sigma.len());
    for k in 0..sigma.len() {
        if sigma[k] < threshold {
            continue;
        }
        coeffs[k] = u.column(k).dot(&y_vec) / sigma[k];
    }
    let y_hat = &a * &(v_t.transpose() * &coeffs);
    let y_mean = y_vec.mean();
    let ss_tot: f64 = y_vec.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let ss_res: f64 = (&y_vec - &y_hat).iter().map(|r| r.powi(2)).sum();
    if ss_tot < 1e-30 {
        return 0.0;
    }
    1.0 - ss_res / ss_tot
}

/// Center a matrix column-wise (subtract each column's mean).
///
/// WHY: PCA requires zero-mean columns. Centralising the centering step here
/// eliminates the direct nalgebra dependency from callers that only need to
/// prepare data before calling `svd_right_singular_vectors`.
///
/// Returns a new matrix with the same shape. Empty input returns empty output.
pub fn center_columns_flat(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n_rows = data.len();
    if n_rows == 0 {
        return vec![];
    }
    let n_cols = data[0].len();
    if n_cols == 0 {
        return data.to_vec();
    }
    let col_means: Vec<f64> = (0..n_cols)
        .map(|j| data.iter().map(|row| row[j]).sum::<f64>() / n_rows as f64)
        .collect();
    data.iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(j, &v)| v - col_means[j])
                .collect()
        })
        .collect()
}

/// Project rows of `data` onto the first `k` right singular vectors.
///
/// WHY: PCA projection is the workhorse of dimensionality reduction. Taking
/// `v_t_rows` from `svd_right_singular_vectors` and data from
/// `center_columns_flat` gives a fully nalgebra-free projection pipeline.
///
/// `data` is the row-major data matrix. `v_t_rows[i]` is the i-th right
/// singular vector (i-th row of V^T). Each output row has `k` elements:
/// `result[i][j] = sum_col(data[i][col] * v_t_rows[j][col])`.
///
/// Clamps `k` to the number of available singular vectors.
pub fn project_rows_onto_components(
    data: &[Vec<f64>],
    v_t_rows: &[Vec<f64>],
    k: usize,
) -> Vec<Vec<f64>> {
    let use_k = k.min(v_t_rows.len());
    data.iter()
        .map(|row| {
            (0..use_k)
                .map(|comp| {
                    row.iter()
                        .zip(v_t_rows[comp].iter())
                        .map(|(a, b)| a * b)
                        .sum::<f64>()
                })
                .collect()
        })
        .collect()
}

/// Ordinary least squares via SVD on normal equations.
///
/// WHY: OLS solved through the normal equations `(X^T X) beta = X^T y` is
/// numerically stabler with SVD than LU when columns are nearly collinear.
/// Centralising this eliminates the DMatrix/DVector dependency from CLI bins
/// that perform regression.
///
/// `design_rows` is the design matrix in row-major order; `y` must have the
/// same length. Returns `None` for degenerate input (empty, size mismatch,
/// or zero-width design matrix). The SVD tolerance is 1e-12 of the largest
/// singular value.
pub fn ols_svd_solve(design_rows: &[Vec<f64>], y: &[f64]) -> Option<Vec<f64>> {
    use nalgebra::{DMatrix, DVector};
    let n = design_rows.len();
    if n == 0 || n != y.len() {
        return None;
    }
    let k = design_rows.first()?.len();
    if k == 0 {
        return None;
    }
    let flat: Vec<f64> = design_rows
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect();
    let x = DMatrix::from_row_slice(n, k, &flat);
    let y_vec = DVector::from_column_slice(y);
    let xt = x.transpose();
    let xtx = &xt * &x;
    let xty = &xt * &y_vec;
    xtx.svd(true, true)
        .solve(&xty, 1e-12)
        .ok()
        .map(|beta| beta.iter().copied().collect())
}

/// Coefficient of determination (R^2) from pre-computed predictions.
///
/// WHY: When predictions are already available (from OLS beta applied to rows),
/// R^2 is a simple O(n) calculation. This avoids re-running SVD just to get the
/// quality metric. Returns 0.0 for empty input, length mismatch, or zero
/// total variance.
pub fn r_squared_from_predictions(y_true: &[f64], y_pred: &[f64]) -> f64 {
    if y_true.len() != y_pred.len() || y_true.is_empty() {
        return 0.0;
    }
    let mean = y_true.iter().sum::<f64>() / y_true.len() as f64;
    let ss_tot: f64 = y_true.iter().map(|yi| (yi - mean).powi(2)).sum();
    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(yt, yp)| (yt - yp).powi(2))
        .sum();
    if ss_tot < 1e-30 {
        return 0.0;
    }
    1.0 - ss_res / ss_tot
}

/// Compute a histogram of values.
///
/// Returns (bin_centers, counts).
pub fn histogram(values: &[f64], n_bins: usize, min: f64, max: f64) -> (Vec<f64>, Vec<usize>) {
    if values.is_empty() || n_bins == 0 || max <= min {
        return (vec![], vec![]);
    }

    let mut counts = vec![0; n_bins];
    let bin_width = (max - min) / n_bins as f64;
    let bin_centers: Vec<f64> = (0..n_bins)
        .map(|i| min + (i as f64 + 0.5) * bin_width)
        .collect();

    for &v in values {
        if v >= min && v <= max {
            let mut bin = ((v - min) / bin_width).floor() as usize;
            if bin >= n_bins {
                bin = n_bins - 1;
            }
            counts[bin] += 1;
        }
    }

    (bin_centers, counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_basic() {
        assert!((mean(&[1.0, 2.0, 3.0, 4.0, 5.0]) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_mean_empty() {
        assert!((mean(&[]) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_mean_single() {
        assert!((mean(&[42.0]) - 42.0).abs() < 1e-12);
    }

    #[test]
    fn test_std_dev_basic() {
        let vals = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let mu = mean(&vals);
        let sd = std_dev(&vals, mu);
        // Known population std dev is 2.0, sample std dev ~2.138
        assert!((sd - 2.138).abs() < 0.01, "std_dev={}", sd);
    }

    #[test]
    fn test_std_dev_empty() {
        assert!((std_dev(&[], 0.0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_std_dev_single() {
        assert!((std_dev(&[5.0], 5.0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_median_odd() {
        let mut vals = vec![3.0, 1.0, 2.0];
        assert!((median(&mut vals) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_median_even() {
        let mut vals = vec![4.0, 1.0, 3.0, 2.0];
        assert!((median(&mut vals) - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_median_empty() {
        assert!((median(&mut []) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_median_single() {
        let mut vals = vec![42.0];
        assert!((median(&mut vals) - 42.0).abs() < 1e-12);
    }

    // --- chi_squared_p_value ---

    #[test]
    fn test_chi_squared_p_value_at_zero() {
        // Entire probability mass lies above 0 for a non-negative r.v.
        let p = chi_squared_p_value(2.0, 0.0);
        assert!((p - 1.0).abs() < 1e-12, "p={}", p);
    }

    #[test]
    fn test_chi_squared_p_value_large_statistic() {
        // Very large statistic -> negligible tail probability.
        let p = chi_squared_p_value(2.0, 100.0);
        assert!(p < 1e-20, "p={}", p);
    }

    #[test]
    fn test_chi_squared_p_value_known_analytic() {
        // chi2(df=2) is Exponential(1/2): CDF = 1 - exp(-x/2).
        // At x=2.0: p-value = exp(-1.0) ~= 0.36787944117.
        let p = chi_squared_p_value(2.0, 2.0);
        assert!((p - (-1.0_f64).exp()).abs() < 1e-9, "p={}", p);
    }

    // --- normal_two_tailed_p_value ---

    #[test]
    fn test_normal_two_tailed_at_zero() {
        // z=0: 2*(1 - Phi(0)) = 2*0.5 = 1.0 exactly.
        let p = normal_two_tailed_p_value(0.0);
        assert!((p - 1.0).abs() < 1e-12, "p={}", p);
    }

    #[test]
    fn test_normal_two_tailed_large_z() {
        // Large |z| -> nearly zero p-value.
        let p = normal_two_tailed_p_value(10.0);
        assert!(p < 1e-20, "p={}", p);
    }

    #[test]
    fn test_normal_two_tailed_symmetric() {
        // Two-tailed p-value is symmetric: p(z) == p(-z).
        let p_pos = normal_two_tailed_p_value(1.5);
        let p_neg = normal_two_tailed_p_value(-1.5);
        assert!(
            (p_pos - p_neg).abs() < 1e-15,
            "asymmetry: {} vs {}",
            p_pos,
            p_neg
        );
    }

    #[test]
    fn test_normal_two_tailed_known_z196() {
        // z ~= 1.96 gives the canonical alpha=0.05 threshold.
        let p = normal_two_tailed_p_value(1.959_963_985);
        assert!((p - 0.05).abs() < 1e-6, "p={}", p);
    }

    // --- students_t_two_tailed_p_value ---

    #[test]
    fn test_students_t_at_zero() {
        // t=0: 2*(1 - T_cdf(0)) = 1.0 for any dof.
        let p = students_t_two_tailed_p_value(0.0, 10.0);
        assert!((p - 1.0).abs() < 1e-12, "p={}", p);
    }

    #[test]
    fn test_students_t_large_t() {
        // Large |t| -> negligible p-value.
        // NOTE: dof=5 has heavy tails (Cauchy-like), so the tail at t=100 is ~2.45e-08,
        // not as extreme as a normal tail. Threshold reflects actual t-distribution behavior.
        let p = students_t_two_tailed_p_value(100.0, 5.0);
        assert!(p < 1e-4, "p={}", p);
    }

    #[test]
    fn test_students_t_symmetric() {
        // Two-tailed p-value is symmetric.
        let p_pos = students_t_two_tailed_p_value(2.0, 10.0);
        let p_neg = students_t_two_tailed_p_value(-2.0, 10.0);
        assert!(
            (p_pos - p_neg).abs() < 1e-15,
            "asymmetry: {} vs {}",
            p_pos,
            p_neg
        );
    }

    #[test]
    fn test_students_t_approaches_normal_at_large_dof() {
        // t(dof=1000) at z=1.96 should be very close to normal_two_tailed_p_value(1.96).
        // Tolerance 1e-3: statrs t vs normal CDF implementations diverge slightly at dof=1000.
        let p_t = students_t_two_tailed_p_value(1.959_963_985, 1000.0);
        let p_n = normal_two_tailed_p_value(1.959_963_985);
        assert!((p_t - p_n).abs() < 1e-3, "t={} normal={}", p_t, p_n);
    }

    // --- singular_values ---

    #[test]
    fn singular_values_identity_2x2() {
        // 2x2 identity matrix has singular values [1, 1]
        let mat = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let svals = singular_values(&mat);
        assert_eq!(svals.len(), 2);
        for s in &svals {
            assert!((s - 1.0).abs() < 1e-12, "sval={}", s);
        }
    }

    #[test]
    fn singular_values_diagonal_2x2() {
        // Diagonal [[3,0],[0,2]] has singular values {3, 2}
        let mat = vec![vec![3.0, 0.0], vec![0.0, 2.0]];
        let mut svals = singular_values(&mat);
        svals.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert!((svals[0] - 3.0).abs() < 1e-10, "svals={:?}", svals);
        assert!((svals[1] - 2.0).abs() < 1e-10, "svals={:?}", svals);
    }

    #[test]
    fn singular_values_empty_returns_empty() {
        assert!(singular_values(&[]).is_empty());
        assert!(singular_values(&[vec![]]).is_empty());
    }

    // --- svd_right_singular_vectors ---

    #[test]
    fn svd_right_singular_vectors_identity_2x2() {
        let mat = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let (svals, v_t) = svd_right_singular_vectors(&mat);
        assert_eq!(svals.len(), 2);
        // V^T rows are orthonormal
        assert_eq!(v_t.len(), 2);
        assert_eq!(v_t[0].len(), 2);
        let norm_row0: f64 = v_t[0].iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm_row0 - 1.0).abs() < 1e-12, "row0 norm={}", norm_row0);
    }

    #[test]
    fn svd_right_singular_vectors_v_t_orthonormal() {
        // 3x2 matrix: rows of V^T should be orthonormal (2 vectors in R^2)
        let mat = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let (_svals, v_t) = svd_right_singular_vectors(&mat);
        assert_eq!(v_t.len(), 2);
        // Each row is unit-length
        for row in &v_t {
            let n: f64 = row.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!((n - 1.0).abs() < 1e-12, "row norm={}", n);
        }
        // Rows are orthogonal
        let dot: f64 = v_t[0].iter().zip(v_t[1].iter()).map(|(a, b)| a * b).sum();
        assert!(dot.abs() < 1e-12, "dot={}", dot);
    }

    // --- svd_projection_r_squared ---

    #[test]
    fn svd_projection_r_squared_perfect_fit() {
        // y is exactly in the column space of A -> R^2 should be ~1.0
        let design = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let y = vec![2.0, 3.0, 5.0]; // exactly 2*col0 + 3*col1
        let r2 = svd_projection_r_squared(&design, &y);
        assert!((r2 - 1.0).abs() < 1e-10, "r2={}", r2);
    }

    #[test]
    fn svd_projection_r_squared_zero_variance() {
        // y is constant -> R^2 is 0 (no variance to explain)
        let design = vec![vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![5.0, 5.0, 5.0];
        let r2 = svd_projection_r_squared(&design, &y);
        assert!((r2 - 0.0).abs() < 1e-10, "r2={}", r2);
    }

    #[test]
    fn svd_projection_r_squared_empty() {
        assert!((svd_projection_r_squared(&[], &[]) - 0.0).abs() < 1e-12);
    }
}

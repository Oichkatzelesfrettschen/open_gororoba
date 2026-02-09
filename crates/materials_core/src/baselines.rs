//! OLS linear regression baselines for materials property prediction.
//!
//! Provides SVD-based ordinary least squares with train/test splitting
//! and standard evaluation metrics (MAE, RMSE, R^2).

use nalgebra::{DMatrix, DVector};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

/// Results from a baseline regression run.
#[derive(Debug, Clone)]
pub struct RegressionResult {
    /// Fitted coefficients (excluding intercept).
    pub coefficients: Vec<f64>,
    /// Intercept (bias) term.
    pub intercept: f64,
    /// Mean absolute error on the test set.
    pub mae: f64,
    /// Root mean squared error on the test set.
    pub rmse: f64,
    /// Coefficient of determination (R^2) on the test set.
    pub r_squared: f64,
    /// Number of training samples.
    pub n_train: usize,
    /// Number of test samples.
    pub n_test: usize,
}

/// Deterministic train/test split using ChaCha8 PRNG.
///
/// Returns (train_indices, test_indices).  `test_fraction` is the
/// fraction of data to hold out for testing (e.g. 0.2 for 80/20 split).
pub fn train_test_split(n: usize, test_fraction: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);

    let n_test = ((n as f64) * test_fraction).round() as usize;
    let n_test = n_test.max(1).min(n - 1);

    let test = indices[..n_test].to_vec();
    let train = indices[n_test..].to_vec();
    (train, test)
}

/// Fit OLS regression via SVD pseudoinverse.
///
/// The design matrix `x_train` has one row per sample and one column per
/// feature.  An intercept column is appended internally.
///
/// Returns (coefficients_including_intercept, intercept).
/// The intercept is the last element of the coefficient vector.
pub fn ols_fit(x_train: &DMatrix<f64>, y_train: &DVector<f64>) -> (DVector<f64>, f64) {
    let n = x_train.nrows();
    let p = x_train.ncols();

    // Append ones column for intercept
    let mut x_aug = DMatrix::zeros(n, p + 1);
    x_aug.view_mut((0, 0), (n, p)).copy_from(x_train);
    for i in 0..n {
        x_aug[(i, p)] = 1.0;
    }

    // SVD-based pseudoinverse solve: beta = pinv(X) * y
    let svd = x_aug.svd(true, true);
    let beta = svd
        .solve(y_train, 1e-12)
        .unwrap_or_else(|_| DVector::zeros(p + 1));

    let intercept = beta[p];
    let coefficients = beta.rows(0, p).into_owned();
    (coefficients, intercept)
}

/// Predict targets from features using fitted coefficients.
pub fn predict(x: &DMatrix<f64>, coefficients: &DVector<f64>, intercept: f64) -> DVector<f64> {
    x * coefficients + DVector::from_element(x.nrows(), intercept)
}

/// Compute MAE, RMSE, and R^2 from true and predicted values.
pub fn evaluate(y_true: &DVector<f64>, y_pred: &DVector<f64>) -> (f64, f64, f64) {
    let n = y_true.len() as f64;
    let diff = y_true - y_pred;

    let mae = diff.iter().map(|d| d.abs()).sum::<f64>() / n;
    let mse = diff.dot(&diff) / n;
    let rmse = mse.sqrt();

    // R^2 = 1 - SS_res / SS_tot
    let y_mean = y_true.mean();
    let ss_tot: f64 = y_true.iter().map(|y| (y - y_mean).powi(2)).sum();
    let ss_res: f64 = diff.iter().map(|d| d.powi(2)).sum();
    let r_squared = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };

    (mae, rmse, r_squared)
}

/// Run the full baseline pipeline: split, fit, predict, evaluate.
///
/// `features` is a slice of feature vectors (one per sample).
/// `targets` is a slice of target values.
/// `test_fraction` is the hold-out fraction (e.g. 0.2).
/// `seed` is the PRNG seed for reproducibility.
pub fn run_baseline(
    features: &[Vec<f64>],
    targets: &[f64],
    test_fraction: f64,
    seed: u64,
) -> Result<RegressionResult, String> {
    let n = features.len();
    if n != targets.len() {
        return Err(format!(
            "Feature/target length mismatch: {} vs {}",
            n,
            targets.len()
        ));
    }
    if n < 3 {
        return Err("Need at least 3 samples for train/test split".to_string());
    }

    let p = features[0].len();
    if p == 0 {
        return Err("Feature vectors are empty".to_string());
    }

    let (train_idx, test_idx) = train_test_split(n, test_fraction, seed);

    // Build design matrices
    let x_train = DMatrix::from_fn(train_idx.len(), p, |i, j| features[train_idx[i]][j]);
    let y_train = DVector::from_fn(train_idx.len(), |i, _| targets[train_idx[i]]);
    let x_test = DMatrix::from_fn(test_idx.len(), p, |i, j| features[test_idx[i]][j]);
    let y_test = DVector::from_fn(test_idx.len(), |i, _| targets[test_idx[i]]);

    let (coefficients, intercept) = ols_fit(&x_train, &y_train);
    let y_pred = predict(&x_test, &coefficients, intercept);
    let (mae, rmse, r_squared) = evaluate(&y_test, &y_pred);

    Ok(RegressionResult {
        coefficients: coefficients.as_slice().to_vec(),
        intercept,
        mae,
        rmse,
        r_squared,
        n_train: train_idx.len(),
        n_test: test_idx.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ols_perfect_line() {
        // y = 2x + 1, should get R^2 ~= 1.0
        let n = 50;
        let x = DMatrix::from_fn(n, 1, |i, _| i as f64);
        let y = DVector::from_fn(n, |i, _| 2.0 * (i as f64) + 1.0);

        let (coeff, intercept) = ols_fit(&x, &y);
        assert!(
            (coeff[0] - 2.0).abs() < 1e-8,
            "Slope should be ~2.0, got {}",
            coeff[0]
        );
        assert!(
            (intercept - 1.0).abs() < 1e-8,
            "Intercept should be ~1.0, got {intercept}"
        );

        let y_pred = predict(&x, &coeff, intercept);
        let (mae, _rmse, r2) = evaluate(&y, &y_pred);
        assert!(r2 > 0.9999, "R^2 should be ~1.0, got {r2}");
        assert!(mae < 1e-8, "MAE should be ~0, got {mae}");
    }

    #[test]
    fn test_ols_with_noise() {
        // y = x + noise
        let mut rng = ChaCha8Rng::seed_from_u64(12345);
        let n = 100;
        let x = DMatrix::from_fn(n, 1, |i, _| i as f64);
        let y = DVector::from_fn(n, |i, _| {
            let noise: f64 = rng.gen_range(-5.0..5.0);
            (i as f64) + noise
        });

        let (coeff, _intercept) = ols_fit(&x, &y);
        let y_pred = predict(&x, &coeff, _intercept);
        let (_mae, _rmse, r2) = evaluate(&y, &y_pred);

        // With noise, R^2 should be between 0 and 1
        assert!(r2 > 0.0 && r2 < 1.0, "R^2 should be in (0,1), got {r2}");
        // Slope should be approximately 1.0
        assert!(
            (coeff[0] - 1.0).abs() < 0.5,
            "Slope should be ~1.0, got {}",
            coeff[0]
        );
    }

    #[test]
    fn test_train_test_split_sizes() {
        let (train, test) = train_test_split(100, 0.2, 42);
        assert_eq!(train.len() + test.len(), 100);
        assert_eq!(test.len(), 20);
        assert_eq!(train.len(), 80);
    }

    #[test]
    fn test_train_test_split_deterministic() {
        let (train1, test1) = train_test_split(100, 0.2, 42);
        let (train2, test2) = train_test_split(100, 0.2, 42);
        assert_eq!(train1, train2);
        assert_eq!(test1, test2);

        // Different seed should give different split
        let (train3, _test3) = train_test_split(100, 0.2, 99);
        assert_ne!(train1, train3);
    }

    #[test]
    fn test_evaluate_metrics() {
        let y_true = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y_pred = DVector::from_vec(vec![1.1, 2.2, 2.8, 4.1, 4.9]);

        let (mae, rmse, r2) = evaluate(&y_true, &y_pred);

        // MAE = mean(|0.1, 0.2, 0.2, 0.1, 0.1|) = 0.14
        assert!((mae - 0.14).abs() < 1e-10, "MAE should be 0.14, got {mae}");

        // MSE = mean(0.01, 0.04, 0.04, 0.01, 0.01) = 0.022
        // RMSE = sqrt(0.022) ~ 0.1483
        assert!(
            (rmse - 0.022_f64.sqrt()).abs() < 1e-10,
            "RMSE should be ~0.1483, got {rmse}"
        );

        // SS_tot = 10.0, SS_res = 0.11, R^2 = 1 - 0.11/10 = 0.989
        assert!((r2 - 0.989).abs() < 1e-10, "R^2 should be 0.989, got {r2}");
    }

    #[test]
    fn test_run_baseline_pipeline() {
        // Create a simple dataset: y = 3*x1 + 2*x2 + 1
        let n = 200;
        let features: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64, (i * 2) as f64]).collect();
        let targets: Vec<f64> = features
            .iter()
            .map(|f| 3.0 * f[0] + 2.0 * f[1] + 1.0)
            .collect();

        let result = run_baseline(&features, &targets, 0.2, 42).unwrap();
        assert!(
            result.r_squared > 0.999,
            "Perfect linear data should give R^2 ~1.0, got {}",
            result.r_squared
        );
        assert_eq!(result.n_train + result.n_test, n);
    }
}

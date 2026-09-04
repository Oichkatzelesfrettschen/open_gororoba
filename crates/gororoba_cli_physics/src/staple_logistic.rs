//! Shared logistic fitting and score metrics for Staples grouped validation.

use anyhow::{Result, bail};
use rayon::prelude::*;

/// Fixed iteration budget used by the grouped logistic fits.
pub const MAX_NEWTON_ITERS: usize = 25;
/// Relative deviance tolerance.  The penalized deviance runs near 4e6 on the
/// full benchmark, where an absolute 1e-8 sits about a decade above the f64
/// resolution of the sum and depends on the rayon reduction order, so the
/// stopping rule would never fire and every fold would silently exhaust the
/// iteration cap.
pub const DEVIANCE_REL_TOL: f64 = 1e-8;
const N_CALIBRATION_BINS: usize = 10;

/// Sigmoid branched on the sign of `eta` so neither exp() overflows.
pub fn sigmoid(eta: f64) -> f64 {
    if eta >= 0.0 {
        1.0 / (1.0 + (-eta).exp())
    } else {
        let e = eta.exp();
        e / (1.0 + e)
    }
}

/// softplus(eta) = ln(1 + exp(eta)), evaluated through ln_1p on the negative
/// branch so a large positive eta returns eta rather than inf.
fn softplus(eta: f64) -> f64 {
    eta.max(0.0) + (-(eta.abs())).exp().ln_1p()
}

/// Per-sample logistic loss from the logit: softplus(eta) - y*eta.
pub fn logit_loss(eta: f64, y: f64) -> f64 {
    softplus(eta) - y * eta
}

/// In-place Cholesky factorization of a symmetric positive definite matrix
/// stored row-major, returning the lower triangle. Ridge regularizes the slope
/// block; the unpenalized intercept still needs positive total logistic weight.
fn cholesky(a: &mut [f64], p: usize) -> Result<()> {
    for i in 0..p {
        for j in 0..=i {
            let mut s = a[i * p + j];
            for k in 0..j {
                s -= a[i * p + k] * a[j * p + k];
            }
            if i == j {
                if !s.is_finite() || s <= 0.0 {
                    bail!("Hessian lost positive definiteness at pivot {i}");
                }
                a[i * p + i] = s.sqrt();
            } else {
                a[i * p + j] = s / a[j * p + j];
            }
        }
        for j in (i + 1)..p {
            a[i * p + j] = 0.0;
        }
    }
    Ok(())
}

/// Solve `L L^T x = b` for a lower-triangular Cholesky factor.
fn cholesky_solve(l: &[f64], b: &[f64], p: usize) -> Vec<f64> {
    let mut y = vec![0.0f64; p];
    for i in 0..p {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * p + k] * y[k];
        }
        y[i] = s / l[i * p + i];
    }
    let mut x = vec![0.0f64; p];
    for i in (0..p).rev() {
        let mut s = y[i];
        for k in (i + 1)..p {
            s -= l[k * p + i] * x[k];
        }
        x[i] = s / l[i * p + i];
    }
    x
}

// ---------------------------------------------------------------------------
// IRLS
// ---------------------------------------------------------------------------

/// Fitted coefficients on standardized features plus the fit trajectory.
#[derive(Debug, Clone)]
pub struct FitResult {
    /// `beta[0]` is the intercept; `beta[1..]` follow `cols` order.
    pub beta: Vec<f64>,
    /// Penalized deviances evaluated before each Newton update.
    pub deviance: Vec<f64>,
    /// Number of Newton updates attempted within the fixed iteration budget.
    pub iterations: usize,
    /// Whether the relative deviance stopping condition fired.
    pub converged: bool,
}

/// Newton-Raphson (IRLS) logistic fit over `rows` of a row-major f32 feature
/// block, using the feature columns named in `cols` after standardization by
/// `(mean, std)`.  The ridge penalty applies to the standardized slopes only;
/// the unpenalized intercept avoids shrinking the fitted prevalence toward 0.5.
/// Repeated row indices retain multiplicity for cluster resampling. Validation
/// covers selected feature columns and active labels; unused rows may be absent
/// from the statistical population. Invalid inputs and nonfinite Newton states
/// return an error. An exhausted iteration budget returns `converged = false`.
#[allow(clippy::too_many_arguments)]
pub fn fit_irls(
    feats: &[f32],
    stride: usize,
    cols: &[usize],
    rows: &[u32],
    labels: &[u8],
    mean: &[f64],
    std: &[f64],
    ridge: f64,
) -> Result<FitResult> {
    if stride == 0 || labels.len().checked_mul(stride) != Some(feats.len()) {
        bail!("feature block length must equal label count times positive stride");
    }
    if rows.is_empty() {
        bail!("logistic fitting requires active rows");
    }
    if cols.len() != mean.len() || cols.len() != std.len() {
        bail!("column, mean, and standard-deviation lengths must agree");
    }
    if !ridge.is_finite() || ridge < 0.0 {
        bail!("ridge must be finite and nonnegative");
    }
    for (position, &column) in cols.iter().enumerate() {
        if column >= stride
            || !mean[position].is_finite()
            || !std[position].is_finite()
            || std[position] <= 0.0
        {
            bail!("invalid column or standardization at feature position {position}");
        }
    }
    let mut positives = 0usize;
    for &row in rows {
        let row = row as usize;
        if row >= labels.len() || labels[row] > 1 {
            bail!("invalid row index or binary label at row {row}");
        }
        positives += usize::from(labels[row]);
        for (position, &column) in cols.iter().enumerate() {
            let standardized =
                (f64::from(feats[row * stride + column]) - mean[position]) / std[position];
            if !standardized.is_finite() {
                bail!("nonfinite standardized feature at row {row}, column {column}");
            }
        }
    }
    if positives == 0 || positives == rows.len() {
        bail!("logistic fitting requires both classes among active rows");
    }
    let d = cols.len();
    let p = d + 1;
    let mut beta = vec![0.0f64; p];
    // Unpenalized intercept-only start: the log odds of the observed rate.
    let pos = rows.iter().filter(|&&r| labels[r as usize] == 1).count() as f64;
    let n = rows.len() as f64;
    let rate = (pos / n).clamp(1e-12, 1.0 - 1e-12);
    beta[0] = (rate / (1.0 - rate)).ln();

    let mut trajectory = Vec::with_capacity(MAX_NEWTON_ITERS + 1);
    let mut prev_dev = f64::INFINITY;
    let mut iterations = 0usize;
    let mut converged = false;

    for _ in 0..MAX_NEWTON_ITERS {
        iterations += 1;
        // One pass accumulates the deviance, the gradient and the upper
        // triangle of X^T W X.  Chunked so rayon reduces partial sums.
        let (dev, grad, hess) = rows
            .par_chunks(1 << 16)
            .map(|chunk| {
                let mut dev = 0.0f64;
                let mut grad = vec![0.0f64; p];
                let mut hess = vec![0.0f64; p * p];
                let mut x = vec![0.0f64; p];
                x[0] = 1.0;
                for &r in chunk {
                    let base = r as usize * stride;
                    for (k, &c) in cols.iter().enumerate() {
                        x[k + 1] = (f64::from(feats[base + c]) - mean[k]) / std[k];
                    }
                    let mut eta = 0.0f64;
                    for j in 0..p {
                        eta += beta[j] * x[j];
                    }
                    let y = f64::from(labels[r as usize]);
                    dev += logit_loss(eta, y);
                    let mu = sigmoid(eta);
                    let w = mu * (1.0 - mu);
                    let resid = y - mu;
                    for j in 0..p {
                        grad[j] += resid * x[j];
                        let wxj = w * x[j];
                        for k in j..p {
                            hess[j * p + k] += wxj * x[k];
                        }
                    }
                }
                (dev, grad, hess)
            })
            .reduce(
                || (0.0f64, vec![0.0f64; p], vec![0.0f64; p * p]),
                |mut a, b| {
                    a.0 += b.0;
                    for j in 0..p {
                        a.1[j] += b.1[j];
                    }
                    for j in 0..(p * p) {
                        a.2[j] += b.2[j];
                    }
                    a
                },
            );

        let mut grad = grad;
        let mut hess = hess;
        for j in 1..p {
            grad[j] -= ridge * beta[j];
            hess[j * p + j] += ridge;
        }
        for j in 0..p {
            for k in 0..j {
                hess[j * p + k] = hess[k * p + j];
            }
        }
        let penalty: f64 = ridge * beta[1..].iter().map(|b| b * b).sum::<f64>() * 0.5;
        let pen_dev = 2.0 * (dev + penalty);
        if !pen_dev.is_finite() || grad.iter().chain(&hess).any(|value| !value.is_finite()) {
            bail!("nonfinite Newton objective, gradient, or Hessian");
        }
        trajectory.push(pen_dev);

        cholesky(&mut hess, p)?;
        let step = cholesky_solve(&hess, &grad, p);
        for j in 0..p {
            beta[j] += step[j];
        }
        if beta.iter().any(|coefficient| !coefficient.is_finite()) {
            bail!("nonfinite Newton coefficient");
        }

        if (prev_dev - pen_dev).abs() / prev_dev.abs().max(1.0) < DEVIANCE_REL_TOL {
            converged = true;
            break;
        }
        prev_dev = pen_dev;
    }

    Ok(FitResult {
        beta,
        deviance: trajectory,
        iterations,
        converged,
    })
}

// ---------------------------------------------------------------------------
// metrics
// ---------------------------------------------------------------------------

fn valid_sorted_scores<Score: Copy + PartialOrd + Into<f64>>(
    scores: &[Score],
    labels: &[u8],
) -> bool {
    !scores.is_empty()
        && scores.len() == labels.len()
        && scores.iter().all(|&score| score.into().is_finite())
        && labels.iter().all(|&label| label <= 1)
        && scores.windows(2).all(|pair| pair[0] >= pair[1])
}

fn valid_weights<Score: Copy + PartialOrd + Into<f64>>(
    scores: &[Score],
    labels: &[u8],
    weights: &[f64],
) -> bool {
    valid_sorted_scores(scores, labels)
        && scores.len() == weights.len()
        && weights
            .iter()
            .all(|weight| weight.is_finite() && *weight >= 0.0)
        && weights.iter().sum::<f64>().is_finite()
}

/// Weighted ROC-AUC over samples pre-sorted by descending score, with tied
/// scores collapsed into one group so a tie contributes half its rectangle.
/// This is the Mann-Whitney statistic with average ranks.
/// Invalid inputs or missing positive-weight classes return NaN.
pub fn weighted_auc<Score: Copy + PartialOrd + Into<f64>>(
    scores: &[Score],
    labels: &[u8],
    weights: &[f64],
) -> f64 {
    if !valid_weights(scores, labels, weights) {
        return f64::NAN;
    }
    let n = scores.len();
    let mut total_pos = 0.0f64;
    let mut total_neg = 0.0f64;
    for i in 0..n {
        if labels[i] == 1 {
            total_pos += weights[i];
        } else {
            total_neg += weights[i];
        }
    }
    if total_pos <= 0.0 || total_neg <= 0.0 {
        return f64::NAN;
    }
    // Descending order, so the negatives a positive outranks are the ones
    // still ahead in the scan; a tied group splits its rectangle in half.
    let mut auc = 0.0f64;
    let mut neg_seen = 0.0f64;
    let mut i = 0usize;
    while i < n {
        let mut j = i;
        let mut gp = 0.0f64;
        let mut gn = 0.0f64;
        while j < n && scores[j] == scores[i] {
            let w = weights[j];
            if labels[j] == 1 {
                gp += w
            } else {
                gn += w
            }
            j += 1;
        }
        auc += gp * ((total_neg - neg_seen - gn) + 0.5 * gn);
        neg_seen += gn;
        i = j;
    }
    auc / (total_pos * total_neg)
}

/// Weighted average precision over samples pre-sorted by descending score.
/// One precision/recall point per distinct score, matching the
/// distinct-threshold convention, and the same tie grouping as `weighted_auc`.
/// Invalid inputs or absent positive-class weight return NaN.
pub fn weighted_average_precision<Score: Copy + PartialOrd + Into<f64>>(
    scores: &[Score],
    labels: &[u8],
    weights: &[f64],
) -> f64 {
    if !valid_weights(scores, labels, weights) {
        return f64::NAN;
    }
    let n = scores.len();
    let total_pos: f64 = (0..n).filter(|&i| labels[i] == 1).map(|i| weights[i]).sum();
    if total_pos <= 0.0 {
        return f64::NAN;
    }
    let mut tp = 0.0f64;
    let mut fp = 0.0f64;
    let mut prev_recall = 0.0f64;
    let mut ap = 0.0f64;
    let mut i = 0usize;
    while i < n {
        let mut j = i;
        while j < n && scores[j] == scores[i] {
            let w = weights[j];
            if labels[j] == 1 {
                tp += w
            } else {
                fp += w
            }
            j += 1;
        }
        let denom = tp + fp;
        if denom > 0.0 {
            let recall = tp / total_pos;
            ap += (recall - prev_recall) * (tp / denom);
            prev_recall = recall;
        }
        i = j;
    }
    ap
}

/// Unweighted metrics on the out-of-fold logits.
pub struct PointMetrics {
    pub roc_auc: f64,
    pub pr_auc: f64,
    pub log_loss: f64,
    pub brier: f64,
    pub ece: f64,
    pub calibration: Vec<(f64, f64, usize)>,
}

/// Evaluate descending finite logits with binary labels. Invalid inputs return
/// NaN metrics and an empty calibration table; AUC also needs both classes.
pub fn point_metrics(sorted_logit: &[f32], sorted_label: &[u8]) -> PointMetrics {
    if !valid_sorted_scores(sorted_logit, sorted_label) {
        return PointMetrics {
            roc_auc: f64::NAN,
            pr_auc: f64::NAN,
            log_loss: f64::NAN,
            brier: f64::NAN,
            ece: f64::NAN,
            calibration: Vec::new(),
        };
    }
    let n = sorted_logit.len();
    let ones = vec![1.0f64; n];
    let roc_auc = weighted_auc(sorted_logit, sorted_label, &ones);
    let pr_auc = weighted_average_precision(sorted_logit, sorted_label, &ones);
    let (loss_sum, brier_sum) = (0..n)
        .into_par_iter()
        .map(|i| {
            let eta = f64::from(sorted_logit[i]);
            let y = f64::from(sorted_label[i]);
            let p = sigmoid(eta);
            (logit_loss(eta, y), (p - y) * (p - y))
        })
        .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
    // Equal-count bins walk the already-sorted order, so a bin is a
    // contiguous slice and no second sort is needed.
    let mut calibration = Vec::with_capacity(N_CALIBRATION_BINS);
    let mut ece = 0.0f64;
    for b in 0..N_CALIBRATION_BINS {
        let lo = n * b / N_CALIBRATION_BINS;
        let hi = n * (b + 1) / N_CALIBRATION_BINS;
        if hi <= lo {
            continue;
        }
        let count = hi - lo;
        let mean_pred: f64 = sorted_logit[lo..hi]
            .iter()
            .map(|&e| sigmoid(f64::from(e)))
            .sum::<f64>()
            / count as f64;
        let observed: f64 = sorted_label[lo..hi]
            .iter()
            .map(|&y| f64::from(y))
            .sum::<f64>()
            / count as f64;
        ece += (count as f64 / n as f64) * (mean_pred - observed).abs();
        calibration.push((mean_pred, observed, count));
    }
    PointMetrics {
        roc_auc,
        pr_auc,
        log_loss: loss_sum / n as f64,
        brier: brier_sum / n as f64,
        ece,
        calibration,
    }
}

/// Nearest-rank percentile of a sample sorted in ascending order.
/// Empty, nonfinite, unsorted samples or quantiles outside [0, 1] return NaN.
pub fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty()
        || !q.is_finite()
        || !(0.0..=1.0).contains(&q)
        || sorted.iter().any(|value| !value.is_finite())
        || sorted.windows(2).any(|pair| pair[0] > pair[1])
    {
        return f64::NAN;
    }
    let rank = (q * sorted.len() as f64).ceil().max(1.0) as usize;
    sorted[rank.min(sorted.len()) - 1]
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    const RIDGE: f64 = 1e-6;

    #[test]
    fn duplicate_rows_match_materialized_resampling() {
        let features = [-1.0, 0.0, 1.0, 2.0];
        let labels = [0, 1, 0, 1];
        let selected_rows = [0, 0, 1, 2, 3, 3];
        let resampled = fit_irls(
            &features,
            1,
            &[0],
            &selected_rows,
            &labels,
            &[0.0],
            &[1.0],
            RIDGE,
        )
        .expect("duplicate row indices retain sample multiplicity");
        let materialized_features: Vec<f32> = selected_rows
            .iter()
            .map(|&row| features[row as usize])
            .collect();
        let materialized_labels: Vec<u8> = selected_rows
            .iter()
            .map(|&row| labels[row as usize])
            .collect();
        let materialized = fit_irls(
            &materialized_features,
            1,
            &[0],
            &[0, 1, 2, 3, 4, 5],
            &materialized_labels,
            &[0.0],
            &[1.0],
            RIDGE,
        )
        .expect("materialized fit");
        assert_eq!(resampled.beta, materialized.beta);
        assert_eq!(resampled.deviance, materialized.deviance);
        assert!(resampled.converged);
    }

    #[test]
    fn fitting_rejects_invalid_dimensions_labels_and_standardization() {
        let features = [-1.0, 0.0, 1.0, 2.0];
        let labels = [0, 1, 0, 1];
        let rows = [0, 1, 2, 3];
        assert!(fit_irls(&features, 0, &[0], &rows, &labels, &[0.0], &[1.0], RIDGE).is_err());
        assert!(
            fit_irls(
                &features[..3],
                1,
                &[0],
                &rows,
                &labels,
                &[0.0],
                &[1.0],
                RIDGE
            )
            .is_err()
        );
        assert!(fit_irls(&features, 1, &[1], &rows, &labels, &[0.0], &[1.0], RIDGE).is_err());
        assert!(fit_irls(&features, 1, &[0], &[], &labels, &[0.0], &[1.0], RIDGE).is_err());
        assert!(fit_irls(&features, 1, &[0], &[4], &labels, &[0.0], &[1.0], RIDGE).is_err());
        assert!(
            fit_irls(
                &features,
                1,
                &[0],
                &rows,
                &[0, 2, 0, 1],
                &[0.0],
                &[1.0],
                RIDGE
            )
            .is_err()
        );
        assert!(fit_irls(&features, 1, &[0], &[0, 2], &labels, &[0.0], &[1.0], RIDGE).is_err());
        assert!(fit_irls(&features, 1, &[0], &rows, &labels, &[], &[1.0], RIDGE).is_err());
        for invalid_scale in [0.0, -1.0, f64::INFINITY, f64::NAN] {
            assert!(
                fit_irls(
                    &features,
                    1,
                    &[0],
                    &rows,
                    &labels,
                    &[0.0],
                    &[invalid_scale],
                    RIDGE
                )
                .is_err()
            );
        }
        for invalid_ridge in [-1.0, f64::INFINITY, f64::NAN] {
            assert!(
                fit_irls(
                    &features,
                    1,
                    &[0],
                    &rows,
                    &labels,
                    &[0.0],
                    &[1.0],
                    invalid_ridge
                )
                .is_err()
            );
        }
        assert!(
            fit_irls(
                &features,
                1,
                &[0],
                &rows,
                &labels,
                &[f64::NAN],
                &[1.0],
                RIDGE
            )
            .is_err()
        );
    }

    #[test]
    fn fitting_checks_only_active_features_and_rejects_nonfinite_newton_state() {
        let features = [-1.0, 0.0, 1.0, 2.0, f32::NAN];
        let labels = [0, 1, 0, 1, 2];
        assert!(
            fit_irls(
                &features,
                1,
                &[0],
                &[0, 1, 2, 3],
                &labels,
                &[0.0],
                &[1.0],
                RIDGE
            )
            .is_ok()
        );
        assert!(
            fit_irls(
                &features,
                1,
                &[0],
                &[0, 1, 2, 3, 4],
                &[0, 1, 0, 1, 0],
                &[0.0],
                &[1.0],
                RIDGE
            )
            .is_err()
        );
        // Finite standardized inputs can still overflow the Hessian products.
        assert!(
            fit_irls(
                &features,
                1,
                &[0],
                &[0, 1, 2, 3],
                &labels,
                &[0.0],
                &[1e-200],
                RIDGE
            )
            .is_err()
        );
    }

    #[test]
    fn metric_inputs_reject_nan_unsorted_scores_and_invalid_weights() {
        for scores in [[f32::NAN, 0.0], [0.0, 1.0], [f32::INFINITY, 0.0]] {
            assert!(weighted_auc(&scores, &[1, 0], &[1.0, 1.0]).is_nan());
            assert!(weighted_average_precision(&scores, &[1, 0], &[1.0, 1.0]).is_nan());
            assert!(point_metrics(&scores, &[1, 0]).log_loss.is_nan());
        }
        for weights in [[-1.0, 1.0], [f64::NAN, 1.0], [f64::MAX, f64::MAX]] {
            assert!(weighted_auc(&[1.0, 0.0], &[1, 0], &weights).is_nan());
        }
        assert!(weighted_auc(&[1.0, 0.0], &[1], &[1.0, 1.0]).is_nan());
        assert!(weighted_auc(&[1.0, 0.0], &[1, 0], &[1.0]).is_nan());
        assert!(weighted_auc(&[1.0, 0.0], &[1, 2], &[1.0, 1.0]).is_nan());
        assert!(weighted_auc(&[1.0, 0.0], &[1, 1], &[1.0, 1.0]).is_nan());
        assert!(point_metrics(&[], &[]).calibration.is_empty());
    }

    #[test]
    fn extreme_logits_keep_loss_finite_and_percentiles_require_order() {
        assert_eq!(logit_loss(1000.0, 0.0), 1000.0);
        assert_eq!(logit_loss(-1000.0, 1.0), 1000.0);
        assert_eq!(sigmoid(1000.0), 1.0);
        assert_eq!(sigmoid(-1000.0), 0.0);
        assert_eq!(percentile(&[1.0, 2.0, 3.0], 0.5), 2.0);
        assert!(percentile(&[2.0, 1.0], 0.5).is_nan());
        assert!(percentile(&[f64::NAN], 0.5).is_nan());
        assert!(percentile(&[1.0], f64::NAN).is_nan());
    }

    #[test]
    fn newton_recovers_known_logistic_coefficients() {
        let n = 20_000usize;
        let truth = [0.9f64, -1.4, 0.6];
        let intercept = -0.3f64;
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut feats = Vec::with_capacity(n * 3);
        let mut labels = Vec::with_capacity(n);
        for _ in 0..n {
            let mut eta = intercept;
            for t in truth.iter() {
                // Box-Muller from two uniforms keeps the test free of rand_distr.
                let u1: f64 = rng.random::<f64>().max(1e-12);
                let u2: f64 = rng.random::<f64>();
                let x: f64 = (-2.0f64 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                feats.push(x as f32);
                eta += t * x;
            }
            let p = sigmoid(eta);
            labels.push(u8::from(rng.random::<f64>() < p));
        }
        let rows: Vec<u32> = (0..n as u32).collect();
        let fit = fit_irls(
            &feats,
            3,
            &[0, 1, 2],
            &rows,
            &labels,
            &[0.0, 0.0, 0.0],
            &[1.0, 1.0, 1.0],
            RIDGE,
        )
        .expect("fit converges");
        assert!((fit.beta[0] - intercept).abs() < 0.08, "{:?}", fit.beta);
        for (k, t) in truth.iter().enumerate() {
            assert!((fit.beta[k + 1] - t).abs() < 0.08, "{:?}", fit.beta);
        }
    }

    #[test]
    fn rank_auc_handles_ties() {
        // Descending order: 3.0 (pos), 2.0 (pos), 2.0 (neg), 1.0 (neg).
        // Pairs: (3,2neg)=1, (3,1)=1, (2pos,2neg)=0.5, (2pos,1)=1 -> 3.5/4.
        let scores = [3.0f32, 2.0, 2.0, 1.0];
        let labels = [1u8, 1, 0, 0];
        let w = [1.0f64; 4];
        assert!((weighted_auc(&scores, &labels, &w) - 0.875).abs() < 1e-12);
    }

    #[test]
    fn average_precision_hand_example() {
        // Descending: pos, neg, pos, neg.  Distinct thresholds give
        // precision 1/1 at recall 1/2 and 2/3 at recall 1.
        let scores = [4.0f32, 3.0, 2.0, 1.0];
        let labels = [1u8, 0, 1, 0];
        let w = [1.0f64; 4];
        let ap = weighted_average_precision(&scores, &labels, &w);
        let expect = 0.5 * 1.0 + 0.5 * (2.0 / 3.0);
        assert!((ap - expect).abs() < 1e-12, "{ap}");
    }

    #[test]
    fn unit_weights_reproduce_unweighted_auc() {
        let scores = [5.0f32, 4.0, 4.0, 3.0, 2.0, 1.0, 1.0];
        let labels = [1u8, 0, 1, 1, 0, 0, 1];
        let ones = vec![1.0f64; scores.len()];
        let twos = vec![2.0f64; scores.len()];
        let a = weighted_auc(&scores, &labels, &ones);
        // A uniform reweighting cancels in the normalization, so the
        // weighted statistic must equal the unweighted one.
        let b = weighted_auc(&scores, &labels, &twos);
        assert!((a - b).abs() < 1e-12);
        assert!(a > 0.0 && a < 1.0);
    }
}

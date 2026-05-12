//! L2-regularized weighted logistic regression with standard scaling.
//!
//! Two record types plus four pipeline helpers:
//!   * `ScaledFeatureSet`     -- column means + standard deviations from `fit_scaler`
//!   * `LogisticModel`        -- weight vector + bias from `train_logistic_model`
//!   * `fit_scaler` / `apply_scaler`   -- per-column standardize
//!   * `train_logistic_model` / `predict_scores`  -- training + inference
//!
//! The trainer carries a positive-class up-weight (negatives/positives)
//! so the loss does not collapse on heavily imbalanced labels.
//!
//! All items are `pub(super)`; depends only on `super::stats::mean`
//! and `super::metrics::sigmoid`.

use super::metrics::sigmoid;
use super::stats::mean;

#[derive(Debug, Clone)]
pub(super) struct ScaledFeatureSet {
    pub(super) means: Vec<f64>,
    pub(super) scales: Vec<f64>,
}

#[derive(Debug, Clone)]
pub(super) struct LogisticModel {
    pub(super) weights: Vec<f64>,
    pub(super) bias: f64,
}

pub(super) fn fit_scaler(rows: &[Vec<f64>]) -> ScaledFeatureSet {
    if rows.is_empty() {
        return ScaledFeatureSet {
            means: Vec::new(),
            scales: Vec::new(),
        };
    }
    let dim = rows[0].len();
    let mut means = vec![0.0; dim];
    let mut scales = vec![1.0; dim];
    for idx in 0..dim {
        let column = rows.iter().map(|row| row[idx]).collect::<Vec<_>>();
        means[idx] = mean(&column);
        let variance = column
            .iter()
            .map(|value| {
                let delta = *value - means[idx];
                delta * delta
            })
            .sum::<f64>()
            / column.len().max(1) as f64;
        let scale = variance.sqrt();
        scales[idx] = if scale.is_finite() && scale > 0.0 {
            scale
        } else {
            1.0
        };
    }
    ScaledFeatureSet { means, scales }
}

pub(super) fn apply_scaler(scaler: &ScaledFeatureSet, rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
    rows.iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(idx, value)| (value - scaler.means[idx]) / scaler.scales[idx])
                .collect::<Vec<_>>()
        })
        .collect()
}

pub(super) fn train_logistic_model(
    rows: &[Vec<f64>],
    labels: &[f64],
    learning_rate: f64,
    epochs: usize,
    lambda: f64,
) -> LogisticModel {
    let dim = rows.first().map(Vec::len).unwrap_or(0);
    let mut weights = vec![0.0; dim];
    let mut bias = 0.0;
    let positives = labels.iter().copied().sum::<f64>().max(1.0);
    let negatives = (labels.len() as f64 - positives).max(1.0);
    let positive_weight = negatives / positives;
    for _ in 0..epochs {
        let mut grad_w = vec![0.0; dim];
        let mut grad_b = 0.0;
        for (row, label) in rows.iter().zip(labels.iter().copied()) {
            let linear = bias
                + row
                    .iter()
                    .zip(weights.iter())
                    .map(|(value, weight)| value * weight)
                    .sum::<f64>();
            let prediction = sigmoid(linear);
            let error = prediction - label;
            let weight = if label > 0.5 { positive_weight } else { 1.0 };
            for idx in 0..dim {
                grad_w[idx] += error * row[idx] * weight;
            }
            grad_b += error * weight;
        }
        let denom = rows.len().max(1) as f64;
        for idx in 0..dim {
            grad_w[idx] = grad_w[idx] / denom + lambda * weights[idx];
            weights[idx] -= learning_rate * grad_w[idx];
        }
        bias -= learning_rate * grad_b / denom;
    }
    LogisticModel { weights, bias }
}

pub(super) fn predict_scores(model: &LogisticModel, rows: &[Vec<f64>]) -> Vec<f64> {
    rows.iter()
        .map(|row| {
            let linear = model.bias
                + row
                    .iter()
                    .zip(model.weights.iter())
                    .map(|(value, weight)| value * weight)
                    .sum::<f64>();
            sigmoid(linear)
        })
        .collect()
}

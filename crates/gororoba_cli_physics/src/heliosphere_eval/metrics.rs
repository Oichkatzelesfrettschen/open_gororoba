//! Classifier-metric helpers: sigmoid, threshold-based confusion
//! matrix (precision/recall/F1/false-alert rate), AUPRC, AUROC, and
//! threshold-search.
//!
//! All functions are pure -- no parent-module type dependencies. Uses
//! `super::stats::ratio_usize` for safe division.

use super::stats::ratio_usize;

pub(super) fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        let z = (-value).exp();
        1.0 / (1.0 + z)
    } else {
        let z = value.exp();
        z / (1.0 + z)
    }
}

/// Compute precision, recall, F1, predicted-positive count, and
/// false-alert rate for a given score-threshold cutoff.
///
/// Returns `(precision, recall, f1, tp+fp, false_alert_rate)`.
pub(super) fn threshold_metrics(
    scores: &[f64],
    labels: &[bool],
    threshold: f64,
) -> (f64, f64, f64, usize, f64) {
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut fn_ = 0usize;
    let mut tn = 0usize;
    for (score, label) in scores.iter().zip(labels.iter().copied()) {
        let predicted = score.is_finite() && *score >= threshold;
        match (predicted, label) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fn_ += 1,
            (false, false) => tn += 1,
        }
    }
    let precision = ratio_usize(tp, tp + fp);
    let recall = ratio_usize(tp, tp + fn_);
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    let false_alert_rate = ratio_usize(fp, fp + tn);
    (precision, recall, f1, tp + fp, false_alert_rate)
}

/// Sweep candidate thresholds (deduped and stepped to <=64 points)
/// and return the one maximizing `F1 + 0.1*precision + 0.1*recall`.
pub(super) fn best_threshold(scores: &[f64], labels: &[bool]) -> f64 {
    if scores.is_empty() {
        return 0.5;
    }
    let mut candidates = scores
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    candidates.sort_by(|a, b| a.total_cmp(b));
    candidates.dedup_by(|a, b| (*a - *b).abs() < 1.0e-9);
    if candidates.len() > 64 {
        candidates = candidates
            .into_iter()
            .step_by((scores.len() / 64).max(1))
            .collect();
    }
    let mut best = (0.5_f64, f64::NEG_INFINITY);
    for threshold in candidates {
        let (precision, recall, f1, _, _) = threshold_metrics(scores, labels, threshold);
        let score = f1 + 0.1 * precision + 0.1 * recall;
        if score.total_cmp(&best.1).is_gt() {
            best = (threshold, score);
        }
    }
    best.0
}

/// Area under the precision-recall curve.
pub(super) fn auprc(scores: &[f64], labels: &[bool]) -> f64 {
    let mut ranked = scores
        .iter()
        .copied()
        .zip(labels.iter().copied())
        .filter(|(score, _)| score.is_finite())
        .collect::<Vec<_>>();
    ranked.sort_by(|(a, _), (b, _)| b.total_cmp(a));
    let positives = labels.iter().filter(|value| **value).count().max(1) as f64;
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut prev_recall = 0.0;
    let mut area = 0.0;
    for (_, label) in ranked {
        if label {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        let precision = tp / f64::max(tp + fp, 1.0);
        let recall = tp / positives;
        area += precision * (recall - prev_recall);
        prev_recall = recall;
    }
    area.clamp(0.0, 1.0)
}

/// Area under the ROC curve via rank-sum identity. Returns NaN when
/// there are no positives or no negatives.
pub(super) fn auroc(scores: &[f64], labels: &[bool]) -> f64 {
    let mut ranked = scores
        .iter()
        .copied()
        .zip(labels.iter().copied())
        .filter(|(score, _)| score.is_finite())
        .collect::<Vec<_>>();
    ranked.sort_by(|(a, _), (b, _)| a.total_cmp(b));
    let positives = labels.iter().filter(|value| **value).count() as f64;
    let negatives = labels.iter().filter(|value| !**value).count() as f64;
    if positives == 0.0 || negatives == 0.0 {
        return f64::NAN;
    }
    let mut rank_sum = 0.0;
    for (idx, (_, label)) in ranked.iter().enumerate() {
        if *label {
            rank_sum += idx as f64 + 1.0;
        }
    }
    ((rank_sum - positives * (positives + 1.0) / 2.0) / (positives * negatives)).clamp(0.0, 1.0)
}

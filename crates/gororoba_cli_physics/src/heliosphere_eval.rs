//! Shared heliosphere evaluation helpers for predictive, invariance, and
//! sparsification experiments.

use anyhow::{Context, Result, bail};
use chrono::{DateTime, Duration, Utc};
use csv::ReaderBuilder;
use data_core::{
    HELIOSPHERE_INVARIANT_DIM, HeliosphereEventWindow, HeliosphereFeatureRow,
    HeliosphereInvariantSample, HeliosphereTransformMode, compute_invariant_samples,
    fetch_donki_event_labels, heliosphere_row_datetime, labels_to_prediction_windows,
    transform_feature_rows_with_stats,
};
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

const DESCRIPTOR_DIM: usize = 4;

/// Stable row key used to join raw, transformed, and invariant views.
pub type RowKey = (String, String, String, u16, u16, u8);

/// One invariant sample augmented with official-label and descriptor context.
#[derive(Debug, Clone, Serialize)]
pub struct LabeledInvariantSample {
    pub key: RowKey,
    pub window_name: String,
    pub mission: String,
    pub product: String,
    pub timestamp_utc: String,
    pub label_positive: bool,
    pub scalar_event_score: f64,
    pub weighted_channels: [f64; HELIOSPHERE_INVARIANT_DIM],
    pub descriptor_channels: [f64; DESCRIPTOR_DIM],
}

/// Scalar evaluation metrics for one predictive lane.
#[derive(Debug, Clone, Serialize)]
pub struct BinaryMetrics {
    pub name: String,
    pub threshold: f64,
    pub positive_rows: usize,
    pub negative_rows: usize,
    pub predicted_positive_rows: usize,
    pub auprc: f64,
    pub auroc: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub false_alert_rate: f64,
    pub median_lead_time_hours: Option<f64>,
}

/// Chronological train/validation/test split sizes for one mission.
#[derive(Debug, Clone, Serialize)]
pub struct MissionSplitSummary {
    pub mission: String,
    pub train_rows: usize,
    pub validation_rows: usize,
    pub test_rows: usize,
}

/// Cross-mission descriptor stability summary.
#[derive(Debug, Clone, Serialize)]
pub struct MissionInvarianceSummary {
    pub mission: String,
    pub positive_rows: usize,
    pub negative_rows: usize,
    pub positive_mean_weighted_norm: f64,
    pub negative_mean_weighted_norm: f64,
    pub positive_descriptor_mean: f64,
    pub leave_one_mission_out_cosine: f64,
}

/// Sparse-preservation metrics comparing two event masks.
#[derive(Debug, Clone, Serialize)]
pub struct SparseMaskSummary {
    pub name: String,
    pub active_rows: usize,
    pub active_fraction: f64,
    pub event_label_recall: f64,
    pub event_label_precision: f64,
    pub density_mean: f64,
    pub speed_mean: f64,
    pub temperature_mean: f64,
    pub bmag_mean: f64,
}

#[derive(Debug, Clone)]
struct ScaledFeatureSet {
    means: Vec<f64>,
    scales: Vec<f64>,
}

/// Load heliosphere feature rows from a CSV cube.
pub fn load_heliosphere_rows(path: &Path) -> Result<Vec<HeliosphereFeatureRow>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .with_context(|| format!("open {}", path.display()))?;
    let mut rows = Vec::new();
    for row in reader.deserialize() {
        rows.push(row.with_context(|| format!("deserialize {}", path.display()))?);
    }
    Ok(rows)
}

/// Stable join key for one heliosphere row.
pub fn row_key(row: &HeliosphereFeatureRow) -> RowKey {
    (
        row.window_name.clone(),
        row.mission.clone(),
        row.product.clone(),
        row.year,
        row.doy,
        row.hour,
    )
}

fn sample_key(sample: &HeliosphereInvariantSample) -> RowKey {
    (
        sample.window_name.clone(),
        sample.mission.clone(),
        sample.product.clone(),
        sample.year,
        sample.doy,
        sample.hour,
    )
}

/// Build official-label-joined invariant samples and per-mission split summaries.
pub fn build_labeled_samples(
    rows: &[HeliosphereFeatureRow],
    cache_root: &Path,
    horizon_hours: i64,
) -> Result<(Vec<LabeledInvariantSample>, Vec<MissionSplitSummary>)> {
    let start_date = rows
        .iter()
        .filter_map(heliosphere_row_datetime)
        .map(|value| value.date_naive())
        .min()
        .ok_or_else(|| anyhow::anyhow!("cube contains no timestamped rows"))?;
    let end_date = rows
        .iter()
        .filter_map(heliosphere_row_datetime)
        .map(|value| value.date_naive())
        .max()
        .ok_or_else(|| anyhow::anyhow!("cube contains no timestamped rows"))?;
    let labels = fetch_donki_event_labels(
        start_date - Duration::days(2),
        end_date + Duration::days(2),
        cache_root,
    )?;
    let mission_set = rows
        .iter()
        .map(|row| row.mission.clone())
        .collect::<BTreeSet<_>>();
    let mut mission_windows = BTreeMap::new();
    for mission in &mission_set {
        mission_windows.insert(
            mission.clone(),
            labels_to_prediction_windows(&labels, mission, horizon_hours),
        );
    }

    let invariants = compute_invariant_samples(rows);
    let mut grouped: BTreeMap<(String, String, String), Vec<HeliosphereInvariantSample>> =
        BTreeMap::new();
    for sample in invariants {
        grouped
            .entry((
                sample.window_name.clone(),
                sample.mission.clone(),
                sample.product.clone(),
            ))
            .or_default()
            .push(sample);
    }

    let mut output = Vec::new();
    for ((_window, mission, _product), mut group) in grouped {
        group.sort_by(|a, b| {
            (
                a.year,
                a.doy,
                a.hour,
                a.timestamp_utc.as_str(),
                a.product.as_str(),
            )
                .cmp(&(
                    b.year,
                    b.doy,
                    b.hour,
                    b.timestamp_utc.as_str(),
                    b.product.as_str(),
                ))
        });
        let mission_window_refs = mission_windows
            .get(&mission)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        for idx in 0..group.len() {
            let sample = &group[idx];
            let Ok(timestamp) = parse_timestamp(&sample.timestamp_utc) else {
                continue;
            };
            let descriptor_channels = descriptor_channels(&group, idx);
            output.push(LabeledInvariantSample {
                key: sample_key(sample),
                window_name: sample.window_name.clone(),
                mission: sample.mission.clone(),
                product: sample.product.clone(),
                timestamp_utc: sample.timestamp_utc.clone(),
                label_positive: mission_window_refs
                    .iter()
                    .any(|window| contains_time(window, timestamp)),
                scalar_event_score: sample.inherited_event_score.unwrap_or(0.0),
                weighted_channels: sample.weighted_channels,
                descriptor_channels,
            });
        }
    }
    output.sort_by(|a, b| {
        (
            a.mission.as_str(),
            a.timestamp_utc.as_str(),
            a.product.as_str(),
            a.window_name.as_str(),
        )
            .cmp(&(
                b.mission.as_str(),
                b.timestamp_utc.as_str(),
                b.product.as_str(),
                b.window_name.as_str(),
            ))
    });
    let split_summary = mission_splits(&output);
    Ok((output, split_summary))
}

/// Evaluate scalar, invariant-only, and invariant-plus-descriptor predictors.
pub fn evaluate_predictive_models(samples: &[LabeledInvariantSample]) -> Result<Vec<BinaryMetrics>> {
    if samples.is_empty() {
        bail!("no labeled invariant samples available");
    }
    let splits = split_samples(samples);
    if splits.train.is_empty() || splits.validation.is_empty() || splits.test.is_empty() {
        bail!("need non-empty train/validation/test splits");
    }

    let scalar_validation_scores = splits
        .validation
        .iter()
        .map(|sample| sample.scalar_event_score)
        .collect::<Vec<_>>();
    let scalar_threshold = best_threshold(
        &scalar_validation_scores,
        &splits
            .validation
            .iter()
            .map(|sample| sample.label_positive)
            .collect::<Vec<_>>(),
    );

    let best_single = best_single_invariant_threshold(&splits.validation);

    let invariant_train = feature_matrix(&splits.train, FeatureMode::Invariants);
    let invariant_validation = feature_matrix(&splits.validation, FeatureMode::Invariants);
    let invariant_test = feature_matrix(&splits.test, FeatureMode::Invariants);
    let invariant_scaler = fit_scaler(&invariant_train);
    let invariant_train_scaled = apply_scaler(&invariant_scaler, &invariant_train);
    let invariant_validation_scaled = apply_scaler(&invariant_scaler, &invariant_validation);
    let invariant_test_scaled = apply_scaler(&invariant_scaler, &invariant_test);
    let invariant_model = train_logistic_model(
        &invariant_train_scaled,
        &binary_labels(&splits.train),
        0.02,
        250,
        1e-3,
    );
    let invariant_validation_scores = predict_scores(&invariant_model, &invariant_validation_scaled);
    let invariant_threshold = best_threshold(
        &invariant_validation_scores,
        &splits
            .validation
            .iter()
            .map(|sample| sample.label_positive)
            .collect::<Vec<_>>(),
    );

    let hybrid_train = feature_matrix(&splits.train, FeatureMode::InvariantsAndDescriptors);
    let hybrid_validation =
        feature_matrix(&splits.validation, FeatureMode::InvariantsAndDescriptors);
    let hybrid_test = feature_matrix(&splits.test, FeatureMode::InvariantsAndDescriptors);
    let hybrid_scaler = fit_scaler(&hybrid_train);
    let hybrid_train_scaled = apply_scaler(&hybrid_scaler, &hybrid_train);
    let hybrid_validation_scaled = apply_scaler(&hybrid_scaler, &hybrid_validation);
    let hybrid_test_scaled = apply_scaler(&hybrid_scaler, &hybrid_test);
    let hybrid_model = train_logistic_model(
        &hybrid_train_scaled,
        &binary_labels(&splits.train),
        0.02,
        300,
        1e-3,
    );
    let hybrid_validation_scores = predict_scores(&hybrid_model, &hybrid_validation_scaled);
    let hybrid_threshold = best_threshold(
        &hybrid_validation_scores,
        &splits
            .validation
            .iter()
            .map(|sample| sample.label_positive)
            .collect::<Vec<_>>(),
    );

    Ok(vec![
        evaluate_scores(
            "scalar_event_score",
            &splits.test,
            &splits
                .test
                .iter()
                .map(|sample| sample.scalar_event_score)
                .collect::<Vec<_>>(),
            scalar_threshold,
        ),
        evaluate_scores(
            "best_single_invariant_threshold",
            &splits.test,
            &splits
                .test
                .iter()
                .map(|sample| sample.weighted_channels[best_single.index])
                .collect::<Vec<_>>(),
            best_single.threshold,
        ),
        evaluate_scores(
            "invariant_logistic",
            &splits.test,
            &predict_scores(&invariant_model, &invariant_test_scaled),
            invariant_threshold,
        ),
        evaluate_scores(
            "invariant_plus_algebra_logistic",
            &splits.test,
            &predict_scores(&hybrid_model, &hybrid_test_scaled),
            hybrid_threshold,
        ),
    ])
}

/// Summarize leave-one-mission-out descriptor stability.
pub fn summarize_cross_mission_invariance(
    samples: &[LabeledInvariantSample],
) -> Vec<MissionInvarianceSummary> {
    let mut grouped: BTreeMap<String, Vec<&LabeledInvariantSample>> = BTreeMap::new();
    for sample in samples {
        grouped.entry(sample.mission.clone()).or_default().push(sample);
    }

    grouped
        .iter()
        .map(|(mission, group)| {
            let positive = group
                .iter()
                .copied()
                .filter(|sample| sample.label_positive)
                .collect::<Vec<_>>();
            let negative = group
                .iter()
                .copied()
                .filter(|sample| !sample.label_positive)
                .collect::<Vec<_>>();
            let mission_vector = mean_feature_vector(&positive);
            let rest_vector = mean_feature_vector(
                &grouped
                    .iter()
                    .filter(|(other, _)| *other != mission)
                    .flat_map(|(_, rows)| rows.iter().copied())
                    .filter(|sample| sample.label_positive)
                    .collect::<Vec<_>>(),
            );
            MissionInvarianceSummary {
                mission: mission.clone(),
                positive_rows: positive.len(),
                negative_rows: negative.len(),
                positive_mean_weighted_norm: mean(
                    &positive
                        .iter()
                        .map(|sample| l2_norm_sq(&sample.weighted_channels).sqrt())
                        .collect::<Vec<_>>(),
                ),
                negative_mean_weighted_norm: mean(
                    &negative
                        .iter()
                        .map(|sample| l2_norm_sq(&sample.weighted_channels).sqrt())
                        .collect::<Vec<_>>(),
                ),
                positive_descriptor_mean: mean(
                    &positive
                        .iter()
                        .map(|sample| sample.descriptor_channels[0])
                        .collect::<Vec<_>>(),
                ),
                leave_one_mission_out_cosine: cosine_similarity(&mission_vector, &rest_vector),
            }
        })
        .collect()
}

/// Compute sparse-preservation summaries for baseline and algebra-derived masks.
pub fn summarize_sparse_masks(
    raw_rows: &[HeliosphereFeatureRow],
    cache_root: &Path,
    horizon_hours: i64,
) -> Result<(SparseMaskSummary, SparseMaskSummary)> {
    let transformed = transform_feature_rows_with_stats(
        raw_rows,
        HeliosphereTransformMode::RobustDifferencedCentered,
    );
    let baseline_index = transformed
        .rows
        .iter()
        .map(|row| (row_key(row), row.event_active()))
        .collect::<BTreeMap<_, _>>();
    let baseline_rows = raw_rows
        .iter()
        .filter(|row| *baseline_index.get(&row_key(row)).unwrap_or(&false))
        .collect::<Vec<_>>();

    let (samples, _) = build_labeled_samples(raw_rows, cache_root, horizon_hours)?;
    let algebra_index = algebra_event_mask(&samples)
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    let algebra_rows = raw_rows
        .iter()
        .filter(|row| *algebra_index.get(&row_key(row)).unwrap_or(&false))
        .collect::<Vec<_>>();

    let labels = label_index(&samples);
    Ok((
        sparse_summary("robust_event_mask", &baseline_rows, raw_rows.len(), &labels),
        sparse_summary("algebra_adaptive_mask", &algebra_rows, raw_rows.len(), &labels),
    ))
}

/// Return the algebra-derived event mask for each invariant sample.
pub fn algebra_event_mask(samples: &[LabeledInvariantSample]) -> Vec<(RowKey, bool)> {
    let mut grouped: BTreeMap<(String, String, String), Vec<&LabeledInvariantSample>> =
        BTreeMap::new();
    for sample in samples {
        grouped
            .entry((
                sample.window_name.clone(),
                sample.mission.clone(),
                sample.product.clone(),
            ))
            .or_default()
            .push(sample);
    }

    let mut result = Vec::new();
    for mut group in grouped.into_values() {
        group.sort_by(|a, b| {
            (
                a.timestamp_utc.as_str(),
                a.mission.as_str(),
                a.product.as_str(),
            )
                .cmp(&(b.timestamp_utc.as_str(), b.mission.as_str(), b.product.as_str()))
        });
        if group.len() < 4 {
            for sample in group {
                result.push((sample.key.clone(), true));
            }
            continue;
        }
        let raw_scores = group
            .iter()
            .map(|sample| {
                let descriptor_rms = (sample
                    .descriptor_channels
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    / DESCRIPTOR_DIM as f64)
                    .sqrt();
                let invariant_rms = (sample
                    .weighted_channels
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    / HELIOSPHERE_INVARIANT_DIM as f64)
                    .sqrt();
                descriptor_rms + invariant_rms
            })
            .collect::<Vec<_>>();
        let smoothed = median_filter_3(&raw_scores);
        let baseline = finite_median(&smoothed);
        let spread = (1.4826 * finite_mad(&smoothed, baseline)).max(1.0e-6);
        let on = baseline + 2.5 * spread;
        let off = baseline + 1.25 * spread;
        let flags = hysteresis_mask(&smoothed, on, off);
        let dilated = dilate_mask(&flags, 1);
        let merged = merge_small_gaps(&dilated, 1);
        for (sample, active) in group.into_iter().zip(merged) {
            result.push((sample.key.clone(), active));
        }
    }
    result
}

#[derive(Debug, Clone, Copy)]
enum FeatureMode {
    Invariants,
    InvariantsAndDescriptors,
}

#[derive(Debug, Clone)]
struct LogisticModel {
    weights: Vec<f64>,
    bias: f64,
}

#[derive(Debug, Clone)]
struct ThresholdedInvariant {
    index: usize,
    threshold: f64,
}

#[derive(Debug, Clone)]
struct SampleSplits<'a> {
    train: Vec<&'a LabeledInvariantSample>,
    validation: Vec<&'a LabeledInvariantSample>,
    test: Vec<&'a LabeledInvariantSample>,
}

fn parse_timestamp(value: &str) -> Result<DateTime<Utc>> {
    Ok(DateTime::parse_from_rfc3339(value)
        .with_context(|| format!("parse timestamp {value}"))?
        .with_timezone(&Utc))
}

fn contains_time(window: &HeliosphereEventWindow, timestamp: DateTime<Utc>) -> bool {
    let start = DateTime::parse_from_rfc3339(&window.window_start_utc)
        .map(|value| value.with_timezone(&Utc));
    let end = DateTime::parse_from_rfc3339(&window.window_end_utc)
        .map(|value| value.with_timezone(&Utc));
    match (start, end) {
        (Ok(start), Ok(end)) => timestamp >= start && timestamp <= end,
        _ => false,
    }
}

fn descriptor_channels(
    group: &[HeliosphereInvariantSample],
    idx: usize,
) -> [f64; DESCRIPTOR_DIM] {
    let current = &group[idx].weighted_channels;
    let prev = idx.checked_sub(1).map(|index| &group[index].weighted_channels);
    let prev2 = idx.checked_sub(2).map(|index| &group[index].weighted_channels);
    let norm_sq = l2_norm_sq(current);
    let delta_norm = prev
        .map(|value| (norm_sq - l2_norm_sq(value)).abs())
        .unwrap_or(0.0);
    let associator = match (prev2, prev) {
        (Some(a), Some(b)) => {
            let a_cd = to_cd16(a);
            let b_cd = to_cd16(b);
            let c_cd = to_cd16(current);
            cd_kernel::cd_associator_norm(&a_cd, &b_cd, &c_cd)
        }
        _ => 0.0,
    };
    let mean_abs = current.iter().map(|value| value.abs()).sum::<f64>() / current.len() as f64;
    [norm_sq, delta_norm, associator, mean_abs]
}

fn mission_splits(samples: &[LabeledInvariantSample]) -> Vec<MissionSplitSummary> {
    let mut grouped: BTreeMap<String, Vec<&LabeledInvariantSample>> = BTreeMap::new();
    for sample in samples {
        grouped.entry(sample.mission.clone()).or_default().push(sample);
    }
    grouped
        .into_iter()
        .map(|(mission, mut group)| {
            group.sort_by_key(|sample| sample.timestamp_utc.clone());
            let n = group.len();
            let train_end = ((n as f64) * 0.70).round() as usize;
            let val_end = ((n as f64) * 0.85).round() as usize;
            MissionSplitSummary {
                mission,
                train_rows: train_end.min(n),
                validation_rows: val_end.saturating_sub(train_end).min(n.saturating_sub(train_end)),
                test_rows: n.saturating_sub(val_end),
            }
        })
        .collect()
}

fn split_samples(samples: &[LabeledInvariantSample]) -> SampleSplits<'_> {
    let mut grouped: BTreeMap<String, Vec<&LabeledInvariantSample>> = BTreeMap::new();
    for sample in samples {
        grouped.entry(sample.mission.clone()).or_default().push(sample);
    }
    let mut train = Vec::new();
    let mut validation = Vec::new();
    let mut test = Vec::new();
    for mut group in grouped.into_values() {
        group.sort_by_key(|sample| sample.timestamp_utc.clone());
        let n = group.len();
        let train_end = ((n as f64) * 0.70).round() as usize;
        let val_end = ((n as f64) * 0.85).round() as usize;
        train.extend(group.iter().take(train_end).copied());
        validation.extend(
            group.iter()
                .skip(train_end)
                .take(val_end.saturating_sub(train_end))
                .copied(),
        );
        test.extend(group.iter().skip(val_end).copied());
    }
    SampleSplits {
        train,
        validation,
        test,
    }
}

fn feature_matrix(samples: &[&LabeledInvariantSample], mode: FeatureMode) -> Vec<Vec<f64>> {
    samples
        .iter()
        .map(|sample| match mode {
            FeatureMode::Invariants => sample.weighted_channels.to_vec(),
            FeatureMode::InvariantsAndDescriptors => sample
                .weighted_channels
                .iter()
                .copied()
                .chain(sample.descriptor_channels.iter().copied())
                .collect::<Vec<_>>(),
        })
        .collect()
}

fn binary_labels(samples: &[&LabeledInvariantSample]) -> Vec<f64> {
    samples
        .iter()
        .map(|sample| if sample.label_positive { 1.0 } else { 0.0 })
        .collect()
}

fn fit_scaler(rows: &[Vec<f64>]) -> ScaledFeatureSet {
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

fn apply_scaler(scaler: &ScaledFeatureSet, rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
    rows.iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(idx, value)| (value - scaler.means[idx]) / scaler.scales[idx])
                .collect::<Vec<_>>()
        })
        .collect()
}

fn train_logistic_model(
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

fn predict_scores(model: &LogisticModel, rows: &[Vec<f64>]) -> Vec<f64> {
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

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        let z = (-value).exp();
        1.0 / (1.0 + z)
    } else {
        let z = value.exp();
        z / (1.0 + z)
    }
}

fn best_threshold(scores: &[f64], labels: &[bool]) -> f64 {
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

fn best_single_invariant_threshold(samples: &[&LabeledInvariantSample]) -> ThresholdedInvariant {
    let labels = samples
        .iter()
        .map(|sample| sample.label_positive)
        .collect::<Vec<_>>();
    let mut best = ThresholdedInvariant {
        index: 0,
        threshold: 0.0,
    };
    let mut best_score = f64::NEG_INFINITY;
    for idx in 0..HELIOSPHERE_INVARIANT_DIM {
        let scores = samples
            .iter()
            .map(|sample| sample.weighted_channels[idx])
            .collect::<Vec<_>>();
        let threshold = best_threshold(&scores, &labels);
        let (_, _, f1, _, _) = threshold_metrics(&scores, &labels, threshold);
        if f1.total_cmp(&best_score).is_gt() {
            best = ThresholdedInvariant {
                index: idx,
                threshold,
            };
            best_score = f1;
        }
    }
    best
}

fn evaluate_scores(
    name: &str,
    samples: &[&LabeledInvariantSample],
    scores: &[f64],
    threshold: f64,
) -> BinaryMetrics {
    let labels = samples
        .iter()
        .map(|sample| sample.label_positive)
        .collect::<Vec<_>>();
    let (precision, recall, f1, predicted_positive_rows, false_alert_rate) =
        threshold_metrics(scores, &labels, threshold);
    BinaryMetrics {
        name: name.to_string(),
        threshold,
        positive_rows: labels.iter().filter(|value| **value).count(),
        negative_rows: labels.iter().filter(|value| !**value).count(),
        predicted_positive_rows,
        auprc: auprc(scores, &labels),
        auroc: auroc(scores, &labels),
        precision,
        recall,
        f1,
        false_alert_rate,
        median_lead_time_hours: median_lead_time_hours(samples, scores, threshold),
    }
}

fn threshold_metrics(
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

fn auprc(scores: &[f64], labels: &[bool]) -> f64 {
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

fn auroc(scores: &[f64], labels: &[bool]) -> f64 {
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

fn median_lead_time_hours(
    samples: &[&LabeledInvariantSample],
    scores: &[f64],
    threshold: f64,
) -> Option<f64> {
    let mut grouped: BTreeMap<&str, Vec<(&LabeledInvariantSample, f64)>> = BTreeMap::new();
    for (sample, score) in samples.iter().copied().zip(scores.iter().copied()) {
        grouped.entry(sample.mission.as_str()).or_default().push((sample, score));
    }
    let mut leads = Vec::new();
    for rows in grouped.values_mut() {
        rows.sort_by_key(|(sample, _)| sample.timestamp_utc.clone());
        let positive_indices = rows
            .iter()
            .enumerate()
            .filter(|(_, (sample, _))| sample.label_positive)
            .map(|(idx, _)| idx)
            .collect::<Vec<_>>();
        for positive_idx in positive_indices {
            let event_time = parse_timestamp(&rows[positive_idx].0.timestamp_utc).ok()?;
            let mut earliest_prediction = None;
            for (sample, score) in rows[..=positive_idx].iter().rev() {
                if *score >= threshold {
                    earliest_prediction = parse_timestamp(&sample.timestamp_utc).ok();
                } else if earliest_prediction.is_some() {
                    break;
                }
            }
            if let Some(start) = earliest_prediction {
                let hours = (event_time - start).num_minutes() as f64 / 60.0;
                if hours.is_finite() && hours >= 0.0 {
                    leads.push(hours);
                }
            }
        }
    }
    finite_median_opt(&leads)
}

fn label_index(samples: &[LabeledInvariantSample]) -> BTreeMap<RowKey, bool> {
    samples
        .iter()
        .map(|sample| (sample.key.clone(), sample.label_positive))
        .collect()
}

fn sparse_summary(
    name: &str,
    rows: &[&HeliosphereFeatureRow],
    total_rows: usize,
    label_index: &BTreeMap<RowKey, bool>,
) -> SparseMaskSummary {
    let active_rows = rows.len();
    let labeled_active = rows
        .iter()
        .filter(|row| *label_index.get(&row_key(row)).unwrap_or(&false))
        .count();
    let total_labeled = label_index.values().filter(|value| **value).count();
    SparseMaskSummary {
        name: name.to_string(),
        active_rows,
        active_fraction: ratio_usize(active_rows, total_rows),
        event_label_recall: ratio_usize(labeled_active, total_labeled),
        event_label_precision: ratio_usize(labeled_active, active_rows),
        density_mean: mean(
            &rows.iter()
                .map(|row| row.density_cm3)
                .filter(|value| value.is_finite())
                .collect::<Vec<_>>(),
        ),
        speed_mean: mean(
            &rows.iter()
                .map(|row| row.speed_kms)
                .filter(|value| value.is_finite())
                .collect::<Vec<_>>(),
        ),
        temperature_mean: mean(
            &rows.iter()
                .map(|row| row.temperature_k)
                .filter(|value| value.is_finite())
                .collect::<Vec<_>>(),
        ),
        bmag_mean: mean(
            &rows.iter()
                .map(|row| row.b_mag)
                .filter(|value| value.is_finite())
                .collect::<Vec<_>>(),
        ),
    }
}

fn mean_feature_vector(samples: &[&LabeledInvariantSample]) -> Vec<f64> {
    if samples.is_empty() {
        return vec![0.0; HELIOSPHERE_INVARIANT_DIM + DESCRIPTOR_DIM];
    }
    let mut out = vec![0.0; HELIOSPHERE_INVARIANT_DIM + DESCRIPTOR_DIM];
    for sample in samples {
        for (idx, value) in out.iter_mut().enumerate().take(HELIOSPHERE_INVARIANT_DIM) {
            *value += sample.weighted_channels[idx];
        }
        for idx in 0..DESCRIPTOR_DIM {
            out[HELIOSPHERE_INVARIANT_DIM + idx] += sample.descriptor_channels[idx];
        }
    }
    for value in &mut out {
        *value /= samples.len() as f64;
    }
    out
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>();
    let norm_a = a.iter().map(|value| value * value).sum::<f64>().sqrt();
    let norm_b = b.iter().map(|value| value * value).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

fn l2_norm_sq<const N: usize>(values: &[f64; N]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>()
}

fn ratio_usize(num: usize, denom: usize) -> f64 {
    if denom == 0 {
        0.0
    } else {
        num as f64 / denom as f64
    }
}

fn mean(values: &[f64]) -> f64 {
    let finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return f64::NAN;
    }
    finite.iter().sum::<f64>() / finite.len() as f64
}

fn finite_median(values: &[f64]) -> f64 {
    finite_median_opt(values).unwrap_or(0.0)
}

fn finite_median_opt(values: &[f64]) -> Option<f64> {
    let mut finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return None;
    }
    finite.sort_by(|a, b| a.total_cmp(b));
    let mid = finite.len() / 2;
    Some(if finite.len() % 2 == 0 {
        0.5 * (finite[mid - 1] + finite[mid])
    } else {
        finite[mid]
    })
}

fn finite_mad(values: &[f64], median: f64) -> f64 {
    let deviations = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .map(|value| (value - median).abs())
        .collect::<Vec<_>>();
    finite_median(&deviations)
}

fn median_filter_3(values: &[f64]) -> Vec<f64> {
    if values.len() < 3 {
        return values.to_vec();
    }
    let mut out = Vec::with_capacity(values.len());
    for idx in 0..values.len() {
        let start = idx.saturating_sub(1);
        let end = (idx + 1).min(values.len() - 1);
        out.push(finite_median(&values[start..=end]));
    }
    out
}

fn hysteresis_mask(values: &[f64], on: f64, off: f64) -> Vec<bool> {
    let mut active = false;
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        if !active && *value >= on {
            active = true;
        } else if active && *value <= off {
            active = false;
        }
        out.push(active);
    }
    out
}

fn dilate_mask(values: &[bool], radius: usize) -> Vec<bool> {
    let mut out = vec![false; values.len()];
    for (idx, active) in values.iter().copied().enumerate() {
        if !active {
            continue;
        }
        let start = idx.saturating_sub(radius);
        let end = (idx + radius).min(values.len().saturating_sub(1));
        out[start..=end].fill(true);
    }
    out
}

fn merge_small_gaps(values: &[bool], max_gap: usize) -> Vec<bool> {
    let mut out = values.to_vec();
    let mut idx = 0usize;
    while idx < out.len() {
        if out[idx] {
            idx += 1;
            continue;
        }
        let gap_start = idx;
        while idx < out.len() && !out[idx] {
            idx += 1;
        }
        let gap_end = idx;
        let gap_len = gap_end.saturating_sub(gap_start);
        if gap_len <= max_gap
            && gap_start > 0
            && gap_end < out.len()
            && out[gap_start - 1]
            && out[gap_end]
        {
            out[gap_start..gap_end].fill(true);
        }
    }
    out
}

/// Public channel names for the algebra descriptor extension.
pub const HELIOSPHERE_DESCRIPTOR_CHANNEL_NAMES: [&str; DESCRIPTOR_DIM] = [
    "weighted_norm_sq",
    "delta_weighted_norm_sq",
    "rolling_associator_norm",
    "mean_abs_weighted_channel",
];

fn to_cd16(values: &[f64; HELIOSPHERE_INVARIANT_DIM]) -> [f64; 16] {
    let mut out = [0.0_f64; 16];
    out[..HELIOSPHERE_INVARIANT_DIM].copy_from_slice(values);
    out
}

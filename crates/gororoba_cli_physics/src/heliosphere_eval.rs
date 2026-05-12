//! Shared heliosphere evaluation helpers for predictive, invariance, and
//! sparsification experiments.

use anyhow::{Context, Result, bail};
use chrono::{DateTime, Duration, Utc};
use csv::ReaderBuilder;
use data_core::{
    HELIOSPHERE_INVARIANT_CHANNEL_NAMES, HELIOSPHERE_INVARIANT_DIM, HeliosphereEventSource,
    HeliosphereEventWindow, HeliosphereFeatureRow, HeliosphereInvariantSample,
    HeliosphereTransformMode, SparseHardwareEnvelope, compute_invariant_samples,
    estimate_sparse_execution_plan, fetch_donki_event_labels, fetch_official_forecast_residuals,
    heliosphere_row_datetime, labels_to_prediction_windows, transform_feature_rows_with_stats,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    hash::{DefaultHasher, Hash, Hasher},
    path::Path,
};

const DESCRIPTOR_DIM: usize = 8;

// Public summary record types (LabeledInvariantSample, BinaryMetrics,
// MissionSplitSummary, MissionInvarianceSummary, SparseMaskSummary,
// CounterfactualPredictiveSummary, CounterfactualSparseSummary,
// SeededSparsePolicySummary, SparsePolicyTransferSpec,
// SparsePolicyDatasetContext, LabelCoverageRow) plus the RowKey type
// alias live in the `public_types` submodule.
pub mod public_types;
pub use public_types::{
    BinaryMetrics, CounterfactualPredictiveSummary, CounterfactualSparseSummary, LabelCoverageRow,
    LabeledInvariantSample, MissionInvarianceSummary, MissionSplitSummary, RowKey,
    SeededSparsePolicySummary, SparseMaskSummary, SparsePolicyDatasetContext,
    SparsePolicyTransferSpec,
};

type OccupancyTileKey = (String, String, String, u16, u16, u8, Option<u32>);

#[derive(Debug, Clone)]
struct ScaledFeatureSet {
    means: Vec<f64>,
    scales: Vec<f64>,
}

#[derive(Debug, Clone)]
struct NormalizationParams {
    medians: [f64; HELIOSPHERE_INVARIANT_DIM],
    scales: [f64; HELIOSPHERE_INVARIANT_DIM],
}

#[derive(Debug, Clone)]
struct NormalizedSample {
    normalized_channels: [f64; HELIOSPHERE_INVARIANT_DIM],
    normalized_descriptor_channels: [f64; DESCRIPTOR_DIM],
}

#[derive(Debug, Clone, Copy)]
enum ViewMode {
    Raw,
    Normalized,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NormalizationStrategy {
    Global,
    Mission,
    MissionProduct,
}

impl NormalizationStrategy {
    fn label(self) -> &'static str {
        match self {
            Self::Global => "global_quiet",
            Self::Mission => "mission_quiet",
            Self::MissionProduct => "mission_product_quiet",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DescriptorProfile {
    Full,
    DeltaAssociator,
    AssociatorOnly,
    TakensSedenion,
    TakensComparison,
}

impl DescriptorProfile {
    fn label(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::DeltaAssociator => "delta_associator",
            Self::AssociatorOnly => "associator_only",
            Self::TakensSedenion => "takens_sedenion",
            Self::TakensComparison => "takens_comparison",
        }
    }

    fn from_label(value: &str) -> Option<Self> {
        match value {
            "full" => Some(Self::Full),
            "delta_associator" => Some(Self::DeltaAssociator),
            "associator_only" => Some(Self::AssociatorOnly),
            "takens_sedenion" => Some(Self::TakensSedenion),
            "takens_comparison" => Some(Self::TakensComparison),
            _ => None,
        }
    }
}

impl NormalizationStrategy {
    fn from_label(value: &str) -> Option<Self> {
        match value {
            "global_quiet" => Some(Self::Global),
            "mission_quiet" => Some(Self::Mission),
            "mission_product_quiet" => Some(Self::MissionProduct),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum SparsePolicyKind {
    InvariantBudget,
    HybridBudget,
}

#[derive(Debug, Clone)]
struct ThresholdedSparsePolicy {
    name: &'static str,
    mask: BTreeMap<RowKey, bool>,
}

#[derive(Debug, Clone)]
struct FittedSparsePolicyProfile {
    name: &'static str,
    scaler: ScaledFeatureSet,
    model: LogisticModel,
    threshold: f64,
    descriptor_profile: Option<DescriptorProfile>,
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

fn occupancy_tile_key(sample: &LabeledInvariantSample) -> OccupancyTileKey {
    let hour_bucket = (sample.key.5 / 6) * 6;
    (
        sample.window_name.clone(),
        sample.mission.clone(),
        sample.product.clone(),
        sample.key.3,
        sample.key.4,
        hour_bucket,
        None,
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
                channels: sample.channels,
                uncertainty_scales: sample.uncertainty_scales,
                weighted_channels: sample.weighted_channels,
                b_field: sample.b_field,
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

/// Build and cache the label-joined invariant samples for repeated sparse-policy
/// transfer evaluation.
pub fn build_sparse_policy_dataset_context(
    rows: &[HeliosphereFeatureRow],
    cache_root: &Path,
    horizon_hours: i64,
) -> Result<SparsePolicyDatasetContext> {
    let (samples, _) = build_labeled_samples(rows, cache_root, horizon_hours)?;
    let positive_sample_count = samples
        .iter()
        .filter(|sample| sample.label_positive)
        .count();
    Ok(SparsePolicyDatasetContext {
        positive_sample_count,
        samples,
    })
}

/// Summarize official label coverage for each mission/product lane.
pub fn summarize_label_coverage(
    rows: &[HeliosphereFeatureRow],
    cache_root: &Path,
) -> Result<Vec<LabelCoverageRow>> {
    let (start_date, end_date) = cube_date_bounds(rows)?;
    let labels = fetch_donki_event_labels(
        start_date - Duration::days(2),
        end_date + Duration::days(2),
        cache_root,
    )?;
    let residuals =
        fetch_official_forecast_residuals(start_date, end_date, cache_root).unwrap_or_default();
    let mut grouped: BTreeMap<(String, String), Vec<&HeliosphereFeatureRow>> = BTreeMap::new();
    for row in rows {
        grouped
            .entry((row.mission.clone(), row.product.clone()))
            .or_default()
            .push(row);
    }
    let mut coverage = grouped
        .into_iter()
        .map(|((mission, product), rows)| {
            let sources = labels
                .iter()
                .filter(|label| {
                    label
                        .mission_targets
                        .iter()
                        .any(|target| normalize_text(target) == normalize_text(&mission))
                })
                .map(|label| format!("{:?}", label.source))
                .collect::<BTreeSet<_>>();
            let windows_6 = labels_to_prediction_windows(&labels, &mission, 6);
            let windows_12 = labels_to_prediction_windows(&labels, &mission, 12);
            let windows_24 = labels_to_prediction_windows(&labels, &mission, 24);
            let positive_row_count_6h = positive_row_count(&rows, &windows_6);
            let positive_row_count_12h = positive_row_count(&rows, &windows_12);
            let positive_row_count_24h = positive_row_count(&rows, &windows_24);
            let forecast_residual_count = residuals
                .iter()
                .filter(|row| normalize_text(&row.mission) == normalize_text(&mission))
                .count();
            let has_observed_overlap = windows_24
                .iter()
                .any(|window| !matches!(window.source, HeliosphereEventSource::DonkiEnlilImpact))
                && positive_row_count_24h > 0;
            let has_forecast_only_overlap = positive_row_count_24h > 0 && !has_observed_overlap;
            let coverage_status = if has_observed_overlap {
                "observed_label_overlap"
            } else if has_forecast_only_overlap {
                "forecast_target_overlap"
            } else if !sources.is_empty() {
                "official_sources_no_overlap"
            } else {
                "unlabeled"
            }
            .to_string();
            let mut blocked_reasons = Vec::new();
            if sources.is_empty() {
                blocked_reasons.push("no_official_targeted_labels".to_string());
            }
            if !sources.is_empty() && positive_row_count_24h == 0 {
                blocked_reasons.push("official_labels_present_but_no_cube_overlap".to_string());
            }
            if forecast_residual_count == 0
                && matches!(
                    normalize_text(&mission).as_str(),
                    "omni" | "ace" | "wind" | "soho"
                )
            {
                blocked_reasons.push("no_scoreboard_residuals_in_window".to_string());
            }
            LabelCoverageRow {
                mission,
                product,
                row_count: rows.len(),
                source_families: sources.into_iter().collect(),
                positive_window_count_6h: windows_6.len(),
                positive_window_count_12h: windows_12.len(),
                positive_window_count_24h: windows_24.len(),
                positive_row_count_6h,
                positive_row_count_12h,
                positive_row_count_24h,
                forecast_residual_count,
                coverage_status,
                blocked_reasons,
            }
        })
        .collect::<Vec<_>>();
    coverage.sort_by(|a, b| {
        (a.mission.as_str(), a.product.as_str()).cmp(&(b.mission.as_str(), b.product.as_str()))
    });
    Ok(coverage)
}

/// Evaluate scalar, invariant-only, and invariant-plus-descriptor predictors.
pub fn evaluate_predictive_models(
    samples: &[LabeledInvariantSample],
) -> Result<Vec<BinaryMetrics>> {
    if samples.is_empty() {
        bail!("no labeled invariant samples available");
    }
    let splits = split_samples(samples);
    if splits.train.is_empty() || splits.validation.is_empty() || splits.test.is_empty() {
        bail!("need non-empty train/validation/test splits");
    }
    let normalized = build_normalized_samples(samples);

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

    let best_single_raw =
        best_single_invariant_threshold(&splits.validation, ViewMode::Raw, &normalized);
    let best_single_normalized =
        best_single_invariant_threshold(&splits.validation, ViewMode::Normalized, &normalized);

    let invariant_train = feature_matrix(
        &splits.train,
        ViewMode::Raw,
        FeatureMode::Invariants,
        &normalized,
    );
    let invariant_validation = feature_matrix(
        &splits.validation,
        ViewMode::Raw,
        FeatureMode::Invariants,
        &normalized,
    );
    let invariant_test = feature_matrix(
        &splits.test,
        ViewMode::Raw,
        FeatureMode::Invariants,
        &normalized,
    );
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
    let invariant_validation_scores =
        predict_scores(&invariant_model, &invariant_validation_scaled);
    let invariant_threshold = best_threshold(
        &invariant_validation_scores,
        &splits
            .validation
            .iter()
            .map(|sample| sample.label_positive)
            .collect::<Vec<_>>(),
    );

    let hybrid_train = feature_matrix(
        &splits.train,
        ViewMode::Raw,
        FeatureMode::InvariantsAndDescriptors,
        &normalized,
    );
    let hybrid_validation = feature_matrix(
        &splits.validation,
        ViewMode::Raw,
        FeatureMode::InvariantsAndDescriptors,
        &normalized,
    );
    let hybrid_test = feature_matrix(
        &splits.test,
        ViewMode::Raw,
        FeatureMode::InvariantsAndDescriptors,
        &normalized,
    );
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

    let normalized_invariant_train = feature_matrix(
        &splits.train,
        ViewMode::Normalized,
        FeatureMode::Invariants,
        &normalized,
    );
    let normalized_invariant_validation = feature_matrix(
        &splits.validation,
        ViewMode::Normalized,
        FeatureMode::Invariants,
        &normalized,
    );
    let normalized_invariant_test = feature_matrix(
        &splits.test,
        ViewMode::Normalized,
        FeatureMode::Invariants,
        &normalized,
    );
    let normalized_invariant_scaler = fit_scaler(&normalized_invariant_train);
    let normalized_invariant_train_scaled =
        apply_scaler(&normalized_invariant_scaler, &normalized_invariant_train);
    let normalized_invariant_validation_scaled = apply_scaler(
        &normalized_invariant_scaler,
        &normalized_invariant_validation,
    );
    let normalized_invariant_test_scaled =
        apply_scaler(&normalized_invariant_scaler, &normalized_invariant_test);
    let normalized_invariant_model = train_logistic_model(
        &normalized_invariant_train_scaled,
        &binary_labels(&splits.train),
        0.02,
        250,
        1e-3,
    );
    let normalized_invariant_validation_scores = predict_scores(
        &normalized_invariant_model,
        &normalized_invariant_validation_scaled,
    );
    let normalized_invariant_threshold = best_threshold(
        &normalized_invariant_validation_scores,
        &splits
            .validation
            .iter()
            .map(|sample| sample.label_positive)
            .collect::<Vec<_>>(),
    );

    let normalized_hybrid_train = feature_matrix(
        &splits.train,
        ViewMode::Normalized,
        FeatureMode::InvariantsAndDescriptors,
        &normalized,
    );
    let normalized_hybrid_validation = feature_matrix(
        &splits.validation,
        ViewMode::Normalized,
        FeatureMode::InvariantsAndDescriptors,
        &normalized,
    );
    let normalized_hybrid_test = feature_matrix(
        &splits.test,
        ViewMode::Normalized,
        FeatureMode::InvariantsAndDescriptors,
        &normalized,
    );
    let normalized_hybrid_scaler = fit_scaler(&normalized_hybrid_train);
    let normalized_hybrid_train_scaled =
        apply_scaler(&normalized_hybrid_scaler, &normalized_hybrid_train);
    let normalized_hybrid_validation_scaled =
        apply_scaler(&normalized_hybrid_scaler, &normalized_hybrid_validation);
    let normalized_hybrid_test_scaled =
        apply_scaler(&normalized_hybrid_scaler, &normalized_hybrid_test);
    let normalized_hybrid_model = train_logistic_model(
        &normalized_hybrid_train_scaled,
        &binary_labels(&splits.train),
        0.02,
        300,
        1e-3,
    );
    let normalized_hybrid_validation_scores = predict_scores(
        &normalized_hybrid_model,
        &normalized_hybrid_validation_scaled,
    );
    let normalized_hybrid_threshold = best_threshold(
        &normalized_hybrid_validation_scores,
        &splits
            .validation
            .iter()
            .map(|sample| sample.label_positive)
            .collect::<Vec<_>>(),
    );

    Ok(vec![
        evaluate_scores(
            "raw",
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
            "raw",
            "best_single_invariant_threshold",
            &splits.test,
            &splits
                .test
                .iter()
                .map(|sample| sample.weighted_channels[best_single_raw.index])
                .collect::<Vec<_>>(),
            best_single_raw.threshold,
        ),
        evaluate_scores(
            "raw",
            "invariant_logistic",
            &splits.test,
            &predict_scores(&invariant_model, &invariant_test_scaled),
            invariant_threshold,
        ),
        evaluate_scores(
            "raw",
            "invariant_plus_algebra_logistic",
            &splits.test,
            &predict_scores(&hybrid_model, &hybrid_test_scaled),
            hybrid_threshold,
        ),
        evaluate_scores(
            "normalized",
            "best_single_invariant_threshold",
            &splits.test,
            &splits
                .test
                .iter()
                .map(|sample| {
                    normalized
                        .get(&sample.key)
                        .map(|row| row.normalized_channels[best_single_normalized.index])
                        .unwrap_or(0.0)
                })
                .collect::<Vec<_>>(),
            best_single_normalized.threshold,
        ),
        evaluate_scores(
            "normalized",
            "invariant_logistic",
            &splits.test,
            &predict_scores(
                &normalized_invariant_model,
                &normalized_invariant_test_scaled,
            ),
            normalized_invariant_threshold,
        ),
        evaluate_scores(
            "normalized",
            "invariant_plus_algebra_logistic",
            &splits.test,
            &predict_scores(&normalized_hybrid_model, &normalized_hybrid_test_scaled),
            normalized_hybrid_threshold,
        ),
    ])
}

/// Challenge recorded predictive falsifications with alternate normalization
/// families and algebra descriptor subsets.
pub fn evaluate_predictive_counterfactuals(
    samples: &[LabeledInvariantSample],
) -> Result<Vec<CounterfactualPredictiveSummary>> {
    if samples.is_empty() {
        bail!("no labeled invariant samples available");
    }
    let splits = split_samples(samples);
    if splits.train.is_empty() || splits.validation.is_empty() || splits.test.is_empty() {
        bail!("need non-empty train/validation/test splits");
    }

    let mut rows = Vec::new();
    rows.push(train_counterfactual_predictive_model(
        &splits,
        ViewMode::Raw,
        "raw",
        None,
        None,
    ));
    for descriptor_profile in [
        DescriptorProfile::Full,
        DescriptorProfile::DeltaAssociator,
        DescriptorProfile::AssociatorOnly,
        DescriptorProfile::TakensSedenion,
        DescriptorProfile::TakensComparison,
    ] {
        rows.push(train_counterfactual_predictive_model(
            &splits,
            ViewMode::Raw,
            "raw",
            None,
            Some(descriptor_profile),
        ));
    }
    for strategy in [
        NormalizationStrategy::Global,
        NormalizationStrategy::Mission,
        NormalizationStrategy::MissionProduct,
    ] {
        let normalized = build_normalized_samples_with_strategy(samples, strategy);
        rows.push(train_counterfactual_predictive_model(
            &splits,
            ViewMode::Normalized,
            strategy.label(),
            Some(&normalized),
            None,
        ));
        for descriptor_profile in [
            DescriptorProfile::Full,
            DescriptorProfile::DeltaAssociator,
            DescriptorProfile::AssociatorOnly,
            DescriptorProfile::TakensSedenion,
            DescriptorProfile::TakensComparison,
        ] {
            rows.push(train_counterfactual_predictive_model(
                &splits,
                ViewMode::Normalized,
                strategy.label(),
                Some(&normalized),
                Some(descriptor_profile),
            ));
        }
    }
    Ok(rows)
}

/// Summarize leave-one-mission-out descriptor stability.
pub fn summarize_cross_mission_invariance(
    samples: &[LabeledInvariantSample],
) -> Vec<MissionInvarianceSummary> {
    let normalized = build_normalized_samples(samples);
    let mut grouped: BTreeMap<String, Vec<&LabeledInvariantSample>> = BTreeMap::new();
    for sample in samples {
        grouped
            .entry(sample.mission.clone())
            .or_default()
            .push(sample);
    }

    let mut summaries = Vec::new();
    for view_mode in [ViewMode::Raw, ViewMode::Normalized] {
        for (mission, group) in &grouped {
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
            let mission_vector = mean_feature_vector(&positive, view_mode, &normalized);
            let rest_positive = grouped
                .iter()
                .filter(|(other, _)| *other != mission)
                .flat_map(|(_, rows)| rows.iter().copied())
                .filter(|sample| sample.label_positive)
                .collect::<Vec<_>>();
            let rest_vector = mean_feature_vector(&rest_positive, view_mode, &normalized);
            summaries.push(MissionInvarianceSummary {
                feature_mode: match view_mode {
                    ViewMode::Raw => "raw".to_string(),
                    ViewMode::Normalized => "normalized".to_string(),
                },
                mission: mission.clone(),
                positive_rows: positive.len(),
                negative_rows: negative.len(),
                positive_mean_weighted_norm: mean(
                    &positive
                        .iter()
                        .map(|sample| sample_invariant_norm(sample, view_mode, &normalized))
                        .collect::<Vec<_>>(),
                ),
                negative_mean_weighted_norm: mean(
                    &negative
                        .iter()
                        .map(|sample| sample_invariant_norm(sample, view_mode, &normalized))
                        .collect::<Vec<_>>(),
                ),
                positive_descriptor_mean: mean(
                    &positive
                        .iter()
                        .map(|sample| sample_descriptor_value(sample, view_mode, &normalized, 0))
                        .collect::<Vec<_>>(),
                ),
                leave_one_mission_out_cosine: cosine_similarity(&mission_vector, &rest_vector),
                blocking_channels: top_blocking_channels(&mission_vector, &rest_vector, 3),
            });
        }
    }
    summaries
}

/// Compute sparse-policy summaries for the robust baseline and budgeted policies.
pub fn summarize_sparse_policies(
    raw_rows: &[HeliosphereFeatureRow],
    cache_root: &Path,
    horizon_hours: i64,
    grid: usize,
) -> Result<Vec<SparseMaskSummary>> {
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
    let normalized = build_normalized_samples(&samples);
    let invariant_policy = fit_sparse_budget_policy(
        &samples,
        &normalized,
        SparsePolicyKind::InvariantBudget,
        grid,
        12.0,
    )?;
    let hybrid_policy = fit_sparse_budget_policy(
        &samples,
        &normalized,
        SparsePolicyKind::HybridBudget,
        grid,
        12.0,
    )?;
    let invariant_rows = raw_rows
        .iter()
        .filter(|row| *invariant_policy.mask.get(&row_key(row)).unwrap_or(&false))
        .collect::<Vec<_>>();
    let hybrid_rows = raw_rows
        .iter()
        .filter(|row| *hybrid_policy.mask.get(&row_key(row)).unwrap_or(&false))
        .collect::<Vec<_>>();
    let labels = label_index(&samples);
    let time_index = raw_time_index(raw_rows);
    Ok(vec![
        sparse_summary(
            "robust_baseline",
            &baseline_rows,
            raw_rows.len(),
            &labels,
            &time_index,
            &baseline_index,
            &samples,
        ),
        sparse_summary(
            invariant_policy.name,
            &invariant_rows,
            raw_rows.len(),
            &labels,
            &time_index,
            &invariant_policy.mask,
            &samples,
        ),
        sparse_summary(
            hybrid_policy.name,
            &hybrid_rows,
            raw_rows.len(),
            &labels,
            &time_index,
            &hybrid_policy.mask,
            &samples,
        ),
    ])
}

/// Challenge sparse-policy falsifications with alternate normalization families
/// and descriptor subsets under the same hard memory budget.
pub fn summarize_sparse_policy_counterfactuals(
    raw_rows: &[HeliosphereFeatureRow],
    cache_root: &Path,
    horizon_hours: i64,
    grid: usize,
) -> Result<Vec<CounterfactualSparseSummary>> {
    summarize_sparse_policy_counterfactuals_with_seed(raw_rows, cache_root, horizon_hours, grid, 0)
}

/// Challenge sparse policies with alternate normalization families and
/// descriptor subsets using a deterministic split seed.
pub fn summarize_sparse_policy_counterfactuals_with_seed(
    raw_rows: &[HeliosphereFeatureRow],
    cache_root: &Path,
    horizon_hours: i64,
    grid: usize,
    split_seed: u64,
) -> Result<Vec<CounterfactualSparseSummary>> {
    let (samples, _) = build_labeled_samples(raw_rows, cache_root, horizon_hours)?;
    let mut rows = Vec::new();
    for strategy in [
        NormalizationStrategy::Global,
        NormalizationStrategy::Mission,
        NormalizationStrategy::MissionProduct,
    ] {
        let normalized =
            build_normalized_samples_with_strategy_and_seed(&samples, strategy, split_seed);
        rows.push(thresholded_sparse_policy_summary(
            &samples,
            &normalized,
            strategy.label(),
            None,
            grid,
        )?);
        for descriptor_profile in [
            DescriptorProfile::Full,
            DescriptorProfile::DeltaAssociator,
            DescriptorProfile::AssociatorOnly,
            DescriptorProfile::TakensSedenion,
            DescriptorProfile::TakensComparison,
        ] {
            rows.push(thresholded_sparse_policy_summary(
                &samples,
                &normalized,
                strategy.label(),
                Some(descriptor_profile),
                grid,
            )?);
        }
    }
    Ok(rows)
}

/// Seeded sparse-policy rows for promoted mainline evaluation.
pub fn summarize_seeded_sparse_policy_rows(
    raw_rows: &[HeliosphereFeatureRow],
    cache_root: &Path,
    horizon_hours: i64,
    grid: usize,
    split_seed: u64,
) -> Result<(usize, Vec<SeededSparsePolicySummary>)> {
    let (samples, _) = build_labeled_samples(raw_rows, cache_root, horizon_hours)?;
    let positive_sample_count = samples
        .iter()
        .filter(|sample| sample.label_positive)
        .count();
    let counterfactuals = summarize_sparse_policy_counterfactuals_with_seed(
        raw_rows,
        cache_root,
        horizon_hours,
        grid,
        split_seed,
    )?;
    let rows = counterfactuals
        .into_iter()
        .map(|row| SeededSparsePolicySummary {
            split_seed,
            normalization_strategy: row.normalization_strategy,
            descriptor_profile: row.descriptor_profile,
            active_rows: row.active_rows,
            active_fraction: row.active_fraction,
            occupancy_tiles_active: row.occupancy_tiles_active,
            occupancy_tiles_total: row.occupancy_tiles_total,
            occupancy_tile_fraction: row.occupancy_tile_fraction,
            event_label_recall: row.event_label_recall,
            event_label_precision: row.event_label_precision,
            sparse_bf16_aa_projected_gib: row.sparse_bf16_aa_projected_gib,
            median_lead_time_hours: row.median_lead_time_hours,
        })
        .collect::<Vec<_>>();
    Ok((positive_sample_count, rows))
}

/// Seeded evaluation for one targeted sparse-policy configuration.
pub fn summarize_targeted_seeded_sparse_policy(
    raw_rows: &[HeliosphereFeatureRow],
    cache_root: &Path,
    horizon_hours: i64,
    grid: usize,
    split_seed: u64,
    normalization_strategy: &str,
    descriptor_profile: &str,
) -> Result<(usize, SeededSparsePolicySummary)> {
    let strategy = NormalizationStrategy::from_label(normalization_strategy)
        .with_context(|| format!("unknown normalization strategy '{normalization_strategy}'"))?;
    let descriptor = if descriptor_profile == "invariants_only" {
        None
    } else {
        Some(
            DescriptorProfile::from_label(descriptor_profile)
                .with_context(|| format!("unknown descriptor profile '{descriptor_profile}'"))?,
        )
    };
    let (samples, _) = build_labeled_samples(raw_rows, cache_root, horizon_hours)?;
    let positive_sample_count = samples
        .iter()
        .filter(|sample| sample.label_positive)
        .count();
    let normalized =
        build_normalized_samples_with_strategy_and_seed(&samples, strategy, split_seed);
    let row = thresholded_sparse_policy_summary(
        &samples,
        &normalized,
        normalization_strategy,
        descriptor,
        grid,
    )?;
    Ok((
        positive_sample_count,
        SeededSparsePolicySummary {
            split_seed,
            normalization_strategy: row.normalization_strategy,
            descriptor_profile: row.descriptor_profile,
            active_rows: row.active_rows,
            active_fraction: row.active_fraction,
            occupancy_tiles_active: row.occupancy_tiles_active,
            occupancy_tiles_total: row.occupancy_tiles_total,
            occupancy_tile_fraction: row.occupancy_tile_fraction,
            event_label_recall: row.event_label_recall,
            event_label_precision: row.event_label_precision,
            sparse_bf16_aa_projected_gib: row.sparse_bf16_aa_projected_gib,
            median_lead_time_hours: row.median_lead_time_hours,
        },
    ))
}

/// Seeded transfer evaluation for one targeted sparse-policy configuration.
///
/// The policy is fit on `training_rows` and then applied unchanged to
/// `target_rows`, which is the correct cross-cube evaluation path for unlabeled
/// transfer cubes.
pub fn summarize_transferred_seeded_sparse_policy(
    training_rows: &[HeliosphereFeatureRow],
    target_rows: &[HeliosphereFeatureRow],
    cache_root: &Path,
    spec: &SparsePolicyTransferSpec<'_>,
) -> Result<(usize, SeededSparsePolicySummary)> {
    let training_context =
        build_sparse_policy_dataset_context(training_rows, cache_root, spec.horizon_hours)?;
    let target_context =
        build_sparse_policy_dataset_context(target_rows, cache_root, spec.horizon_hours)?;
    summarize_transferred_seeded_sparse_policy_from_contexts(
        &training_context,
        &target_context,
        spec,
    )
}

/// Cached-context version of sparse-policy transfer evaluation.
pub fn summarize_transferred_seeded_sparse_policy_from_contexts(
    training: &SparsePolicyDatasetContext,
    target: &SparsePolicyDatasetContext,
    spec: &SparsePolicyTransferSpec<'_>,
) -> Result<(usize, SeededSparsePolicySummary)> {
    let strategy =
        NormalizationStrategy::from_label(spec.normalization_strategy).with_context(|| {
            format!(
                "unknown normalization strategy '{}'",
                spec.normalization_strategy
            )
        })?;
    let descriptor = if spec.descriptor_profile == "invariants_only" {
        None
    } else {
        Some(
            DescriptorProfile::from_label(spec.descriptor_profile).with_context(|| {
                format!("unknown descriptor profile '{}'", spec.descriptor_profile)
            })?,
        )
    };
    let training_normalized = build_normalized_samples_with_strategy_and_seed(
        &training.samples,
        strategy,
        spec.split_seed,
    );
    let fitted = fit_sparse_budget_policy_profile_model_with_seed(
        &training.samples,
        &training_normalized,
        descriptor,
        spec.grid,
        12.0,
        spec.split_seed,
    )?;
    let target_normalized =
        build_normalized_samples_with_strategy_and_seed(&target.samples, strategy, spec.split_seed);
    let target_refs = target.samples.iter().collect::<Vec<_>>();
    let target_matrix = feature_matrix_with_descriptor_profile(
        &target_refs,
        ViewMode::Normalized,
        &target_normalized,
        fitted.descriptor_profile,
    );
    let target_scaled = apply_scaler(&fitted.scaler, &target_matrix);
    let target_scores = predict_scores(&fitted.model, &target_scaled);
    let active_index = target
        .samples
        .iter()
        .zip(target_scores)
        .map(|(sample, score)| {
            (
                sample.key.clone(),
                score.is_finite() && score >= fitted.threshold,
            )
        })
        .collect::<BTreeMap<_, _>>();
    let label_index = label_index(&target.samples);
    let time_index = target
        .samples
        .iter()
        .map(|sample| (sample.key.clone(), sample.timestamp_utc.clone()))
        .collect::<BTreeMap<_, _>>();
    let summary = sparse_summary_from_active_index(
        &active_index,
        target.samples.len(),
        &label_index,
        &time_index,
        &target.samples,
    );
    let projected_gib = estimate_sparse_execution_plan(
        spec.grid,
        summary.occupancy_tile_fraction,
        None,
        SparseHardwareEnvelope {
            cuda_vram_budget_bytes: None,
            cuda_l2_bytes: None,
            cuda_shared_mem_per_block: None,
            cuda_managed_memory: None,
            cuda_concurrent_managed_access: None,
            cpu_l3_safe_working_set_bytes: None,
            prefer_sparse_tile: false,
        },
    )
    .memory
    .sparse_bf16_aa_projected_gib;
    Ok((
        target.positive_sample_count,
        SeededSparsePolicySummary {
            split_seed: spec.split_seed,
            normalization_strategy: spec.normalization_strategy.to_string(),
            descriptor_profile: spec.descriptor_profile.to_string(),
            active_rows: summary.active_rows,
            active_fraction: summary.active_fraction,
            occupancy_tiles_active: summary.occupancy_tiles_active,
            occupancy_tiles_total: summary.occupancy_tiles_total,
            occupancy_tile_fraction: summary.occupancy_tile_fraction,
            event_label_recall: summary.event_label_recall,
            event_label_precision: summary.event_label_precision,
            sparse_bf16_aa_projected_gib: projected_gib,
            median_lead_time_hours: summary.median_lead_time_hours,
        },
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
                .cmp(&(
                    b.timestamp_utc.as_str(),
                    b.mission.as_str(),
                    b.product.as_str(),
                ))
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
    let end =
        DateTime::parse_from_rfc3339(&window.window_end_utc).map(|value| value.with_timezone(&Utc));
    match (start, end) {
        (Ok(start), Ok(end)) => timestamp >= start && timestamp <= end,
        _ => false,
    }
}

fn descriptor_channels(group: &[HeliosphereInvariantSample], idx: usize) -> [f64; DESCRIPTOR_DIM] {
    let vectors = group
        .iter()
        .map(|sample| sample.weighted_channels)
        .collect::<Vec<_>>();
    let mut out = [0.0; DESCRIPTOR_DIM];
    let base = descriptor_channels_from_arrays(&vectors, idx);
    out[..4].copy_from_slice(&base);

    // Compute Takens descriptors (using 4-step delay of 4D B-field)
    let takens = takens_descriptors(group, idx);
    out[4..8].copy_from_slice(&takens);
    out
}

trait HasBField {
    fn b_field(&self) -> [f64; 4];
}

impl HasBField for HeliosphereInvariantSample {
    fn b_field(&self) -> [f64; 4] {
        self.b_field
    }
}

impl HasBField for &LabeledInvariantSample {
    fn b_field(&self) -> [f64; 4] {
        self.b_field
    }
}

fn takens_descriptors<T: HasBField>(group: &[T], idx: usize) -> [f64; 4] {
    let get_v16 = |target_idx: usize| -> Option<[f64; 16]> {
        if target_idx < 3 {
            return None;
        }
        let mut v16 = [0.0; 16];
        for i in 0..4 {
            let s = &group[target_idx - 3 + i];
            v16[i * 4..i * 4 + 4].copy_from_slice(&s.b_field());
        }
        Some(v16)
    };

    let v_curr = get_v16(idx);
    let v_prev = idx.checked_sub(1).and_then(get_v16);
    let v_prev2 = idx.checked_sub(2).and_then(get_v16);

    match (v_prev2, v_prev, v_curr) {
        (Some(a), Some(b), Some(c)) => {
            let sedenion_assoc = cd_kernel::cd_associator_norm(&a, &b, &c);

            let mut a_oct = [0.0; 16];
            let mut b_oct = [0.0; 16];
            let mut c_oct = [0.0; 16];
            a_oct[..8].copy_from_slice(&a[..8]);
            b_oct[..8].copy_from_slice(&b[..8]);
            c_oct[..8].copy_from_slice(&c[..8]);
            let octonion_assoc = cd_kernel::cd_associator_norm(&a_oct, &b_oct, &c_oct);

            let mut a_rand = a;
            a_rand.reverse();
            let mut b_rand = b;
            b_rand.reverse();
            let mut c_rand = c;
            c_rand.reverse();
            let random_assoc = cd_kernel::cd_associator_norm(&a_rand, &b_rand, &c_rand);

            let euclidean = (l2_norm_sq(&a) + l2_norm_sq(&b) + l2_norm_sq(&c)).sqrt();

            [sedenion_assoc, octonion_assoc, random_assoc, euclidean]
        }
        _ => [0.0; 4],
    }
}

fn descriptor_channels_from_arrays(
    vectors: &[[f64; HELIOSPHERE_INVARIANT_DIM]],
    idx: usize,
) -> [f64; 4] {
    let current = &vectors[idx];
    let prev = idx.checked_sub(1).map(|index| &vectors[index]);
    let prev2 = idx.checked_sub(2).map(|index| &vectors[index]);
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
        grouped
            .entry(sample.mission.clone())
            .or_default()
            .push(sample);
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
                validation_rows: val_end
                    .saturating_sub(train_end)
                    .min(n.saturating_sub(train_end)),
                test_rows: n.saturating_sub(val_end),
            }
        })
        .collect()
}

fn split_samples(samples: &[LabeledInvariantSample]) -> SampleSplits<'_> {
    split_samples_with_seed(samples, 0)
}

fn split_samples_with_seed(
    samples: &[LabeledInvariantSample],
    split_seed: u64,
) -> SampleSplits<'_> {
    let mut grouped: BTreeMap<String, Vec<&LabeledInvariantSample>> = BTreeMap::new();
    for sample in samples {
        grouped
            .entry(sample.mission.clone())
            .or_default()
            .push(sample);
    }
    let mut train = Vec::new();
    let mut validation = Vec::new();
    let mut test = Vec::new();
    for mut group in grouped.into_values() {
        if split_seed == 0 {
            group.sort_by_key(|sample| sample.timestamp_utc.clone());
        } else {
            group.sort_by(|a, b| {
                seeded_split_rank(a, split_seed)
                    .cmp(&seeded_split_rank(b, split_seed))
                    .then_with(|| a.timestamp_utc.cmp(&b.timestamp_utc))
            });
        }
        let n = group.len();
        let train_end = ((n as f64) * 0.70).round() as usize;
        let val_end = ((n as f64) * 0.85).round() as usize;
        train.extend(group.iter().take(train_end).copied());
        validation.extend(
            group
                .iter()
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

fn seeded_split_rank(sample: &LabeledInvariantSample, split_seed: u64) -> u64 {
    let mut hasher = DefaultHasher::new();
    split_seed.hash(&mut hasher);
    sample.key.hash(&mut hasher);
    sample.timestamp_utc.hash(&mut hasher);
    hasher.finish()
}

fn feature_matrix(
    samples: &[&LabeledInvariantSample],
    view_mode: ViewMode,
    mode: FeatureMode,
    normalized: &BTreeMap<RowKey, NormalizedSample>,
) -> Vec<Vec<f64>> {
    match mode {
        FeatureMode::Invariants => {
            feature_matrix_with_descriptor_profile(samples, view_mode, normalized, None)
        }
        FeatureMode::InvariantsAndDescriptors => feature_matrix_with_descriptor_profile(
            samples,
            view_mode,
            normalized,
            Some(DescriptorProfile::Full),
        ),
    }
}

fn feature_matrix_with_descriptor_profile(
    samples: &[&LabeledInvariantSample],
    view_mode: ViewMode,
    normalized: &BTreeMap<RowKey, NormalizedSample>,
    descriptor_profile: Option<DescriptorProfile>,
) -> Vec<Vec<f64>> {
    samples
        .iter()
        .map(|sample| {
            let mut row = invariant_vector(sample, view_mode, normalized).to_vec();
            if let Some(profile) = descriptor_profile {
                row.extend(selected_descriptor_values(
                    sample, view_mode, normalized, profile,
                ));
            }
            row
        })
        .collect()
}

fn binary_labels(samples: &[&LabeledInvariantSample]) -> Vec<f64> {
    samples
        .iter()
        .map(|sample| if sample.label_positive { 1.0 } else { 0.0 })
        .collect()
}

fn cube_date_bounds(
    rows: &[HeliosphereFeatureRow],
) -> Result<(chrono::NaiveDate, chrono::NaiveDate)> {
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
    Ok((start_date, end_date))
}

fn positive_row_count(
    rows: &[&HeliosphereFeatureRow],
    windows: &[HeliosphereEventWindow],
) -> usize {
    rows.iter()
        .filter_map(|row| heliosphere_row_datetime(row))
        .filter(|timestamp| {
            windows
                .iter()
                .any(|window| contains_time(window, *timestamp))
        })
        .count()
}

fn normalize_text(value: &str) -> String {
    value
        .trim()
        .to_ascii_lowercase()
        .replace([' ', '_', '/'], "-")
        .replace("--", "-")
}

fn build_normalized_samples(
    samples: &[LabeledInvariantSample],
) -> BTreeMap<RowKey, NormalizedSample> {
    build_normalized_samples_with_strategy_and_seed(
        samples,
        NormalizationStrategy::MissionProduct,
        0,
    )
}

fn build_normalized_samples_with_strategy(
    samples: &[LabeledInvariantSample],
    strategy: NormalizationStrategy,
) -> BTreeMap<RowKey, NormalizedSample> {
    build_normalized_samples_with_strategy_and_seed(samples, strategy, 0)
}

fn build_normalized_samples_with_strategy_and_seed(
    samples: &[LabeledInvariantSample],
    strategy: NormalizationStrategy,
    split_seed: u64,
) -> BTreeMap<RowKey, NormalizedSample> {
    let splits = split_samples_with_seed(samples, split_seed);
    let train_keys = splits
        .train
        .iter()
        .map(|sample| sample.key.clone())
        .collect::<BTreeSet<_>>();
    let train_quiet = samples
        .iter()
        .filter(|sample| train_keys.contains(&sample.key) && !sample.label_positive)
        .collect::<Vec<_>>();
    let all_sample_refs = samples.iter().collect::<Vec<_>>();
    let global_params = fit_normalization_params(if train_quiet.is_empty() {
        &all_sample_refs
    } else {
        &train_quiet
    });

    let mut group_params = BTreeMap::new();
    let mut grouped_train: BTreeMap<(String, Option<String>), Vec<&LabeledInvariantSample>> =
        BTreeMap::new();
    for sample in samples {
        if train_keys.contains(&sample.key) {
            let group_key = match strategy {
                NormalizationStrategy::Global => continue,
                NormalizationStrategy::Mission => (sample.mission.clone(), None),
                NormalizationStrategy::MissionProduct => {
                    (sample.mission.clone(), Some(sample.product.clone()))
                }
            };
            grouped_train.entry(group_key).or_default().push(sample);
        }
    }
    for (group_key, group_samples) in grouped_train {
        let quiet = group_samples
            .iter()
            .copied()
            .filter(|sample| !sample.label_positive)
            .collect::<Vec<_>>();
        let params = fit_normalization_params(if quiet.is_empty() {
            &group_samples
        } else {
            &quiet
        });
        group_params.insert(group_key, params);
    }

    let mut normalized = BTreeMap::new();
    let mut grouped_all: BTreeMap<(String, String, String), Vec<&LabeledInvariantSample>> =
        BTreeMap::new();
    for sample in samples {
        grouped_all
            .entry((
                sample.window_name.clone(),
                sample.mission.clone(),
                sample.product.clone(),
            ))
            .or_default()
            .push(sample);
    }
    for ((_window, mission, product), mut group) in grouped_all {
        group.sort_by(|a, b| {
            (
                a.timestamp_utc.as_str(),
                a.mission.as_str(),
                a.product.as_str(),
            )
                .cmp(&(
                    b.timestamp_utc.as_str(),
                    b.mission.as_str(),
                    b.product.as_str(),
                ))
        });
        let params = match strategy {
            NormalizationStrategy::Global => global_params.clone(),
            NormalizationStrategy::Mission => group_params
                .get(&(mission.clone(), None))
                .cloned()
                .unwrap_or_else(|| global_params.clone()),
            NormalizationStrategy::MissionProduct => group_params
                .get(&(mission.clone(), Some(product.clone())))
                .cloned()
                .unwrap_or_else(|| global_params.clone()),
        };
        let channels = group
            .iter()
            .map(|sample| normalize_channels(sample, &params))
            .collect::<Vec<_>>();
        for idx in 0..group.len() {
            let mut descriptor = [0.0; DESCRIPTOR_DIM];
            let base = descriptor_channels_from_arrays(&channels, idx);
            descriptor[..4].copy_from_slice(&base);
            let takens = takens_descriptors(&group, idx);
            descriptor[4..8].copy_from_slice(&takens);

            normalized.insert(
                group[idx].key.clone(),
                NormalizedSample {
                    normalized_channels: channels[idx],
                    normalized_descriptor_channels: descriptor,
                },
            );
        }
    }
    normalized
}

fn fit_normalization_params(samples: &[&LabeledInvariantSample]) -> NormalizationParams {
    let mut medians = [0.0_f64; HELIOSPHERE_INVARIANT_DIM];
    let mut scales = [1.0_f64; HELIOSPHERE_INVARIANT_DIM];
    for idx in 0..HELIOSPHERE_INVARIANT_DIM {
        let values = samples
            .iter()
            .map(|sample| sample.channels[idx])
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        let uncertainties = samples
            .iter()
            .map(|sample| sample.uncertainty_scales[idx])
            .filter(|value| value.is_finite() && *value > 0.0)
            .collect::<Vec<_>>();
        let median = finite_median(&values);
        let robust_sigma = 1.4826 * finite_mad(&values, median);
        let uncertainty_sigma = finite_median_opt(&uncertainties).unwrap_or(0.0);
        let std_sigma = finite_std(&values, median);
        medians[idx] = if median.is_finite() { median } else { 0.0 };
        scales[idx] = [robust_sigma, uncertainty_sigma, std_sigma, 1.0]
            .into_iter()
            .filter(|value| value.is_finite() && *value > 1.0e-6)
            .fold(1.0, f64::max);
    }
    NormalizationParams { medians, scales }
}

fn normalize_channels(
    sample: &LabeledInvariantSample,
    params: &NormalizationParams,
) -> [f64; HELIOSPHERE_INVARIANT_DIM] {
    let mut out = [0.0_f64; HELIOSPHERE_INVARIANT_DIM];
    for (idx, slot) in out.iter_mut().enumerate().take(HELIOSPHERE_INVARIANT_DIM) {
        let value = sample.channels[idx];
        *slot = if value.is_finite() {
            (value - params.medians[idx]) / params.scales[idx]
        } else {
            0.0
        };
    }
    out
}

fn invariant_vector(
    sample: &LabeledInvariantSample,
    view_mode: ViewMode,
    normalized: &BTreeMap<RowKey, NormalizedSample>,
) -> [f64; HELIOSPHERE_INVARIANT_DIM] {
    match view_mode {
        ViewMode::Raw => sample.weighted_channels,
        ViewMode::Normalized => normalized
            .get(&sample.key)
            .map(|row| row.normalized_channels)
            .unwrap_or([0.0_f64; HELIOSPHERE_INVARIANT_DIM]),
    }
}

fn descriptor_vector(
    sample: &LabeledInvariantSample,
    view_mode: ViewMode,
    normalized: &BTreeMap<RowKey, NormalizedSample>,
) -> [f64; DESCRIPTOR_DIM] {
    match view_mode {
        ViewMode::Raw => sample.descriptor_channels,
        ViewMode::Normalized => normalized
            .get(&sample.key)
            .map(|row| row.normalized_descriptor_channels)
            .unwrap_or([0.0_f64; DESCRIPTOR_DIM]),
    }
}

fn selected_descriptor_values(
    sample: &LabeledInvariantSample,
    view_mode: ViewMode,
    normalized: &BTreeMap<RowKey, NormalizedSample>,
    descriptor_profile: DescriptorProfile,
) -> Vec<f64> {
    let descriptor = descriptor_vector(sample, view_mode, normalized);
    match descriptor_profile {
        DescriptorProfile::Full => descriptor.to_vec(),
        DescriptorProfile::DeltaAssociator => vec![descriptor[1], descriptor[2]],
        DescriptorProfile::AssociatorOnly => vec![descriptor[2]],
        DescriptorProfile::TakensSedenion => vec![descriptor[4]],
        DescriptorProfile::TakensComparison => descriptor[4..8].to_vec(),
    }
}

fn sample_invariant_value(
    sample: &LabeledInvariantSample,
    view_mode: ViewMode,
    normalized: &BTreeMap<RowKey, NormalizedSample>,
    idx: usize,
) -> f64 {
    invariant_vector(sample, view_mode, normalized)[idx]
}

fn sample_descriptor_value(
    sample: &LabeledInvariantSample,
    view_mode: ViewMode,
    normalized: &BTreeMap<RowKey, NormalizedSample>,
    idx: usize,
) -> f64 {
    descriptor_vector(sample, view_mode, normalized)[idx]
}

fn sample_invariant_norm(
    sample: &LabeledInvariantSample,
    view_mode: ViewMode,
    normalized: &BTreeMap<RowKey, NormalizedSample>,
) -> f64 {
    l2_norm_sq(&invariant_vector(sample, view_mode, normalized)).sqrt()
}

fn top_blocking_channels(mission_vector: &[f64], rest_vector: &[f64], top_n: usize) -> Vec<String> {
    let names = HELIOSPHERE_INVARIANT_CHANNEL_NAMES
        .iter()
        .map(|value| (*value).to_string())
        .chain(
            HELIOSPHERE_DESCRIPTOR_CHANNEL_NAMES
                .iter()
                .map(|value| (*value).to_string()),
        )
        .collect::<Vec<_>>();
    let mut ranked = mission_vector
        .iter()
        .zip(rest_vector.iter())
        .enumerate()
        .map(|(idx, (a, b))| (idx, (a - b).abs()))
        .collect::<Vec<_>>();
    ranked.sort_by(|(_, a), (_, b)| b.total_cmp(a));
    ranked
        .into_iter()
        .take(top_n)
        .map(|(idx, _)| names.get(idx).cloned().unwrap_or_else(|| format!("f{idx}")))
        .collect()
}

fn fit_sparse_budget_policy(
    samples: &[LabeledInvariantSample],
    normalized: &BTreeMap<RowKey, NormalizedSample>,
    kind: SparsePolicyKind,
    grid: usize,
    budget_gib: f64,
) -> Result<ThresholdedSparsePolicy> {
    let splits = split_samples(samples);
    let (name, feature_mode) = match kind {
        SparsePolicyKind::InvariantBudget => ("invariant_budget_policy", FeatureMode::Invariants),
        SparsePolicyKind::HybridBudget => (
            "hybrid_budget_policy",
            FeatureMode::InvariantsAndDescriptors,
        ),
    };
    let train = feature_matrix(
        &splits.train,
        ViewMode::Normalized,
        feature_mode,
        normalized,
    );
    let validation = feature_matrix(
        &splits.validation,
        ViewMode::Normalized,
        feature_mode,
        normalized,
    );
    let all_samples = samples.iter().collect::<Vec<_>>();
    let all_matrix = feature_matrix(&all_samples, ViewMode::Normalized, feature_mode, normalized);
    let scaler = fit_scaler(&train);
    let train_scaled = apply_scaler(&scaler, &train);
    let validation_scaled = apply_scaler(&scaler, &validation);
    let all_scaled = apply_scaler(&scaler, &all_matrix);
    let model = train_logistic_model(
        &train_scaled,
        &binary_labels(&splits.train),
        0.02,
        300,
        1e-3,
    );
    let validation_scores = predict_scores(&model, &validation_scaled);
    let validation_labels = splits
        .validation
        .iter()
        .map(|sample| sample.label_positive)
        .collect::<Vec<_>>();
    let threshold = best_budgeted_threshold(
        &splits.validation,
        &validation_scores,
        &validation_labels,
        grid,
        budget_gib,
    );
    let all_scores = predict_scores(&model, &all_scaled);
    let mask = samples
        .iter()
        .zip(all_scores)
        .map(|(sample, score)| (sample.key.clone(), score.is_finite() && score >= threshold))
        .collect::<BTreeMap<_, _>>();
    Ok(ThresholdedSparsePolicy { name, mask })
}

fn fit_sparse_budget_policy_profile(
    samples: &[LabeledInvariantSample],
    normalized: &BTreeMap<RowKey, NormalizedSample>,
    descriptor_profile: Option<DescriptorProfile>,
    grid: usize,
    budget_gib: f64,
) -> Result<ThresholdedSparsePolicy> {
    fit_sparse_budget_policy_profile_with_seed(
        samples,
        normalized,
        descriptor_profile,
        grid,
        budget_gib,
        0,
    )
}

fn fit_sparse_budget_policy_profile_with_seed(
    samples: &[LabeledInvariantSample],
    normalized: &BTreeMap<RowKey, NormalizedSample>,
    descriptor_profile: Option<DescriptorProfile>,
    grid: usize,
    budget_gib: f64,
    split_seed: u64,
) -> Result<ThresholdedSparsePolicy> {
    let fitted = fit_sparse_budget_policy_profile_model_with_seed(
        samples,
        normalized,
        descriptor_profile,
        grid,
        budget_gib,
        split_seed,
    )?;
    let all_samples = samples.iter().collect::<Vec<_>>();
    let all_matrix = feature_matrix_with_descriptor_profile(
        &all_samples,
        ViewMode::Normalized,
        normalized,
        descriptor_profile,
    );
    let all_scaled = apply_scaler(&fitted.scaler, &all_matrix);
    let all_scores = predict_scores(&fitted.model, &all_scaled);
    let mask = samples
        .iter()
        .zip(all_scores)
        .map(|(sample, score)| {
            (
                sample.key.clone(),
                score.is_finite() && score >= fitted.threshold,
            )
        })
        .collect::<BTreeMap<_, _>>();
    Ok(ThresholdedSparsePolicy {
        name: fitted.name,
        mask,
    })
}

fn fit_sparse_budget_policy_profile_model_with_seed(
    samples: &[LabeledInvariantSample],
    normalized: &BTreeMap<RowKey, NormalizedSample>,
    descriptor_profile: Option<DescriptorProfile>,
    grid: usize,
    budget_gib: f64,
    split_seed: u64,
) -> Result<FittedSparsePolicyProfile> {
    let splits = split_samples_with_seed(samples, split_seed);
    let name = match descriptor_profile {
        None => "invariant_budget_policy",
        Some(profile) => match profile {
            DescriptorProfile::Full => "hybrid_budget_policy_full",
            DescriptorProfile::DeltaAssociator => "hybrid_budget_policy_delta_associator",
            DescriptorProfile::AssociatorOnly => "hybrid_budget_policy_associator_only",
            DescriptorProfile::TakensSedenion => "hybrid_budget_policy_takens_sedenion",
            DescriptorProfile::TakensComparison => "hybrid_budget_policy_takens_comparison",
        },
    };
    let train = feature_matrix_with_descriptor_profile(
        &splits.train,
        ViewMode::Normalized,
        normalized,
        descriptor_profile,
    );
    let validation = feature_matrix_with_descriptor_profile(
        &splits.validation,
        ViewMode::Normalized,
        normalized,
        descriptor_profile,
    );
    let scaler = fit_scaler(&train);
    let train_scaled = apply_scaler(&scaler, &train);
    let validation_scaled = apply_scaler(&scaler, &validation);
    let model = train_logistic_model(
        &train_scaled,
        &binary_labels(&splits.train),
        0.02,
        300,
        1e-3,
    );
    let validation_scores = predict_scores(&model, &validation_scaled);
    let validation_labels = splits
        .validation
        .iter()
        .map(|sample| sample.label_positive)
        .collect::<Vec<_>>();
    let threshold = best_budgeted_threshold(
        &splits.validation,
        &validation_scores,
        &validation_labels,
        grid,
        budget_gib,
    );
    Ok(FittedSparsePolicyProfile {
        name,
        scaler,
        model,
        threshold,
        descriptor_profile,
    })
}

fn thresholded_sparse_policy_summary(
    samples: &[LabeledInvariantSample],
    normalized: &BTreeMap<RowKey, NormalizedSample>,
    normalization_strategy: &str,
    descriptor_profile: Option<DescriptorProfile>,
    grid: usize,
) -> Result<CounterfactualSparseSummary> {
    let policy =
        fit_sparse_budget_policy_profile(samples, normalized, descriptor_profile, grid, 12.0)?;
    let label_index = label_index(samples);
    let time_index = samples
        .iter()
        .map(|sample| (sample.key.clone(), sample.timestamp_utc.clone()))
        .collect::<BTreeMap<_, _>>();
    let summary = sparse_summary_from_active_index(
        &policy.mask,
        samples.len(),
        &label_index,
        &time_index,
        samples,
    );
    let projected_gib = estimate_sparse_execution_plan(
        grid,
        summary.occupancy_tile_fraction,
        None,
        SparseHardwareEnvelope {
            cuda_vram_budget_bytes: None,
            cuda_l2_bytes: None,
            cuda_shared_mem_per_block: None,
            cuda_managed_memory: None,
            cuda_concurrent_managed_access: None,
            cpu_l3_safe_working_set_bytes: None,
            prefer_sparse_tile: false,
        },
    )
    .memory
    .sparse_bf16_aa_projected_gib;
    Ok(CounterfactualSparseSummary {
        normalization_strategy: normalization_strategy.to_string(),
        descriptor_profile: descriptor_profile
            .map(DescriptorProfile::label)
            .unwrap_or("invariants_only")
            .to_string(),
        active_rows: summary.active_rows,
        active_fraction: summary.active_fraction,
        occupancy_tiles_active: summary.occupancy_tiles_active,
        occupancy_tiles_total: summary.occupancy_tiles_total,
        occupancy_tile_fraction: summary.occupancy_tile_fraction,
        event_label_recall: summary.event_label_recall,
        event_label_precision: summary.event_label_precision,
        sparse_bf16_aa_projected_gib: projected_gib,
        median_lead_time_hours: summary.median_lead_time_hours,
    })
}

fn best_budgeted_threshold(
    samples: &[&LabeledInvariantSample],
    scores: &[f64],
    labels: &[bool],
    grid: usize,
    budget_gib: f64,
) -> f64 {
    if scores.is_empty() {
        return 1.0;
    }
    let mut candidates = scores
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    candidates.sort_by(|a, b| a.total_cmp(b));
    candidates.dedup_by(|a, b| (*a - *b).abs() < 1.0e-9);
    if candidates.len() > 96 {
        let step = (candidates.len() / 96).max(1);
        candidates = candidates.into_iter().step_by(step).collect();
    }
    let mut best_under_budget = None::<(f64, f64, f64)>;
    let mut lowest_memory = None::<(f64, f64)>;
    for threshold in candidates {
        let occupancy_tile_fraction =
            occupancy_tile_fraction_for_scores(samples, scores, threshold);
        let projected_gib = estimate_sparse_execution_plan(
            grid,
            occupancy_tile_fraction,
            None,
            SparseHardwareEnvelope {
                cuda_vram_budget_bytes: None,
                cuda_l2_bytes: None,
                cuda_shared_mem_per_block: None,
                cuda_managed_memory: None,
                cuda_concurrent_managed_access: None,
                cpu_l3_safe_working_set_bytes: None,
                prefer_sparse_tile: false,
            },
        )
        .memory
        .sparse_bf16_aa_projected_gib;
        let (_, recall, _, _, _) = threshold_metrics(scores, labels, threshold);
        if projected_gib <= budget_gib {
            match best_under_budget {
                Some((_, best_recall, best_fraction))
                    if recall.total_cmp(&best_recall).is_lt()
                        || (recall.total_cmp(&best_recall).is_eq()
                            && occupancy_tile_fraction.total_cmp(&best_fraction).is_ge()) => {}
                _ => best_under_budget = Some((threshold, recall, occupancy_tile_fraction)),
            }
        }
        match lowest_memory {
            Some((_, best_gib)) if projected_gib.total_cmp(&best_gib).is_ge() => {}
            _ => lowest_memory = Some((threshold, projected_gib)),
        }
    }
    best_under_budget
        .map(|(threshold, _, _)| threshold)
        .or_else(|| lowest_memory.map(|(threshold, _)| threshold))
        .unwrap_or(1.0)
}

fn raw_time_index(rows: &[HeliosphereFeatureRow]) -> BTreeMap<RowKey, String> {
    rows.iter()
        .filter_map(|row| {
            heliosphere_row_datetime(row).map(|timestamp| (row_key(row), timestamp.to_rfc3339()))
        })
        .collect()
}

fn median_mask_lead_time_hours(
    time_index: &BTreeMap<RowKey, String>,
    label_index: &BTreeMap<RowKey, bool>,
    active_index: &BTreeMap<RowKey, bool>,
) -> Option<f64> {
    let mut grouped: BTreeMap<String, Vec<(String, bool, bool)>> = BTreeMap::new();
    for (key, timestamp) in time_index {
        let mission = key.1.clone();
        grouped.entry(mission).or_default().push((
            timestamp.clone(),
            *label_index.get(key).unwrap_or(&false),
            *active_index.get(key).unwrap_or(&false),
        ));
    }
    let mut leads = Vec::new();
    for rows in grouped.values_mut() {
        rows.sort_by(|a, b| a.0.cmp(&b.0));
        for positive_idx in rows
            .iter()
            .enumerate()
            .filter_map(|(idx, (_, positive, _))| (*positive).then_some(idx))
        {
            let event_time = parse_timestamp(&rows[positive_idx].0).ok()?;
            let mut earliest_prediction = None;
            for (timestamp, _positive, active) in rows[..=positive_idx].iter().rev() {
                if *active {
                    earliest_prediction = parse_timestamp(timestamp).ok();
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

// sigmoid / threshold_metrics / best_threshold / auprc / auroc live
// in the `metrics` submodule.
mod metrics;
use metrics::{auprc, auroc, best_threshold, sigmoid, threshold_metrics};

fn best_single_invariant_threshold(
    samples: &[&LabeledInvariantSample],
    view_mode: ViewMode,
    normalized: &BTreeMap<RowKey, NormalizedSample>,
) -> ThresholdedInvariant {
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
            .map(|sample| sample_invariant_value(sample, view_mode, normalized, idx))
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
    feature_mode: &str,
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
        feature_mode: feature_mode.to_string(),
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

fn train_counterfactual_predictive_model(
    splits: &SampleSplits<'_>,
    view_mode: ViewMode,
    normalization_strategy: &str,
    normalized: Option<&BTreeMap<RowKey, NormalizedSample>>,
    descriptor_profile: Option<DescriptorProfile>,
) -> CounterfactualPredictiveSummary {
    let empty = BTreeMap::new();
    let normalized = normalized.unwrap_or(&empty);
    let train = feature_matrix_with_descriptor_profile(
        &splits.train,
        view_mode,
        normalized,
        descriptor_profile,
    );
    let validation = feature_matrix_with_descriptor_profile(
        &splits.validation,
        view_mode,
        normalized,
        descriptor_profile,
    );
    let test = feature_matrix_with_descriptor_profile(
        &splits.test,
        view_mode,
        normalized,
        descriptor_profile,
    );
    let scaler = fit_scaler(&train);
    let train_scaled = apply_scaler(&scaler, &train);
    let validation_scaled = apply_scaler(&scaler, &validation);
    let test_scaled = apply_scaler(&scaler, &test);
    let model = train_logistic_model(
        &train_scaled,
        &binary_labels(&splits.train),
        0.02,
        300,
        1e-3,
    );
    let validation_scores = predict_scores(&model, &validation_scaled);
    let threshold = best_threshold(
        &validation_scores,
        &splits
            .validation
            .iter()
            .map(|sample| sample.label_positive)
            .collect::<Vec<_>>(),
    );
    let metrics = evaluate_scores(
        match view_mode {
            ViewMode::Raw => "raw",
            ViewMode::Normalized => "normalized",
        },
        descriptor_profile
            .map(DescriptorProfile::label)
            .unwrap_or("invariants_only"),
        &splits.test,
        &predict_scores(&model, &test_scaled),
        threshold,
    );
    CounterfactualPredictiveSummary {
        view_mode: metrics.feature_mode,
        normalization_strategy: normalization_strategy.to_string(),
        descriptor_profile: descriptor_profile
            .map(DescriptorProfile::label)
            .unwrap_or("invariants_only")
            .to_string(),
        threshold: metrics.threshold,
        positive_rows: metrics.positive_rows,
        negative_rows: metrics.negative_rows,
        predicted_positive_rows: metrics.predicted_positive_rows,
        auprc: metrics.auprc,
        auroc: metrics.auroc,
        precision: metrics.precision,
        recall: metrics.recall,
        f1: metrics.f1,
        false_alert_rate: metrics.false_alert_rate,
        median_lead_time_hours: metrics.median_lead_time_hours,
    }
}


fn median_lead_time_hours(
    samples: &[&LabeledInvariantSample],
    scores: &[f64],
    threshold: f64,
) -> Option<f64> {
    let mut grouped: BTreeMap<&str, Vec<(&LabeledInvariantSample, f64)>> = BTreeMap::new();
    for (sample, score) in samples.iter().copied().zip(scores.iter().copied()) {
        grouped
            .entry(sample.mission.as_str())
            .or_default()
            .push((sample, score));
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

fn occupancy_tile_totals(samples: &[LabeledInvariantSample]) -> usize {
    samples
        .iter()
        .map(occupancy_tile_key)
        .collect::<BTreeSet<_>>()
        .len()
}

fn occupancy_tile_stats_from_mask(
    samples: &[LabeledInvariantSample],
    active_index: &BTreeMap<RowKey, bool>,
) -> (usize, usize, f64) {
    let total_tiles = occupancy_tile_totals(samples);
    let mut active_tiles = BTreeSet::new();
    for sample in samples {
        if *active_index.get(&sample.key).unwrap_or(&false) {
            active_tiles.insert(occupancy_tile_key(sample));
        }
    }
    let active_count = active_tiles.len();
    (
        active_count,
        total_tiles,
        ratio_usize(active_count, total_tiles.max(1)),
    )
}

fn occupancy_tile_fraction_for_scores(
    samples: &[&LabeledInvariantSample],
    scores: &[f64],
    threshold: f64,
) -> f64 {
    let total_tiles = samples
        .iter()
        .map(|sample| occupancy_tile_key(sample))
        .collect::<BTreeSet<_>>()
        .len();
    let mut active_tiles = BTreeSet::new();
    for (sample, score) in samples.iter().zip(scores.iter()) {
        if score.is_finite() && *score >= threshold {
            active_tiles.insert(occupancy_tile_key(sample));
        }
    }
    ratio_usize(active_tiles.len(), total_tiles.max(1))
}

fn sparse_summary(
    name: &str,
    rows: &[&HeliosphereFeatureRow],
    total_rows: usize,
    label_index: &BTreeMap<RowKey, bool>,
    time_index: &BTreeMap<RowKey, String>,
    active_index: &BTreeMap<RowKey, bool>,
    samples: &[LabeledInvariantSample],
) -> SparseMaskSummary {
    let active_rows = rows.len();
    let labeled_active = rows
        .iter()
        .filter(|row| *label_index.get(&row_key(row)).unwrap_or(&false))
        .count();
    let total_labeled = label_index.values().filter(|value| **value).count();
    let (occupancy_tiles_active, occupancy_tiles_total, occupancy_tile_fraction) =
        occupancy_tile_stats_from_mask(samples, active_index);
    SparseMaskSummary {
        name: name.to_string(),
        active_rows,
        active_fraction: ratio_usize(active_rows, total_rows),
        occupancy_tiles_active,
        occupancy_tiles_total,
        occupancy_tile_fraction,
        event_label_recall: ratio_usize(labeled_active, total_labeled),
        event_label_precision: ratio_usize(labeled_active, active_rows),
        density_mean: mean(
            &rows
                .iter()
                .map(|row| row.density_cm3)
                .filter(|value| value.is_finite())
                .collect::<Vec<_>>(),
        ),
        speed_mean: mean(
            &rows
                .iter()
                .map(|row| row.speed_kms)
                .filter(|value| value.is_finite())
                .collect::<Vec<_>>(),
        ),
        temperature_mean: mean(
            &rows
                .iter()
                .map(|row| row.temperature_k)
                .filter(|value| value.is_finite())
                .collect::<Vec<_>>(),
        ),
        bmag_mean: mean(
            &rows
                .iter()
                .map(|row| row.b_mag)
                .filter(|value| value.is_finite())
                .collect::<Vec<_>>(),
        ),
        median_lead_time_hours: median_mask_lead_time_hours(time_index, label_index, active_index),
    }
}

fn sparse_summary_from_active_index(
    active_index: &BTreeMap<RowKey, bool>,
    total_rows: usize,
    label_index: &BTreeMap<RowKey, bool>,
    time_index: &BTreeMap<RowKey, String>,
    samples: &[LabeledInvariantSample],
) -> SparseMaskSummary {
    let active_rows = active_index.values().filter(|value| **value).count();
    let labeled_active = active_index
        .iter()
        .filter(|(_, active)| **active)
        .filter(|(key, _)| *label_index.get(*key).unwrap_or(&false))
        .count();
    let total_labeled = label_index.values().filter(|value| **value).count();
    let (occupancy_tiles_active, occupancy_tiles_total, occupancy_tile_fraction) =
        occupancy_tile_stats_from_mask(samples, active_index);
    SparseMaskSummary {
        name: "counterfactual".to_string(),
        active_rows,
        active_fraction: ratio_usize(active_rows, total_rows),
        occupancy_tiles_active,
        occupancy_tiles_total,
        occupancy_tile_fraction,
        event_label_recall: ratio_usize(labeled_active, total_labeled),
        event_label_precision: ratio_usize(labeled_active, active_rows),
        density_mean: f64::NAN,
        speed_mean: f64::NAN,
        temperature_mean: f64::NAN,
        bmag_mean: f64::NAN,
        median_lead_time_hours: median_mask_lead_time_hours(time_index, label_index, active_index),
    }
}

fn mean_feature_vector(
    samples: &[&LabeledInvariantSample],
    view_mode: ViewMode,
    normalized: &BTreeMap<RowKey, NormalizedSample>,
) -> Vec<f64> {
    if samples.is_empty() {
        return vec![0.0; HELIOSPHERE_INVARIANT_DIM + DESCRIPTOR_DIM];
    }
    let mut out = vec![0.0; HELIOSPHERE_INVARIANT_DIM + DESCRIPTOR_DIM];
    for sample in samples {
        for (idx, value) in out.iter_mut().enumerate().take(HELIOSPHERE_INVARIANT_DIM) {
            *value += sample_invariant_value(sample, view_mode, normalized, idx);
        }
        for idx in 0..DESCRIPTOR_DIM {
            out[HELIOSPHERE_INVARIANT_DIM + idx] +=
                sample_descriptor_value(sample, view_mode, normalized, idx);
        }
    }
    for value in &mut out {
        *value /= samples.len() as f64;
    }
    out
}

// Pure statistical helpers moved to the `stats` submodule for PH-MOD
// modularization. Re-exported here at the parent module scope so all
// existing call sites in heliosphere_eval.rs continue to resolve them.
mod stats;
use stats::{
    cosine_similarity, finite_mad, finite_median, finite_median_opt, finite_std, l2_norm_sq,
    mean, ratio_usize,
};

// Mask-ops helpers (median_filter_3, hysteresis_mask, dilate_mask,
// merge_small_gaps) live in the `mask_ops` submodule.
mod mask_ops;
use mask_ops::{dilate_mask, hysteresis_mask, median_filter_3, merge_small_gaps};

/// Public channel names for the algebra descriptor extension.
pub const HELIOSPHERE_DESCRIPTOR_CHANNEL_NAMES: [&str; DESCRIPTOR_DIM] = [
    "weighted_norm_sq",
    "delta_weighted_norm_sq",
    "rolling_associator_norm",
    "mean_abs_weighted_channel",
    "takens_sedenion_assoc",
    "takens_octonion_pair",
    "takens_random_orthogonal",
    "takens_euclidean_baseline",
];

fn to_cd16(values: &[f64; HELIOSPHERE_INVARIANT_DIM]) -> [f64; 16] {
    let mut out = [0.0_f64; 16];
    out[..HELIOSPHERE_INVARIANT_DIM].copy_from_slice(values);
    out
}

#[cfg(test)]
pub(crate) fn assert_takens_descriptor_sedenion_lane_matches_scalar_reference() {
    use self::tests::sample;
    use crate::heliosphere_eval::{HasBField, takens_descriptors};

    let group = (0..6).map(sample).collect::<Vec<_>>();
    let descriptor = takens_descriptors(&group, 5);

    let get_v16 = |target_idx: usize| -> [f64; 16] {
        let mut v16 = [0.0; 16];
        for i in 0..4 {
            let s = &group[target_idx - 3 + i];
            v16[i * 4..i * 4 + 4].copy_from_slice(&s.b_field());
        }
        v16
    };
    let a = get_v16(3);
    let b = get_v16(4);
    let c = get_v16(5);
    let expected = cd_kernel::cd_associator_norm(&a, &b, &c);

    assert!((descriptor[0] - expected).abs() < 1.0e-12);
    assert!(descriptor[3] > 0.0);
}

#[cfg(test)]
mod tests {
    use data_core::{HELIOSPHERE_INVARIANT_DIM, HeliosphereInvariantSample};

    pub(super) fn sample(idx: usize) -> HeliosphereInvariantSample {
        let base = idx as f64 + 1.0;
        HeliosphereInvariantSample {
            window_name: "unit".to_string(),
            mission: "voyager1".to_string(),
            product: "bfield".to_string(),
            year: 2000,
            doy: idx as u16,
            hour: idx as u8,
            timestamp_utc: format!("2000-01-{:02}T00:00:00Z", idx + 1),
            channels: [0.0; HELIOSPHERE_INVARIANT_DIM],
            uncertainty_scales: [1.0; HELIOSPHERE_INVARIANT_DIM],
            weighted_channels: [0.0; HELIOSPHERE_INVARIANT_DIM],
            b_field: [base, base + 0.1, base + 0.2, base + 0.3],
            inherited_event_score: None,
            inherited_event_mask: None,
            inherited_event_segment_id: None,
        }
    }

    #[test]
    fn takens_descriptor_sedenion_lane_matches_scalar_reference() {
        super::assert_takens_descriptor_sedenion_lane_matches_scalar_reference();
    }
}

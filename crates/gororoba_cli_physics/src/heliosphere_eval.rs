//! Shared heliosphere evaluation helpers for predictive, invariance, and
//! sparsification experiments.

use anyhow::{Context, Result, bail};
use chrono::Duration;
use csv::ReaderBuilder;
use data_core::{
    HELIOSPHERE_INVARIANT_CHANNEL_NAMES, HELIOSPHERE_INVARIANT_DIM, HeliosphereEventSource,
    HeliosphereFeatureRow, HeliosphereInvariantSample, HeliosphereTransformMode,
    SparseHardwareEnvelope, compute_invariant_samples, estimate_sparse_execution_plan,
    fetch_donki_event_labels, fetch_official_forecast_residuals, heliosphere_row_datetime,
    labels_to_prediction_windows, transform_feature_rows_with_stats,
};
use std::{
    collections::{BTreeMap, BTreeSet},
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
struct ThresholdedInvariant {
    index: usize,
    threshold: f64,
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

// sigmoid / threshold_metrics / best_threshold / auprc / auroc live
// in the `metrics` submodule.
mod metrics;
use metrics::{auprc, auroc, best_threshold, threshold_metrics};

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
    cosine_similarity, finite_mad, finite_median, finite_median_opt, mean, ratio_usize,
};

// Mask-ops helpers (median_filter_3, hysteresis_mask, dilate_mask,
// merge_small_gaps) live in the `mask_ops` submodule.
mod mask_ops;
use mask_ops::{dilate_mask, hysteresis_mask, median_filter_3, merge_small_gaps};

// Occupancy-tile helpers (label_index, occupancy_tile_key,
// occupancy_tile_totals, occupancy_tile_stats_from_mask,
// occupancy_tile_fraction_for_scores) live in the `occupancy` submodule.
mod occupancy;
use occupancy::{label_index, occupancy_tile_stats_from_mask};

// Descriptor-channel helpers (HasBField trait, descriptor_channels,
// takens_descriptors, descriptor_channels_from_arrays, to_cd16) live
// in the `descriptors` submodule.
mod descriptors;
use descriptors::{descriptor_channels, descriptor_channels_from_arrays, takens_descriptors};

// Split helpers (SampleSplits struct, mission_splits, split_samples,
// split_samples_with_seed, seeded_split_rank) live in the `splits`
// submodule.
mod splits;
use splits::{SampleSplits, mission_splits, split_samples, split_samples_with_seed};

// Logistic-regression helpers (ScaledFeatureSet, LogisticModel,
// fit_scaler, apply_scaler, train_logistic_model, predict_scores)
// live in the `logistic` submodule.
mod logistic;
use logistic::{
    LogisticModel, ScaledFeatureSet, apply_scaler, fit_scaler, predict_scores,
    train_logistic_model,
};

// Sample-vector access + normalization helpers
// (fit_normalization_params, normalize_channels, invariant_vector,
// descriptor_vector, selected_descriptor_values, sample_*) live in
// the `vectors` submodule.
mod vectors;
use vectors::{
    fit_normalization_params, invariant_vector, normalize_channels, sample_descriptor_value,
    sample_invariant_norm, sample_invariant_value, selected_descriptor_values,
};

// Time / window / text-key helpers (parse_timestamp, contains_time,
// cube_date_bounds, positive_row_count, normalize_text) live in the
// `windows` submodule.
mod windows;
use windows::{
    contains_time, cube_date_bounds, median_mask_lead_time_hours, normalize_text, parse_timestamp,
    positive_row_count, raw_time_index,
};

// Sparse-budget policy fitters and the budget-aware threshold sweep
// live in the `sparse_policy` submodule.
mod sparse_policy;
use sparse_policy::{
    fit_sparse_budget_policy, fit_sparse_budget_policy_profile,
    fit_sparse_budget_policy_profile_model_with_seed,
};

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

#[cfg(test)]
pub(crate) fn assert_takens_descriptor_sedenion_lane_matches_scalar_reference() {
    use self::descriptors::{HasBField, takens_descriptors};
    use self::tests::sample;

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

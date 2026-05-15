//! Sparse-budget policy fitters and the budget-aware threshold sweep.
//!
//! Pipeline:
//!   * `best_budgeted_threshold`            -- sweep candidate
//!     thresholds and pick the one with the highest recall whose
//!     projected sparse-execution memory fits inside `budget_gib`,
//!     falling back to the lowest-memory threshold otherwise.
//!   * `fit_sparse_budget_policy`           -- invariant- or
//!     hybrid-feature variant; returns a per-sample boolean
//!     activation mask wrapped in a `ThresholdedSparsePolicy`.
//!   * `fit_sparse_budget_policy_profile` / `*_with_seed` /
//!     `*_model_with_seed` -- profile-aware variants that respect a
//!     `DescriptorProfile` selector and produce either a mask or the
//!     full `FittedSparsePolicyProfile` record.
//!
//! All items pub(super); accesses parent's private types directly.

use std::collections::BTreeMap;

use anyhow::Result;
use data_core::{SparseHardwareEnvelope, estimate_sparse_execution_plan};

use super::{
    DescriptorProfile, FeatureMode, FittedSparsePolicyProfile, NormalizedSample, SparsePolicyKind,
    ThresholdedSparsePolicy, ViewMode, binary_labels, feature_matrix,
    feature_matrix_with_descriptor_profile,
    logistic::{apply_scaler, fit_scaler, predict_scores, train_logistic_model},
    metrics::threshold_metrics,
    occupancy::occupancy_tile_fraction_for_scores,
    public_types::{LabeledInvariantSample, RowKey},
    splits::{split_samples, split_samples_with_seed},
};

pub(super) fn best_budgeted_threshold(
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

pub(super) fn fit_sparse_budget_policy(
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

pub(super) fn fit_sparse_budget_policy_profile(
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

pub(super) fn fit_sparse_budget_policy_profile_with_seed(
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

pub(super) fn fit_sparse_budget_policy_profile_model_with_seed(
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

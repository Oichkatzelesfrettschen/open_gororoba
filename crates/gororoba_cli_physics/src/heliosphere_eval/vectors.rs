//! Sample-vector helpers: invariant + descriptor channel access and
//! robust per-channel normalization.
//!
//! Pipeline:
//!   * `fit_normalization_params` -- per-channel medians + robust
//!     sigmas (max of MAD/uncertainty/std/1) from a slice of samples
//!   * `normalize_channels`       -- apply those params to one sample
//!   * `invariant_vector` / `descriptor_vector` -- raw or normalized
//!     view selectors
//!   * `selected_descriptor_values` -- profile-filtered descriptor row
//!   * `sample_invariant_value` / `sample_descriptor_value` -- single-
//!     channel accessors
//!   * `sample_invariant_norm`    -- L2 norm in the current view mode
//!
//! Submodule accesses private types in the parent module (Rust allows
//! child modules to see parent's private items). All exports are
//! `pub(super)`.

use std::collections::BTreeMap;

use data_core::HELIOSPHERE_INVARIANT_DIM;

use super::{
    DESCRIPTOR_DIM, DescriptorProfile, NormalizationParams, NormalizedSample, ViewMode,
    public_types::{LabeledInvariantSample, RowKey},
    stats::{finite_mad, finite_median, finite_median_opt, finite_std, l2_norm_sq},
};

pub(super) fn fit_normalization_params(samples: &[&LabeledInvariantSample]) -> NormalizationParams {
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

pub(super) fn normalize_channels(
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

pub(super) fn invariant_vector(
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

pub(super) fn descriptor_vector(
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

pub(super) fn selected_descriptor_values(
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

pub(super) fn sample_invariant_value(
    sample: &LabeledInvariantSample,
    view_mode: ViewMode,
    normalized: &BTreeMap<RowKey, NormalizedSample>,
    idx: usize,
) -> f64 {
    invariant_vector(sample, view_mode, normalized)[idx]
}

pub(super) fn sample_descriptor_value(
    sample: &LabeledInvariantSample,
    view_mode: ViewMode,
    normalized: &BTreeMap<RowKey, NormalizedSample>,
    idx: usize,
) -> f64 {
    descriptor_vector(sample, view_mode, normalized)[idx]
}

pub(super) fn sample_invariant_norm(
    sample: &LabeledInvariantSample,
    view_mode: ViewMode,
    normalized: &BTreeMap<RowKey, NormalizedSample>,
) -> f64 {
    l2_norm_sq(&invariant_vector(sample, view_mode, normalized)).sqrt()
}

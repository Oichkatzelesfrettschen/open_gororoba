//! Time-series mask operations used by the heliosphere sparse-policy
//! and hysteresis pipelines.
//!
//! `median_filter_3` smooths a numeric series; `hysteresis_mask`
//! converts a value series to an on/off mask with separate trigger
//! and release thresholds; `dilate_mask` widens a bool mask by N
//! neighbors; `merge_small_gaps` fills brief gaps inside an active
//! region. All four are pure; no parent-module dependencies beyond
//! `super::stats::finite_median`.

use super::stats::finite_median;

pub(super) fn median_filter_3(values: &[f64]) -> Vec<f64> {
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

pub(super) fn hysteresis_mask(values: &[f64], on: f64, off: f64) -> Vec<bool> {
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

pub(super) fn dilate_mask(values: &[bool], radius: usize) -> Vec<bool> {
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

pub(super) fn merge_small_gaps(values: &[bool], max_gap: usize) -> Vec<bool> {
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

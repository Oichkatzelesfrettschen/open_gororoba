//! RMS / weighted-RMS statistics over fit-residual rows.
//!
//! Functions:
//!   * `rms_from_iter`              -- unweighted RMS over any iterator
//!   * `optional_rms`               -- None on empty slice, else RMS
//!   * `collect_option_values`      -- flatten `Iterator<Option<f64>>`
//!   * `weighted_rms_from_rows`     -- 1/sigma^2-weighted RMS over
//!     `IndependentRefitRow` (before / after-WLS variants)
//!   * `weighted_rms_from_rows_gls` -- 1/sigma^2-weighted RMS over
//!     the after-GLS residual column
//!   * `closest_residual`           -- pick the wrap-period-shifted
//!     candidate closest to a baseline
//!   * `row_dot`                    -- dot product of a `DMatrix` row
//!     with a `DVector`
//!
//! All items `pub(super)`.

use nalgebra::{DMatrix, DVector};

use super::IndependentRefitRow;

pub(super) fn row_dot(matrix: &DMatrix<f64>, row: usize, coefficients: &DVector<f64>) -> f64 {
    (0..matrix.ncols())
        .map(|col| matrix[(row, col)] * coefficients[col])
        .sum()
}

pub(super) fn closest_residual(candidate: f64, baseline: f64, period_s: f64) -> f64 {
    [-1.0, 0.0, 1.0]
        .into_iter()
        .map(|offset| candidate + offset * period_s)
        .min_by(|left, right| (left - baseline).abs().total_cmp(&(right - baseline).abs()))
        .unwrap_or(candidate)
}

pub(super) fn rms_from_iter(values: impl Iterator<Item = f64>) -> f64 {
    let mut count = 0usize;
    let mut sumsq = 0.0;
    for value in values {
        count += 1;
        sumsq += value * value;
    }
    if count == 0 {
        0.0
    } else {
        (sumsq / count as f64).sqrt()
    }
}

pub(super) fn optional_rms(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some(rms_from_iter(values.iter().copied()))
    }
}

pub(super) fn collect_option_values(values: impl Iterator<Item = Option<f64>>) -> Vec<f64> {
    values.flatten().collect::<Vec<_>>()
}

pub(super) fn weighted_rms_from_rows(rows: &[IndependentRefitRow], before: bool) -> f64 {
    let mut weighted_sum = 0.0;
    let mut total_weight = 0.0;
    for row in rows {
        let sigma = (row.uncertainty_us * 1.0e-6).max(1.0e-18);
        let residual = if before {
            row.residual_before_us * 1.0e-6
        } else {
            row.residual_after_wls_us * 1.0e-6
        };
        let weight = 1.0 / (sigma * sigma);
        weighted_sum += weight * residual * residual;
        total_weight += weight;
    }
    if total_weight == 0.0 {
        0.0
    } else {
        (weighted_sum / total_weight).sqrt() * 1.0e6
    }
}

pub(super) fn weighted_rms_from_rows_gls(rows: &[IndependentRefitRow]) -> f64 {
    let mut weighted_sum = 0.0;
    let mut total_weight = 0.0;
    for row in rows {
        let sigma = (row.uncertainty_us * 1.0e-6).max(1.0e-18);
        let residual = row.residual_after_gls_us * 1.0e-6;
        let weight = 1.0 / (sigma * sigma);
        weighted_sum += weight * residual * residual;
        total_weight += weight;
    }
    if total_weight == 0.0 {
        0.0
    } else {
        (weighted_sum / total_weight).sqrt() * 1.0e6
    }
}

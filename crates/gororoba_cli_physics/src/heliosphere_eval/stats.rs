//! Pure statistical helpers used by the heliosphere evaluation pipeline.
//!
//! Each function filters non-finite values and returns a sensible
//! sentinel (NaN, 0.0, or 1.0) for empty inputs, matching the
//! original inline definitions. No dependencies on the parent module.

pub(super) fn ratio_usize(num: usize, denom: usize) -> f64 {
    if denom == 0 {
        0.0
    } else {
        num as f64 / denom as f64
    }
}

pub(super) fn mean(values: &[f64]) -> f64 {
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

pub(super) fn finite_std(values: &[f64], center: f64) -> f64 {
    let finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return 1.0;
    }
    let variance = finite
        .iter()
        .map(|value| {
            let delta = *value - center;
            delta * delta
        })
        .sum::<f64>()
        / finite.len() as f64;
    let std = variance.sqrt();
    if std.is_finite() && std > 1.0e-6 {
        std
    } else {
        1.0
    }
}

pub(super) fn finite_median(values: &[f64]) -> f64 {
    finite_median_opt(values).unwrap_or(0.0)
}

pub(super) fn finite_median_opt(values: &[f64]) -> Option<f64> {
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

pub(super) fn finite_mad(values: &[f64], median: f64) -> f64 {
    let deviations = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .map(|value| (value - median).abs())
        .collect::<Vec<_>>();
    finite_median(&deviations)
}

pub(super) fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>();
    let norm_a = a.iter().map(|value| value * value).sum::<f64>().sqrt();
    let norm_b = b.iter().map(|value| value * value).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

pub(super) fn l2_norm_sq<const N: usize>(values: &[f64; N]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>()
}

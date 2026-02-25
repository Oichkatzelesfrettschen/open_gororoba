//! Quality gates for simulation time-series acceptance.

use thiserror::Error;

#[derive(Debug, Clone, Copy)]
pub struct RhoQualityThresholds {
    pub max_abs_drift_final: f64,
    pub max_std_dev: f64,
}

impl Default for RhoQualityThresholds {
    fn default() -> Self {
        Self {
            max_abs_drift_final: 5.0e-3,
            max_std_dev: 5.0e-3,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RhoTraceQuality {
    pub sample_count: usize,
    pub finite_count: usize,
    pub nan_count: usize,
    pub inf_count: usize,
    pub first_non_finite_index: Option<usize>,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub final_value: f64,
    pub abs_drift_final: f64,
    pub max_abs_drift_from_one: f64,
}

impl RhoTraceQuality {
    pub fn all_finite(&self) -> bool {
        self.finite_count == self.sample_count
    }
}

#[derive(Debug, Error)]
pub enum RhoQualityError {
    #[error("rho trace is empty")]
    EmptyTrace,
    #[error(
        "rho trace has non-finite values: finite={finite_count}/{sample_count}, nan={nan_count}, inf={inf_count}, first_bad_index={first_non_finite_index:?}"
    )]
    NonFinite {
        sample_count: usize,
        finite_count: usize,
        nan_count: usize,
        inf_count: usize,
        first_non_finite_index: Option<usize>,
    },
    #[error(
        "final density drift too large: abs_drift_final={abs_drift_final:.6e} > threshold={threshold:.6e}"
    )]
    DriftTooLarge {
        abs_drift_final: f64,
        threshold: f64,
    },
    #[error("density std-dev too large: std_dev={std_dev:.6e} > threshold={threshold:.6e}")]
    StdDevTooLarge { std_dev: f64, threshold: f64 },
}

pub fn assess_rho_trace(rho: &[f64]) -> RhoTraceQuality {
    let sample_count = rho.len();
    if sample_count == 0 {
        return RhoTraceQuality {
            sample_count: 0,
            finite_count: 0,
            nan_count: 0,
            inf_count: 0,
            first_non_finite_index: None,
            min: f64::NAN,
            max: f64::NAN,
            mean: f64::NAN,
            std_dev: f64::NAN,
            final_value: f64::NAN,
            abs_drift_final: f64::NAN,
            max_abs_drift_from_one: f64::NAN,
        };
    }

    let mut finite_count = 0usize;
    let mut nan_count = 0usize;
    let mut inf_count = 0usize;
    let mut first_non_finite_index = None;

    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut max_abs_drift_from_one = 0.0f64;

    for (idx, &v) in rho.iter().enumerate() {
        if v.is_finite() {
            finite_count += 1;
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
            sum += v;
            let drift = (v - 1.0).abs();
            if drift > max_abs_drift_from_one {
                max_abs_drift_from_one = drift;
            }
        } else {
            if v.is_nan() {
                nan_count += 1;
            } else {
                inf_count += 1;
            }
            if first_non_finite_index.is_none() {
                first_non_finite_index = Some(idx);
            }
        }
    }

    let mean = if finite_count > 0 {
        sum / finite_count as f64
    } else {
        f64::NAN
    };
    let mut variance_sum = 0.0f64;
    if finite_count > 0 {
        for &v in rho {
            if v.is_finite() {
                let d = v - mean;
                variance_sum += d * d;
            }
        }
    }
    let std_dev = if finite_count > 0 {
        (variance_sum / finite_count as f64).sqrt()
    } else {
        f64::NAN
    };

    let final_value = rho[rho.len() - 1];
    let abs_drift_final = (final_value - 1.0).abs();

    if finite_count == 0 {
        min = f64::NAN;
        max = f64::NAN;
        max_abs_drift_from_one = f64::NAN;
    }

    RhoTraceQuality {
        sample_count,
        finite_count,
        nan_count,
        inf_count,
        first_non_finite_index,
        min,
        max,
        mean,
        std_dev,
        final_value,
        abs_drift_final,
        max_abs_drift_from_one,
    }
}

pub fn validate_rho_trace(
    rho: &[f64],
    thresholds: RhoQualityThresholds,
) -> Result<RhoTraceQuality, RhoQualityError> {
    if rho.is_empty() {
        return Err(RhoQualityError::EmptyTrace);
    }

    let quality = assess_rho_trace(rho);
    if !quality.all_finite() {
        return Err(RhoQualityError::NonFinite {
            sample_count: quality.sample_count,
            finite_count: quality.finite_count,
            nan_count: quality.nan_count,
            inf_count: quality.inf_count,
            first_non_finite_index: quality.first_non_finite_index,
        });
    }
    if quality.abs_drift_final > thresholds.max_abs_drift_final {
        return Err(RhoQualityError::DriftTooLarge {
            abs_drift_final: quality.abs_drift_final,
            threshold: thresholds.max_abs_drift_final,
        });
    }
    if quality.std_dev > thresholds.max_std_dev {
        return Err(RhoQualityError::StdDevTooLarge {
            std_dev: quality.std_dev,
            threshold: thresholds.max_std_dev,
        });
    }

    Ok(quality)
}

#[derive(Debug, Clone, Copy)]
pub struct ScalarTraceThresholds {
    pub min_abs_max: f64,
    pub min_std_dev: f64,
}

impl Default for ScalarTraceThresholds {
    fn default() -> Self {
        Self {
            min_abs_max: 1.0e-12,
            min_std_dev: 1.0e-12,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScalarTraceQuality {
    pub sample_count: usize,
    pub finite_count: usize,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub max_abs_value: f64,
}

#[derive(Debug, Error)]
pub enum ScalarTraceError {
    #[error("{name} trace is empty")]
    Empty { name: String },
    #[error("{name} trace has non-finite values: finite={finite_count}/{sample_count}")]
    NonFinite {
        name: String,
        sample_count: usize,
        finite_count: usize,
    },
    #[error(
        "{name} trace appears trivial: max_abs_value={max_abs_value:.6e} < threshold={threshold:.6e}"
    )]
    MaxAbsTooSmall {
        name: String,
        max_abs_value: f64,
        threshold: f64,
    },
    #[error(
        "{name} trace appears nearly constant: std_dev={std_dev:.6e} < threshold={threshold:.6e}"
    )]
    StdDevTooSmall {
        name: String,
        std_dev: f64,
        threshold: f64,
    },
}

pub fn assess_scalar_trace(values: &[f64]) -> ScalarTraceQuality {
    let sample_count = values.len();
    if sample_count == 0 {
        return ScalarTraceQuality {
            sample_count: 0,
            finite_count: 0,
            min: f64::NAN,
            max: f64::NAN,
            mean: f64::NAN,
            std_dev: f64::NAN,
            max_abs_value: f64::NAN,
        };
    }

    let mut finite_count = 0usize;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut max_abs_value = 0.0f64;
    for &v in values {
        if v.is_finite() {
            finite_count += 1;
            min = min.min(v);
            max = max.max(v);
            sum += v;
            max_abs_value = max_abs_value.max(v.abs());
        }
    }

    let mean = if finite_count > 0 {
        sum / finite_count as f64
    } else {
        f64::NAN
    };
    let mut var_sum = 0.0f64;
    if finite_count > 0 {
        for &v in values {
            if v.is_finite() {
                let d = v - mean;
                var_sum += d * d;
            }
        }
    }
    let std_dev = if finite_count > 0 {
        (var_sum / finite_count as f64).sqrt()
    } else {
        f64::NAN
    };

    if finite_count == 0 {
        min = f64::NAN;
        max = f64::NAN;
        max_abs_value = f64::NAN;
    }

    ScalarTraceQuality {
        sample_count,
        finite_count,
        min,
        max,
        mean,
        std_dev,
        max_abs_value,
    }
}

pub fn validate_scalar_trace_signal(
    name: &str,
    values: &[f64],
    thresholds: ScalarTraceThresholds,
) -> Result<ScalarTraceQuality, ScalarTraceError> {
    if values.is_empty() {
        return Err(ScalarTraceError::Empty {
            name: name.to_string(),
        });
    }

    let quality = assess_scalar_trace(values);
    if quality.finite_count != quality.sample_count {
        return Err(ScalarTraceError::NonFinite {
            name: name.to_string(),
            sample_count: quality.sample_count,
            finite_count: quality.finite_count,
        });
    }
    if quality.max_abs_value < thresholds.min_abs_max {
        return Err(ScalarTraceError::MaxAbsTooSmall {
            name: name.to_string(),
            max_abs_value: quality.max_abs_value,
            threshold: thresholds.min_abs_max,
        });
    }
    if quality.std_dev < thresholds.min_std_dev {
        return Err(ScalarTraceError::StdDevTooSmall {
            name: name.to_string(),
            std_dev: quality.std_dev,
            threshold: thresholds.min_std_dev,
        });
    }
    Ok(quality)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_detects_non_finite_values() {
        let q = assess_rho_trace(&[1.0, f64::NAN, f64::INFINITY, 1.0]);
        assert_eq!(q.sample_count, 4);
        assert_eq!(q.finite_count, 2);
        assert_eq!(q.nan_count, 1);
        assert_eq!(q.inf_count, 1);
        assert_eq!(q.first_non_finite_index, Some(1));
    }

    #[test]
    fn quality_accepts_stable_trace() {
        let thresholds = RhoQualityThresholds::default();
        let q = validate_rho_trace(&[1.0, 1.0001, 0.9999, 1.0], thresholds)
            .expect("stable trace should pass");
        assert!(q.all_finite());
        assert!(q.abs_drift_final <= thresholds.max_abs_drift_final);
    }

    #[test]
    fn quality_rejects_large_drift() {
        let thresholds = RhoQualityThresholds {
            max_abs_drift_final: 1.0e-4,
            max_std_dev: 1.0,
        };
        let err = validate_rho_trace(&[1.0, 1.0, 1.01], thresholds).expect_err("drift must fail");
        assert!(matches!(err, RhoQualityError::DriftTooLarge { .. }));
    }

    #[test]
    fn scalar_trace_detects_nontrivial_signal() {
        let q = validate_scalar_trace_signal(
            "u_rms",
            &[0.01, 0.011, 0.012, 0.0115],
            ScalarTraceThresholds::default(),
        )
        .expect("signal should pass");
        assert!(q.max_abs_value > 0.0);
        assert!(q.std_dev > 0.0);
    }

    #[test]
    fn scalar_trace_rejects_constant_zero() {
        let err = validate_scalar_trace_signal(
            "enstrophy",
            &[0.0, 0.0, 0.0, 0.0],
            ScalarTraceThresholds::default(),
        )
        .expect_err("constant zero should fail");
        assert!(matches!(err, ScalarTraceError::MaxAbsTooSmall { .. }));
    }
}

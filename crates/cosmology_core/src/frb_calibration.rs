//! Source-level empirical calibration of a frozen distance predictor.
//!
//! Split conformal intervals provide marginal coverage for exchangeable sources
//! when predictor fitting and selection precede calibration. Repeated bursts from
//! one source are not independent calibration observations. Calibration sources
//! and test sources must be disjoint. Coverage on a localized discovery sample
//! does not establish transport to another survey or conditional coverage at a
//! particular sky position, dispersion measure, or redshift.
//!
//! Scores are absolute comoving-distance errors in Mpc. Positive infinity records
//! a failed prediction without deleting its source. An unbounded interval records
//! inadequate finite calibration information rather than physical precision.
//! Operational coverage counts failed test predictions as uncovered. Infinite
//! calibration scores alone do not establish conformal coverage for a pipeline
//! that fails to construct prediction sets for some test sources.

use statrs::distribution::{Beta, Binomial, ContinuousCDF, DiscreteCDF};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CalibrationError {
    InvalidCoverage,
    InvalidScore,
    InvalidPrediction,
    InvalidCounts,
    NumericalFailure,
}

impl std::fmt::Display for CalibrationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::InvalidCoverage => "coverage must be finite and strictly between zero and one",
            Self::InvalidScore => {
                "absolute error must be nonnegative and finite or positive infinity"
            }
            Self::InvalidPrediction => "predicted distance must be finite and nonnegative",
            Self::InvalidCounts => {
                "coverage counts require 0 <= covered <= total, with positive total representable exactly as f64"
            }
            Self::NumericalFailure => "calibration calculation exceeds finite numerical precision",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for CalibrationError {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConformalQuantile {
    Finite { radius_mpc: f64 },
    Unbounded,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DistancePredictionInterval {
    Finite { lower_mpc: f64, upper_mpc: f64 },
    Unbounded,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplitConformalCalibration {
    count: usize,
    rank: usize,
    coverage: f64,
    quantile: ConformalQuantile,
}

impl SplitConformalCalibration {
    pub fn count(&self) -> usize {
        self.count
    }

    /// One-based order-statistic rank, including the implicit infinite score.
    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn coverage(&self) -> f64 {
        self.coverage
    }

    pub fn quantile(&self) -> ConformalQuantile {
        self.quantile
    }

    /// Construct a nonnegative distance interval around a valid frozen prediction.
    /// Overflow is a numerical error, distinct from an unbounded conformal rank.
    pub fn interval(
        &self,
        predicted_mpc: f64,
    ) -> Result<DistancePredictionInterval, CalibrationError> {
        if !predicted_mpc.is_finite() || predicted_mpc < 0.0 {
            return Err(CalibrationError::InvalidPrediction);
        }
        match self.quantile {
            ConformalQuantile::Unbounded => Ok(DistancePredictionInterval::Unbounded),
            ConformalQuantile::Finite { radius_mpc } => {
                let upper_mpc = predicted_mpc + radius_mpc;
                if !upper_mpc.is_finite() {
                    return Err(CalibrationError::NumericalFailure);
                }
                Ok(DistancePredictionInterval::Finite {
                    lower_mpc: (predicted_mpc - radius_mpc).max(0.0),
                    upper_mpc,
                })
            }
        }
    }
}

fn validate_coverage(coverage: f64) -> Result<(), CalibrationError> {
    if coverage.is_finite() && coverage > 0.0 && coverage < 1.0 {
        Ok(())
    } else {
        Err(CalibrationError::InvalidCoverage)
    }
}

/// Select ceil((n+1)*coverage), retaining ties and failed predictions in n.
/// An empty sample, an order statistic beyond n, or an infinite selected score
/// yields an unbounded quantile. The predictor must remain fixed independently
/// of calibration scores; quantile selection performs no fitting or source grouping.
pub fn calibrate_absolute_errors(
    scores_mpc: &[f64],
    coverage: f64,
) -> Result<SplitConformalCalibration, CalibrationError> {
    validate_coverage(coverage)?;
    if scores_mpc
        .iter()
        .any(|score| score.is_nan() || *score < 0.0)
    {
        return Err(CalibrationError::InvalidScore);
    }
    let augmented_count = scores_mpc
        .len()
        .checked_add(1)
        .ok_or(CalibrationError::InvalidCounts)?;
    if augmented_count as u128 > (1_u128 << 53) {
        return Err(CalibrationError::InvalidCounts);
    }
    let rank = ((augmented_count as f64) * coverage).ceil() as usize;
    let quantile = if rank > scores_mpc.len() {
        ConformalQuantile::Unbounded
    } else {
        let mut sorted = scores_mpc.to_vec();
        sorted.sort_by(f64::total_cmp);
        let selected = sorted[rank - 1];
        if selected.is_infinite() {
            ConformalQuantile::Unbounded
        } else {
            ConformalQuantile::Finite {
                radius_mpc: selected,
            }
        }
    };
    Ok(SplitConformalCalibration {
        count: scores_mpc.len(),
        rank,
        coverage,
        quantile,
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoverageAssessment {
    pub covered: u64,
    pub total: u64,
    pub empirical_coverage: f64,
    pub undercoverage_p_value: f64,
    pub clopper_pearson_lower: f64,
    pub clopper_pearson_upper: f64,
}

/// Assess coverage with a fixed source denominator; failed predictions count
/// as uncovered. The one-sided exact p-value is P[Binomial(total, nominal) <=
/// covered]. The confidence limits are two-sided 95% Clopper-Pearson limits.
/// Binomial interpretation requires independent sources with a common coverage
/// probability, conditional on the frozen calibration artifact. Shared selection
/// errors or repeat-source dependence require a separately justified model.
/// Rejecting nominal coverage for a frozen artifact does not refute the split
/// conformal guarantee, which is marginal over calibration samples.
pub fn assess_coverage(
    covered: u64,
    total: u64,
    nominal_coverage: f64,
) -> Result<CoverageAssessment, CalibrationError> {
    validate_coverage(nominal_coverage)?;
    if total == 0 || covered > total || total >= (1_u64 << 53) {
        return Err(CalibrationError::InvalidCounts);
    }
    let binomial =
        Binomial::new(nominal_coverage, total).map_err(|_| CalibrationError::NumericalFailure)?;
    let undercoverage_p_value = binomial.cdf(covered);
    let clopper_pearson_lower = if covered == 0 {
        0.0
    } else {
        Beta::new(covered as f64, (total - covered + 1) as f64)
            .map_err(|_| CalibrationError::NumericalFailure)?
            .inverse_cdf(0.025)
    };
    let clopper_pearson_upper = if covered == total {
        1.0
    } else {
        Beta::new((covered + 1) as f64, (total - covered) as f64)
            .map_err(|_| CalibrationError::NumericalFailure)?
            .inverse_cdf(0.975)
    };
    if [
        undercoverage_p_value,
        clopper_pearson_lower,
        clopper_pearson_upper,
    ]
    .iter()
    .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        || clopper_pearson_lower > clopper_pearson_upper
    {
        return Err(CalibrationError::NumericalFailure);
    }
    Ok(CoverageAssessment {
        covered,
        total,
        empirical_coverage: covered as f64 / total as f64,
        undercoverage_p_value,
        clopper_pearson_lower,
        clopper_pearson_upper,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_ties_and_intervals_match_hand_calculation() {
        let calibration = calibrate_absolute_errors(&[4.0, 1.0, 2.0, 2.0], 0.6).unwrap();
        assert_eq!(calibration.count(), 4);
        assert_eq!(calibration.rank(), 3);
        assert_eq!(calibration.coverage(), 0.6);
        assert_eq!(
            calibration.quantile(),
            ConformalQuantile::Finite { radius_mpc: 2.0 }
        );
        assert_eq!(
            calibration.interval(1.0).unwrap(),
            DistancePredictionInterval::Finite {
                lower_mpc: 0.0,
                upper_mpc: 3.0
            }
        );
        assert_eq!(
            calibration.interval(5.0).unwrap(),
            DistancePredictionInterval::Finite {
                lower_mpc: 3.0,
                upper_mpc: 7.0
            }
        );
        assert_eq!(
            calibrate_absolute_errors(&[2.0, 4.0, 2.0, 1.0], 0.6).unwrap(),
            calibration
        );
    }

    #[test]
    fn insufficient_and_infinite_scores_are_unbounded() {
        for scores in [vec![], vec![0.0; 8], vec![f64::INFINITY; 9]] {
            let calibration = calibrate_absolute_errors(&scores, 0.9).unwrap();
            assert_eq!(calibration.quantile(), ConformalQuantile::Unbounded);
            assert_eq!(
                calibration.interval(10.0).unwrap(),
                DistancePredictionInterval::Unbounded
            );
        }
        let finite = calibrate_absolute_errors(&[0.0; 9], 0.9).unwrap();
        assert_eq!(finite.rank(), 9);
        assert_eq!(
            finite.quantile(),
            ConformalQuantile::Finite { radius_mpc: 0.0 }
        );
        let scores = [1.0, 2.0, f64::INFINITY];
        assert_eq!(calibrate_absolute_errors(&scores, 0.5).unwrap().count(), 3);
        assert_eq!(
            calibrate_absolute_errors(&scores, 0.5).unwrap().quantile(),
            ConformalQuantile::Finite { radius_mpc: 2.0 }
        );
    }

    #[test]
    fn higher_coverage_never_narrows_quantile() {
        let scores = [3.0, 1.0, 8.0, 2.0, 5.0, 13.0, 0.0, 21.0, 1.0];
        let mut previous = 0.0;
        for coverage in [0.1, 0.5, 0.8, 0.9, 0.95, 0.99] {
            let radius = match calibrate_absolute_errors(&scores, coverage)
                .unwrap()
                .quantile()
            {
                ConformalQuantile::Finite { radius_mpc } => radius_mpc,
                ConformalQuantile::Unbounded => f64::INFINITY,
            };
            assert!(radius >= previous);
            previous = radius;
        }
    }

    #[test]
    fn invalid_values_and_interval_overflow_are_explicit() {
        for coverage in [0.0, 1.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                calibrate_absolute_errors(&[], coverage),
                Err(CalibrationError::InvalidCoverage)
            );
            assert_eq!(
                assess_coverage(0, 1, coverage),
                Err(CalibrationError::InvalidCoverage)
            );
        }
        for score in [-1.0, f64::NEG_INFINITY, f64::NAN] {
            assert_eq!(
                calibrate_absolute_errors(&[score], 0.5),
                Err(CalibrationError::InvalidScore)
            );
        }
        let calibration = calibrate_absolute_errors(&[f64::MAX], 0.5).unwrap();
        for prediction in [-1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                calibration.interval(prediction),
                Err(CalibrationError::InvalidPrediction)
            );
        }
        assert_eq!(
            calibration.interval(f64::MAX),
            Err(CalibrationError::NumericalFailure)
        );
        for (covered, total) in [(0, 0), (2, 1), (0, 1_u64 << 53)] {
            assert_eq!(
                assess_coverage(covered, total, 0.9),
                Err(CalibrationError::InvalidCounts)
            );
        }
    }

    #[test]
    fn exact_binomial_and_confidence_limits_match_analytic_cases() {
        let empty_success = assess_coverage(0, 1, 0.9).unwrap();
        assert!((empty_success.undercoverage_p_value - 0.1).abs() < 1e-14);
        assert_eq!(empty_success.clopper_pearson_lower, 0.0);
        assert!((empty_success.clopper_pearson_upper - 0.975).abs() < 1e-12);
        let success = assess_coverage(1, 1, 0.9).unwrap();
        assert_eq!(success.undercoverage_p_value, 1.0);
        assert!((success.clopper_pearson_lower - 0.025).abs() < 1e-12);
        assert_eq!(success.clopper_pearson_upper, 1.0);
        let half = assess_coverage(1, 2, 0.5).unwrap();
        assert!((half.undercoverage_p_value - 0.75).abs() < 1e-14);
        assert!((half.clopper_pearson_lower - (1.0 - 0.975_f64.sqrt())).abs() < 1e-12);
        assert!((half.clopper_pearson_upper - 0.975_f64.sqrt()).abs() < 1e-12);
        let failures_retained = assess_coverage(8, 10, 0.9).unwrap();
        assert_eq!(failures_retained.total, 10);
        assert_eq!(failures_retained.empirical_coverage, 0.8);
        let expected = 1.0 - 0.9_f64.powi(10) - 10.0 * 0.9_f64.powi(9) * 0.1;
        assert!((failures_retained.undercoverage_p_value - expected).abs() < 1e-12);
    }
}

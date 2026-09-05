//! Paired detector utility with separate decision and event exposure units.

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

/// Counts share labels, exposure, operating policy, and matching rules across detectors.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectionCounts {
    pub true_positives: u64,
    pub false_positives: u64,
}

/// Event counts require one-to-one matching; unmatched duplicate alerts count as false positives.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Accounting {
    Sample {
        decisions: u64,
        positive_decisions: u64,
        baseline: DetectionCounts,
        augmented: DetectionCounts,
    },
    Event {
        exposure_hours: f64,
        true_events: u64,
        baseline: DetectionCounts,
        augmented: DetectionCounts,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum UtilityUnit {
    PerDecision,
    PerExposureHour,
}

/// Each draw resamples baseline and augmentation together with their common exposure.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PairedDraw {
    pub accounting: Accounting,
    /// Signed incremental overhead in the same value units as detection benefit.
    pub additional_overhead: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UtilityInput {
    pub accounting: Accounting,
    pub benefit_per_true_detection: f64,
    /// Total incremental cost over the accounting exposure; savings are negative.
    pub additional_overhead: f64,
    pub paired_draws: Option<Vec<PairedDraw>>,
}

/// Normalized utility is `a - r * b - k`, where `r` is false-alert cost / benefit.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Coefficients {
    pub a: f64,
    pub b: f64,
    pub k: f64,
}

/// Strictly positive utility regions over the domain `cost_ratio >= 0`.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RatioFrontier {
    AllPositive,
    NoPositive { zero_at: Option<f64> },
    BreakEvenEverywhere,
    PositiveBelow { exclusive_ratio: f64 },
    PositiveAbove { exclusive_ratio: f64 },
}

/// Empirical paired-draw percentiles; coverage depends on the supplied resampling design.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct UtilityInterval {
    pub lower: f64,
    pub upper: f64,
    pub draws: usize,
    pub lower_quantile: f64,
    pub upper_quantile: f64,
}

fn exact_count(value: u64) -> Result<f64> {
    ensure!(
        value <= (1_u64 << 53),
        "count exceeds exact f64 integer range"
    );
    Ok(value as f64)
}

impl Accounting {
    pub fn unit(&self) -> UtilityUnit {
        match self {
            Self::Sample { .. } => UtilityUnit::PerDecision,
            Self::Event { .. } => UtilityUnit::PerExposureHour,
        }
    }

    fn coefficients(&self, benefit: f64, overhead: f64) -> Result<Coefficients> {
        ensure!(
            benefit.is_finite() && benefit > 0.0,
            "benefit must be finite and positive"
        );
        ensure!(overhead.is_finite(), "overhead must be finite");
        let (denominator, positives, negatives, baseline, augmented) = match self {
            Self::Sample {
                decisions,
                positive_decisions,
                baseline,
                augmented,
            } => {
                ensure!(
                    *decisions > 0 && positive_decisions <= decisions,
                    "invalid sample exposure or positive count"
                );
                (
                    exact_count(*decisions)?,
                    *positive_decisions,
                    Some(decisions - positive_decisions),
                    baseline,
                    augmented,
                )
            }
            Self::Event {
                exposure_hours,
                true_events,
                baseline,
                augmented,
            } => {
                ensure!(
                    exposure_hours.is_finite() && *exposure_hours > 0.0,
                    "event exposure must be finite and positive"
                );
                (*exposure_hours, *true_events, None, baseline, augmented)
            }
        };
        exact_count(positives)?;
        for counts in [baseline, augmented] {
            ensure!(
                counts.true_positives <= positives,
                "true positives exceed common true-event or positive-decision count"
            );
            if let Some(negative_count) = negatives {
                ensure!(
                    counts.false_positives <= negative_count,
                    "false positives exceed negative decisions"
                );
            }
        }
        let coefficients = Coefficients {
            a: (exact_count(augmented.true_positives)? - exact_count(baseline.true_positives)?)
                / denominator,
            b: (exact_count(augmented.false_positives)? - exact_count(baseline.false_positives)?)
                / denominator,
            k: (overhead / benefit) / denominator,
        };
        coefficients.validate()?;
        Ok(coefficients)
    }
}

impl Coefficients {
    fn validate(&self) -> Result<()> {
        ensure!(
            [self.a, self.b, self.k]
                .iter()
                .all(|value| value.is_finite()),
            "utility coefficients must be finite"
        );
        Ok(())
    }

    pub fn evaluate(&self, cost_ratio: f64, normalized_overhead_shift: f64) -> Result<f64> {
        self.validate()?;
        ensure!(
            cost_ratio.is_finite() && cost_ratio >= 0.0,
            "cost ratio must be finite and nonnegative"
        );
        ensure!(
            normalized_overhead_shift.is_finite(),
            "overhead shift must be finite"
        );
        let utility = self.a - cost_ratio * self.b - (self.k + normalized_overhead_shift);
        ensure!(utility.is_finite(), "utility arithmetic overflow");
        Ok(utility)
    }

    pub fn ratio_frontier(&self, normalized_overhead_shift: f64) -> Result<RatioFrontier> {
        let intercept = self.evaluate(0.0, normalized_overhead_shift)?;
        if self.b == 0.0 {
            return Ok(if intercept > 0.0 {
                RatioFrontier::AllPositive
            } else if intercept < 0.0 {
                RatioFrontier::NoPositive { zero_at: None }
            } else {
                RatioFrontier::BreakEvenEverywhere
            });
        }
        let boundary = intercept / self.b;
        ensure!(boundary.is_finite(), "ratio frontier arithmetic overflow");
        Ok(if self.b > 0.0 {
            if boundary > 0.0 {
                RatioFrontier::PositiveBelow {
                    exclusive_ratio: boundary,
                }
            } else {
                RatioFrontier::NoPositive {
                    zero_at: (boundary == 0.0).then_some(0.0),
                }
            }
        } else if boundary < 0.0 {
            RatioFrontier::AllPositive
        } else {
            RatioFrontier::PositiveAbove {
                exclusive_ratio: boundary,
            }
        })
    }
}

impl UtilityInput {
    pub fn unit(&self) -> UtilityUnit {
        self.accounting.unit()
    }

    pub fn coefficients(&self) -> Result<Coefficients> {
        self.accounting
            .coefficients(self.benefit_per_true_detection, self.additional_overhead)
    }

    pub fn evaluate(&self, cost_ratio: f64, normalized_overhead_shift: f64) -> Result<f64> {
        self.coefficients()?
            .evaluate(cost_ratio, normalized_overhead_shift)
    }

    pub fn ratio_frontier(&self, normalized_overhead_shift: f64) -> Result<RatioFrontier> {
        self.coefficients()?
            .ratio_frontier(normalized_overhead_shift)
    }

    /// An additive overhead scenario shifts every draw, preserving paired overhead variation.
    pub fn interval(
        &self,
        cost_ratio: f64,
        normalized_overhead_shift: f64,
    ) -> Result<Option<UtilityInterval>> {
        self.evaluate(cost_ratio, normalized_overhead_shift)?;
        let Some(draws) = &self.paired_draws else {
            return Ok(None);
        };
        ensure!(
            draws.len() >= 2,
            "uncertainty requires at least two paired draws"
        );
        let mut values = Vec::with_capacity(draws.len());
        for draw in draws {
            ensure!(
                draw.accounting.unit() == self.unit(),
                "paired draw accounting units differ"
            );
            values.push(
                draw.accounting
                    .coefficients(self.benefit_per_true_detection, draw.additional_overhead)?
                    .evaluate(cost_ratio, normalized_overhead_shift)?,
            );
        }
        values.sort_by(f64::total_cmp);
        let percentile = |quantile: f64| {
            let position = quantile * (values.len() - 1) as f64;
            let lower = position.floor() as usize;
            let upper = position.ceil() as usize;
            let fraction = position - lower as f64;
            values[lower] * (1.0 - fraction) + values[upper] * fraction
        };
        let interval = UtilityInterval {
            lower: percentile(0.025),
            upper: percentile(0.975),
            draws: values.len(),
            lower_quantile: 0.025,
            upper_quantile: 0.975,
        };
        ensure!(
            interval.lower.is_finite() && interval.upper.is_finite(),
            "utility percentile arithmetic overflow"
        );
        Ok(Some(interval))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counts(true_positives: u64, false_positives: u64) -> DetectionCounts {
        DetectionCounts {
            true_positives,
            false_positives,
        }
    }

    fn event() -> UtilityInput {
        UtilityInput {
            accounting: Accounting::Event {
                exposure_hours: 2.0,
                true_events: 5,
                baseline: counts(2, 4),
                augmented: counts(4, 2),
            },
            benefit_per_true_detection: 10.0,
            additional_overhead: 10.0,
            paired_draws: None,
        }
    }

    #[test]
    fn units_and_signed_overhead_match_direct_total_utility() {
        let input = event();
        assert_eq!(input.unit(), UtilityUnit::PerExposureHour);
        assert_eq!(input.evaluate(3.0, 0.0).unwrap(), 3.5);
        assert_eq!(input.evaluate(3.0, -0.5).unwrap(), 4.0);
        let sample = UtilityInput {
            accounting: Accounting::Sample {
                decisions: 100,
                positive_decisions: 5,
                baseline: counts(2, 4),
                augmented: counts(4, 2),
            },
            ..input
        };
        assert_eq!(sample.unit(), UtilityUnit::PerDecision);
        assert!((sample.evaluate(3.0, 0.0).unwrap() - 0.07).abs() < 1e-12);
    }

    #[test]
    fn frontier_sign_reversal_and_zero_slopes() {
        let coefficient = Coefficients {
            a: 2.0,
            b: 1.0,
            k: 0.0,
        };
        assert_eq!(
            coefficient.ratio_frontier(0.0).unwrap(),
            RatioFrontier::PositiveBelow {
                exclusive_ratio: 2.0
            }
        );
        let reverse = Coefficients {
            a: -2.0,
            b: -1.0,
            k: 0.0,
        };
        assert_eq!(
            reverse.ratio_frontier(0.0).unwrap(),
            RatioFrontier::PositiveAbove {
                exclusive_ratio: 2.0
            }
        );
        assert!(reverse.evaluate(3.0, 0.0).unwrap() > 0.0);
        assert_eq!(
            event().ratio_frontier(0.0).unwrap(),
            RatioFrontier::AllPositive
        );
        for (intercept, expected) in [
            (1.0, RatioFrontier::AllPositive),
            (0.0, RatioFrontier::BreakEvenEverywhere),
            (-1.0, RatioFrontier::NoPositive { zero_at: None }),
        ] {
            assert_eq!(
                Coefficients {
                    a: intercept,
                    b: 0.0,
                    k: 0.0
                }
                .ratio_frontier(0.0)
                .unwrap(),
                expected
            );
        }
        assert_eq!(
            coefficient.ratio_frontier(2.0).unwrap(),
            RatioFrontier::NoPositive { zero_at: Some(0.0) }
        );
        assert_eq!(
            coefficient.ratio_frontier(3.0).unwrap(),
            RatioFrontier::NoPositive { zero_at: None }
        );
        assert_eq!(
            reverse.ratio_frontier(-2.0).unwrap(),
            RatioFrontier::PositiveAbove {
                exclusive_ratio: 0.0
            }
        );
    }

    #[test]
    fn zero_events_preserve_false_alert_cost_without_inventing_detection_rate() {
        let input = UtilityInput {
            accounting: Accounting::Event {
                exposure_hours: 1.0,
                true_events: 0,
                baseline: counts(0, 5),
                augmented: counts(0, 2),
            },
            additional_overhead: 0.0,
            ..event()
        };
        assert_eq!(input.evaluate(2.0, 0.0).unwrap(), 6.0);
    }

    #[test]
    fn missing_uncertainty_stays_unknown_and_draws_preserve_overhead() {
        let mut input = event();
        assert!(input.interval(1.0, 0.0).unwrap().is_none());
        input.paired_draws = Some(vec![
            PairedDraw {
                accounting: input.accounting.clone(),
                additional_overhead: 0.0,
            },
            PairedDraw {
                accounting: input.accounting.clone(),
                additional_overhead: 20.0,
            },
        ]);
        let interval = input.interval(1.0, 0.0).unwrap().unwrap();
        assert!((interval.lower - 1.025).abs() < 1e-12);
        assert!((interval.upper - 1.975).abs() < 1e-12);
        let shifted = input.interval(1.0, 0.5).unwrap().unwrap();
        assert!((shifted.lower - interval.lower + 0.5).abs() < 1e-12);
        input.paired_draws = Some(vec![]);
        assert!(input.interval(1.0, 0.0).is_err());
    }

    #[test]
    fn invalid_counts_exposure_and_numeric_overflow_fail_closed() {
        let invalid_accounts = [
            Accounting::Sample {
                decisions: 0,
                positive_decisions: 0,
                baseline: counts(0, 0),
                augmented: counts(0, 0),
            },
            Accounting::Sample {
                decisions: 2,
                positive_decisions: 3,
                baseline: counts(0, 0),
                augmented: counts(0, 0),
            },
            Accounting::Sample {
                decisions: 2,
                positive_decisions: 1,
                baseline: counts(0, 2),
                augmented: counts(0, 0),
            },
            Accounting::Event {
                exposure_hours: 1.0,
                true_events: 1,
                baseline: counts(2, 0),
                augmented: counts(0, 0),
            },
            Accounting::Event {
                exposure_hours: 0.0,
                true_events: 0,
                baseline: counts(0, 0),
                augmented: counts(0, 0),
            },
            Accounting::Event {
                exposure_hours: f64::NAN,
                true_events: 0,
                baseline: counts(0, 0),
                augmented: counts(0, 0),
            },
            Accounting::Event {
                exposure_hours: 1.0,
                true_events: 0,
                baseline: counts(0, u64::MAX),
                augmented: counts(0, 0),
            },
        ];
        for accounting in invalid_accounts {
            assert!(
                UtilityInput {
                    accounting,
                    ..event()
                }
                .coefficients()
                .is_err()
            );
        }
        for benefit in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::MIN_POSITIVE] {
            assert!(
                UtilityInput {
                    benefit_per_true_detection: benefit,
                    additional_overhead: f64::MAX,
                    ..event()
                }
                .coefficients()
                .is_err()
            );
        }
        assert!(event().evaluate(-1.0, 0.0).is_err());
        assert!(event().evaluate(f64::INFINITY, 0.0).is_err());
        assert!(event().evaluate(1.0, f64::NAN).is_err());
        assert!(
            Coefficients {
                a: 1.0,
                b: f64::MAX,
                k: 0.0
            }
            .evaluate(2.0, 0.0)
            .is_err()
        );
    }

    #[test]
    fn supplied_draws_cannot_mix_sample_and_event_units() {
        let mut input = event();
        let draw = PairedDraw {
            accounting: Accounting::Sample {
                decisions: 2,
                positive_decisions: 1,
                baseline: counts(0, 0),
                augmented: counts(1, 0),
            },
            additional_overhead: 0.0,
        };
        input.paired_draws = Some(vec![draw.clone(), draw]);
        assert!(input.interval(1.0, 0.0).is_err());
    }
}

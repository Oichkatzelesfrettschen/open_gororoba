//! Paired model metrics, standalone rankings, and conditional intervals.

use anyhow::{Result, ensure};
use gororoba_cli_physics::{
    staple_calibration::PreparedDataset,
    staple_logistic::{logit_loss, weighted_auc, weighted_average_precision},
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{EPSILON, REPLICATES};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub(super) struct Metrics {
    pub(super) roc_auc: f64,
    pub(super) average_precision: f64,
    pub(super) log_loss: f64,
}

impl Metrics {
    pub(super) fn delta(self, baseline: Self) -> Self {
        Self {
            roc_auc: self.roc_auc - baseline.roc_auc,
            average_precision: self.average_precision - baseline.average_precision,
            log_loss: self.log_loss - baseline.log_loss,
        }
    }
}

pub(super) fn metrics(logits: &[f64], labels: &[u8], weights: &[f64]) -> Result<Metrics> {
    ensure!(
        logits.len() == labels.len() && logits.len() == weights.len(),
        "metric dimension mismatch"
    );
    ensure!(
        logits.iter().all(|value| value.is_finite()),
        "nonfinite predicted logits"
    );
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.par_sort_unstable_by(|&left, &right| logits[right].total_cmp(&logits[left]));
    let sorted: Vec<f64> = order.iter().map(|&index| logits[index]).collect();
    let sorted_labels: Vec<u8> = order.iter().map(|&index| labels[index]).collect();
    let sorted_weights: Vec<f64> = order.iter().map(|&index| weights[index]).collect();
    let loss: f64 = logits
        .par_iter()
        .zip(labels)
        .zip(weights)
        .map(|((&logit, &label), &weight)| weight * logit_loss(logit, f64::from(label)))
        .sum();
    let result = Metrics {
        roc_auc: weighted_auc(&sorted, &sorted_labels, &sorted_weights),
        average_precision: weighted_average_precision(&sorted, &sorted_labels, &sorted_weights),
        log_loss: loss / weights.iter().sum::<f64>(),
    };
    ensure!(
        [result.roc_auc, result.average_precision, result.log_loss]
            .iter()
            .all(|value| value.is_finite()),
        "undefined weighted model metrics"
    );
    Ok(result)
}

pub(super) struct Ranking {
    pub(super) scores: Vec<f64>,
    pub(super) labels: Vec<u8>,
    pub(super) files: Vec<u16>,
    pub(super) gradient: Vec<f64>,
}

impl Ranking {
    fn new(data: &PreparedDataset, rows: &[u32], values: &[f64]) -> Self {
        let mut order = rows.to_vec();
        order.par_sort_unstable_by(|&left, &right| {
            values[right as usize].total_cmp(&values[left as usize])
        });
        Self {
            scores: order.iter().map(|&row| values[row as usize]).collect(),
            labels: order.iter().map(|&row| data.labels[row as usize]).collect(),
            files: order
                .iter()
                .map(|&row| data.file_index[row as usize])
                .collect(),
            gradient: order.iter().map(|&row| data.dbdt[row as usize]).collect(),
        }
    }

    pub(super) fn auc_strata(&self, counts: &[u32], threshold: f64) -> Result<[f64; 3]> {
        self.auc_strata_scores(&self.scores, counts, threshold)
    }

    pub(super) fn auc_strata_scores(
        &self,
        scores: &[f64],
        counts: &[u32],
        threshold: f64,
    ) -> Result<[f64; 3]> {
        let mut output = [0.0; 3];
        for (stratum, result) in output.iter_mut().enumerate() {
            let weights: Vec<f64> = self
                .files
                .iter()
                .enumerate()
                .map(|(index, &file)| {
                    let included = self.labels[index] == 0
                        || stratum == 0
                        || (stratum == 1 && self.gradient[index] <= threshold)
                        || (stratum == 2 && self.gradient[index] > threshold);
                    if included {
                        f64::from(counts[usize::from(file)])
                    } else {
                        0.0
                    }
                })
                .collect();
            *result = weighted_auc(scores, &self.labels, &weights);
            ensure!(
                result.is_finite(),
                "undefined standalone AUC in stratum {stratum}"
            );
        }
        Ok(output)
    }

    pub(super) fn positive_divisor_oracle(&self, rms: f64) -> Result<Value> {
        ensure!(
            rms.is_finite() && rms > 0.0,
            "oracle requires a positive finite divisor"
        );
        let mut maximum_positive_log_shift_error = 0.0_f64;
        let mut maximum_positive_f32_log_shift_error = 0.0_f64;
        let mut zeros = 0usize;
        let mut minimum_positive = f64::INFINITY;
        let mut previous_raw = f64::INFINITY;
        let mut previous_scaled = f64::INFINITY;
        let mut merged_adjacent_pairs = 0usize;
        for &raw in &self.scores {
            let scaled = raw / rms;
            ensure!(scaled.is_finite(), "nonfinite frozen standalone score");
            if previous_raw.is_finite() {
                ensure!(
                    scaled <= previous_scaled,
                    "finite-precision frozen RMS reversed score ordering"
                );
                if raw != previous_raw && scaled == previous_scaled {
                    merged_adjacent_pairs += 1;
                }
            }
            if raw == 0.0 {
                zeros += 1;
            } else {
                minimum_positive = minimum_positive.min(raw);
                let difference = ((scaled + EPSILON).ln() + rms.ln() - (raw + EPSILON).ln()).abs();
                maximum_positive_log_shift_error = maximum_positive_log_shift_error.max(difference);
                let storage_difference = (f64::from((scaled + EPSILON).ln() as f32) + rms.ln()
                    - f64::from((raw + EPSILON).ln() as f32))
                .abs();
                maximum_positive_f32_log_shift_error =
                    maximum_positive_f32_log_shift_error.max(storage_difference);
            }
            previous_raw = raw;
            previous_scaled = scaled;
        }
        Ok(
            json!({"score_order_preserved":true,"ranking_and_tie_partition_preserved":merged_adjacent_pairs==0,"merged_adjacent_pairs":merged_adjacent_pairs,"zero_numerators":zeros,"minimum_positive_numerator":minimum_positive,"max_positive_log_shift_error":maximum_positive_log_shift_error,"max_positive_f32_log_shift_error":maximum_positive_f32_log_shift_error,"zero_log_shift_error":rms.ln().abs(),"boundary":"Positive real division preserves ranks; f64 division can merge adjacent scores. Standalone AUC uses actual divided f64 scores and reports deviation from raw numerator ranks. Additive epsilon and f32 storage permit log-feature shift deviations; zero numerators are excluded from the positive log-shift error. Standardized-feature invariance remains unasserted."}),
        )
    }
}

pub(super) struct Rankings {
    pub(super) associator: Ranking,
    pub(super) pvi: Vec<Ranking>,
}

impl Rankings {
    pub(super) fn new(data: &PreparedDataset, rows: &[u32]) -> Self {
        Self {
            associator: Ranking::new(data, rows, &data.raw_assoc),
            pvi: [&data.daily_pvi, &data.pvi_numerator, &data.rolling_pvi]
                .iter()
                .map(|values| Ranking::new(data, rows, values))
                .collect(),
        }
    }
}

pub(super) fn linear_percentile(sorted: &[f64], quantile: f64) -> Result<f64> {
    ensure!(
        !sorted.is_empty() && (0.0..=1.0).contains(&quantile),
        "invalid percentile request"
    );
    ensure!(
        sorted.iter().all(|value| value.is_finite())
            && sorted.windows(2).all(|pair| pair[0] <= pair[1]),
        "percentile requires sorted finite values"
    );
    let position = (sorted.len() - 1) as f64 * quantile;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    Ok(sorted[lower] + (sorted[upper] - sorted[lower]) * position.fract())
}

pub(super) fn interval(mut values: Vec<f64>) -> Result<Value> {
    ensure!(
        values.len() == REPLICATES,
        "conditional interval requires all100 planned draws"
    );
    values.sort_by(f64::total_cmp);
    let lower = linear_percentile(&values, 0.025)?;
    let upper = linear_percentile(&values, 0.975)?;
    let decision = if lower > 0.0 {
        "positive_under_conditional_procedure"
    } else if upper < 0.0 {
        "negative_under_conditional_procedure"
    } else {
        "inconclusive"
    };
    Ok(
        json!({"replicates":values.len(),"percentile_2_5":lower,"percentile_97_5":upper,"median":linear_percentile(&values,0.5)?,"adverse_nonpositive_count":values.iter().filter(|&&value| value <= 0.0).count(),"denominator":REPLICATES,"monte_carlo_resolution":1.0/101.0,"decision":decision,"coverage_boundary":"Empirical linear percentiles conditional on seed42 fold0 partition;100 draws limit tail precision and establish approximate intervals rather than demonstrated95-percent coverage."}),
    )
}

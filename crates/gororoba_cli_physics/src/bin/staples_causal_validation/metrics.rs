//! Paired discrimination metrics and exact daily-weight AUC sufficient statistics.

use anyhow::{Result, ensure};
use gororoba_cli_physics::staple_logistic::{logit_loss, weighted_auc, weighted_average_precision};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeSet;

use super::{Config, admission::Dataset, fitting::Model, splits::draw_counts};

#[derive(Clone, Deserialize, Serialize)]
pub(super) struct PointMetrics {
    pub(super) roc_auc: f64,
    pub(super) average_precision: f64,
    pub(super) log_loss: f64,
    pub(super) rows: usize,
    pub(super) positives: usize,
    pub(super) files: usize,
    pub(super) positive_files: usize,
}

#[derive(Clone, Deserialize, Serialize)]
pub(super) struct AucKernel {
    pub(super) file_ids: Vec<u16>,
    pub(super) positives: Vec<u64>,
    pub(super) negatives: Vec<u64>,
    // Matrix entry owns positive-file/negative-file wins with half credit for ties.
    pub(super) wins: Vec<f64>,
}

impl AucKernel {
    fn from_sorted(scores: &[(f64, u8, u16)]) -> Self {
        let file_ids: Vec<u16> = scores
            .iter()
            .map(|row| row.2)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let count = file_ids.len();
        let mut positives = vec![0_u64; count];
        let mut negatives = vec![0_u64; count];
        let mut wins = vec![0.0; count * count];
        let mut group_positive = vec![0_u64; count];
        let mut group_negative = vec![0_u64; count];
        let mut cursor = 0;
        while cursor < scores.len() {
            group_positive.fill(0);
            group_negative.fill(0);
            let mut end = cursor + 1;
            while end < scores.len() && scores[end].0 == scores[cursor].0 {
                end += 1;
            }
            for &(_, label, file) in &scores[cursor..end] {
                let index = file_ids.binary_search(&file).unwrap();
                if label == 1 {
                    group_positive[index] += 1;
                } else {
                    group_negative[index] += 1;
                }
            }
            for (negative_file, &negative_count) in group_negative
                .iter()
                .enumerate()
                .filter(|(_, count)| **count > 0)
            {
                for positive_file in 0..count {
                    wins[positive_file * count + negative_file] += (positives[positive_file]
                        as f64
                        + 0.5 * group_positive[positive_file] as f64)
                        * negative_count as f64;
                }
            }
            for index in 0..count {
                positives[index] += group_positive[index];
                negatives[index] += group_negative[index];
            }
            cursor = end;
        }
        Self {
            file_ids,
            positives,
            negatives,
            wins,
        }
    }

    pub(super) fn validate(&self) -> Result<()> {
        let count = self.file_ids.len();
        ensure!(
            count > 0 && self.file_ids.windows(2).all(|pair| pair[0] < pair[1]),
            "invalid AUC daily identities"
        );
        ensure!(
            self.positives.len() == count
                && self.negatives.len() == count
                && self.wins.len() == count * count,
            "invalid AUC kernel dimensions"
        );
        for positive in 0..count {
            for negative in 0..count {
                let wins = self.wins[positive * count + negative];
                ensure!(
                    wins.is_finite()
                        && wins >= 0.0
                        && wins
                            <= self.positives[positive] as f64 * self.negatives[negative] as f64,
                    "invalid retained pairwise win count"
                );
            }
        }
        Ok(())
    }

    pub(super) fn auc(&self, counts: &[u32]) -> Result<f64> {
        let count = self.file_ids.len();
        ensure!(
            counts.len() == count,
            "bootstrap membership dimension mismatch"
        );
        let positives: f64 = self
            .positives
            .iter()
            .zip(counts)
            .map(|(&value, &weight)| value as f64 * f64::from(weight))
            .sum();
        let negatives: f64 = self
            .negatives
            .iter()
            .zip(counts)
            .map(|(&value, &weight)| value as f64 * f64::from(weight))
            .sum();
        ensure!(
            positives > 0.0 && negatives > 0.0,
            "bootstrap draw lacks a label class"
        );
        let mut wins = 0.0;
        for (positive, &positive_weight) in
            counts.iter().enumerate().filter(|(_, value)| **value > 0)
        {
            let weighted: f64 = self.wins[positive * count..(positive + 1) * count]
                .iter()
                .zip(counts)
                .map(|(&value, &weight)| value * f64::from(weight))
                .sum();
            wins += f64::from(positive_weight) * weighted;
        }
        let result = wins / (positives * negatives);
        ensure!(
            result.is_finite() && (0.0..=1.0).contains(&result),
            "undefined paired AUC"
        );
        Ok(result)
    }
}

pub(super) fn evaluate(
    data: &Dataset,
    model: &Model,
    width_index: usize,
    year: i32,
    retain_kernel: bool,
) -> Result<(PointMetrics, Option<AucKernel>)> {
    let mut scores: Vec<(f64, u8, u16)> = data
        .rows
        .par_iter()
        .filter(|row| row.year == year)
        .map(|row| (model.predict(row, width_index), row.label, row.file))
        .collect();
    ensure!(
        !scores.is_empty() && scores.iter().all(|row| row.0.is_finite()),
        "empty or nonfinite model predictions in year {year}"
    );
    scores.par_sort_unstable_by(|left, right| right.0.total_cmp(&left.0));
    let logits: Vec<f64> = scores.iter().map(|row| row.0).collect();
    let labels: Vec<u8> = scores.iter().map(|row| row.1).collect();
    let weights = vec![1.0; scores.len()];
    let result = PointMetrics {
        roc_auc: weighted_auc(&logits, &labels, &weights),
        average_precision: weighted_average_precision(&logits, &labels, &weights),
        log_loss: scores
            .iter()
            .map(|row| logit_loss(row.0, f64::from(row.1)))
            .sum::<f64>()
            / scores.len() as f64,
        rows: scores.len(),
        positives: labels.iter().map(|&label| usize::from(label)).sum(),
        files: scores
            .iter()
            .map(|row| row.2)
            .collect::<BTreeSet<_>>()
            .len(),
        positive_files: scores
            .iter()
            .filter(|row| row.1 == 1)
            .map(|row| row.2)
            .collect::<BTreeSet<_>>()
            .len(),
    };
    ensure!(
        [result.roc_auc, result.average_precision, result.log_loss]
            .iter()
            .all(|value| value.is_finite())
            && result.positives > 0
            && result.positives < result.rows,
        "undefined model metrics or absent class in year {year}"
    );
    let kernel = if retain_kernel {
        let kernel = AucKernel::from_sorted(&scores);
        kernel.validate()?;
        ensure!(
            (kernel.auc(&vec![1; kernel.file_ids.len()])? - result.roc_auc).abs() < 1e-12,
            "AUC sufficient-statistic oracle differs from rank metric"
        );
        Some(kernel)
    } else {
        None
    };
    Ok((result, kernel))
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    let position = quantile * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower as f64)
}

pub(super) fn target_declaration(target: f64) -> Value {
    json!({"value":target,"metric":"paired_roc_auc_increment","basis":"historical_investigator_declared_discrimination_target","application_based_justification":"unestablished","scope":"Target attainment is separate from effect evidence, algebra specificity and practical utility."})
}

pub(super) fn practical_utility_boundary() -> Value {
    json!({"status":"unassessed","required_evidence":["Defined application and operating point","Admitted event prevalence and ordinary-day false alarms","Missed-event, false-alarm and resource costs","Paired operational outcomes"],"scope":"An aggregate ROC-AUC increment alone establishes neither usefulness nor uselessness."})
}

pub(super) fn discrimination_assessment(lower: f64, upper: f64, target: f64) -> Result<Value> {
    ensure!(
        lower.is_finite() && upper.is_finite() && target.is_finite() && lower <= upper,
        "discrimination assessment requires finite ordered interval endpoints and target"
    );
    let effect = if lower > 0.0 {
        "positive"
    } else if upper < 0.0 {
        "negative"
    } else {
        "inconclusive"
    };
    let target_comparison = if lower > target {
        "above"
    } else if upper < target {
        "below"
    } else {
        "touches_or_crosses"
    };
    Ok(
        json!({"schema_version":2,"assessment_status":"assessed","interval":[lower,upper],"effect_assessment":effect,"declared_discrimination_target":target_declaration(target),"target_comparison":target_comparison,"practical_utility":practical_utility_boundary()}),
    )
}

pub(super) fn bootstrap(
    kernels: &[(usize, i32, AucKernel, AucKernel)],
    config: &Config,
) -> Result<Value> {
    ensure!(
        kernels.len() == config.widths.len() * config.final_years.len(),
        "bootstrap requires exact width/year kernel set"
    );
    let mut support = Vec::new();
    for &year in &config.final_years {
        let canonical = &kernels.iter().find(|row| row.1 == year).unwrap().3;
        canonical.validate()?;
        let positive_files = canonical
            .positives
            .iter()
            .filter(|&&value| value > 0)
            .count();
        support.push(json!({"year":year,"admitted_files":canonical.file_ids.len(),"positive_files":positive_files}));
        for (_, _, baseline, augmented) in kernels.iter().filter(|row| row.1 == year) {
            baseline.validate()?;
            augmented.validate()?;
            ensure!(
                baseline.file_ids == canonical.file_ids
                    && augmented.file_ids == canonical.file_ids
                    && baseline.positives == canonical.positives
                    && baseline.negatives == canonical.negatives
                    && augmented.positives == canonical.positives
                    && augmented.negatives == canonical.negatives,
                "width/model daily support or labels differ"
            );
        }
    }
    if support.iter().any(|row| {
        row["admitted_files"].as_u64().unwrap() < 30 || row["positive_files"].as_u64().unwrap() < 10
    }) {
        return Ok(
            json!({"discrimination_assessment":{"schema_version":2,"assessment_status":"insufficient_support","reason":"Each final year requires at least 30 admitted files and 10 positive-containing files.","support":support,"interval":null,"effect_assessment":"inconclusive","declared_discrimination_target":target_declaration(config.minimum_increment),"target_comparison":"unassessed","practical_utility":practical_utility_boundary()},"support":support,"planned_draws":config.bootstrap_draws,"completed_draws":0,"interval":null}),
        );
    }
    let mut random = ChaCha8Rng::seed_from_u64(config.bootstrap_seed);
    let mut records = Vec::new();
    let mut global = Vec::new();
    for index in 0..config.bootstrap_draws {
        let counts: Vec<_> = config
            .final_years
            .iter()
            .map(|year| {
                let kernel = &kernels.iter().find(|row| row.1 == *year).unwrap().2;
                (*year, draw_counts(kernel.file_ids.len(), &mut random))
            })
            .collect();
        let mut increments = Vec::new();
        let mut minimum = f64::INFINITY;
        for (width, year, baseline, augmented) in kernels {
            let multiplicity = &counts.iter().find(|row| row.0 == *year).unwrap().1;
            let increment = augmented.auc(multiplicity)? - baseline.auc(multiplicity)?;
            minimum = minimum.min(increment);
            increments.push(json!({"width":width,"year":year,"increment":increment}));
        }
        global.push(minimum);
        records.push(json!({"index":index,"year_file_multiplicities":counts,"increments":increments,"global_minimum":minimum}));
    }
    global.sort_unstable_by(f64::total_cmp);
    let lower = percentile(&global, 0.025);
    let upper = percentile(&global, 0.975);
    Ok(
        json!({"discrimination_assessment":discrimination_assessment(lower,upper,config.minimum_increment)?,"interval":[lower,upper],"median":percentile(&global,0.5),"support":support,"planned_draws":config.bootstrap_draws,"completed_draws":records.len(),"monte_carlo_resolution":1.0/(config.bootstrap_draws+1) as f64,"uncertainty":"Approximate whole-file bootstrap conditional on frozen training estimates and observed final epochs; interday independence remains unestablished.","records":records}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pairwise_kernel_matches_tied_weighted_rank_auc() {
        let scores = [
            (3.0, 1, 0),
            (2.0, 0, 1),
            (2.0, 1, 1),
            (1.0, 0, 0),
            (0.0, 1, 2),
            (0.0, 0, 2),
        ];
        let kernel = AucKernel::from_sorted(&scores);
        kernel.validate().unwrap();
        for counts in [[1, 1, 1], [2, 0, 3], [0, 2, 1]] {
            let observed = kernel.auc(&counts).unwrap();
            let weights: Vec<f64> = scores
                .iter()
                .map(|row| f64::from(counts[row.2 as usize]))
                .collect();
            let expected = weighted_auc(
                &scores.iter().map(|row| row.0).collect::<Vec<_>>(),
                &scores.iter().map(|row| row.1).collect::<Vec<_>>(),
                &weights,
            );
            assert!((observed - expected).abs() < 1e-12);
        }
    }
    #[test]
    fn insufficient_support_survives_summary_extraction_without_target_or_utility_verdict() {
        let config = super::super::test_config();
        for (file_count, positive_file_count) in [(29_u16, 10_u16), (30, 9)] {
            let kernel = AucKernel {
                file_ids: (0..file_count).collect(),
                positives: (0..file_count)
                    .map(|index| u64::from(index < positive_file_count))
                    .collect(),
                negatives: vec![1; usize::from(file_count)],
                wins: vec![0.0; usize::from(file_count).pow(2)],
            };
            let mut kernels = Vec::new();
            for &width in &config.widths {
                for &year in &config.final_years {
                    kernels.push((width, year, kernel.clone(), kernel.clone()));
                }
            }
            let bootstrap = bootstrap(&kernels, &config).unwrap();
            let summary_assessment = super::super::assessment_from_bootstrap(&bootstrap).unwrap();
            assert_eq!(bootstrap["completed_draws"], 0);
            assert_eq!(
                summary_assessment["assessment_status"],
                "insufficient_support"
            );
            assert_eq!(summary_assessment["effect_assessment"], "inconclusive");
            assert_eq!(summary_assessment["target_comparison"], "unassessed");
            assert_eq!(
                summary_assessment["practical_utility"]["status"],
                "unassessed"
            );
            assert_eq!(
                summary_assessment["support"][0]["admitted_files"],
                file_count
            );
            assert!(
                summary_assessment["reason"]
                    .as_str()
                    .unwrap()
                    .contains("30 admitted files")
            );
        }
        assert!(
            super::super::assessment_from_bootstrap(
                &json!({"decision":"inconclusive_insufficient_support"})
            )
            .is_err()
        );
    }

    #[test]
    fn retained_internal_and_external_intervals_keep_positive_effect_below_target() {
        for (claim, relative_path, interval_key) in [
            (
                "C-1743",
                "data/output/audit/staples-causal-validation/findings.toml",
                "primary_interval",
            ),
            (
                "C-1754",
                "data/output/audit/external-crossing-intake-amendment/findings.toml",
                "global_minimum_increment_interval",
            ),
        ] {
            let path = repo_root::path!(relative_path);
            let source = std::fs::read_to_string(&path).unwrap();
            let retained: toml::Value = toml::from_str(&source).unwrap();
            let interval = retained[interval_key].as_array().unwrap();
            let target = retained["minimum_increment"].as_float().unwrap();
            let report = discrimination_assessment(
                interval[0].as_float().unwrap(),
                interval[1].as_float().unwrap(),
                target,
            )
            .unwrap();
            assert_eq!(report["effect_assessment"], "positive");
            assert_eq!(report["target_comparison"], "below");
            assert_eq!(report["practical_utility"]["status"], "unassessed");
            println!(
                "{}",
                json!({"claim_id":claim,"source":relative_path,"source_sha256":super::super::evidence::hash_file(&path).unwrap(),"assessment":report})
            );
        }
    }

    #[test]
    fn discrimination_target_cannot_adjudicate_effect_or_practical_utility() {
        for (lower, upper, effect) in [
            (0.001, 0.003, "positive"),
            (-0.003, -0.001, "negative"),
            (-0.001, 0.003, "inconclusive"),
            (0.0, 0.003, "inconclusive"),
            (-0.003, 0.0, "inconclusive"),
        ] {
            for (target, comparison) in [
                (lower - 0.001, "above"),
                (lower, "touches_or_crosses"),
                (upper, "touches_or_crosses"),
                (upper + 0.001, "below"),
            ] {
                let report = discrimination_assessment(lower, upper, target).unwrap();
                assert_eq!(report["effect_assessment"], effect);
                assert_eq!(report["target_comparison"], comparison);
                assert_eq!(report["practical_utility"]["status"], "unassessed");
            }
        }
        for (lower, upper, target) in [
            (0.003, 0.001, 0.005),
            (f64::NAN, 0.003, 0.005),
            (0.001, f64::INFINITY, 0.005),
            (0.001, 0.003, f64::NAN),
        ] {
            assert!(discrimination_assessment(lower, upper, target).is_err());
        }
    }
}

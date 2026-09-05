//! Training calibration, standardization, and paired logistic evaluation.

use anyhow::{Result, ensure};
use gororoba_cli_physics::{staple_calibration::PreparedDataset, staple_logistic::fit_irls};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde_json::{Value, json};
use std::collections::BTreeSet;

use super::{
    ARMS, EPSILON, PVI_COLUMN, RIDGE, STRIDE,
    metrics::{Rankings, metrics},
    splitting::{draw_counts, folds, selected_rows, unit_counts},
};

pub(super) fn training_rms(data: &PreparedDataset, counts: &[u32]) -> Result<f64> {
    let mut squares = 0.0;
    let mut increments = 0.0;
    for file in &data.files {
        let multiplicity = f64::from(counts[usize::from(file.id)]);
        squares += file.increment_square_sum * multiplicity;
        increments += file.increment_count as f64 * multiplicity;
    }
    let rms = (squares / increments).sqrt();
    ensure!(
        rms.is_finite() && rms > 0.0,
        "invalid training calibration RMS"
    );
    Ok(rms)
}

pub(super) fn replace_calibration(
    features: &mut [f32],
    data: &PreparedDataset,
    arm: &str,
    rms: f64,
) -> Result<()> {
    ensure!(ARMS.contains(&arm), "unknown calibration arm {arm}");
    features
        .par_chunks_mut(STRIDE)
        .enumerate()
        .for_each(|(row, values)| {
            values[PVI_COLUMN] = match arm {
                "frozen_training" => (data.pvi_numerator[row] / rms + EPSILON).ln() as f32,
                "prewindow_rolling" => data.rolling_log_pvi[row],
                _ => data.features[row * STRIDE + PVI_COLUMN],
            };
        });
    ensure!(
        features.iter().all(|value| value.is_finite()),
        "nonfinite calibrated features"
    );
    Ok(())
}

pub(super) fn standardize(
    features: &[f32],
    rows: &[u32],
) -> Result<([f64; STRIDE], [f64; STRIDE])> {
    ensure!(!rows.is_empty(), "empty training rows");
    let sum = rows
        .par_chunks(65536)
        .map(|chunk| {
            let mut result = [0.0; STRIDE];
            for &row in chunk {
                for feature in 0..STRIDE {
                    result[feature] += f64::from(features[row as usize * STRIDE + feature]);
                }
            }
            result
        })
        .reduce(
            || [0.0; STRIDE],
            |mut left, right| {
                for feature in 0..STRIDE {
                    left[feature] += right[feature];
                }
                left
            },
        );
    let mean = sum.map(|value| value / rows.len() as f64);
    let squares = rows
        .par_chunks(65536)
        .map(|chunk| {
            let mut result = [0.0; STRIDE];
            for &row in chunk {
                for feature in 0..STRIDE {
                    let delta =
                        f64::from(features[row as usize * STRIDE + feature]) - mean[feature];
                    result[feature] += delta * delta;
                }
            }
            result
        })
        .reduce(
            || [0.0; STRIDE],
            |mut left, right| {
                for feature in 0..STRIDE {
                    left[feature] += right[feature];
                }
                left
            },
        );
    let scale = squares.map(|value| (value / rows.len() as f64).sqrt());
    ensure!(
        mean.iter().all(|value| value.is_finite())
            && scale.iter().all(|value| value.is_finite() && *value > 0.0),
        "nonfinite or zero training scale"
    );
    Ok((mean, scale))
}

// Keep paired training and evaluation populations explicit at the fitting boundary.
#[allow(clippy::too_many_arguments)]
fn evaluate_arm(
    data: &PreparedDataset,
    features: &mut [f32],
    arm_index: usize,
    train_counts: &[u32],
    train_rows: &[u32],
    test_rows: &[u32],
    test_counts: &[u32],
    rankings: Option<&Rankings>,
    threshold: f64,
) -> Result<(Value, Vec<Vec<f64>>)> {
    let arm = ARMS[arm_index];
    let rms = training_rms(data, train_counts)?;
    replace_calibration(features, data, arm, rms)?;
    let (mean, scale) = standardize(features, train_rows)?;
    let labels: Vec<u8> = test_rows
        .iter()
        .map(|&row| data.labels[row as usize])
        .collect();
    let weights: Vec<f64> = test_rows
        .iter()
        .map(|&row| f64::from(test_counts[usize::from(data.file_index[row as usize])]))
        .collect();
    let mut fits = Vec::new();
    let mut evaluations = Vec::new();
    let mut predictions = Vec::new();
    for feature_count in [6, 7] {
        let columns: Vec<usize> = (0..feature_count).collect();
        let fit = fit_irls(
            features,
            STRIDE,
            &columns,
            train_rows,
            &data.labels,
            &mean[..feature_count],
            &scale[..feature_count],
            RIDGE,
        )?;
        ensure!(
            fit.converged,
            "{arm} model {feature_count} exhausted Newton budget; diagnostics={}",
            json!({"coefficients":fit.beta,"deviance":fit.deviance,"iterations":fit.iterations,"mean":mean,"std":scale,"training_rms":rms})
        );
        let logits: Vec<f64> = test_rows
            .par_iter()
            .map(|&row| {
                let offset = row as usize * STRIDE;
                fit.beta[0]
                    + (0..feature_count)
                        .map(|feature| {
                            fit.beta[feature + 1]
                                * (f64::from(features[offset + feature]) - mean[feature])
                                / scale[feature]
                        })
                        .sum::<f64>()
            })
            .collect();
        evaluations.push(metrics(&logits, &labels, &weights)?);
        fits.push(json!({"feature_count":feature_count,"coefficients":fit.beta,"penalized_deviance_trajectory":fit.deviance,"iterations":fit.iterations,"converged":fit.converged}));
        predictions.push(logits);
    }
    let standalone = if let Some(rankings) = rankings {
        let associator = rankings.associator.auc_strata(test_counts, threshold)?;
        let raw_pvi = rankings.pvi[arm_index].auc_strata(test_counts, threshold)?;
        let (pvi, oracle) = if arm == "frozen_training" {
            let scores: Vec<f64> = rankings.pvi[arm_index]
                .scores
                .iter()
                .map(|value| value / rms)
                .collect();
            let mut oracle = rankings.pvi[arm_index].positive_divisor_oracle(rms)?;
            let calibrated =
                rankings.pvi[arm_index].auc_strata_scores(&scores, test_counts, threshold)?;
            oracle["raw_numerator_auc"] = json!(raw_pvi);
            oracle["calibrated_minus_raw_auc"] = json!(
                calibrated
                    .iter()
                    .zip(raw_pvi)
                    .map(|(left, right)| left - right)
                    .collect::<Vec<_>>()
            );
            (calibrated, oracle)
        } else {
            (raw_pvi, Value::Null)
        };
        let delta: Vec<f64> = associator
            .iter()
            .zip(pvi)
            .map(|(left, right)| left - right)
            .collect();
        let minimum = delta.iter().copied().fold(f64::INFINITY, f64::min);
        json!({"strata":["bulk","low_gradient_positive","high_gradient_positive"],"associator_auc":associator,"pvi_auc":pvi,"delta_auc":delta,"minimum_delta_auc":minimum,"frozen_divisor_oracle":oracle})
    } else {
        Value::Null
    };
    Ok((
        json!({"arm":arm,"training_rms":rms,"training_rows_with_multiplicity":train_rows.len(),"training_mean":mean,"training_population_std":scale,"fits":fits,"baseline":evaluations[0],"augmented":evaluations[1],"delta":evaluations[1].delta(evaluations[0]),"standalone":standalone}),
        predictions,
    ))
}

pub(super) fn run_cv(data: &PreparedDataset, seed: u64, threshold: f64) -> Result<Value> {
    let file_ids: Vec<u16> = data.files.iter().map(|file| file.id).collect();
    let assignments = folds(&file_ids, seed);
    let mut features = data.features.clone();
    let mut arm_results = Vec::new();
    for (arm_index, arm) in ARMS.iter().enumerate() {
        let mut predictions = vec![vec![f64::NAN; data.labels.len()]; 2];
        let mut fold_results = Vec::new();
        for (fold_index, test_ids) in assignments.iter().enumerate() {
            eprintln!("CV seed {seed} arm {arm} fold {fold_index}");
            let test_set: BTreeSet<u16> = test_ids.iter().copied().collect();
            let train_ids: Vec<u16> = file_ids
                .iter()
                .copied()
                .filter(|id| !test_set.contains(id))
                .collect();
            let train_counts = unit_counts(&train_ids, data.files.len());
            let test_counts = unit_counts(test_ids, data.files.len());
            let train_rows = selected_rows(data, &train_counts);
            let test_rows = selected_rows(data, &test_counts);
            let rankings = if seed == 42 && fold_index == 0 {
                Some(Rankings::new(data, &test_rows))
            } else {
                None
            };
            let (result, logits) = evaluate_arm(
                data,
                &mut features,
                arm_index,
                &train_counts,
                &train_rows,
                &test_rows,
                &test_counts,
                rankings.as_ref(),
                threshold,
            )?;
            for model in 0..2 {
                for (position, &row) in test_rows.iter().enumerate() {
                    predictions[model][row as usize] = logits[model][position];
                }
            }
            fold_results.push(result);
        }
        let weights = vec![1.0; data.labels.len()];
        let baseline = metrics(&predictions[0], &data.labels, &weights)?;
        let augmented = metrics(&predictions[1], &data.labels, &weights)?;
        arm_results.push(json!({"arm":ARMS[arm_index],"folds":fold_results,"out_of_fold_baseline":baseline,"out_of_fold_augmented":augmented,"out_of_fold_delta":augmented.delta(baseline)}));
    }
    Ok(
        json!({"seed":seed,"file_assignments":assignments,"arms":arm_results,"population_rows":data.labels.len(),"prediction_precision":"f64 logits from f32 log features and f64 coefficients"}),
    )
}

// Explicit partition inputs prevent a sampled file crossing the holdout boundary.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_bootstrap(
    data: &PreparedDataset,
    index: usize,
    threshold: f64,
    rankings: &Rankings,
    train_ids: &[u16],
    test_ids: &[u16],
    test_rows: &[u32],
) -> Result<Value> {
    let seed = 10000 + index as u64;
    let mut random = ChaCha8Rng::seed_from_u64(seed);
    let train_counts = draw_counts(train_ids, data.files.len(), &mut random);
    let test_counts = draw_counts(test_ids, data.files.len(), &mut random);
    let train_rows = selected_rows(data, &train_counts);
    let mut features = data.features.clone();
    let mut arms = Vec::new();
    for (arm_index, arm) in ARMS.iter().enumerate() {
        eprintln!("bootstrap {index} arm {arm}");
        let (result, _) = evaluate_arm(
            data,
            &mut features,
            arm_index,
            &train_counts,
            &train_rows,
            test_rows,
            &test_counts,
            Some(rankings),
            threshold,
        )?;
        arms.push(result);
    }
    let minimum_increment = arms
        .iter()
        .map(|arm| arm["delta"]["roc_auc"].as_f64().unwrap())
        .fold(f64::INFINITY, f64::min);
    let minimum_calibration = arms
        .iter()
        .map(|arm| arm["standalone"]["minimum_delta_auc"].as_f64().unwrap())
        .fold(f64::INFINITY, f64::min);
    Ok(
        json!({"index":index,"seed":seed,"partition_seed":42,"held_out_fold":0,"train_file_counts":train_counts,"test_file_counts":test_counts,"arms":arms,"global_minimum_increment":minimum_increment,"global_minimum_calibration":minimum_calibration,"sampling_stream":"ChaCha8(seed); unbiased independent training draws followed by independent test draws"}),
    )
}

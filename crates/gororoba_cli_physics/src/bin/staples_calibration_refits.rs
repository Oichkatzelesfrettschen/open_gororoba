//! E-280 calibration-context comparison and E-281 grouped logistic refitting.
//! Every inference record binds the admitted corpus, sealed protocol and source.

use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use gororoba_cli_physics::{
    staple_calibration::{PreparedDataset, prepare, validate_sample_order},
    staple_logistic::{
        DEVIANCE_REL_TOL, fit_irls, logit_loss, weighted_auc, weighted_average_precision,
    },
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
};

const STRIDE: usize = 7;
const PVI_COLUMN: usize = 4;
const EPSILON: f64 = 1e-12;
const RIDGE: f64 = 1e-6;
const SEEDS: [u64; 5] = [42, 43, 44, 45, 46];
const ARMS: [&str; 3] = ["daily_file", "frozen_training", "prewindow_rolling"];
const REPLICATES: usize = 100;
const PROTOCOL_HASH: &str = "20bdf30fe878db6e92977df82088bea79104cf81fd3353f4c736effdc362b4d3";
const SOURCE_FILES: [(&str, &[u8]); 17] = [
    (
        "crates/gororoba_cli_physics/src/bin/staples_calibration_refits.rs",
        include_bytes!("staples_calibration_refits.rs"),
    ),
    (
        "crates/gororoba_cli_physics/src/staple_calibration.rs",
        include_bytes!("../staple_calibration.rs"),
    ),
    (
        "crates/gororoba_cli_physics/src/staple_logistic.rs",
        include_bytes!("../staple_logistic.rs"),
    ),
    (
        "crates/gororoba_cli_physics/src/staple_associator.rs",
        include_bytes!("../staple_associator.rs"),
    ),
    (
        "crates/gororoba_cli_physics/src/staple_controls.rs",
        include_bytes!("../staple_controls.rs"),
    ),
    ("Cargo.lock", include_bytes!("../../../../Cargo.lock")),
    ("Cargo.toml", include_bytes!("../../../../Cargo.toml")),
    (
        "crates/gororoba_cli_physics/Cargo.toml",
        include_bytes!("../../Cargo.toml"),
    ),
    (
        "crates/cd_kernel/src/mult_table.rs",
        include_bytes!("../../../cd_kernel/src/mult_table.rs"),
    ),
    (
        "crates/cd_kernel/src/cayley_dickson/mod.rs",
        include_bytes!("../../../cd_kernel/src/cayley_dickson/mod.rs"),
    ),
    (
        "crates/cd_kernel/src/cayley_dickson/arith.rs",
        include_bytes!("../../../cd_kernel/src/cayley_dickson/arith.rs"),
    ),
    (
        "crates/cd_kernel/src/lib.rs",
        include_bytes!("../../../cd_kernel/src/lib.rs"),
    ),
    (
        "crates/cd_kernel/Cargo.toml",
        include_bytes!("../../../cd_kernel/Cargo.toml"),
    ),
    (
        ".cargo/config.toml",
        include_bytes!("../../../../.cargo/config.toml"),
    ),
    (
        "rust-toolchain.toml",
        include_bytes!("../../../../rust-toolchain.toml"),
    ),
    (
        "vendor/proc-macro-error2/src/lib.rs",
        include_bytes!("../../../../vendor/proc-macro-error2/src/lib.rs"),
    ),
    (
        "data/output/audit/staples-calibration-grouped-refits/protocol.toml",
        include_bytes!(
            "../../../../data/output/audit/staples-calibration-grouped-refits/protocol.toml"
        ),
    ),
];

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Mode {
    All,
    Cv,
    Bootstrap,
    Summarize,
}

#[derive(Parser)]
#[command(
    about = "Sealed calibration-context and grouped-refit experiments on the complete Staples corpus"
)]
struct Args {
    #[arg(long)]
    input_root: PathBuf,
    #[arg(long)]
    scores: PathBuf,
    #[arg(long)]
    file_map: PathBuf,
    #[arg(long)]
    catalog: PathBuf,
    #[arg(long)]
    out_dir: PathBuf,
    #[arg(long)]
    protocol: PathBuf,
    #[arg(long)]
    prepare_only: bool,
    #[arg(long, value_enum, default_value = "all")]
    mode: Mode,
    #[arg(long)]
    cv_seed: Option<u64>,
    #[arg(long)]
    bootstrap_index: Option<usize>,
}

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn hash_file(path: &Path) -> Result<String> {
    let mut input = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let length = input.read(&mut buffer)?;
        if length == 0 {
            break;
        }
        hasher.update(&buffer[..length]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn source_root() -> Result<PathBuf> {
    std::env::current_dir()?
        .ancestors()
        .find(|path| {
            path.join("crates/gororoba_cli_physics/src/staple_logistic.rs")
                .is_file()
        })
        .map(Path::to_path_buf)
        .context("launch runner from its source checkout")
}

fn source_identity(root: &Path) -> Result<Value> {
    let mut sources = serde_json::Map::new();
    for (relative, compiled) in SOURCE_FILES {
        let observed = hash_file(&root.join(relative))?;
        ensure!(
            observed == digest(compiled),
            "running binary source differs from {relative}; rebuild before inference"
        );
        sources.insert(relative.to_owned(), json!(observed));
    }
    Ok(Value::Object(sources))
}

fn atomic_json(path: &Path, value: &Value) -> Result<()> {
    ensure!(!path.exists(), "refusing to overwrite {}", path.display());
    let temporary = path.with_extension(format!("json.{}.tmp", std::process::id()));
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    serde_json::to_writer_pretty(&mut output, value)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    fs::rename(&temporary, path)?;
    Ok(())
}

fn folds(file_ids: &[u16], seed: u64) -> Vec<Vec<u16>> {
    let mut shuffled = file_ids.to_vec();
    shuffled.sort_unstable();
    let mut random = ChaCha8Rng::seed_from_u64(seed);
    for index in (1..shuffled.len()).rev() {
        let other = random.random_range(0..=index);
        shuffled.swap(index, other);
    }
    let mut output = vec![Vec::new(); 5];
    for (index, file) in shuffled.into_iter().enumerate() {
        output[index % 5].push(file);
    }
    for fold in &mut output {
        fold.sort_unstable();
    }
    output
}

fn draw_counts(ids: &[u16], file_count: usize, random: &mut ChaCha8Rng) -> Vec<u32> {
    let mut counts = vec![0; file_count];
    for _ in ids {
        let selected = random.random_range(0..ids.len());
        counts[usize::from(ids[selected])] += 1;
    }
    counts
}

fn unit_counts(ids: &[u16], file_count: usize) -> Vec<u32> {
    let mut counts = vec![0; file_count];
    for &id in ids {
        counts[usize::from(id)] = 1;
    }
    counts
}

fn selected_rows(data: &PreparedDataset, counts: &[u32]) -> Vec<u32> {
    data.file_index
        .iter()
        .enumerate()
        .flat_map(|(row, &file)| {
            std::iter::repeat_n(row as u32, counts[usize::from(file)] as usize)
        })
        .collect()
}

fn training_rms(data: &PreparedDataset, counts: &[u32]) -> Result<f64> {
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

fn replace_calibration(
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

fn standardize(features: &[f32], rows: &[u32]) -> Result<([f64; STRIDE], [f64; STRIDE])> {
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

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct Metrics {
    roc_auc: f64,
    average_precision: f64,
    log_loss: f64,
}

impl Metrics {
    fn delta(self, baseline: Self) -> Self {
        Self {
            roc_auc: self.roc_auc - baseline.roc_auc,
            average_precision: self.average_precision - baseline.average_precision,
            log_loss: self.log_loss - baseline.log_loss,
        }
    }
}

fn metrics(logits: &[f64], labels: &[u8], weights: &[f64]) -> Result<Metrics> {
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

struct Ranking {
    scores: Vec<f64>,
    labels: Vec<u8>,
    files: Vec<u16>,
    gradient: Vec<f64>,
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

    fn auc_strata(&self, counts: &[u32], threshold: f64) -> Result<[f64; 3]> {
        self.auc_strata_scores(&self.scores, counts, threshold)
    }

    fn auc_strata_scores(
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

    fn positive_divisor_oracle(&self, rms: f64) -> Result<Value> {
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

struct Rankings {
    associator: Ranking,
    pvi: Vec<Ranking>,
}

impl Rankings {
    fn new(data: &PreparedDataset, rows: &[u32]) -> Self {
        Self {
            associator: Ranking::new(data, rows, &data.raw_assoc),
            pvi: [&data.daily_pvi, &data.pvi_numerator, &data.rolling_pvi]
                .iter()
                .map(|values| Ranking::new(data, rows, values))
                .collect(),
        }
    }
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

fn run_cv(data: &PreparedDataset, seed: u64, threshold: f64) -> Result<Value> {
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
fn run_bootstrap(
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

fn linear_percentile(sorted: &[f64], quantile: f64) -> Result<f64> {
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

fn interval(mut values: Vec<f64>) -> Result<Value> {
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

fn validate_payload(kind: &str, index: u64, payload: &Value) -> Result<()> {
    ensure!(kind == "cv" || kind == "bootstrap", "unknown record kind");
    let file_ids: Vec<u16> = (0..813).collect();
    let assignments = folds(&file_ids, if kind == "cv" { index } else { 42 });
    if kind == "cv" {
        ensure!(SEEDS.contains(&index), "CV seed outside sealed plan");
        ensure!(
            payload["file_assignments"] == json!(assignments),
            "CV partition differs from sealed seeded assignment"
        );
    } else {
        ensure!(
            index < REPLICATES as u64,
            "bootstrap index outside sealed plan"
        );
        let test_ids = &assignments[0];
        let train_ids: Vec<u16> = file_ids
            .iter()
            .copied()
            .filter(|id| !test_ids.contains(id))
            .collect();
        let mut random = ChaCha8Rng::seed_from_u64(10000 + index);
        ensure!(
            payload["train_file_counts"] == json!(draw_counts(&train_ids, 813, &mut random)),
            "training draw or membership mismatch"
        );
        ensure!(
            payload["test_file_counts"] == json!(draw_counts(test_ids, 813, &mut random)),
            "test draw or membership mismatch"
        );
    }
    let arms = payload["arms"].as_array().context("record lacks arms")?;
    ensure!(arms.len() == ARMS.len(), "record arm count mismatch");
    for (position, arm) in arms.iter().enumerate() {
        ensure!(arm["arm"] == ARMS[position], "record arm order mismatch");
        let evaluations = if kind == "cv" {
            let folds = arm["folds"].as_array().context("CV record lacks folds")?;
            ensure!(folds.len() == 5, "CV record fold count mismatch");
            for metric_key in [
                "out_of_fold_baseline",
                "out_of_fold_augmented",
                "out_of_fold_delta",
            ] {
                validate_metrics(&arm[metric_key])?;
            }
            validate_metric_pair(
                &arm["out_of_fold_baseline"],
                &arm["out_of_fold_augmented"],
                &arm["out_of_fold_delta"],
            )?;
            if index == 42 {
                validate_standalone(&folds[0]["standalone"], ARMS[position])?;
            }
            folds.clone()
        } else {
            vec![arm.clone()]
        };
        for evaluation in evaluations {
            for metric_key in ["baseline", "augmented", "delta"] {
                validate_metrics(&evaluation[metric_key])?;
            }
            validate_metric_pair(
                &evaluation["baseline"],
                &evaluation["augmented"],
                &evaluation["delta"],
            )?;
            let fits = evaluation["fits"].as_array().context("missing fits")?;
            ensure!(fits.len() == 2, "each arm requires both model fits");
            for (model, fit) in fits.iter().enumerate() {
                ensure!(
                    fit["converged"] == true && fit["feature_count"] == model + 6,
                    "invalid model completion"
                );
                let coefficients = fit["coefficients"]
                    .as_array()
                    .context("missing coefficients")?;
                ensure!(
                    coefficients.len() == model + 7
                        && coefficients
                            .iter()
                            .all(|value| value.as_f64().is_some_and(f64::is_finite)),
                    "invalid fitted coefficients"
                );
                let deviance = fit["penalized_deviance_trajectory"]
                    .as_array()
                    .context("missing deviance")?;
                ensure!(
                    deviance.len() >= 2
                        && deviance.len() <= 25
                        && deviance
                            .iter()
                            .all(|value| value.as_f64().is_some_and(f64::is_finite)),
                    "invalid fit trajectory"
                );
                ensure!(
                    fit["iterations"] == deviance.len(),
                    "fit iteration count mismatch"
                );
                let previous = deviance[deviance.len() - 2].as_f64().unwrap();
                let final_value = deviance[deviance.len() - 1].as_f64().unwrap();
                ensure!(
                    (previous - final_value).abs() / previous.abs().max(1.0) < DEVIANCE_REL_TOL,
                    "recorded trajectory fails convergence criterion"
                );
            }
            for key in ["training_mean", "training_population_std"] {
                let values = evaluation[key]
                    .as_array()
                    .context("missing standardization")?;
                ensure!(
                    values.len() == STRIDE
                        && values
                            .iter()
                            .all(|value| value.as_f64().is_some_and(|value| value.is_finite()
                                && (key == "training_mean" || value > 0.0))),
                    "invalid standardization"
                );
            }
            ensure!(
                evaluation["training_rms"]
                    .as_f64()
                    .is_some_and(|value| value.is_finite() && value > 0.0),
                "invalid recorded RMS"
            );
            if kind == "bootstrap" || !evaluation["standalone"].is_null() {
                validate_standalone(&evaluation["standalone"], ARMS[position])?;
            }
        }
    }
    if kind == "cv" {
        ensure!(payload["seed"] == index, "CV seed mismatch");
    } else {
        ensure!(
            payload["index"] == index
                && payload["seed"] == 10000 + index
                && payload["partition_seed"] == 42
                && payload["held_out_fold"] == 0,
            "bootstrap configuration mismatch"
        );
        for key in ["global_minimum_increment", "global_minimum_calibration"] {
            ensure!(
                payload[key].as_f64().is_some_and(f64::is_finite),
                "invalid bootstrap endpoint {key}"
            );
        }
        let increment = arms
            .iter()
            .map(|arm| arm["delta"]["roc_auc"].as_f64().unwrap())
            .fold(f64::INFINITY, f64::min);
        let calibration = arms
            .iter()
            .map(|arm| arm["standalone"]["minimum_delta_auc"].as_f64().unwrap())
            .fold(f64::INFINITY, f64::min);
        ensure!(
            (payload["global_minimum_increment"].as_f64().unwrap() - increment).abs() < 1e-12
                && (payload["global_minimum_calibration"].as_f64().unwrap() - calibration).abs()
                    < 1e-12,
            "global paired minima mismatch"
        );
    }
    Ok(())
}

fn validate_standalone(value: &Value, arm: &str) -> Result<()> {
    ensure!(
        value["strata"] == json!(["bulk", "low_gradient_positive", "high_gradient_positive"]),
        "standalone stratum mismatch"
    );
    let mut vectors = Vec::new();
    for key in ["associator_auc", "pvi_auc", "delta_auc"] {
        let values: Vec<f64> = serde_json::from_value(value[key].clone())?;
        ensure!(
            values.len() == 3
                && values.iter().all(|value| value.is_finite()
                    && (key == "delta_auc" || (0.0..=1.0).contains(value))),
            "invalid standalone vector"
        );
        vectors.push(values);
    }
    for ((associator, pvi), delta) in vectors[0].iter().zip(&vectors[1]).zip(&vectors[2]) {
        ensure!(
            (delta - (associator - pvi)).abs() < 1e-12,
            "standalone delta mismatch"
        );
    }
    let minimum = vectors[2].iter().copied().fold(f64::INFINITY, f64::min);
    ensure!(
        value["minimum_delta_auc"]
            .as_f64()
            .is_some_and(|value| (value - minimum).abs() < 1e-12),
        "standalone minimum mismatch"
    );
    if arm == "frozen_training" {
        ensure!(
            value["frozen_divisor_oracle"]["score_order_preserved"] == true,
            "missing frozen rank oracle"
        );
    }
    Ok(())
}

fn validate_metrics(value: &Value) -> Result<()> {
    let metrics: Metrics = serde_json::from_value(value.clone())?;
    ensure!(
        [metrics.roc_auc, metrics.average_precision, metrics.log_loss]
            .iter()
            .all(|value| value.is_finite()),
        "invalid metric value"
    );
    Ok(())
}

fn validate_metric_pair(baseline: &Value, augmented: &Value, delta: &Value) -> Result<()> {
    let baseline: Metrics = serde_json::from_value(baseline.clone())?;
    let augmented: Metrics = serde_json::from_value(augmented.clone())?;
    let delta: Metrics = serde_json::from_value(delta.clone())?;
    for metrics in [baseline, augmented] {
        ensure!(
            (0.0..=1.0).contains(&metrics.roc_auc)
                && (0.0..=1.0).contains(&metrics.average_precision)
                && metrics.log_loss >= 0.0,
            "metric outside mathematical domain"
        );
    }
    let recomputed = augmented.delta(baseline);
    ensure!(
        (delta.roc_auc - recomputed.roc_auc).abs() < 1e-12
            && (delta.average_precision - recomputed.average_precision).abs() < 1e-12
            && (delta.log_loss - recomputed.log_loss).abs() < 1e-12,
        "paired metric delta mismatch"
    );
    Ok(())
}

fn read_record(path: &Path, identity: &str, kind: &str, index: u64) -> Result<Value> {
    let record: Value = serde_json::from_reader(File::open(path)?)?;
    ensure!(
        record["identity"] == identity && record["kind"] == kind && record["index"] == index,
        "stale or mismatched record {}",
        path.display()
    );
    ensure!(
        record["payload_sha256"] == digest(&serde_json::to_vec(&record["payload"])?),
        "record payload checksum mismatch"
    );
    validate_payload(kind, index, &record["payload"])?;
    Ok(record["payload"].clone())
}

fn record_path(directory: &Path, kind: &str, index: u64) -> PathBuf {
    if kind == "cv" {
        directory.join(format!("cv-seed-{index}.json"))
    } else {
        directory.join(format!("bootstrap-{index:03}.json"))
    }
}

// Record identity and execution inputs stay explicit for resume validation.
#[allow(clippy::too_many_arguments)]
fn execute_record(
    directory: &Path,
    source_root: &Path,
    sources: &Value,
    identity: &str,
    kind: &str,
    index: u64,
    execute: impl FnOnce() -> Result<Value>,
) -> Result<()> {
    ensure!(
        source_identity(source_root)? == *sources,
        "source identity changed before {kind} {index}"
    );
    let output = record_path(directory, kind, index);
    let failure = directory.join(format!("failure-{kind}-{index}.json"));
    ensure!(
        !failure.exists(),
        "retained failure blocks retry: {}",
        failure.display()
    );
    if output.exists() {
        read_record(&output, identity, kind, index)?;
        eprintln!("reusing verified {kind} record {index}");
        return Ok(());
    }
    match execute() {
        Ok(payload) => {
            validate_payload(kind, index, &payload)?;
            ensure!(
                source_identity(source_root)? == *sources,
                "source identity changed during {kind} {index}"
            );
            atomic_json(
                &output,
                &json!({"identity":identity,"kind":kind,"index":index,"payload_sha256":digest(&serde_json::to_vec(&payload)?),"payload":payload}),
            )
        }
        Err(error) => {
            atomic_json(
                &failure,
                &json!({"identity":identity,"kind":kind,"index":index,"error":format!("{error:#}"),"status":"invalid planned replicate; explicit review required"}),
            )?;
            Err(error)
        }
    }
}

fn summarize(directory: &Path, identity: &str) -> Result<Value> {
    let expected: BTreeSet<String> = SEEDS
        .iter()
        .map(|seed| format!("cv-seed-{seed}.json"))
        .chain((0..REPLICATES).map(|index| format!("bootstrap-{index:03}.json")))
        .collect();
    let observed: BTreeSet<String> = fs::read_dir(directory)?
        .map(|entry| entry.map(|entry| entry.file_name().to_string_lossy().into_owned()))
        .collect::<std::io::Result<Vec<_>>>()?
        .into_iter()
        .filter(|name| {
            name.starts_with("cv-seed-")
                || name.starts_with("bootstrap-")
                || name.starts_with("failure-")
        })
        .collect();
    ensure!(
        observed == expected,
        "aggregate requires exact planned file set: missing {:?}, extra {:?}",
        expected.difference(&observed).collect::<Vec<_>>(),
        observed.difference(&expected).collect::<Vec<_>>()
    );
    let cv: Vec<Value> = SEEDS
        .iter()
        .map(|&seed| read_record(&record_path(directory, "cv", seed), identity, "cv", seed))
        .collect::<Result<_>>()?;
    let bootstrap: Vec<Value> = (0..REPLICATES)
        .map(|index| {
            read_record(
                &record_path(directory, "bootstrap", index as u64),
                identity,
                "bootstrap",
                index as u64,
            )
        })
        .collect::<Result<_>>()?;
    let mut arms = Vec::new();
    for (arm_index, arm) in ARMS.iter().enumerate() {
        let increments: Vec<f64> = cv
            .iter()
            .map(|record| {
                record["arms"][arm_index]["out_of_fold_delta"]["roc_auc"]
                    .as_f64()
                    .unwrap()
            })
            .collect();
        let bootstrap_increments: Vec<f64> = bootstrap
            .iter()
            .map(|record| {
                record["arms"][arm_index]["delta"]["roc_auc"]
                    .as_f64()
                    .unwrap()
            })
            .collect();
        let standalone: Vec<f64> = bootstrap
            .iter()
            .map(|record| {
                record["arms"][arm_index]["standalone"]["minimum_delta_auc"]
                    .as_f64()
                    .unwrap()
            })
            .collect();
        arms.push(json!({"arm":arm,"cv_seeds":SEEDS,"cv_delta_roc_auc":increments,"observed_split_minimum":increments.iter().copied().fold(f64::INFINITY,f64::min),"observed_split_maximum":increments.iter().copied().fold(f64::NEG_INFINITY,f64::max),"split_boundary":"Five correlated grouped split sensitivity measurements; range is not a population confidence interval.","primary_fixed_partition_point":cv[0]["arms"][arm_index]["folds"][0],"conditional_increment_descriptive":interval(bootstrap_increments)?,"conditional_standalone_minimum_descriptive":interval(standalone)?}));
    }
    Ok(
        json!({"identity":identity,"completed_cv_seeds":SEEDS,"completed_bootstrap_replicates":REPLICATES,"completed_fits":750,"arms":arms,"global_increment":interval(bootstrap.iter().map(|record|record["global_minimum_increment"].as_f64().unwrap()).collect())?,"global_standalone_margin":interval(bootstrap.iter().map(|record|record["global_minimum_calibration"].as_f64().unwrap()).collect())?,"separate_estimands":"Global increment and global standalone margin answer separate questions; per-arm intervals are descriptive. Average precision and log loss are secondary tradeoffs in individual records.","decision_boundary":"Primary percentile procedure is conditional on the fixed seed42 fold0 partition. An interval containing zero remains inconclusive and establishes no redundancy."}),
    )
}

struct OutputLock(PathBuf);
impl Drop for OutputLock {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_file(&self.0) {
            eprintln!("output lock cleanup failed: {error}");
        }
    }
}

fn admit_protocol(args: &Args, data: &PreparedDataset) -> Result<Value> {
    ensure!(
        hash_file(&args.protocol)? == PROTOCOL_HASH,
        "protocol differs from sealed declaration"
    );
    let protocol: toml::Value = toml::from_str(&fs::read_to_string(&args.protocol)?)?;
    for (actual, expected) in [
        ("scores_sha256", "scores_expected_sha256"),
        ("file_map_sha256", "file_map_expected_sha256"),
        ("catalog_sha256", "catalog_expected_sha256"),
    ] {
        ensure!(
            data.evidence[actual].as_str() == protocol["inputs"][expected].as_str(),
            "admission hash mismatch {actual}"
        );
    }
    for (actual, expected) in [
        ("exported_files", "expected_legacy_files"),
        ("rows_before_warmup", "expected_legacy_rows"),
        ("positives_before_warmup", "expected_legacy_positives"),
    ] {
        ensure!(
            data.evidence[actual].as_u64()
                == protocol["inputs"][expected]
                    .as_integer()
                    .map(|value| value as u64),
            "admission count mismatch {actual}"
        );
    }
    ensure!(
        hash_file(
            &args.input_root.join(
                protocol["inputs"]["matched_manifest"]
                    .as_str()
                    .context("missing matched manifest")?
            )
        )? == protocol["inputs"]["matched_manifest_sha256"]
            .as_str()
            .context("missing manifest hash")?,
        "matched manifest hash mismatch"
    );
    ensure!(
        data.files.iter().all(|file| file.retained_rows > 0),
        "common population loses an entire planned file"
    );
    validate_sample_order(&data.files)?;
    ensure!(
        data.labels.len() <= u32::MAX as usize,
        "row indices exceed u32 representation"
    );
    Ok(
        json!({"protocol_sha256":PROTOCOL_HASH,"rolling_increment_count":256,"cv_seeds":SEEDS,"fold_count":5,"bootstrap_replicates":REPLICATES,"bootstrap_seed_start":10000,"arms":ARMS,"ridge":RIDGE,"log_epsilon":EPSILON,"rayon_threads":6}),
    )
}

fn retain_preparation_result(
    args: &Args,
    sources: &Value,
    result: Result<PreparedDataset>,
) -> Result<PreparedDataset> {
    result.map_err(|error| {
        let inputs: Vec<Value> = [
            ("scores", &args.scores),
            ("file_map", &args.file_map),
            ("catalog", &args.catalog),
        ]
        .into_iter()
        .map(|(role, path)| match hash_file(path) {
            Ok(hash) => json!({"role":role,"path":path,"sha256":hash}),
            Err(error) => json!({"role":role,"path":path,"identity_error":format!("{error:#}")}),
        })
        .collect();
        let evidence = json!({
            "status":"preparation_rejected_before_inference",
            "protocol_sha256":PROTOCOL_HASH,
            "sources":sources,
            "input_root":args.input_root,
            "inputs":inputs,
            "identity_boundary":"Input hashes observed after preparation failed; raw-file identities and partial dataset measurements are unavailable. Inputs may have changed during preparation.",
            "data":null,
            "files":null,
            "error":format!("{error:#}"),
        });
        retain_failure(&args.out_dir, &evidence, error)
    })
}

fn retain_failure(directory: &Path, evidence: &Value, error: anyhow::Error) -> anyhow::Error {
    let retained = (|| -> Result<PathBuf> {
        let identity = digest(&serde_json::to_vec(evidence)?);
        let destination = directory.join(format!("failure-admission-{identity}.json"));
        if destination.exists() {
            let existing: Value = serde_json::from_reader(File::open(&destination)?)?;
            ensure!(
                existing == *evidence,
                "retained admission evidence checksum collision"
            );
        } else {
            atomic_json(&destination, evidence)?;
        }
        Ok(destination)
    })();
    match retained {
        Ok(destination) => error.context(format!(
            "admission diagnostics retained at {}",
            destination.display()
        )),
        Err(retention_error) => error.context(format!(
            "admission diagnostic retention failed: {retention_error:#}"
        )),
    }
}

fn retain_admission_result(
    directory: &Path,
    sources: &Value,
    data: &PreparedDataset,
    result: Result<Value>,
) -> Result<Value> {
    match result {
        Ok(configuration) => Ok(configuration),
        Err(error) => {
            let evidence = json!({
                "status": "admission_rejected_before_inference",
                "protocol_sha256": PROTOCOL_HASH,
                "sources": sources,
                "data": data.evidence,
                "files": data.files,
                "error": format!("{error:#}"),
            });
            Err(retain_failure(directory, &evidence, error))
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.cv_seed.is_none_or(|seed| SEEDS.contains(&seed)),
        "CV seed outside sealed plan"
    );
    ensure!(
        args.bootstrap_index.is_none_or(|index| index < REPLICATES),
        "bootstrap index outside sealed plan"
    );
    ensure!(
        hash_file(&args.protocol)? == PROTOCOL_HASH,
        "protocol differs from sealed declaration"
    );
    rayon::ThreadPoolBuilder::new()
        .num_threads(6)
        .build_global()?;
    fs::create_dir_all(&args.out_dir)?;
    let lock_path = args.out_dir.join(".runner.lock");
    let mut lock_file = OpenOptions::new().write(true).create_new(true).open(&lock_path).context("output directory is locked; inspect retained process and lock before explicit recovery")?;
    writeln!(lock_file, "{}", std::process::id())?;
    let _lock = OutputLock(lock_path);
    let root = source_root()?;
    let sources = source_identity(&root)?;
    let preparation = prepare(
        &args.input_root,
        &args.scores,
        &args.file_map,
        &args.catalog,
        256,
    );
    let data = retain_preparation_result(&args, &sources, preparation)?;
    let configuration =
        retain_admission_result(&args.out_dir, &sources, &data, admit_protocol(&args, &data))?;
    let mut positive_gradients: Vec<f64> = data
        .dbdt
        .iter()
        .zip(&data.labels)
        .filter_map(|(&gradient, &label)| (label == 1).then_some(gradient))
        .collect();
    positive_gradients.par_sort_unstable_by(f64::total_cmp);
    let threshold = linear_percentile(&positive_gradients, 0.5)?;
    let provenance = json!({"configuration":configuration,"sources":sources,"data":data.evidence,"files":data.files,"common_population_positive_gradient_median":threshold,"gradient_boundary":"Median dbdt of positive rows after common warmup; fixed descriptive benchmark threshold, not the original full-corpus threshold.","executable_sha256":hash_file(&std::env::current_exe()?)?});
    let identity = digest(&serde_json::to_vec(&provenance)?);
    let dataset_path = args.out_dir.join("dataset.json");
    let dataset_record = json!({"identity":identity,"provenance":provenance});
    if dataset_path.exists() {
        let existing: Value = serde_json::from_reader(File::open(&dataset_path)?)?;
        ensure!(
            existing == dataset_record,
            "output directory contains stale dataset/source/protocol identity"
        );
    } else {
        atomic_json(&dataset_path, &dataset_record)?;
    }
    if args.prepare_only {
        return Ok(());
    }
    if matches!(args.mode, Mode::All | Mode::Cv) {
        for seed in SEEDS
            .into_iter()
            .filter(|seed| args.cv_seed.is_none_or(|selected| selected == *seed))
        {
            execute_record(
                &args.out_dir,
                &root,
                &sources,
                &identity,
                "cv",
                seed,
                || run_cv(&data, seed, threshold),
            )?;
        }
    }
    if matches!(args.mode, Mode::All | Mode::Bootstrap) {
        let file_ids: Vec<u16> = data.files.iter().map(|file| file.id).collect();
        let primary_folds = folds(&file_ids, 42);
        let test_ids = &primary_folds[0];
        let test_set: BTreeSet<u16> = test_ids.iter().copied().collect();
        let train_ids: Vec<u16> = file_ids
            .iter()
            .copied()
            .filter(|id| !test_set.contains(id))
            .collect();
        let test_rows = selected_rows(&data, &unit_counts(test_ids, data.files.len()));
        let rankings = Rankings::new(&data, &test_rows);
        for index in (0..REPLICATES).filter(|index| {
            args.bootstrap_index
                .is_none_or(|selected| selected == *index)
        }) {
            execute_record(
                &args.out_dir,
                &root,
                &sources,
                &identity,
                "bootstrap",
                index as u64,
                || {
                    run_bootstrap(
                        &data, index, threshold, &rankings, &train_ids, test_ids, &test_rows,
                    )
                },
            )?;
        }
    }
    if matches!(args.mode, Mode::All | Mode::Summarize)
        && args.cv_seed.is_none()
        && args.bootstrap_index.is_none()
    {
        let summary = summarize(&args.out_dir, &identity)?;
        let destination = args.out_dir.join("summary.json");
        if destination.exists() {
            let existing: Value = serde_json::from_reader(File::open(&destination)?)?;
            ensure!(
                existing == summary,
                "retained summary differs from validated records"
            );
        } else {
            atomic_json(&destination, &summary)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gororoba_cli_physics::staple_calibration::FileEvidence;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn fixture() -> PreparedDataset {
        let files = (0..2)
            .map(|id| FileEvidence {
                id,
                path: format!("file-{id}"),
                increment_square_sum: if id == 0 { 8.0 } else { 36.0 },
                increment_count: if id == 0 { 2 } else { 4 },
                sha256: String::new(),
                finite_samples: 6,
                discarded_raw_rows: 0,
                scored_rows: 6,
                retained_rows: 6,
                first_timestamp: String::new(),
                last_timestamp: String::new(),
                min_cadence_milliseconds: 1000,
                max_cadence_milliseconds: 1000,
                nonpositive_cadence_count: 0,
                equal_timestamp_pair_count: 0,
                backward_timestamp_pair_count: 0,
                positive_submillisecond_pair_count: 0,
                kept_index_label_sha256: String::new(),
            })
            .collect();
        PreparedDataset {
            features: (0..12)
                .flat_map(|row| (0..STRIDE).map(move |feature| ((row + feature + 1) as f32).ln()))
                .collect(),
            labels: (0..12).map(|row| (row % 2) as u8).collect(),
            file_index: (0..12).map(|row| (row / 6) as u16).collect(),
            pvi_numerator: (0..12).map(|row| row as f64).collect(),
            raw_assoc: vec![1.0; 12],
            daily_pvi: vec![2.0; 12],
            rolling_pvi: vec![3.0; 12],
            rolling_log_pvi: vec![3.0f32.ln(); 12],
            dbdt: vec![1.0; 12],
            files,
            evidence: Value::Null,
        }
    }

    #[test]
    fn seeded_folds_and_resampled_clusters_preserve_membership() {
        let ids: Vec<u16> = (0..813).collect();
        let assignment = folds(&ids, 42);
        assert_eq!(assignment, folds(&ids, 42));
        assert_ne!(assignment, folds(&ids, 43));
        let mut all: Vec<u16> = assignment.iter().flatten().copied().collect();
        all.sort_unstable();
        assert_eq!(all, ids);
        let test = &assignment[0];
        let train: Vec<u16> = ids
            .iter()
            .copied()
            .filter(|id| !test.contains(id))
            .collect();
        let mut random = ChaCha8Rng::seed_from_u64(10000);
        let training = draw_counts(&train, 813, &mut random);
        let testing = draw_counts(test, 813, &mut random);
        assert_eq!(training.iter().sum::<u32>(), train.len() as u32);
        assert_eq!(testing.iter().sum::<u32>(), test.len() as u32);
        assert!(training.iter().any(|&count| count > 1));
        assert!(
            training
                .iter()
                .zip(&testing)
                .all(|(left, right)| *left == 0 || *right == 0)
        );
    }

    #[test]
    fn multiplicities_control_rms_and_training_standardization() {
        let data = fixture();
        let counts = [2, 1];
        let rows = selected_rows(&data, &counts);
        assert_eq!(rows.len(), 18);
        assert_eq!(rows.iter().filter(|&&row| row == 0).count(), 2);
        assert_eq!(rows.iter().filter(|&&row| row == 6).count(), 1);
        assert!((training_rms(&data, &counts).unwrap() - (52.0f64 / 8.0).sqrt()).abs() < 1e-12);
        let training = selected_rows(&data, &[1, 0]);
        let before = standardize(&data.features, &training).unwrap();
        let mut changed = data.features.clone();
        changed[6 * STRIDE..].fill(1e20);
        assert_eq!(before, standardize(&changed, &training).unwrap());
        assert!(training_rms(&data, &[0, 0]).is_err());
    }

    #[test]
    fn calibration_replaces_only_pvi_column_and_raw_divisor_preserves_ties() {
        let data = fixture();
        let mut features = data.features.clone();
        replace_calibration(&mut features, &data, "frozen_training", 2.0).unwrap();
        for row in 0..12 {
            for column in 0..STRIDE {
                if column == PVI_COLUMN {
                    assert_eq!(
                        features[row * STRIDE + column],
                        (data.pvi_numerator[row] / 2.0 + EPSILON).ln() as f32
                    );
                } else {
                    assert_eq!(
                        features[row * STRIDE + column],
                        data.features[row * STRIDE + column]
                    );
                }
            }
        }
        replace_calibration(&mut features, &data, "prewindow_rolling", 2.0).unwrap();
        assert!(
            (0..12).all(|row| features[row * STRIDE + PVI_COLUMN] == data.rolling_log_pvi[row])
        );
        replace_calibration(&mut features, &data, "daily_file", 2.0).unwrap();
        assert_eq!(features, data.features);
        let ranking = Ranking {
            scores: vec![4.0, 2.0, 2.0, 0.0],
            labels: vec![1, 0, 1, 0],
            files: vec![0; 4],
            gradient: vec![0.0; 4],
        };
        let oracle = ranking.positive_divisor_oracle(2.0).unwrap();
        assert_eq!(oracle["zero_numerators"], 1);
        assert_eq!(oracle["ranking_and_tie_partition_preserved"], true);
        assert_eq!(
            weighted_auc(&ranking.scores, &ranking.labels, &[1.0; 4]),
            weighted_auc(&[2.0, 1.0, 1.0, 0.0], &ranking.labels, &[1.0; 4])
        );
    }

    fn valid_bootstrap() -> Value {
        let ids: Vec<u16> = (0..813).collect();
        let assignment = folds(&ids, 42);
        let test = &assignment[0];
        let train: Vec<u16> = ids
            .iter()
            .copied()
            .filter(|id| !test.contains(id))
            .collect();
        let mut random = ChaCha8Rng::seed_from_u64(10000);
        let train_counts = draw_counts(&train, 813, &mut random);
        let test_counts = draw_counts(test, 813, &mut random);
        let arms:Vec<Value>=ARMS.iter().map(|arm|json!({"arm":arm,"training_rms":2.0,"training_mean":vec![0.0;7],"training_population_std":vec![1.0;7],"baseline":{"roc_auc":0.5,"average_precision":0.5,"log_loss":0.7},"augmented":{"roc_auc":0.6,"average_precision":0.6,"log_loss":0.6},"delta":{"roc_auc":0.1,"average_precision":0.1,"log_loss":-0.1},"fits":[{"feature_count":6,"coefficients":vec![0.0;7],"converged":true,"iterations":2,"penalized_deviance_trajectory":[1.0,1.0]},{"feature_count":7,"coefficients":vec![0.0;8],"converged":true,"iterations":2,"penalized_deviance_trajectory":[1.0,1.0]}],"standalone":{"strata":["bulk","low_gradient_positive","high_gradient_positive"],"associator_auc":[0.6,0.6,0.6],"pvi_auc":[0.5,0.5,0.5],"delta_auc":[0.1,0.1,0.1],"minimum_delta_auc":0.1,"frozen_divisor_oracle":{"score_order_preserved":true,"ranking_and_tie_partition_preserved":true}}})).collect();
        json!({"index":0,"seed":10000,"partition_seed":42,"held_out_fold":0,"train_file_counts":train_counts,"test_file_counts":test_counts,"arms":arms,"global_minimum_increment":0.1,"global_minimum_calibration":0.1})
    }

    #[test]
    fn record_validation_rejects_changed_partitions_missing_arms_and_false_minima() {
        let valid = valid_bootstrap();
        validate_payload("bootstrap", 0, &valid).unwrap();
        let mut changed = valid.clone();
        changed["train_file_counts"][0] = json!(999);
        assert!(validate_payload("bootstrap", 0, &changed).is_err());
        let mut changed = valid.clone();
        changed["arms"][1]["standalone"] = Value::Null;
        assert!(validate_payload("bootstrap", 0, &changed).is_err());
        let mut changed = valid.clone();
        changed["global_minimum_increment"] = json!(0.2);
        assert!(validate_payload("bootstrap", 0, &changed).is_err());
        let mut changed = valid.clone();
        changed["arms"].as_array_mut().unwrap().pop();
        assert!(validate_payload("bootstrap", 0, &changed).is_err());
        let mut changed = valid;
        changed["arms"][0]["fits"][0]["converged"] = json!(false);
        assert!(validate_payload("bootstrap", 0, &changed).is_err());
    }

    #[test]
    fn linear_intervals_require_complete_replicates_and_keep_zero_inconclusive() {
        assert_eq!(linear_percentile(&[0.0, 10.0], 0.025).unwrap(), 0.25);
        assert!(interval(vec![1.0; 99]).is_err());
        assert_eq!(
            interval(vec![0.0; 100]).unwrap()["decision"],
            "inconclusive"
        );
        assert_eq!(
            interval(vec![1.0; 100]).unwrap()["adverse_nonpositive_count"],
            0
        );
    }

    #[test]
    fn finite_division_tie_merges_are_measured_and_scored() {
        let smallest = f64::from_bits(1);
        let ranking = Ranking {
            scores: vec![2.0 * smallest, smallest, 0.0],
            labels: vec![1, 0, 1],
            files: vec![0; 3],
            gradient: vec![0.0; 3],
        };
        let oracle = ranking.positive_divisor_oracle(2.0).unwrap();
        assert_eq!(oracle["score_order_preserved"], true);
        assert_eq!(oracle["merged_adjacent_pairs"], 1);
        assert_eq!(oracle["ranking_and_tie_partition_preserved"], false);
        let divided: Vec<f64> = ranking.scores.iter().map(|value| value / 2.0).collect();
        assert_ne!(
            weighted_auc(&ranking.scores, &ranking.labels, &[1.0; 3]),
            weighted_auc(&divided, &ranking.labels, &[1.0; 3])
        );
    }

    #[test]
    fn cv_records_validate_paired_out_of_fold_results_and_primary_standalone() {
        let bootstrap = valid_bootstrap();
        let arms:Vec<Value>=bootstrap["arms"].as_array().unwrap().iter().map(|arm|json!({"arm":arm["arm"],"folds":vec![arm.clone();5],"out_of_fold_baseline":arm["baseline"],"out_of_fold_augmented":arm["augmented"],"out_of_fold_delta":arm["delta"]})).collect();
        let ids: Vec<u16> = (0..813).collect();
        let valid = json!({"seed":42,"file_assignments":folds(&ids,42),"arms":arms});
        validate_payload("cv", 42, &valid).unwrap();
        let mut changed = valid.clone();
        changed["arms"][0]["out_of_fold_delta"]["roc_auc"] = json!(0.2);
        assert!(validate_payload("cv", 42, &changed).is_err());
        let mut changed = valid;
        changed["arms"][0]["folds"][0]["standalone"] = Value::Null;
        assert!(validate_payload("cv", 42, &changed).is_err());
    }

    #[test]
    fn aggregate_rejects_missing_and_unplanned_records() {
        let directory =
            std::env::temp_dir().join(format!("staples-aggregate-test-{}", std::process::id()));
        fs::create_dir(&directory).unwrap();
        assert!(summarize(&directory, "identity").is_err());
        atomic_json(&directory.join("bootstrap-100.json"), &json!({})).unwrap();
        let error = summarize(&directory, "identity").unwrap_err().to_string();
        assert!(error.contains("bootstrap-100.json"));
        fs::remove_file(directory.join("bootstrap-100.json")).unwrap();
        fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn preparation_errors_retain_available_identity_and_original_error() {
        let directory = std::env::temp_dir().join(format!(
            "staples-preparation-failure-{}",
            std::process::id()
        ));
        fs::create_dir(&directory).unwrap();
        let args = Args {
            input_root: directory.clone(),
            scores: directory.join("scores.csv"),
            file_map: directory.join("files.csv"),
            catalog: directory.join("catalog.csv"),
            out_dir: directory.clone(),
            protocol: directory.join("protocol.toml"),
            prepare_only: true,
            mode: Mode::All,
            cv_seed: None,
            bootstrap_index: None,
        };
        fs::write(&args.file_map, "file_id,path\n0,missing-raw.csv\n").unwrap();
        fs::write(&args.catalog, "TIMESTAMP\n2020-01-01T00:00:00Z\n").unwrap();
        let sources = json!({"source":"digest"});
        for (header, expected) in [
            ("wrong\n", "unexpected retained score header"),
            (
                "file_id,assoc,dbdt,rot,bmag,label,cumrot6,maxrot6,pvi6,gram6,scram,chperm\n",
                "reconstruct file 0: missing-raw.csv",
            ),
        ] {
            fs::write(&args.scores, header).unwrap();
            let result = prepare(
                &args.input_root,
                &args.scores,
                &args.file_map,
                &args.catalog,
                256,
            );
            let error = retain_preparation_result(&args, &sources, result)
                .err()
                .unwrap();
            assert!(format!("{error:#}").contains(expected));
            assert!(error.to_string().contains("diagnostics retained"));
        }
        let records: Vec<PathBuf> = fs::read_dir(&directory)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.extension()
                    .is_some_and(|extension| extension == "json")
            })
            .collect();
        assert_eq!(records.len(), 2);
        for path in records {
            let record: Value = serde_json::from_reader(File::open(&path).unwrap()).unwrap();
            assert_eq!(record["status"], "preparation_rejected_before_inference");
            assert!(record["data"].is_null());
            assert!(record["files"].is_null());
            assert_eq!(
                record["inputs"][1]["sha256"],
                hash_file(&args.file_map).unwrap()
            );
            assert_eq!(record["sources"], sources);
        }
        let error = retain_failure(
            &directory.join("missing-parent"),
            &json!({}),
            anyhow::anyhow!("original rejection"),
        );
        assert!(format!("{error:#}").contains("original rejection"));
        assert!(error.to_string().contains("retention failed"));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn rejected_admission_retains_cadence_evidence_and_distinct_failures() {
        let directory = std::env::temp_dir().join(format!(
            "staples-admission-failure-test-{}",
            std::process::id()
        ));
        fs::create_dir(&directory).unwrap();
        let mut data = fixture();
        data.files[1].nonpositive_cadence_count = 3;
        data.evidence = json!({"exported_files":2,"rows_before_warmup":12});
        let sources = json!({"scientific_source_sha256":"source-digest"});
        for message in [
            "chronological gate rejected",
            "chronological gate rejected",
            "different rejection",
        ] {
            let result =
                retain_admission_result(&directory, &sources, &data, Err(anyhow::anyhow!(message)));
            assert!(result.is_err());
        }
        let retained: Vec<PathBuf> = fs::read_dir(&directory)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect();
        assert_eq!(retained.len(), 2);
        for path in retained {
            let record: Value = serde_json::from_reader(File::open(&path).unwrap()).unwrap();
            assert_eq!(record["status"], "admission_rejected_before_inference");
            assert_eq!(record["files"][1]["nonpositive_cadence_count"], 3);
            assert_eq!(record["data"], data.evidence);
            assert_eq!(record["sources"], sources);
            assert!(
                path.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with("failure-admission-")
            );
            fs::remove_file(path).unwrap();
        }
        fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn serialized_float_checksums_survive_reload_and_stale_identity_is_rejected() {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "staples-record-{}-{}.json",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut payload = valid_bootstrap();
        payload["roundtrip_probe"] = json!(2.4641099530828617e-9);
        let record = json!({"identity":"expected","kind":"bootstrap","index":0,"payload_sha256":digest(&serde_json::to_vec(&payload).unwrap()),"payload":payload});
        atomic_json(&path, &record).unwrap();
        let reloaded = read_record(&path, "expected", "bootstrap", 0).unwrap();
        assert_eq!(reloaded, payload);
        assert!(read_record(&path, "changed-protocol-or-source", "bootstrap", 0).is_err());
        assert!(atomic_json(&path, &record).is_err());
        fs::remove_file(path).unwrap();
        let mut random = ChaCha8Rng::seed_from_u64(7);
        for _ in 0..10000 {
            let value = json!({"value":random.random::<f64>()});
            let encoded = serde_json::to_vec(&value).unwrap();
            let restored: Value = serde_json::from_slice(&encoded).unwrap();
            assert_eq!(encoded, serde_json::to_vec(&restored).unwrap());
        }
    }
}

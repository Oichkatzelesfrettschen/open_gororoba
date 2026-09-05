//! Sealed protocol admission and resumable evidence records.

use anyhow::{Context, Result, ensure};
use gororoba_cli_physics::{
    staple_calibration::{PreparedDataset, validate_sample_order},
    staple_logistic::DEVIANCE_REL_TOL,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
};

use super::{
    ARMS, Args, EPSILON, PROTOCOL_HASH, REPLICATES, RIDGE, SEEDS, SOURCE_FILES, STRIDE,
    metrics::{Metrics, interval},
    splitting::{draw_counts, folds},
};

pub(super) fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub(super) fn hash_file(path: &Path) -> Result<String> {
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

pub(super) fn source_root() -> Result<PathBuf> {
    std::env::current_dir()?
        .ancestors()
        .find(|path| {
            path.join("crates/gororoba_cli_physics/src/staple_logistic.rs")
                .is_file()
        })
        .map(Path::to_path_buf)
        .context("launch runner from its source checkout")
}

pub(super) fn source_identity(root: &Path) -> Result<Value> {
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

pub(super) fn atomic_json(path: &Path, value: &Value) -> Result<()> {
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

pub(super) fn validate_payload(kind: &str, index: u64, payload: &Value) -> Result<()> {
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

pub(super) fn read_record(path: &Path, identity: &str, kind: &str, index: u64) -> Result<Value> {
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
pub(super) fn execute_record(
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

pub(super) fn summarize(directory: &Path, identity: &str) -> Result<Value> {
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

pub(super) struct OutputLock(pub(super) PathBuf);
impl Drop for OutputLock {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_file(&self.0) {
            eprintln!("output lock cleanup failed: {error}");
        }
    }
}

pub(super) fn admit_protocol(args: &Args, data: &PreparedDataset) -> Result<Value> {
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

pub(super) fn retain_preparation_result(
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

pub(super) fn retain_failure(
    directory: &Path,
    evidence: &Value,
    error: anyhow::Error,
) -> anyhow::Error {
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

pub(super) fn retain_admission_result(
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

//! Independently validate retained grouped-refit records without refitting models.
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, error::Error, fs, path::Path, process::Command};

type AuditResult<T> = Result<T, Box<dyn Error>>;
const ARMS: [&str; 3] = ["daily_file", "frozen_training", "prewindow_rolling"];
fn number(value: &Value) -> f64 {
    let number = value.as_f64().expect("numeric field");
    assert!(number.is_finite());
    number
}
fn near(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-12 * (1.0 + expected.abs()),
        "numeric mismatch {actual} versus {expected}"
    );
}
fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
fn read(path: &Path) -> AuditResult<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
fn assignments(seed: u64) -> Vec<Vec<u16>> {
    let mut ids: Vec<u16> = (0..813).collect();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    for position in (1..ids.len()).rev() {
        let selected = rng.random_range(0..=position);
        ids.swap(position, selected);
    }
    let mut folds = vec![Vec::new(); 5];
    for (position, id) in ids.into_iter().enumerate() {
        folds[position % 5].push(id);
    }
    for fold in &mut folds {
        fold.sort_unstable();
    }
    folds
}
fn sampled(ids: &[u16], rng: &mut ChaCha8Rng) -> Vec<u32> {
    let mut counts = vec![0; 813];
    for _ in 0..ids.len() {
        counts[usize::from(ids[rng.random_range(0..ids.len())])] += 1;
    }
    counts
}
fn check_metrics(baseline: &Value, augmented: &Value, delta: &Value) {
    for key in ["roc_auc", "average_precision", "log_loss"] {
        let left = number(&baseline[key]);
        let right = number(&augmented[key]);
        if key != "log_loss" {
            assert!((0.0..=1.0).contains(&left) && (0.0..=1.0).contains(&right));
        } else {
            assert!(left >= 0.0 && right >= 0.0);
        }
        near(number(&delta[key]), right - left);
    }
}
fn check_evaluation(evaluation: &Value, counts: &[u32], files: &[Value]) -> usize {
    let total_rows: u64 = files
        .iter()
        .enumerate()
        .map(|(index, file)| file["retained_rows"].as_u64().unwrap() * u64::from(counts[index]))
        .sum();
    assert_eq!(evaluation["training_rows_with_multiplicity"], total_rows);
    let mut square_sum = 0.0;
    let mut increment_count = 0.0;
    for (index, file) in files.iter().enumerate() {
        square_sum += number(&file["increment_square_sum"]) * f64::from(counts[index]);
        increment_count += number(&file["increment_count"]) * f64::from(counts[index]);
    }
    near(
        number(&evaluation["training_rms"]),
        (square_sum / increment_count).sqrt(),
    );
    for key in ["training_mean", "training_population_std"] {
        let values = evaluation[key].as_array().unwrap();
        assert_eq!(values.len(), 7);
        for value in values {
            let entry = number(value);
            if key == "training_population_std" {
                assert!(entry > 0.0);
            }
        }
    }
    check_metrics(
        &evaluation["baseline"],
        &evaluation["augmented"],
        &evaluation["delta"],
    );
    let fits = evaluation["fits"].as_array().unwrap();
    assert_eq!(fits.len(), 2);
    for (index, fit) in fits.iter().enumerate() {
        assert_eq!(fit["feature_count"], index + 6);
        assert_eq!(fit["converged"], true);
        let coefficients = fit["coefficients"].as_array().unwrap();
        assert_eq!(coefficients.len(), index + 7);
        for coefficient in coefficients {
            number(coefficient);
        }
        let trajectory = fit["penalized_deviance_trajectory"].as_array().unwrap();
        assert!((2..=25).contains(&trajectory.len()));
        assert_eq!(fit["iterations"], trajectory.len());
        for value in trajectory {
            assert!(number(value) >= 0.0);
        }
        let prior = number(&trajectory[trajectory.len() - 2]);
        let last = number(trajectory.last().unwrap());
        assert!(
            (prior - last).abs() / prior.abs().max(1.0) < 1e-8,
            "convergence contradiction"
        );
    }
    fits.len()
}
fn check_standalone(value: &Value, frozen: bool) -> f64 {
    assert_eq!(
        value["strata"],
        json!(["bulk", "low_gradient_positive", "high_gradient_positive"])
    );
    for key in ["associator_auc", "pvi_auc", "delta_auc"] {
        assert_eq!(value[key].as_array().unwrap().len(), 3);
    }
    let mut minimum = f64::INFINITY;
    for index in 0..3 {
        let associator = number(&value["associator_auc"][index]);
        let pvi = number(&value["pvi_auc"][index]);
        assert!((0.0..=1.0).contains(&associator) && (0.0..=1.0).contains(&pvi));
        let delta = number(&value["delta_auc"][index]);
        near(delta, associator - pvi);
        minimum = minimum.min(delta);
    }
    near(number(&value["minimum_delta_auc"]), minimum);
    if frozen {
        let oracle = &value["frozen_divisor_oracle"];
        assert_eq!(oracle["score_order_preserved"], true);
        assert_eq!(
            oracle["ranking_and_tie_partition_preserved"],
            oracle["merged_adjacent_pairs"].as_u64().unwrap() == 0
        );
        for index in 0..3 {
            near(
                number(&oracle["calibrated_minus_raw_auc"][index]),
                number(&value["pvi_auc"][index]) - number(&oracle["raw_numerator_auc"][index]),
            );
        }
    }
    minimum
}
fn record(path: &Path, identity: &str, kind: &str, index: u64) -> AuditResult<Value> {
    let record = read(path)?;
    assert_eq!(record["identity"], identity);
    assert_eq!(record["kind"], kind);
    assert_eq!(record["index"], index);
    assert_eq!(
        record["payload_sha256"],
        hash(&serde_json::to_vec(&record["payload"])?)
    );
    Ok(record["payload"].clone())
}
fn quantile(sorted: &[f64], probability: f64) -> f64 {
    let position = probability * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    sorted[lower] * (1.0 - position.fract()) + sorted[upper] * position.fract()
}
fn check_interval(summary: &Value, values: &mut [f64]) -> Value {
    assert_eq!(values.len(), 100);
    values.sort_by(f64::total_cmp);
    let lower = quantile(values, 0.025);
    let upper = quantile(values, 0.975);
    let median = quantile(values, 0.5);
    near(number(&summary["percentile_2_5"]), lower);
    near(number(&summary["percentile_97_5"]), upper);
    near(number(&summary["median"]), median);
    let adverse = values.iter().filter(|&&value| value <= 0.0).count();
    assert_eq!(summary["adverse_nonpositive_count"], adverse);
    assert_eq!(summary["replicates"], 100);
    assert_eq!(summary["denominator"], 100);
    near(number(&summary["monte_carlo_resolution"]), 1.0 / 101.0);
    assert_eq!(
        summary["decision"],
        if lower > 0.0 {
            "positive_under_conditional_procedure"
        } else if upper < 0.0 {
            "negative_under_conditional_procedure"
        } else {
            "inconclusive"
        }
    );
    json!({"lower":lower,"upper":upper,"median":median,"adverse_nonpositive":adverse})
}
fn options(arguments: &[String]) -> AuditResult<(bool, Option<&str>)> {
    let mut complete = false;
    let mut source_ref = None;
    let mut arguments = arguments.iter();
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--complete" if !complete => complete = true,
            "--source-ref" if source_ref.is_none() => {
                let reference = arguments
                    .next()
                    .ok_or("--source-ref requires a full commit ID")?;
                if reference.len() != 40 || !reference.bytes().all(|byte| byte.is_ascii_hexdigit())
                {
                    return Err("--source-ref requires a full 40-digit commit ID".into());
                }
                source_ref = Some(reference.as_str());
            }
            _ => return Err(format!("unexpected or repeated argument: {argument}").into()),
        }
    }
    Ok((complete, source_ref))
}

fn source_bytes(root: &Path, path: &str, source_ref: Option<&str>) -> AuditResult<Vec<u8>> {
    match source_ref {
        None => Ok(fs::read(root.join(path))?),
        Some(reference) => {
            let output = Command::new("git")
                .arg("-C")
                .arg(root)
                .args([
                    "--no-replace-objects",
                    "show",
                    &format!("{reference}^{{commit}}:{path}"),
                ])
                .output()?;
            if !output.status.success() {
                return Err(format!(
                    "read source {reference}:{path}: {}",
                    String::from_utf8_lossy(&output.stderr)
                )
                .into());
            }
            Ok(output.stdout)
        }
    }
}

#[test]
fn source_reference_options_are_explicit_and_strict() {
    let parse = |arguments: &[&str]| {
        options(
            &arguments
                .iter()
                .map(|argument| (*argument).to_owned())
                .collect::<Vec<_>>(),
        )
        .map(|(complete, reference)| (complete, reference.map(str::to_owned)))
    };
    assert_eq!(parse(&[]).unwrap(), (false, None));
    assert_eq!(parse(&["--complete"]).unwrap(), (true, None));
    let reference = "291ce5023d4ae8ca30c5ee4535bd9152520233ef";
    assert_eq!(
        parse(&["--source-ref", reference, "--complete"]).unwrap(),
        (true, Some(reference.to_owned()))
    );
    for arguments in [
        vec!["--source-ref"],
        vec!["--source-ref", "HEAD"],
        vec!["--complet"],
        vec!["--complete", "--complete"],
        vec!["--source-ref", reference, "--source-ref", reference],
    ] {
        assert!(parse(&arguments).is_err());
    }
}

fn main() -> AuditResult<()> {
    let args: Vec<String> = std::env::args().collect();
    let root = Path::new(args.get(1).ok_or("source checkout argument required")?);
    let (complete, source_ref) = options(&args[2..])?;
    let audit = root.join("data/output/audit/staples-calibration-grouped-refits");
    let results = audit.join("results");
    let dataset = read(&results.join("dataset.json"))?;
    let identity = dataset["identity"].as_str().unwrap();
    assert_eq!(identity, hash(&serde_json::to_vec(&dataset["provenance"])?));
    for (path, expected) in dataset["provenance"]["sources"].as_object().unwrap() {
        assert_eq!(
            expected.as_str().unwrap(),
            hash(&source_bytes(root, path, source_ref)?),
            "source hash mismatch {path}"
        );
    }
    eprintln!(
        "Verified production source hashes against {}",
        source_ref.unwrap_or("live checkout")
    );
    assert_eq!(
        dataset["provenance"]["configuration"]["protocol_sha256"],
        hash(&fs::read(audit.join("protocol.toml"))?)
    );
    let files = dataset["provenance"]["files"].as_array().unwrap();
    assert_eq!(files.len(), 813);
    for (index, file) in files.iter().enumerate() {
        assert_eq!(file["id"], index);
    }
    let names: BTreeSet<String> = fs::read_dir(&results)?
        .map(|entry| entry.map(|entry| entry.file_name().to_string_lossy().into_owned()))
        .collect::<Result<_, _>>()?;
    assert!(
        !names.iter().any(|name| name.starts_with("failure-")),
        "retained failure blocks completion"
    );
    let expected: BTreeSet<String> = (42..=46)
        .map(|seed| format!("cv-seed-{seed}.json"))
        .chain((0..100).map(|index| format!("bootstrap-{index:03}.json")))
        .collect();
    let observed: BTreeSet<String> = names
        .iter()
        .filter(|name| {
            (name.starts_with("cv-seed-") || name.starts_with("bootstrap-"))
                && name.ends_with(".json")
        })
        .cloned()
        .collect();
    assert!(observed.is_subset(&expected), "unplanned record");
    if complete {
        assert_eq!(observed, expected, "incomplete planned record set");
    }
    let primary = assignments(42);
    let test_ids = &primary[0];
    let train_ids: Vec<u16> = (0..813).filter(|id| !test_ids.contains(id)).collect();
    let mut fit_count = 0usize;
    let mut cv_records = Vec::new();
    let mut bootstrap_records = Vec::new();
    for seed in 42..=46 {
        let payload = record(
            &results.join(format!("cv-seed-{seed}.json")),
            identity,
            "cv",
            seed,
        )?;
        assert_eq!(payload["seed"], seed);
        let folds = assignments(seed);
        assert_eq!(payload["file_assignments"], json!(folds));
        assert_eq!(
            payload["population_rows"],
            dataset["provenance"]["data"]["rows_after_warmup"]
        );
        assert_eq!(payload["arms"].as_array().unwrap().len(), 3);
        for (arm_index, arm) in payload["arms"].as_array().unwrap().iter().enumerate() {
            assert_eq!(arm["arm"], ARMS[arm_index]);
            assert_eq!(arm["folds"].as_array().unwrap().len(), 5);
            check_metrics(
                &arm["out_of_fold_baseline"],
                &arm["out_of_fold_augmented"],
                &arm["out_of_fold_delta"],
            );
            for (fold_index, evaluation) in arm["folds"].as_array().unwrap().iter().enumerate() {
                let counts: Vec<u32> = (0..813)
                    .map(|id| u32::from(!folds[fold_index].contains(&id)))
                    .collect();
                fit_count += check_evaluation(evaluation, &counts, files);
                if seed == 42 && fold_index == 0 {
                    check_standalone(&evaluation["standalone"], arm_index == 1);
                }
            }
        }
        cv_records.push(payload);
    }
    for index in 0..100 {
        let filename = format!("bootstrap-{index:03}.json");
        if !observed.contains(&filename) {
            continue;
        }
        let payload = record(&results.join(filename), identity, "bootstrap", index)?;
        assert_eq!(payload["seed"], 10000 + index);
        assert_eq!(payload["index"], index);
        assert_eq!(payload["held_out_fold"], 0);
        assert_eq!(payload["partition_seed"], 42);
        let mut rng = ChaCha8Rng::seed_from_u64(10000 + index);
        let training = sampled(&train_ids, &mut rng);
        let testing = sampled(test_ids, &mut rng);
        assert_eq!(payload["train_file_counts"], json!(training));
        assert_eq!(payload["test_file_counts"], json!(testing));
        assert!(
            training
                .iter()
                .zip(&testing)
                .all(|(left, right)| *left == 0 || *right == 0)
        );
        assert_eq!(training.iter().sum::<u32>() as usize, train_ids.len());
        assert_eq!(testing.iter().sum::<u32>() as usize, test_ids.len());
        assert_eq!(payload["arms"].as_array().unwrap().len(), 3);
        let mut increment = f64::INFINITY;
        let mut margin = f64::INFINITY;
        for (arm_index, arm) in payload["arms"].as_array().unwrap().iter().enumerate() {
            assert_eq!(arm["arm"], ARMS[arm_index]);
            fit_count += check_evaluation(arm, &training, files);
            increment = increment.min(number(&arm["delta"]["roc_auc"]));
            margin = margin.min(check_standalone(&arm["standalone"], arm_index == 1));
            assert_eq!(
                arm["standalone"]["associator_auc"],
                payload["arms"][0]["standalone"]["associator_auc"]
            );
        }
        assert_eq!(number(&payload["global_minimum_increment"]), increment);
        assert_eq!(number(&payload["global_minimum_calibration"]), margin);
        bootstrap_records.push(payload);
    }
    assert_eq!(fit_count, 150 + 6 * bootstrap_records.len());
    let mut arm_report = Vec::new();
    let mut intervals = Value::Null;
    for arm_index in 0..3 {
        let deltas: Vec<f64> = cv_records
            .iter()
            .map(|record| number(&record["arms"][arm_index]["out_of_fold_delta"]["roc_auc"]))
            .collect();
        arm_report.push(json!({"arm":ARMS[arm_index],"cv_roc_auc_deltas":deltas,"minimum":deltas.iter().copied().fold(f64::INFINITY,f64::min),"maximum":deltas.iter().copied().fold(f64::NEG_INFINITY,f64::max)}));
    }
    if complete {
        assert_eq!(fit_count, 750);
        let summary = read(&results.join("summary.json"))?;
        assert_eq!(summary["identity"], identity);
        assert_eq!(summary["completed_fits"], 750);
        assert_eq!(summary["completed_cv_seeds"], json!([42, 43, 44, 45, 46]));
        assert_eq!(summary["completed_bootstrap_replicates"], 100);
        assert_eq!(summary["arms"].as_array().unwrap().len(), 3);
        let mut global_increments: Vec<f64> = bootstrap_records
            .iter()
            .map(|record| number(&record["global_minimum_increment"]))
            .collect();
        let mut global_margins: Vec<f64> = bootstrap_records
            .iter()
            .map(|record| number(&record["global_minimum_calibration"]))
            .collect();
        intervals = json!({"global_increment":check_interval(&summary["global_increment"],&mut global_increments),"global_standalone":check_interval(&summary["global_standalone_margin"],&mut global_margins)});
        for arm_index in 0..3 {
            assert_eq!(summary["arms"][arm_index]["arm"], ARMS[arm_index]);
            assert_eq!(
                summary["arms"][arm_index]["primary_fixed_partition_point"],
                cv_records[0]["arms"][arm_index]["folds"][0]
            );
            let mut increment: Vec<f64> = bootstrap_records
                .iter()
                .map(|record| number(&record["arms"][arm_index]["delta"]["roc_auc"]))
                .collect();
            let mut margin: Vec<f64> = bootstrap_records
                .iter()
                .map(|record| number(&record["arms"][arm_index]["standalone"]["minimum_delta_auc"]))
                .collect();
            check_interval(
                &summary["arms"][arm_index]["conditional_increment_descriptive"],
                &mut increment,
            );
            check_interval(
                &summary["arms"][arm_index]["conditional_standalone_minimum_descriptive"],
                &mut margin,
            );
            assert_eq!(
                summary["arms"][arm_index]["cv_delta_roc_auc"],
                arm_report[arm_index]["cv_roc_auc_deltas"]
            );
            near(
                number(&summary["arms"][arm_index]["observed_split_minimum"]),
                number(&arm_report[arm_index]["minimum"]),
            );
            near(
                number(&summary["arms"][arm_index]["observed_split_maximum"]),
                number(&arm_report[arm_index]["maximum"]),
            );
        }
    }
    let report = json!({"schema_version":1,"complete":complete,"identity":identity,"cv_records":cv_records.len(),"bootstrap_records":bootstrap_records.len(),"converged_fits":fit_count,"checks":["source and protocol hashes","dataset and payload checksums","seeded partitions and paired training/test counts","training RMS and row multiplicities","fit stopping criterion and finite coefficients","paired metrics and standalone strata","global minima","exact planned set and summary linear percentiles in complete mode"],"cv":arm_report,"conditional_intervals":intervals});
    serde_json::to_writer_pretty(std::io::stdout().lock(), &report)?;
    Ok(())
}

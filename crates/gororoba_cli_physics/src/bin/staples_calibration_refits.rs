//! E-280 calibration-context comparison and E-281 grouped logistic refitting.
//! Every inference record binds the admitted corpus, sealed protocol and source.

use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use gororoba_cli_physics::staple_calibration::prepare;
use rayon::prelude::*;
use serde_json::{Value, json};
use std::{
    collections::BTreeSet,
    fs::{self, File, OpenOptions},
    io::Write,
    path::PathBuf,
};

#[path = "staples_calibration_refits/evidence.rs"]
mod evidence;
#[path = "staples_calibration_refits/fitting.rs"]
mod fitting;
#[path = "staples_calibration_refits/metrics.rs"]
mod metrics;
#[path = "staples_calibration_refits/splitting.rs"]
mod splitting;

use evidence::{
    OutputLock, admit_protocol, atomic_json, digest, execute_record, hash_file,
    retain_admission_result, retain_preparation_result, source_identity, source_root, summarize,
};
use fitting::{run_bootstrap, run_cv};
use metrics::{Rankings, linear_percentile};
use splitting::{folds, selected_rows, unit_counts};

const STRIDE: usize = 7;
const PVI_COLUMN: usize = 4;
const EPSILON: f64 = 1e-12;
const RIDGE: f64 = 1e-6;
const SEEDS: [u64; 5] = [42, 43, 44, 45, 46];
const ARMS: [&str; 3] = ["daily_file", "frozen_training", "prewindow_rolling"];
const REPLICATES: usize = 100;
const PROTOCOL_HASH: &str = "20bdf30fe878db6e92977df82088bea79104cf81fd3353f4c736effdc362b4d3";
const SOURCE_FILES: [(&str, &[u8]); 21] = [
    (
        "crates/gororoba_cli_physics/src/bin/staples_calibration_refits/splitting.rs",
        include_bytes!("staples_calibration_refits/splitting.rs"),
    ),
    (
        "crates/gororoba_cli_physics/src/bin/staples_calibration_refits/fitting.rs",
        include_bytes!("staples_calibration_refits/fitting.rs"),
    ),
    (
        "crates/gororoba_cli_physics/src/bin/staples_calibration_refits/metrics.rs",
        include_bytes!("staples_calibration_refits/metrics.rs"),
    ),
    (
        "crates/gororoba_cli_physics/src/bin/staples_calibration_refits/evidence.rs",
        include_bytes!("staples_calibration_refits/evidence.rs"),
    ),
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
    use super::{
        evidence::{read_record, retain_failure, validate_payload},
        fitting::{replace_calibration, standardize, training_rms},
        metrics::{Ranking, interval},
        splitting::draw_counts,
        *,
    };
    use gororoba_cli_physics::{
        staple_calibration::{FileEvidence, PreparedDataset},
        staple_logistic::weighted_auc,
    };
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha8Rng;
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

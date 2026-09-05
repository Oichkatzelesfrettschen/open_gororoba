//! Preregistered timestamp-causal epoch and exact-support control validation.

use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeSet, fs, path::PathBuf};

#[path = "staples_causal_validation/admission.rs"]
mod admission;
#[path = "staples_causal_validation/evidence.rs"]
mod evidence;
#[path = "staples_causal_validation/features.rs"]
mod features;
#[path = "staples_causal_validation/fitting.rs"]
mod fitting;
#[path = "staples_causal_validation/metrics.rs"]
mod metrics;
#[path = "staples_causal_validation/splits.rs"]
mod splits;

const PROTOCOL_SHA256: &str = "ff90167ae3d30e2e65fb99bbec4387b07f94fb7e4586c5e520c14dd70aa3aea6";

#[derive(Clone, Deserialize, Serialize)]
struct Config {
    training_years: Vec<i32>,
    validation_years: Vec<i32>,
    final_years: Vec<i32>,
    widths: [usize; 3],
    control_seeds: Vec<u64>,
    bootstrap_draws: usize,
    bootstrap_seed: u64,
    ridge: f64,
    log_epsilon: f64,
    minimum_increment: f64,
    label_radius_seconds: i64,
    maximum_gap_seconds: i64,
    maximum_feature_span_seconds: i64,
    fill_value: f64,
    threads: usize,
    file_map: String,
    file_map_sha256: String,
    catalog: String,
    catalog_sha256: String,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Mode {
    Prepare,
    All,
    External,
}

#[derive(Parser)]
#[command(about = "Sealed timestamp-causal epoch validation with exact-support controls")]
struct Args {
    #[arg(long)]
    input_root: PathBuf,
    #[arg(long)]
    out_dir: PathBuf,
    #[arg(long)]
    protocol: PathBuf,
    #[arg(long, value_enum, default_value = "all")]
    mode: Mode,
    #[arg(long)]
    external_manifest: Option<PathBuf>,
    #[arg(long)]
    file_map: Option<PathBuf>,
    #[arg(long)]
    models_dir: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    run(&args)
}

fn validate_external(manifest: &Value, config: &Config, crossings: &[i64]) -> Result<bool> {
    use chrono::Datelike;
    ensure!(
        manifest["probe"] == "d"
            && manifest["protocol_sha256"] == PROTOCOL_SHA256
            && manifest["catalog_sha256"] == config.catalog_sha256
            && manifest["complete_accounting"] == true,
        "external manifest identity/accounting mismatch"
    );
    let expected_dates: BTreeSet<String> = crossings
        .iter()
        .filter_map(|&nanos| {
            let time = chrono::DateTime::<chrono::Utc>::from_timestamp_nanos(nanos);
            config
                .final_years
                .contains(&time.year())
                .then(|| time.date_naive().to_string())
        })
        .collect();
    let planned: Vec<String> = serde_json::from_value(manifest["planned_dates"].clone())?;
    ensure!(
        planned.iter().cloned().collect::<BTreeSet<_>>() == expected_dates
            && planned.len() == expected_dates.len(),
        "external dates differ from all declared V2 crossing dates"
    );
    let results = manifest["results"]
        .as_array()
        .context("external results missing")?;
    ensure!(
        results.len() == planned.len(),
        "external date accounting incomplete"
    );
    for (date, result) in planned.iter().zip(results) {
        ensure!(
            result["date"] == *date
                && matches!(result["status"].as_str(), Some("admitted" | "failed")),
            "external result date/status mismatch"
        );
    }
    let field = &manifest["field_semantics"];
    ensure!(
        field["name"] == "thd_fgs_gse"
            && field["type"] == "double"
            && field["size"] == json!([3])
            && field["units"]
                .as_str()
                .is_some_and(|units| units.starts_with("nT GSE"))
            && field["fill"]
                .as_str()
                .and_then(|value| value.parse::<f64>().ok())
                == Some(config.fill_value),
        "external units/dimension/fill identity mismatch"
    );
    Ok(results.iter().all(|result| result["status"] == "admitted"))
}

fn model_path(width: usize, tensor: Option<usize>, config: &Config) -> String {
    format!("model-width-{width}-{}.json", fitting::name(tensor, config))
}

fn run(args: &Args) -> Result<()> {
    let _lock = evidence::OutputLock::acquire(&args.out_dir)?;
    let sources = evidence::source_identity()?;
    let context = json!({"sources":sources,"protocol_path":args.protocol,"expected_protocol_sha256":PROTOCOL_SHA256,"input_root":args.input_root,"executable_sha256":evidence::hash_file(&std::env::current_exe()?)?});
    let result = run_locked(args, &sources);
    result.map_err(|error| evidence::retain_failure(&args.out_dir, &context, error))
}

fn run_locked(args: &Args, sources: &Value) -> Result<()> {
    let protocol = fs::read(&args.protocol)?;
    ensure!(
        evidence::digest(&protocol) == PROTOCOL_SHA256,
        "protocol differs from preregistered hash"
    );
    let config: Config = toml::from_str(std::str::from_utf8(&protocol)?)?;
    splits::validate(&config)?;
    ensure!(
        config.widths == [64, 256, 1024]
            && config.control_seeds == (1000..1019).collect::<Vec<_>>()
            && config.bootstrap_draws == 2000,
        "protocol dimensions differ from compiled campaign"
    );
    rayon::ThreadPoolBuilder::new()
        .num_threads(config.threads)
        .build_global()?;
    let failures: Vec<_> = fs::read_dir(&args.out_dir)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("failure-"))
        .collect();
    ensure!(
        failures.is_empty(),
        "retained execution failure requires explicit review before retry"
    );
    let probe = if args.mode == Mode::External {
        "d"
    } else {
        "a"
    };
    let crossings = admission::catalog(
        &args.input_root.join(&config.catalog),
        probe,
        &config.catalog_sha256,
    )?;
    let external_manifest = if args.mode == Mode::External {
        let path = args
            .external_manifest
            .as_ref()
            .context("external mode requires --external-manifest")?;
        let bytes = fs::read(path)?;
        let manifest: Value = serde_json::from_slice(&bytes)?;
        if !validate_external(&manifest, &config, &crossings)? {
            let blocked = json!({"status":"blocked_external_intake","protocol_sha256":PROTOCOL_SHA256,"manifest_sha256":evidence::digest(&bytes),"sources":sources,"planned_dates":manifest["planned_dates"],"failed_results":manifest["results"].as_array().unwrap().iter().filter(|result|result["status"] != "admitted").collect::<Vec<_>>(),"executable_sha256":evidence::hash_file(&std::env::current_exe()?)?,"numerical_rows_consumed":0,"reason":"Every planned external date must pass admission; preserve failed dates without sorting or dropping."});
            evidence::preserve(&args.out_dir.join("blocked-external-intake.json"), &blocked)?;
            eprintln!("external inference blocked by retained intake failure");
            return Ok(());
        }
        Some((manifest, evidence::digest(&bytes)))
    } else {
        ensure!(
            args.external_manifest.is_none()
                && args.file_map.is_none()
                && args.models_dir.is_none(),
            "external overrides require external mode"
        );
        None
    };
    let map_path = if args.mode == Mode::External {
        args.file_map
            .as_ref()
            .context("external mode requires --file-map")?
            .clone()
    } else {
        args.input_root.join(&config.file_map)
    };
    let map_hash = evidence::hash_file(&map_path)?;
    if args.mode != Mode::External {
        ensure!(
            map_hash == config.file_map_sha256,
            "primary file map hash mismatch"
        );
    }
    let entries = admission::file_map(&map_path)?;
    if let Some((manifest, _)) = &external_manifest {
        for ((id, path), result) in entries.iter().zip(manifest["results"].as_array().unwrap()) {
            ensure!(
                result["raw_path"] == *path
                    && evidence::hash_file(&args.input_root.join(path))?
                        == result["sha256"]
                            .as_str()
                            .context("external digest missing")?,
                "external raw association/digest mismatch for file {id}"
            );
        }
        ensure!(
            entries.len() == manifest["planned_dates"].as_array().unwrap().len(),
            "external map drops planned dates"
        );
    }
    let ensemble = features::Ensemble::new(&config.control_seeds)?;
    let data = admission::prepare(
        &args.input_root,
        &args.out_dir,
        &entries,
        probe,
        &config,
        &crossings,
        &ensemble,
    )?;
    ensure!(
        data.files
            .iter()
            .all(|file| config.training_years.contains(&file.year)
                || config.validation_years.contains(&file.year)
                || config.final_years.contains(&file.year)),
        "raw file epoch outside sealed plan"
    );
    if let Some((manifest, _)) = &external_manifest {
        for file in &data.files {
            ensure!(
                manifest["planned_dates"][usize::from(file.id)] == file.date,
                "external raw date differs from planned date for file {}",
                file.id
            );
        }
    }
    let provenance = json!({"protocol_sha256":PROTOCOL_SHA256,"config":config,"sources":sources,"file_map_sha256":map_hash,"catalog_sha256":config.catalog_sha256,"probe":probe,"files":data.files,"tensors":ensemble.declarations,"external_manifest_sha256":external_manifest.as_ref().map(|row|&row.1),"executable_sha256":evidence::hash_file(&std::env::current_exe()?)?,"rows":data.rows.len(),"representation":"f32 natural-log features; f64 training statistics, coefficients, logits and metrics","support_encoding":"For each admitted decision: little-endian u16 file ID; u64 decision raw index, common first raw index, feature first raw index, feature last raw index and common sample count; i64 decision and latest-feature Unix nanoseconds; u8 label. Each binary record is 59 bytes, with no header. Admitted segments contain consecutive unique valid raw rows.","labels_encoding":"For each admitted decision: little-endian i64 decision Unix nanoseconds followed by u8 binary label.","decision_order":"Advance timestamp, close preceding batch, apply gap rule, score preceding history, then inspect current vector. Current invalid vectors affect only subsequent decisions; unknown timestamps reset immediately."});
    let identity = evidence::digest(&serde_json::to_vec(&provenance)?);
    evidence::preserve(
        &args.out_dir.join("dataset.json"),
        &json!({"identity":identity,"provenance":provenance}),
    )?;
    if args.mode == Mode::Prepare {
        return Ok(());
    }
    execute_campaign(args, &config, sources, &data, &identity)
}

fn execute_campaign(
    args: &Args,
    config: &Config,
    sources: &Value,
    data: &admission::Dataset,
    identity: &str,
) -> Result<()> {
    let mut expected = BTreeSet::from(["dataset.json".to_owned()]);
    let years: Vec<i32> = if args.mode == Mode::External {
        config.final_years.clone()
    } else {
        config
            .validation_years
            .iter()
            .chain(&config.final_years)
            .copied()
            .collect()
    };
    let mut points = Vec::<(usize, Option<usize>, i32, metrics::PointMetrics)>::new();
    let mut kernels = Vec::<(usize, i32, Option<usize>, metrics::AucKernel)>::new();
    let trained_identity = if args.mode == Mode::External {
        let directory = args
            .models_dir
            .as_ref()
            .context("external mode requires --models-dir")?;
        let dataset: Value = serde_json::from_slice(&fs::read(directory.join("dataset.json"))?)?;
        ensure!(
            dataset["identity"] == evidence::digest(&serde_json::to_vec(&dataset["provenance"])?),
            "training dataset identity checksum mismatch"
        );
        ensure!(
            dataset["provenance"]["protocol_sha256"] == PROTOCOL_SHA256
                && dataset["provenance"]["probe"] == "a"
                && dataset["provenance"]["config"] == serde_json::to_value(config)?,
            "external model source differs from primary training protocol"
        );
        ensure!(
            dataset["provenance"]["sources"] == *sources,
            "external feature implementation differs from training source"
        );
        Some(
            dataset["identity"]
                .as_str()
                .context("training identity missing")?
                .to_owned(),
        )
    } else {
        None
    };
    for (width_index, &width) in config.widths.iter().enumerate() {
        for tensor in std::iter::once(None).chain((0..20).map(Some)) {
            let model_name = model_path(width, tensor, config);
            let payload = if let Some(training_identity) = &trained_identity {
                evidence::read_record(
                    &args.models_dir.as_ref().unwrap().join(&model_name),
                    training_identity,
                )?
            } else {
                expected.insert(model_name.clone());
                evidence::record(&args.out_dir.join(&model_name), identity, sources, || {
                    eprintln!("fit width {width} {}", fitting::name(tensor, config));
                    Ok(serde_json::to_value(fitting::fit(
                        data,
                        config,
                        width_index,
                        tensor,
                    )?)?)
                })?
            };
            let model: fitting::Model = serde_json::from_value(payload)?;
            model.validate(config)?;
            ensure!(
                model.width == width && model.tensor == tensor,
                "model filename/parameter identity mismatch"
            );
            let points_name = format!(
                "points-width-{width}-{}.json",
                fitting::name(tensor, config)
            );
            expected.insert(points_name.clone());
            let point_payload = evidence::record(
                &args.out_dir.join(&points_name),
                identity,
                sources,
                || {
                    let mut records = Vec::new();
                    for &year in &years {
                        eprintln!(
                            "evaluate width {width} {} year {year}",
                            fitting::name(tensor, config)
                        );
                        let retain = tensor.is_none_or(|tensor| tensor == 0)
                            && config.final_years.contains(&year);
                        let (metrics, kernel) =
                            metrics::evaluate(data, &model, width_index, year, retain)?;
                        records.push(json!({"year":year,"metrics":metrics,"auc_kernel":kernel}));
                    }
                    Ok(
                        json!({"width":width,"tensor":tensor,"models_training_identity":trained_identity.as_deref().unwrap_or(identity),"years":records}),
                    )
                },
            )?;
            ensure!(
                point_payload["width"] == width
                    && point_payload["tensor"] == serde_json::to_value(tensor)?,
                "point record identity mismatch"
            );
            ensure!(
                point_payload["models_training_identity"]
                    == trained_identity.as_deref().unwrap_or(identity),
                "point record frozen training identity mismatch"
            );
            let records = point_payload["years"]
                .as_array()
                .context("point years missing")?;
            ensure!(records.len() == years.len(), "point epoch set incomplete");
            for (&year, record) in years.iter().zip(records) {
                ensure!(record["year"] == year, "point epoch order mismatch");
                let point: metrics::PointMetrics =
                    serde_json::from_value(record["metrics"].clone())?;
                ensure!(
                    [point.roc_auc, point.average_precision, point.log_loss]
                        .iter()
                        .all(|value| value.is_finite())
                        && point.positives > 0
                        && point.positives < point.rows,
                    "invalid retained point metric"
                );
                points.push((width, tensor, year, point));
                if tensor.is_none_or(|tensor| tensor == 0) && config.final_years.contains(&year) {
                    let kernel: metrics::AucKernel =
                        serde_json::from_value(record["auc_kernel"].clone())?;
                    kernel.validate()?;
                    kernels.push((width, year, tensor, kernel));
                } else {
                    ensure!(record["auc_kernel"].is_null(), "unplanned AUC kernel");
                }
            }
        }
    }
    let paired: Vec<_> = config
        .widths
        .iter()
        .flat_map(|&width| config.final_years.iter().map(move |&year| (width, year)))
        .map(|(width, year)| {
            let baseline = kernels
                .iter()
                .find(|row| row.0 == width && row.1 == year && row.2.is_none())
                .context("baseline kernel missing")?
                .3
                .clone();
            let canonical = kernels
                .iter()
                .find(|row| row.0 == width && row.1 == year && row.2 == Some(0))
                .context("canonical kernel missing")?
                .3
                .clone();
            Ok((width, year, baseline, canonical))
        })
        .collect::<Result<Vec<_>>>()?;
    expected.insert("bootstrap.json".to_owned());
    let bootstrap = evidence::record(
        &args.out_dir.join("bootstrap.json"),
        identity,
        sources,
        || metrics::bootstrap(&paired, config),
    )?;
    let mut panels = Vec::new();
    let mut all_canonical_above = true;
    for &width in &config.widths {
        for &year in &years {
            let baseline = &points
                .iter()
                .find(|row| row.0 == width && row.1.is_none() && row.2 == year)
                .context("missing baseline points")?
                .3;
            let canonical = &points
                .iter()
                .find(|row| row.0 == width && row.1 == Some(0) && row.2 == year)
                .context("missing canonical points")?
                .3;
            let controls: Vec<_> = (1..20).map(|index| {
            let point=&points.iter().find(|row|row.0==width&&row.1==Some(index)&&row.2==year).unwrap().3;
            json!({"seed":config.control_seeds[index-1],"metrics":point,"increment":point.roc_auc-baseline.roc_auc})
        }).collect();
            let above = controls
                .iter()
                .filter(|row| row["metrics"]["roc_auc"].as_f64().unwrap() >= canonical.roc_auc)
                .count();
            if config.final_years.contains(&year) {
                all_canonical_above &= above == 0;
            }
            panels.push(json!({"width":width,"year":year,"baseline":baseline,"canonical":canonical,"canonical_increment":canonical.roc_auc-baseline.roc_auc,"controls":controls,"controls_at_least_canonical":above,"finite_ensemble_tail_rank":(1+above) as f64/20.0}));
        }
    }
    evidence::exact_records(&args.out_dir, &expected)?;
    let summary = json!({"identity":identity,"status":"complete","completed_models":63,"new_training_fits":if args.mode==Mode::External {0}else{63},"point_records":63,"planned_record_names":expected,"bootstrap_decision":bootstrap["decision"],"primary_interval":bootstrap["interval"],"minimum_increment":config.minimum_increment,"panels":panels,"canonical_exceeds_every_control_on_every_final_panel":all_canonical_above,"mechanism_decision":"inconclusive_comparison_uncertainty","mechanism_boundary":"Finite ensemble point ranks describe exact-support comparisons; confidence intervals for canonical-minus-control comparisons were not run. Pointwise reproduction motivates revising an algebra-specific explanation independently of useful prediction.","external_models_training_identity":trained_identity,"claim_boundary":"Archived-data timestamp-causal localization on selected crossing days; historical online availability, precursor utility, general ordinary-day false alarms and independent-mission transfer remain unmeasured."});
    evidence::record(
        &args.out_dir.join("summary.json"),
        identity,
        sources,
        || Ok(summary),
    )?;
    Ok(())
}

#[cfg(test)]
fn test_config() -> Config {
    Config {
        training_years: vec![2007, 2008, 2009, 2010, 2011, 2012],
        validation_years: vec![2013, 2014],
        final_years: vec![2015, 2016],
        widths: [1, 2, 4],
        control_seeds: (1000..1019).collect(),
        bootstrap_draws: 2000,
        bootstrap_seed: 20260904,
        ridge: 1e-6,
        log_epsilon: 1e-12,
        minimum_increment: 0.005,
        label_radius_seconds: 120,
        maximum_gap_seconds: 30,
        maximum_feature_span_seconds: 30,
        fill_value: -1e30,
        threads: 6,
        file_map: String::new(),
        file_map_sha256: String::new(),
        catalog: String::new(),
        catalog_sha256: String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn external_gate_preserves_failed_date_and_rejects_incomplete_or_wrong_units() {
        let config = test_config();
        let crossing = chrono::DateTime::parse_from_rfc3339("2015-01-01T12:00:00Z")
            .unwrap()
            .timestamp_nanos_opt()
            .unwrap();
        let mut manifest = json!({"probe":"d","protocol_sha256":PROTOCOL_SHA256,"catalog_sha256":config.catalog_sha256,"complete_accounting":true,"planned_dates":["2015-01-01"],"results":[{"date":"2015-01-01","status":"failed"}],"field_semantics":{"name":"thd_fgs_gse","type":"double","size":[3],"units":"nT GSE (All Qs)","fill":"-1.0E30"}});
        assert!(!validate_external(&manifest, &config, &[crossing]).unwrap());
        manifest["results"][0]["status"] = json!("admitted");
        assert!(validate_external(&manifest, &config, &[crossing]).unwrap());
        manifest["field_semantics"]["units"] = json!("nT GSM");
        assert!(validate_external(&manifest, &config, &[crossing]).is_err());
        manifest["field_semantics"]["units"] = json!("nT GSE");
        manifest["results"] = json!([]);
        assert!(validate_external(&manifest, &config, &[crossing]).is_err());
    }
}

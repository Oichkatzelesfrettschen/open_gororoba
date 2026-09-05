//! Independent retained-record closure, support, and bootstrap replay audit.
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

fn digest(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
fn array(value: &Value) -> &[Value] {
    value.as_array().expect("array")
}
fn number(value: &Value) -> u64 {
    value.as_u64().expect("unsigned integer")
}
fn real(value: &Value) -> f64 {
    value.as_f64().expect("number")
}
fn near(left: f64, right: f64) -> Result<()> {
    ensure!(
        left.is_finite() && right.is_finite() && (left - right).abs() < 1e-12,
        "numeric mismatch {left} != {right}"
    );
    Ok(())
}
fn read(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
fn validate_record(record: &Value, identity: &Value) -> Result<()> {
    ensure!(&record["identity"] == identity, "record identity mismatch");
    ensure!(
        record["payload_sha256"] == digest(&serde_json::to_vec(&record["payload"])?),
        "record payload checksum mismatch"
    );
    Ok(())
}
fn payload(root: &Path, name: &str, identity: &Value) -> Result<Value> {
    let record = read(&root.join(name))?;
    validate_record(&record, identity).with_context(|| name.to_owned())?;
    Ok(record["payload"].clone())
}
fn same_set(observed: &BTreeSet<String>, expected: &BTreeSet<String>) -> Result<()> {
    ensure!(
        observed == expected,
        "record set mismatch: missing {:?}; extra {:?}",
        expected.difference(observed).collect::<Vec<_>>(),
        observed.difference(expected).collect::<Vec<_>>()
    );
    Ok(())
}
fn decision(interval: [f64; 2], threshold: f64) -> &'static str {
    if interval[0] > threshold {
        "supports_declared_useful_increment"
    } else if interval[1] < threshold {
        "rejects_declared_useful_increment"
    } else {
        "inconclusive"
    }
}
fn model_check(model: &Value, width: u64, tensor: Value, rows: u64, positives: u64) -> Result<()> {
    ensure!(
        model["width"] == width && model["tensor"] == tensor,
        "model width/tensor mismatch"
    );
    ensure!(model["converged"] == true, "unconverged model");
    ensure!(
        model["training_rows"] == rows && model["training_positives"] == positives,
        "training membership mismatch"
    );
    ensure!(
        number(&model["iterations"]) > 0
            && number(&model["iterations"]) == array(&model["deviance"]).len() as u64,
        "iteration accounting mismatch"
    );
    let dimension = if tensor.is_null() { 6 } else { 7 };
    ensure!(
        array(&model["coefficients"]).len() == dimension + 1
            && array(&model["means"]).len() == dimension
            && array(&model["scales"]).len() == dimension,
        "fit dimension mismatch"
    );
    for key in ["coefficients", "means", "scales", "deviance"] {
        ensure!(
            array(&model[key])
                .iter()
                .all(|value| real(value).is_finite()),
            "nonfinite fit"
        );
    }
    ensure!(
        array(&model["scales"])
            .iter()
            .all(|value| real(value) > 0.0),
        "invalid scale"
    );
    ensure!(model["ridge"] == 0.000001, "ridge changed");
    Ok(())
}
struct Kernel {
    ids: Vec<u64>,
    positives: Vec<f64>,
    negatives: Vec<f64>,
    wins: Vec<f64>,
}
impl Kernel {
    fn load(value: &Value) -> Result<Self> {
        let kernel = Self {
            ids: array(&value["file_ids"]).iter().map(number).collect(),
            positives: array(&value["positives"]).iter().map(real).collect(),
            negatives: array(&value["negatives"]).iter().map(real).collect(),
            wins: array(&value["wins"]).iter().map(real).collect(),
        };
        let count = kernel.ids.len();
        ensure!(
            count > 0
                && kernel.ids.windows(2).all(|pair| pair[0] < pair[1])
                && kernel.positives.len() == count
                && kernel.negatives.len() == count
                && kernel.wins.len() == count * count,
            "kernel shape"
        );
        for (index, &win) in kernel.wins.iter().enumerate() {
            ensure!(
                win.is_finite()
                    && win >= 0.0
                    && win <= kernel.positives[index / count] * kernel.negatives[index % count],
                "kernel win bound"
            );
        }
        Ok(kernel)
    }
    fn auc(&self, counts: &[f64]) -> f64 {
        let positives: f64 = self
            .positives
            .iter()
            .zip(counts)
            .map(|(value, weight)| value * weight)
            .sum();
        let negatives: f64 = self
            .negatives
            .iter()
            .zip(counts)
            .map(|(value, weight)| value * weight)
            .sum();
        let wins: f64 = self
            .wins
            .chunks(counts.len())
            .zip(counts)
            .map(|(row, weight)| {
                weight
                    * row
                        .iter()
                        .zip(counts)
                        .map(|(win, other)| win * other)
                        .sum::<f64>()
            })
            .sum();
        wins / (positives * negatives)
    }
}
fn quantile(sorted: &[f64], fraction: f64) -> f64 {
    let position = fraction * (sorted.len() - 1) as f64;
    let low = position.floor() as usize;
    let high = position.ceil() as usize;
    sorted[low] + (sorted[high] - sorted[low]) * (position - low as f64)
}
fn main() -> Result<()> {
    let argument = std::env::args()
        .nth(1)
        .context("usage: completion-audit RESULTS")?;
    let root = Path::new(&argument);
    let dataset = read(&root.join("dataset.json"))?;
    let identity = &dataset["identity"];
    let provenance = &dataset["provenance"];
    ensure!(
        *identity == digest(&serde_json::to_vec(provenance)?),
        "dataset identity checksum"
    );
    let config = &provenance["config"];
    ensure!(
        provenance["protocol_sha256"]
            == digest(&fs::read(
                root.parent()
                    .context("campaign directory")?
                    .join("protocol.toml")
            )?),
        "protocol digest mismatch"
    );
    for (path, expected_hash) in provenance["sources"].as_object().context("source hashes")? {
        ensure!(
            *expected_hash == digest(&fs::read(path)?),
            "retained source changed: {path}"
        );
    }
    for (field, expected) in [
        ("widths", json!([64, 256, 1024])),
        (
            "training_years",
            json!([2007, 2008, 2009, 2010, 2011, 2012]),
        ),
        ("validation_years", json!([2013, 2014])),
        ("final_years", json!([2015, 2016])),
        ("control_seeds", json!((1000..=1018).collect::<Vec<_>>())),
    ] {
        ensure!(config[field] == expected, "sealed config mismatch {field}");
    }
    ensure!(
        config["bootstrap_seed"] == 20260904
            && config["bootstrap_draws"] == 2000
            && config["minimum_increment"] == 0.005,
        "sealed bootstrap config mismatch"
    );
    let mut names = vec![
        ("baseline".to_owned(), Value::Null),
        ("canonical".to_owned(), json!(0)),
    ];
    names.extend((1000..=1018).map(|seed| (format!("scramble-{seed}"), json!(seed - 999))));
    let mut expected: BTreeSet<String> = ["dataset.json", "bootstrap.json", "summary.json"]
        .into_iter()
        .map(str::to_owned)
        .collect();
    for width in [64, 256, 1024] {
        for (name, _) in &names {
            for kind in ["model", "points"] {
                expected.insert(format!("{kind}-width-{width}-{name}.json"));
            }
        }
    }
    let observed: BTreeSet<String> = fs::read_dir(root)?
        .map(|entry| entry.map(|entry| entry.file_name().to_string_lossy().into_owned()))
        .collect::<std::io::Result<BTreeSet<_>>>()?
        .into_iter()
        .filter(|name| name != "supports")
        .collect();
    same_set(&observed, &expected)?;
    let summary = payload(root, "summary.json", identity)?;
    let mut planned = expected.clone();
    planned.remove("summary.json");
    same_set(
        &array(&summary["planned_record_names"])
            .iter()
            .map(|value| value.as_str().unwrap().to_owned())
            .collect(),
        &planned,
    )?;
    ensure!(
        summary["status"] == "complete"
            && summary["completed_models"] == 63
            && summary["point_records"] == 63
            && summary["new_training_fits"] == 63,
        "completion counters mismatch"
    );
    let tensors = array(&provenance["tensors"]);
    ensure!(tensors.len() == 20, "tensor count");
    for (index, tensor) in tensors.iter().enumerate() {
        ensure!(
            tensor["terms"] == 1848
                && tensor["dimension"] == 16
                && tensor["support_sha256"] == tensors[0]["support_sha256"],
            "exact tensor support mismatch"
        );
        ensure!(
            tensor["seed"]
                == if index == 0 {
                    Value::Null
                } else {
                    json!(999 + index)
                },
            "tensor seed mismatch"
        );
    }
    ensure!(
        tensors
            .iter()
            .map(|tensor| tensor["coefficients_sha256"].as_str().unwrap())
            .collect::<BTreeSet<_>>()
            .len()
            == 20,
        "duplicate coefficient tensor"
    );
    let files = array(&provenance["files"]);
    let mut annual = BTreeMap::<u64, (u64, u64, u64, u64)>::new();
    let mut support_names = BTreeSet::new();
    for (index, file) in files.iter().enumerate() {
        ensure!(file["id"] == index, "file ID order");
        let path = file["support_path"].as_str().unwrap();
        support_names.insert(path.to_owned());
        let bytes = fs::read(root.join(path))?;
        ensure!(
            file["support_sha256"] == digest(&bytes) && file["support_bytes"] == bytes.len(),
            "support digest/size mismatch {path}"
        );
        ensure!(
            bytes.len() % 59 == 0 && file["admitted_decisions"] == bytes.len() / 59,
            "support record count mismatch"
        );
        let mut labels = Sha256::new();
        let mut positives = 0_u64;
        let mut previous = None;
        for record in bytes.chunks_exact(59) {
            let u64_at =
                |offset| u64::from_le_bytes(record[offset..offset + 8].try_into().unwrap());
            let decision_time = i64::from_le_bytes(record[42..50].try_into().unwrap());
            let feature_time = i64::from_le_bytes(record[50..58].try_into().unwrap());
            ensure!(
                u16::from_le_bytes(record[0..2].try_into().unwrap()) as usize == index,
                "support file identity"
            );
            ensure!(
                u64_at(34) == 1031
                    && u64_at(18) == u64_at(10) + 1025
                    && u64_at(26) == u64_at(18) + 5
                    && u64_at(2) == u64_at(26) + 1,
                "support window boundary"
            );
            ensure!(
                feature_time < decision_time
                    && previous.is_none_or(|previous| previous < decision_time),
                "support causal/chronological boundary"
            );
            previous = Some(decision_time);
            ensure!(record[58] <= 1, "invalid label");
            positives += u64::from(record[58]);
            labels.update(&record[42..50]);
            labels.update(&record[58..59]);
        }
        ensure!(
            file["positive_decisions"] == positives
                && file["label_sha256"] == hex(&labels.finalize()),
            "label digest/count mismatch"
        );
        let entry = annual.entry(number(&file["year"])).or_default();
        entry.0 += number(&file["admitted_decisions"]);
        entry.1 += positives;
        entry.2 += u64::from(number(&file["admitted_decisions"]) > 0);
        entry.3 += u64::from(positives > 0);
    }
    let observed_support = fs::read_dir(root.join("supports"))?
        .map(|entry| entry.map(|entry| format!("supports/{}", entry.file_name().to_string_lossy())))
        .collect::<std::io::Result<BTreeSet<_>>>()?;
    same_set(&observed_support, &support_names)?;
    ensure!(
        provenance["rows"] == annual.values().map(|row| row.0).sum::<u64>(),
        "dataset row sum"
    );
    let training_rows = annual
        .iter()
        .filter(|(year, _)| **year <= 2012)
        .map(|(_, row)| row.0)
        .sum();
    let training_positives = annual
        .iter()
        .filter(|(year, _)| **year <= 2012)
        .map(|(_, row)| row.1)
        .sum();
    let mut kernels = BTreeMap::<(u64, u64, bool), Kernel>::new();
    let mut models = BTreeMap::new();
    let mut point_metrics = BTreeMap::new();
    let mut iterations = Vec::new();
    for width in [64, 256, 1024] {
        for (name, tensor) in &names {
            let model = payload(root, &format!("model-width-{width}-{name}.json"), identity)?;
            model_check(
                &model,
                width,
                tensor.clone(),
                training_rows,
                training_positives,
            )?;
            iterations.push(number(&model["iterations"]));
            models.insert((width, name.clone()), model);
            let points = payload(root, &format!("points-width-{width}-{name}.json"), identity)?;
            ensure!(
                points["width"] == width
                    && points["tensor"] == *tensor
                    && points["models_training_identity"] == *identity,
                "point identity"
            );
            ensure!(
                array(&points["years"])
                    .iter()
                    .map(|row| number(&row["year"]))
                    .collect::<Vec<_>>()
                    == vec![2013, 2014, 2015, 2016],
                "point epochs"
            );
            for row in array(&points["years"]) {
                let year = number(&row["year"]);
                let counts = annual[&year];
                let metrics = &row["metrics"];
                point_metrics.insert((width, year, name.clone()), metrics.clone());
                ensure!(
                    metrics["rows"] == counts.0
                        && metrics["positives"] == counts.1
                        && metrics["files"] == counts.2
                        && metrics["positive_files"] == counts.3,
                    "point support accounting"
                );
                if year >= 2015 && (name == "baseline" || name == "canonical") {
                    let kernel = Kernel::load(&row["auc_kernel"])?;
                    let selected: Vec<_> = files
                        .iter()
                        .filter(|file| {
                            file["year"] == year && number(&file["admitted_decisions"]) > 0
                        })
                        .collect();
                    ensure!(
                        kernel.ids
                            == selected
                                .iter()
                                .map(|file| number(&file["id"]))
                                .collect::<Vec<_>>(),
                        "kernel file parity"
                    );
                    for (offset, file) in selected.iter().enumerate() {
                        near(kernel.positives[offset], real(&file["positive_decisions"]))?;
                        near(
                            kernel.negatives[offset],
                            real(&file["admitted_decisions"]) - real(&file["positive_decisions"]),
                        )?;
                    }
                    near(
                        kernel.auc(&vec![1.0; kernel.ids.len()]),
                        real(&metrics["roc_auc"]),
                    )?;
                    kernels.insert((width, year, name == "canonical"), kernel);
                } else {
                    ensure!(row["auc_kernel"].is_null(), "unexpected retained kernel");
                }
            }
        }
    }
    for width in [64, 256, 1024] {
        let baseline = &models[&(width, "baseline".into())];
        for (name, _) in &names[1..] {
            let model = &models[&(width, name.clone())];
            for field in ["means", "scales"] {
                ensure!(
                    array(&model[field])[..6] == *array(&baseline[field]),
                    "training preprocessing parity"
                );
            }
        }
    }
    let mut summary_panels = BTreeSet::new();
    for panel in array(&summary["panels"]) {
        let width = number(&panel["width"]);
        let year = number(&panel["year"]);
        ensure!(
            summary_panels.insert((width, year)),
            "duplicate summary panel"
        );
        let baseline = &point_metrics[&(width, year, "baseline".into())];
        let canonical = &point_metrics[&(width, year, "canonical".into())];
        ensure!(
            panel["baseline"] == *baseline && panel["canonical"] == *canonical,
            "summary metrics mismatch"
        );
        near(
            real(&panel["canonical_increment"]),
            real(&canonical["roc_auc"]) - real(&baseline["roc_auc"]),
        )?;
        ensure!(
            array(&panel["controls"]).len() == 19,
            "summary control count"
        );
        let mut at_least = 0;
        for (index, control) in array(&panel["controls"]).iter().enumerate() {
            let seed = 1000 + index;
            ensure!(control["seed"] == seed, "summary control seed order");
            let metrics = &point_metrics[&(width, year, format!("scramble-{seed}"))];
            ensure!(control["metrics"] == *metrics, "control metrics mismatch");
            near(
                real(&control["increment"]),
                real(&metrics["roc_auc"]) - real(&baseline["roc_auc"]),
            )?;
            at_least += usize::from(real(&metrics["roc_auc"]) >= real(&canonical["roc_auc"]));
        }
        ensure!(
            panel["controls_at_least_canonical"] == at_least,
            "control rank mismatch"
        );
        near(
            real(&panel["finite_ensemble_tail_rank"]),
            (at_least + 1) as f64 / 20.0,
        )?;
    }
    ensure!(
        summary_panels
            == [64, 256, 1024]
                .into_iter()
                .flat_map(|width| [2013, 2014, 2015, 2016].map(|year| (width, year)))
                .collect(),
        "summary panel closure"
    );
    let bootstrap = payload(root, "bootstrap.json", identity)?;
    ensure!(
        bootstrap["planned_draws"] == 2000
            && bootstrap["completed_draws"] == 2000
            && array(&bootstrap["records"]).len() == 2000,
        "bootstrap count"
    );
    let mut minima = Vec::new();
    for (index, record) in array(&bootstrap["records"]).iter().enumerate() {
        ensure!(record["index"] == index, "bootstrap draw order");
        let mut weights = BTreeMap::new();
        for row in array(&record["year_file_multiplicities"]) {
            let year = number(&row[0]);
            let counts: Vec<f64> = array(&row[1]).iter().map(real).collect();
            let file_count = kernels[&(64, year, false)].ids.len();
            ensure!(
                counts.len() == file_count
                    && counts
                        .iter()
                        .all(|value| *value >= 0.0 && value.fract() == 0.0)
                    && counts.iter().sum::<f64>() == file_count as f64,
                "bootstrap daily multiplicities"
            );
            ensure!(
                weights.insert(year, counts).is_none(),
                "duplicate bootstrap epoch"
            );
        }
        ensure!(
            weights.keys().copied().collect::<Vec<_>>() == vec![2015, 2016],
            "bootstrap epochs"
        );
        let mut observed_pairs = BTreeSet::new();
        let mut minimum = f64::INFINITY;
        for row in array(&record["increments"]) {
            let width = number(&row["width"]);
            let year = number(&row["year"]);
            ensure!(
                observed_pairs.insert((width, year)),
                "duplicate bootstrap panel"
            );
            let increment = kernels[&(width, year, true)].auc(&weights[&year])
                - kernels[&(width, year, false)].auc(&weights[&year]);
            near(increment, real(&row["increment"]))?;
            minimum = minimum.min(increment);
        }
        ensure!(
            observed_pairs
                == [64, 256, 1024]
                    .into_iter()
                    .flat_map(|width| [2015, 2016].map(|year| (width, year)))
                    .collect(),
            "bootstrap panel closure"
        );
        near(minimum, real(&record["global_minimum"]))?;
        minima.push(minimum);
    }
    minima.sort_by(f64::total_cmp);
    let interval = [quantile(&minima, 0.025), quantile(&minima, 0.975)];
    for (index, value) in interval.iter().enumerate() {
        near(*value, real(&bootstrap["interval"][index]))?;
        near(*value, real(&summary["primary_interval"][index]))?;
    }
    let verdict = decision(interval, 0.005);
    ensure!(
        bootstrap["decision"] == verdict && summary["bootstrap_decision"] == verdict,
        "threshold verdict"
    );
    ensure!(
        summary["mechanism_decision"] == "inconclusive_comparison_uncertainty",
        "mechanism inference overreach"
    );
    let mut mutation_results = Vec::new();
    let mut missing = expected.clone();
    missing.remove("model-width-64-canonical.json");
    ensure!(
        same_set(&missing, &expected).is_err(),
        "missing record probe"
    );
    mutation_results.push("missing model rejected");
    let mut extra = expected.clone();
    extra.insert("unplanned.json".into());
    ensure!(same_set(&extra, &expected).is_err(), "extra record probe");
    mutation_results.push("extra record rejected");
    let original = &models[&(64, "baseline".into())];
    for field in ["training_rows", "converged", "width"] {
        let mut changed = original.clone();
        changed[field] = if field == "converged" {
            json!(false)
        } else {
            json!(0)
        };
        ensure!(
            model_check(&changed, 64, Value::Null, training_rows, training_positives).is_err(),
            "model mutation probe"
        );
        mutation_results.push(field);
    }
    let mut corrupt = read(&root.join("model-width-64-baseline.json"))?;
    corrupt["payload"]["training_rows"] = json!(0);
    ensure!(
        validate_record(&corrupt, identity).is_err(),
        "payload corruption probe"
    );
    mutation_results.push("payload corruption rejected");
    ensure!(
        decision([0.005, 0.007], 0.005) == "inconclusive"
            && decision([0.001, 0.005], 0.005) == "inconclusive",
        "threshold touching probe"
    );
    mutation_results.push("threshold touching remains inconclusive");
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"status":"passed","dataset_identity":identity,"exact_root_records":expected.len(),"model_count":63,"point_count":63,"point_epoch_panels":252,"support_files":files.len(),"support_rows":provenance["rows"],"training_rows":training_rows,"training_positives":training_positives,"annual_support":annual,"iterations_min":iterations.iter().min(),"iterations_max":iterations.iter().max(),"all_models_converged":true,"tensor_support_sha256":tensors[0]["support_sha256"],"bootstrap_draws_replayed":2000,"bootstrap_panel_increments_replayed":12000,"primary_interval":interval,"verdict":verdict,"mechanism_decision":summary["mechanism_decision"],"control_point_ranks":array(&summary["panels"]).iter().map(|panel|json!({"width":panel["width"],"year":panel["year"],"controls_at_least_canonical":panel["controls_at_least_canonical"]})).collect::<Vec<_>>(),"mutation_probes":mutation_results,"boundary":"Replays retained support, fit membership counts, model records and paired bootstrap arithmetic. Does not refit raw features or establish external transfer; seeded RNG draws are retained inputs, not independently regenerated."})
        )?
    );
    Ok(())
}

//! Separately sealed geometric augmentation and joint paired uncertainty.
use super::{
    Args, Config, Mode, PROTOCOL_SHA256, admission, control_uncertainty, evidence, features,
    fitting, metrics, model_path, splits,
};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::{collections::BTreeSet, fs, path::Path};

const GRAM_RANK_RELATIVE_TOLERANCE: f64 = 1e-10;

fn parent(directory: &Path, config: &Config) -> Result<(String, Value)> {
    let dataset: Value = serde_json::from_slice(&fs::read(directory.join("dataset.json"))?)?;
    let identity = dataset["identity"]
        .as_str()
        .context("missing parent identity")?
        .to_owned();
    ensure!(
        identity == evidence::digest(&serde_json::to_vec(&dataset["provenance"])?),
        "parent checksum mismatch"
    );
    ensure!(
        dataset["provenance"]["config"] == serde_json::to_value(config)?
            && dataset["provenance"]["probe"] == "a"
            && dataset["provenance"]["protocol_sha256"] == PROTOCOL_SHA256,
        "parent configuration mismatch"
    );
    Ok((identity, dataset))
}

fn prepare(
    args: &Args,
    config: &Config,
    dataset: &Value,
    years: &[i32],
) -> Result<admission::Dataset> {
    let files = dataset["provenance"]["files"]
        .as_array()
        .context("parent files absent")?;
    let selected: Vec<_> = files
        .iter()
        .filter(|file| {
            file["year"]
                .as_i64()
                .is_some_and(|year| years.contains(&(year as i32)))
        })
        .collect();
    let ids: BTreeSet<_> = selected
        .iter()
        .map(|file| file["id"].as_u64().context("file id missing"))
        .collect::<Result<_>>()?;
    ensure!(
        !ids.is_empty() && ids.len() == selected.len(),
        "empty or duplicate epoch inputs"
    );
    let map = args.input_root.join(&config.file_map);
    ensure!(
        evidence::hash_file(&map)? == config.file_map_sha256,
        "map checksum mismatch"
    );
    let entries: Vec<_> = admission::file_map(&map)?
        .into_iter()
        .filter(|(id, _)| ids.contains(&u64::from(*id)))
        .collect();
    ensure!(entries.len() == ids.len(), "map omits selected files");
    for (id, path) in &entries {
        let original = selected
            .iter()
            .find(|file| file["id"] == *id)
            .context("file association absent")?;
        ensure!(
            original["path"] == *path
                && original["sha256"] == evidence::hash_file(&args.input_root.join(path))?,
            "raw identity mismatch for {id}"
        );
    }
    let catalog = admission::catalog(
        &args.input_root.join(&config.catalog),
        "a",
        &config.catalog_sha256,
    )?;
    let ensemble = features::Ensemble::new(&config.control_seeds)?;
    let data = admission::prepare(
        &args.input_root,
        &args.out_dir,
        &entries,
        "a",
        config,
        &catalog,
        &ensemble,
    )?;
    ensure!(
        data.files.len() == selected.len() && data.rows.iter().all(|row| years.contains(&row.year)),
        "epoch admission differs"
    );
    for file in &data.files {
        let original = selected
            .iter()
            .find(|entry| entry["id"] == file.id)
            .context("parent association absent")?;
        ensure!(
            serde_json::to_value(file)? == **original,
            "raw/support/label admission differs for {}",
            file.id
        );
    }
    Ok(data)
}

fn spectrum(mut gram: [[f64; 8]; 8]) -> Result<[f64; 8]> {
    for _ in 0..512 {
        let mut pivot = (0, 1);
        for first in 0..8 {
            for second in first + 1..8 {
                if gram[first][second].abs() > gram[pivot.0][pivot.1].abs() {
                    pivot = (first, second);
                }
            }
        }
        let scale = (0..8)
            .map(|index| gram[index][index].abs())
            .fold(0.0, f64::max);
        if gram[pivot.0][pivot.1].abs() <= 1e-13 * scale.max(1.0) {
            return Ok(std::array::from_fn(|index| gram[index][index]));
        }
        let (first, second) = pivot;
        let angle =
            0.5 * (2.0 * gram[first][second]).atan2(gram[second][second] - gram[first][first]);
        let (sine, cosine) = angle.sin_cos();
        let diagonal_first = gram[first][first];
        let diagonal_second = gram[second][second];
        let off = gram[first][second];
        for (other, row) in gram.into_iter().enumerate() {
            if other != first && other != second {
                let left = row[first];
                let right = row[second];
                gram[other][first] = cosine * left - sine * right;
                gram[first][other] = gram[other][first];
                gram[other][second] = sine * left + cosine * right;
                gram[second][other] = gram[other][second];
            }
        }
        gram[first][first] = cosine * cosine * diagonal_first - 2.0 * sine * cosine * off
            + sine * sine * diagonal_second;
        gram[second][second] = sine * sine * diagonal_first
            + 2.0 * sine * cosine * off
            + cosine * cosine * diagonal_second;
        gram[first][second] = 0.0;
        gram[second][first] = 0.0;
    }
    anyhow::bail!("design Gram eigensolver exhausted fixed budget")
}

fn design(data: &admission::Dataset, config: &Config, width_index: usize) -> Result<Value> {
    let rows = splits::training_rows(data, config);
    ensure!(
        rows.len() == data.rows.len(),
        "design contains held-out rows"
    );
    let features: Vec<_> = data
        .rows
        .iter()
        .flat_map(|row| fitting::feature_values(row, width_index, None, true))
        .collect();
    let (means, scales) = fitting::standardize(&features, &rows, 7)?;
    let mut gram = [[0.0; 8]; 8];
    for &row in &rows {
        let vector: [f64; 8] = std::array::from_fn(|column| {
            if column == 0 {
                1.0
            } else {
                (f64::from(features[row as usize * 7 + column - 1]) - means[column - 1])
                    / scales[column - 1]
            }
        });
        for first in 0..8 {
            for second in 0..8 {
                gram[first][second] += vector[first] * vector[second] / rows.len() as f64;
            }
        }
    }
    let eigenvalues = spectrum(gram)?;
    ensure!(
        eigenvalues.iter().all(|value| value.is_finite()),
        "nonfinite design spectrum"
    );
    let maximum = eigenvalues.iter().copied().fold(0.0, f64::max);
    let minimum = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    ensure!(
        maximum > 0.0 && minimum >= -GRAM_RANK_RELATIVE_TOLERANCE * maximum,
        "invalid design Gram spectrum"
    );
    let rank = eigenvalues
        .iter()
        .filter(|&&value| value > maximum * GRAM_RANK_RELATIVE_TOLERANCE)
        .count();
    Ok(
        json!({"width":config.widths[width_index],"rows":rows.len(),"columns_with_intercept":8,"rank":rank,"condition_2":if rank==8 {Some((maximum/minimum).sqrt())}else{None},"gram_eigenvalues":eigenvalues,"gram":gram,"gram_relative_rank_tolerance":GRAM_RANK_RELATIVE_TOLERANCE,"means":means,"scales":scales,"method":"symmetric Jacobi spectrum of standardized training Gram; singular-value relative tolerance 1e-5; Gram formation squares condition number"}),
    )
}

pub(super) fn run(args: &Args, config: &Config, sources: &Value) -> Result<()> {
    ensure!(
        args.external_manifest.is_none()
            && args.file_map.is_none()
            && args.equivalence_margin.is_none(),
        "geometric continuation uses sealed primary inputs and undeclared equivalence margin"
    );
    let parent_dir = args.models_dir.as_ref().context("requires --models-dir")?;
    ensure!(
        fs::canonicalize(parent_dir)? != fs::canonicalize(&args.out_dir)?,
        "output aliases parent"
    );
    let (parent_identity, dataset) = parent(parent_dir, config)?;
    let protocol_path = args
        .geometric_protocol
        .as_ref()
        .context("requires --geometric-protocol")?;
    let protocol_hash = args
        .geometric_protocol_sha256
        .as_ref()
        .context("requires admitted --geometric-protocol-sha256")?;
    let protocol_bytes = fs::read(protocol_path)?;
    ensure!(
        evidence::digest(&protocol_bytes) == *protocol_hash,
        "geometric protocol checksum mismatch"
    );
    let protocol: Value = toml::from_str(std::str::from_utf8(&protocol_bytes)?)?;
    ensure!(
        protocol["parent_identity"] == parent_identity
            && protocol["admitted_sources"] == *sources
            && protocol["parent_sources_sha256"]
                == evidence::digest(&serde_json::to_vec(&dataset["provenance"]["sources"])?),
        "source amendment or parent identity mismatch"
    );
    ensure!(
        protocol["family_size"] == 120
            && protocol["gram_rank_relative_tolerance"] == GRAM_RANK_RELATIVE_TOLERANCE
            && config.training_years == (2007..=2012).collect::<Vec<_>>()
            && config.final_years == [2015, 2016],
        "protocol gates differ from compiled continuation"
    );
    let identity = evidence::digest(&serde_json::to_vec(
        &json!({"parent":parent_identity,"protocol":protocol_hash,"sources":sources}),
    )?);
    if args.mode == Mode::GeometricTrain {
        ensure!(
            args.geometric_models_dir.is_none(),
            "training cannot consume geometric models"
        );
        let data = prepare(args, config, &dataset, &config.training_years)?;
        let mut diagnostics = Vec::new();
        for width_index in 0..3 {
            diagnostics.push(design(&data, config, width_index)?);
        }
        evidence::record(
            &args.out_dir.join("training-design.json"),
            &identity,
            sources,
            || Ok(json!({"designs":diagnostics,"files":data.files,"final_rows_consumed":0})),
        )?;
        ensure!(
            diagnostics.iter().all(|row| row["rank"] == 8),
            "training design rank gate failed; retained diagnostics precede fitting"
        );
        let mut model_receipts = Vec::new();
        for (index, width) in config.widths.iter().enumerate() {
            let name = format!("model-geometric-width-{width}.json");
            let payload = evidence::record(&args.out_dir.join(&name), &identity, sources, || {
                Ok(serde_json::to_value(fitting::fit_geometric(
                    &data, config, index,
                )?)?)
            })?;
            let model: fitting::Model = serde_json::from_value(payload)?;
            model.validate(config)?;
            ensure!(
                model.geometric_capacity && model.tensor.is_none() && model.width == *width,
                "geometric identity mismatch"
            );
            ensure!(
                serde_json::to_value(&model.means)? == diagnostics[index]["means"]
                    && serde_json::to_value(&model.scales)? == diagnostics[index]["scales"],
                "fit transforms differ from design gate"
            );
            model_receipts.push(
                json!({"path":name,"sha256":evidence::hash_file(&args.out_dir.join(&name))?}),
            );
        }
        evidence::preserve(
            &args.out_dir.join("training-freeze.json"),
            &json!({"identity":identity,"parent_identity":parent_identity,"protocol_sha256":protocol_hash,"models":model_receipts,"design_sha256":evidence::hash_file(&args.out_dir.join("training-design.json"))?,"final_rows_consumed":0,"status":"training_frozen"}),
        )?;
        return Ok(());
    }
    let training = args
        .geometric_models_dir
        .as_ref()
        .context("evaluate requires --geometric-models-dir")?;
    let recovery = args
        .recovery_dir
        .as_ref()
        .context("evaluate requires --recovery-dir")?;
    ensure!(
        fs::canonicalize(training)? != fs::canonicalize(&args.out_dir)?
            && fs::canonicalize(recovery)? != fs::canonicalize(&args.out_dir)?,
        "evaluation aliases retained input"
    );
    let freeze: Value = serde_json::from_slice(&fs::read(training.join("training-freeze.json"))?)?;
    ensure!(
        freeze["identity"] == identity
            && freeze["parent_identity"] == parent_identity
            && freeze["protocol_sha256"] == *protocol_hash
            && freeze["final_rows_consumed"] == 0
            && freeze["status"] == "training_frozen",
        "training freeze mismatch"
    );
    ensure!(
        freeze["design_sha256"] == evidence::hash_file(&training.join("training-design.json"))?,
        "design receipt mismatch"
    );
    let diagnostics = evidence::read_record(&training.join("training-design.json"), &identity)?;
    ensure!(
        diagnostics["final_rows_consumed"] == 0
            && diagnostics["designs"]
                .as_array()
                .context("design rows missing")?
                .len()
                == 3
            && diagnostics["designs"]
                .as_array()
                .unwrap()
                .iter()
                .all(|row| row["rank"] == 8),
        "training rank admission failed"
    );
    let receipts = freeze["models"]
        .as_array()
        .context("frozen model receipts absent")?;
    ensure!(receipts.len() == 3, "frozen model set incomplete");
    let mut models = Vec::new();
    for (index, width) in config.widths.iter().enumerate() {
        let name = format!("model-geometric-width-{width}.json");
        ensure!(
            receipts[index]["path"] == name
                && receipts[index]["sha256"] == evidence::hash_file(&training.join(&name))?,
            "frozen model receipt mismatch"
        );
        let model: fitting::Model =
            serde_json::from_value(evidence::read_record(&training.join(&name), &identity)?)?;
        model.validate(config)?;
        ensure!(
            model.geometric_capacity && model.tensor.is_none() && model.width == *width,
            "model selector mismatch"
        );
        models.push(model);
    }
    // Final raw files are opened only after all training gates and model receipts pass.
    let data = prepare(args, config, &dataset, &config.final_years)?;
    let recovered: Value =
        serde_json::from_slice(&fs::read(recovery.join("recovery-identity.json"))?)?;
    let recovery_identity = recovered["identity"]
        .as_str()
        .context("recovery identity absent")?;
    ensure!(
        recovery_identity == evidence::digest(&serde_json::to_vec(&recovered["analysis"])?)
            && recovered["analysis"]["parent_identity"] == parent_identity
            && protocol["recovery_identity"] == recovery_identity,
        "recovery checksum/parent mismatch"
    );
    ensure!(
        recovered["analysis"]["files"] == serde_json::to_value(&data.files)?,
        "recovered final admission differs"
    );
    let mut kernels = Vec::new();
    let mut geometric = Vec::new();
    for (index, &width) in config.widths.iter().enumerate() {
        let canonical: fitting::Model = serde_json::from_value(evidence::read_record(
            &parent_dir.join(model_path(width, Some(0), config)),
            &parent_identity,
        )?)?;
        canonical.validate(config)?;
        ensure!(
            !canonical.geometric_capacity
                && canonical.tensor == Some(0)
                && canonical.width == width,
            "parent canonical identity mismatch"
        );
        for &year in &config.final_years {
            let original = evidence::read_record(
                &parent_dir.join(format!("points-width-{width}-canonical.json")),
                &parent_identity,
            )?;
            let reference = original["years"]
                .as_array()
                .context("parent years absent")?
                .iter()
                .find(|row| row["year"] == year)
                .context("parent year absent")?;
            let (point, replayed) = metrics::evaluate(&data, &canonical, index, year, true)?;
            let expected: metrics::PointMetrics =
                serde_json::from_value(reference["metrics"].clone())?;
            ensure!(
                (point.roc_auc - expected.roc_auc).abs() < 1e-12
                    && point.rows == expected.rows
                    && point.positives == expected.positives
                    && point.files == expected.files
                    && point.positive_files == expected.positive_files,
                "canonical point reproduction failed"
            );
            let replayed = replayed.context("canonical kernel missing")?;
            for tensor in 0..20 {
                let name = format!("kernel-width-{width}-tensor-{tensor}-year-{year}.json");
                let record = evidence::read_record(&recovery.join(&name), recovery_identity)?;
                ensure!(
                    record["parent_points_sha256"]
                        == evidence::hash_file(&parent_dir.join(format!(
                            "points-width-{width}-{}.json",
                            fitting::name(Some(tensor), config)
                        )))?,
                    "recovery point receipt mismatch"
                );
                ensure!(
                    record["model_sha256"]
                        == evidence::hash_file(&parent_dir.join(model_path(
                            width,
                            Some(tensor),
                            config
                        )))?,
                    "recovery model receipt mismatch"
                );
                let kernel: metrics::AucKernel = serde_json::from_value(record["kernel"].clone())?;
                kernel.validate()?;
                if tensor == 0 {
                    ensure!(
                        serde_json::to_value(&kernel)? == serde_json::to_value(&replayed)?,
                        "canonical kernel differs from recovery"
                    );
                }
                kernels.push((width, year, tensor, kernel));
            }
            let record = evidence::record(
                &args
                    .out_dir
                    .join(format!("kernel-geometric-width-{width}-year-{year}.json")),
                &identity,
                sources,
                || {
                    let (point, kernel) =
                        metrics::evaluate(&data, &models[index], index, year, true)?;
                    Ok(
                        json!({"metrics":point,"kernel":kernel,"training_freeze_sha256":evidence::hash_file(&training.join("training-freeze.json"))?,"canonical_metrics":point_to_value(&expected)?}),
                    )
                },
            )?;
            ensure!(
                record["training_freeze_sha256"]
                    == evidence::hash_file(&training.join("training-freeze.json"))?,
                "retained geometric kernel model receipt changed"
            );
            geometric.push((
                width,
                year,
                serde_json::from_value(record["kernel"].clone())?,
            ));
        }
    }
    let report = evidence::record(
        &args.out_dir.join("joint-control-uncertainty.json"),
        &identity,
        sources,
        || control_uncertainty::bootstrap_extended(&kernels, &geometric, config, None),
    )?;
    ensure!(report["family_size"] == 120, "joint family incomplete");
    evidence::preserve(
        &args.out_dir.join("geometric-evaluation-identity.json"),
        &json!({"identity":identity,"protocol_sha256":protocol_hash,"parent_identity":parent_identity,"recovery_identity":recovery_identity,"training_freeze_sha256":evidence::hash_file(&training.join("training-freeze.json"))?,"analysis_status":"post_hoc","family_size":120,"minimum_useful_increment":0.005,"equivalence_margin":null}),
    )?;
    Ok(())
}

fn point_to_value(point: &metrics::PointMetrics) -> Result<Value> {
    Ok(serde_json::to_value(point)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn admitted_protocol_seals_actual_compiled_source_set() -> Result<()> {
        let current = std::env::current_dir()?;
        let root=current.ancestors().find(|path|path.join("data/output/audit/claim-family-evidence-adjudication/geometric-capacity-protocol.toml").is_file()).context("protocol checkout missing")?;
        let protocol: Value = toml::from_str(&fs::read_to_string(root.join(
            "data/output/audit/claim-family-evidence-adjudication/geometric-capacity-protocol.toml",
        ))?)?;
        ensure!(
            protocol["admitted_sources"] == evidence::source_identity()?,
            "protocol source amendment is stale"
        );
        Ok(())
    }
    #[test]
    fn gram_rank_detects_duplicate_predictor() {
        let mut gram =
            std::array::from_fn(|row| std::array::from_fn(|column| f64::from(row == column)));
        gram[0][1] = 1.0;
        gram[1][0] = 1.0;
        let values = spectrum(gram).unwrap();
        assert_eq!(
            values
                .iter()
                .filter(|&&value| value > 2.0 * GRAM_RANK_RELATIVE_TOLERANCE)
                .count(),
            7
        );
        assert!((values.iter().sum::<f64>() - 8.0).abs() < 1e-12);
    }
}

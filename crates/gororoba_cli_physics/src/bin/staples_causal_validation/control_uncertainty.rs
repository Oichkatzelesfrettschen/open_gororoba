//! Frozen-model kernel recovery and post-hoc simultaneous paired uncertainty.
use super::{
    Args, Config, PROTOCOL_SHA256, admission, evidence, features, fitting, metrics, model_path,
    splits,
};
use anyhow::{Context, Result, ensure};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde_json::{Value, json};
use std::{collections::BTreeSet, fs};

fn compatible(left: &metrics::AucKernel, right: &metrics::AucKernel) -> Result<()> {
    left.validate()?;
    right.validate()?;
    ensure!(
        left.file_ids == right.file_ids
            && left.positives == right.positives
            && left.negatives == right.negatives,
        "paired file identities or class counts differ"
    );
    Ok(())
}

fn adjudicate(lower: f64, upper: f64, margin: Option<f64>) -> Value {
    json!({"direction":if lower > 0.0 {"canonical_superiority"} else if upper < 0.0 {"control_superiority"} else {"insufficient_precision"},
        "practical_equivalence":margin.map(|value| lower >= -value && upper <= value),
        "equivalence_margin":margin})
}

fn admit_utility_manifest(manifest: &str, parent_digest: &str) -> Result<()> {
    let addition = "[[bin]]\nname = \"crossing-utility-frontier\"\npath = \"src/bin/crossing_utility_frontier.rs\"\n\n";
    ensure!(
        manifest.matches(addition).count() == 1
            && evidence::digest(manifest.replacen(addition, "", 1).as_bytes()) == parent_digest,
        "manifest differs beyond the independently added utility binary"
    );
    Ok(())
}

pub(super) fn recover(args: &Args, config: &Config, sources: &Value) -> Result<()> {
    ensure!(
        args.external_manifest.is_none() && args.file_map.is_none(),
        "recovery uses sealed primary inputs"
    );
    if let Some(margin) = args.equivalence_margin {
        ensure!(
            margin.is_finite() && margin > 0.0 && margin <= 1.0,
            "equivalence margin must lie in (0,1]"
        );
    }
    let parent = args
        .models_dir
        .as_ref()
        .context("recover-controls requires --models-dir")?;
    ensure!(
        fs::canonicalize(parent)? != fs::canonicalize(&args.out_dir)?,
        "recovery output must differ from parent"
    );
    let parent_bytes = fs::read(parent.join("dataset.json"))?;
    let dataset: Value = serde_json::from_slice(&parent_bytes)?;
    let parent_identity = dataset["identity"]
        .as_str()
        .context("parent identity missing")?;
    ensure!(
        parent_identity == evidence::digest(&serde_json::to_vec(&dataset["provenance"])?),
        "parent dataset checksum mismatch"
    );
    let provenance = &dataset["provenance"];
    for (path, digest) in provenance["sources"]
        .as_object()
        .context("parent sources missing")?
    {
        if path == "crates/gororoba_cli_physics/Cargo.toml" && sources.get(path) != Some(digest) {
            // The utility executable adds no dependency or feature to frozen models.
            let manifest = include_str!("../../../Cargo.toml");
            admit_utility_manifest(
                manifest,
                digest.as_str().context("manifest digest missing")?,
            )?;
            continue;
        }
        if path != "crates/gororoba_cli_physics/src/bin/staples_causal_validation.rs"
            && path != "crates/gororoba_cli_physics/src/bin/staples_causal_validation/evidence.rs"
        {
            ensure!(
                sources.get(path) == Some(digest),
                "frozen scientific/build source differs: {path}"
            );
        }
    }
    ensure!(
        provenance["protocol_sha256"] == PROTOCOL_SHA256
            && provenance["probe"] == "a"
            && provenance["config"] == serde_json::to_value(config)?,
        "parent protocol/config/probe mismatch"
    );
    let map_path = args.input_root.join(&config.file_map);
    ensure!(
        evidence::hash_file(&map_path)? == config.file_map_sha256,
        "primary map checksum mismatch"
    );
    let files = provenance["files"]
        .as_array()
        .context("parent file evidence missing")?;
    let planned: Vec<_> = files
        .iter()
        .filter(|file| {
            file["year"]
                .as_i64()
                .is_some_and(|year| config.final_years.contains(&(year as i32)))
        })
        .collect();
    let ids: BTreeSet<_> = planned
        .iter()
        .map(|file| file["id"].as_u64().context("parent file id missing"))
        .collect::<Result<_>>()?;
    ensure!(
        !ids.is_empty() && ids.len() == planned.len(),
        "invalid parent final file set"
    );
    let entries: Vec<_> = admission::file_map(&map_path)?
        .into_iter()
        .filter(|(id, _)| ids.contains(&u64::from(*id)))
        .collect();
    ensure!(entries.len() == ids.len(), "map omits parent final files");
    for (id, path) in &entries {
        let recorded = planned
            .iter()
            .find(|file| file["id"] == *id)
            .context("parent file association missing")?;
        ensure!(
            recorded["path"] == *path
                && recorded["sha256"] == evidence::hash_file(&args.input_root.join(path))?,
            "raw parent identity mismatch for file {id}"
        );
    }
    let crossings = admission::catalog(
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
        &crossings,
        &ensemble,
    )?;
    for file in &data.files {
        let original = planned
            .iter()
            .find(|entry| entry["id"] == file.id)
            .context("recovered file missing parent")?;
        ensure!(
            serde_json::to_value(file)? == **original,
            "recovered support/labels/evidence differ for file {}",
            file.id
        );
    }
    ensure!(data.files.len() == planned.len(), "recovery omitted files");
    let analysis = json!({"schema":"paired_control_uncertainty_v1","analysis_status":"post_hoc",
        "parent_identity":parent_identity,"parent_dataset_sha256":evidence::digest(&parent_bytes),
        "sources":sources,"executable_sha256":evidence::hash_file(&std::env::current_exe()?)?,
        "protocol_sha256":PROTOCOL_SHA256,"equivalence_margin":args.equivalence_margin,
        "files":data.files,"new_training_fits":0,"draws":config.bootstrap_draws,"seed":config.bootstrap_seed,
        "interval_method":"95 percent simultaneous centered maximum absolute deviation whole-file bootstrap across 114 contrasts"});
    let identity = evidence::digest(&serde_json::to_vec(&analysis)?);
    evidence::preserve(
        &args.out_dir.join("recovery-identity.json"),
        &json!({"identity":identity,"analysis":analysis}),
    )?;
    let mut kernels = Vec::new();
    for (width_index, &width) in config.widths.iter().enumerate() {
        for tensor in 0..20 {
            let model_name = model_path(width, Some(tensor), config);
            let model: fitting::Model = serde_json::from_value(evidence::read_record(
                &parent.join(&model_name),
                parent_identity,
            )?)?;
            model.validate(config)?;
            ensure!(
                model.width == width && model.tensor == Some(tensor),
                "parent model association mismatch"
            );
            let point_name = format!(
                "points-width-{width}-{}.json",
                fitting::name(Some(tensor), config)
            );
            let original = evidence::read_record(&parent.join(&point_name), parent_identity)?;
            ensure!(
                original["width"] == width
                    && original["tensor"] == tensor
                    && original["models_training_identity"] == parent_identity,
                "parent points association mismatch"
            );
            for &year in &config.final_years {
                let reference: Vec<_> = original["years"]
                    .as_array()
                    .context("parent years missing")?
                    .iter()
                    .filter(|record| record["year"] == year)
                    .collect();
                ensure!(
                    reference.len() == 1,
                    "parent final year missing or duplicated"
                );
                let path = args.out_dir.join(format!(
                    "kernel-width-{width}-tensor-{tensor}-year-{year}.json"
                ));
                let recovered = evidence::record(&path, &identity, sources, || {
                    eprintln!("recover frozen width {width} tensor {tensor} year {year}");
                    let (point, kernel) =
                        metrics::evaluate(&data, &model, width_index, year, true)?;
                    Ok(
                        json!({"metrics":point,"kernel":kernel,"model_sha256":evidence::hash_file(&parent.join(&model_name))?,"parent_points_sha256":evidence::hash_file(&parent.join(&point_name))?}),
                    )
                })?;
                ensure!(
                    recovered["model_sha256"] == evidence::hash_file(&parent.join(&model_name))?
                        && recovered["parent_points_sha256"]
                            == evidence::hash_file(&parent.join(&point_name))?,
                    "recovery parent receipt changed"
                );
                let actual: metrics::PointMetrics =
                    serde_json::from_value(recovered["metrics"].clone())?;
                let expected: metrics::PointMetrics =
                    serde_json::from_value(reference[0]["metrics"].clone())?;
                ensure!(
                    (actual.roc_auc - expected.roc_auc).abs() < 1e-12
                        && actual.rows == expected.rows
                        && actual.positives == expected.positives
                        && actual.files == expected.files
                        && actual.positive_files == expected.positive_files,
                    "frozen prediction/support reproduction failed"
                );
                let kernel: metrics::AucKernel =
                    serde_json::from_value(recovered["kernel"].clone())?;
                kernel.validate()?;
                ensure!(
                    (kernel.auc(&vec![1; kernel.file_ids.len()])? - actual.roc_auc).abs() < 1e-12,
                    "recovered kernel point oracle failed"
                );
                kernels.push((width, year, tensor, kernel));
            }
        }
    }
    let report = evidence::record(
        &args.out_dir.join("control-uncertainty.json"),
        &identity,
        sources,
        || bootstrap(&kernels, config, args.equivalence_margin),
    )?;
    eprintln!("control uncertainty: {}", report["status"]);
    Ok(())
}

fn bootstrap(
    kernels: &[(usize, i32, usize, metrics::AucKernel)],
    config: &Config,
    margin: Option<f64>,
) -> Result<Value> {
    bootstrap_extended(kernels, &[], config, margin)
}

pub(super) fn bootstrap_extended(
    kernels: &[(usize, i32, usize, metrics::AucKernel)],
    geometric: &[(usize, i32, metrics::AucKernel)],
    config: &Config,
    margin: Option<f64>,
) -> Result<Value> {
    ensure!(
        kernels.len() == 120,
        "expected 120 canonical/control kernels"
    );
    let keys: BTreeSet<_> = kernels.iter().map(|row| (row.0, row.1, row.2)).collect();
    let expected: BTreeSet<_> = config
        .widths
        .iter()
        .flat_map(|&width| {
            config
                .final_years
                .iter()
                .flat_map(move |&year| (0..20).map(move |tensor| (width, year, tensor)))
        })
        .collect();
    ensure!(
        keys == expected,
        "missing or duplicate width/year/tensor kernel"
    );
    if !geometric.is_empty() {
        let expected_panels: BTreeSet<_> = config
            .widths
            .iter()
            .flat_map(|&width| config.final_years.iter().map(move |&year| (width, year)))
            .collect();
        ensure!(
            geometric.len() == 6
                && geometric
                    .iter()
                    .map(|row| (row.0, row.1))
                    .collect::<BTreeSet<_>>()
                    == expected_panels,
            "geometric panel set incomplete"
        );
    }
    let mut contrasts = Vec::new();
    for &year in &config.final_years {
        let support = &kernels.iter().find(|row| row.1 == year).unwrap().3;
        ensure!(
            support.file_ids.len() >= 30
                && support.positives.iter().filter(|&&count| count > 0).count() >= 10,
            "insufficient file/class support"
        );
        for row in kernels.iter().filter(|row| row.1 == year) {
            compatible(support, &row.3)?;
        }
        for &width in &config.widths {
            let canonical = &kernels
                .iter()
                .find(|row| row.0 == width && row.1 == year && row.2 == 0)
                .unwrap()
                .3;
            for tensor in 1..20 {
                let control = &kernels
                    .iter()
                    .find(|row| row.0 == width && row.1 == year && row.2 == tensor)
                    .unwrap()
                    .3;
                let weights = vec![1; canonical.file_ids.len()];
                contrasts.push((
                    width,
                    year,
                    tensor,
                    canonical,
                    control,
                    canonical.auc(&weights)? - control.auc(&weights)?,
                ));
            }
            if let Some((_, _, control)) =
                geometric.iter().find(|row| row.0 == width && row.1 == year)
            {
                compatible(canonical, control)?;
                let weights = vec![1; canonical.file_ids.len()];
                contrasts.push((
                    width,
                    year,
                    20,
                    canonical,
                    control,
                    canonical.auc(&weights)? - control.auc(&weights)?,
                ));
            }
        }
    }
    let mut random = ChaCha8Rng::seed_from_u64(config.bootstrap_seed);
    let mut maxima = Vec::new();
    let mut draw_receipts = Vec::new();
    for _ in 0..config.bootstrap_draws {
        let memberships: Vec<_> = config
            .final_years
            .iter()
            .map(|&year| {
                let count = kernels
                    .iter()
                    .find(|row| row.1 == year)
                    .unwrap()
                    .3
                    .file_ids
                    .len();
                (year, splits::draw_counts(count, &mut random))
            })
            .collect();
        let differences: Result<Vec<_>> = contrasts
            .par_iter()
            .map(|(_, year, _, canonical, control, point)| {
                let weights = &memberships.iter().find(|row| row.0 == *year).unwrap().1;
                Ok((canonical.auc(weights)? - control.auc(weights)? - point).abs())
            })
            .collect();
        maxima.push(differences?.into_iter().fold(0.0, f64::max));
        draw_receipts.push(memberships);
    }
    maxima.sort_unstable_by(f64::total_cmp);
    let rank = ((0.95 * (maxima.len() + 1) as f64).ceil() as usize).min(maxima.len()) - 1;
    let radius = maxima[rank];
    let comparisons:Vec<_>=contrasts.iter().map(|(width,year,tensor,_,_,point)| {
        let lower=(point-radius).max(-1.0); let upper=(point+radius).min(1.0);
        if *tensor == 20 {
            json!({"width":width,"year":year,"control":"geometric_capacity","seed":null,"canonical_minus_control":point,"simultaneous_interval":[lower,upper],"adjudication":adjudicate(lower,upper,margin)})
        } else {
            json!({"width":width,"year":year,"seed":config.control_seeds[tensor-1],"canonical_minus_control":point,"simultaneous_interval":[lower,upper],"adjudication":adjudicate(lower,upper,margin)})
        }
    }).collect();
    Ok(
        json!({"status":"completed_exact_support_uncertainty","analysis_status":"post_hoc","comparisons":comparisons,
        "family_size":contrasts.len(),"confidence":0.95,"simultaneous_radius":radius,"draws":config.bootstrap_draws,"seed":config.bootstrap_seed,"year_file_multiplicities":draw_receipts,
        "method":"centered maximum absolute bootstrap deviation with common whole-file weights per year across every contrast; approximate simultaneous coverage",
        "assumptions":"Conditions on frozen training fits, observed final years and catalog-selected days. Interday independence is unestablished; bootstrap coverage is approximate. Post-hoc protocol has seen point rankings. Training uncertainty and selection bias remain outside the interval.",
        "equivalence_margin":margin,"margin_source":if margin.is_some(){"explicit caller assumption"}else{"undeclared; equivalence unadjudicated"},
        "declared_discrimination_target":super::metrics::target_declaration(config.minimum_increment),"practical_utility":super::metrics::practical_utility_boundary(),"target_scope":"Historical canonical-minus-baseline target; these canonical-minus-control intervals adjudicate a separate comparison.",
        "matched_geometric_capacity":if geometric.is_empty(){json!({"status":"blocked_missing_training_fit_and_predictions","required":"Declare matched feature count, six-sample support, dimensionality, fixed normalization and logistic ridge/fitting budget; fit geometric transform/model using training years only; freeze before final predictions. Extend the simultaneous family before inspecting new outcomes."})}else{json!({"status":"included_in_same_simultaneous_family","comparisons":6})}}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn utility_manifest_admission_rejects_every_undeclared_delta() {
        let parent = "[package]\nname = \"fixture\"\nversion = \"0.1.0\"\n\n";
        let addition = "[[bin]]\nname = \"crossing-utility-frontier\"\npath = \"src/bin/crossing_utility_frontier.rs\"\n\n";
        let manifest = format!("{parent}{addition}");
        let digest = evidence::digest(parent.as_bytes());
        admit_utility_manifest(&manifest, &digest).unwrap();
        for rejected in [
            parent.to_owned(),
            format!("{manifest}{addition}"),
            format!("{manifest}[dependencies]\nanyhow = \"1\"\n"),
            manifest.replace("0.1.0", "0.2.0"),
        ] {
            assert!(admit_utility_manifest(&rejected, &digest).is_err());
        }
        assert!(
            admit_utility_manifest(
                include_str!("../../../Cargo.toml"),
                "ad463ce5e684b984244f537e1b236f8f4943a7efe8c4ddbcca48b0ca494be645"
            )
            .is_err()
        );
    }
    fn kernel() -> metrics::AucKernel {
        metrics::AucKernel {
            file_ids: vec![0, 1],
            positives: vec![1, 1],
            negatives: vec![1, 1],
            wins: vec![0.5; 4],
        }
    }
    #[test]
    fn paired_support_rejects_identity_labels_and_invalid_wins() {
        let original = kernel();
        let mut changed = original.clone();
        changed.file_ids[1] = 2;
        assert!(compatible(&original, &changed).is_err());
        changed = original.clone();
        changed.positives[1] = 2;
        assert!(compatible(&original, &changed).is_err());
        changed = original.clone();
        changed.wins[0] = f64::NAN;
        assert!(compatible(&original, &changed).is_err());
    }
    #[test]
    fn absent_kernels_and_unset_margin_cannot_establish_equivalence() {
        assert!(bootstrap(&[], &crate::test_config(), None).is_err());
        assert!(adjudicate(-0.01, 0.01, None)["practical_equivalence"].is_null());
        assert_eq!(
            adjudicate(-0.01, 0.01, Some(0.02))["practical_equivalence"],
            true
        );
        assert_eq!(
            adjudicate(-0.03, 0.01, Some(0.02))["practical_equivalence"],
            false
        );
    }
    #[test]
    fn identical_controls_have_zero_simultaneous_radius() -> Result<()> {
        let mut config = crate::test_config();
        config.bootstrap_draws = 20;
        let kernel = metrics::AucKernel {
            file_ids: (0..30).collect(),
            positives: vec![1; 30],
            negatives: vec![1; 30],
            wins: vec![0.5; 900],
        };
        let mut kernels = Vec::new();
        for width in config.widths {
            for &year in &config.final_years {
                for tensor in 0..20 {
                    kernels.push((width, year, tensor, kernel.clone()));
                }
            }
        }
        let report = bootstrap(&kernels, &config, None)?;
        assert_eq!(report["simultaneous_radius"], 0.0);
        assert_eq!(report["comparisons"].as_array().unwrap().len(), 114);
        let geometric: Vec<_> = config
            .widths
            .iter()
            .flat_map(|&width| config.final_years.iter().map(move |&year| (width, year)))
            .map(|(width, year)| (width, year, kernel.clone()))
            .collect();
        let extended = bootstrap_extended(&kernels, &geometric, &config, None)?;
        assert_eq!(extended["family_size"], 120);
        assert_eq!(
            extended["year_file_multiplicities"],
            report["year_file_multiplicities"]
        );
        assert_eq!(extended["simultaneous_radius"], 0.0);
        assert!(bootstrap_extended(&kernels, &geometric[..5], &config, None).is_err());
        let mut changed_geometric = geometric.clone();
        changed_geometric[0].2.wins[0] = 0.0;
        let changed_report = bootstrap_extended(&kernels, &changed_geometric, &config, None)?;
        assert!(changed_report["simultaneous_radius"].as_f64().unwrap() > 0.0);
        kernels[1].3.file_ids[0] = 99;
        assert!(bootstrap(&kernels, &config, None).is_err());
        Ok(())
    }
}

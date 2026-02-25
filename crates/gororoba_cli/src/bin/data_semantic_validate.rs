use anyhow::Result;
use chrono::{DateTime, Utc};
use clap::{ArgAction, Parser};
use csv::ReaderBuilder;
use gororoba_cli::data_governance::{
    DEFAULT_ARTIFACTS_MANIFEST_PATH, DEFAULT_EVIDENCE_MANIFEST_PATH,
    DEFAULT_EXTERNAL_PROVENANCE_PATH, DEFAULT_EXTERNAL_SOURCES_PATH, DEFAULT_GOVERNANCE_PATH,
    DEFAULT_SEMANTIC_VALIDATORS_PATH, artifacts_manifest_matches, blocked_source_deadline_issues,
    collect_files_under, load_artifacts_manifest, load_data_governance, load_evidence_manifest,
    load_external_hashes, load_external_sources, load_semantic_validators,
    missing_semantic_lane_validators, parse_deadline_utc, source_rule_for_path,
    validators_for_lane,
};
use serde::Serialize;
use std::collections::BTreeSet;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "data-semantic-validate",
    about = "Run semantic validation contracts for all governed data lanes"
)]
struct Args {
    #[arg(long, default_value = DEFAULT_GOVERNANCE_PATH)]
    governance: PathBuf,
    #[arg(long, default_value = DEFAULT_SEMANTIC_VALIDATORS_PATH)]
    semantic_validators: PathBuf,
    #[arg(long, default_value = DEFAULT_EXTERNAL_SOURCES_PATH)]
    external_sources: PathBuf,
    #[arg(long, default_value = DEFAULT_EXTERNAL_PROVENANCE_PATH)]
    external_provenance: PathBuf,
    #[arg(long, default_value = DEFAULT_ARTIFACTS_MANIFEST_PATH)]
    artifacts_manifest: PathBuf,
    #[arg(long, default_value = DEFAULT_EVIDENCE_MANIFEST_PATH)]
    evidence_manifest: PathBuf,
    #[arg(long)]
    lane: Vec<String>,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long, default_value_t = false, action = ArgAction::Set)]
    fail_fast: bool,
    #[arg(long, default_value_t = false, action = ArgAction::Set)]
    fail_on_unverifiable: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValidationStatus {
    Pass,
    Fail,
    Unverifiable,
}

impl ValidationStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Fail => "fail",
            Self::Unverifiable => "unverifiable",
        }
    }
}

#[derive(Debug, Default)]
struct ValidatorOutcome {
    checked_files: usize,
    errors: Vec<String>,
    unverifiable: Vec<String>,
}

impl ValidatorOutcome {
    fn status(&self) -> ValidationStatus {
        if !self.errors.is_empty() {
            ValidationStatus::Fail
        } else if !self.unverifiable.is_empty() {
            ValidationStatus::Unverifiable
        } else {
            ValidationStatus::Pass
        }
    }

    fn from_errors(checked_files: usize, errors: Vec<String>) -> Self {
        Self {
            checked_files,
            errors,
            unverifiable: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct ValidatorResult {
    lane_id: String,
    validator_id: String,
    validator_kind: String,
    status: String,
    checked_files: usize,
    error_count: usize,
    errors: Vec<String>,
    unverifiable_count: usize,
    unverifiable: Vec<String>,
}

#[derive(Debug, Serialize)]
struct SemanticValidationReport {
    generated_at_utc: String,
    total_validators: usize,
    passed: usize,
    failed: usize,
    unverifiable: usize,
    results: Vec<ValidatorResult>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let governance = load_data_governance(&args.governance)?;
    let validators = load_semantic_validators(&args.semantic_validators)?;
    let external_sources = load_external_sources(&args.external_sources)?;
    let external_hashes = load_external_hashes(&args.external_provenance)?;
    let artifacts_manifest = load_artifacts_manifest(&args.artifacts_manifest)?;
    let evidence_manifest = load_evidence_manifest(&args.evidence_manifest)?;

    let missing_lanes = missing_semantic_lane_validators(&governance, &validators);
    let mut results = Vec::new();
    if !missing_lanes.is_empty() {
        for lane_id in &missing_lanes {
            results.push(ValidatorResult {
                lane_id: lane_id.clone(),
                validator_id: "<coverage>".to_string(),
                validator_kind: "coverage".to_string(),
                status: "fail".to_string(),
                checked_files: 0,
                error_count: 1,
                errors: vec![format!("lane {lane_id} has no enabled semantic validator")],
                unverifiable_count: 0,
                unverifiable: Vec::new(),
            });
        }
    }

    for lane in &governance.lane {
        if !args.lane.is_empty() && !args.lane.iter().any(|wanted| wanted == &lane.id) {
            continue;
        }
        let lane_files = collect_files_under(std::path::Path::new(&lane.root))?;
        for validator in validators_for_lane(&lane.id, &validators) {
            let outcome = run_validator(
                validator.validator_kind.as_str(),
                &lane_files,
                lane.id.as_str(),
                &external_sources,
                &external_hashes,
                &artifacts_manifest,
                &evidence_manifest,
            );
            if args.fail_fast
                && (outcome.status() == ValidationStatus::Fail
                    || (args.fail_on_unverifiable
                        && outcome.status() == ValidationStatus::Unverifiable))
            {
                let result = ValidatorResult {
                    lane_id: lane.id.clone(),
                    validator_id: validator.id.clone(),
                    validator_kind: validator.validator_kind.clone(),
                    status: outcome.status().as_str().to_string(),
                    checked_files: outcome.checked_files,
                    error_count: outcome.errors.len(),
                    errors: outcome.errors,
                    unverifiable_count: outcome.unverifiable.len(),
                    unverifiable: outcome.unverifiable,
                };
                results.push(result);
                return write_and_exit(&args, &results);
            }

            results.push(ValidatorResult {
                lane_id: lane.id.clone(),
                validator_id: validator.id.clone(),
                validator_kind: validator.validator_kind.clone(),
                status: outcome.status().as_str().to_string(),
                checked_files: outcome.checked_files,
                error_count: outcome.errors.len(),
                errors: outcome.errors,
                unverifiable_count: outcome.unverifiable.len(),
                unverifiable: outcome.unverifiable,
            });
        }
    }

    write_and_exit(&args, &results)
}

fn write_and_exit(args: &Args, results: &[ValidatorResult]) -> Result<()> {
    let passed = results.iter().filter(|r| r.status == "pass").count();
    let failed = results.iter().filter(|r| r.status == "fail").count();
    let unverifiable = results
        .iter()
        .filter(|r| r.status == "unverifiable")
        .count();
    let report = SemanticValidationReport {
        generated_at_utc: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
        total_validators: results.len(),
        passed,
        failed,
        unverifiable,
        results: results.to_vec(),
    };

    println!("DATA_SEMANTIC_VALIDATE");
    println!("  total_validators={}", report.total_validators);
    println!("  passed={}", report.passed);
    println!("  failed={}", report.failed);
    println!("  unverifiable={}", report.unverifiable);

    if let Some(out) = &args.out {
        let body = if out.extension().and_then(|s| s.to_str()) == Some("json") {
            serde_json::to_string_pretty(&report)?
        } else {
            toml::to_string_pretty(&report)?
        };
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(out, body + "\n")?;
        println!("WROTE {}", out.display());
    }

    if report.failed > 0 {
        anyhow::bail!("semantic validation failed: {} validator(s)", report.failed);
    }
    if args.fail_on_unverifiable && report.unverifiable > 0 {
        anyhow::bail!(
            "semantic validation has {} unverifiable validator(s)",
            report.unverifiable
        );
    }
    Ok(())
}

fn run_validator(
    kind: &str,
    lane_files: &[String],
    lane_id: &str,
    external_sources: &gororoba_cli::data_governance::ExternalSourcesRoot,
    external_hashes: &std::collections::HashMap<
        String,
        gororoba_cli::data_governance::ExternalHashEntry,
    >,
    artifacts_manifest: &std::collections::HashMap<
        String,
        gororoba_cli::data_governance::ArtifactManifestEntry,
    >,
    evidence_manifest: &std::collections::HashMap<
        String,
        gororoba_cli::data_governance::EvidenceManifestEntry,
    >,
) -> ValidatorOutcome {
    match kind {
        "external_dataset_semantics" => {
            validate_external_lane(lane_files, external_sources, external_hashes)
        }
        "artifacts_manifest_coverage" => ValidatorOutcome::from_errors(
            lane_files.len(),
            validate_artifacts_lane(lane_files, artifacts_manifest),
        ),
        "evidence_manifest_integrity" => ValidatorOutcome::from_errors(
            lane_files.len(),
            validate_evidence_lane(lane_files, evidence_manifest),
        ),
        "csv_lane_structure" => validate_csv_lane(lane_files),
        "toml_lane_structure" => {
            ValidatorOutcome::from_errors(lane_files.len(), validate_toml_lane(lane_files))
        }
        "binary_lane_structure" | "h5_lane_structure" => {
            ValidatorOutcome::from_errors(lane_files.len(), validate_binary_lane(lane_files))
        }
        "generic_lane_structure" => ValidatorOutcome::from_errors(
            lane_files.len(),
            validate_generic_lane(lane_files, lane_id),
        ),
        other => ValidatorOutcome::from_errors(
            lane_files.len(),
            vec![format!("unsupported validator_kind {other}")],
        ),
    }
}

fn validate_external_lane(
    lane_files: &[String],
    external_sources: &gororoba_cli::data_governance::ExternalSourcesRoot,
    external_hashes: &std::collections::HashMap<
        String,
        gororoba_cli::data_governance::ExternalHashEntry,
    >,
) -> ValidatorOutcome {
    let mut errors = Vec::new();
    let mut unverifiable = Vec::new();
    let now = Utc::now();
    let mut scientific_checks = BTreeSet::new();

    for path in lane_files {
        if path == "data/external/README.md"
            || path == "data/external/PROVENANCE.local.json"
            || path == "data/external/SOURCES.toml"
        {
            continue;
        }
        let rule = source_rule_for_path(path, external_sources);
        if rule.is_none() {
            errors.push(format!("missing source rule for {path}"));
        }
        if !external_hashes.contains_key(path) {
            errors.push(format!("missing external hash row for {path}"));
        } else if let Some(rule) = rule
            && let Some(hash_row) = external_hashes.get(path)
        {
            if hash_row.source_id.trim().is_empty() {
                errors.push(format!("external hash missing source_id for {path}"));
            } else if hash_row.source_id != rule.id {
                errors.push(format!(
                    "external hash source_id mismatch for {path}: {} != {}",
                    hash_row.source_id, rule.id
                ));
            }
            if hash_row.source_canonical_url.trim().is_empty() {
                errors.push(format!(
                    "external hash missing source_canonical_url for {path}"
                ));
            }
        }
    }

    for issue in blocked_source_deadline_issues(external_sources, now) {
        errors.push(format!("blocked-source policy issue: {issue}"));
    }

    for path in lane_files {
        if path == "data/external/README.md"
            || path == "data/external/PROVENANCE.local.json"
            || path == "data/external/SOURCES.toml"
        {
            continue;
        }
        let Some(rule) = source_rule_for_path(path, external_sources) else {
            continue;
        };
        if should_skip_blocked_semantics(path, external_sources, now) {
            match run_source_scientific_validators(path, rule, external_hashes) {
                Ok(true) => {
                    scientific_checks.insert(path.to_string());
                }
                Ok(false) => {
                    unverifiable.push(format!("blocked source semantics not runnable yet: {path}"));
                }
                Err(err) => {
                    errors.push(format!(
                        "blocked source semantic contract failed for {path}: {err}"
                    ));
                }
            }
            continue;
        }
        if path != "data/external/jarvis_dft_3d.json"
            && let Err(err) = validate_file_semantics(path)
        {
            errors.push(err);
            continue;
        }
        match run_source_scientific_validators(path, rule, external_hashes) {
            Ok(true) => {
                scientific_checks.insert(path.to_string());
            }
            Ok(false) => {
                if !has_registered_scientific_checker(path) {
                    unverifiable.push(format!(
                        "no scientific checker registered for active external dataset: {path}"
                    ));
                }
            }
            Err(err) => {
                errors.push(format!(
                    "source scientific validator failed for {path}: {err}"
                ));
            }
        }
    }

    if let Some(path) = existing_path("data/external/chime_frb_cat2.csv")
        && !should_skip_blocked_semantics(path.to_string_lossy().as_ref(), external_sources, now)
    {
        match data_core::parse_chime_csv(&path) {
            Ok(rows) if !rows.is_empty() => {
                if let Err(err) = check_chime_rows(&rows) {
                    errors.push(format!("CHIME Cat2 scientific checks failed: {err}"));
                } else {
                    scientific_checks.insert("data/external/chime_frb_cat2.csv".to_string());
                }
            }
            Ok(_) => errors.push("CHIME Cat2 parsed but returned zero rows".to_string()),
            Err(err) => errors.push(format!("CHIME Cat2 parse failed: {err}")),
        }
    }

    if let Some(path) = existing_path("data/external/chime_frb_cat1.csv")
        && !should_skip_blocked_semantics(path.to_string_lossy().as_ref(), external_sources, now)
    {
        match data_core::parse_chime_csv(&path) {
            Ok(rows) if !rows.is_empty() => {
                if let Err(err) = check_chime_rows(&rows) {
                    errors.push(format!("CHIME Cat1 scientific checks failed: {err}"));
                } else {
                    scientific_checks.insert("data/external/chime_frb_cat1.csv".to_string());
                }
            }
            Ok(_) => errors.push("CHIME Cat1 parsed but returned zero rows".to_string()),
            Err(err) => errors.push(format!("CHIME Cat1 parse failed: {err}")),
        }
    }

    if let Some(path) = existing_path("data/external/atnf_pulsars.csv")
        && !should_skip_blocked_semantics(path.to_string_lossy().as_ref(), external_sources, now)
    {
        match data_core::parse_atnf_csv(&path) {
            Ok(rows) if !rows.is_empty() => {
                if let Err(err) = check_atnf_rows(&rows) {
                    errors.push(format!("ATNF scientific checks failed: {err}"));
                } else {
                    scientific_checks.insert("data/external/atnf_pulsars.csv".to_string());
                }
            }
            Ok(_) => errors.push("ATNF parsed but returned zero rows".to_string()),
            Err(err) => errors.push(format!("ATNF parse failed: {err}")),
        }
    }

    if let Some(path) = existing_path("data/external/sdss_dr18_quasars.csv")
        && !should_skip_blocked_semantics(path.to_string_lossy().as_ref(), external_sources, now)
    {
        match data_core::parse_sdss_quasar_csv(&path) {
            Ok(rows) if !rows.is_empty() => {
                if let Err(err) = check_sdss_rows(&rows) {
                    errors.push(format!("SDSS scientific checks failed: {err}"));
                } else {
                    scientific_checks.insert("data/external/sdss_dr18_quasars.csv".to_string());
                }
            }
            Ok(_) => errors.push("SDSS parsed but returned zero rows".to_string()),
            Err(err) => errors.push(format!("SDSS parse failed: {err}")),
        }
    }

    if let Some(path) = existing_path("data/external/gaia_dr3_nearby.csv")
        && !should_skip_blocked_semantics(path.to_string_lossy().as_ref(), external_sources, now)
    {
        match data_core::parse_gaia_csv(&path) {
            Ok(rows) if !rows.is_empty() => {
                if let Err(err) = check_gaia_rows(&rows) {
                    errors.push(format!("Gaia scientific checks failed: {err}"));
                } else {
                    scientific_checks.insert("data/external/gaia_dr3_nearby.csv".to_string());
                }
            }
            Ok(_) => errors.push("Gaia parsed but returned zero rows".to_string()),
            Err(err) => errors.push(format!("Gaia parse failed: {err}")),
        }
    }

    if let Some(path) = existing_path("data/external/fermi_gbm_grbs.csv")
        && !should_skip_blocked_semantics(path.to_string_lossy().as_ref(), external_sources, now)
    {
        match data_core::parse_fermi_gbm_csv(&path) {
            Ok(rows) if !rows.is_empty() => {
                if let Err(err) = check_fermi_rows(&rows) {
                    errors.push(format!("Fermi scientific checks failed: {err}"));
                } else {
                    scientific_checks.insert("data/external/fermi_gbm_grbs.csv".to_string());
                }
            }
            Ok(_) => errors.push("Fermi GBM parsed but returned zero rows".to_string()),
            Err(err) => errors.push(format!("Fermi GBM parse failed: {err}")),
        }
    }

    if let Some(path) = existing_path("data/external/PantheonPlusSH0ES.dat")
        && !should_skip_blocked_semantics(path.to_string_lossy().as_ref(), external_sources, now)
    {
        match data_core::parse_pantheon_dat(&path) {
            Ok(rows) if !rows.is_empty() => {
                if let Err(err) = check_pantheon_rows(&rows) {
                    errors.push(format!("Pantheon scientific checks failed: {err}"));
                } else {
                    scientific_checks.insert("data/external/PantheonPlusSH0ES.dat".to_string());
                }
            }
            Ok(_) => errors.push("Pantheon parsed but returned zero rows".to_string()),
            Err(err) => errors.push(format!("Pantheon parse failed: {err}")),
        }
    }

    if let Some(path) = existing_path("data/external/union3_chain_1.txt")
        && !should_skip_blocked_semantics(path.to_string_lossy().as_ref(), external_sources, now)
    {
        match data_core::parse_union3_chain(&path) {
            Ok(rows) if !rows.is_empty() => {
                if let Err(err) = check_union3_rows(&rows) {
                    errors.push(format!("Union3 scientific checks failed: {err}"));
                } else {
                    scientific_checks.insert("data/external/union3_chain_1.txt".to_string());
                }
            }
            Ok(_) => errors.push("Union3 parsed but returned zero rows".to_string()),
            Err(err) => errors.push(format!("Union3 parse failed: {err}")),
        }
    }

    if let Some(path) = existing_path("data/external/hip_main.dat")
        && !should_skip_blocked_semantics(path.to_string_lossy().as_ref(), external_sources, now)
        && let Err(err) = data_core::catalogs::hipparcos::validate_hipparcos_format(&path, 400)
    {
        errors.push(format!("Hipparcos format validation failed: {err}"));
    } else if existing_path("data/external/hip_main.dat").is_some() {
        scientific_checks.insert("data/external/hip_main.dat".to_string());
    }

    if let Some(path) = existing_path("data/external/landsat_c2l2_sr_sample.json")
        && !should_skip_blocked_semantics(path.to_string_lossy().as_ref(), external_sources, now)
        && let Err(err) = data_core::catalogs::landsat::validate_stac_schema(&path)
    {
        errors.push(format!("Landsat STAC schema validation failed: {err}"));
    } else if let Some(path) = existing_path("data/external/landsat_c2l2_sr_sample.json")
        && !should_skip_blocked_semantics(path.to_string_lossy().as_ref(), external_sources, now)
    {
        let cloud_cover = data_core::catalogs::landsat::extract_cloud_cover(&path);
        let asset_count = data_core::catalogs::landsat::count_stac_assets(&path);
        match (cloud_cover, asset_count) {
            (Ok(cloud), Ok(assets)) => {
                if assets == 0 {
                    errors.push("Landsat STAC has zero assets".to_string());
                }
                if let Some(value) = cloud
                    && !(0.0..=100.0).contains(&value)
                {
                    errors.push(format!("Landsat cloud cover out of range [0,100]: {value}"));
                }
                if assets > 0 {
                    scientific_checks
                        .insert("data/external/landsat_c2l2_sr_sample.json".to_string());
                }
            }
            (Err(err), _) => errors.push(format!("Landsat cloud cover extraction failed: {err}")),
            (_, Err(err)) => errors.push(format!("Landsat asset count failed: {err}")),
        }
    }

    if let Some(path) = existing_path("data/external/sorce_tsi_daily.csv")
        && !should_skip_blocked_semantics(path.to_string_lossy().as_ref(), external_sources, now)
    {
        match data_core::parse_sorce_csv(&path) {
            Ok(rows) if !rows.is_empty() => {
                if let Err(err) = check_sorce_rows(&rows) {
                    errors.push(format!("SORCE scientific checks failed: {err}"));
                } else {
                    scientific_checks.insert("data/external/sorce_tsi_daily.csv".to_string());
                }
            }
            Ok(_) => errors.push("SORCE parsed but returned zero rows".to_string()),
            Err(err) => errors.push(format!("SORCE parse failed: {err}")),
        }
    }

    if let Some(path) = existing_path("data/external/tsis1_tsi_daily.csv")
        && !should_skip_blocked_semantics(path.to_string_lossy().as_ref(), external_sources, now)
    {
        match data_core::parse_tsi_csv(&path) {
            Ok(rows) if !rows.is_empty() => {
                if let Err(err) = check_tsi_rows(&rows) {
                    errors.push(format!("TSIS scientific checks failed: {err}"));
                } else {
                    scientific_checks.insert("data/external/tsis1_tsi_daily.csv".to_string());
                }
            }
            Ok(_) => errors.push("TSIS parsed but returned zero rows".to_string()),
            Err(err) => errors.push(format!("TSIS parse failed: {err}")),
        }
    }

    if let Some(path) = existing_path("data/external/nanograv_15yr_freespectrum.csv")
        && !should_skip_blocked_semantics(path.to_string_lossy().as_ref(), external_sources, now)
    {
        match data_core::parse_nanograv_free_spectrum(&path) {
            Ok(rows) if !rows.is_empty() => {
                if let Err(err) = check_nanograv_rows(&rows) {
                    errors.push(format!("NANOGrav scientific checks failed: {err}"));
                } else {
                    scientific_checks
                        .insert("data/external/nanograv_15yr_freespectrum.csv".to_string());
                }
            }
            Ok(_) => errors.push("NANOGrav parsed but returned zero rows".to_string()),
            Err(err) => errors.push(format!("NANOGrav parse failed: {err}")),
        }
    }

    if let Some(path) = existing_path("data/external/jpl_planets_2020_2030.csv")
        && !should_skip_blocked_semantics(path.to_string_lossy().as_ref(), external_sources, now)
    {
        match data_core::geophysical::jpl_ephemeris::parse_jpl_ephemeris_csv(&path) {
            Ok(rows) if !rows.is_empty() => {
                if let Err(err) = check_jpl_rows(&rows) {
                    errors.push(format!("JPL ephemeris scientific checks failed: {err}"));
                } else {
                    scientific_checks.insert("data/external/jpl_planets_2020_2030.csv".to_string());
                }
            }
            Ok(_) => errors.push("JPL ephemeris parsed but returned zero rows".to_string()),
            Err(err) => errors.push(format!("JPL ephemeris parse failed: {err}")),
        }
    }

    if let Some(path) = existing_path("data/external/swarm_magnetic_sample.csv")
        && !should_skip_blocked_semantics(path.to_string_lossy().as_ref(), external_sources, now)
    {
        match data_core::geophysical::swarm::parse_swarm_csv(&path) {
            Ok(rows) if !rows.is_empty() => {
                if let Err(err) = check_swarm_rows(&rows) {
                    errors.push(format!("Swarm scientific checks failed: {err}"));
                } else if let Err(err) =
                    data_core::geophysical::swarm::check_timestamp_monotonicity(&rows)
                {
                    errors.push(format!("Swarm monotonicity check failed: {err}"));
                } else {
                    scientific_checks.insert("data/external/swarm_magnetic_sample.csv".to_string());
                }
            }
            Ok(_) => errors.push("Swarm CSV parsed but returned zero rows".to_string()),
            Err(err) => errors.push(format!("Swarm parse failed: {err}")),
        }
    }

    if let Some(path) = existing_path("data/external/jarvis_dft_3d.json")
        && !should_skip_blocked_semantics(path.to_string_lossy().as_ref(), external_sources, now)
    {
        match data_core::parse_jarvis_json(&path) {
            Ok(rows) if !rows.is_empty() => {
                if let Err(err) = check_jarvis_rows(&rows) {
                    errors.push(format!("JARVIS scientific checks failed: {err}"));
                } else {
                    scientific_checks.insert("data/external/jarvis_dft_3d.json".to_string());
                }
            }
            Ok(_) => errors.push("JARVIS parsed but returned zero rows".to_string()),
            Err(err) => errors.push(format!("JARVIS parse failed: {err}")),
        }
    }

    if let Some(path) = existing_path("data/external/GWTC-3_confident.csv")
        && !should_skip_blocked_semantics(path.to_string_lossy().as_ref(), external_sources, now)
    {
        match data_core::parse_gwtc3_csv(&path) {
            Ok(rows) if !rows.is_empty() => {
                if let Err(err) = check_gwtc_rows(&rows) {
                    errors.push(format!("GWTC scientific checks failed: {err}"));
                } else {
                    scientific_checks.insert("data/external/GWTC-3_confident.csv".to_string());
                }
            }
            Ok(_) => errors.push("GWTC parsed but returned zero rows".to_string()),
            Err(err) => errors.push(format!("GWTC parse failed: {err}")),
        }
    }

    if let Some(path) = existing_path("data/external/mcgill_magnetars.csv")
        && !should_skip_blocked_semantics(path.to_string_lossy().as_ref(), external_sources, now)
    {
        match data_core::catalogs::mcgill::parse_mcgill_csv(&path) {
            Ok(rows) if !rows.is_empty() => {
                if let Err(err) = check_mcgill_rows(&rows) {
                    errors.push(format!("McGill scientific checks failed: {err}"));
                } else {
                    scientific_checks.insert("data/external/mcgill_magnetars.csv".to_string());
                }
            }
            Ok(_) => errors.push("McGill parsed but returned zero rows".to_string()),
            Err(err) => errors.push(format!("McGill parse failed: {err}")),
        }
    }

    for path in lane_files {
        if path == "data/external/README.md"
            || path == "data/external/PROVENANCE.local.json"
            || path == "data/external/SOURCES.toml"
        {
            continue;
        }
        if should_skip_blocked_semantics(path, external_sources, now) {
            continue;
        }
        if has_registered_scientific_checker(path) && !scientific_checks.contains(path) {
            unverifiable.push(format!(
                "scientific checker is registered but did not run successfully: {path}"
            ));
        }
    }

    ValidatorOutcome {
        checked_files: lane_files.len(),
        errors,
        unverifiable,
    }
}

fn validate_artifacts_lane(
    lane_files: &[String],
    artifacts_manifest: &std::collections::HashMap<
        String,
        gororoba_cli::data_governance::ArtifactManifestEntry,
    >,
) -> Vec<String> {
    let mut errors = Vec::new();
    for path in lane_files {
        if path == "data/artifacts/ARTIFACTS_MANIFEST.csv"
            || path == "data/artifacts/README.md"
            || path == "data/artifacts/PROVENANCE.local.json"
        {
            continue;
        }
        if artifacts_manifest_matches(path, artifacts_manifest) {
            continue;
        }
        if path.ends_with(".PROVENANCE.json") {
            let base = path.trim_end_matches(".PROVENANCE.json");
            if artifacts_manifest_matches(base, artifacts_manifest) {
                continue;
            }
        }
        errors.push(format!("artifact file missing manifest row: {path}"));
    }
    errors
}

fn validate_evidence_lane(
    lane_files: &[String],
    evidence_manifest: &std::collections::HashMap<
        String,
        gororoba_cli::data_governance::EvidenceManifestEntry,
    >,
) -> Vec<String> {
    let mut errors = Vec::new();
    for path in lane_files {
        if path == "data/evidence/MANIFEST.toml" {
            continue;
        }
        let Some(entry) = evidence_manifest.get(path) else {
            errors.push(format!("evidence file missing MANIFEST row: {path}"));
            continue;
        };
        if entry.sha256.len() != 64 {
            errors.push(format!("evidence entry has non-SHA256 hash length: {path}"));
        }
        if entry.status != "present" {
            errors.push(format!(
                "evidence entry status is not present: {path} ({})",
                entry.status
            ));
        }
    }
    errors
}

fn validate_csv_lane(lane_files: &[String]) -> ValidatorOutcome {
    let mut errors = Vec::new();
    for path in lane_files {
        if !path.ends_with(".csv") {
            continue;
        }
        if let Err(err) = validate_csv_file(path) {
            errors.push(err);
        }
    }
    ValidatorOutcome {
        checked_files: lane_files.len(),
        errors,
        unverifiable: Vec::new(),
    }
}

fn validate_toml_lane(lane_files: &[String]) -> Vec<String> {
    let mut errors = Vec::new();
    for path in lane_files {
        if !path.ends_with(".toml") {
            continue;
        }
        if let Err(err) = validate_toml_file(path) {
            errors.push(err);
        }
    }
    errors
}

fn validate_binary_lane(lane_files: &[String]) -> Vec<String> {
    let mut errors = Vec::new();
    for path in lane_files {
        if let Err(err) = validate_binary_nonempty(path) {
            errors.push(err);
        }
    }
    errors
}

fn validate_generic_lane(lane_files: &[String], _lane_id: &str) -> Vec<String> {
    let mut errors = Vec::new();
    for path in lane_files {
        if let Err(err) = validate_file_semantics(path) {
            errors.push(err);
        }
    }
    errors
}

fn validate_file_semantics(path: &str) -> Result<(), String> {
    if path.ends_with(".csv") {
        validate_csv_file(path)
    } else if path.ends_with(".toml") {
        validate_toml_file(path)
    } else if path.ends_with(".json") {
        validate_json_file(path)
    } else if path.ends_with(".pdf") {
        validate_pdf_magic(path)
    } else {
        validate_binary_nonempty(path)
    }
}

fn validate_csv_file(path: &str) -> Result<(), String> {
    let mut reader = ReaderBuilder::new()
        .comment(Some(b'#'))
        .flexible(true)
        .from_path(path)
        .map_err(|err| format!("CSV open failed {path}: {err}"))?;
    let headers = reader
        .headers()
        .map_err(|err| format!("CSV headers failed {path}: {err}"))?;
    if headers.is_empty() {
        return Err(format!("CSV headers empty for {path}"));
    }
    let mut row_count = 0usize;
    for record in reader.records() {
        record.map_err(|err| format!("CSV row parse failed {path}: {err}"))?;
        row_count += 1;
        if row_count >= 1 {
            break;
        }
    }
    if row_count == 0 {
        return Err(format!("CSV has no data rows: {path}"));
    }
    Ok(())
}

fn validate_toml_file(path: &str) -> Result<(), String> {
    let raw =
        std::fs::read_to_string(path).map_err(|err| format!("TOML read failed {path}: {err}"))?;
    toml::from_str::<toml::Value>(&raw)
        .map_err(|err| format!("TOML parse failed {path}: {err}"))?;
    Ok(())
}

fn validate_json_file(path: &str) -> Result<(), String> {
    let raw =
        std::fs::read_to_string(path).map_err(|err| format!("JSON read failed {path}: {err}"))?;
    serde_json::from_str::<serde_json::Value>(&raw)
        .map_err(|err| format!("JSON parse failed {path}: {err}"))?;
    Ok(())
}

fn validate_pdf_magic(path: &str) -> Result<(), String> {
    let data = std::fs::read(path).map_err(|err| format!("PDF read failed {path}: {err}"))?;
    if data.len() < 5 || !data.starts_with(b"%PDF-") {
        return Err(format!("PDF magic mismatch for {path}"));
    }
    Ok(())
}

fn validate_binary_nonempty(path: &str) -> Result<(), String> {
    let metadata =
        std::fs::metadata(path).map_err(|err| format!("metadata failed {path}: {err}"))?;
    if metadata.len() == 0 {
        return Err(format!("file is empty: {path}"));
    }
    Ok(())
}

fn should_skip_blocked_semantics(
    path: &str,
    external_sources: &gororoba_cli::data_governance::ExternalSourcesRoot,
    now: DateTime<Utc>,
) -> bool {
    if let Some(rule) = source_rule_for_path(path, external_sources)
        && rule.is_blocked()
    {
        if let Some(deadline) = parse_deadline_utc(&rule.resolution_deadline_utc) {
            return now <= deadline;
        }
        return true;
    }
    false
}

fn existing_path(path: &str) -> Option<PathBuf> {
    let candidate = PathBuf::from(path);
    if candidate.exists() {
        Some(candidate)
    } else {
        None
    }
}

fn run_source_scientific_validators(
    path: &str,
    rule: &gororoba_cli::data_governance::ExternalSourceRule,
    external_hashes: &std::collections::HashMap<
        String,
        gororoba_cli::data_governance::ExternalHashEntry,
    >,
) -> Result<bool, String> {
    if rule.scientific_validator_refs.is_empty() {
        return Ok(false);
    }
    for validator in &rule.scientific_validator_refs {
        match validator.as_str() {
            "replay_contract_hash" => validate_replay_contract_hash(path, rule, external_hashes)?,
            "swarm_physics_invariants"
            | "tsi_physics_invariants"
            | "materials_json_invariants"
            | "heavy_ion_raa_invariants"
            | "spectroscopy_numeric_invariants"
            | "radio_catalog_numeric_invariants" => validate_numeric_signal_contract(path)?,
            "blocked_manifest_only" => validate_blocked_manifest_contract(path, rule)?,
            other => {
                return Err(format!(
                    "unsupported scientific_validator_refs entry {other} for source {}",
                    rule.id
                ));
            }
        }
    }
    Ok(true)
}

fn validate_replay_contract_hash(
    path: &str,
    rule: &gororoba_cli::data_governance::ExternalSourceRule,
    external_hashes: &std::collections::HashMap<
        String,
        gororoba_cli::data_governance::ExternalHashEntry,
    >,
) -> Result<(), String> {
    if !rule.is_active() {
        return Err(format!(
            "replay_contract_hash used by non-active source {} ({path})",
            rule.id
        ));
    }
    if !rule
        .retrieval_method
        .starts_with("cargo run -p gororoba_cli --bin ")
    {
        return Err(format!(
            "source {} has non-Rust retrieval_method for replay_contract_hash",
            rule.id
        ));
    }
    if !external_hashes.contains_key(path) {
        return Err(format!(
            "source {} missing external hash row for replay contract path {}",
            rule.id, path
        ));
    }
    Ok(())
}

fn validate_numeric_signal_contract(path: &str) -> Result<(), String> {
    if path == "data/external/jarvis_dft_3d.json" {
        // This lane uses a dedicated JARVIS scientific checker below.
        return Ok(());
    }
    let raw = std::fs::read(path).map_err(|err| format!("read failed {path}: {err}"))?;
    if raw.is_empty() {
        return Err(format!("numeric signal contract got empty file: {path}"));
    }
    if path.ends_with(".json") {
        let value: serde_json::Value = serde_json::from_slice(&raw)
            .map_err(|err| format!("json parse failed for {path}: {err}"))?;
        if !json_contains_number(&value) {
            return Err(format!("json contains no numeric values: {path}"));
        }
        return Ok(());
    }
    if path.ends_with(".toml") {
        let text = String::from_utf8_lossy(&raw);
        let value: toml::Value = toml::from_str(text.as_ref())
            .map_err(|err| format!("toml parse failed for {path}: {err}"))?;
        if !toml_contains_number(&value) {
            return Err(format!("toml contains no numeric values: {path}"));
        }
        return Ok(());
    }
    if path.ends_with(".csv") || path.ends_with(".dat") || path.ends_with(".txt") {
        let text = String::from_utf8_lossy(&raw);
        if !text_contains_number(text.as_ref()) {
            return Err(format!("text/csv file contains no numeric values: {path}"));
        }
    }
    Ok(())
}

fn json_contains_number(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Number(_) => true,
        serde_json::Value::Array(items) => items.iter().any(json_contains_number),
        serde_json::Value::Object(map) => map.values().any(json_contains_number),
        _ => false,
    }
}

fn toml_contains_number(value: &toml::Value) -> bool {
    match value {
        toml::Value::Integer(_) | toml::Value::Float(_) => true,
        toml::Value::Array(items) => items.iter().any(toml_contains_number),
        toml::Value::Table(map) => map.values().any(toml_contains_number),
        _ => false,
    }
}

fn text_contains_number(text: &str) -> bool {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .flat_map(|line| line.split([',', ';', '\t', ' ']))
        .any(|token| token.parse::<f64>().is_ok())
}

fn validate_blocked_manifest_contract(
    path: &str,
    rule: &gororoba_cli::data_governance::ExternalSourceRule,
) -> Result<(), String> {
    if !rule.is_blocked() {
        return Err(format!(
            "blocked_manifest_only used by non-blocked source {} ({path})",
            rule.id
        ));
    }
    if rule.manual_manifest_refs.is_empty() {
        return Err(format!(
            "source {} has blocked_manifest_only but no manual_manifest_refs",
            rule.id
        ));
    }
    for manifest in &rule.manual_manifest_refs {
        if !PathBuf::from(manifest).exists() {
            return Err(format!(
                "source {} references missing manual manifest {}",
                rule.id, manifest
            ));
        }
    }
    if rule.blocked_action_plan.is_empty() {
        return Err(format!(
            "source {} has blocked_manifest_only but no blocked_action_plan",
            rule.id
        ));
    }
    for plan_ref in &rule.blocked_action_plan {
        let path_text = plan_ref.split('#').next().unwrap_or(plan_ref).trim();
        if path_text.is_empty() {
            return Err(format!(
                "source {} has empty blocked_action_plan reference",
                rule.id
            ));
        }
        if !PathBuf::from(path_text).exists() {
            return Err(format!(
                "source {} references missing blocked_action_plan {}",
                rule.id, plan_ref
            ));
        }
    }
    Ok(())
}

fn has_registered_scientific_checker(path: &str) -> bool {
    matches!(
        path,
        "data/external/chime_frb_cat1.csv"
            | "data/external/chime_frb_cat2.csv"
            | "data/external/atnf_pulsars.csv"
            | "data/external/sdss_dr18_quasars.csv"
            | "data/external/gaia_dr3_nearby.csv"
            | "data/external/fermi_gbm_grbs.csv"
            | "data/external/PantheonPlusSH0ES.dat"
            | "data/external/union3_chain_1.txt"
            | "data/external/hip_main.dat"
            | "data/external/landsat_c2l2_sr_sample.json"
            | "data/external/sorce_tsi_daily.csv"
            | "data/external/tsis1_tsi_daily.csv"
            | "data/external/nanograv_15yr_freespectrum.csv"
            | "data/external/jpl_planets_2020_2030.csv"
            | "data/external/swarm_magnetic_sample.csv"
            | "data/external/jarvis_dft_3d.json"
            | "data/external/GWTC-3_confident.csv"
            | "data/external/mcgill_magnetars.csv"
    )
}

fn check_chime_rows(rows: &[data_core::catalogs::chime::FrbEvent]) -> Result<(), String> {
    let mut has_positive_dm = false;
    for row in rows {
        if row.ra.is_finite() && !(0.0..=360.0).contains(&row.ra) {
            return Err(format!("ra out of range [0,360]: {}", row.ra));
        }
        if row.dec.is_finite() && !(-90.0..=90.0).contains(&row.dec) {
            return Err(format!("dec out of range [-90,90]: {}", row.dec));
        }
        if row.bonsai_dm.is_finite() {
            if row.bonsai_dm <= 0.0 {
                return Err(format!("bonsai_dm must be > 0, got {}", row.bonsai_dm));
            }
            has_positive_dm = true;
        }
    }
    if !has_positive_dm {
        return Err("no finite positive bonsai_dm values found".to_string());
    }
    Ok(())
}

fn check_atnf_rows(rows: &[data_core::catalogs::atnf::Pulsar]) -> Result<(), String> {
    let mut has_positive_period = false;
    for row in rows {
        if row.ra.is_finite() && !(0.0..=360.0).contains(&row.ra) {
            return Err(format!("ra out of range [0,360]: {}", row.ra));
        }
        if row.dec.is_finite() && !(-90.0..=90.0).contains(&row.dec) {
            return Err(format!("dec out of range [-90,90]: {}", row.dec));
        }
        if row.p0.is_finite() {
            if row.p0 <= 0.0 {
                return Err(format!("spin period p0 must be > 0, got {}", row.p0));
            }
            has_positive_period = true;
        }
    }
    if !has_positive_period {
        return Err("no finite positive spin periods found".to_string());
    }
    Ok(())
}

fn check_sdss_rows(rows: &[data_core::catalogs::sdss::SdssQuasar]) -> Result<(), String> {
    let mut has_finite_redshift = false;
    for row in rows {
        if row.ra.is_finite() && !(0.0..=360.0).contains(&row.ra) {
            return Err(format!("ra out of range [0,360]: {}", row.ra));
        }
        if row.dec.is_finite() && !(-90.0..=90.0).contains(&row.dec) {
            return Err(format!("dec out of range [-90,90]: {}", row.dec));
        }
        if row.z.is_finite() {
            if row.z < 0.0 {
                return Err(format!("redshift must be >= 0, got {}", row.z));
            }
            has_finite_redshift = true;
        }
    }
    if !has_finite_redshift {
        return Err("no finite redshift values found".to_string());
    }
    Ok(())
}

fn check_gaia_rows(rows: &[data_core::catalogs::gaia::GaiaSource]) -> Result<(), String> {
    let mut has_positive_parallax = false;
    for row in rows {
        if row.ra.is_finite() && !(0.0..=360.0).contains(&row.ra) {
            return Err(format!("ra out of range [0,360]: {}", row.ra));
        }
        if row.dec.is_finite() && !(-90.0..=90.0).contains(&row.dec) {
            return Err(format!("dec out of range [-90,90]: {}", row.dec));
        }
        if row.parallax.is_finite() {
            if row.parallax <= 0.0 {
                return Err(format!("parallax must be > 0, got {}", row.parallax));
            }
            has_positive_parallax = true;
        }
    }
    if !has_positive_parallax {
        return Err("no finite positive parallax values found".to_string());
    }
    Ok(())
}

fn check_fermi_rows(rows: &[data_core::catalogs::fermi_gbm::GrbEvent]) -> Result<(), String> {
    let mut has_positive_t90 = false;
    for row in rows {
        if row.t90.is_finite() {
            if row.t90 <= 0.0 {
                return Err(format!("t90 must be > 0, got {}", row.t90));
            }
            has_positive_t90 = true;
        }
    }
    if !has_positive_t90 {
        return Err("no finite positive t90 values found".to_string());
    }
    Ok(())
}

fn check_pantheon_rows(rows: &[data_core::catalogs::pantheon::Supernova]) -> Result<(), String> {
    let mut has_finite_distance_modulus = false;
    for row in rows {
        if row.z_cmb.is_finite() && row.z_cmb < 0.0 {
            return Err(format!("z_cmb must be >= 0, got {}", row.z_cmb));
        }
        if row.mu.is_finite() {
            has_finite_distance_modulus = true;
        }
    }
    if !has_finite_distance_modulus {
        return Err("no finite distance modulus values found".to_string());
    }
    Ok(())
}

fn check_union3_rows(rows: &[data_core::catalogs::union3::Union3ChainRow]) -> Result<(), String> {
    for row in rows {
        if !row.weight.is_finite() || row.weight <= 0.0 {
            return Err(format!(
                "chain weight must be finite and > 0, got {}",
                row.weight
            ));
        }
        if !row.minus_log_posterior.is_finite() {
            return Err("minus_log_posterior must be finite".to_string());
        }
    }
    Ok(())
}

fn check_sorce_rows(rows: &[data_core::catalogs::sorce::SorceMeasurement]) -> Result<(), String> {
    for row in rows {
        if !row.tsi.is_finite() || row.tsi <= 0.0 {
            return Err(format!("TSI must be finite and > 0, got {}", row.tsi));
        }
        if !row.jd.is_finite() {
            return Err("SORCE JD must be finite".to_string());
        }
    }
    Ok(())
}

fn check_tsi_rows(rows: &[data_core::catalogs::tsi::TsiMeasurement]) -> Result<(), String> {
    let mut has_positive = false;
    for row in rows {
        if !row.tsi.is_finite() {
            return Err("TSIS TSI must be finite".to_string());
        }
        if row.tsi < 0.0 {
            return Err(format!("TSIS TSI must be >= 0, got {}", row.tsi));
        }
        if row.tsi > 0.0 {
            has_positive = true;
        }
        if !row.jd.is_finite() {
            return Err("TSIS JD must be finite".to_string());
        }
    }
    if !has_positive {
        return Err("TSIS has no positive irradiance samples".to_string());
    }
    Ok(())
}

fn check_nanograv_rows(
    rows: &[data_core::catalogs::nanograv::FreeSpectrumPoint],
) -> Result<(), String> {
    for row in rows {
        if !row.frequency.is_finite() || row.frequency <= 0.0 {
            return Err(format!(
                "NANOGrav frequency must be finite and > 0, got {}",
                row.frequency
            ));
        }
        if row.log10_rho_lo.is_finite()
            && row.log10_rho_hi.is_finite()
            && row.log10_rho.is_finite()
            && !(row.log10_rho_lo <= row.log10_rho && row.log10_rho <= row.log10_rho_hi)
        {
            return Err("NANOGrav interval ordering violated: lo <= median <= hi".to_string());
        }
    }
    Ok(())
}

fn check_jpl_rows(
    rows: &[data_core::geophysical::jpl_ephemeris::EphemerisPoint],
) -> Result<(), String> {
    for row in rows {
        if !row.jd.is_finite() {
            return Err("JPL JD must be finite".to_string());
        }
        if row.ra.is_finite() && !(0.0..=360.0).contains(&row.ra) {
            return Err(format!("JPL ra out of range [0,360]: {}", row.ra));
        }
        if row.dec.is_finite() && !(-90.0..=90.0).contains(&row.dec) {
            return Err(format!("JPL dec out of range [-90,90]: {}", row.dec));
        }
        if !row.delta.is_finite() || row.delta <= 0.0 {
            return Err(format!(
                "JPL delta must be finite and > 0, got {}",
                row.delta
            ));
        }
    }
    Ok(())
}

fn check_swarm_rows(rows: &[data_core::geophysical::swarm::SwarmRecord]) -> Result<(), String> {
    for row in rows {
        if !(-90.0..=90.0).contains(&row.latitude) {
            return Err(format!(
                "Swarm latitude out of range [-90,90]: {}",
                row.latitude
            ));
        }
        if !(-180.0..=180.0).contains(&row.longitude) {
            return Err(format!(
                "Swarm longitude out of range [-180,180]: {}",
                row.longitude
            ));
        }
        if !row.radius.is_finite() || row.radius <= 0.0 {
            return Err(format!(
                "Swarm radius must be finite and > 0, got {}",
                row.radius
            ));
        }
        if !row.f_total.is_finite() || row.f_total <= 0.0 {
            return Err(format!(
                "Swarm total field must be finite and > 0, got {}",
                row.f_total
            ));
        }
    }
    Ok(())
}

fn check_jarvis_rows(rows: &[data_core::catalogs::jarvis::JarvisMaterial]) -> Result<(), String> {
    let mut invalid = 0usize;
    for row in rows {
        if row.jid.trim().is_empty() {
            invalid += 1;
            continue;
        }
        if row.formula.trim().is_empty() && row.elements.is_empty() {
            invalid += 1;
        }
    }
    if rows.is_empty() {
        return Err("JARVIS dataset is empty".to_string());
    }
    // Require at least 95% rows to have basic compositional metadata.
    let invalid_ratio = invalid as f64 / rows.len() as f64;
    if invalid_ratio > 0.05 {
        return Err(format!(
            "JARVIS invalid metadata ratio too high: {:.2}% ({} / {})",
            invalid_ratio * 100.0,
            invalid,
            rows.len()
        ));
    }
    Ok(())
}

fn check_gwtc_rows(rows: &[data_core::catalogs::gwtc::GwEvent]) -> Result<(), String> {
    for row in rows {
        if row.mass_1_source <= 0.0 || row.mass_2_source <= 0.0 {
            return Err(format!(
                "GWTC event {} has non-positive source masses",
                row.id
            ));
        }
        if row.luminosity_distance <= 0.0 {
            return Err(format!(
                "GWTC event {} has non-positive luminosity distance",
                row.id
            ));
        }
        if row.p_astro.is_finite() && !(0.0..=1.0).contains(&row.p_astro) {
            return Err(format!(
                "GWTC event {} has p_astro outside [0,1]: {}",
                row.id, row.p_astro
            ));
        }
    }
    Ok(())
}

fn check_mcgill_rows(rows: &[data_core::catalogs::mcgill::Magnetar]) -> Result<(), String> {
    for row in rows {
        if row.ra.is_finite() && !(0.0..=360.0).contains(&row.ra) {
            return Err(format!("McGill ra out of range [0,360]: {}", row.ra));
        }
        if row.dec.is_finite() && !(-90.0..=90.0).contains(&row.dec) {
            return Err(format!("McGill dec out of range [-90,90]: {}", row.dec));
        }
        if row.period.is_finite() && row.period <= 0.0 {
            return Err(format!(
                "McGill period must be positive when finite, got {}",
                row.period
            ));
        }
    }
    Ok(())
}

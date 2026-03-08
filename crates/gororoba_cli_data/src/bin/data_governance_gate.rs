use anyhow::Result;
use clap::{ArgAction, Parser};
use glob::Pattern;
use gororoba_cli::data_governance::{
    AuditInputs, DEFAULT_ARTIFACTS_MANIFEST_PATH, DEFAULT_EVIDENCE_MANIFEST_PATH,
    DEFAULT_EXPERIMENTS_PATH, DEFAULT_EXTERNAL_PROVENANCE_PATH, DEFAULT_EXTERNAL_SOURCES_PATH,
    DEFAULT_GENERATED_PATTERNS_PATH, DEFAULT_GOVERNANCE_PATH, DEFAULT_SEMANTIC_VALIDATORS_PATH,
    audit_data, blocked_source_deadline_issues, external_source_contract_issues, git_ignored_paths,
    git_tracked_paths, list_required_manifests, load_artifacts_manifest, load_data_governance,
    load_evidence_manifest, load_experiment_output_patterns, load_external_hashes,
    load_external_sources, load_generated_origin_patterns, load_semantic_validators,
    missing_semantic_lane_validators, source_rule_for_path,
};
use std::{collections::BTreeSet, path::PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "data-governance-gate",
    about = "Fail-closed gate for data origin, naming, and gitignore governance"
)]
struct Args {
    #[arg(long, default_value = "data")]
    root: PathBuf,
    #[arg(long, default_value = DEFAULT_GOVERNANCE_PATH)]
    governance: PathBuf,
    #[arg(long, default_value = DEFAULT_EXPERIMENTS_PATH)]
    experiments: PathBuf,
    #[arg(long, default_value = DEFAULT_GENERATED_PATTERNS_PATH)]
    generated_patterns: PathBuf,
    #[arg(long, default_value = DEFAULT_EXTERNAL_PROVENANCE_PATH)]
    external_provenance: PathBuf,
    #[arg(long, default_value = DEFAULT_ARTIFACTS_MANIFEST_PATH)]
    artifacts_manifest: PathBuf,
    #[arg(long, default_value = DEFAULT_EVIDENCE_MANIFEST_PATH)]
    evidence_manifest: PathBuf,
    #[arg(long, default_value = DEFAULT_EXTERNAL_SOURCES_PATH)]
    external_sources: PathBuf,
    #[arg(long, default_value = DEFAULT_SEMANTIC_VALIDATORS_PATH)]
    semantic_validators: PathBuf,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    enforce_gitignore: bool,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    enforce_naming: bool,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    enforce_origin: bool,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    enforce_semantic: bool,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    enforce_blocked_deadlines: bool,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    enforce_unclassified: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let governance = load_data_governance(&args.governance)?;
    let external_hashes = load_external_hashes(&args.external_provenance)?;
    let artifacts_manifest = load_artifacts_manifest(&args.artifacts_manifest)?;
    let evidence_manifest = load_evidence_manifest(&args.evidence_manifest)?;
    let output_patterns = load_experiment_output_patterns(&args.experiments)?;
    let generated_origin_patterns = load_generated_origin_patterns(&args.generated_patterns)?;
    let manifests = list_required_manifests(&governance);
    let lanes = governance.lane.clone();

    let report = audit_data(&AuditInputs {
        governance: governance.clone(),
        external_hashes,
        artifacts_manifest,
        evidence_manifest,
        experiment_output_patterns: output_patterns,
        generated_origin_patterns,
        root: args.root,
    })?;

    let mut failures = Vec::new();
    if report.strict_unknown_count > 0 {
        failures.push(format!(
            "strict unknown origins: {}",
            report.strict_unknown_count
        ));
    }
    if args.enforce_unclassified {
        let unclassified = report.lane_counts.get("unclassified").copied().unwrap_or(0);
        if unclassified > 0 {
            failures.push(format!("unclassified lane files: {unclassified}"));
            for path in report
                .records
                .iter()
                .filter(|record| record.lane_id == "unclassified")
                .map(|record| record.path.as_str())
                .take(25)
            {
                failures.push(format!("unclassified path: {path}"));
            }
        }
    }

    for (lane, files) in manifests {
        for file in files {
            let path = PathBuf::from(&file);
            if !path.exists() {
                failures.push(format!("lane {lane}: missing required manifest {file}"));
            }
        }
    }

    if args.enforce_naming && !report.naming_violations.is_empty() {
        failures.push(format!(
            "naming violations in governed lanes: {}",
            report.naming_violations.len()
        ));
        for item in report.naming_violations.iter().take(25) {
            failures.push(format!("naming violation: {item}"));
        }
    }

    if args.enforce_gitignore {
        let allowlist: BTreeSet<String> = lanes
            .iter()
            .flat_map(|lane| lane.gitignore_allowlist.iter().cloned())
            .collect();
        let mut auto_ignored = BTreeSet::new();
        let paths_to_check: Vec<String> = report
            .records
            .iter()
            .filter_map(|record| {
                let lane = lanes.iter().find(|lane| lane.id == record.lane_id)?;
                let should_require = lane.gitignore_required
                    && matches!(
                        record.origin_kind.as_str(),
                        "generated_reproducible" | "external_reproducible_fetch"
                    );
                if should_require && !allowlist.contains(&record.path) {
                    // Fast path: the external lane is blanket-ignored by `data/external/**`
                    // and we already excluded explicit allowlisted metadata above.
                    if record.path.starts_with("data/external/") {
                        auto_ignored.insert(record.path.clone());
                        None
                    } else {
                        Some(record.path.clone())
                    }
                } else {
                    None
                }
            })
            .collect();
        let tracked = git_tracked_paths(&PathBuf::from("."), &paths_to_check)?;
        let ignored = git_ignored_paths(&PathBuf::from("."), &paths_to_check)?;
        for path in paths_to_check {
            if tracked.contains(&path) {
                continue;
            }
            if !ignored.contains(&path) && !auto_ignored.contains(&path) {
                failures.push(format!("path must be gitignored but is not: {path}"));
            }
        }
    }

    for lane in &lanes {
        for pattern_text in &lane.forbidden_globs {
            let Ok(pattern) = Pattern::new(pattern_text) else {
                failures.push(format!(
                    "lane {} has invalid forbidden_glob {}",
                    lane.id, pattern_text
                ));
                continue;
            };
            for record in report
                .records
                .iter()
                .filter(|record| record.lane_id == lane.id)
            {
                if pattern.matches(&record.path) {
                    failures.push(format!(
                        "lane {} forbids path pattern {} but matched {}",
                        lane.id, pattern_text, record.path
                    ));
                }
            }
        }
    }

    if args.enforce_origin {
        let sources = load_external_sources(&args.external_sources)?;
        for issue in external_source_contract_issues(&sources) {
            failures.push(format!("external source contract: {issue}"));
        }
        let external_paths: Vec<String> = report
            .records
            .iter()
            .filter(|record| {
                record.path.starts_with("data/external/")
                    && record.path != "data/external/PROVENANCE.local.json"
                    && record.path != "data/external/README.md"
                    && record.path != "data/external/SOURCES.toml"
            })
            .map(|record| record.path.clone())
            .collect();
        for path in external_paths {
            if source_rule_for_path(&path, &sources).is_none() {
                failures.push(format!("external path missing source rule: {path}"));
            }
        }
        if args.enforce_blocked_deadlines {
            let now = chrono::Utc::now();
            for issue in blocked_source_deadline_issues(&sources, now) {
                failures.push(format!("blocked source policy: {issue}"));
            }
        }
    }

    if args.enforce_semantic {
        let semantic_validators = load_semantic_validators(&args.semantic_validators)?;
        for lane in missing_semantic_lane_validators(&governance, &semantic_validators) {
            failures.push(format!("lane missing semantic validator coverage: {lane}"));
        }
    }

    if failures.is_empty() {
        println!("OK: data governance gate passed");
        println!("  total_files={}", report.total_files);
        println!("  strict_unknown_count={}", report.strict_unknown_count);
        println!("  naming_violations={}", report.naming_violations.len());
        return Ok(());
    }

    eprintln!(
        "ERROR: data governance gate failed with {} issue(s)",
        failures.len()
    );
    for failure in failures {
        eprintln!("  - {failure}");
    }
    anyhow::bail!("data governance gate failed");
}

use anyhow::{Context, Result};
use clap::Parser;
use gororoba_cli::data_governance::{
    AuditInputs, DEFAULT_ARTIFACTS_MANIFEST_PATH, DEFAULT_EVIDENCE_MANIFEST_PATH,
    DEFAULT_EXPERIMENTS_PATH, DEFAULT_EXTERNAL_PROVENANCE_PATH, DEFAULT_GENERATED_PATTERNS_PATH,
    DEFAULT_GOVERNANCE_PATH, audit_data, load_artifacts_manifest, load_data_governance,
    load_evidence_manifest, load_experiment_output_patterns, load_external_hashes,
    load_generated_origin_patterns,
};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "data-origin-audit",
    about = "Audit data/* origin coverage and identify unknown-origin lacunae"
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
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long)]
    fail_on_strict_unknown: bool,
    #[arg(long)]
    fail_on_unclassified: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let governance = load_data_governance(&args.governance)?;
    let external_hashes = load_external_hashes(&args.external_provenance)?;
    let artifacts_manifest = load_artifacts_manifest(&args.artifacts_manifest)?;
    let evidence_manifest = load_evidence_manifest(&args.evidence_manifest)?;
    let output_patterns = load_experiment_output_patterns(&args.experiments)?;
    let generated_origin_patterns = load_generated_origin_patterns(&args.generated_patterns)?;

    let report = audit_data(&AuditInputs {
        governance,
        external_hashes,
        artifacts_manifest,
        evidence_manifest,
        experiment_output_patterns: output_patterns,
        generated_origin_patterns,
        root: args.root,
    })?;

    println!("DATA_ORIGIN_AUDIT");
    println!("  total_files={}", report.total_files);
    println!("  strict_unknown_count={}", report.strict_unknown_count);
    println!("  unknown_count={}", report.unknown_paths.len());
    println!(
        "  unclassified_count={}",
        report.lane_counts.get("unclassified").copied().unwrap_or(0)
    );
    for (kind, count) in &report.origin_counts {
        println!("  origin[{kind}]={count}");
    }
    if !report.naming_violations.is_empty() {
        println!("  naming_violations={}", report.naming_violations.len());
    }
    if !report.unknown_paths.is_empty() {
        println!("UNKNOWN_PATHS (first 100)");
        for item in report.unknown_paths.iter().take(100) {
            println!("  {item}");
        }
    }

    if let Some(out) = args.out {
        write_report(&out, &report)?;
        println!("WROTE {}", out.display());
    }

    if args.fail_on_strict_unknown && report.strict_unknown_count > 0 {
        anyhow::bail!(
            "strict unknown origins detected: {}",
            report.strict_unknown_count
        );
    }
    if args.fail_on_unclassified {
        let unclassified = report.lane_counts.get("unclassified").copied().unwrap_or(0);
        if unclassified > 0 {
            anyhow::bail!("unclassified data files detected: {unclassified}");
        }
    }
    Ok(())
}

fn write_report(path: &PathBuf, report: &gororoba_cli::data_governance::AuditReport) -> Result<()> {
    let serialized = if path.extension().and_then(|s| s.to_str()) == Some("json") {
        serde_json::to_string_pretty(report).context("serialize JSON report")?
    } else {
        toml::to_string_pretty(report).context("serialize TOML report")?
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create output directory {}", parent.display()))?;
    }
    std::fs::write(path, serialized + "\n")
        .with_context(|| format!("write report {}", path.display()))?;
    Ok(())
}

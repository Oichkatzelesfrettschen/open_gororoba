use anyhow::{Context, Result};
use clap::{ArgAction, Parser};
use gororoba_cli::data_governance::{
    DEFAULT_EXTERNAL_PROVENANCE_PATH, DEFAULT_EXTERNAL_SOURCES_PATH, collect_files_under,
    load_external_hashes, load_external_sources, parse_deadline_utc, source_rule_for_path,
};
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "external-blocked-burndown",
    about = "Quantify blocked external-source debt and enforce burn-down thresholds"
)]
struct Args {
    #[arg(long, default_value = "data/external")]
    root: PathBuf,
    #[arg(long, default_value = DEFAULT_EXTERNAL_SOURCES_PATH)]
    sources: PathBuf,
    #[arg(long, default_value = DEFAULT_EXTERNAL_PROVENANCE_PATH)]
    provenance: PathBuf,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long, default_value_t = 5)]
    max_sample_paths: usize,
    #[arg(long)]
    max_blocked_files: Option<usize>,
    #[arg(long)]
    max_blocked_sources: Option<usize>,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    fail_on_overdue: bool,
    #[arg(long, default_value_t = false, action = ArgAction::Set)]
    fail_on_missing_deadlines: bool,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    fail_on_missing_action_plan: bool,
}

#[derive(Debug, Serialize)]
struct BlockedSourceBurndown {
    source_id: String,
    access_class: String,
    path_glob: String,
    attempt_deadline_utc: String,
    resolution_deadline_utc: String,
    is_overdue: bool,
    missing_deadlines: bool,
    missing_action_plan: bool,
    missing_action_plan_refs: usize,
    file_count: usize,
    total_size_bytes: u64,
    sample_paths: Vec<String>,
    blocker_note: String,
    evidence_refs: Vec<String>,
    blocked_action_plan: Vec<String>,
}

#[derive(Debug, Serialize)]
struct BlockedBurndownReport {
    generated_at_utc: String,
    total_external_files: usize,
    blocked_files: usize,
    blocked_sources: usize,
    blocked_overdue_sources: usize,
    blocked_missing_deadline_sources: usize,
    blocked_missing_action_plan_sources: usize,
    blocked_missing_action_plan_ref_count: usize,
    blocked_total_size_bytes: u64,
    top_blocked_sources: Vec<BlockedSourceBurndown>,
}

#[derive(Default)]
struct MutableSourceStats {
    access_class: String,
    path_glob: String,
    attempt_deadline_utc: String,
    resolution_deadline_utc: String,
    is_overdue: bool,
    missing_deadlines: bool,
    missing_action_plan: bool,
    missing_action_plan_refs: usize,
    file_count: usize,
    total_size_bytes: u64,
    sample_paths: Vec<String>,
    blocker_note: String,
    evidence_refs: Vec<String>,
    blocked_action_plan: Vec<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let now = chrono::Utc::now();
    let sources = load_external_sources(&args.sources)?;
    let hashes = load_external_hashes(&args.provenance)?;

    let mut files = collect_files_under(&args.root)?;
    files.retain(|path| {
        path != "data/external/README.md"
            && path != "data/external/SOURCES.toml"
            && path != "data/external/PROVENANCE.local.json"
    });
    files.sort();

    let mut by_source: BTreeMap<String, MutableSourceStats> = BTreeMap::new();
    for path in &files {
        let Some(rule) = source_rule_for_path(path, &sources) else {
            continue;
        };
        if !rule.is_blocked() {
            continue;
        }
        let stats = by_source.entry(rule.id.clone()).or_insert_with(|| {
            let missing_deadlines = rule.attempt_deadline_utc.trim().is_empty()
                || rule.resolution_deadline_utc.trim().is_empty();
            let missing_action_plan = rule.blocked_action_plan.is_empty();
            let missing_action_plan_refs = rule
                .blocked_action_plan
                .iter()
                .filter(|item| !plan_ref_exists(item))
                .count();
            let is_overdue = parse_deadline_utc(&rule.resolution_deadline_utc)
                .map(|deadline| deadline < now)
                .unwrap_or(false);
            MutableSourceStats {
                access_class: rule.access_class.clone(),
                path_glob: rule.path_glob.clone(),
                attempt_deadline_utc: rule.attempt_deadline_utc.clone(),
                resolution_deadline_utc: rule.resolution_deadline_utc.clone(),
                is_overdue,
                missing_deadlines,
                missing_action_plan,
                missing_action_plan_refs,
                file_count: 0,
                total_size_bytes: 0,
                sample_paths: Vec::new(),
                blocker_note: rule.blocker_note.clone(),
                evidence_refs: rule.evidence_refs.clone(),
                blocked_action_plan: rule.blocked_action_plan.clone(),
            }
        });
        stats.file_count += 1;
        if let Some(hash) = hashes.get(path) {
            stats.total_size_bytes += hash.size_bytes;
        }
        if stats.sample_paths.len() < args.max_sample_paths {
            stats.sample_paths.push(path.clone());
        }
    }

    let mut top_blocked_sources: Vec<BlockedSourceBurndown> = by_source
        .into_iter()
        .map(|(source_id, stats)| BlockedSourceBurndown {
            source_id,
            access_class: stats.access_class,
            path_glob: stats.path_glob,
            attempt_deadline_utc: stats.attempt_deadline_utc,
            resolution_deadline_utc: stats.resolution_deadline_utc,
            is_overdue: stats.is_overdue,
            missing_deadlines: stats.missing_deadlines,
            missing_action_plan: stats.missing_action_plan,
            missing_action_plan_refs: stats.missing_action_plan_refs,
            file_count: stats.file_count,
            total_size_bytes: stats.total_size_bytes,
            sample_paths: stats.sample_paths,
            blocker_note: stats.blocker_note,
            evidence_refs: stats.evidence_refs,
            blocked_action_plan: stats.blocked_action_plan,
        })
        .collect();
    top_blocked_sources.sort_by_key(|entry| std::cmp::Reverse(entry.file_count));

    let blocked_files = top_blocked_sources
        .iter()
        .map(|s| s.file_count)
        .sum::<usize>();
    let blocked_sources = top_blocked_sources.len();
    let blocked_overdue_sources = top_blocked_sources.iter().filter(|s| s.is_overdue).count();
    let blocked_missing_deadline_sources = top_blocked_sources
        .iter()
        .filter(|s| s.missing_deadlines)
        .count();
    let blocked_missing_action_plan_sources = top_blocked_sources
        .iter()
        .filter(|s| s.missing_action_plan)
        .count();
    let blocked_missing_action_plan_ref_count = top_blocked_sources
        .iter()
        .map(|s| s.missing_action_plan_refs)
        .sum::<usize>();
    let blocked_total_size_bytes = top_blocked_sources
        .iter()
        .map(|s| s.total_size_bytes)
        .sum::<u64>();

    let report = BlockedBurndownReport {
        generated_at_utc: now.to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
        total_external_files: files.len(),
        blocked_files,
        blocked_sources,
        blocked_overdue_sources,
        blocked_missing_deadline_sources,
        blocked_missing_action_plan_sources,
        blocked_missing_action_plan_ref_count,
        blocked_total_size_bytes,
        top_blocked_sources,
    };

    println!("EXTERNAL_BLOCKED_BURNDOWN");
    println!("  total_external_files={}", report.total_external_files);
    println!("  blocked_files={}", report.blocked_files);
    println!("  blocked_sources={}", report.blocked_sources);
    println!(
        "  blocked_overdue_sources={}",
        report.blocked_overdue_sources
    );
    println!(
        "  blocked_missing_deadline_sources={}",
        report.blocked_missing_deadline_sources
    );
    println!(
        "  blocked_missing_action_plan_sources={}",
        report.blocked_missing_action_plan_sources
    );
    println!(
        "  blocked_missing_action_plan_ref_count={}",
        report.blocked_missing_action_plan_ref_count
    );
    println!(
        "  blocked_total_size_bytes={}",
        report.blocked_total_size_bytes
    );

    for entry in report.top_blocked_sources.iter().take(10) {
        println!(
            "SOURCE {} files={} bytes={} overdue={} missing_deadlines={} glob={}",
            entry.source_id,
            entry.file_count,
            entry.total_size_bytes,
            entry.is_overdue,
            entry.missing_deadlines,
            entry.path_glob
        );
    }

    if let Some(out) = &args.out {
        let body = if out.extension().and_then(|s| s.to_str()) == Some("json") {
            serde_json::to_string_pretty(&report).context("serialize JSON report")?
        } else {
            toml::to_string_pretty(&report).context("serialize TOML report")?
        };
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create output directory {}", parent.display()))?;
        }
        std::fs::write(out, body + "\n").with_context(|| format!("write {}", out.display()))?;
        println!("WROTE {}", out.display());
    }

    let mut failures = Vec::new();
    if args.fail_on_overdue && report.blocked_overdue_sources > 0 {
        failures.push(format!(
            "{} blocked source(s) are overdue",
            report.blocked_overdue_sources
        ));
    }
    if args.fail_on_missing_deadlines && report.blocked_missing_deadline_sources > 0 {
        failures.push(format!(
            "{} blocked source(s) are missing deadlines",
            report.blocked_missing_deadline_sources
        ));
    }
    if args.fail_on_missing_action_plan && report.blocked_missing_action_plan_sources > 0 {
        failures.push(format!(
            "{} blocked source(s) are missing blocked_action_plan",
            report.blocked_missing_action_plan_sources
        ));
    }
    if args.fail_on_missing_action_plan && report.blocked_missing_action_plan_ref_count > 0 {
        failures.push(format!(
            "{} blocked_action_plan reference(s) are missing on disk",
            report.blocked_missing_action_plan_ref_count
        ));
    }
    if let Some(max) = args.max_blocked_files
        && report.blocked_files > max
    {
        failures.push(format!(
            "blocked_files={} exceeds configured max {}",
            report.blocked_files, max
        ));
    }
    if let Some(max) = args.max_blocked_sources
        && report.blocked_sources > max
    {
        failures.push(format!(
            "blocked_sources={} exceeds configured max {}",
            report.blocked_sources, max
        ));
    }

    if !failures.is_empty() {
        for failure in failures {
            eprintln!("ERROR: {failure}");
        }
        anyhow::bail!("external blocked burndown failed");
    }
    Ok(())
}

fn plan_ref_exists(reference: &str) -> bool {
    let path_text = reference.split('#').next().unwrap_or(reference).trim();
    if path_text.is_empty() {
        return false;
    }
    std::path::Path::new(path_text).exists()
}

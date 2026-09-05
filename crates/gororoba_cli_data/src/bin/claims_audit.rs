//! Claims audit report generator.
//!
//! Replaces multiple Python analysis scripts with a single Rust binary.
//! Generates combined or individual audit reports, plus a falsification
//! campaign summary and machine-readable manifest for empirical-core claims.
//!
//! Usage:
//!   claims-audit                                  # all legacy reports to stdout
//!   claims-audit --report id                      # just ID inventory
//!   claims-audit --report status                  # just status inventory
//!   claims-audit --report falsification-campaign  # campaign markdown summary
//!   claims-audit --campaign-manifest-out /tmp/campaign.toml
//!   claims-audit --out reports/audit.md           # write report to file

use chrono::Utc;
use clap::Parser;
use provenance_store::{ControlPlaneCompatKind, ProvenanceStore};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
    process,
};
use toml::Value;

use gororoba_cli::claims::{
    audit,
    parser::{ClaimRow, parse_claim_rows},
};

const EMPIRICAL_CORE_STATUSES: &[&str] = &[
    "Refuted",
    "Closed/Refuted",
    "Closed/Negative-Result",
    "Closed/Methodology-Insufficient",
];
const CLOSED_TASK_STATUS_TOKENS: &[&str] = &["DONE", "REFUTED", "DEFERRED"];
const FAMILY_ALGEBRAIC_STRUCTURAL: &str = "algebraic_structural";
const FAMILY_COSMOLOGY_METAMATERIAL: &str = "cosmology_metamaterial";
const FAMILY_PARTICLE_NUMEROLOGY: &str = "particle_numerology";
const FAMILY_SIGNAL_TRANSPORT: &str = "signal_transport";
const WAVE_REGISTERED_EXPERIMENT: &str = "wave_1_registered_experiment_hardening";
const WAVE_NEW_RUST_FALSIFIER: &str = "wave_2_new_rust_falsifier";

#[derive(Parser)]
#[command(name = "claims-audit", about = "Generate claims audit reports")]
struct Cli {
    /// Repository root directory.
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,

    /// Canonical SQLite control-plane DB used to render the live claims matrix.
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    canonical_db: PathBuf,

    /// Path to legacy claims matrix compatibility file (relative to repo root).
    #[arg(long, default_value = "docs/CLAIMS_EVIDENCE_MATRIX.md")]
    matrix: String,

    /// Path to compatibility claims TOML (relative to repo root).
    #[arg(long, default_value = "registry/claims.toml")]
    claims_toml: String,

    /// Path to compatibility experiments TOML (relative to repo root).
    #[arg(long, default_value = "registry/experiments.toml")]
    experiments_toml: String,

    /// Path to claims tasks registry (relative to repo root).
    #[arg(long, default_value = "registry/claims_tasks.toml")]
    claims_tasks_toml: String,

    /// Path to claim tickets registry (relative to repo root).
    #[arg(long, default_value = "registry/claim_tickets.toml")]
    claim_tickets_toml: String,

    /// Path to compatibility insights TOML (relative to repo root).
    #[arg(long, default_value = "registry/insights.toml")]
    insights_toml: String,

    /// Specific report to generate (default: all).
    #[arg(long, value_parser = ["id", "status", "staleness", "contradictions", "bold-tokens", "priority", "falsification-campaign", "all"])]
    report: Option<String>,

    /// Staleness threshold date (claims verified before this are stale).
    #[arg(long, default_value = "2025-06-01")]
    stale_before: String,

    /// Output file for markdown/text report (default: stdout).
    #[arg(long)]
    out: Option<PathBuf>,

    /// Optional TOML output for the falsification campaign manifest.
    #[arg(long)]
    campaign_manifest_out: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct ClaimsToml {
    #[serde(default, rename = "claim")]
    claims: Vec<ClaimTomlRow>,
}

#[derive(Debug, Clone, Deserialize)]
struct ClaimTomlRow {
    id: String,
    statement: String,
    status: String,
    #[serde(default)]
    last_verified: String,
    #[serde(default)]
    where_stated: String,
}

#[derive(Debug, Deserialize)]
struct ExperimentsToml {
    #[serde(default, rename = "experiment")]
    experiments: Vec<ExperimentTomlRow>,
}

#[derive(Debug, Clone, Deserialize)]
struct ExperimentTomlRow {
    id: String,
    #[serde(default)]
    binary: String,
    #[serde(default)]
    deterministic: bool,
    #[serde(default)]
    gpu: bool,
    #[serde(default)]
    claims: Vec<String>,
    #[serde(default)]
    claim_refs: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ClaimsTasksToml {
    #[serde(default, rename = "task")]
    tasks: Vec<ClaimTaskTomlRow>,
}

#[derive(Debug, Clone, Deserialize)]
struct ClaimTaskTomlRow {
    id: String,
    claim_id: String,
    status_token: String,
}

#[derive(Debug, Deserialize)]
struct ClaimTicketsToml {
    #[serde(default, rename = "ticket")]
    tickets: Vec<ClaimTicketTomlRow>,
}

#[derive(Debug, Clone, Deserialize)]
struct ClaimTicketTomlRow {
    id: String,
    claim_range_start: u32,
    claim_range_end: u32,
    #[serde(default)]
    claims_referenced: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct InsightsToml {
    #[serde(default, rename = "insight")]
    insights: Vec<InsightTomlRow>,
}

#[derive(Debug, Clone, Deserialize)]
struct InsightTomlRow {
    id: String,
    #[serde(default)]
    claims: Vec<String>,
    #[serde(default)]
    related_claims: Vec<String>,
}

#[derive(Debug, Serialize)]
struct FalsificationCampaignManifest {
    generated_at_utc: String,
    source_claims_toml: String,
    source_experiments_toml: String,
    source_claims_tasks_toml: String,
    source_claim_tickets_toml: String,
    source_insights_toml: String,
    all_claim_count: usize,
    all_claim_lane_counts: BTreeMap<String, usize>,
    all_claim_ids_by_lane: BTreeMap<String, Vec<String>>,
    superseded_claim_ids: Vec<String>,
    empirical_core_statuses: Vec<String>,
    target_claim_count: usize,
    status_counts: BTreeMap<String, usize>,
    family_counts: BTreeMap<String, usize>,
    wave_counts: BTreeMap<String, usize>,
    coverage: FalsificationCoverage,
    claims: Vec<FalsificationCampaignClaimRow>,
}

#[derive(Debug, Serialize)]
struct FalsificationCoverage {
    with_registered_experiments: usize,
    without_registered_experiments: usize,
    with_gpu_experiments: usize,
    with_tasks: usize,
    with_open_tasks: usize,
    with_ticket_coverage: usize,
    with_insight_coverage: usize,
    claims_needing_new_experiment: Vec<String>,
    claims_missing_task_coverage: Vec<String>,
    claims_missing_ticket_coverage: Vec<String>,
    claims_missing_insight_coverage: Vec<String>,
    claims_missing_all_linkage: Vec<String>,
    claims_missing_experiment_ticket_and_insight: Vec<String>,
}

#[derive(Debug, Serialize)]
struct FalsificationCampaignClaimRow {
    claim_id: String,
    status: String,
    family: String,
    wave: String,
    last_verified: String,
    statement: String,
    where_stated: String,
    experiment_ids: Vec<String>,
    experiment_binaries: Vec<String>,
    registered_gpu_experiment: bool,
    task_ids: Vec<String>,
    open_task_ids: Vec<String>,
    ticket_ids: Vec<String>,
    insight_ids: Vec<String>,
    requires_new_experiment: bool,
    reproducibility_floor: String,
    canonical_falsifier: String,
    independent_countercheck: String,
    acceptance_bar: Vec<String>,
}

struct CampaignRegistryInputs<'a> {
    source_claims_toml: &'a str,
    source_experiments_toml: &'a str,
    source_claims_tasks_toml: &'a str,
    source_claim_tickets_toml: &'a str,
    source_insights_toml: &'a str,
    claims: &'a [ClaimTomlRow],
    experiments: &'a [ExperimentTomlRow],
    tasks: &'a [ClaimTaskTomlRow],
    tickets: &'a [ClaimTicketTomlRow],
    insights: &'a [InsightTomlRow],
}

fn main() {
    let cli = Cli::parse();
    let repo_root = cli.repo_root.canonicalize().unwrap_or_else(|e| {
        eprintln!("ERROR: Cannot resolve repo root {:?}: {e}", cli.repo_root);
        process::exit(2);
    });

    let matrix_path = repo_root.join(&cli.matrix);
    let report_type = cli.report.as_deref().unwrap_or("all");

    if report_type != "falsification-campaign" && cli.campaign_manifest_out.is_some() {
        eprintln!("ERROR: --campaign-manifest-out requires --report falsification-campaign");
        process::exit(2);
    }

    let output = match report_type {
        "all" => {
            let (claims, matrix_label) = load_claim_rows(&repo_root, &cli, &matrix_path);
            let docs_dir = repo_root.join("docs");
            let doc_corpus = audit::collect_doc_corpus(&docs_dir);
            render_all_audits(&claims, &matrix_label, &doc_corpus, &cli.stale_before)
        }
        "id" => {
            let (claims, matrix_label) = load_claim_rows(&repo_root, &cli, &matrix_path);
            let inv = audit::id_inventory(&claims);
            audit::render_id_inventory(&inv, &matrix_label)
        }
        "status" => {
            let (claims, matrix_label) = load_claim_rows(&repo_root, &cli, &matrix_path);
            let inv = audit::status_inventory(&claims);
            audit::render_status_inventory(&inv, &matrix_label)
        }
        "staleness" => {
            let (claims, matrix_label) = load_claim_rows(&repo_root, &cli, &matrix_path);
            let report = audit::staleness_report(&claims, &cli.stale_before);
            audit::render_staleness_report(&report, &cli.stale_before, &matrix_label)
        }
        "contradictions" => {
            let (claims, matrix_label) = load_claim_rows(&repo_root, &cli, &matrix_path);
            let contras = audit::status_contradictions(&claims);
            audit::render_contradictions(&contras, &matrix_label)
        }
        "bold-tokens" => {
            let (claims, matrix_label) = load_claim_rows(&repo_root, &cli, &matrix_path);
            let tokens = audit::bold_tokens_inventory(&claims);
            audit::render_bold_tokens(&tokens, &matrix_label)
        }
        "priority" => {
            let (claims, matrix_label) = load_claim_rows(&repo_root, &cli, &matrix_path);
            let docs_dir = repo_root.join("docs");
            let doc_corpus = audit::collect_doc_corpus(&docs_dir);
            let prio = audit::priority_ranking(&claims, &doc_corpus);
            audit::render_priority_ranking(&prio, &matrix_label)
        }
        "falsification-campaign" => {
            let claims_path = repo_root.join(&cli.claims_toml);
            let experiments_path = repo_root.join(&cli.experiments_toml);
            let tasks_path = repo_root.join(&cli.claims_tasks_toml);
            let tickets_path = repo_root.join(&cli.claim_tickets_toml);
            let insights_path = repo_root.join(&cli.insights_toml);
            let claims: ClaimsToml = load_toml(&claims_path);
            let experiments: ExperimentsToml = load_toml(&experiments_path);
            let tasks: ClaimsTasksToml = load_toml(&tasks_path);
            let tickets: ClaimTicketsToml = load_toml(&tickets_path);
            let insights: InsightsToml = load_toml(&insights_path);
            let manifest = build_falsification_campaign_manifest(CampaignRegistryInputs {
                source_claims_toml: &claims_path.display().to_string(),
                source_experiments_toml: &experiments_path.display().to_string(),
                source_claims_tasks_toml: &tasks_path.display().to_string(),
                source_claim_tickets_toml: &tickets_path.display().to_string(),
                source_insights_toml: &insights_path.display().to_string(),
                claims: &claims.claims,
                experiments: &experiments.experiments,
                tasks: &tasks.tasks,
                tickets: &tickets.tickets,
                insights: &insights.insights,
            });
            if let Some(path) = &cli.campaign_manifest_out {
                write_output(
                    path,
                    &toml::to_string_pretty(&manifest).unwrap_or_else(|e| {
                        eprintln!("ERROR: Cannot serialize falsification campaign manifest: {e}");
                        process::exit(2);
                    }),
                );
            }
            render_falsification_campaign_report(&manifest)
        }
        _ => unreachable!("clap validates report type"),
    };

    match cli.out {
        Some(out_path) => write_output(&out_path, &output),
        None => print!("{output}"),
    }
}

fn load_claim_rows(
    repo_root: &std::path::Path,
    cli: &Cli,
    matrix_path: &std::path::Path,
) -> (Vec<ClaimRow>, String) {
    let canonical_db_path = repo_root.join(&cli.canonical_db);
    if canonical_db_path.exists() {
        let mut store = ProvenanceStore::open(&canonical_db_path).unwrap_or_else(|e| {
            eprintln!(
                "ERROR: Cannot open canonical DB {}: {e}",
                canonical_db_path.display()
            );
            process::exit(2);
        });
        let claims_toml = store
            .control_plane_compat_text(ControlPlaneCompatKind::Claims)
            .unwrap_or_else(|e| {
                eprintln!(
                    "ERROR: Cannot render claims compatibility text from {}: {e}",
                    canonical_db_path.display()
                );
                process::exit(2);
            });
        let claims = parse_claim_rows_from_toml(&claims_toml).unwrap_or_else(|e| {
            eprintln!("ERROR: Cannot build claims audit rows from canonical DB: {e}");
            process::exit(2);
        });
        return (
            claims,
            "registry/canonical/control_plane.sqlite3 (rendered legacy matrix)".to_string(),
        );
    }

    if !matrix_path.exists() {
        eprintln!("ERROR: Missing matrix: {}", matrix_path.display());
        process::exit(2);
    }

    let matrix_text = fs::read_to_string(matrix_path).unwrap_or_else(|e| {
        eprintln!("ERROR: Cannot read matrix: {e}");
        process::exit(2);
    });
    (parse_claim_rows(&matrix_text), cli.matrix.clone())
}

fn parse_claim_rows_from_toml(raw: &str) -> Result<Vec<ClaimRow>, String> {
    let value: Value = toml::from_str(raw).map_err(|e| format!("parse claims TOML: {e}"))?;
    let claims = value
        .get("claim")
        .and_then(Value::as_array)
        .ok_or_else(|| "claims array missing".to_string())?;
    let mut out = Vec::with_capacity(claims.len());
    for (idx, claim) in claims.iter().enumerate() {
        let table = claim
            .as_table()
            .ok_or_else(|| "claim row must be table".to_string())?;
        let claim_id = table_str(table, "id").to_string();
        let claim_num = claim_id
            .strip_prefix("C-")
            .ok_or_else(|| format!("invalid claim id {claim_id}"))?
            .parse::<u32>()
            .map_err(|e| format!("parse numeric claim id {claim_id}: {e}"))?;
        let last_verified = table_str(table, "last_verified").to_string();
        let last_verified_date = if last_verified.len() >= 10 {
            Some(last_verified[..10].to_string())
        } else {
            None
        };
        let status = table_str(table, "status").to_string();
        let evidence_notes = provenance_core::falsifier_text::project_optional(
            table.get("what_would_verify_refute").cloned(),
        )
        .map_err(|error| format!("invalid falsifier for {claim_id}: {error}"))?
        .or_else(|| {
            table
                .get("status_note")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_default();
        out.push(ClaimRow {
            claim_id,
            claim_num,
            claim_text: table_str(table, "statement").to_string(),
            where_stated: table_str(table, "where_stated").to_string(),
            status_cell: format!("**{}**", status),
            status_token: status,
            last_verified,
            last_verified_date,
            evidence_notes,
            lineno: idx + 1,
        });
    }
    Ok(out)
}

fn table_str<'a>(table: &'a toml::map::Map<String, Value>, key: &str) -> &'a str {
    table.get(key).and_then(Value::as_str).unwrap_or("")
}

fn render_all_audits(
    claims: &[ClaimRow],
    matrix_label: &str,
    doc_corpus: &str,
    stale_before: &str,
) -> String {
    let mut output = String::new();
    let inv = audit::id_inventory(claims);
    output.push_str(&audit::render_id_inventory(&inv, matrix_label));
    output.push_str("---\n\n");

    let status = audit::status_inventory(claims);
    output.push_str(&audit::render_status_inventory(&status, matrix_label));
    output.push_str("---\n\n");

    let stale = audit::staleness_report(claims, stale_before);
    output.push_str(&audit::render_staleness_report(
        &stale,
        stale_before,
        matrix_label,
    ));
    output.push_str("---\n\n");

    let contras = audit::status_contradictions(claims);
    output.push_str(&audit::render_contradictions(&contras, matrix_label));
    output.push_str("---\n\n");

    let tokens = audit::bold_tokens_inventory(claims);
    output.push_str(&audit::render_bold_tokens(&tokens, matrix_label));
    output.push_str("---\n\n");

    let prio = audit::priority_ranking(claims, doc_corpus);
    output.push_str(&audit::render_priority_ranking(&prio, matrix_label));
    output
}

fn load_toml<T: for<'de> Deserialize<'de>>(path: &std::path::Path) -> T {
    let raw = fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("ERROR: Cannot read {}: {e}", path.display());
        process::exit(2);
    });
    toml::from_str(&raw).unwrap_or_else(|e| {
        eprintln!("ERROR: Cannot parse {}: {e}", path.display());
        process::exit(2);
    })
}

fn build_falsification_campaign_manifest(
    inputs: CampaignRegistryInputs<'_>,
) -> FalsificationCampaignManifest {
    let mut targeted_claims = inputs
        .claims
        .iter()
        .filter(|claim| is_empirical_core_status(&claim.status))
        .cloned()
        .collect::<Vec<_>>();
    targeted_claims.sort_by_key(|claim| claim_numeric_id(&claim.id));

    let mut experiments_by_claim: BTreeMap<String, Vec<ExperimentTomlRow>> = BTreeMap::new();
    for experiment in inputs.experiments {
        let claim_ids = experiment
            .claims
            .iter()
            .chain(experiment.claim_refs.iter())
            .cloned()
            .collect::<BTreeSet<_>>();
        for claim_id in claim_ids {
            experiments_by_claim
                .entry(claim_id)
                .or_default()
                .push(experiment.clone());
        }
    }
    for rows in experiments_by_claim.values_mut() {
        rows.sort_by(|left, right| left.id.cmp(&right.id));
    }

    let mut tasks_by_claim: BTreeMap<String, Vec<ClaimTaskTomlRow>> = BTreeMap::new();
    for task in inputs.tasks {
        tasks_by_claim
            .entry(task.claim_id.clone())
            .or_default()
            .push(task.clone());
    }
    for rows in tasks_by_claim.values_mut() {
        rows.sort_by(|left, right| left.id.cmp(&right.id));
    }

    let mut insights_by_claim: BTreeMap<String, Vec<InsightTomlRow>> = BTreeMap::new();
    for insight in inputs.insights {
        let claim_ids = insight
            .claims
            .iter()
            .chain(insight.related_claims.iter())
            .cloned()
            .collect::<BTreeSet<_>>();
        for claim_id in claim_ids {
            insights_by_claim
                .entry(claim_id)
                .or_default()
                .push(insight.clone());
        }
    }
    for rows in insights_by_claim.values_mut() {
        rows.sort_by(|left, right| left.id.cmp(&right.id));
    }

    let mut all_claim_lane_counts = BTreeMap::new();
    let mut all_claim_ids_by_lane: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut superseded_claim_ids = Vec::new();
    for claim in inputs.claims {
        let lane = overall_claim_lane(&claim.status);
        *all_claim_lane_counts.entry(lane.to_string()).or_insert(0) += 1;
        all_claim_ids_by_lane
            .entry(lane.to_string())
            .or_default()
            .push(claim.id.clone());
        if lane == "superseded" {
            superseded_claim_ids.push(claim.id.clone());
        }
    }
    for claim_ids in all_claim_ids_by_lane.values_mut() {
        claim_ids.sort_by_key(|claim_id| claim_numeric_id(claim_id));
    }
    superseded_claim_ids.sort();

    let mut claims_rows = Vec::with_capacity(targeted_claims.len());
    let mut status_counts = BTreeMap::new();
    let mut family_counts = BTreeMap::new();
    let mut wave_counts = BTreeMap::new();
    let mut with_registered_experiments = 0usize;
    let mut with_gpu_experiments = 0usize;
    let mut with_tasks = 0usize;
    let mut with_open_tasks = 0usize;
    let mut with_ticket_coverage = 0usize;
    let mut with_insight_coverage = 0usize;
    let mut claims_needing_new_experiment = Vec::new();
    let mut claims_missing_task_coverage = Vec::new();
    let mut claims_missing_ticket_coverage = Vec::new();
    let mut claims_missing_insight_coverage = Vec::new();
    let mut claims_missing_all_linkage = Vec::new();
    let mut claims_missing_experiment_ticket_and_insight = Vec::new();

    for claim in targeted_claims {
        *status_counts.entry(claim.status.clone()).or_insert(0) += 1;

        let family = classify_campaign_family(&claim);
        *family_counts.entry(family.to_string()).or_insert(0) += 1;

        let experiment_rows = experiments_by_claim
            .get(&claim.id)
            .cloned()
            .unwrap_or_default();
        let task_rows = tasks_by_claim.get(&claim.id).cloned().unwrap_or_default();
        let ticket_ids = matching_ticket_ids(&claim.id, inputs.tickets);
        let insight_rows = insights_by_claim
            .get(&claim.id)
            .cloned()
            .unwrap_or_default();

        let wave = if experiment_rows.is_empty() {
            WAVE_NEW_RUST_FALSIFIER.to_string()
        } else {
            WAVE_REGISTERED_EXPERIMENT.to_string()
        };
        *wave_counts.entry(wave.clone()).or_insert(0) += 1;

        let registered_gpu_experiment = experiment_rows.iter().any(|row| row.gpu);
        let requires_new_experiment = experiment_rows.is_empty();
        let open_task_ids = task_rows
            .iter()
            .filter(|row| !CLOSED_TASK_STATUS_TOKENS.contains(&row.status_token.as_str()))
            .map(|row| row.id.clone())
            .collect::<Vec<_>>();

        if !experiment_rows.is_empty() {
            with_registered_experiments += 1;
        }
        if registered_gpu_experiment {
            with_gpu_experiments += 1;
        }
        if !task_rows.is_empty() {
            with_tasks += 1;
        } else {
            claims_missing_task_coverage.push(claim.id.clone());
        }
        if !open_task_ids.is_empty() {
            with_open_tasks += 1;
        }
        if !ticket_ids.is_empty() {
            with_ticket_coverage += 1;
        } else {
            claims_missing_ticket_coverage.push(claim.id.clone());
        }
        if !insight_rows.is_empty() {
            with_insight_coverage += 1;
        } else {
            claims_missing_insight_coverage.push(claim.id.clone());
        }
        if requires_new_experiment {
            claims_needing_new_experiment.push(claim.id.clone());
        }
        if requires_new_experiment && ticket_ids.is_empty() && insight_rows.is_empty() {
            claims_missing_experiment_ticket_and_insight.push(claim.id.clone());
        }
        if requires_new_experiment
            && task_rows.is_empty()
            && ticket_ids.is_empty()
            && insight_rows.is_empty()
        {
            claims_missing_all_linkage.push(claim.id.clone());
        }

        let canonical_falsifier = canonical_falsifier(&experiment_rows, &claim.id, family);
        let independent_countercheck = independent_countercheck_recipe(family).to_string();
        let reproducibility_floor =
            reproducibility_floor(&experiment_rows, requires_new_experiment).to_string();
        let acceptance_bar = acceptance_bar(&experiment_rows, family, requires_new_experiment)
            .into_iter()
            .map(str::to_string)
            .collect();

        claims_rows.push(FalsificationCampaignClaimRow {
            claim_id: claim.id.clone(),
            status: claim.status.clone(),
            family: family.to_string(),
            wave,
            last_verified: claim.last_verified.clone(),
            statement: claim.statement.clone(),
            where_stated: claim.where_stated.clone(),
            experiment_ids: experiment_rows.iter().map(|row| row.id.clone()).collect(),
            experiment_binaries: experiment_rows
                .iter()
                .map(|row| row.binary.clone())
                .filter(|binary| !binary.is_empty())
                .collect(),
            registered_gpu_experiment,
            task_ids: task_rows.iter().map(|row| row.id.clone()).collect(),
            open_task_ids,
            ticket_ids,
            insight_ids: insight_rows.iter().map(|row| row.id.clone()).collect(),
            requires_new_experiment,
            reproducibility_floor,
            canonical_falsifier,
            independent_countercheck,
            acceptance_bar,
        });
    }

    FalsificationCampaignManifest {
        generated_at_utc: Utc::now().to_rfc3339(),
        source_claims_toml: inputs.source_claims_toml.to_string(),
        source_experiments_toml: inputs.source_experiments_toml.to_string(),
        source_claims_tasks_toml: inputs.source_claims_tasks_toml.to_string(),
        source_claim_tickets_toml: inputs.source_claim_tickets_toml.to_string(),
        source_insights_toml: inputs.source_insights_toml.to_string(),
        all_claim_count: inputs.claims.len(),
        all_claim_lane_counts,
        all_claim_ids_by_lane,
        superseded_claim_ids,
        empirical_core_statuses: EMPIRICAL_CORE_STATUSES
            .iter()
            .map(|status| status.to_string())
            .collect(),
        target_claim_count: claims_rows.len(),
        status_counts,
        family_counts,
        wave_counts,
        coverage: FalsificationCoverage {
            with_registered_experiments,
            without_registered_experiments: claims_rows.len() - with_registered_experiments,
            with_gpu_experiments,
            with_tasks,
            with_open_tasks,
            with_ticket_coverage,
            with_insight_coverage,
            claims_needing_new_experiment,
            claims_missing_task_coverage,
            claims_missing_ticket_coverage,
            claims_missing_insight_coverage,
            claims_missing_all_linkage,
            claims_missing_experiment_ticket_and_insight,
        },
        claims: claims_rows,
    }
}

fn render_falsification_campaign_report(manifest: &FalsificationCampaignManifest) -> String {
    let mut out = String::new();
    out.push_str("# Falsification Campaign Summary\n\n");
    out.push_str(&format!("Generated at: `{}`\n", manifest.generated_at_utc));
    out.push_str(&format!(
        "Sources: `{}`, `{}`, `{}`, `{}`, `{}`\n\n",
        manifest.source_claims_toml,
        manifest.source_experiments_toml,
        manifest.source_claims_tasks_toml,
        manifest.source_claim_tickets_toml,
        manifest.source_insights_toml
    ));

    out.push_str("## All-Claims Snapshot\n\n");
    out.push_str(&format!(
        "- Claims in compatibility export: {}\n",
        manifest.all_claim_count
    ));
    for (lane, count) in &manifest.all_claim_lane_counts {
        out.push_str(&format!("- {lane}: {count}\n"));
    }
    for (lane, claim_ids) in &manifest.all_claim_ids_by_lane {
        if claim_ids.is_empty() {
            continue;
        }
        out.push_str(&format!("- {lane}_claim_ids: {}\n", claim_ids.join(", ")));
    }
    if manifest.superseded_claim_ids.is_empty() {
        out.push_str("- superseded_claim_ids: none\n\n");
    } else {
        out.push_str(&format!(
            "- superseded_claim_ids: {}\n\n",
            manifest.superseded_claim_ids.join(", ")
        ));
    }

    out.push_str("## Target Set\n\n");
    out.push_str(&format!(
        "- Empirical-core target claims: {}\n",
        manifest.target_claim_count
    ));
    out.push_str(&format!(
        "- Target statuses: {}\n\n",
        manifest.empirical_core_statuses.join(", ")
    ));

    out.push_str("### Status counts\n\n");
    for (status, count) in &manifest.status_counts {
        out.push_str(&format!("- {status}: {count}\n"));
    }
    out.push('\n');

    out.push_str("### Family counts\n\n");
    for (family, count) in &manifest.family_counts {
        out.push_str(&format!("- {family}: {count}\n"));
    }
    out.push('\n');

    out.push_str("### Wave counts\n\n");
    for (wave, count) in &manifest.wave_counts {
        out.push_str(&format!("- {wave}: {count}\n"));
    }
    out.push('\n');

    out.push_str("## Coverage\n\n");
    out.push_str(&format!(
        "- Claims with registered experiments: {}\n",
        manifest.coverage.with_registered_experiments
    ));
    out.push_str(&format!(
        "- Claims without registered experiments: {}\n",
        manifest.coverage.without_registered_experiments
    ));
    out.push_str(&format!(
        "- Claims with GPU-backed registered experiments: {}\n",
        manifest.coverage.with_gpu_experiments
    ));
    out.push_str(&format!(
        "- Claims with task coverage: {}\n",
        manifest.coverage.with_tasks
    ));
    out.push_str(&format!(
        "- Claims with open task coverage: {}\n",
        manifest.coverage.with_open_tasks
    ));
    out.push_str(&format!(
        "- Claims with ticket coverage: {}\n\n",
        manifest.coverage.with_ticket_coverage
    ));
    out.push_str(&format!(
        "- Claims with insight coverage: {}\n\n",
        manifest.coverage.with_insight_coverage
    ));

    out.push_str("### Claims needing new experiment/verifier work\n\n");
    if manifest.coverage.claims_needing_new_experiment.is_empty() {
        out.push_str("- None\n\n");
    } else {
        for chunk in manifest.coverage.claims_needing_new_experiment.chunks(12) {
            out.push_str(&format!("- {}\n", chunk.join(", ")));
        }
        out.push('\n');
    }

    out.push_str("### Claims missing task coverage\n\n");
    if manifest.coverage.claims_missing_task_coverage.is_empty() {
        out.push_str("- None\n\n");
    } else {
        for chunk in manifest.coverage.claims_missing_task_coverage.chunks(12) {
            out.push_str(&format!("- {}\n", chunk.join(", ")));
        }
        out.push('\n');
    }

    out.push_str("### Claims missing ticket coverage\n\n");
    if manifest.coverage.claims_missing_ticket_coverage.is_empty() {
        out.push_str("- None\n\n");
    } else {
        for chunk in manifest.coverage.claims_missing_ticket_coverage.chunks(12) {
            out.push_str(&format!("- {}\n", chunk.join(", ")));
        }
        out.push('\n');
    }

    out.push_str("### Claims missing insight coverage\n\n");
    if manifest.coverage.claims_missing_insight_coverage.is_empty() {
        out.push_str("- None\n\n");
    } else {
        for chunk in manifest.coverage.claims_missing_insight_coverage.chunks(12) {
            out.push_str(&format!("- {}\n", chunk.join(", ")));
        }
        out.push('\n');
    }

    out.push_str("### Claims missing all linkage\n\n");
    if manifest.coverage.claims_missing_all_linkage.is_empty() {
        out.push_str("- None\n\n");
    } else {
        for chunk in manifest.coverage.claims_missing_all_linkage.chunks(12) {
            out.push_str(&format!("- {}\n", chunk.join(", ")));
        }
        out.push('\n');
    }

    out.push_str("### Claims missing experiment, ticket, and insight coverage\n\n");
    if manifest
        .coverage
        .claims_missing_experiment_ticket_and_insight
        .is_empty()
    {
        out.push_str("- None\n\n");
    } else {
        for chunk in manifest
            .coverage
            .claims_missing_experiment_ticket_and_insight
            .chunks(12)
        {
            out.push_str(&format!("- {}\n", chunk.join(", ")));
        }
        out.push('\n');
    }

    out.push_str("## Family Worklist\n\n");
    for family in [
        FAMILY_ALGEBRAIC_STRUCTURAL,
        FAMILY_COSMOLOGY_METAMATERIAL,
        FAMILY_PARTICLE_NUMEROLOGY,
        FAMILY_SIGNAL_TRANSPORT,
    ] {
        let mut family_rows = manifest
            .claims
            .iter()
            .filter(|row| row.family == family)
            .collect::<Vec<_>>();
        if family_rows.is_empty() {
            continue;
        }
        family_rows.sort_by_key(|row| claim_numeric_id(&row.claim_id));
        out.push_str(&format!("### {family}\n\n"));
        for row in family_rows {
            let experiments = if row.experiment_ids.is_empty() {
                "none".to_string()
            } else {
                row.experiment_ids.join(", ")
            };
            let tickets = if row.ticket_ids.is_empty() {
                "none".to_string()
            } else {
                row.ticket_ids.join(", ")
            };
            let open_tasks = if row.open_task_ids.is_empty() {
                "none".to_string()
            } else {
                row.open_task_ids.join(", ")
            };
            let insights = if row.insight_ids.is_empty() {
                "none".to_string()
            } else {
                row.insight_ids.join(", ")
            };
            out.push_str(&format!(
                "- {} ({}, {}): experiments={}; open_tasks={}; tickets={}; insights={}; countercheck={}\n",
                row.claim_id,
                row.status,
                row.wave,
                experiments,
                open_tasks,
                tickets,
                insights,
                row.independent_countercheck
            ));
        }
        out.push('\n');
    }

    out
}

fn is_empirical_core_status(status: &str) -> bool {
    EMPIRICAL_CORE_STATUSES.contains(&status)
}

fn overall_claim_lane(status: &str) -> &'static str {
    match status {
        "Verified" | "Established" => "positive_complete",
        "Superseded" => "superseded",
        "Refuted"
        | "Closed/Refuted"
        | "Closed/Negative-Result"
        | "Closed/Methodology-Insufficient" => "empirical_falsification_core",
        "Closed/Research-Program"
        | "Closed/Toy"
        | "Closed/Analogy"
        | "Closed/Source-Insufficient"
        | "Closed/Obstructed" => "administrative_closed",
        "Partial" | "Provisional" | "Theoretical" | "Inconclusive" => "open_investigation",
        _ => "other",
    }
}

fn classify_campaign_family(claim: &ClaimTomlRow) -> &'static str {
    let text = claim.statement.to_ascii_lowercase();

    let signal_transport_keywords = [
        "ghost frequency",
        "chime",
        "frb",
        "pulsar",
        "ultrametric",
        "wow!",
        "wavelet",
        "betti",
        "lbm",
        "velocity fields",
        "helicity",
        "doppler",
        "gwtc-3",
        "sky positions",
        "dispersion measures",
        "snia",
        "r_aa",
        "alice",
    ];
    if signal_transport_keywords
        .iter()
        .any(|keyword| text.contains(keyword))
    {
        return FAMILY_SIGNAL_TRANSPORT;
    }

    let cosmology_metamaterial_keywords = [
        "dark energy",
        "lambdacdm",
        "luminosity distance",
        "equation of state",
        "gravastar",
        "warp drive",
        "warp ring",
        "metamaterial",
        "eta_wake",
        "rosetta",
        "h(z)",
        "pantheon+",
        "cosmology",
        "wake",
    ];
    if cosmology_metamaterial_keywords
        .iter()
        .any(|keyword| text.contains(keyword))
    {
        return FAMILY_COSMOLOGY_METAMATERIAL;
    }

    let particle_numerology_keywords = [
        "pmns",
        "particle masses",
        "pdg",
        "yukawa",
        "neutrino",
        "bosonic string",
        "f4 26d",
        "42^2",
        "1764",
        "planck mass",
        "mixing",
    ];
    if particle_numerology_keywords
        .iter()
        .any(|keyword| text.contains(keyword))
    {
        return FAMILY_PARTICLE_NUMEROLOGY;
    }

    FAMILY_ALGEBRAIC_STRUCTURAL
}

fn matching_ticket_ids(claim_id: &str, tickets: &[ClaimTicketTomlRow]) -> Vec<String> {
    let claim_num = claim_numeric_id(claim_id);
    let mut ids = tickets
        .iter()
        .filter(|ticket| ticket_covers_claim(ticket, claim_id, claim_num))
        .map(|ticket| ticket.id.clone())
        .collect::<Vec<_>>();
    ids.sort();
    ids.dedup();
    ids
}

fn ticket_covers_claim(ticket: &ClaimTicketTomlRow, claim_id: &str, claim_num: u32) -> bool {
    if ticket
        .claims_referenced
        .iter()
        .any(|candidate| candidate == claim_id)
    {
        return true;
    }
    ticket.claim_range_start != 0
        && ticket.claim_range_end != 0
        && (ticket.claim_range_start..=ticket.claim_range_end).contains(&claim_num)
}

fn claim_numeric_id(claim_id: &str) -> u32 {
    claim_id
        .strip_prefix("C-")
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(0)
}

fn canonical_falsifier(
    experiment_rows: &[ExperimentTomlRow],
    claim_id: &str,
    family: &str,
) -> String {
    if let Some(experiment) = experiment_rows.iter().find(|row| row.deterministic) {
        return format!(
            "registered experiment {} ({})",
            experiment.id, experiment.binary
        );
    }
    if let Some(experiment) = experiment_rows.first() {
        return format!(
            "registered experiment {} ({})",
            experiment.id, experiment.binary
        );
    }
    match family {
        FAMILY_COSMOLOGY_METAMATERIAL => format!(
            "{claim_id}: add Rust baseline-model comparison or parameter-termination verifier"
        ),
        FAMILY_PARTICLE_NUMEROLOGY => {
            format!("{claim_id}: add Rust blind-matching and invariance verifier")
        }
        FAMILY_SIGNAL_TRANSPORT => {
            format!("{claim_id}: add Rust null-family and ablation experiment")
        }
        _ => format!("{claim_id}: add Rust exact-enumeration or consistency verifier"),
    }
}

fn independent_countercheck_recipe(family: &str) -> &'static str {
    match family {
        FAMILY_COSMOLOGY_METAMATERIAL => {
            "alternate baseline model plus parameter-termination audit"
        }
        FAMILY_PARTICLE_NUMEROLOGY => "blind matching baseline plus unit/base invariance",
        FAMILY_SIGNAL_TRANSPORT => "alternate null family plus ablation or resolution guard",
        _ => "exact enumeration plus external-CSV or representation consistency cross-check",
    }
}

fn reproducibility_floor(
    experiment_rows: &[ExperimentTomlRow],
    requires_new_experiment: bool,
) -> &'static str {
    if requires_new_experiment {
        return "new_rust_offline_check_required";
    }
    if experiment_rows.iter().any(|row| row.gpu) {
        return "cpu_smoke_plus_gpu_acceptance";
    }
    if experiment_rows.iter().any(|row| row.deterministic) {
        return "deterministic_replay";
    }
    "seeded_replay"
}

fn acceptance_bar(
    experiment_rows: &[ExperimentTomlRow],
    family: &str,
    requires_new_experiment: bool,
) -> [&'static str; 5] {
    let family_countercheck = independent_countercheck_recipe(family);
    if requires_new_experiment {
        return [
            "cached inputs and provenance recorded",
            "new Rust offline verifier or experiment added",
            family_countercheck,
            "claim/task/ticket surfaces updated together",
            "full repo gate passes after the batch",
        ];
    }
    if experiment_rows.iter().any(|row| row.gpu) {
        return [
            "cached inputs and provenance recorded",
            "CPU smoke path passes",
            "registered GPU acceptance run passes seeded tolerance checks",
            family_countercheck,
            "claim/task/ticket surfaces updated together",
        ];
    }
    [
        "cached inputs and provenance recorded",
        "registered experiment replay passes",
        family_countercheck,
        "claim/task/ticket surfaces updated together",
        "full repo gate passes after the batch",
    ]
}

fn write_output(path: &std::path::Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).ok();
    }
    fs::write(path, contents).unwrap_or_else(|e| {
        eprintln!("ERROR: Cannot write {}: {e}", path.display());
        process::exit(2);
    });
    eprintln!("Wrote: {}", path.display());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn categorized_falsifiers_replace_only_absent_status_note_fallbacks() {
        let source = "[[claim]]\nid='C-100'\nstatus='Provisional'\nstatus_note='older status note'\n[claim.what_would_verify_refute]\nverification_outcomes=['verify C-101']\nrevision_outcomes=['revise C-102']\nabandonment_outcomes=['abandon C-103']\ninconclusive_outcomes=['inconclusive C-104']\n";
        let rows = parse_claim_rows_from_toml(source).unwrap();
        for reference in ["C-101", "C-102", "C-103", "C-104"] {
            assert!(rows[0].evidence_notes.contains(reference));
        }
        assert!(!rows[0].evidence_notes.contains("older status note"));
        assert!(
            parse_claim_rows_from_toml(&source.replace("revision_outcomes", "unknown_category"))
                .is_err()
        );
    }

    fn sample_claims() -> Vec<ClaimTomlRow> {
        vec![
            ClaimTomlRow {
                id: "C-009".to_string(),
                statement: "Tensor-network experiment exhibits entropy scaling S ~ log(L) + L^0.5."
                    .to_string(),
                status: "Refuted".to_string(),
                last_verified: "2026-02-06".to_string(),
                where_stated: "docs/C009.md".to_string(),
            },
            ClaimTomlRow {
                id: "C-402".to_string(),
                statement: "Metamaterial gravitational coupling can reduce warp drive energy requirements."
                    .to_string(),
                status: "Refuted".to_string(),
                last_verified: "2026-03-01".to_string(),
                where_stated: "docs/C402.md".to_string(),
            },
            ClaimTomlRow {
                id: "C-069".to_string(),
                statement: "Three octonionic subalgebra principal angles reproduce PMNS neutrino mixing angles."
                    .to_string(),
                status: "Refuted".to_string(),
                last_verified: "2026-02-04".to_string(),
                where_stated: "docs/C069.md".to_string(),
            },
            ClaimTomlRow {
                id: "C-071".to_string(),
                statement: "FRB dispersion measures exhibit p-adic ultrametric structure."
                    .to_string(),
                status: "Refuted".to_string(),
                last_verified: "2026-02-04".to_string(),
                where_stated: "docs/C071.md".to_string(),
            },
            ClaimTomlRow {
                id: "C-022".to_string(),
                statement: "Toy model for surreal birthday mapping.".to_string(),
                status: "Closed/Analogy".to_string(),
                last_verified: "2026-02-05".to_string(),
                where_stated: "docs/C022.md".to_string(),
            },
        ]
    }

    fn sample_experiments() -> Vec<ExperimentTomlRow> {
        vec![
            ExperimentTomlRow {
                id: "E-006".to_string(),
                binary: "gravastar-sweep".to_string(),
                deterministic: true,
                gpu: false,
                claims: vec!["C-402".to_string()],
                claim_refs: vec![],
            },
            ExperimentTomlRow {
                id: "E-002".to_string(),
                binary: "multi-dataset-ultrametric".to_string(),
                deterministic: false,
                gpu: true,
                claims: vec!["C-071".to_string()],
                claim_refs: vec![],
            },
        ]
    }

    fn sample_tasks() -> Vec<ClaimTaskTomlRow> {
        vec![
            ClaimTaskTomlRow {
                id: "CTASK-0001".to_string(),
                claim_id: "C-009".to_string(),
                status_token: "TODO".to_string(),
            },
            ClaimTaskTomlRow {
                id: "CTASK-0002".to_string(),
                claim_id: "C-402".to_string(),
                status_token: "DONE".to_string(),
            },
        ]
    }

    fn sample_tickets() -> Vec<ClaimTicketTomlRow> {
        vec![
            ClaimTicketTomlRow {
                id: "TICKET-C001-C050".to_string(),
                claim_range_start: 1,
                claim_range_end: 50,
                claims_referenced: vec![],
            },
            ClaimTicketTomlRow {
                id: "TICKET-C401-C427".to_string(),
                claim_range_start: 401,
                claim_range_end: 427,
                claims_referenced: vec![],
            },
        ]
    }

    fn sample_insights() -> Vec<InsightTomlRow> {
        vec![
            InsightTomlRow {
                id: "I-001".to_string(),
                claims: vec!["C-071".to_string()],
                related_claims: vec![],
            },
            InsightTomlRow {
                id: "I-002".to_string(),
                claims: vec![],
                related_claims: vec!["C-402".to_string()],
            },
        ]
    }

    #[test]
    fn test_classify_campaign_family_examples() {
        let claims = sample_claims();
        assert_eq!(
            classify_campaign_family(&claims[0]),
            FAMILY_ALGEBRAIC_STRUCTURAL
        );
        assert_eq!(
            classify_campaign_family(&claims[1]),
            FAMILY_COSMOLOGY_METAMATERIAL
        );
        assert_eq!(
            classify_campaign_family(&claims[2]),
            FAMILY_PARTICLE_NUMEROLOGY
        );
        assert_eq!(
            classify_campaign_family(&claims[3]),
            FAMILY_SIGNAL_TRANSPORT
        );
    }

    #[test]
    fn test_build_falsification_campaign_manifest_filters_and_counts() {
        let claims = sample_claims();
        let experiments = sample_experiments();
        let tasks = sample_tasks();
        let tickets = sample_tickets();
        let insights = sample_insights();
        let manifest = build_falsification_campaign_manifest(CampaignRegistryInputs {
            source_claims_toml: "registry/claims.toml",
            source_experiments_toml: "registry/experiments.toml",
            source_claims_tasks_toml: "registry/claims_tasks.toml",
            source_claim_tickets_toml: "registry/claim_tickets.toml",
            source_insights_toml: "registry/insights.toml",
            claims: &claims,
            experiments: &experiments,
            tasks: &tasks,
            tickets: &tickets,
            insights: &insights,
        });

        assert_eq!(manifest.all_claim_count, 5);
        assert_eq!(
            manifest
                .all_claim_lane_counts
                .get("empirical_falsification_core"),
            Some(&4)
        );
        assert_eq!(
            manifest.all_claim_lane_counts.get("administrative_closed"),
            Some(&1)
        );
        assert_eq!(
            manifest
                .all_claim_ids_by_lane
                .get("empirical_falsification_core"),
            Some(&vec![
                "C-009".to_string(),
                "C-069".to_string(),
                "C-071".to_string(),
                "C-402".to_string(),
            ])
        );
        assert_eq!(manifest.target_claim_count, 4);
        assert_eq!(manifest.status_counts.get("Refuted"), Some(&4));
        assert_eq!(
            manifest.family_counts.get(FAMILY_ALGEBRAIC_STRUCTURAL),
            Some(&1)
        );
        assert_eq!(
            manifest.family_counts.get(FAMILY_COSMOLOGY_METAMATERIAL),
            Some(&1)
        );
        assert_eq!(
            manifest.family_counts.get(FAMILY_PARTICLE_NUMEROLOGY),
            Some(&1)
        );
        assert_eq!(
            manifest.family_counts.get(FAMILY_SIGNAL_TRANSPORT),
            Some(&1)
        );
        assert_eq!(
            manifest.coverage.claims_needing_new_experiment,
            vec!["C-009".to_string(), "C-069".to_string()]
        );
        assert_eq!(manifest.coverage.with_registered_experiments, 2);
        assert_eq!(manifest.coverage.with_gpu_experiments, 1);
        assert_eq!(manifest.coverage.with_tasks, 2);
        assert_eq!(manifest.coverage.with_open_tasks, 1);
        assert_eq!(manifest.coverage.with_insight_coverage, 2);
        assert_eq!(
            manifest.coverage.claims_missing_task_coverage,
            vec!["C-069".to_string(), "C-071".to_string()]
        );
        assert_eq!(
            manifest.coverage.claims_missing_insight_coverage,
            vec!["C-009".to_string(), "C-069".to_string()]
        );
        assert_eq!(
            manifest.coverage.claims_missing_all_linkage,
            vec!["C-069".to_string()]
        );
        assert_eq!(
            manifest
                .coverage
                .claims_missing_experiment_ticket_and_insight,
            vec!["C-069".to_string()]
        );
    }

    #[test]
    fn test_wave_and_ticket_assignment() {
        let claims = sample_claims();
        let experiments = sample_experiments();
        let tasks = sample_tasks();
        let tickets = sample_tickets();
        let insights = sample_insights();
        let manifest = build_falsification_campaign_manifest(CampaignRegistryInputs {
            source_claims_toml: "registry/claims.toml",
            source_experiments_toml: "registry/experiments.toml",
            source_claims_tasks_toml: "registry/claims_tasks.toml",
            source_claim_tickets_toml: "registry/claim_tickets.toml",
            source_insights_toml: "registry/insights.toml",
            claims: &claims,
            experiments: &experiments,
            tasks: &tasks,
            tickets: &tickets,
            insights: &insights,
        });

        let c009 = manifest
            .claims
            .iter()
            .find(|row| row.claim_id == "C-009")
            .expect("C-009 present");
        assert_eq!(c009.wave, WAVE_NEW_RUST_FALSIFIER);
        assert_eq!(c009.ticket_ids, vec!["TICKET-C001-C050".to_string()]);
        assert_eq!(
            c009.independent_countercheck,
            "exact enumeration plus external-CSV or representation consistency cross-check"
        );

        let c071 = manifest
            .claims
            .iter()
            .find(|row| row.claim_id == "C-071")
            .expect("C-071 present");
        assert_eq!(c071.wave, WAVE_REGISTERED_EXPERIMENT);
        assert!(c071.registered_gpu_experiment);
        assert_eq!(c071.insight_ids, vec!["I-001".to_string()]);
        assert_eq!(c071.reproducibility_floor, "cpu_smoke_plus_gpu_acceptance");

        let c402 = manifest
            .claims
            .iter()
            .find(|row| row.claim_id == "C-402")
            .expect("C-402 present");
        assert_eq!(c402.insight_ids, vec!["I-002".to_string()]);
    }

    #[test]
    fn test_overall_claim_lane_classification() {
        assert_eq!(overall_claim_lane("Verified"), "positive_complete");
        assert_eq!(overall_claim_lane("Superseded"), "superseded");
        assert_eq!(
            overall_claim_lane("Refuted"),
            "empirical_falsification_core"
        );
        assert_eq!(
            overall_claim_lane("Closed/Methodology-Insufficient"),
            "empirical_falsification_core"
        );
        assert_eq!(overall_claim_lane("Closed/Toy"), "administrative_closed");
        assert_eq!(overall_claim_lane("Provisional"), "open_investigation");
        assert_eq!(overall_claim_lane("Weird"), "other");
    }
}

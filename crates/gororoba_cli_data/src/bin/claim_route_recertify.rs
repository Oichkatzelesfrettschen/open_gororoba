use anyhow::{Context, Result};
use chrono::Utc;
use clap::Parser;
use glob::glob;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

#[derive(Parser, Debug)]
#[command(
    name = "claim-route-recertify",
    about = "Map claims and insights onto routed dataset analyses and classify structural versus scientifically re-executed status"
)]
struct Cli {
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,

    #[arg(long, default_value = "registry/claims.toml")]
    claims: PathBuf,

    #[arg(long, default_value = "registry/insights.toml")]
    insights: PathBuf,

    #[arg(long, default_value = "registry/experiments.toml")]
    experiments: PathBuf,

    #[arg(long)]
    routing_report: Option<PathBuf>,

    #[arg(long)]
    predictive_report: Option<PathBuf>,

    #[arg(long)]
    invariance_report: Option<PathBuf>,

    #[arg(long)]
    sparse_policy_report: Option<PathBuf>,

    #[arg(long)]
    null_classification_report: Option<PathBuf>,

    #[arg(long)]
    out_toml: Option<PathBuf>,

    #[arg(long)]
    out_md: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct ClaimsFile {
    #[serde(default, rename = "claim")]
    claims: Vec<ClaimRecord>,
}

#[derive(Debug, Clone, Deserialize)]
struct ClaimRecord {
    id: String,
    statement: String,
    status: String,
}

#[derive(Debug, Deserialize)]
struct InsightsFile {
    #[serde(default, rename = "insight")]
    insights: Vec<InsightRecord>,
}

#[derive(Debug, Clone, Deserialize)]
struct InsightRecord {
    id: String,
    title: String,
    #[serde(default)]
    status: String,
    #[serde(default)]
    claims: Vec<String>,
    #[serde(default)]
    related_claims: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ExperimentsFile {
    #[serde(default, rename = "experiment")]
    experiments: Vec<ExperimentRecord>,
}

#[derive(Debug, Clone, Deserialize)]
struct ExperimentRecord {
    id: String,
    binary: String,
    #[serde(default)]
    claims: Vec<String>,
    #[serde(default)]
    claim_refs: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RoutingReport {
    #[serde(default)]
    rows: Vec<RouteRow>,
}

#[derive(Debug, Clone, Deserialize)]
struct RouteRow {
    dataset_key: String,
    #[serde(default)]
    analysis_binaries: Vec<String>,
    status: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
enum RecertificationStatus {
    StructurallyIndexed,
    RouteLinked,
    ScientificallyReexecuted,
    PredictivelyReexecuted,
    CrossMissionNormalized,
    CompressionPreserved,
    ArchiveStructureNull,
    ResidualCandidate,
}

#[derive(Debug, Serialize)]
struct ClaimRecertificationRow {
    claim_id: String,
    claim_status: String,
    recertification_status: RecertificationStatus,
    statement: String,
    linked_experiment_ids: Vec<String>,
    linked_binaries: Vec<String>,
    linked_dataset_keys: Vec<String>,
    fully_analyzed_dataset_keys: Vec<String>,
    cube_route_dataset_keys: Vec<String>,
    evidence_notes: Vec<String>,
}

#[derive(Debug, Serialize)]
struct InsightRecertificationRow {
    insight_id: String,
    insight_status: String,
    recertification_status: RecertificationStatus,
    title: String,
    claim_refs: Vec<String>,
    scientifically_reexecuted_claims: Vec<String>,
    route_linked_claims: Vec<String>,
}

#[derive(Debug, Serialize)]
struct Report {
    generated_at_utc: String,
    routing_report: String,
    predictive_report: Option<String>,
    invariance_report: Option<String>,
    sparse_policy_report: Option<String>,
    null_classification_report: Option<String>,
    claim_count: usize,
    insight_count: usize,
    claim_status_counts: BTreeMap<String, usize>,
    insight_status_counts: BTreeMap<String, usize>,
    route_dataset_keys_with_cube_consumers: Vec<String>,
    claims: Vec<ClaimRecertificationRow>,
    insights: Vec<InsightRecertificationRow>,
    needs_registry_linkage: Vec<String>,
}

#[derive(Debug, Default)]
struct ClaimAccumulator {
    experiment_ids: BTreeSet<String>,
    binaries: BTreeSet<String>,
    dataset_keys: BTreeSet<String>,
    fully_analyzed_dataset_keys: BTreeSet<String>,
    cube_route_dataset_keys: BTreeSet<String>,
}

#[derive(Debug, Deserialize)]
struct PredictiveReport {
    #[serde(default)]
    models: Vec<PredictiveModelRow>,
}

#[derive(Debug, Deserialize)]
struct PredictiveModelRow {
    feature_mode: String,
    name: String,
    auroc: f64,
}

#[derive(Debug, Deserialize)]
struct InvarianceReport {
    #[serde(default)]
    missions: Vec<InvarianceRow>,
}

#[derive(Debug, Deserialize)]
struct InvarianceRow {
    feature_mode: String,
    leave_one_mission_out_cosine: f64,
}

#[derive(Debug, Deserialize)]
struct SparsePolicyReport {
    #[serde(default)]
    policies: Vec<SparsePolicyRow>,
    #[serde(default)]
    execution_plans: Vec<SparsePlanRow>,
}

#[derive(Debug, Deserialize)]
struct SparsePolicyRow {
    name: String,
    event_label_recall: f64,
}

#[derive(Debug, Deserialize)]
struct SparsePlanRow {
    mask_name: String,
    sparse_bf16_aa_projected_gib: f64,
}

#[derive(Debug, Deserialize)]
struct NullClassificationReport {
    #[serde(default)]
    rows: Vec<NullClassificationRow>,
}

#[derive(Debug, Deserialize)]
struct NullClassificationRow {
    dataset: String,
    classification: String,
}

#[derive(Debug, Default)]
struct EvidenceContext {
    predictive_strengthened: bool,
    normalized_invariance_improved: bool,
    compression_preserved: bool,
    archive_null_datasets: BTreeSet<String>,
    residual_candidate_datasets: BTreeSet<String>,
    needs_registry_linkage: Vec<String>,
}

struct BuildInputs {
    claims: ClaimsFile,
    insights: InsightsFile,
    experiments: ExperimentsFile,
    routing: RoutingReport,
    routing_report: PathBuf,
    predictive_report: Option<PathBuf>,
    invariance_report: Option<PathBuf>,
    sparse_policy_report: Option<PathBuf>,
    null_classification_report: Option<PathBuf>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let routing_report = match cli.routing_report {
        Some(path) => cli.repo_root.join(path),
        None => latest_report(&cli.repo_root, "reports/dataset_routing_*.toml")?,
    };
    let predictive_report = resolve_optional_report(
        &cli.repo_root,
        cli.predictive_report,
        "reports/heliosphere_predictive_eval_*.toml",
    )?;
    let invariance_report = resolve_optional_report(
        &cli.repo_root,
        cli.invariance_report,
        "reports/heliosphere_cross_mission_invariance_*.toml",
    )?;
    let sparse_policy_report = resolve_optional_report(
        &cli.repo_root,
        cli.sparse_policy_report,
        "reports/heliosphere_sparse_policy_*.toml",
    )?;
    let null_classification_report = resolve_optional_report(
        &cli.repo_root,
        cli.null_classification_report,
        "reports/catalog_feature_null_classification_*.toml",
    )?;

    let claims_path = cli.repo_root.join(&cli.claims);
    let insights_path = cli.repo_root.join(&cli.insights);
    let experiments_path = cli.repo_root.join(&cli.experiments);

    let claims: ClaimsFile = toml::from_str(&fs::read_to_string(&claims_path)?)
        .with_context(|| format!("parse {}", claims_path.display()))?;
    let insights: InsightsFile = toml::from_str(&fs::read_to_string(&insights_path)?)
        .with_context(|| format!("parse {}", insights_path.display()))?;
    let experiments: ExperimentsFile = toml::from_str(&fs::read_to_string(&experiments_path)?)
        .with_context(|| format!("parse {}", experiments_path.display()))?;
    let routing: RoutingReport = toml::from_str(&fs::read_to_string(&routing_report)?)
        .with_context(|| format!("parse {}", routing_report.display()))?;
    let evidence = load_evidence_context(
        predictive_report.as_deref(),
        invariance_report.as_deref(),
        sparse_policy_report.as_deref(),
        null_classification_report.as_deref(),
    )?;

    let report = build_report(
        BuildInputs {
            claims,
            insights,
            experiments,
            routing,
            routing_report: routing_report.clone(),
            predictive_report: predictive_report.clone(),
            invariance_report: invariance_report.clone(),
            sparse_policy_report: sparse_policy_report.clone(),
            null_classification_report: null_classification_report.clone(),
        },
        &evidence,
    );
    let date = Utc::now().date_naive();
    let out_toml = cli.out_toml.unwrap_or_else(|| {
        PathBuf::from("reports").join(format!("claim_route_recertification_{}.toml", date))
    });
    let out_md = cli.out_md.unwrap_or_else(|| {
        PathBuf::from("reports").join(format!("claim_route_recertification_{}.md", date))
    });
    if let Some(parent) = out_toml.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = out_md.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&out_toml, toml::to_string_pretty(&report)?)
        .with_context(|| format!("write {}", out_toml.display()))?;
    fs::write(&out_md, render_markdown(&report))
        .with_context(|| format!("write {}", out_md.display()))?;

    println!("claims = {}", report.claim_count);
    println!("insights = {}", report.insight_count);
    println!("toml = {}", out_toml.display());
    println!("md = {}", out_md.display());
    Ok(())
}

fn build_report(inputs: BuildInputs, evidence: &EvidenceContext) -> Report {
    let BuildInputs {
        claims,
        insights,
        experiments,
        routing,
        routing_report,
        predictive_report,
        invariance_report,
        sparse_policy_report,
        null_classification_report,
    } = inputs;
    let mut routes_by_binary: BTreeMap<String, Vec<RouteRow>> = BTreeMap::new();
    let mut cube_route_dataset_keys = BTreeSet::new();
    for row in routing.rows {
        let has_cube_consumer = row
            .analysis_binaries
            .iter()
            .any(|binary| binary == "catalog-feature-cube" || binary == "catalog-feature-algebra");
        if has_cube_consumer {
            cube_route_dataset_keys.insert(row.dataset_key.clone());
        }
        for binary in &row.analysis_binaries {
            routes_by_binary
                .entry(binary.clone())
                .or_default()
                .push(row.clone());
        }
    }

    let mut claim_accumulators: BTreeMap<String, ClaimAccumulator> = BTreeMap::new();
    for experiment in experiments.experiments {
        let mut claim_ids = BTreeSet::new();
        claim_ids.extend(experiment.claims);
        claim_ids.extend(experiment.claim_refs);
        if claim_ids.is_empty() {
            continue;
        }
        let Some(route_rows) = routes_by_binary.get(&experiment.binary) else {
            continue;
        };
        for claim_id in claim_ids {
            let entry = claim_accumulators.entry(claim_id).or_default();
            entry.experiment_ids.insert(experiment.id.clone());
            entry.binaries.insert(experiment.binary.clone());
            for route in route_rows {
                entry.dataset_keys.insert(route.dataset_key.clone());
                if route.status == "fully_analyzed" {
                    entry.fully_analyzed_dataset_keys
                        .insert(route.dataset_key.clone());
                }
                if route
                    .analysis_binaries
                    .iter()
                    .any(|binary| binary == "catalog-feature-cube" || binary == "catalog-feature-algebra")
                {
                    entry.cube_route_dataset_keys
                        .insert(route.dataset_key.clone());
                }
            }
        }
    }

    let mut claim_rows = Vec::new();
    let mut claim_status_counts = BTreeMap::new();
    let mut claim_status_map = BTreeMap::new();
    for claim in claims.claims {
        let accumulator = claim_accumulators.get(&claim.id);
        let (recertification_status, evidence_notes) = classify_claim(accumulator, evidence);
        *claim_status_counts
            .entry(status_label(recertification_status).to_string())
            .or_insert(0) += 1;
        claim_status_map.insert(claim.id.clone(), recertification_status);
        claim_rows.push(ClaimRecertificationRow {
            claim_id: claim.id,
            claim_status: claim.status,
            recertification_status,
            statement: claim.statement,
            linked_experiment_ids: accumulator
                .map(|acc| set_to_vec(&acc.experiment_ids))
                .unwrap_or_default(),
            linked_binaries: accumulator
                .map(|acc| set_to_vec(&acc.binaries))
                .unwrap_or_default(),
            linked_dataset_keys: accumulator
                .map(|acc| set_to_vec(&acc.dataset_keys))
                .unwrap_or_default(),
            fully_analyzed_dataset_keys: accumulator
                .map(|acc| set_to_vec(&acc.fully_analyzed_dataset_keys))
                .unwrap_or_default(),
            cube_route_dataset_keys: accumulator
                .map(|acc| set_to_vec(&acc.cube_route_dataset_keys))
                .unwrap_or_default(),
            evidence_notes,
        });
    }
    claim_rows.sort_by(|a, b| a.claim_id.cmp(&b.claim_id));

    let mut insight_rows = Vec::new();
    let mut insight_status_counts = BTreeMap::new();
    for insight in insights.insights {
        let claim_refs = insight_claim_refs(&insight);
        let scientifically_reexecuted_claims = claim_refs
            .iter()
            .filter(|claim_id| {
                claim_status_map
                    .get(*claim_id)
                    .copied()
                    .unwrap_or(RecertificationStatus::StructurallyIndexed)
                    == RecertificationStatus::ScientificallyReexecuted
            })
            .cloned()
            .collect::<Vec<_>>();
        let route_linked_claims = claim_refs
            .iter()
            .filter(|claim_id| {
                status_rank(
                    claim_status_map
                        .get(*claim_id)
                        .copied()
                        .unwrap_or(RecertificationStatus::StructurallyIndexed),
                ) >= status_rank(RecertificationStatus::RouteLinked)
            })
            .cloned()
            .collect::<Vec<_>>();
        let recertification_status = claim_refs
            .iter()
            .filter_map(|claim_id| claim_status_map.get(claim_id).copied())
            .max_by_key(|status| status_rank(*status))
            .unwrap_or(RecertificationStatus::StructurallyIndexed);
        *insight_status_counts
            .entry(status_label(recertification_status).to_string())
            .or_insert(0) += 1;
        insight_rows.push(InsightRecertificationRow {
            insight_id: insight.id,
            insight_status: insight.status,
            recertification_status,
            title: insight.title,
            claim_refs,
            scientifically_reexecuted_claims,
            route_linked_claims,
        });
    }
    insight_rows.sort_by(|a, b| a.insight_id.cmp(&b.insight_id));

    Report {
        generated_at_utc: Utc::now().to_rfc3339(),
        routing_report: routing_report.display().to_string(),
        predictive_report: predictive_report
            .as_ref()
            .map(|path| path.display().to_string()),
        invariance_report: invariance_report
            .as_ref()
            .map(|path| path.display().to_string()),
        sparse_policy_report: sparse_policy_report
            .as_ref()
            .map(|path| path.display().to_string()),
        null_classification_report: null_classification_report
            .as_ref()
            .map(|path| path.display().to_string()),
        claim_count: claim_rows.len(),
        insight_count: insight_rows.len(),
        claim_status_counts,
        insight_status_counts,
        route_dataset_keys_with_cube_consumers: cube_route_dataset_keys.into_iter().collect(),
        claims: claim_rows,
        insights: insight_rows,
        needs_registry_linkage: evidence.needs_registry_linkage.clone(),
    }
}

fn latest_report(repo_root: &Path, pattern: &str) -> Result<PathBuf> {
    let absolute = repo_root.join(pattern);
    let pattern_text = absolute
        .to_str()
        .with_context(|| format!("non-utf8 path {}", absolute.display()))?;
    let mut matches = glob(pattern_text)?
        .filter_map(|entry| entry.ok())
        .collect::<Vec<_>>();
    matches.sort();
    matches
        .pop()
        .with_context(|| format!("no report matched {}", absolute.display()))
}

fn resolve_optional_report(
    repo_root: &Path,
    explicit: Option<PathBuf>,
    pattern: &str,
) -> Result<Option<PathBuf>> {
    if let Some(path) = explicit {
        return Ok(Some(repo_root.join(path)));
    }
    latest_report_optional(repo_root, pattern)
}

fn latest_report_optional(repo_root: &Path, pattern: &str) -> Result<Option<PathBuf>> {
    let absolute = repo_root.join(pattern);
    let pattern_text = absolute
        .to_str()
        .with_context(|| format!("non-utf8 path {}", absolute.display()))?;
    let mut matches = glob(pattern_text)?
        .filter_map(|entry| entry.ok())
        .collect::<Vec<_>>();
    matches.sort();
    Ok(matches.pop())
}

fn load_evidence_context(
    predictive_report: Option<&Path>,
    invariance_report: Option<&Path>,
    sparse_policy_report: Option<&Path>,
    null_classification_report: Option<&Path>,
) -> Result<EvidenceContext> {
    let mut context = EvidenceContext::default();
    if let Some(path) = predictive_report {
        let report: PredictiveReport = toml::from_str(&fs::read_to_string(path)?)
            .with_context(|| format!("parse {}", path.display()))?;
        let scalar_auroc = report
            .models
            .iter()
            .find(|row| row.feature_mode == "raw" && row.name == "scalar_event_score")
            .map(|row| row.auroc)
            .unwrap_or(f64::NAN);
        let best_non_scalar_auroc = report
            .models
            .iter()
            .filter(|row| row.name != "scalar_event_score")
            .map(|row| row.auroc)
            .fold(f64::NEG_INFINITY, f64::max);
        context.predictive_strengthened =
            scalar_auroc.is_finite() && best_non_scalar_auroc > scalar_auroc;
        if context.predictive_strengthened {
            context
                .needs_registry_linkage
                .push("predictive evidence updated; registry claim linkage remains report-first".to_string());
        }
    }
    if let Some(path) = invariance_report {
        let report: InvarianceReport = toml::from_str(&fs::read_to_string(path)?)
            .with_context(|| format!("parse {}", path.display()))?;
        let raw_mean = mean_report_values(
            &report
                .missions
                .iter()
                .filter(|row| row.feature_mode == "raw")
                .map(|row| row.leave_one_mission_out_cosine)
                .collect::<Vec<_>>(),
        );
        let normalized_mean = mean_report_values(
            &report
                .missions
                .iter()
                .filter(|row| row.feature_mode == "normalized")
                .map(|row| row.leave_one_mission_out_cosine)
                .collect::<Vec<_>>(),
        );
        context.normalized_invariance_improved =
            raw_mean.is_finite() && normalized_mean.is_finite() && normalized_mean > raw_mean;
    }
    if let Some(path) = sparse_policy_report {
        let report: SparsePolicyReport = toml::from_str(&fs::read_to_string(path)?)
            .with_context(|| format!("parse {}", path.display()))?;
        let baseline = report.policies.iter().find(|row| row.name == "robust_baseline");
        let invariant = report
            .policies
            .iter()
            .find(|row| row.name == "invariant_budget_policy");
        let invariant_plan = report
            .execution_plans
            .iter()
            .find(|row| row.mask_name == "invariant_budget_policy");
        if let (Some(baseline), Some(invariant), Some(plan)) = (baseline, invariant, invariant_plan)
        {
            context.compression_preserved = plan.sparse_bf16_aa_projected_gib <= 12.0
                && invariant.event_label_recall >= baseline.event_label_recall;
        }
    }
    if let Some(path) = null_classification_report {
        let report: NullClassificationReport = toml::from_str(&fs::read_to_string(path)?)
            .with_context(|| format!("parse {}", path.display()))?;
        for row in report.rows {
            match row.classification.as_str() {
                "archive_structure_null" => {
                    context
                        .archive_null_datasets
                        .insert(normalize_dataset_key(&row.dataset));
                }
                "residual_astrophysical_candidate" => {
                    context
                        .residual_candidate_datasets
                        .insert(normalize_dataset_key(&row.dataset));
                }
                _ => {}
            }
        }
    }
    Ok(context)
}

fn classify_claim(
    accumulator: Option<&ClaimAccumulator>,
    evidence: &EvidenceContext,
) -> (RecertificationStatus, Vec<String>) {
    let mut notes = Vec::new();
    let Some(acc) = accumulator else {
        return (RecertificationStatus::StructurallyIndexed, notes);
    };
    let dataset_keys = acc
        .dataset_keys
        .iter()
        .map(|value| normalize_dataset_key(value))
        .collect::<BTreeSet<_>>();
    if dataset_keys
        .iter()
        .any(|dataset| evidence.residual_candidate_datasets.contains(dataset))
    {
        notes.push("residualized algebra+topology survived deconfounding".to_string());
        return (RecertificationStatus::ResidualCandidate, notes);
    }
    if dataset_keys
        .iter()
        .any(|dataset| evidence.archive_null_datasets.contains(dataset))
    {
        notes.push("residualized algebra/topology classified this lane as archive structure".to_string());
        return (RecertificationStatus::ArchiveStructureNull, notes);
    }
    if evidence.predictive_strengthened
        && acc
            .binaries
            .iter()
            .any(|binary| binary == "heliosphere-predictive-eval")
    {
        notes.push("normalized heliosphere invariants beat the scalar baseline".to_string());
        return (RecertificationStatus::PredictivelyReexecuted, notes);
    }
    if evidence.compression_preserved
        && acc
            .binaries
            .iter()
            .any(|binary| binary == "heliosphere-sparse-preservation")
    {
        notes.push("budgeted sparse policy preserved recall under the 12 GiB constraint".to_string());
        return (RecertificationStatus::CompressionPreserved, notes);
    }
    if evidence.normalized_invariance_improved
        && acc
            .binaries
            .iter()
            .any(|binary| binary == "heliosphere-cross-mission-invariance")
    {
        notes.push("cross-mission normalization improved leave-one-mission-out similarity".to_string());
        return (RecertificationStatus::CrossMissionNormalized, notes);
    }
    if !acc.fully_analyzed_dataset_keys.is_empty() {
        return (RecertificationStatus::ScientificallyReexecuted, notes);
    }
    if !acc.dataset_keys.is_empty() {
        return (RecertificationStatus::RouteLinked, notes);
    }
    (RecertificationStatus::StructurallyIndexed, notes)
}

fn insight_claim_refs(insight: &InsightRecord) -> Vec<String> {
    let mut refs = BTreeSet::new();
    refs.extend(insight.claims.iter().cloned());
    refs.extend(insight.related_claims.iter().cloned());
    refs.into_iter().collect()
}

fn set_to_vec(values: &BTreeSet<String>) -> Vec<String> {
    values.iter().cloned().collect()
}

fn mean_report_values(values: &[f64]) -> f64 {
    let finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return f64::NAN;
    }
    finite.iter().sum::<f64>() / finite.len() as f64
}

fn normalize_dataset_key(value: &str) -> String {
    value
        .trim()
        .to_ascii_lowercase()
        .replace([' ', '_', '/'], "-")
}

fn status_rank(status: RecertificationStatus) -> usize {
    match status {
        RecertificationStatus::StructurallyIndexed => 0,
        RecertificationStatus::RouteLinked => 1,
        RecertificationStatus::ScientificallyReexecuted => 2,
        RecertificationStatus::CrossMissionNormalized => 3,
        RecertificationStatus::PredictivelyReexecuted => 4,
        RecertificationStatus::CompressionPreserved => 5,
        RecertificationStatus::ArchiveStructureNull => 6,
        RecertificationStatus::ResidualCandidate => 7,
    }
}

fn status_label(status: RecertificationStatus) -> &'static str {
    match status {
        RecertificationStatus::StructurallyIndexed => "structurally_indexed",
        RecertificationStatus::RouteLinked => "route_linked",
        RecertificationStatus::ScientificallyReexecuted => "scientifically_reexecuted",
        RecertificationStatus::PredictivelyReexecuted => "predictively_reexecuted",
        RecertificationStatus::CrossMissionNormalized => "cross_mission_normalized",
        RecertificationStatus::CompressionPreserved => "compression_preserved",
        RecertificationStatus::ArchiveStructureNull => "archive_structure_null",
        RecertificationStatus::ResidualCandidate => "residual_candidate",
    }
}

fn render_markdown(report: &Report) -> String {
    let mut out = String::new();
    out.push_str("# Claim Route Recertification\n\n");
    out.push_str(&format!(
        "Generated at `{}` from `{}`.\n\n",
        report.generated_at_utc, report.routing_report
    ));
    out.push_str("## Claim Status Counts\n\n");
    for (status, count) in &report.claim_status_counts {
        out.push_str(&format!("- `{status}`: {count}\n"));
    }
    out.push_str("\n## Insight Status Counts\n\n");
    for (status, count) in &report.insight_status_counts {
        out.push_str(&format!("- `{status}`: {count}\n"));
    }
    out.push_str("\n## Cube-Capable Route Datasets\n\n");
    for dataset_key in &report.route_dataset_keys_with_cube_consumers {
        out.push_str(&format!("- `{dataset_key}`\n"));
    }
    out.push_str("\n## Scientifically Re-Executed Claims\n\n");
    for row in &report.claims {
        if status_rank(row.recertification_status)
            >= status_rank(RecertificationStatus::ScientificallyReexecuted)
        {
            out.push_str(&format!(
                "- `{}` [{}] via `{}` on `{}`\n",
                row.claim_id,
                status_label(row.recertification_status),
                row.linked_binaries.join(", "),
                row.fully_analyzed_dataset_keys.join(", ")
            ));
        }
    }
    if !report.needs_registry_linkage.is_empty() {
        out.push_str("\n## Needs Registry Linkage\n\n");
        for item in &report.needs_registry_linkage {
            out.push_str(&format!("- {item}\n"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{RecertificationStatus, status_rank};

    #[test]
    fn residual_candidate_ranks_above_archive_null() {
        assert!(status_rank(RecertificationStatus::ResidualCandidate)
            > status_rank(RecertificationStatus::ArchiveStructureNull));
        assert!(status_rank(RecertificationStatus::PredictivelyReexecuted)
            > status_rank(RecertificationStatus::ScientificallyReexecuted));
    }
}

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
    claim_count: usize,
    insight_count: usize,
    claim_status_counts: BTreeMap<String, usize>,
    insight_status_counts: BTreeMap<String, usize>,
    route_dataset_keys_with_cube_consumers: Vec<String>,
    claims: Vec<ClaimRecertificationRow>,
    insights: Vec<InsightRecertificationRow>,
}

#[derive(Debug, Default)]
struct ClaimAccumulator {
    experiment_ids: BTreeSet<String>,
    binaries: BTreeSet<String>,
    dataset_keys: BTreeSet<String>,
    fully_analyzed_dataset_keys: BTreeSet<String>,
    cube_route_dataset_keys: BTreeSet<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let routing_report = match cli.routing_report {
        Some(path) => cli.repo_root.join(path),
        None => latest_report(&cli.repo_root, "reports/dataset_routing_*.toml")?,
    };

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

    let report = build_report(claims, insights, experiments, routing, &routing_report);
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

fn build_report(
    claims: ClaimsFile,
    insights: InsightsFile,
    experiments: ExperimentsFile,
    routing: RoutingReport,
    routing_report: &Path,
) -> Report {
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
        let recertification_status = classify_claim(accumulator);
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
                matches!(
                    claim_status_map
                        .get(*claim_id)
                        .copied()
                        .unwrap_or(RecertificationStatus::StructurallyIndexed),
                    RecertificationStatus::RouteLinked
                        | RecertificationStatus::ScientificallyReexecuted
                )
            })
            .cloned()
            .collect::<Vec<_>>();
        let recertification_status = classify_insight(
            claim_refs.len(),
            scientifically_reexecuted_claims.len(),
            route_linked_claims.len(),
        );
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
        claim_count: claim_rows.len(),
        insight_count: insight_rows.len(),
        claim_status_counts,
        insight_status_counts,
        route_dataset_keys_with_cube_consumers: cube_route_dataset_keys.into_iter().collect(),
        claims: claim_rows,
        insights: insight_rows,
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

fn classify_claim(accumulator: Option<&ClaimAccumulator>) -> RecertificationStatus {
    match accumulator {
        Some(acc) if !acc.fully_analyzed_dataset_keys.is_empty() => {
            RecertificationStatus::ScientificallyReexecuted
        }
        Some(acc) if !acc.dataset_keys.is_empty() => RecertificationStatus::RouteLinked,
        _ => RecertificationStatus::StructurallyIndexed,
    }
}

fn classify_insight(
    claim_ref_count: usize,
    scientifically_reexecuted_count: usize,
    route_linked_count: usize,
) -> RecertificationStatus {
    if claim_ref_count > 0 && scientifically_reexecuted_count == claim_ref_count {
        return RecertificationStatus::ScientificallyReexecuted;
    }
    if route_linked_count > 0 {
        return RecertificationStatus::RouteLinked;
    }
    RecertificationStatus::StructurallyIndexed
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

fn status_label(status: RecertificationStatus) -> &'static str {
    match status {
        RecertificationStatus::StructurallyIndexed => "structurally_indexed",
        RecertificationStatus::RouteLinked => "route_linked",
        RecertificationStatus::ScientificallyReexecuted => "scientifically_reexecuted",
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
        if row.recertification_status == RecertificationStatus::ScientificallyReexecuted {
            out.push_str(&format!(
                "- `{}` via `{}` on `{}`\n",
                row.claim_id,
                row.linked_binaries.join(", "),
                row.fully_analyzed_dataset_keys.join(", ")
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{RecertificationStatus, classify_insight};

    #[test]
    fn insight_requires_all_claims_for_full_reexecution() {
        assert_eq!(
            classify_insight(2, 2, 2),
            RecertificationStatus::ScientificallyReexecuted
        );
        assert_eq!(classify_insight(2, 1, 2), RecertificationStatus::RouteLinked);
        assert_eq!(
            classify_insight(0, 0, 0),
            RecertificationStatus::StructurallyIndexed
        );
    }
}

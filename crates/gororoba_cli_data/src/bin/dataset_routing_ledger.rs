use anyhow::{Context, Result};
use chrono::Utc;
use clap::Parser;
use glob::glob;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

#[derive(Parser, Debug)]
#[command(
    name = "dataset-routing-ledger",
    about = "Generate dataset -> parser -> analysis -> report routing ledger"
)]
struct Cli {
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,

    #[arg(long, default_value = "registry/dataset_routes.toml")]
    routes: PathBuf,

    #[arg(long)]
    out_toml: Option<PathBuf>,

    #[arg(long)]
    out_md: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize)]
struct RouteFile {
    #[serde(default)]
    route: Vec<RouteSpec>,
}

#[derive(Debug, Clone, Deserialize)]
struct RouteSpec {
    dataset_key: String,
    family: String,
    provider_surface: String,
    ingress_binary: String,
    #[serde(default)]
    local_globs: Vec<String>,
    #[serde(default)]
    parser_ref: String,
    #[serde(default)]
    analysis_binaries: Vec<String>,
    #[serde(default)]
    report_patterns: Vec<String>,
    #[serde(default)]
    notes: String,
}

#[derive(Debug, Clone, Serialize)]
struct RouteRow {
    dataset_key: String,
    family: String,
    provider_surface: String,
    ingress_binary: String,
    parser_ref: String,
    analysis_binaries: Vec<String>,
    report_patterns: Vec<String>,
    cache_present: bool,
    cache_match_count: usize,
    report_present: bool,
    report_match_count: usize,
    status: String,
    notes: String,
}

#[derive(Debug, Serialize)]
struct LedgerReport {
    generated_at_utc: String,
    route_count: usize,
    status_counts: BTreeMap<String, usize>,
    rows: Vec<RouteRow>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let routes_path = cli.repo_root.join(&cli.routes);
    let content = fs::read_to_string(&routes_path)
        .with_context(|| format!("read {}", routes_path.display()))?;
    let file: RouteFile =
        toml::from_str(&content).with_context(|| format!("parse {}", routes_path.display()))?;

    let report = build_report(&cli.repo_root, file.route)?;
    let date = Utc::now().date_naive();
    let out_toml = cli
        .out_toml
        .unwrap_or_else(|| PathBuf::from("reports").join(format!("dataset_routing_{}.toml", date)));
    let out_md = cli
        .out_md
        .unwrap_or_else(|| PathBuf::from("reports").join(format!("dataset_routing_{}.md", date)));
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

    println!("routes = {}", report.route_count);
    println!("toml = {}", out_toml.display());
    println!("md = {}", out_md.display());
    Ok(())
}

fn build_report(repo_root: &Path, specs: Vec<RouteSpec>) -> Result<LedgerReport> {
    let mut rows = Vec::new();
    let mut status_counts = BTreeMap::new();
    for spec in specs {
        let cache_match_count = count_matches(repo_root, &spec.local_globs)?;
        let report_match_count = count_matches(repo_root, &spec.report_patterns)?;
        let cache_present = cache_match_count > 0;
        let report_present = report_match_count > 0;
        let status = derive_status(&spec, report_present);
        *status_counts.entry(status.clone()).or_insert(0) += 1;
        rows.push(RouteRow {
            dataset_key: spec.dataset_key,
            family: spec.family,
            provider_surface: spec.provider_surface,
            ingress_binary: spec.ingress_binary,
            parser_ref: spec.parser_ref,
            analysis_binaries: spec.analysis_binaries,
            report_patterns: spec.report_patterns,
            cache_present,
            cache_match_count,
            report_present,
            report_match_count,
            status,
            notes: spec.notes,
        });
    }
    rows.sort_by(|a, b| {
        (&a.family, &a.dataset_key, &a.provider_surface).cmp(&(
            &b.family,
            &b.dataset_key,
            &b.provider_surface,
        ))
    });
    Ok(LedgerReport {
        generated_at_utc: Utc::now().to_rfc3339(),
        route_count: rows.len(),
        status_counts,
        rows,
    })
}

fn count_matches(repo_root: &Path, patterns: &[String]) -> Result<usize> {
    let mut count = 0usize;
    for pattern in patterns {
        let absolute = repo_root.join(pattern);
        let Some(pattern_text) = absolute.to_str() else {
            continue;
        };
        for entry in glob(pattern_text)? {
            if entry.is_ok() {
                count += 1;
            }
        }
    }
    Ok(count)
}

fn derive_status(spec: &RouteSpec, report_present: bool) -> String {
    if spec.parser_ref.trim().is_empty() {
        return "fetch_only".to_string();
    }
    if spec.analysis_binaries.is_empty() {
        return "parse_only".to_string();
    }
    if report_present {
        return "fully_analyzed".to_string();
    }
    "analyzable".to_string()
}

fn render_markdown(report: &LedgerReport) -> String {
    let mut out = String::new();
    out.push_str("# Dataset Routing Ledger\n\n");
    out.push_str(&format!(
        "Generated at `{}` with {} routes.\n\n",
        report.generated_at_utc, report.route_count
    ));
    out.push_str("## Status Counts\n\n");
    for (status, count) in &report.status_counts {
        out.push_str(&format!("- `{status}`: {count}\n"));
    }
    out.push_str("\n## Routes\n\n");
    out.push_str(
        "| Family | Dataset | Provider | Parser | Analyses | Cache | Reports | Status |\n",
    );
    out.push_str("| --- | --- | --- | --- | --- | --- | --- | --- |\n");
    for row in &report.rows {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} ({}) | {} ({}) | {} |\n",
            row.family,
            row.dataset_key,
            row.provider_surface,
            if row.parser_ref.is_empty() {
                "--"
            } else {
                &row.parser_ref
            },
            if row.analysis_binaries.is_empty() {
                "--".to_string()
            } else {
                row.analysis_binaries.join(", ")
            },
            if row.cache_present { "yes" } else { "no" },
            row.cache_match_count,
            if row.report_present { "yes" } else { "no" },
            row.report_match_count,
            row.status,
        ));
    }
    out
}

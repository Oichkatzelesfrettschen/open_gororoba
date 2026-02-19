//! Synchronize top-level counters in registry/project.toml with canonical registries.
//!
//! Managed counters in `[project]`:
//! - claim_count
//! - insight_count
//! - experiment_count
//! - complete_experiment_count
//! - binary_count

use std::collections::HashSet;
use std::path::PathBuf;

use clap::Parser;
use regex::Regex;

#[derive(Parser)]
#[command(name = "project-counter-sync")]
#[command(about = "Synchronize [project] counters in registry/project.toml")]
struct Args {
    /// Path to registry/project.toml.
    #[arg(long, default_value = "registry/project.toml")]
    project: PathBuf,

    /// Path to registry/claims.toml.
    #[arg(long, default_value = "registry/claims.toml")]
    claims: PathBuf,

    /// Path to registry/insights.toml.
    #[arg(long, default_value = "registry/insights.toml")]
    insights: PathBuf,

    /// Path to registry/experiments.toml.
    #[arg(long, default_value = "registry/experiments.toml")]
    experiments: PathBuf,

    /// Path to registry/binaries.toml.
    #[arg(long, default_value = "registry/binaries.toml")]
    binaries: PathBuf,

    /// Check-only mode: exit non-zero if drift exists.
    #[arg(long, default_value_t = false)]
    check: bool,
}

#[derive(serde::Deserialize)]
struct ClaimsRegistry {
    #[serde(default)]
    claim: Vec<toml::Value>,
}

#[derive(serde::Deserialize)]
struct InsightsRegistry {
    #[serde(default)]
    insight: Vec<toml::Value>,
}

#[derive(serde::Deserialize)]
struct BinariesRegistry {
    #[serde(default)]
    binary: Vec<toml::Value>,
}

#[derive(serde::Deserialize)]
struct ExperimentsRegistry {
    #[serde(default)]
    experiment: Vec<ExperimentRow>,
}

#[derive(serde::Deserialize)]
struct ExperimentRow {
    #[serde(default)]
    status: Option<String>,
}

#[derive(Debug, Clone, Copy)]
struct ProjectCounts {
    claim_count: usize,
    insight_count: usize,
    experiment_count: usize,
    complete_experiment_count: usize,
    binary_count: usize,
}

impl ProjectCounts {
    fn get(self, key: &str) -> Option<usize> {
        match key {
            "claim_count" => Some(self.claim_count),
            "insight_count" => Some(self.insight_count),
            "experiment_count" => Some(self.experiment_count),
            "complete_experiment_count" => Some(self.complete_experiment_count),
            "binary_count" => Some(self.binary_count),
            _ => None,
        }
    }
}

fn read_toml<T: serde::de::DeserializeOwned>(path: &PathBuf) -> Result<T, String> {
    let text =
        std::fs::read_to_string(path).map_err(|err| format!("read {}: {err}", path.display()))?;
    toml::from_str::<T>(&text).map_err(|err| format!("parse {}: {err}", path.display()))
}

fn compute_counts(args: &Args) -> Result<ProjectCounts, String> {
    let claims: ClaimsRegistry = read_toml(&args.claims)?;
    let insights: InsightsRegistry = read_toml(&args.insights)?;
    let experiments: ExperimentsRegistry = read_toml(&args.experiments)?;
    let binaries: BinariesRegistry = read_toml(&args.binaries)?;

    let complete_experiment_count = experiments
        .experiment
        .iter()
        .filter(|entry| {
            let status = entry.status.as_deref().unwrap_or("").trim().to_lowercase();
            status == "completed" || status == "complete" || status == "done"
        })
        .count();

    Ok(ProjectCounts {
        claim_count: claims.claim.len(),
        insight_count: insights.insight.len(),
        experiment_count: experiments.experiment.len(),
        complete_experiment_count,
        binary_count: binaries.binary.len(),
    })
}

fn sync_project_block(
    content: &str,
    counts: ProjectCounts,
) -> Result<(String, Vec<String>), String> {
    let assignment_re = Regex::new(
        r"^(\s*)(claim_count|insight_count|experiment_count|complete_experiment_count|binary_count)\s*=\s*([0-9]+)(\s*(?:#.*)?)$",
    )
    .map_err(|err| format!("compile assignment regex: {err}"))?;

    let mut in_project = false;
    let mut saw_project = false;
    let mut seen_keys: HashSet<String> = HashSet::new();
    let mut changes = Vec::new();
    let mut out_lines = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            in_project = trimmed == "[project]";
            if in_project {
                saw_project = true;
            }
            out_lines.push(line.to_string());
            continue;
        }

        if in_project
            && let Some(caps) = assignment_re.captures(line) {
                let indent = caps.get(1).map(|m| m.as_str()).unwrap_or("");
                let key = caps.get(2).map(|m| m.as_str()).unwrap_or("");
                let current = caps
                    .get(3)
                    .and_then(|m| m.as_str().parse::<usize>().ok())
                    .unwrap_or(usize::MAX);
                let suffix = caps.get(4).map(|m| m.as_str()).unwrap_or("");

                seen_keys.insert(key.to_string());

                let expected = counts
                    .get(key)
                    .ok_or_else(|| format!("internal error: unexpected key {key}"))?;
                if current != expected {
                    changes.push(format!("{key}: {current} -> {expected}"));
                }
                out_lines.push(format!("{indent}{key} = {expected}{suffix}"));
                continue;
            }

        out_lines.push(line.to_string());
    }

    if !saw_project {
        return Err("missing [project] table in project.toml".to_string());
    }

    for key in [
        "claim_count",
        "insight_count",
        "experiment_count",
        "complete_experiment_count",
        "binary_count",
    ] {
        if !seen_keys.contains(key) {
            return Err(format!("missing [project] key: {key}"));
        }
    }

    Ok((out_lines.join("\n"), changes))
}

fn main() {
    let args = Args::parse();

    let counts = match compute_counts(&args) {
        Ok(counts) => counts,
        Err(err) => {
            eprintln!("ERROR: {err}");
            std::process::exit(1);
        }
    };

    let project_content = match std::fs::read_to_string(&args.project) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("ERROR: read {}: {err}", args.project.display());
            std::process::exit(1);
        }
    };

    let (rewritten, changes) = match sync_project_block(&project_content, counts) {
        Ok(out) => out,
        Err(err) => {
            eprintln!("ERROR: {err}");
            std::process::exit(1);
        }
    };

    if changes.is_empty() {
        println!("project-counter-sync: already synchronized");
        return;
    }

    for change in &changes {
        println!("drift: {change}");
    }

    if args.check {
        eprintln!(
            "ERROR: {} counter drift entries found in {}",
            changes.len(),
            args.project.display()
        );
        std::process::exit(1);
    }

    if let Err(err) = std::fs::write(&args.project, format!("{rewritten}\n")) {
        eprintln!("ERROR: write {}: {err}", args.project.display());
        std::process::exit(1);
    }

    println!(
        "project-counter-sync: updated {} counter entries in {}",
        changes.len(),
        args.project.display()
    );
}

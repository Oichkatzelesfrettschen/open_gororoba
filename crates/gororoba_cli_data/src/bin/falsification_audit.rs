use anyhow::{Context, Result};
use chrono::Utc;
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, path::PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "falsification-audit",
    about = "Aggregate heliosphere and catalog counter-tests into a single falsification-survival report"
)]
struct Cli {
    #[arg(long)]
    heliosphere_audit_report: PathBuf,

    #[arg(long)]
    baseline_null_report: PathBuf,

    #[arg(long)]
    challenge_null_report: PathBuf,

    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Debug, Deserialize, Serialize)]
struct HeliosphereAuditReport {
    #[serde(default)]
    verdicts: Vec<HeliosphereVerdictRow>,
    #[serde(default)]
    predictive_counterfactuals: Vec<HeliospherePredictiveRow>,
    #[serde(default)]
    sparse_counterfactuals: Vec<HeliosphereSparseRow>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct HeliosphereVerdictRow {
    falsification_id: String,
    survives_challenge: bool,
    baseline_configuration: String,
    challenger_configuration: String,
    #[serde(default)]
    notes: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct HeliospherePredictiveRow {
    view_mode: String,
    normalization_strategy: String,
    descriptor_profile: String,
    auroc: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct HeliosphereSparseRow {
    normalization_strategy: String,
    descriptor_profile: String,
    event_label_recall: f64,
    active_fraction: f64,
    sparse_bf16_aa_projected_gib: f64,
}

#[derive(Debug, Deserialize)]
struct NullClassificationReport {
    #[serde(default)]
    vector_mode: String,
    #[serde(default)]
    rows: Vec<NullClassificationRow>,
}

#[derive(Debug, Clone, Deserialize)]
struct NullClassificationRow {
    dataset: String,
    classification: String,
    #[serde(default)]
    notes: Vec<String>,
}

#[derive(Debug, Serialize)]
struct CatalogVerdictRow {
    dataset: String,
    baseline_classification: String,
    challenge_classification: String,
    survives_challenge: bool,
    notes: Vec<String>,
}

#[derive(Debug, Serialize)]
struct Report {
    generated_at_utc: String,
    heliosphere_audit_report: String,
    baseline_null_report: String,
    challenge_null_report: String,
    heliosphere_verdicts: Vec<HeliosphereVerdictRow>,
    catalog_verdicts: Vec<CatalogVerdictRow>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let out = cli.out.unwrap_or_else(|| {
        PathBuf::from("reports").join(format!(
            "falsification_audit_{}.toml",
            Utc::now().date_naive()
        ))
    });

    let heliosphere: HeliosphereAuditReport = toml::from_str(
        &fs::read_to_string(&cli.heliosphere_audit_report)
            .with_context(|| format!("read {}", cli.heliosphere_audit_report.display()))?,
    )?;
    let baseline: NullClassificationReport = toml::from_str(
        &fs::read_to_string(&cli.baseline_null_report)
            .with_context(|| format!("read {}", cli.baseline_null_report.display()))?,
    )?;
    let challenge: NullClassificationReport = toml::from_str(
        &fs::read_to_string(&cli.challenge_null_report)
            .with_context(|| format!("read {}", cli.challenge_null_report.display()))?,
    )?;

    let challenge_rows = challenge
        .rows
        .into_iter()
        .map(|row| (normalize_dataset(&row.dataset), row))
        .collect::<BTreeMap<_, _>>();
    let mut catalog_verdicts = baseline
        .rows
        .into_iter()
        .map(|row| {
            let challenged = challenge_rows
                .get(&normalize_dataset(&row.dataset))
                .cloned()
                .unwrap_or(NullClassificationRow {
                    dataset: row.dataset.clone(),
                    classification: "inconclusive".to_string(),
                    notes: vec!["dataset_missing_from_challenge_report".to_string()],
                });
            let survives = row.classification == "archive_structure_null"
                && challenged.classification == "archive_structure_null";
            let mut notes = row.notes.clone();
            notes.extend(challenged.notes.clone());
            if !challenge.vector_mode.is_empty() {
                notes.push(format!("challenge_vector_mode={}", challenge.vector_mode));
            }
            CatalogVerdictRow {
                dataset: row.dataset,
                baseline_classification: row.classification,
                challenge_classification: challenged.classification,
                survives_challenge: survives,
                notes,
            }
        })
        .collect::<Vec<_>>();
    catalog_verdicts.sort_by(|a, b| a.dataset.cmp(&b.dataset));

    let heliosphere_verdicts = if heliosphere.predictive_counterfactuals.is_empty()
        && heliosphere.sparse_counterfactuals.is_empty()
    {
        heliosphere.verdicts
    } else {
        recompute_heliosphere_verdicts(
            &heliosphere.predictive_counterfactuals,
            &heliosphere.sparse_counterfactuals,
        )
    };

    let report = Report {
        generated_at_utc: Utc::now().to_rfc3339(),
        heliosphere_audit_report: cli.heliosphere_audit_report.display().to_string(),
        baseline_null_report: cli.baseline_null_report.display().to_string(),
        challenge_null_report: cli.challenge_null_report.display().to_string(),
        heliosphere_verdicts,
        catalog_verdicts,
    };
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&out, toml::to_string_pretty(&report)?)?;
    println!("out = {}", out.display());
    Ok(())
}

fn normalize_dataset(value: &str) -> String {
    value
        .trim()
        .to_ascii_lowercase()
        .replace([' ', '_', '/'], "-")
}

fn recompute_heliosphere_verdicts(
    predictive: &[HeliospherePredictiveRow],
    sparse: &[HeliosphereSparseRow],
) -> Vec<HeliosphereVerdictRow> {
    let best_raw = predictive
        .iter()
        .filter(|row| row.view_mode == "raw")
        .max_by(|a, b| a.auroc.total_cmp(&b.auroc))
        .cloned();
    let best_normalized = predictive
        .iter()
        .filter(|row| row.view_mode == "normalized")
        .max_by(|a, b| a.auroc.total_cmp(&b.auroc))
        .cloned();
    let best_invariant_predictive = predictive
        .iter()
        .filter(|row| row.descriptor_profile == "invariants_only")
        .max_by(|a, b| a.auroc.total_cmp(&b.auroc))
        .cloned();
    let best_algebra_predictive = predictive
        .iter()
        .filter(|row| row.descriptor_profile != "invariants_only")
        .max_by(|a, b| a.auroc.total_cmp(&b.auroc))
        .cloned();
    let best_invariant_sparse = sparse
        .iter()
        .filter(|row| row.descriptor_profile == "invariants_only")
        .filter(|row| row.sparse_bf16_aa_projected_gib <= 12.0)
        .max_by(compare_sparse_rows)
        .cloned();
    let best_algebra_sparse = sparse
        .iter()
        .filter(|row| row.descriptor_profile != "invariants_only")
        .filter(|row| row.sparse_bf16_aa_projected_gib <= 12.0)
        .max_by(compare_sparse_rows)
        .cloned();
    let mut out = Vec::new();
    if let (Some(raw), Some(normalized)) = (best_raw, best_normalized) {
        out.push(HeliosphereVerdictRow {
            falsification_id: "normalized_underperforms_raw".to_string(),
            survives_challenge: normalized.auroc <= raw.auroc,
            baseline_configuration: predictive_label(&raw),
            challenger_configuration: predictive_label(&normalized),
            notes: vec![format!(
                "best_raw_auroc={:.6}, best_normalized_auroc={:.6}",
                raw.auroc, normalized.auroc
            )],
        });
    }
    if let (Some(invariant), Some(algebra)) = (best_invariant_predictive, best_algebra_predictive) {
        out.push(HeliosphereVerdictRow {
            falsification_id: "algebra_adds_no_predictive_gain".to_string(),
            survives_challenge: algebra.auroc <= invariant.auroc,
            baseline_configuration: predictive_label(&invariant),
            challenger_configuration: predictive_label(&algebra),
            notes: vec![format!(
                "best_invariant_auroc={:.6}, best_algebra_auroc={:.6}",
                invariant.auroc, algebra.auroc
            )],
        });
    }
    if let (Some(invariant), Some(algebra)) = (best_invariant_sparse, best_algebra_sparse) {
        out.push(HeliosphereVerdictRow {
            falsification_id: "algebra_adds_no_sparse_gain".to_string(),
            survives_challenge: !sparse_beats(&algebra, &invariant),
            baseline_configuration: sparse_label(&invariant),
            challenger_configuration: sparse_label(&algebra),
            notes: vec![format!(
                "best_invariant_recall={:.6}, best_algebra_recall={:.6}, best_invariant_gib={:.6}, best_algebra_gib={:.6}",
                invariant.event_label_recall,
                algebra.event_label_recall,
                invariant.sparse_bf16_aa_projected_gib,
                algebra.sparse_bf16_aa_projected_gib
            )],
        });
    }
    out
}

fn predictive_label(row: &HeliospherePredictiveRow) -> String {
    format!(
        "{}:{}:{}",
        row.view_mode, row.normalization_strategy, row.descriptor_profile
    )
}

fn sparse_label(row: &HeliosphereSparseRow) -> String {
    format!("{}:{}", row.normalization_strategy, row.descriptor_profile)
}

fn compare_sparse_rows(a: &&HeliosphereSparseRow, b: &&HeliosphereSparseRow) -> std::cmp::Ordering {
    a.event_label_recall
        .total_cmp(&b.event_label_recall)
        .then_with(|| b.active_fraction.total_cmp(&a.active_fraction))
}

fn sparse_beats(a: &HeliosphereSparseRow, b: &HeliosphereSparseRow) -> bool {
    a.event_label_recall > b.event_label_recall
        || ((a.event_label_recall - b.event_label_recall).abs() < 1.0e-9
            && a.active_fraction < b.active_fraction)
}

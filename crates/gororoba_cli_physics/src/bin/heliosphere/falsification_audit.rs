use anyhow::Result;
use chrono::Utc;
use clap::Args;
use gororoba_cli_physics::heliosphere_eval::{
    BinaryMetrics, CounterfactualPredictiveSummary, CounterfactualSparseSummary,
    build_labeled_samples, evaluate_predictive_counterfactuals, evaluate_predictive_models,
    load_heliosphere_rows, summarize_sparse_policies, summarize_sparse_policy_counterfactuals,
};
use serde::Serialize;
use std::{fs, path::PathBuf};

#[derive(Args, Debug)]
pub struct Cli {
    #[arg(long)]
    cube_csv: PathBuf,

    #[arg(long, default_value = ".")]
    repo_root: PathBuf,

    #[arg(long, default_value_t = 24)]
    horizon_hours: i64,

    #[arg(long, default_value_t = 1024)]
    grid: usize,

    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct VerdictRow {
    falsification_id: String,
    survives_challenge: bool,
    baseline_configuration: String,
    challenger_configuration: String,
    notes: Vec<String>,
}

#[derive(Debug, Serialize)]
struct Report {
    generated_at_utc: String,
    cube_csv: String,
    horizon_hours: i64,
    grid: usize,
    labeled_sample_count: usize,
    positive_sample_count: usize,
    baseline_models: Vec<BinaryMetrics>,
    predictive_counterfactuals: Vec<CounterfactualPredictiveSummary>,
    baseline_sparse_policies: Vec<gororoba_cli_physics::heliosphere_eval::SparseMaskSummary>,
    sparse_counterfactuals: Vec<CounterfactualSparseSummary>,
    verdicts: Vec<VerdictRow>,
    notes: Vec<String>,
}

pub fn run(cli: Cli) -> Result<()> {
    let out = cli.out.unwrap_or_else(|| {
        let cube_name = cli
            .cube_csv
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("cube");
        PathBuf::from("reports").join(format!(
            "heliosphere_falsification_audit_{}_{}.toml",
            cube_name,
            Utc::now().date_naive()
        ))
    });
    let rows = load_heliosphere_rows(&cli.cube_csv)?;
    let cache_root = cli.repo_root.join("data/external");
    let (samples, _) = build_labeled_samples(&rows, &cache_root, cli.horizon_hours)?;
    let positive_sample_count = samples
        .iter()
        .filter(|sample| sample.label_positive)
        .count();
    if positive_sample_count == 0 {
        anyhow::bail!(
            "no official positive windows overlapped {}; choose a different cube or label horizon",
            cli.cube_csv.display()
        );
    }

    let baseline_models = evaluate_predictive_models(&samples)?;
    let predictive_counterfactuals = evaluate_predictive_counterfactuals(&samples)?;
    let baseline_sparse_policies =
        summarize_sparse_policies(&rows, &cache_root, cli.horizon_hours, cli.grid)?;
    let sparse_counterfactuals =
        summarize_sparse_policy_counterfactuals(&rows, &cache_root, cli.horizon_hours, cli.grid)?;
    let verdicts = build_verdicts(&predictive_counterfactuals, &sparse_counterfactuals);

    let report = Report {
        generated_at_utc: Utc::now().to_rfc3339(),
        cube_csv: cli.cube_csv.display().to_string(),
        horizon_hours: cli.horizon_hours,
        grid: cli.grid,
        labeled_sample_count: samples.len(),
        positive_sample_count,
        baseline_models,
        predictive_counterfactuals,
        baseline_sparse_policies,
        sparse_counterfactuals,
        verdicts,
        notes: vec![
            "This audit does not assume the first falsification was correct; it re-challenges normalized, algebraic, and sparse-policy negatives with nearby alternatives."
                .to_string(),
            "A falsification only survives here if the best challenger still fails to beat the baseline on the same task."
                .to_string(),
        ],
    };
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&out, toml::to_string_pretty(&report)?)?;
    println!("samples = {}", report.labeled_sample_count);
    println!("positives = {}", report.positive_sample_count);
    println!("out = {}", out.display());
    Ok(())
}

fn build_verdicts(
    predictive: &[CounterfactualPredictiveSummary],
    sparse: &[CounterfactualSparseSummary],
) -> Vec<VerdictRow> {
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
        .cloned()
        .max_by(compare_sparse_rows);
    let best_algebra_sparse = sparse
        .iter()
        .filter(|row| row.descriptor_profile != "invariants_only")
        .filter(|row| row.sparse_bf16_aa_projected_gib <= 12.0)
        .cloned()
        .max_by(compare_sparse_rows);

    let mut verdicts = Vec::new();
    if let (Some(raw), Some(normalized)) = (best_raw, best_normalized) {
        verdicts.push(VerdictRow {
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
        verdicts.push(VerdictRow {
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
        let algebra_beats = sparse_beats(&algebra, &invariant);
        verdicts.push(VerdictRow {
            falsification_id: "algebra_adds_no_sparse_gain".to_string(),
            survives_challenge: !algebra_beats,
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
    verdicts
}

fn predictive_label(row: &CounterfactualPredictiveSummary) -> String {
    format!(
        "{}:{}:{}",
        row.view_mode, row.normalization_strategy, row.descriptor_profile
    )
}

fn sparse_label(row: &CounterfactualSparseSummary) -> String {
    format!("{}:{}", row.normalization_strategy, row.descriptor_profile)
}

fn compare_sparse_rows(
    a: &CounterfactualSparseSummary,
    b: &CounterfactualSparseSummary,
) -> std::cmp::Ordering {
    a.event_label_recall
        .total_cmp(&b.event_label_recall)
        .then_with(|| b.active_fraction.total_cmp(&a.active_fraction))
}

fn sparse_beats(a: &CounterfactualSparseSummary, b: &CounterfactualSparseSummary) -> bool {
    a.event_label_recall > b.event_label_recall
        || ((a.event_label_recall - b.event_label_recall).abs() < 1.0e-9
            && a.active_fraction < b.active_fraction)
}

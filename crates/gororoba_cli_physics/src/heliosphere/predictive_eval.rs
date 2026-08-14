use anyhow::Result;
use chrono::Utc;
use clap::Args;
use crate::heliosphere_eval::{
    BinaryMetrics, HELIOSPHERE_DESCRIPTOR_CHANNEL_NAMES, LabelCoverageRow, MissionSplitSummary,
    build_labeled_samples, evaluate_predictive_models, load_heliosphere_rows,
    summarize_label_coverage,
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

    #[arg(long)]
    out: Option<PathBuf>,

    #[arg(long)]
    coverage_out: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct Report {
    generated_at_utc: String,
    cube_csv: String,
    row_count: usize,
    labeled_sample_count: usize,
    positive_sample_count: usize,
    horizon_hours: i64,
    invariant_channel_names: Vec<String>,
    descriptor_channel_names: Vec<String>,
    mission_splits: Vec<MissionSplitSummary>,
    models: Vec<BinaryMetrics>,
    notes: Vec<String>,
}

#[derive(Debug, Serialize)]
struct CoverageReport {
    generated_at_utc: String,
    cube_csv: String,
    horizons_hours: Vec<i64>,
    rows: Vec<LabelCoverageRow>,
    notes: Vec<String>,
}

pub fn run(cli: Cli) -> Result<()> {
    let out = cli.out.unwrap_or_else(|| {
        PathBuf::from("reports").join(format!(
            "heliosphere_predictive_eval_{}.toml",
            Utc::now().date_naive()
        ))
    });
    let coverage_out = cli.coverage_out.unwrap_or_else(|| {
        let cube_name = cli
            .cube_csv
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("cube");
        PathBuf::from("reports").join(format!(
            "heliosphere_label_coverage_{}_{}.toml",
            cube_name,
            Utc::now().date_naive()
        ))
    });
    let rows = load_heliosphere_rows(&cli.cube_csv)?;
    let cache_root = cli.repo_root.join("data/external");
    let coverage_rows = summarize_label_coverage(&rows, &cache_root)?;
    let (samples, split_summary) = build_labeled_samples(&rows, &cache_root, cli.horizon_hours)?;
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
    let models = evaluate_predictive_models(&samples)?;
    let report = Report {
        generated_at_utc: Utc::now().to_rfc3339(),
        cube_csv: cli.cube_csv.display().to_string(),
        row_count: rows.len(),
        labeled_sample_count: samples.len(),
        positive_sample_count,
        horizon_hours: cli.horizon_hours,
        invariant_channel_names: data_core::HELIOSPHERE_INVARIANT_CHANNEL_NAMES
            .iter()
            .map(|value| (*value).to_string())
            .collect(),
        descriptor_channel_names: HELIOSPHERE_DESCRIPTOR_CHANNEL_NAMES
            .iter()
            .map(|value| (*value).to_string())
            .collect(),
        mission_splits: split_summary,
        models,
        notes: vec![
            "Official labels are fetched from NASA DONKI and cached under data/external/space_weather/donki."
                .to_string(),
            "Prediction windows span [event_time - horizon_hours, event_time].".to_string(),
            "Models are reported for both raw and cross-mission-normalized invariant spaces."
                .to_string(),
            "Mission-targeted DONKI ENLIL arrivals are official forecast-target windows; CCMC Scoreboard arrivals remain the observed Earth/L1 arrival source."
                .to_string(),
        ],
    };
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = coverage_out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&out, toml::to_string_pretty(&report)?)?;
    let coverage_report = CoverageReport {
        generated_at_utc: Utc::now().to_rfc3339(),
        cube_csv: cli.cube_csv.display().to_string(),
        horizons_hours: vec![6, 12, 24],
        rows: coverage_rows,
        notes: vec![
            "Coverage is strict and official-only: DONKI onset families, DONKI CME/ENLIL target windows, and CCMC Scoreboard arrivals/residuals."
                .to_string(),
            "A forecast-target overlap is not the same thing as a validated observed arrival.".to_string(),
        ],
    };
    fs::write(&coverage_out, toml::to_string_pretty(&coverage_report)?)?;
    println!("samples = {}", report.labeled_sample_count);
    println!("positives = {}", report.positive_sample_count);
    println!("out = {}", out.display());
    println!("coverage_out = {}", coverage_out.display());
    Ok(())
}

use anyhow::Result;
use chrono::Utc;
use clap::Parser;
use gororoba_cli_physics::heliosphere_eval::{
    BinaryMetrics, HELIOSPHERE_DESCRIPTOR_CHANNEL_NAMES, MissionSplitSummary,
    build_labeled_samples, evaluate_predictive_models, load_heliosphere_rows,
};
use serde::Serialize;
use std::{fs, path::PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "heliosphere-predictive-eval",
    about = "Evaluate heliosphere invariant and algebraic predictors against official DONKI event windows"
)]
struct Cli {
    #[arg(long)]
    cube_csv: PathBuf,

    #[arg(long, default_value = ".")]
    repo_root: PathBuf,

    #[arg(long, default_value_t = 24)]
    horizon_hours: i64,

    #[arg(long)]
    out: Option<PathBuf>,
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

fn main() -> Result<()> {
    let cli = Cli::parse();
    let out = cli.out.unwrap_or_else(|| {
        PathBuf::from("reports").join(format!(
            "heliosphere_predictive_eval_{}.toml",
            Utc::now().date_naive()
        ))
    });
    let rows = load_heliosphere_rows(&cli.cube_csv)?;
    let cache_root = cli.repo_root.join("data/external");
    let (samples, split_summary) = build_labeled_samples(&rows, &cache_root, cli.horizon_hours)?;
    let positive_sample_count = samples.iter().filter(|sample| sample.label_positive).count();
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
            "Algebra descriptors are used as adaptive geometry features, not as standalone physics evidence."
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

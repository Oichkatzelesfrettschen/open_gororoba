use anyhow::Result;
use chrono::Utc;
use clap::Parser;
use gororoba_cli_physics::heliosphere_eval::{
    HELIOSPHERE_DESCRIPTOR_CHANNEL_NAMES, build_labeled_samples, load_heliosphere_rows,
    summarize_cross_mission_invariance,
};
use serde::Serialize;
use std::{fs, path::PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "heliosphere-cross-mission-invariance",
    about = "Measure cross-mission stability of heliosphere invariant and algebraic descriptors"
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
    labeled_sample_count: usize,
    positive_sample_count: usize,
    horizon_hours: i64,
    invariant_channel_names: Vec<String>,
    descriptor_channel_names: Vec<String>,
    missions: Vec<gororoba_cli_physics::heliosphere_eval::MissionInvarianceSummary>,
    notes: Vec<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let out = cli.out.unwrap_or_else(|| {
        PathBuf::from("reports").join(format!(
            "heliosphere_cross_mission_invariance_{}.toml",
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
    let missions = summarize_cross_mission_invariance(&samples);
    let report = Report {
        generated_at_utc: Utc::now().to_rfc3339(),
        cube_csv: cli.cube_csv.display().to_string(),
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
        missions,
        notes: vec![
            "Leave-one-mission-out cosine compares each mission's positive-window descriptor mean against the positive-window mean from the remaining missions."
                .to_string(),
            "The report now includes raw and uncertainty-aware normalized sections in the same mission table."
                .to_string(),
            "High stability only becomes interesting if it survives uncertainty-aware normalization and official-label filtering."
                .to_string(),
        ],
    };
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&out, toml::to_string_pretty(&report)?)?;
    println!("missions = {}", report.missions.len());
    println!("out = {}", out.display());
    Ok(())
}

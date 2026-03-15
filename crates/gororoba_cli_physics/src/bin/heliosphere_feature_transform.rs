use anyhow::{Context, Result};
use chrono::Utc;
use clap::Parser;
use csv::{ReaderBuilder, WriterBuilder};
use data_core::{HeliosphereFeatureRow, HeliosphereTransformMode, transform_feature_rows};
use serde::Serialize;
use std::{fs, path::PathBuf, str::FromStr};

#[derive(Parser, Debug)]
#[command(
    name = "heliosphere-feature-transform",
    about = "Normalize and/or difference heliosphere feature cubes without dropping rows"
)]
struct Cli {
    #[arg(long)]
    cube_csv: PathBuf,

    #[arg(long)]
    mode: String,

    #[arg(long)]
    window: Option<String>,

    #[arg(long)]
    out_csv: Option<PathBuf>,

    #[arg(long)]
    out_manifest: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct Report {
    generated_at_utc: String,
    cube_csv: String,
    selected_window: Option<String>,
    mode: String,
    input_rows: usize,
    output_rows: usize,
    input_active_fraction_estimate: f64,
    output_active_fraction_estimate: f64,
    groups: usize,
    notes: Vec<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let mode = parse_mode(&cli.mode)?;
    let date = Utc::now().date_naive();
    let out_csv = cli.out_csv.unwrap_or_else(|| {
        let window = cli.window.as_deref().unwrap_or("all");
        PathBuf::from("reports").join(format!(
            "heliosphere_feature_transform_{window}_{}_{}.csv",
            cli.mode, date
        ))
    });
    let out_manifest = cli.out_manifest.unwrap_or_else(|| {
        let window = cli.window.as_deref().unwrap_or("all");
        PathBuf::from("reports").join(format!(
            "heliosphere_feature_transform_{window}_{}_{}.toml",
            cli.mode, date
        ))
    });

    let rows = load_rows(&cli.cube_csv)?;
    let selected: Vec<HeliosphereFeatureRow> = rows
        .into_iter()
        .filter(|row| {
            cli.window
                .as_ref()
                .map(|value| row.window_name == *value)
                .unwrap_or(true)
        })
        .collect();
    let transformed = transform_feature_rows(&selected, mode);

    if let Some(parent) = out_csv.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut writer = WriterBuilder::new()
        .has_headers(true)
        .from_path(&out_csv)
        .with_context(|| format!("create {}", out_csv.display()))?;
    for row in &transformed {
        writer
            .serialize(row)
            .with_context(|| format!("write {}", out_csv.display()))?;
    }
    writer.flush()?;

    let report = Report {
        generated_at_utc: Utc::now().to_rfc3339(),
        cube_csv: cli.cube_csv.display().to_string(),
        selected_window: cli.window.clone(),
        mode: cli.mode.clone(),
        input_rows: selected.len(),
        output_rows: transformed.len(),
        input_active_fraction_estimate: estimate_active_fraction(&selected),
        output_active_fraction_estimate: estimate_active_fraction(&transformed),
        groups: count_groups(&transformed),
        notes: vec![
            "Rows are grouped by window_name/mission/product before transformation.".to_string(),
            "Differencing keeps row count by zero-filling the first row in each group.".to_string(),
            "Differenced-normalized applies differencing first and z-score normalization second."
                .to_string(),
        ],
    };
    if let Some(parent) = out_manifest.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&out_manifest, toml::to_string_pretty(&report)?)
        .with_context(|| format!("write {}", out_manifest.display()))?;
    println!("input_rows = {}", report.input_rows);
    println!("output_rows = {}", report.output_rows);
    println!(
        "active_fraction_estimate: {:.5} -> {:.5}",
        report.input_active_fraction_estimate, report.output_active_fraction_estimate
    );
    println!("out_csv = {}", out_csv.display());
    println!("out_manifest = {}", out_manifest.display());
    Ok(())
}

fn parse_mode(value: &str) -> Result<HeliosphereTransformMode> {
    HeliosphereTransformMode::from_str(value).map_err(anyhow::Error::msg)
}

fn load_rows(path: &PathBuf) -> Result<Vec<HeliosphereFeatureRow>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .with_context(|| format!("open {}", path.display()))?;
    let mut rows = Vec::new();
    for row in reader.deserialize() {
        rows.push(row.with_context(|| format!("deserialize {}", path.display()))?);
    }
    Ok(rows)
}

fn estimate_active_fraction(rows: &[HeliosphereFeatureRow]) -> f64 {
    let energies: Vec<f64> = rows.iter().map(|row| row.signal_energy()).collect();
    let finite: Vec<f64> = energies
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.is_empty() {
        return 0.0;
    }
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    let var = finite
        .iter()
        .map(|value| {
            let delta = *value - mean;
            delta * delta
        })
        .sum::<f64>()
        / finite.len() as f64;
    let std = var.sqrt();
    let active = finite
        .iter()
        .filter(|value| **value > mean + 0.5 * std)
        .count();
    (active as f64 / finite.len() as f64).clamp(0.0, 1.0)
}

fn count_groups(rows: &[HeliosphereFeatureRow]) -> usize {
    use std::collections::BTreeSet;
    rows.iter()
        .map(|row| {
            (
                row.window_name.clone(),
                row.mission.clone(),
                row.product.clone(),
            )
        })
        .collect::<BTreeSet<_>>()
        .len()
}

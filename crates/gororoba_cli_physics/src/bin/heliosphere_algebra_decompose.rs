use anyhow::{Context, Result};
use cd_kernel::{cd_associator_norm, cd_norm_sq};
use chrono::Utc;
use clap::Parser;
use csv::ReaderBuilder;
use data_core::{HELIOSPHERE_FEATURE_DIM, HeliosphereFeatureRow};
use serde::Serialize;
use std::{collections::BTreeMap, fs, path::PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "heliosphere-algebra-decompose",
    about = "Compute algebraic descriptors from heliosphere feature cubes"
)]
struct Cli {
    #[arg(long)]
    cube_csv: PathBuf,

    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct GroupSummary {
    window_name: String,
    mission: String,
    product: String,
    row_count: usize,
    mean_norm_sq: f64,
    mean_signal_energy: f64,
    mean_associator_norm: f64,
    max_associator_norm: f64,
    dominant_channel: String,
    dominant_channel_mean_abs: f64,
}

#[derive(Debug, Serialize)]
struct Report {
    generated_at_utc: String,
    cube_csv: String,
    feature_dim: usize,
    group_count: usize,
    groups: Vec<GroupSummary>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let out = cli.out.unwrap_or_else(|| {
        PathBuf::from("reports").join(format!(
            "heliosphere_algebra_decompose_{}.toml",
            Utc::now().date_naive()
        ))
    });
    let rows = load_rows(&cli.cube_csv)?;
    let groups = summarize_groups(&rows);
    let report = Report {
        generated_at_utc: Utc::now().to_rfc3339(),
        cube_csv: cli.cube_csv.display().to_string(),
        feature_dim: HELIOSPHERE_FEATURE_DIM,
        group_count: groups.len(),
        groups,
    };
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&out, toml::to_string_pretty(&report)?)
        .with_context(|| format!("write {}", out.display()))?;
    println!("group_count = {}", report.group_count);
    println!("out = {}", out.display());
    Ok(())
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

fn summarize_groups(rows: &[HeliosphereFeatureRow]) -> Vec<GroupSummary> {
    let mut grouped: BTreeMap<(String, String, String), Vec<&HeliosphereFeatureRow>> =
        BTreeMap::new();
    for row in rows {
        grouped
            .entry((
                row.window_name.clone(),
                row.mission.clone(),
                row.product.clone(),
            ))
            .or_default()
            .push(row);
    }

    grouped
        .into_iter()
        .map(|((window_name, mission, product), group)| {
            let vectors: Vec<[f64; HELIOSPHERE_FEATURE_DIM]> =
                group.iter().map(|row| row.algebra_vector()).collect();
            let norms: Vec<f64> = vectors.iter().map(|vector| cd_norm_sq(vector)).collect();
            let signal_energy: Vec<f64> = group.iter().map(|row| row.signal_energy()).collect();
            let mut associators = Vec::new();
            for triple in vectors.windows(3) {
                associators.push(cd_associator_norm(&triple[0], &triple[1], &triple[2]));
            }

            let channel_means = mean_abs_channels(&vectors);
            let (dominant_idx, dominant_value) = channel_means
                .iter()
                .copied()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .unwrap_or((HELIOSPHERE_FEATURE_DIM - 1, 0.0));

            GroupSummary {
                window_name,
                mission,
                product,
                row_count: group.len(),
                mean_norm_sq: mean(&norms),
                mean_signal_energy: mean(&signal_energy),
                mean_associator_norm: mean(&associators),
                max_associator_norm: associators.into_iter().fold(0.0, f64::max),
                dominant_channel: data_core::HELIOSPHERE_CHANNEL_NAMES[dominant_idx].to_string(),
                dominant_channel_mean_abs: dominant_value,
            }
        })
        .collect()
}

fn mean(values: &[f64]) -> f64 {
    let finite: Vec<f64> = values.iter().copied().filter(|value| value.is_finite()).collect();
    if finite.is_empty() {
        return f64::NAN;
    }
    finite.iter().sum::<f64>() / finite.len() as f64
}

fn mean_abs_channels(vectors: &[[f64; HELIOSPHERE_FEATURE_DIM]]) -> [f64; HELIOSPHERE_FEATURE_DIM] {
    let mut sums = [0.0_f64; HELIOSPHERE_FEATURE_DIM];
    let mut counts = [0usize; HELIOSPHERE_FEATURE_DIM];
    for vector in vectors {
        for (idx, value) in vector.iter().enumerate() {
            if value.is_finite() {
                sums[idx] += value.abs();
                counts[idx] += 1;
            }
        }
    }
    let mut out = [0.0_f64; HELIOSPHERE_FEATURE_DIM];
    for idx in 0..HELIOSPHERE_FEATURE_DIM {
        if counts[idx] > 0 {
            out[idx] = sums[idx] / counts[idx] as f64;
        }
    }
    out
}

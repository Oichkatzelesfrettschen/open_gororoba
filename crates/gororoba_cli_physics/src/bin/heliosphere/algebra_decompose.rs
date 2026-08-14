use anyhow::{Context, Result};
use cd_kernel::{cd_associator_norm, cd_norm_sq};
use chrono::Utc;
use clap::Args;
use csv::ReaderBuilder;
use data_core::{
    HELIOSPHERE_DYNAMIC_CHANNEL_NAMES, HELIOSPHERE_FEATURE_DIM, HeliosphereFeatureRow,
};
use serde::Serialize;
use std::{collections::BTreeMap, fs, path::PathBuf, str::FromStr};

#[derive(Args, Debug)]
pub struct Cli {
    #[arg(long)]
    cube_csv: PathBuf,

    #[arg(long, default_value = "legacy")]
    descriptor_mode: String,

    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DescriptorMode {
    Legacy,
    DynamicBiasFree,
    Both,
}

impl FromStr for DescriptorMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "legacy" => Ok(Self::Legacy),
            "dynamic-bias-free" => Ok(Self::DynamicBiasFree),
            "both" => Ok(Self::Both),
            other => Err(format!(
                "unsupported descriptor mode '{other}'; expected legacy, dynamic-bias-free, or both"
            )),
        }
    }
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
struct ModeSummary {
    descriptor_mode: String,
    groups: Vec<GroupSummary>,
}

#[derive(Debug, Serialize)]
struct Report {
    generated_at_utc: String,
    cube_csv: String,
    feature_dim: usize,
    mode_count: usize,
    modes: Vec<ModeSummary>,
}

pub fn run(cli: Cli) -> Result<()> {
    let descriptor_mode =
        DescriptorMode::from_str(&cli.descriptor_mode).map_err(anyhow::Error::msg)?;
    let out = cli.out.unwrap_or_else(|| {
        PathBuf::from("reports").join(format!(
            "heliosphere_algebra_decompose_{}.toml",
            Utc::now().date_naive()
        ))
    });
    let rows = load_rows(&cli.cube_csv)?;
    let modes = summarize_modes(&rows, descriptor_mode);
    let report = Report {
        generated_at_utc: Utc::now().to_rfc3339(),
        cube_csv: cli.cube_csv.display().to_string(),
        feature_dim: HELIOSPHERE_FEATURE_DIM,
        mode_count: modes.len(),
        modes,
    };
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&out, toml::to_string_pretty(&report)?)
        .with_context(|| format!("write {}", out.display()))?;
    println!("mode_count = {}", report.mode_count);
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

fn summarize_modes(rows: &[HeliosphereFeatureRow], mode: DescriptorMode) -> Vec<ModeSummary> {
    match mode {
        DescriptorMode::Legacy => vec![ModeSummary {
            descriptor_mode: "legacy".to_string(),
            groups: summarize_groups(rows, DescriptorMode::Legacy),
        }],
        DescriptorMode::DynamicBiasFree => vec![ModeSummary {
            descriptor_mode: "dynamic-bias-free".to_string(),
            groups: summarize_groups(rows, DescriptorMode::DynamicBiasFree),
        }],
        DescriptorMode::Both => vec![
            ModeSummary {
                descriptor_mode: "legacy".to_string(),
                groups: summarize_groups(rows, DescriptorMode::Legacy),
            },
            ModeSummary {
                descriptor_mode: "dynamic-bias-free".to_string(),
                groups: summarize_groups(rows, DescriptorMode::DynamicBiasFree),
            },
        ],
    }
}

fn summarize_groups(rows: &[HeliosphereFeatureRow], mode: DescriptorMode) -> Vec<GroupSummary> {
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
            let vectors: Vec<[f64; HELIOSPHERE_FEATURE_DIM]> = group
                .iter()
                .map(|row| match mode {
                    DescriptorMode::Legacy | DescriptorMode::Both => row.algebra_vector(),
                    DescriptorMode::DynamicBiasFree => row.algebra_vector_dynamic_bias_free(),
                })
                .collect();
            let norms: Vec<f64> = vectors.iter().map(|vector| cd_norm_sq(vector)).collect();
            let signal_energy: Vec<f64> = group.iter().map(|row| row.signal_energy()).collect();
            let mut associators = Vec::new();
            for triple in vectors.windows(3) {
                associators.push(cd_associator_norm(&triple[0], &triple[1], &triple[2]));
            }

            let (dominant_channel, dominant_value) = match mode {
                DescriptorMode::Legacy | DescriptorMode::Both => {
                    let channel_means = mean_abs_channels(&vectors);
                    let (dominant_idx, dominant_value) = channel_means
                        .iter()
                        .copied()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                        .unwrap_or((HELIOSPHERE_FEATURE_DIM - 1, 0.0));
                    (
                        data_core::HELIOSPHERE_CHANNEL_NAMES[dominant_idx].to_string(),
                        dominant_value,
                    )
                }
                DescriptorMode::DynamicBiasFree => dominant_dynamic_channel(&group),
            };

            GroupSummary {
                window_name,
                mission,
                product,
                row_count: group.len(),
                mean_norm_sq: mean(&norms),
                mean_signal_energy: mean(&signal_energy),
                mean_associator_norm: mean(&associators),
                max_associator_norm: associators.into_iter().fold(0.0, f64::max),
                dominant_channel,
                dominant_channel_mean_abs: dominant_value,
            }
        })
        .collect()
}

fn dominant_dynamic_channel(group: &[&HeliosphereFeatureRow]) -> (String, f64) {
    let mut sums = [0.0_f64; data_core::HELIOSPHERE_DYNAMIC_DIM];
    let mut counts = [0usize; data_core::HELIOSPHERE_DYNAMIC_DIM];
    for row in group {
        for (idx, value) in row.dynamic_signal_channels().iter().enumerate() {
            if value.is_finite() {
                sums[idx] += value.abs();
                counts[idx] += 1;
            }
        }
    }
    let mut dominant_idx = 0usize;
    let mut dominant_value = f64::NEG_INFINITY;
    for idx in 0..data_core::HELIOSPHERE_DYNAMIC_DIM {
        let mean_abs = if counts[idx] == 0 {
            0.0
        } else {
            sums[idx] / counts[idx] as f64
        };
        if mean_abs.total_cmp(&dominant_value).is_gt() {
            dominant_idx = idx;
            dominant_value = mean_abs;
        }
    }
    (
        HELIOSPHERE_DYNAMIC_CHANNEL_NAMES[dominant_idx].to_string(),
        dominant_value.max(0.0),
    )
}

fn mean(values: &[f64]) -> f64 {
    let finite: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
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

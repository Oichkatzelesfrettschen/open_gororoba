use anyhow::{Context, Result};
use cd_kernel::cd_associator_norm;
use clap::Parser;
use csv::{ReaderBuilder, WriterBuilder};
use data_core::HeliosphereFeatureRow;
use serde::Serialize;
use std::{collections::BTreeMap, path::PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "heliosphere-quench-scan",
    about = "Map the algebraic quenching point across heliocentric distance using Magnetic Takens embedding"
)]
struct Cli {
    #[arg(long)]
    cube_csv: PathBuf,

    #[arg(long)]
    out_csv: PathBuf,

    #[arg(long, default_value_t = 5.0)]
    bin_size_au: f64,
}

#[derive(Debug, Clone, Serialize)]
struct QuenchBin {
    r_center_au: f64,
    mean_associator: f64,
    median_associator: f64,
    max_associator: f64,
    sample_count: usize,
    mission_diversity: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut reader = ReaderBuilder::new()
        .from_path(&cli.cube_csv)
        .with_context(|| format!("open {}", cli.cube_csv.display()))?;

    let mut all_rows = Vec::new();
    for result in reader.deserialize::<HeliosphereFeatureRow>() {
        let r = result?;
        if r.bx.is_finite() && r.by.is_finite() && r.bz.is_finite() && r.b_mag.is_finite() && r.b_mag > 0.0 {
            all_rows.push(r);
        }
    }

    // Group rows by mission+product to compute temporal Takens associators
    let mut mission_groups: BTreeMap<(String, String), Vec<HeliosphereFeatureRow>> = BTreeMap::new();
    for row in all_rows {
        mission_groups.entry((row.mission.clone(), row.product.clone())).or_default().push(row);
    }

    let mut associator_data: Vec<(f64, f64, String)> = Vec::new(); // (r_au, associator, mission)

    println!("[1/2] Computing Takens associators with r_au awareness...");
    for ((mission, _product), rows) in mission_groups {
        if rows.len() < 6 { continue; } // Need at least 4 for v16 + 2 more for triple

        let mut embedded_vectors = Vec::new();
        let mut r_aus = Vec::new();

        for window in rows.windows(4) {
            let mut v16 = [0.0; 16];
            let local_mean_b = (window[0].b_mag + window[1].b_mag + window[2].b_mag + window[3].b_mag) / 4.0;
            if local_mean_b <= 0.0 { continue; }

            for i in 0..4 {
                v16[i * 4 + 0] = window[i].bx / local_mean_b;
                v16[i * 4 + 1] = window[i].by / local_mean_b;
                v16[i * 4 + 2] = window[i].bz / local_mean_b;
                v16[i * 4 + 3] = (window[i].b_mag - local_mean_b) / local_mean_b;
            }
            embedded_vectors.push(v16);
            r_aus.push(window[3].r_au);
        }

        for (i, w) in embedded_vectors.windows(3).enumerate() {
            let norm = cd_associator_norm(&w[0], &w[1], &w[2]);
            if norm.is_finite() {
                associator_data.push((r_aus[i + 2], norm, mission.clone()));
            }
        }
    }

    println!("[2/2] Aggregating {} samples into radial bins...", associator_data.len());
    let mut bins: BTreeMap<i32, Vec<(f64, String)>> = BTreeMap::new();
    for (r, norm, mission) in associator_data {
        let bin_idx = (r / cli.bin_size_au).floor() as i32;
        bins.entry(bin_idx).or_default().push((norm, mission));
    }

    let mut writer = WriterBuilder::new().from_path(&cli.out_csv)?;
    for (bin_idx, samples) in bins {
        let r_center = (bin_idx as f64 + 0.5) * cli.bin_size_au;
        let mut values: Vec<f64> = samples.iter().map(|(v, _)| *v).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let median = values[values.len() / 2];
        let max = *values.last().unwrap_or(&0.0);
        let mission_count = samples.iter().map(|(_, m)| m).collect::<std::collections::HashSet<_>>().len();

        writer.serialize(QuenchBin {
            r_center_au: r_center,
            mean_associator: mean,
            median_associator: median,
            max_associator: max,
            sample_count: values.len(),
            mission_diversity: mission_count,
        })?;
    }

    Ok(())
}

use anyhow::{Context, Result};
use cd_kernel::cd_associator_norm;
use clap::Parser;
use csv::{ReaderBuilder, WriterBuilder};
use data_core::{
    compute_invariant_samples, HeliosphereFeatureRow,
};
use serde::Serialize;
use std::{collections::BTreeMap, path::PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "heliosphere-quench-scan",
    about = "Map the algebraic quenching point across heliocentric distance"
)]
struct Cli {
    #[arg(long)]
    cube_csv: PathBuf,

    #[arg(long)]
    out_csv: PathBuf,

    #[arg(long, default_value_t = 1.0)]
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
        all_rows.push(result?);
    }

    // Group rows by mission+product to compute temporal associators
    let mut mission_groups: BTreeMap<(String, String), Vec<HeliosphereFeatureRow>> = BTreeMap::new();
    for row in all_rows {
        mission_groups.entry((row.mission.clone(), row.product.clone())).or_default().push(row);
    }

    let mut associator_data: Vec<(f64, f64, String)> = Vec::new(); // (r_au, associator, mission)

    println!("[1/2] Computing associators with r_au awareness...");
    for ((mission, _product), rows) in mission_groups {
        let inv_samples = compute_invariant_samples(&rows);
        let vectors: Vec<[f64; 16]> = inv_samples.iter().map(|s| {
            let mut v = [0.0; 16];
            for i in 0..10 { v[i] = s.channels[i]; }
            v
        }).collect();

        let norms: Vec<f64> = vectors.windows(3)
            .map(|w| cd_associator_norm(&w[0], &w[1], &w[2]))
            .collect();

        // Map norms back to average r_au of the triple
        for (i, norm) in norms.into_iter().enumerate() {
            let avg_r = (rows[i].r_au + rows[i+1].r_au + rows[i+2].r_au) / 3.0;
            if norm.is_finite() && avg_r.is_finite() {
                associator_data.push((avg_r, norm, mission.clone()));
            }
        }
    }

    println!("[2/2] Aggregating {} associator samples into radial bins...", associator_data.len());
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

    println!("Quench scan complete: {}", cli.out_csv.display());
    Ok(())
}

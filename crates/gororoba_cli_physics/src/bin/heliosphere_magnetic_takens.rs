use anyhow::{Context, Result};
use cd_kernel::cd_associator_norm;
use clap::Parser;
use csv::{ReaderBuilder, WriterBuilder};
use data_core::HeliosphereFeatureRow;
use serde::Serialize;
use std::{collections::BTreeMap, path::PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "heliosphere-magnetic-takens",
    about = "Takens time-delay embedding of magnetic field into 16D Sedenion space"
)]
struct Cli {
    #[arg(long)]
    cube_csv: PathBuf,

    #[arg(long)]
    out_csv: PathBuf,
}

#[derive(Debug, Clone, Serialize)]
struct TakensBin {
    mission: String,
    product: String,
    r_au: f64,
    mean_associator: f64,
    max_associator: f64,
    sample_count: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut reader = ReaderBuilder::new()
        .from_path(&cli.cube_csv)
        .with_context(|| format!("open {}", cli.cube_csv.display()))?;

    let mut all_rows = Vec::new();
    for result in reader.deserialize::<HeliosphereFeatureRow>() {
        let r = result?;
        // ONLY keep rows with valid magnetic field data
        if r.bx.is_finite() && r.by.is_finite() && r.bz.is_finite() && r.b_mag.is_finite() && r.b_mag > 0.0 {
            all_rows.push(r);
        }
    }

    // Group by mission+product
    let mut groups: BTreeMap<(String, String), Vec<HeliosphereFeatureRow>> = BTreeMap::new();
    for row in all_rows {
        groups
            .entry((row.mission.clone(), row.product.clone()))
            .or_default()
            .push(row);
    }

    let mut results = Vec::new();

    println!("[1/2] Computing Takens 16D embedding (4-step delay) for B-field...");
    for ((mission, product), rows) in groups {
        if rows.len() < 10 { continue; }

        let mut embedded_vectors = Vec::new();
        let mut r_aus = Vec::new();

        // 4-step sliding window to build 16D vector from 4D B-field (Bx, By, Bz, |B|)
        // We use relative fluctuations (dB / mean_B) to remove the 1/r^2 heliospheric scaling
        for window in rows.windows(4) {
            let mut v16 = [0.0; 16];
            let local_mean_b = (window[0].b_mag + window[1].b_mag + window[2].b_mag + window[3].b_mag) / 4.0;
            
            if local_mean_b <= 0.0 { continue; }

            for i in 0..4 {
                // Difference from local mean, normalized by local mean (dimensionless)
                v16[i * 4 + 0] = (window[i].bx) / local_mean_b;
                v16[i * 4 + 1] = (window[i].by) / local_mean_b;
                v16[i * 4 + 2] = (window[i].bz) / local_mean_b;
                v16[i * 4 + 3] = (window[i].b_mag - local_mean_b) / local_mean_b;
            }
            
            embedded_vectors.push(v16);
            r_aus.push(window[3].r_au); // tag with latest distance
        }

        // Now compute Sedenion associator over triples of these 16D states
        let mut associators = Vec::new();
        let mut valid_r = Vec::new();
        for (i, w) in embedded_vectors.windows(3).enumerate() {
            let norm = cd_associator_norm(&w[0], &w[1], &w[2]);
            if norm.is_finite() {
                associators.push(norm);
                valid_r.push(r_aus[i + 2]);
            }
        }

        if associators.is_empty() { continue; }

        let mean = associators.iter().sum::<f64>() / associators.len() as f64;
        let max = associators.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mean_r = valid_r.iter().sum::<f64>() / valid_r.len() as f64;

        results.push(TakensBin {
            mission,
            product,
            r_au: mean_r,
            mean_associator: mean,
            max_associator: max,
            sample_count: associators.len(),
        });
    }

    // Sort by heliocentric distance
    results.sort_by(|a, b| a.r_au.partial_cmp(&b.r_au).unwrap());

    println!("[2/2] Writing clean magnetic quench scan...");
    let mut writer = WriterBuilder::new().from_path(&cli.out_csv)?;
    for res in results {
        writer.serialize(res)?;
    }

    Ok(())
}

use anyhow::{Context, Result};
use clap::Parser;
use csv::{ReaderBuilder, WriterBuilder};
use data_core::{HeliosphereFeatureRow, magnetic_plasma_takens_embed, magnetic_takens_embed};
use serde::Serialize;
use std::{collections::BTreeMap, path::PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "heliosphere-magnetic-takens",
    about = "Takens time-delay embedding of magnetic field into Cayley-Dickson space"
)]
struct Cli {
    #[arg(long)]
    cube_csv: PathBuf,

    #[arg(long)]
    out_csv: PathBuf,

    /// Exclude rows from these missions (repeatable, case-sensitive).
    #[arg(long = "exclude-mission")]
    exclude_missions: Vec<String>,

    /// Embedding dimension (must be power of 2, >= 16). Default 16 = sedenion.
    #[arg(long, default_value_t = 16)]
    embedding_dim: usize,

    /// Takens lag in time steps (1 = consecutive hourly, 2 = every other, ...).
    #[arg(long, default_value_t = 1)]
    takens_lag: usize,

    /// Embedding mode: "magnetic" (pure B-field) or "magnetic-plasma" (B + plasma).
    /// magnetic-plasma uses 4 steps x 8 channels (always 32D, ignores --embedding-dim).
    #[arg(long, default_value = "magnetic")]
    embedding_mode: String,
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

    let exclude: std::collections::HashSet<&str> =
        cli.exclude_missions.iter().map(|s| s.as_str()).collect();
    if !exclude.is_empty() {
        println!("Excluding missions: {:?}", cli.exclude_missions);
    }

    let mut all_rows = Vec::new();
    for result in reader.deserialize::<HeliosphereFeatureRow>() {
        let r = result?;
        if exclude.contains(r.mission.as_str()) {
            continue;
        }
        if r.bx.is_finite()
            && r.by.is_finite()
            && r.bz.is_finite()
            && r.b_mag.is_finite()
            && r.b_mag > 0.0
        {
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

    // Sort rows by time within each group
    for rows in groups.values_mut() {
        rows.sort_by(|a, b| {
            a.year
                .cmp(&b.year)
                .then(a.doy.cmp(&b.doy))
                .then(a.hour.cmp(&b.hour))
        });
    }

    let is_mixed = cli.embedding_mode == "magnetic-plasma";
    let dim = if is_mixed { 32 } else { cli.embedding_dim };
    let steps = if is_mixed { 4 } else { dim / 4 };
    let mut results = Vec::new();

    println!(
        "[1/2] Computing Takens {}D embedding ({}-step delay, lag={}, mode={}) ...",
        dim, steps, cli.takens_lag, cli.embedding_mode
    );
    for ((mission, product), rows) in groups {
        if rows.len() < steps + 5 {
            continue;
        }

        let (embedded_vectors, meta_idx) = if is_mixed {
            let (vecs, idx, eligible) = magnetic_plasma_takens_embed(&rows, cli.takens_lag);
            let eligible_frac =
                eligible.iter().filter(|&&e| e).count() as f64 / eligible.len() as f64;
            println!(
                "  {mission}/{product}: {:.1}% eligible ({}/{})",
                eligible_frac * 100.0,
                eligible.iter().filter(|&&e| e).count(),
                eligible.len()
            );
            (vecs, idx)
        } else {
            magnetic_takens_embed(&rows, dim, cli.takens_lag)
        };

        let associators =
            cd_kernel::batch_sliding_associator_norms_parallel(&embedded_vectors, dim);

        if associators.is_empty() {
            continue;
        }

        // associator[k] corresponds to triple (emb[k], emb[k+1], emb[k+2])
        // spatial tag comes from meta_idx[k+2] (last row of last vector in triple)
        let tagged_r: Vec<f64> = (0..associators.len())
            .map(|k| rows[meta_idx[k + 2]].r_au)
            .collect();

        let mean = associators.iter().sum::<f64>() / associators.len() as f64;
        let max = associators
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let mean_r = tagged_r.iter().sum::<f64>() / tagged_r.len() as f64;

        results.push(TakensBin {
            mission,
            product,
            r_au: mean_r,
            mean_associator: mean,
            max_associator: max,
            sample_count: associators.len(),
        });
    }

    results.sort_by(|a, b| a.r_au.partial_cmp(&b.r_au).unwrap());

    println!("[2/2] Writing magnetic Takens results...");
    let mut writer = WriterBuilder::new().from_path(&cli.out_csv)?;
    for res in results {
        writer.serialize(res)?;
    }

    Ok(())
}

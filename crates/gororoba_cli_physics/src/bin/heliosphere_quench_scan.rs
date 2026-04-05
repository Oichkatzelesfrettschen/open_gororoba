use anyhow::{Context, Result};
use clap::Parser;
use csv::{ReaderBuilder, WriterBuilder};
use data_core::{HeliosphereFeatureRow, magnetic_takens_embed};
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

    #[arg(long, default_value_t = 10.0)]
    lat_bin_size_deg: f64,

    /// Exclude rows from these missions (repeatable, case-sensitive).
    #[arg(long = "exclude-mission")]
    exclude_missions: Vec<String>,

    /// Exclude rows from these window names (repeatable, case-sensitive).
    #[arg(long = "exclude-window")]
    exclude_windows: Vec<String>,

    /// Embedding dimension (must be power of 2, >= 16). Default 16 = sedenion.
    #[arg(long, default_value_t = 16)]
    embedding_dim: usize,

    /// Takens lag in time steps (1 = consecutive hourly, 2 = every other, ...).
    #[arg(long, default_value_t = 1)]
    takens_lag: usize,
}

#[derive(Debug, Clone, Serialize)]
struct QuenchBin {
    r_center_au: f64,
    lat_center_deg: f64,
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

    let exclude_m: std::collections::HashSet<&str> =
        cli.exclude_missions.iter().map(|s| s.as_str()).collect();
    let exclude_w: std::collections::HashSet<&str> =
        cli.exclude_windows.iter().map(|s| s.as_str()).collect();
    if !exclude_m.is_empty() {
        println!("Excluding missions: {:?}", cli.exclude_missions);
    }
    if !exclude_w.is_empty() {
        println!("Excluding windows: {:?}", cli.exclude_windows);
    }

    let mut all_rows = Vec::new();
    let mut skipped = 0usize;
    for result in reader.deserialize::<HeliosphereFeatureRow>() {
        let r = result?;
        if exclude_m.contains(r.mission.as_str()) || exclude_w.contains(r.window_name.as_str()) {
            skipped += 1;
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
    if skipped > 0 {
        println!("Filtered out {} rows by mission/window exclusion.", skipped);
    }

    let mut mission_groups: BTreeMap<(String, String), Vec<HeliosphereFeatureRow>> =
        BTreeMap::new();
    for row in all_rows {
        mission_groups
            .entry((row.mission.clone(), row.product.clone()))
            .or_default()
            .push(row);
    }

    for rows in mission_groups.values_mut() {
        rows.sort_by(|a, b| {
            a.year
                .cmp(&b.year)
                .then(a.doy.cmp(&b.doy))
                .then(a.hour.cmp(&b.hour))
        });
    }

    let steps = cli.embedding_dim / 4;
    // (r_au, lat_deg, associator_norm, mission_name)
    let mut associator_data: Vec<(f64, f64, f64, String)> = Vec::new();

    println!(
        "[1/2] Computing Takens {}D associators ({}-step, lag={}) with (r, lat) binning...",
        cli.embedding_dim, steps, cli.takens_lag
    );
    for ((mission, _product), rows) in &mission_groups {
        if rows.len() < steps + 5 {
            continue;
        }

        let (embedded_vectors, meta_idx) =
            magnetic_takens_embed(rows, cli.embedding_dim, cli.takens_lag);

        let associators = cd_kernel::batch_sliding_associator_norms_parallel(
            &embedded_vectors,
            cli.embedding_dim,
        );

        // associator[k] = triple (emb[k], emb[k+1], emb[k+2])
        // spatial tag from meta_idx[k+2] (last row of last vector in triple)
        for (k, &norm) in associators.iter().enumerate() {
            let tag_row = meta_idx[k + 2];
            associator_data.push((
                rows[tag_row].r_au,
                rows[tag_row].lat_deg,
                norm,
                mission.clone(),
            ));
        }
    }

    println!(
        "[2/2] Aggregating {} samples into 3D bins (r, lat)...",
        associator_data.len()
    );
    let mut bins: BTreeMap<(i32, i32), Vec<(f64, String)>> = BTreeMap::new();
    for (r, lat, norm, mission) in associator_data {
        let r_bin = (r / cli.bin_size_au).floor() as i32;
        let lat_bin = (lat / cli.lat_bin_size_deg).floor() as i32;
        bins.entry((r_bin, lat_bin))
            .or_default()
            .push((norm, mission));
    }

    let mut writer = WriterBuilder::new().from_path(&cli.out_csv)?;
    for ((r_bin, lat_bin), samples) in bins {
        let r_center = (r_bin as f64 + 0.5) * cli.bin_size_au;
        let lat_center = (lat_bin as f64 + 0.5) * cli.lat_bin_size_deg;

        let mut values: Vec<f64> = samples.iter().map(|(v, _)| *v).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let median = values[values.len() / 2];
        let max = *values.last().unwrap_or(&0.0);
        let mission_count = samples
            .iter()
            .map(|(_, m)| m)
            .collect::<std::collections::HashSet<_>>()
            .len();

        writer.serialize(QuenchBin {
            r_center_au: r_center,
            lat_center_deg: lat_center,
            mean_associator: mean,
            median_associator: median,
            max_associator: max,
            sample_count: values.len(),
            mission_diversity: mission_count,
        })?;
    }

    Ok(())
}

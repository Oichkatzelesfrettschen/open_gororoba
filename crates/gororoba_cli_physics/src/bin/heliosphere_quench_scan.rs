use anyhow::{Context, Result};
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

    #[arg(long, default_value_t = 10.0)]
    lat_bin_size_deg: f64,

    /// Exclude rows from these missions (repeatable, case-sensitive).
    #[arg(long = "exclude-mission")]
    exclude_missions: Vec<String>,

    /// Exclude rows from these window names (repeatable, case-sensitive).
    #[arg(long = "exclude-window")]
    exclude_windows: Vec<String>,
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
        if r.bx.is_finite() && r.by.is_finite() && r.bz.is_finite() && r.b_mag.is_finite() && r.b_mag > 0.0 {
            all_rows.push(r);
        }
    }
    if skipped > 0 {
        println!("Filtered out {} rows by mission/window exclusion.", skipped);
    }

    // Group rows by mission+product to compute temporal Takens associators
    let mut mission_groups: BTreeMap<(String, String), Vec<HeliosphereFeatureRow>> = BTreeMap::new();
    for row in all_rows {
        mission_groups.entry((row.mission.clone(), row.product.clone())).or_default().push(row);
    }

    // Sort rows by time within each group
    for rows in mission_groups.values_mut() {
        rows.sort_by(|a, b| {
            a.year
                .cmp(&b.year)
                .then(a.doy.cmp(&b.doy))
                .then(a.hour.cmp(&b.hour))
        });
    }

    let mut associator_data: Vec<(f64, f64, f64, String)> = Vec::new(); // (r_au, lat_deg, associator, mission)

    println!("[1/2] Computing Takens associators with 3D (r, lat) awareness...");
    for ((mission, _product), rows) in mission_groups {
        if rows.len() < 6 { continue; }

        let mut embedded_vectors = Vec::new();
        let mut r_aus = Vec::new();
        let mut lats = Vec::new();

        // Special handling for MMS: High-cadence dissipation scale (Sprint 50)
        // We use a smaller window or specialized normalization if needed,
        // but for global map consistency, we stick to 4-point Takens.
        let is_mms = mission == "MMS";
        if is_mms {
            println!("  Processing MMS high-cadence tranche: {} samples", rows.len());
        }

        for window in rows.windows(4) {
            let mut v16 = [0.0; 16];
            let local_mean_b = (window[0].b_mag + window[1].b_mag + window[2].b_mag + window[3].b_mag) / 4.0;
            if local_mean_b <= 0.0 { continue; }

            for i in 0..4 {
                v16[i * 4] = window[i].bx / local_mean_b;
                v16[i * 4 + 1] = window[i].by / local_mean_b;
                v16[i * 4 + 2] = window[i].bz / local_mean_b;
                v16[i * 4 + 3] = (window[i].b_mag - local_mean_b) / local_mean_b;
            }
            embedded_vectors.push(v16);
            r_aus.push(window[3].r_au);
            lats.push(window[3].lat_deg);
        }

        let associators = cd_kernel::batch_sedenion_associator_norms_parallel(&embedded_vectors);
        for (i, &norm) in associators.iter().enumerate() {
            // For MMS, r_au is ~1.0, lat is ~0.0 (near Earth)
            associator_data.push((r_aus[i], lats[i], norm, mission.clone()));
        }
    }

    println!("[2/2] Aggregating {} samples into 3D bins (r, lat)...", associator_data.len());
    let mut bins: BTreeMap<(i32, i32), Vec<(f64, String)>> = BTreeMap::new();
    for (r, lat, norm, mission) in associator_data {
        let r_bin = (r / cli.bin_size_au).floor() as i32;
        let lat_bin = (lat / cli.lat_bin_size_deg).floor() as i32;
        bins.entry((r_bin, lat_bin)).or_default().push((norm, mission));
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
        let mission_count = samples.iter().map(|(_, m)| m).collect::<std::collections::HashSet<_>>().len();

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

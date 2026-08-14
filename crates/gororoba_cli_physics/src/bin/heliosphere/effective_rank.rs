//! Compute effective channel independence for CD embeddings.
//!
//! For a given embedding dimension, computes the SVD of the embedding matrix
//! and reports the effective rank (number of singular values above the noise
//! floor). This quantifies how much of the CD dimension is populated with
//! genuinely independent information vs autocorrelated delay copies.
//!
//! Part of the Tower Depth Trichotomy hypothesis test:
//!   - Low dim (8-16): effective rank ~ 4 (all channels correlated)
//!   - Mid dim (32-64): effective rank ~ 12-20 (partial decorrelation)
//!   - High dim (128+): effective rank saturates at 40-60 for 4-channel delay
//!     embedding, far below the nominal dimension

use anyhow::{Context, Result};
use clap::Args;
use csv::ReaderBuilder;
use data_core::{HeliosphereFeatureRow, magnetic_takens_embed};
use stats_core::helpers::singular_values;
use std::{collections::BTreeMap, path::PathBuf};

#[derive(Args, Debug)]
pub struct Cli {
    #[arg(long)]
    cube_csv: PathBuf,

    /// Comma-separated list of dimensions to test.
    #[arg(long, default_value = "8,16,32,64,128,256")]
    dims: String,

    #[arg(long, default_value_t = 1)]
    takens_lag: usize,

    /// Number of channels: 4 (B-field only) or 8 (B-field + plasma).
    #[arg(long, default_value_t = 4)]
    channels: usize,

    /// Max rows per mission to subsample (SVD on full 600K rows is expensive).
    #[arg(long, default_value_t = 5000)]
    max_rows_per_mission: usize,
}

fn effective_rank(embedded: &[Vec<f64>], dim: usize) -> (usize, Vec<f64>) {
    let n = embedded.len();
    if n < 3 || dim < 2 {
        return (0, vec![]);
    }
    let n_use = n.min(5000); // cap for SVD cost
    let svals = singular_values(
        &embedded[..n_use]
            .iter()
            .map(|r| r[..dim].to_vec())
            .collect::<Vec<_>>(),
    );

    // Effective rank at multiple thresholds
    let max_sv = svals.first().copied().unwrap_or(0.0);
    // Use 10% of max as "meaningful" threshold (standard in signal processing)
    let threshold = max_sv * 0.10;
    let eff_rank = svals.iter().filter(|&&s| s > threshold).count();

    (eff_rank, svals)
}

pub fn run(cli: Cli) -> Result<()> {
    let dims: Vec<usize> = cli
        .dims
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();

    println!("=== Effective Channel Independence ===");
    println!(
        "  dims: {:?}, channels={}, lag={}",
        dims, cli.channels, cli.takens_lag
    );

    let mut reader = ReaderBuilder::new()
        .from_path(&cli.cube_csv)
        .with_context(|| format!("open {}", cli.cube_csv.display()))?;

    let mut all_rows: Vec<HeliosphereFeatureRow> = Vec::new();
    for result in reader.deserialize::<HeliosphereFeatureRow>() {
        let r = result?;
        if r.bx.is_finite()
            && r.by.is_finite()
            && r.bz.is_finite()
            && r.b_mag.is_finite()
            && r.b_mag > 0.0
        {
            all_rows.push(r);
        }
    }

    // Group by mission
    let mut mission_groups: BTreeMap<String, Vec<HeliosphereFeatureRow>> = BTreeMap::new();
    for row in all_rows {
        mission_groups
            .entry(row.mission.clone())
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
        rows.truncate(cli.max_rows_per_mission);
    }

    println!(
        "\n{:>6}  {:>10}  {:>10}  {:>12}  {:>10}",
        "dim", "eff_rank", "ratio", "window_hrs", "n_embed"
    );
    println!("{}", "-".repeat(60));

    for &dim in &dims {
        let steps = dim / 4;
        let window_hours = (steps - 1) * cli.takens_lag;

        // Aggregate embeddings across all missions
        let mut all_embedded = Vec::new();
        for rows in mission_groups.values() {
            if cli.channels == 8 {
                let (embedded, _, _) =
                    data_core::plasma_takens_embed_dim(rows, dim, cli.takens_lag);
                all_embedded.extend(embedded);
            } else {
                let (embedded, _) = magnetic_takens_embed(rows, dim, cli.takens_lag);
                all_embedded.extend(embedded);
            }
        }

        if all_embedded.is_empty() {
            println!(
                "{:>6}  {:>10}  {:>10}  {:>12}  {:>10}",
                dim, "N/A", "N/A", window_hours, 0
            );
            continue;
        }

        let (eff_rank, svals) = effective_rank(&all_embedded, dim);
        let ratio = eff_rank as f64 / dim as f64;

        // Show top 5 singular values for context
        let top5: Vec<String> = svals.iter().take(5).map(|s| format!("{:.1}", s)).collect();

        println!(
            "{:>6}  {:>10}  {:>10.3}  {:>12}  {:>10}  sv=[{}]",
            dim,
            eff_rank,
            ratio,
            window_hours,
            all_embedded.len(),
            top5.join(", ")
        );
    }

    // Prediction summary
    println!("\n--- Trichotomy Prediction ---");
    println!("  Low dim (8-16):  eff_rank/dim ~ 0.5-1.0 (all channels useful)");
    println!("  Mid dim (32-64): eff_rank/dim ~ 0.3-0.6 (partial redundancy)");
    println!("  High dim (128+): eff_rank/dim < 0.3 (channel-starved, delay padding)");
    println!("  Fix: genuinely independent diagnostics needed for high-dim to work.");

    Ok(())
}

//! SDO/HMI SHARP CD analysis: detect pre-flare magnetic complexity evolution.
//!
//! Loads SHARP keyword time series from JSOC JSON, builds 15-channel Takens
//! embedding, and runs the CD associator through the flare sequence.
//! The 15 SHARP parameters (USFLUX, MEANGBT, MEANJZH, etc.) capture the
//! full photospheric magnetic complexity of an active region at 12-min cadence.

use anyhow::Result;
use clap::Parser;
use data_core::catalogs::sdo_hmi::{self, SHARP_CHANNELS};
use serde::Serialize;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "sharp-flare-cd")]
struct Cli {
    /// JSOC JSON file path
    #[arg(long)]
    json: PathBuf,

    /// HARPNUM (active region patch number)
    #[arg(long)]
    harpnum: u32,

    /// Number of Takens delay steps
    #[arg(long, default_value_t = 2)]
    steps: usize,

    /// Takens lag (in 12-min increments)
    #[arg(long, default_value_t = 1)]
    lag: usize,

    /// Output JSON
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/sharp_flare_cd.json"
    )]
    out_json: PathBuf,

    /// Precision
    #[arg(long, default_value = "f32")]
    precision: String,
}

#[derive(Debug, Serialize)]
struct SharpCdResult {
    harpnum: u32,
    n_records: usize,
    n_windows: usize,
    embedding_dim: usize,
    channels: usize,
    steps: usize,
    mean_associator: f64,
    median_associator: f64,
    max_associator: f64,
    /// Per-window associator norms with timestamps
    timeseries: Vec<SharpCdPoint>,
}

#[derive(Debug, Serialize)]
struct SharpCdPoint {
    t_rec: String,
    associator_norm: f64,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    println!("=== SHARP Flare CD: HARPNUM {} ===", cli.harpnum);

    let ts =
        sdo_hmi::load_sharp_json(&cli.json, cli.harpnum).map_err(|e| anyhow::anyhow!("{}", e))?;
    println!(
        "  Records: {}, cadence: {} min",
        ts.records.len(),
        ts.cadence_minutes
    );

    let (embedded, indices) = sdo_hmi::sharp_takens_embed(&ts, cli.steps, cli.lag);
    let actual_dim = SHARP_CHANNELS * cli.steps;
    let padded_dim = actual_dim.next_power_of_two();

    println!(
        "  Embedded: {} windows of {}D ({}D padded, {} steps x {} channels)",
        embedded.len(),
        padded_dim,
        actual_dim,
        cli.steps,
        SHARP_CHANNELS
    );

    if embedded.is_empty() {
        anyhow::bail!("No embedded windows");
    }

    // Pad to power of 2
    let padded: Vec<Vec<f64>> = embedded
        .iter()
        .map(|v| {
            let mut p = v.clone();
            p.resize(padded_dim, 0.0);
            p
        })
        .collect();

    let norms =
        cd_kernel::batch_sliding_associator_norms_dispatch(&padded, padded_dim, &cli.precision);

    if norms.is_empty() {
        anyhow::bail!("No associator norms");
    }

    let mean = norms.iter().sum::<f64>() / norms.len() as f64;
    let mut sorted = norms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    let max = sorted.last().copied().unwrap_or(0.0);

    // Build per-window timeseries
    let timeseries: Vec<SharpCdPoint> = norms
        .iter()
        .enumerate()
        .map(|(i, &norm)| {
            let record_idx = if i + 2 < indices.len() {
                indices[i + 2]
            } else {
                indices[indices.len() - 1]
            };
            SharpCdPoint {
                t_rec: ts.records[record_idx].t_rec.clone(),
                associator_norm: norm,
            }
        })
        .collect();

    println!(
        "  Associator: mean={:.4}, median={:.4}, max={:.4}",
        mean, median, max
    );

    // Print evolution
    let n = timeseries.len();
    let step = if n > 10 { n / 10 } else { 1 };
    for (i, pt) in timeseries.iter().enumerate() {
        if i % step == 0 || i == n - 1 {
            println!("    [{}] {} A={:.4}", i, pt.t_rec, pt.associator_norm);
        }
    }

    let result = SharpCdResult {
        harpnum: cli.harpnum,
        n_records: ts.records.len(),
        n_windows: embedded.len(),
        embedding_dim: padded_dim,
        channels: SHARP_CHANNELS,
        steps: cli.steps,
        mean_associator: mean,
        median_associator: median,
        max_associator: max,
        timeseries,
    };

    if let Some(parent) = cli.out_json.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&cli.out_json, serde_json::to_string_pretty(&result)?)?;
    println!("  Wrote {}", cli.out_json.display());

    Ok(())
}

//! Multi-instrument ISM CD analysis: MAG + PWS combined 20-channel embedding.
//!
//! Joins Voyager MAG (4-ch: Br, Bt, Bn, |B|) with PWS (16-ch: spectral power
//! at 10 Hz to 56.2 kHz) by (year, doy, hour) to build a 20-channel Takens
//! embedding for the interstellar medium.
//!
//! The tower depth trichotomy predicts optimal dim ~ 8 * n_channels.
//! With 20 channels: optimal dim ~ 160D (vs 32D for 4-channel B-only).
//! This tests whether multi-instrument ISM data reveals cross-instrument
//! phase coupling invisible to single-instrument analysis.

use anyhow::{Context, Result};
use clap::Args;
use csv::ReaderBuilder;
use serde::Serialize;
use std::{collections::BTreeMap, path::PathBuf};

#[derive(Args)]
pub struct Cli {
    /// MAG CSV (from voyager cdf-ingest)
    #[arg(long)]
    mag_csv: PathBuf,

    /// PWS CSV (from voyager pws-ingest --hourly)
    #[arg(long)]
    pws_csv: PathBuf,

    /// Output JSON
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/ism_multichannel_cd.json"
    )]
    out_json: PathBuf,

    /// Embedding dimension (must be power of 2, multiple of 20 rounded up)
    #[arg(long, default_value_t = 32)]
    embedding_dim: usize,

    /// Precision: "f32" or "f64"
    #[arg(long, default_value = "f32")]
    precision: String,

    /// Minimum year
    #[arg(long, default_value_t = 2019)]
    year_min: u16,

    /// Maximum year
    #[arg(long, default_value_t = 2022)]
    year_max: u16,
}

const MAG_CHANNELS: usize = 4;
const PWS_CHANNELS: usize = 16;
const TOTAL_CHANNELS: usize = MAG_CHANNELS + PWS_CHANNELS;

#[derive(Debug, Clone)]
struct JoinedRecord {
    year: u16,
    doy: u16,
    hour: u8,
    mag: [f64; MAG_CHANNELS], // Br, Bt, Bn, |B|
    pws: [f64; PWS_CHANNELS], // 16 spectral channels
}

#[derive(Debug, Serialize)]
struct MultiChannelResult {
    total_joined: usize,
    embedding_dim: usize,
    n_channels: usize,
    n_windows: usize,
    mean_associator: f64,
    median_associator: f64,
    max_associator: f64,
    year_range: String,
}

pub fn run(cli: Cli) -> Result<()> {
    println!("=== Voyager ISM Multi-Channel CD ===");
    println!("  MAG: {}", cli.mag_csv.display());
    println!("  PWS: {}", cli.pws_csv.display());
    println!(
        "  dim={}, channels={}, precision={}",
        cli.embedding_dim, TOTAL_CHANNELS, cli.precision
    );

    // Step 1: Load MAG data indexed by (year, doy, hour)
    println!("[1/4] Loading MAG data...");
    let mut mag_map: BTreeMap<(u16, u16, u8), Vec<[f64; MAG_CHANNELS]>> = BTreeMap::new();
    {
        let mut rdr = ReaderBuilder::new()
            .from_path(&cli.mag_csv)
            .with_context(|| format!("open {}", cli.mag_csv.display()))?;
        for result in rdr.records() {
            let record = result?;
            let year: u16 = record.get(3).unwrap_or("0").parse().unwrap_or(0);
            let doy: u16 = record.get(4).unwrap_or("0").parse().unwrap_or(0);
            let hour: u8 = record.get(5).unwrap_or("0").parse().unwrap_or(0);
            if year < cli.year_min || year > cli.year_max {
                continue;
            }

            let bx: f64 = record.get(12).unwrap_or("NaN").parse().unwrap_or(f64::NAN);
            let by: f64 = record.get(13).unwrap_or("NaN").parse().unwrap_or(f64::NAN);
            let bz: f64 = record.get(14).unwrap_or("NaN").parse().unwrap_or(f64::NAN);
            let bmag: f64 = record.get(15).unwrap_or("NaN").parse().unwrap_or(f64::NAN);

            if !bx.is_finite() || !bmag.is_finite() || bmag.abs() > 999.0 {
                continue;
            }

            mag_map
                .entry((year, doy, hour))
                .or_default()
                .push([bx, by, bz, bmag]);
        }
    }
    println!("  MAG: {} unique (year,doy,hour) keys", mag_map.len());

    // Step 2: Load PWS data indexed by (year, doy, hour)
    println!("[2/4] Loading PWS data...");
    let mut pws_map: BTreeMap<(u16, u16, u8), [f64; PWS_CHANNELS]> = BTreeMap::new();
    {
        let mut rdr = ReaderBuilder::new()
            .from_path(&cli.pws_csv)
            .with_context(|| format!("open {}", cli.pws_csv.display()))?;
        for result in rdr.records() {
            let record = result?;
            let year: u16 = record.get(0).unwrap_or("0").parse().unwrap_or(0);
            let doy: u16 = record.get(1).unwrap_or("0").parse().unwrap_or(0);
            let hour: u8 = record.get(2).unwrap_or("0").parse().unwrap_or(0);
            if year < cli.year_min || year > cli.year_max {
                continue;
            }

            let mut pws = [0.0f64; PWS_CHANNELS];
            let mut valid = true;
            for (ch, value) in pws.iter_mut().enumerate() {
                let v: f64 = record
                    .get(3 + ch)
                    .unwrap_or("NaN")
                    .parse()
                    .unwrap_or(f64::NAN);
                if !v.is_finite() || v <= 0.0 {
                    valid = false;
                    break;
                }
                *value = v;
            }
            if !valid {
                continue;
            }
            pws_map.insert((year, doy, hour), pws);
        }
    }
    println!("  PWS: {} unique (year,doy,hour) keys", pws_map.len());

    // Step 3: Inner join on (year, doy, hour)
    println!("[3/4] Joining MAG+PWS...");
    let mut joined: Vec<JoinedRecord> = Vec::new();
    for (key, mag_samples) in &mag_map {
        if let Some(pws) = pws_map.get(key) {
            // Average MAG samples within the hour
            let n = mag_samples.len() as f64;
            let mut mag_avg = [0.0f64; MAG_CHANNELS];
            for s in mag_samples {
                for (avg, sample) in mag_avg.iter_mut().zip(s.iter()) {
                    *avg += *sample;
                }
            }
            for avg in mag_avg.iter_mut().take(MAG_CHANNELS) {
                *avg /= n;
            }

            joined.push(JoinedRecord {
                year: key.0,
                doy: key.1,
                hour: key.2,
                mag: mag_avg,
                pws: *pws,
            });
        }
    }
    joined.sort_by_key(|r| (r.year, r.doy, r.hour));
    println!("  Joined: {} records", joined.len());

    if joined.len() < 10 {
        anyhow::bail!("Too few joined records ({}) for CD analysis", joined.len());
    }

    // Step 4: Build 20-channel Takens embedding and compute CD associator
    println!(
        "[4/4] Computing {}-channel {}D CD associator...",
        TOTAL_CHANNELS, cli.embedding_dim
    );
    let dim = cli.embedding_dim;
    // Steps for Takens delay: dim / TOTAL_CHANNELS, rounded up
    let steps = dim.div_ceil(TOTAL_CHANNELS);
    let actual_dim = steps * TOTAL_CHANNELS;

    // Build normalized embedding vectors
    let window_len = steps;
    if joined.len() < window_len + 2 {
        anyhow::bail!(
            "Need at least {} records for embedding, have {}",
            window_len + 2,
            joined.len()
        );
    }

    let mut embedded: Vec<Vec<f64>> = Vec::new();

    for w_start in 0..=(joined.len() - window_len) {
        let window = &joined[w_start..w_start + window_len];

        // Compute per-channel means for normalization
        let mut mag_means = [0.0f64; MAG_CHANNELS];
        let mut pws_means = [0.0f64; PWS_CHANNELS];
        for r in window {
            for (ch, mean) in mag_means.iter_mut().enumerate() {
                *mean += r.mag[ch].abs();
            }
            for (ch, mean) in pws_means.iter_mut().enumerate() {
                *mean += r.pws[ch];
            }
        }
        let wl = window_len as f64;
        for mean in mag_means.iter_mut().take(MAG_CHANNELS) {
            *mean /= wl;
        }
        for mean in pws_means.iter_mut().take(PWS_CHANNELS) {
            *mean /= wl;
        }

        // Skip if any mean is zero
        let mag_ok = mag_means.iter().all(|&m| m > 1e-30);
        let pws_ok = pws_means.iter().all(|&m| m > 1e-30);
        if !mag_ok || !pws_ok {
            continue;
        }

        // Build normalized vector
        let mut v = vec![0.0f64; actual_dim];
        for (s, r) in window.iter().enumerate() {
            let base = s * TOTAL_CHANNELS;
            for (ch, value) in r.mag.iter().enumerate() {
                v[base + ch] = value / mag_means[ch];
            }
            for (ch, value) in r.pws.iter().enumerate() {
                v[base + MAG_CHANNELS + ch] = value / pws_means[ch];
            }
        }

        // Pad to next power of 2 for CD kernel (requires power-of-2 dimensions)
        let padded_dim = actual_dim.next_power_of_two();
        v.resize(padded_dim, 0.0);

        embedded.push(v);
    }

    let padded_dim = actual_dim.next_power_of_two();
    println!(
        "  Embedded: {} windows of {}D ({}D padded to pow2, {} steps x {} channels)",
        embedded.len(),
        padded_dim,
        actual_dim,
        steps,
        TOTAL_CHANNELS
    );

    // Run CD associator at power-of-2 dimension
    let norms =
        cd_kernel::batch_sliding_associator_norms_dispatch(&embedded, padded_dim, &cli.precision);

    if norms.is_empty() {
        anyhow::bail!("No associator norms computed");
    }

    let mean = norms.iter().sum::<f64>() / norms.len() as f64;
    let mut sorted = norms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    let max = sorted.last().copied().unwrap_or(0.0);

    println!(
        "  Associator: mean={:.6}, median={:.6}, max={:.6}",
        mean, median, max
    );

    let result = MultiChannelResult {
        total_joined: joined.len(),
        embedding_dim: padded_dim,
        n_channels: TOTAL_CHANNELS,
        n_windows: embedded.len(),
        mean_associator: mean,
        median_associator: median,
        max_associator: max,
        year_range: format!("{}-{}", cli.year_min, cli.year_max),
    };

    if let Some(parent) = cli.out_json.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(&result)?;
    std::fs::write(&cli.out_json, &json)?;
    println!("  Wrote {}", cli.out_json.display());

    Ok(())
}

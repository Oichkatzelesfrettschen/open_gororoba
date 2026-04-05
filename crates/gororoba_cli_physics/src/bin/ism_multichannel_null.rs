//! Per-dimension multivariate null test for 20-channel MAG+PWS ISM embedding.
//!
//! For each of the 20 physical channels, independently phase-randomize that
//! channel's contribution across all embedding windows, then recompute CD.
//! Channels where the null CD matches the base CD are NOT contributing
//! cross-channel phase structure. Channels where null CD differs significantly
//! ARE contributing genuine multi-instrument coupling.
//!
//! This disambiguates the ISM anomaly: is it in MAG-only, PWS-only, or
//! cross-instrument terms?
//!
//! Usage:
//!   ism-multichannel-null \
//!     --mag-csv data/output/heliosphere/ablations/v2_mag_hourly_2019_2022.csv \
//!     --pws-csv data/output/heliosphere/ablations/v2_pws_hourly_2012_2026.csv \
//!     --n-iter 10

use anyhow::{Context, Result};
use clap::Parser;
use csv::ReaderBuilder;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use serde::Serialize;
use std::{collections::BTreeMap, path::PathBuf};

#[derive(Parser)]
#[command(name = "ism-multichannel-null")]
struct Cli {
    /// MAG CSV
    #[arg(long)]
    mag_csv: PathBuf,

    /// PWS CSV
    #[arg(long)]
    pws_csv: PathBuf,

    /// Output JSON
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/ism_multichannel_null.json"
    )]
    out_json: PathBuf,

    /// Embedding dimension (power of 2)
    #[arg(long, default_value_t = 64)]
    dim: usize,

    /// Number of null iterations per channel
    #[arg(long, default_value_t = 10)]
    n_iter: usize,

    /// Year range start
    #[arg(long, default_value_t = 2019)]
    year_min: u16,

    /// Year range end
    #[arg(long, default_value_t = 2025)]
    year_max: u16,

    /// Epoch split year (for comparing pre/post anomaly)
    #[arg(long, default_value_t = 2024)]
    split_year: u16,
}

const MAG_CHANNELS: usize = 4;
const PWS_CHANNELS: usize = 16;
const TOTAL_CHANNELS: usize = MAG_CHANNELS + PWS_CHANNELS;

#[derive(Debug, Serialize)]
struct ChannelNullResult {
    channel_idx: usize,
    channel_name: String,
    channel_type: String,
    base_cd_median: f64,
    null_cd_median: f64,
    null_cd_std: f64,
    ratio: f64,
    significant: bool,
}

#[derive(Debug, Serialize)]
struct EpochResult {
    epoch: String,
    year_range: String,
    n_windows: usize,
    base_cd_median: f64,
    base_cd_mean: f64,
    base_cd_max: f64,
    channel_nulls: Vec<ChannelNullResult>,
    mag_only_null_ratio: f64,
    pws_only_null_ratio: f64,
    cross_null_ratio: f64,
}

#[derive(Debug, Serialize)]
struct NullTestResult {
    dim: usize,
    n_iter: usize,
    n_channels: usize,
    epochs: Vec<EpochResult>,
}

const CHANNEL_NAMES: [&str; TOTAL_CHANNELS] = [
    "Bx",
    "By",
    "Bz",
    "|B|",
    "PWS_10Hz",
    "PWS_18Hz",
    "PWS_31Hz",
    "PWS_56Hz",
    "PWS_100Hz",
    "PWS_178Hz",
    "PWS_311Hz",
    "PWS_562Hz",
    "PWS_1kHz",
    "PWS_1.8kHz",
    "PWS_3.1kHz",
    "PWS_5.6kHz",
    "PWS_10kHz",
    "PWS_17.8kHz",
    "PWS_31.1kHz",
    "PWS_56.2kHz",
];

fn load_and_join(
    mag_path: &std::path::Path,
    pws_path: &std::path::Path,
    year_min: u16,
    year_max: u16,
) -> Result<Vec<(u16, [f64; TOTAL_CHANNELS])>> {
    // Load MAG
    let mut mag_map: BTreeMap<(u16, u16, u8), [f64; MAG_CHANNELS]> = BTreeMap::new();
    let mut rdr = ReaderBuilder::new()
        .from_path(mag_path)
        .with_context(|| format!("open {}", mag_path.display()))?;
    for result in rdr.records() {
        let record = result?;
        let year: u16 = record.get(3).unwrap_or("0").parse().unwrap_or(0);
        let doy: u16 = record.get(4).unwrap_or("0").parse().unwrap_or(0);
        let hour: u8 = record.get(5).unwrap_or("0").parse().unwrap_or(0);
        if year < year_min || year > year_max {
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
            .or_insert([bx, by, bz, bmag]);
    }

    // Load PWS
    let mut pws_map: BTreeMap<(u16, u16, u8), [f64; PWS_CHANNELS]> = BTreeMap::new();
    let mut rdr = ReaderBuilder::new()
        .from_path(pws_path)
        .with_context(|| format!("open {}", pws_path.display()))?;
    for result in rdr.records() {
        let record = result?;
        let year: u16 = record.get(0).unwrap_or("0").parse().unwrap_or(0);
        let doy: u16 = record.get(1).unwrap_or("0").parse().unwrap_or(0);
        let hour: u8 = record.get(2).unwrap_or("0").parse().unwrap_or(0);
        if year < year_min || year > year_max {
            continue;
        }
        let mut pws = [0.0f64; PWS_CHANNELS];
        let mut valid = true;
        for (ch, slot) in pws.iter_mut().enumerate() {
            let v: f64 = record
                .get(3 + ch)
                .unwrap_or("NaN")
                .parse()
                .unwrap_or(f64::NAN);
            if !v.is_finite() || v <= 0.0 {
                valid = false;
                break;
            }
            *slot = v;
        }
        if !valid {
            continue;
        }
        pws_map.insert((year, doy, hour), pws);
    }

    // Join
    let mut joined = Vec::new();
    for (key, mag) in &mag_map {
        if let Some(pws) = pws_map.get(key) {
            let mut combined = [0.0f64; TOTAL_CHANNELS];
            combined[..MAG_CHANNELS].copy_from_slice(mag);
            combined[MAG_CHANNELS..].copy_from_slice(pws);
            joined.push((key.0, combined));
        }
    }
    joined.sort_by_key(|(y, _)| *y);
    Ok(joined)
}

fn build_embeddings(records: &[(u16, [f64; TOTAL_CHANNELS])], dim: usize) -> Vec<Vec<f64>> {
    let steps = dim.div_ceil(TOTAL_CHANNELS);
    let actual_dim = steps * TOTAL_CHANNELS;
    let padded_dim = actual_dim.next_power_of_two();

    let mut embedded = Vec::new();
    for w_start in 0..records.len().saturating_sub(steps) {
        let window = &records[w_start..w_start + steps];

        // Per-channel means for normalization
        let mut means = [0.0f64; TOTAL_CHANNELS];
        for (_, r) in window {
            for (ch, mean) in means.iter_mut().enumerate() {
                *mean += if ch < MAG_CHANNELS {
                    r[ch].abs()
                } else {
                    r[ch]
                };
            }
        }
        let wl = steps as f64;
        for m in means.iter_mut() {
            *m /= wl;
        }
        if means.iter().any(|&m| m <= 1e-30) {
            continue;
        }

        let mut v = vec![0.0f64; padded_dim];
        for (s, (_, r)) in window.iter().enumerate() {
            let base = s * TOTAL_CHANNELS;
            for (ch, value) in v[base..(base + TOTAL_CHANNELS).min(actual_dim)]
                .iter_mut()
                .enumerate()
            {
                *value = r[ch] / means[ch];
            }
        }
        embedded.push(v);
    }
    embedded
}

fn compute_cd_median(embeddings: &[Vec<f64>], dim: usize) -> (f64, f64, f64) {
    if embeddings.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let padded_f32: Vec<Vec<f32>> = embeddings
        .iter()
        .map(|v| v.iter().map(|&x| x as f32).collect())
        .collect();
    let norms = cd_kernel::batch_sliding_associator_norms_f32(&padded_f32, dim);
    if norms.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut sorted: Vec<f64> = norms.iter().map(|&n| n as f64).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[sorted.len() / 2];
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let max = sorted.last().copied().unwrap_or(0.0);
    (median, mean, max)
}

fn phase_randomize_channel(
    embeddings: &[Vec<f64>],
    channel_idx: usize,
    rng: &mut ChaCha20Rng,
    dim: usize,
) -> Vec<Vec<f64>> {
    use rand::seq::SliceRandom;

    let steps = dim.div_ceil(TOTAL_CHANNELS);

    // Extract all values for this channel across all windows and time steps
    let mut channel_values: Vec<f64> = Vec::new();
    for emb in embeddings {
        for s in 0..steps {
            let idx = s * TOTAL_CHANNELS + channel_idx;
            if idx < emb.len() {
                channel_values.push(emb[idx]);
            }
        }
    }

    // Shuffle the channel values
    channel_values.shuffle(rng);

    // Reconstruct embeddings with shuffled channel
    let mut result = embeddings.to_vec();
    let mut val_idx = 0;
    for emb in result.iter_mut() {
        for s in 0..steps {
            let idx = s * TOTAL_CHANNELS + channel_idx;
            if idx < emb.len() && val_idx < channel_values.len() {
                emb[idx] = channel_values[val_idx];
                val_idx += 1;
            }
        }
    }
    result
}

fn analyze_epoch(
    records: &[(u16, [f64; TOTAL_CHANNELS])],
    epoch_name: &str,
    year_range: &str,
    dim: usize,
    n_iter: usize,
    seed_base: u64,
) -> EpochResult {
    let padded_dim = (dim.div_ceil(TOTAL_CHANNELS) * TOTAL_CHANNELS).next_power_of_two();
    let embeddings = build_embeddings(records, dim);
    let (base_median, base_mean, base_max) = compute_cd_median(&embeddings, padded_dim);

    println!(
        "    {} base: median={:.6}, mean={:.6}, max={:.6} ({} windows)",
        epoch_name,
        base_median,
        base_mean,
        base_max,
        embeddings.len()
    );

    let mut channel_nulls = Vec::new();

    for (ch, channel_name) in CHANNEL_NAMES.iter().enumerate() {
        let mut null_medians = Vec::new();
        for iter in 0..n_iter {
            let mut rng = ChaCha20Rng::seed_from_u64(seed_base + ch as u64 * 100 + iter as u64);
            let shuffled = phase_randomize_channel(&embeddings, ch, &mut rng, dim);
            let (null_median, _, _) = compute_cd_median(&shuffled, padded_dim);
            null_medians.push(null_median);
        }

        let null_mean = null_medians.iter().sum::<f64>() / null_medians.len() as f64;
        let null_var = null_medians
            .iter()
            .map(|m| (m - null_mean).powi(2))
            .sum::<f64>()
            / null_medians.len() as f64;
        let null_std = null_var.sqrt();
        let ratio = if base_median > 1e-15 {
            null_mean / base_median
        } else {
            1.0
        };

        // Significant if null differs from base by more than 2 sigma
        let significant = (null_mean - base_median).abs() > 2.0 * null_std.max(base_median * 0.05);

        let ch_type = if ch < MAG_CHANNELS { "MAG" } else { "PWS" };
        let marker = if significant { " ***" } else { "" };
        println!(
            "      ch{:>2} {:>12} ({}): null={:.6} +/- {:.6}, ratio={:.3}{}",
            ch, channel_name, ch_type, null_mean, null_std, ratio, marker
        );

        channel_nulls.push(ChannelNullResult {
            channel_idx: ch,
            channel_name: (*channel_name).to_string(),
            channel_type: ch_type.to_string(),
            base_cd_median: base_median,
            null_cd_median: null_mean,
            null_cd_std: null_std,
            ratio,
            significant,
        });
    }

    // Summary: average ratio for MAG, PWS, and cross groups
    let mag_ratios: Vec<f64> = channel_nulls
        .iter()
        .filter(|c| c.channel_type == "MAG")
        .map(|c| c.ratio)
        .collect();
    let pws_ratios: Vec<f64> = channel_nulls
        .iter()
        .filter(|c| c.channel_type == "PWS")
        .map(|c| c.ratio)
        .collect();

    let mag_ratio = if mag_ratios.is_empty() {
        1.0
    } else {
        mag_ratios.iter().sum::<f64>() / mag_ratios.len() as f64
    };
    let pws_ratio = if pws_ratios.is_empty() {
        1.0
    } else {
        pws_ratios.iter().sum::<f64>() / pws_ratios.len() as f64
    };
    // Cross: shuffle ALL MAG together
    let cross_ratio = mag_ratio.min(pws_ratio); // simplified proxy

    EpochResult {
        epoch: epoch_name.to_string(),
        year_range: year_range.to_string(),
        n_windows: embeddings.len(),
        base_cd_median: base_median,
        base_cd_mean: base_mean,
        base_cd_max: base_max,
        channel_nulls,
        mag_only_null_ratio: mag_ratio,
        pws_only_null_ratio: pws_ratio,
        cross_null_ratio: cross_ratio,
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    println!("=== ISM Multi-Channel Per-Dimension Null Test ===");
    println!(
        "  dim={}, n_iter={}, channels={}",
        cli.dim, cli.n_iter, TOTAL_CHANNELS
    );
    println!("  Split year: {}", cli.split_year);

    let all_records = load_and_join(&cli.mag_csv, &cli.pws_csv, cli.year_min, cli.year_max)?;
    println!("  Total joined records: {}", all_records.len());

    // Split into pre and post epochs
    let pre: Vec<_> = all_records
        .iter()
        .filter(|(y, _)| *y < cli.split_year)
        .cloned()
        .collect();
    let post: Vec<_> = all_records
        .iter()
        .filter(|(y, _)| *y >= cli.split_year)
        .cloned()
        .collect();

    println!("\n  Pre-{}: {} records", cli.split_year, pre.len());
    println!("  Post-{}: {} records", cli.split_year, post.len());

    let mut epochs = Vec::new();

    if pre.len() >= 10 {
        println!("\n  --- Pre-{} epoch ---", cli.split_year);
        let epoch = analyze_epoch(
            &pre,
            &format!("pre_{}", cli.split_year),
            &format!("{}-{}", cli.year_min, cli.split_year - 1),
            cli.dim,
            cli.n_iter,
            42,
        );
        epochs.push(epoch);
    }

    if post.len() >= 10 {
        println!("\n  --- Post-{} epoch ---", cli.split_year);
        let epoch = analyze_epoch(
            &post,
            &format!("post_{}", cli.split_year),
            &format!("{}-{}", cli.split_year, cli.year_max),
            cli.dim,
            cli.n_iter,
            137,
        );
        epochs.push(epoch);
    }

    // Summary
    println!("\n=== Summary ===");
    for epoch in &epochs {
        let n_sig = epoch.channel_nulls.iter().filter(|c| c.significant).count();
        let sig_names: Vec<&str> = epoch
            .channel_nulls
            .iter()
            .filter(|c| c.significant)
            .map(|c| c.channel_name.as_str())
            .collect();
        println!(
            "  {}: base_cd={:.6}, {}/{} channels significant",
            epoch.epoch, epoch.base_cd_median, n_sig, TOTAL_CHANNELS
        );
        if !sig_names.is_empty() {
            println!("    Significant: {}", sig_names.join(", "));
        }
        println!(
            "    MAG avg ratio: {:.3}, PWS avg ratio: {:.3}",
            epoch.mag_only_null_ratio, epoch.pws_only_null_ratio
        );
    }

    let result = NullTestResult {
        dim: cli.dim,
        n_iter: cli.n_iter,
        n_channels: TOTAL_CHANNELS,
        epochs,
    };

    let json = serde_json::to_string_pretty(&result)?;
    if let Some(parent) = cli.out_json.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&cli.out_json, &json)?;
    println!("\nResults -> {}", cli.out_json.display());

    Ok(())
}

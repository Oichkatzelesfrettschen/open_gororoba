//! SDO AIA multi-channel CD: 6-band EUV + HMI SHARP through X9.3 flare.
//!
//! Tests the tower depth trichotomy on solar data with genuinely independent
//! channels at the SAME cadence (2-min AIA, no resampling needed).
//! 12 channels: 6 AIA bands x (mean + rms) = temperature-diverse coronal response.

use anyhow::{Context, Result};
use clap::Parser;
use csv::ReaderBuilder;
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};

#[derive(Parser)]
#[command(name = "solar-aia-cd")]
struct Cli {
    /// AIA 6-band CSV (from JSOC DATAMEAN/DATARMS export).
    #[arg(long, default_value = "data/external/sdo_aia/aia_6band_x93.csv")]
    aia_csv: PathBuf,

    /// Embedding dimension (power of 2).
    #[arg(long, default_value_t = 32)]
    embedding_dim: usize,

    /// Precision for CD computation.
    #[arg(long, default_value = "f64")]
    precision: String,

    /// Output JSON path.
    #[arg(long, default_value = "data/output/heliosphere/ablations/solar_aia_cd.json")]
    out_json: PathBuf,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct AiaRow {
    time: String,
    aia94_mean: f64,
    aia131_mean: f64,
    aia171_mean: f64,
    aia193_mean: f64,
    aia211_mean: f64,
    aia335_mean: f64,
    aia94_rms: f64,
    aia131_rms: f64,
    aia171_rms: f64,
    aia193_rms: f64,
    aia211_rms: f64,
    aia335_rms: f64,
}

#[derive(Debug, Serialize)]
struct SolarCdResult {
    n_rows: usize,
    embedding_dim: usize,
    n_channels: usize,
    // 12-channel multi-band
    multi_n_embed: usize,
    multi_eff_rank: usize,
    multi_cd_mean: f64,
    multi_cd_max: f64,
    // 2-channel delay (171A mean + rms only)
    delay_n_embed: usize,
    delay_cd_mean: f64,
    delay_cd_max: f64,
    // Flare signature
    pre_flare_cd_mean: f64,
    flare_cd_mean: f64,
    post_flare_cd_mean: f64,
    flare_enhancement: f64,
    interpretation: String,
}

fn effective_rank(embedded: &[Vec<f64>], dim: usize) -> usize {
    let n = embedded.len().min(5000);
    if n < 3 || dim < 2 { return 0; }
    let mat = DMatrix::from_fn(n, dim, |i, j| embedded[i][j]);
    let svd = mat.svd(false, false);
    let svals: Vec<f64> = svd.singular_values.iter().copied().collect();
    let max_sv = svals.first().copied().unwrap_or(0.0);
    let threshold = max_sv * 0.10;
    svals.iter().filter(|&&s| s > threshold).count()
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let dim = cli.embedding_dim;
    println!("=== SDO AIA Multi-Channel CD through X9.3 Flare ===");
    println!("  dim={}, precision={}", dim, cli.precision);

    let mut reader = ReaderBuilder::new()
        .from_path(&cli.aia_csv)
        .with_context(|| format!("open {}", cli.aia_csv.display()))?;

    let mut rows: Vec<AiaRow> = Vec::new();
    for result in reader.deserialize::<AiaRow>() {
        let r = result?;
        if r.aia171_mean > 0.0 && r.aia94_mean > 0.0 {
            rows.push(r);
        }
    }
    println!("  Loaded {} rows from {}", rows.len(), cli.aia_csv.display());

    // Standardize each channel to zero-mean unit-variance
    let n = rows.len();
    let channels: Vec<Vec<f64>> = vec![
        rows.iter().map(|r| r.aia94_mean).collect(),
        rows.iter().map(|r| r.aia131_mean).collect(),
        rows.iter().map(|r| r.aia171_mean).collect(),
        rows.iter().map(|r| r.aia193_mean).collect(),
        rows.iter().map(|r| r.aia211_mean).collect(),
        rows.iter().map(|r| r.aia335_mean).collect(),
        rows.iter().map(|r| r.aia94_rms).collect(),
        rows.iter().map(|r| r.aia131_rms).collect(),
        rows.iter().map(|r| r.aia171_rms).collect(),
        rows.iter().map(|r| r.aia193_rms).collect(),
        rows.iter().map(|r| r.aia211_rms).collect(),
        rows.iter().map(|r| r.aia335_rms).collect(),
    ];
    let n_ch = channels.len();

    let standardized: Vec<Vec<f64>> = channels
        .iter()
        .map(|ch| {
            let mean = ch.iter().sum::<f64>() / n as f64;
            let var = ch.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
            let std = var.sqrt().max(1e-15);
            ch.iter().map(|x| (x - mean) / std).collect()
        })
        .collect();

    // Build 12-channel embedding
    let steps = dim / n_ch;
    let actual_dim = if steps > 0 { (n_ch * steps).next_power_of_two() } else { dim };
    let raw_dim = n_ch * steps.max(1);
    println!("  12-channel: {} steps x {} channels = {}D (padded to {}D)", steps.max(1), n_ch, raw_dim, actual_dim);

    let _stride = 1; // 2-min cadence, stride=1 step
    let window = steps.max(1);
    let mut multi_embedded = Vec::new();
    if n > window + 2 {
        for start in 0..n - window {
            let mut v = vec![0.0f64; actual_dim];
            for s in 0..steps.max(1) {
                for ch in 0..n_ch {
                    if s * n_ch + ch < raw_dim {
                        v[s * n_ch + ch] = standardized[ch][start + s];
                    }
                }
            }
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-15 {
                for x in v.iter_mut() { *x /= norm; }
            }
            multi_embedded.push(v);
        }
    }

    let multi_eff = if multi_embedded.len() > 5 { effective_rank(&multi_embedded, actual_dim) } else { 0 };
    let multi_norms = cd_kernel::batch_sliding_associator_norms_dispatch(
        &multi_embedded, actual_dim, &cli.precision,
    );
    let multi_mean = if multi_norms.is_empty() { 0.0 } else {
        multi_norms.iter().sum::<f64>() / multi_norms.len() as f64
    };
    let multi_max = multi_norms.iter().cloned().fold(0.0f64, f64::max);

    println!("  12-ch: {} embeddings, eff_rank={}/{}, CD mean={:.6}, max={:.6}",
        multi_embedded.len(), multi_eff, actual_dim, multi_mean, multi_max);

    // Build 2-channel delay embedding (171A mean + rms) for comparison
    let delay_ch = 2;
    let delay_steps = dim / delay_ch;
    let mut delay_embedded = Vec::new();
    if n > delay_steps + 2 {
        for start in 0..n - delay_steps {
            let mut v = vec![0.0f64; dim];
            for s in 0..delay_steps {
                v[s * delay_ch] = standardized[2][start + s]; // 171A mean
                v[s * delay_ch + 1] = standardized[8][start + s]; // 171A rms
            }
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-15 {
                for x in v.iter_mut() { *x /= norm; }
            }
            delay_embedded.push(v);
        }
    }
    let delay_norms = cd_kernel::batch_sliding_associator_norms_dispatch(
        &delay_embedded, dim, &cli.precision,
    );
    let delay_mean = if delay_norms.is_empty() { 0.0 } else {
        delay_norms.iter().sum::<f64>() / delay_norms.len() as f64
    };
    let delay_max = delay_norms.iter().cloned().fold(0.0f64, f64::max);
    println!("  2-ch delay: {} embeddings, CD mean={:.6}, max={:.6}",
        delay_embedded.len(), delay_mean, delay_max);

    // Flare phase analysis (flare peak ~12:02 UT, roughly row 120 of 177)
    let flare_idx = multi_norms.len() * 2 / 3; // approximate flare peak
    let pre = if flare_idx > 20 { &multi_norms[..flare_idx - 10] } else { &multi_norms[..] };
    let during = if flare_idx + 10 < multi_norms.len() {
        &multi_norms[flare_idx - 10..flare_idx + 10]
    } else { &multi_norms[flare_idx..] };
    let post = if flare_idx + 10 < multi_norms.len() {
        &multi_norms[flare_idx + 10..]
    } else { &[] as &[f64] };

    let pre_mean = if pre.is_empty() { 0.0 } else { pre.iter().sum::<f64>() / pre.len() as f64 };
    let flare_mean = if during.is_empty() { 0.0 } else { during.iter().sum::<f64>() / during.len() as f64 };
    let post_mean = if post.is_empty() { 0.0 } else { post.iter().sum::<f64>() / post.len() as f64 };
    let enhancement = if pre_mean > 1e-15 { flare_mean / pre_mean } else { 0.0 };

    println!("\n  Flare phases: pre={:.4}, during={:.4}, post={:.4}, enhancement={:.2}x",
        pre_mean, flare_mean, post_mean, enhancement);

    let interp = if enhancement > 1.5 {
        format!("FLARE DETECTED: {:.1}x CD enhancement during X9.3 with {} channels at {}D", enhancement, n_ch, actual_dim)
    } else {
        format!("Weak flare signal ({:.2}x) with {} channels at {}D", enhancement, n_ch, actual_dim)
    };
    println!("  {}", interp);

    let result = SolarCdResult {
        n_rows: n,
        embedding_dim: actual_dim,
        n_channels: n_ch,
        multi_n_embed: multi_embedded.len(),
        multi_eff_rank: multi_eff,
        multi_cd_mean: multi_mean,
        multi_cd_max: multi_max,
        delay_n_embed: delay_embedded.len(),
        delay_cd_mean: delay_mean,
        delay_cd_max: delay_max,
        pre_flare_cd_mean: pre_mean,
        flare_cd_mean: flare_mean,
        post_flare_cd_mean: post_mean,
        flare_enhancement: enhancement,
        interpretation: interp,
    };

    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.out_json, serde_json::to_string_pretty(&result)?)?;
    println!("  Wrote {}", cli.out_json.display());
    Ok(())
}

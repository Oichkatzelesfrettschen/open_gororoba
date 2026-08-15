//! Solar flare CD associator: detect magnetic topology transitions in SHARP keywords.
//!
//! Uses SWAN-SF SHARP keyword time series (USFLUX, MEANPOT, SHRGT45, R_VALUE)
//! as 4-channel input for 32D Takens embedding. Tests whether the CD associator
//! detects pre-flare buildup (coherent flux rope = low A) and eruption onset (spike).

use anyhow::Result;
use clap::Args;
use csv::ReaderBuilder;
use serde::Serialize;
use std::{fs, path::PathBuf};

#[derive(Args)]
pub struct Cli {
    /// Directory containing SWAN-SF CSV files.
    #[arg(long, default_value = "data/external/sdo_hmi/partition1/FL")]
    data_dir: PathBuf,

    /// GOES flare class prefix to filter (e.g., "X" for X-class only).
    #[arg(long, default_value = "X")]
    flare_class: String,

    #[arg(long, default_value_t = 32)]
    embedding_dim: usize,

    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/solar_flare_cd.json"
    )]
    out_json: PathBuf,
}

#[derive(Debug, Serialize)]
struct FlareResult {
    filename: String,
    goes_class: String,
    harpnum: String,
    n_timesteps: usize,
    mean_associator: f64,
    max_associator: f64,
    n_transitions: usize,
    pre_flare_mean: f64,
    flare_onset_max: f64,
}

#[derive(Debug, Serialize)]
struct SolarOutput {
    n_flares_analyzed: usize,
    results: Vec<FlareResult>,
    mean_pre_flare: f64,
    mean_flare_onset: f64,
    ratio: f64,
    interpretation: String,
}

pub fn run(cli: Cli) -> Result<()> {
    println!("=== Solar Flare CD Analysis ===");

    let entries: Vec<_> = fs::read_dir(&cli.data_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name()
                .to_str()
                .is_some_and(|n| n.starts_with(&cli.flare_class) && n.ends_with(".csv"))
        })
        .collect();

    println!(
        "  Found {} {}-class flare files",
        entries.len(),
        cli.flare_class
    );

    let channels = 4usize;
    let steps = cli.embedding_dim / channels;
    let mut results = Vec::new();

    for entry in &entries {
        let path = entry.path();
        let fname = entry.file_name().to_string_lossy().to_string();

        // Parse GOES class and HARPNUM from filename
        let goes_class = fname.split('@').next().unwrap_or("?").to_string();
        let harpnum = fname
            .split("_ar")
            .nth(1)
            .and_then(|s| s.split('_').next())
            .unwrap_or("?")
            .to_string();

        // Read CSV -- columns: Timestamp, TOTUSJH, TOTBSQ, TOTPOT, ..., R_VALUE, ...
        let content = fs::read_to_string(&path)?;
        let mut reader = ReaderBuilder::new()
            .delimiter(b'\t')
            .has_headers(true)
            .flexible(true)
            .from_reader(content.as_bytes());

        let headers = reader.headers()?.clone();

        // Find column indices for our 4 channels
        let usflux_col = headers.iter().position(|h| h == "USFLUX");
        let meanpot_col = headers.iter().position(|h| h == "MEANPOT");
        let shrgt45_col = headers.iter().position(|h| h == "SHRGT45");
        let r_value_col = headers.iter().position(|h| h == "R_VALUE");

        let (Some(c0), Some(c1), Some(c2), Some(c3)) =
            (usflux_col, meanpot_col, shrgt45_col, r_value_col)
        else {
            continue;
        };

        // Parse time series
        let mut ch0 = Vec::new(); // USFLUX
        let mut ch1 = Vec::new(); // MEANPOT
        let mut ch2 = Vec::new(); // SHRGT45
        let mut ch3 = Vec::new(); // R_VALUE

        for record in reader.records().flatten() {
            let v0: f64 = record
                .get(c0)
                .and_then(|s| s.parse().ok())
                .unwrap_or(f64::NAN);
            let v1: f64 = record
                .get(c1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(f64::NAN);
            let v2: f64 = record
                .get(c2)
                .and_then(|s| s.parse().ok())
                .unwrap_or(f64::NAN);
            let v3: f64 = record
                .get(c3)
                .and_then(|s| s.parse().ok())
                .unwrap_or(f64::NAN);

            if v0.is_finite() && v1.is_finite() && v2.is_finite() && v3.is_finite() {
                ch0.push(v0);
                ch1.push(v1);
                ch2.push(v2);
                ch3.push(v3);
            }
        }

        let n = ch0.len();
        if n < steps + 2 {
            continue;
        }

        // Build 32D Takens embedding using the 4 SHARP keyword channels
        let mut embedded: Vec<Vec<f64>> = Vec::new();
        for w in 0..=n.saturating_sub(steps) {
            // Normalize each channel by its local mean
            let mean0: f64 = (0..steps).map(|s| ch0[w + s]).sum::<f64>() / steps as f64;
            let mean1: f64 = (0..steps).map(|s| ch1[w + s]).sum::<f64>() / steps as f64;
            let mean2: f64 = (0..steps).map(|s| ch2[w + s]).sum::<f64>() / steps as f64;
            let mean3: f64 = (0..steps).map(|s| ch3[w + s]).sum::<f64>() / steps as f64;

            if mean0.abs() < 1e-10 || mean1.abs() < 1e-10 {
                continue;
            }

            let mut v = vec![0.0; cli.embedding_dim];
            for s in 0..steps {
                let i = w + s;
                v[s * channels] = ch0[i] / mean0;
                v[s * channels + 1] = ch1[i] / mean1;
                v[s * channels + 2] = if mean2.abs() > 1e-10 {
                    ch2[i] / mean2
                } else {
                    0.0
                };
                v[s * channels + 3] = if mean3.abs() > 1e-10 {
                    ch3[i] / mean3
                } else {
                    0.0
                };
            }
            embedded.push(v);
        }

        if embedded.len() < 3 {
            continue;
        }

        let norms =
            cd_kernel::batch_sliding_associator_norms_parallel(&embedded, cli.embedding_dim);

        if norms.is_empty() {
            continue;
        }

        let mean_a = norms.iter().sum::<f64>() / norms.len() as f64;
        let max_a = norms.iter().cloned().fold(0.0f64, f64::max);

        // Split: first half = pre-flare, second half = flare onset
        let mid = norms.len() / 2;
        let pre_mean = norms[..mid].iter().sum::<f64>() / mid.max(1) as f64;
        let onset_max = norms[mid..].iter().cloned().fold(0.0f64, f64::max);

        // Detect transitions
        let global_std = {
            let var = norms.iter().map(|&a| (a - mean_a).powi(2)).sum::<f64>() / norms.len() as f64;
            var.sqrt()
        };
        let threshold = global_std * 1.5;
        let tw = 3usize;
        let mut n_trans = 0usize;
        let mut last: Option<usize> = None;
        for i in tw..norms.len().saturating_sub(tw) {
            let pre: f64 = norms[i.saturating_sub(tw)..i].iter().sum::<f64>() / tw as f64;
            let post: f64 = norms[i..(i + tw).min(norms.len())].iter().sum::<f64>()
                / tw.min(norms.len() - i) as f64;
            if (post - pre).abs() > threshold
                && last.is_none_or(|prev| i.saturating_sub(prev) >= tw)
            {
                n_trans += 1;
                last = Some(i);
            }
        }

        results.push(FlareResult {
            filename: fname,
            goes_class,
            harpnum,
            n_timesteps: n,
            mean_associator: mean_a,
            max_associator: max_a,
            n_transitions: n_trans,
            pre_flare_mean: pre_mean,
            flare_onset_max: onset_max,
        });
    }

    println!("  Analyzed {} flares", results.len());

    // Aggregate
    let mean_pre: f64 = if results.is_empty() {
        0.0
    } else {
        results.iter().map(|r| r.pre_flare_mean).sum::<f64>() / results.len() as f64
    };
    let mean_onset: f64 = if results.is_empty() {
        0.0
    } else {
        results.iter().map(|r| r.flare_onset_max).sum::<f64>() / results.len() as f64
    };
    let ratio = if mean_pre > 1e-15 {
        mean_onset / mean_pre
    } else {
        0.0
    };

    println!("  Pre-flare mean A: {:.4}", mean_pre);
    println!("  Flare onset max A: {:.4}", mean_onset);
    println!("  Onset/pre ratio: {:.2}", ratio);

    for r in &results {
        println!(
            "    {}: pre={:.4}, onset_max={:.4}, transitions={}",
            r.goes_class, r.pre_flare_mean, r.flare_onset_max, r.n_transitions
        );
    }

    let interp = format!(
        "Analyzed {} {}-class flares from SWAN-SF. Mean pre-flare A={:.4}, mean onset max A={:.4}, ratio={:.2}. {}",
        results.len(),
        cli.flare_class,
        mean_pre,
        mean_onset,
        ratio,
        if ratio > 1.5 {
            "Flare onset produces HIGHER associator than pre-flare phase -- consistent with boundary disruption phenotype."
        } else if ratio > 1.0 {
            "Moderate elevation at onset."
        } else {
            "No clear onset signature in SHARP keyword embedding."
        }
    );

    let output = SolarOutput {
        n_flares_analyzed: results.len(),
        results,
        mean_pre_flare: mean_pre,
        mean_flare_onset: mean_onset,
        ratio,
        interpretation: interp.clone(),
    };

    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.out_json, serde_json::to_string_pretty(&output)?)?;
    println!("\n  {}", interp);
    println!("  Wrote {}", cli.out_json.display());

    Ok(())
}

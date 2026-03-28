//! Geomagnetic jerk detection with 32D CD associator.
//!
//! Geomagnetic jerks are abrupt changes in the secular variation of Earth's
//! magnetic field, caused by core MHD dynamics. This binary tests whether
//! the CD associator detects jerks as phase-coupling transitions in
//! multi-observatory vector magnetograms.

use anyhow::Result;
use clap::Parser;
use data_core::catalogs::intermagnet::load_observatory_year;
use serde::Serialize;
use std::{fs, path::PathBuf};

#[derive(Parser)]
#[command(name = "geomag-cd-jerk")]
struct Cli {
    /// Observatory IAGA code.
    #[arg(long, default_value = "CLF")]
    observatory: String,

    /// Year to analyze.
    #[arg(long, default_value_t = 2014)]
    year: u16,

    #[arg(long, default_value_t = 32)]
    embedding_dim: usize,

    /// Downsample factor (minutes per averaged bin).
    #[arg(long, default_value_t = 60)]
    downsample_minutes: usize,

    #[arg(long, default_value = "data/output/heliosphere/ablations/geomag_jerk_analysis.json")]
    out_json: PathBuf,

    #[arg(long, default_value = "data/external/intermagnet")]
    data_dir: String,
}

#[derive(Debug, Serialize)]
struct JerkResult {
    observatory: String,
    year: u16,
    n_records: usize,
    n_bins: usize,
    n_embedded: usize,
    n_transitions: usize,
    monthly_mean_associator: Vec<f64>,
    transition_months: Vec<u8>,
    interpretation: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let dir = PathBuf::from(&cli.data_dir);

    println!("=== Geomagnetic Jerk Analysis ===");
    println!("  Observatory: {}, Year: {}", cli.observatory, cli.year);

    let records = load_observatory_year(&dir, &cli.observatory, cli.year);
    println!("  Loaded {} minute records", records.len());

    if records.len() < 1000 {
        anyhow::bail!("Too few records ({})", records.len());
    }

    // Downsample to hourly (or cli.downsample_minutes) bins
    let ds = cli.downsample_minutes;
    let mut bins_x = Vec::new();
    let mut bins_y = Vec::new();
    let mut bins_z = Vec::new();
    let mut bins_month = Vec::new();

    let mut i = 0;
    while i + ds <= records.len() {
        let chunk = &records[i..i + ds];
        let n = chunk.len() as f64;
        let mx: f64 = chunk.iter().map(|r| r.x).sum::<f64>() / n;
        let my: f64 = chunk.iter().map(|r| r.y).sum::<f64>() / n;
        let mz: f64 = chunk.iter().map(|r| r.z).sum::<f64>() / n;

        if mx.is_finite() && my.is_finite() && mz.is_finite() {
            bins_x.push(mx);
            bins_y.push(my);
            bins_z.push(mz);
            bins_month.push(chunk[ds / 2].month);
        }
        i += ds;
    }

    let n_bins = bins_x.len();
    println!("  Downsampled to {} bins ({}min each)", n_bins, ds);

    // Build 32D Takens embedding
    let channels = 4usize;
    let steps = cli.embedding_dim / channels;

    let mut embedded: Vec<Vec<f64>> = Vec::new();
    let mut embed_months: Vec<u8> = Vec::new();

    for w in 0..=n_bins.saturating_sub(steps) {
        let sum_f: f64 = (0..steps)
            .map(|s| {
                let i = w + s;
                (bins_x[i].powi(2) + bins_y[i].powi(2) + bins_z[i].powi(2)).sqrt()
            })
            .sum();
        let mean_f = sum_f / steps as f64;
        if mean_f <= 0.01 || !mean_f.is_finite() { continue; }

        let mut v = vec![0.0; cli.embedding_dim];
        for s in 0..steps {
            let i = w + s;
            let f = (bins_x[i].powi(2) + bins_y[i].powi(2) + bins_z[i].powi(2)).sqrt();
            v[s * channels] = bins_x[i] / mean_f;
            v[s * channels + 1] = bins_y[i] / mean_f;
            v[s * channels + 2] = bins_z[i] / mean_f;
            v[s * channels + 3] = (f - mean_f) / mean_f;
        }
        embedded.push(v);
        embed_months.push(bins_month[w + steps - 1]);
    }

    println!("  Embedded {} vectors ({}D)", embedded.len(), cli.embedding_dim);

    if embedded.len() < 10 {
        anyhow::bail!("Too few embedded vectors");
    }

    let norms = cd_kernel::batch_sliding_associator_norms_parallel(&embedded, cli.embedding_dim);
    println!("  Computed {} associator norms", norms.len());

    // Monthly mean associator
    let mut monthly_sums = vec![0.0f64; 12];
    let mut monthly_counts = vec![0usize; 12];

    for (k, &norm) in norms.iter().enumerate() {
        let month = embed_months[k + 2]; // triple offset
        if month >= 1 && month <= 12 {
            monthly_sums[(month - 1) as usize] += norm;
            monthly_counts[(month - 1) as usize] += 1;
        }
    }

    let monthly_means: Vec<f64> = (0..12)
        .map(|m| {
            if monthly_counts[m] > 0 {
                monthly_sums[m] / monthly_counts[m] as f64
            } else {
                0.0
            }
        })
        .collect();

    println!("\n  Monthly mean associator:");
    let month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    for (m, &mean_a) in monthly_means.iter().enumerate() {
        println!("    {} {}: {:.4} (n={})", month_names[m], cli.year, mean_a, monthly_counts[m]);
    }

    // Detect transitions
    let global_mean = norms.iter().sum::<f64>() / norms.len() as f64;
    let global_std = {
        let var = norms.iter().map(|&a| (a - global_mean).powi(2)).sum::<f64>() / norms.len() as f64;
        var.sqrt()
    };
    let threshold = global_std * 1.5;
    let tw = 20usize; // wider window for slow geomagnetic variations

    let mut transitions = Vec::new();
    if norms.len() > tw * 2 {
        let mut last: Option<usize> = None;
        for i in tw..norms.len().saturating_sub(tw) {
            let pre: f64 = norms[i.saturating_sub(tw)..i].iter().sum::<f64>() / tw as f64;
            let post: f64 = norms[i..(i + tw).min(norms.len())].iter().sum::<f64>()
                / tw.min(norms.len() - i) as f64;
            if (post - pre).abs() > threshold {
                if last.is_some_and(|prev| i.saturating_sub(prev) < tw) { continue; }
                transitions.push(i);
                last = Some(i);
            }
        }
    }

    let transition_months: Vec<u8> = transitions.iter()
        .map(|&t| embed_months.get(t + 2).copied().unwrap_or(0))
        .collect();

    println!("\n  Transitions: {} (months: {:?})", transitions.len(), transition_months);

    let interp = format!(
        "{} {} {}: {} hourly bins, {} transitions. {}",
        cli.observatory, cli.year,
        if transition_months.is_empty() { "No jerk detected" }
        else { "Transitions detected" },
        n_bins, transitions.len(),
        if transition_months.iter().any(|&m| m >= 3 && m <= 6) {
            "Spring 2014 transition detected -- consistent with published 2014 geomagnetic jerk timing."
        } else if transition_months.is_empty() {
            "No transitions above threshold. The geomagnetic field may be too slowly varying for minute-level Takens embedding. Consider daily or weekly downsampling."
        } else {
            "Transitions detected at non-canonical jerk timing. May reflect seasonal or solar-driven variations rather than core-field jerks."
        }
    );
    println!("\n  {}", interp);

    let result = JerkResult {
        observatory: cli.observatory.clone(),
        year: cli.year,
        n_records: records.len(),
        n_bins: n_bins,
        n_embedded: embedded.len(),
        n_transitions: transitions.len(),
        monthly_mean_associator: monthly_means,
        transition_months,
        interpretation: interp,
    };

    if let Some(parent) = cli.out_json.parent() { fs::create_dir_all(parent)?; }
    fs::write(&cli.out_json, serde_json::to_string_pretty(&result)?)?;
    println!("  Wrote {}", cli.out_json.display());

    Ok(())
}

//! Solar cycle cross-correlation: compare 1 AU fleet CD with Voyager CD.
//!
//! Computes yearly CD at 1 AU (from ACE MAG) and correlates with
//! Voyager CD at the same epoch. Tests whether solar-cycle modulation
//! at 1 AU propagates to outer heliosphere CD structure.
//!
//! The expected signal: solar maximum -> higher turbulence -> higher CD
//! at 1 AU. Does this correlate with Voyager CD at ~10-120 AU after
//! accounting for propagation delay (~1 year per AU at ~400 km/s)?
//!
//! Usage:
//!   solar-cycle-crosscorr \
//!     --ace-dir data/external/ace/ \
//!     --v2-lifetime data/output/heliosphere/ablations/v2_lifetime_2au_32d.csv \
//!     --v1-lifetime data/output/heliosphere/ablations/v1_lifetime_2au_32d.csv

use anyhow::{Context, Result};
use clap::Parser;
use serde::Serialize;
use std::{collections::BTreeMap, path::PathBuf};

#[derive(Parser)]
#[command(name = "solar-cycle-crosscorr")]
struct Cli {
    /// Directory with ACE yearly CSVs (from hapi-fetch --yearly)
    #[arg(long)]
    ace_dir: PathBuf,

    /// V2 lifetime CSV (r_center_au, ..., median_associator)
    #[arg(long)]
    v2_lifetime: Option<PathBuf>,

    /// V1 lifetime CSV
    #[arg(long)]
    v1_lifetime: Option<PathBuf>,

    /// Embedding dimension
    #[arg(long, default_value_t = 32)]
    dim: usize,

    /// Output JSON
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/solar_cycle_crosscorr.json"
    )]
    out_json: PathBuf,
}

#[derive(Debug, Serialize)]
struct YearlyCd {
    year: u16,
    n_records: usize,
    n_windows: usize,
    cd_median: f64,
    cd_mean: f64,
    cd_max: f64,
    cd_p90: f64,
}

#[derive(Debug, Serialize)]
struct CrossCorrResult {
    ace_yearly: Vec<YearlyCd>,
    correlation_v1: Option<f64>,
    correlation_v2: Option<f64>,
    solar_cycle_amplitude: f64,
    notes: Vec<String>,
}

fn parse_ace_year(path: &std::path::Path) -> Result<Vec<[f64; 4]>> {
    let content = std::fs::read_to_string(path)?;
    let mut records = Vec::new();
    for line in content.lines() {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 5 || fields[0].starts_with('#') || fields[1].contains("Magnitude") {
            continue;
        }
        let mag: f64 = match fields[1].trim().parse() {
            Ok(v) if v > -1e30 && v < 999.0 => v,
            _ => continue,
        };
        let bx: f64 = match fields[2].trim().parse() {
            Ok(v) if v > -1e30 => v,
            _ => continue,
        };
        let by: f64 = match fields[3].trim().parse() {
            Ok(v) if v > -1e30 => v,
            _ => continue,
        };
        let bz: f64 = match fields[4].trim().parse() {
            Ok(v) if v > -1e30 => v,
            _ => continue,
        };
        records.push([mag, bx, by, bz]);
    }
    Ok(records)
}

fn compute_yearly_cd(records: &[[f64; 4]], dim: usize) -> YearlyCd {
    let steps = dim / 4;
    let mut embeddings: Vec<Vec<f32>> = Vec::new();

    for i in 0..records.len().saturating_sub(steps) {
        let mut v = Vec::with_capacity(dim);
        let mut valid = true;
        for s in 0..steps {
            let r = &records[i + s];
            for &val in r {
                if !val.is_finite() {
                    valid = false;
                    break;
                }
                v.push(val as f32);
            }
            if !valid {
                break;
            }
        }
        if valid && v.len() == dim {
            embeddings.push(v);
        }
    }

    if embeddings.len() < 3 {
        return YearlyCd {
            year: 0,
            n_records: records.len(),
            n_windows: 0,
            cd_median: 0.0,
            cd_mean: 0.0,
            cd_max: 0.0,
            cd_p90: 0.0,
        };
    }

    let norms = cd_kernel::batch_sliding_associator_norms_f32(&embeddings, dim);
    let mut sorted: Vec<f64> = norms.iter().map(|&n| n as f64).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    let median = sorted[n / 2];
    let mean = sorted.iter().sum::<f64>() / n as f64;
    let max = sorted.last().copied().unwrap_or(0.0);
    let p90 = sorted[(n as f64 * 0.9) as usize];

    YearlyCd {
        year: 0,
        n_records: records.len(),
        n_windows: embeddings.len(),
        cd_median: median,
        cd_mean: mean,
        cd_max: max,
        cd_p90: p90,
    }
}

/// Pearson correlation coefficient between two series.
/// Will be used when we match ACE yearly CD to Voyager radial-bin CD by epoch.
#[allow(dead_code)]
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 3 {
        return 0.0;
    }
    let x = &x[..n];
    let y = &y[..n];
    let mx = x.iter().sum::<f64>() / n as f64;
    let my = y.iter().sum::<f64>() / n as f64;
    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for i in 0..n {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    let denom = (vx * vy).sqrt();
    if denom < 1e-30 { 0.0 } else { cov / denom }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    println!("=== Solar Cycle Cross-Correlation ===");
    println!("  ACE dir: {}", cli.ace_dir.display());
    println!("  dim: {}D", cli.dim);

    // Step 1: Compute yearly CD from ACE data
    let mut yearly_cds: BTreeMap<u16, YearlyCd> = BTreeMap::new();

    let entries = std::fs::read_dir(&cli.ace_dir)
        .with_context(|| format!("Reading {}", cli.ace_dir.display()))?;
    let mut ace_files: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().and_then(|e| e.to_str()) == Some("csv")
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("ac_h0_mfi"))
                    .unwrap_or(false)
        })
        .collect();
    ace_files.sort();

    println!("  Found {} ACE yearly files", ace_files.len());

    for path in &ace_files {
        let fname = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        // Extract year from filename: ac_h0_mfi_YYYY.csv
        let year: u16 = fname
            .chars()
            .filter(|c| c.is_ascii_digit())
            .collect::<String>()
            .parse()
            .unwrap_or(0);
        if !(1997..=2025).contains(&year) {
            continue;
        }

        let records = parse_ace_year(path)?;
        if records.len() < 100 {
            println!(
                "  {} {}: only {} records, skipping",
                year,
                fname,
                records.len()
            );
            continue;
        }

        let mut cd = compute_yearly_cd(&records, cli.dim);
        cd.year = year;
        println!(
            "  {}: {} records, {} windows, cd_median={:.3}, cd_p90={:.3}",
            year, cd.n_records, cd.n_windows, cd.cd_median, cd.cd_p90
        );
        yearly_cds.insert(year, cd);
    }

    // Step 2: Load Voyager lifetime profiles
    let mut notes = Vec::new();

    // Solar cycle amplitude: ratio of max/min yearly CD
    let medians: Vec<f64> = yearly_cds.values().map(|c| c.cd_median).collect();
    let solar_amplitude = if !medians.is_empty() {
        let max_cd = medians.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_cd = medians.iter().cloned().fold(f64::INFINITY, f64::min);
        if min_cd > 0.0 { max_cd / min_cd } else { 0.0 }
    } else {
        0.0
    };

    println!("\n  Solar cycle CD amplitude: {:.1}x", solar_amplitude);
    notes.push(format!(
        "ACE yearly CD amplitude: {:.1}x (max/min of median CD)",
        solar_amplitude
    ));

    // Step 3: Cross-correlate with Voyager lifetime profiles
    // The Voyager lifetime CSVs have radial bins with median CD.
    // We need to match by year (or approximate epoch) to compare.
    // For now, report the ACE yearly series and note the correlation approach.

    let correlation_v1 = None;
    let correlation_v2 = None;

    notes.push(
        "Cross-correlation with Voyager requires mapping radial bins to years via V1/V2 trajectory"
            .to_string(),
    );
    notes.push("V2 PLS (1977-2007) provides V_sw for propagation delay estimation".to_string());
    notes.push(
        "Expected: solar max years show higher CD at 1 AU, lagged at Voyager distance".to_string(),
    );

    // Identify solar cycle phases
    if yearly_cds.len() >= 5 {
        let years: Vec<u16> = yearly_cds.keys().copied().collect();
        let cds: Vec<f64> = yearly_cds.values().map(|c| c.cd_median).collect();

        // Find peaks and troughs
        let mut peaks = Vec::new();
        let mut troughs = Vec::new();
        for i in 1..cds.len().saturating_sub(1) {
            if cds[i] > cds[i - 1] && cds[i] > cds[i + 1] {
                peaks.push((years[i], cds[i]));
            }
            if cds[i] < cds[i - 1] && cds[i] < cds[i + 1] {
                troughs.push((years[i], cds[i]));
            }
        }

        if !peaks.is_empty() {
            let peak_strs: Vec<String> = peaks
                .iter()
                .map(|(y, c)| format!("{}({:.1})", y, c))
                .collect();
            println!("  CD peaks: {}", peak_strs.join(", "));
            notes.push(format!("CD peaks at 1 AU: {}", peak_strs.join(", ")));
        }
        if !troughs.is_empty() {
            let trough_strs: Vec<String> = troughs
                .iter()
                .map(|(y, c)| format!("{}({:.1})", y, c))
                .collect();
            println!("  CD troughs: {}", trough_strs.join(", "));
            notes.push(format!("CD troughs at 1 AU: {}", trough_strs.join(", ")));
        }
    }

    let yearly_vec: Vec<YearlyCd> = yearly_cds.into_values().collect();

    let result = CrossCorrResult {
        ace_yearly: yearly_vec,
        correlation_v1,
        correlation_v2,
        solar_cycle_amplitude: solar_amplitude,
        notes,
    };

    let json = serde_json::to_string_pretty(&result)?;
    if let Some(parent) = cli.out_json.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&cli.out_json, &json)?;
    println!("\nResults -> {}", cli.out_json.display());

    Ok(())
}

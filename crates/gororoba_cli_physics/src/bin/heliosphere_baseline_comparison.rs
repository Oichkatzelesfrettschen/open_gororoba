//! 5-way baseline comparison: CD associator vs standard boundary detectors.
//!
//! Runs all detectors on the same data with the same curated labels,
//! reporting detection rate / false alarm rate / offset for each.
//! This is the table that shows what the CD associator adds beyond
//! standard diagnostics.

use anyhow::{Context, Result};
use chrono::{Datelike, NaiveDate};
use clap::Parser;
use data_core::{
    catalogs::themis::parse_themis_fgm_hapi_csv_minutes,
    crossing_lists::{match_crossings, parse_crossing_list},
};
use serde::Serialize;
use spectral_core::boundary_detectors::{
    bmag_gradient_crossings, pvi_crossings, spectral_entropy_crossings,
};
use std::{fs, path::PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "heliosphere-baseline-comparison",
    about = "5-way baseline comparison against curated labels"
)]
struct Cli {
    #[arg(long)]
    start_date: String,
    #[arg(long)]
    end_date: String,
    #[arg(long, default_value_t = 32)]
    embedding_dim: usize,
    #[arg(long, default_value_t = 200.0)]
    max_bmag: f64,
    #[arg(long)]
    crossing_list: PathBuf,
    #[arg(long)]
    crossing_probe: Option<String>,
    #[arg(long, default_value_t = 20)]
    match_tolerance_minutes: usize,
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/baseline_comparison.json"
    )]
    out_json: PathBuf,
    #[arg(long, default_value = "data/external")]
    data_dir: PathBuf,
}

#[derive(Debug, Serialize)]
struct DetectorResult {
    name: String,
    n_detections: usize,
    detection_rate: f64,
    false_alarm_rate: f64,
    mean_offset_minutes: f64,
    median_offset_minutes: f64,
}

#[derive(Debug, Serialize)]
struct ComparisonResult {
    start_date: String,
    end_date: String,
    n_curated_crossings: usize,
    detectors: Vec<DetectorResult>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let start = NaiveDate::parse_from_str(&cli.start_date, "%Y-%m-%d")
        .with_context(|| format!("bad start: {}", cli.start_date))?;
    let end = NaiveDate::parse_from_str(&cli.end_date, "%Y-%m-%d")
        .with_context(|| format!("bad end: {}", cli.end_date))?;

    println!("=== 5-Way Baseline Comparison ===");
    println!("  Window: {} to {}", start, end);

    // Load data
    let mut bx = Vec::new();
    let mut by = Vec::new();
    let mut bz = Vec::new();
    let mut elapsed_hrs = Vec::new();

    for offset in 0..=(end - start).num_days() {
        let date = start + chrono::Duration::days(offset);
        let path = cli.data_dir.join("themis").join(format!(
            "tha_fgm_{:04}_{:03}.csv",
            date.year(),
            date.ordinal()
        ));
        if path.exists() {
            let content = fs::read_to_string(&path)?;
            for r in parse_themis_fgm_hapi_csv_minutes(&content, "THA") {
                if r.b_magnitude <= cli.max_bmag {
                    bx.push(r.bx_gse);
                    by.push(r.by_gse);
                    bz.push(r.bz_gse);
                    // Elapsed hours from midnight of start_date
                    let day_offset = (r.year as f64 - start.year() as f64) * 365.25
                        + (r.doy as f64 - start.ordinal() as f64);
                    elapsed_hrs.push(day_offset * 24.0 + r.hour as f64 + r.minute as f64 / 60.0);
                }
            }
        }
    }

    let n = bx.len();
    println!("  Loaded {} minutes", n);

    // Load curated crossings
    let content = fs::read_to_string(&cli.crossing_list)?;
    let events = parse_crossing_list(&content, cli.crossing_probe.as_deref(), &start, &end);
    let curated_hours: Vec<f64> = events.iter().map(|e| e.elapsed_hours).collect();
    println!("  Curated crossings: {}", curated_hours.len());

    if curated_hours.is_empty() || n < 20 {
        anyhow::bail!("Insufficient data or crossings");
    }

    let tol_hours = cli.match_tolerance_minutes as f64 / 60.0;
    let mut detectors: Vec<DetectorResult> = Vec::new();

    // --- Detector 1: |B| gradient ---
    println!("\n[1/4] |B| gradient (10 nT, 10 min window)...");
    let bmag_idx = bmag_gradient_crossings(&bx, &by, &bz, 10, 10.0);
    let bmag_hours: Vec<f64> = bmag_idx
        .iter()
        .map(|&i| elapsed_hrs[i.min(n - 1)])
        .collect();
    let (bm_matched, bm_offsets) = match_crossings(&curated_hours, &bmag_hours, tol_hours);
    let bm_rev_matched = bmag_hours
        .iter()
        .filter(|&&th| curated_hours.iter().any(|&ch| (ch - th).abs() < tol_hours))
        .count();
    detectors.push(make_result(
        "|B| gradient",
        &curated_hours,
        &bmag_hours,
        bm_matched,
        bm_rev_matched,
        &bm_offsets,
    ));

    // --- Detector 2: PVI ---
    println!("[2/4] PVI (lag=1, threshold=4.0)...");
    let pvi_idx = pvi_crossings(&bx, &by, &bz, 1, 4.0, 10);
    let pvi_hours: Vec<f64> = pvi_idx.iter().map(|&i| elapsed_hrs[i.min(n - 1)]).collect();
    let (pv_matched, pv_offsets) = match_crossings(&curated_hours, &pvi_hours, tol_hours);
    let pv_rev = pvi_hours
        .iter()
        .filter(|&&th| curated_hours.iter().any(|&ch| (ch - th).abs() < tol_hours))
        .count();
    detectors.push(make_result(
        "PVI",
        &curated_hours,
        &pvi_hours,
        pv_matched,
        pv_rev,
        &pv_offsets,
    ));

    // --- Detector 3: Spectral entropy ---
    println!("[3/4] Spectral entropy (window=30, threshold=0.1)...");
    let se_idx = spectral_entropy_crossings(&bx, &by, &bz, 30, 0.1);
    let se_hours: Vec<f64> = se_idx.iter().map(|&i| elapsed_hrs[i.min(n - 1)]).collect();
    let (se_matched, se_offsets) = match_crossings(&curated_hours, &se_hours, tol_hours);
    let se_rev = se_hours
        .iter()
        .filter(|&&th| curated_hours.iter().any(|&ch| (ch - th).abs() < tol_hours))
        .count();
    detectors.push(make_result(
        "Spectral entropy",
        &curated_hours,
        &se_hours,
        se_matched,
        se_rev,
        &se_offsets,
    ));

    // --- Detector 4: CD associator ---
    println!("[4/4] 32D CD associator...");

    let channels = 4usize;
    let steps = cli.embedding_dim / channels;
    let mut embedded: Vec<Vec<f64>> = Vec::new();
    let mut embed_meta: Vec<usize> = Vec::new();

    for w in 0..=n.saturating_sub(steps) {
        let sum_b: f64 = (0..steps)
            .map(|s| (bx[w + s].powi(2) + by[w + s].powi(2) + bz[w + s].powi(2)).sqrt())
            .sum();
        let mean_b = sum_b / steps as f64;
        if mean_b <= 0.01 || !mean_b.is_finite() {
            continue;
        }

        let mut v = vec![0.0; cli.embedding_dim];
        for s in 0..steps {
            let i = w + s;
            let bmag = (bx[i].powi(2) + by[i].powi(2) + bz[i].powi(2)).sqrt();
            v[s * channels] = bx[i] / mean_b;
            v[s * channels + 1] = by[i] / mean_b;
            v[s * channels + 2] = bz[i] / mean_b;
            v[s * channels + 3] = (bmag - mean_b) / mean_b;
        }
        embedded.push(v);
        embed_meta.push(w + steps - 1);
    }

    let assoc_norms =
        cd_kernel::batch_sliding_associator_norms_parallel(&embedded, cli.embedding_dim);
    let assoc_indices: Vec<usize> = (0..assoc_norms.len()).map(|k| embed_meta[k + 2]).collect();

    // Detect transitions
    let global_mean: f64 = assoc_norms.iter().sum::<f64>() / assoc_norms.len() as f64;
    let global_std: f64 = {
        let var = assoc_norms
            .iter()
            .map(|&a| (a - global_mean).powi(2))
            .sum::<f64>()
            / assoc_norms.len() as f64;
        var.sqrt()
    };
    let threshold = global_std * 1.5;
    let trans_window = 10usize;

    let mut cd_indices: Vec<usize> = Vec::new();
    if assoc_norms.len() > trans_window * 2 {
        for i in trans_window..assoc_norms.len().saturating_sub(trans_window) {
            let pre: f64 = assoc_norms[i.saturating_sub(trans_window)..i]
                .iter()
                .sum::<f64>()
                / trans_window as f64;
            let post: f64 = assoc_norms[i..(i + trans_window).min(assoc_norms.len())]
                .iter()
                .sum::<f64>()
                / trans_window.min(assoc_norms.len() - i) as f64;
            if (post - pre).abs() > threshold {
                let dominated = cd_indices
                    .last()
                    .is_some_and(|&prev| i.saturating_sub(prev) < trans_window);
                if !dominated {
                    cd_indices.push(assoc_indices[i]);
                }
            }
        }
    }

    let cd_hours: Vec<f64> = cd_indices
        .iter()
        .map(|&i| elapsed_hrs[i.min(n - 1)])
        .collect();
    let (cd_matched, cd_offsets) = match_crossings(&curated_hours, &cd_hours, tol_hours);
    let cd_rev = cd_hours
        .iter()
        .filter(|&&th| curated_hours.iter().any(|&ch| (ch - th).abs() < tol_hours))
        .count();
    detectors.push(make_result(
        "CD associator (32D)",
        &curated_hours,
        &cd_hours,
        cd_matched,
        cd_rev,
        &cd_offsets,
    ));

    // --- Hybrid union: CD OR PVI ---
    let mut union_hours: Vec<f64> = cd_hours.clone();
    for &ph in &pvi_hours {
        if !union_hours
            .iter()
            .any(|&uh| (uh - ph).abs() < tol_hours * 0.5)
        {
            union_hours.push(ph);
        }
    }
    union_hours.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (union_matched, union_offsets) = match_crossings(&curated_hours, &union_hours, tol_hours);
    let union_rev = union_hours
        .iter()
        .filter(|&&th| curated_hours.iter().any(|&ch| (ch - th).abs() < tol_hours))
        .count();
    detectors.push(make_result(
        "CD+PVI UNION",
        &curated_hours,
        &union_hours,
        union_matched,
        union_rev,
        &union_offsets,
    ));

    // --- Print results ---
    println!(
        "\n=== Results against {} curated crossings ===\n",
        curated_hours.len()
    );
    println!(
        "{:<25} {:>8} {:>10} {:>8} {:>8} {:>8}",
        "Detector", "N det", "Detection", "FA", "Mean", "Median"
    );
    println!("{}", "-".repeat(70));
    for d in &detectors {
        println!(
            "{:<25} {:>8} {:>9.1}% {:>7.1}% {:>7.1}m {:>7.1}m",
            d.name,
            d.n_detections,
            d.detection_rate * 100.0,
            d.false_alarm_rate * 100.0,
            d.mean_offset_minutes,
            d.median_offset_minutes
        );
    }

    let result = ComparisonResult {
        start_date: cli.start_date.clone(),
        end_date: cli.end_date.clone(),
        n_curated_crossings: curated_hours.len(),
        detectors,
    };

    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.out_json, serde_json::to_string_pretty(&result)?)?;
    println!("\nWrote {}", cli.out_json.display());

    Ok(())
}

fn make_result(
    name: &str,
    curated: &[f64],
    detected: &[f64],
    matched: usize,
    rev_matched: usize,
    offsets: &[f64],
) -> DetectorResult {
    let det_rate = if curated.is_empty() {
        0.0
    } else {
        matched as f64 / curated.len() as f64
    };
    let fa_rate = if detected.is_empty() {
        0.0
    } else {
        (detected.len() - rev_matched) as f64 / detected.len() as f64
    };
    let mean_off = if offsets.is_empty() {
        0.0
    } else {
        offsets.iter().sum::<f64>() / offsets.len() as f64
    };
    let med_off = if offsets.is_empty() {
        0.0
    } else {
        offsets[offsets.len() / 2]
    };

    DetectorResult {
        name: name.to_string(),
        n_detections: detected.len(),
        detection_rate: det_rate,
        false_alarm_rate: fa_rate,
        mean_offset_minutes: mean_off,
        median_offset_minutes: med_off,
    }
}

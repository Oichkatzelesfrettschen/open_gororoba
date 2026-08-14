//! Takens tau (lag) sensitivity sweep on THEMIS E-237 data (E-240).
//!
//! WHY: Takens' theorem requires the embedding lag tau to be chosen relative
//! to the characteristic timescale of the underlying dynamical system.  A
//! fixed tau may over- or under-sample the phase-space attractor when applied
//! across missions spanning very different plasma environments.  This sweep
//! tests whether F1 degrades gracefully as tau varies over an order of
//! magnitude, establishing the temporal robustness of the 32D embedding.
//!
//! WHAT: Runs the identical CD associator evaluation pipeline from E-237
//! (THEMIS-A FGM, Staples+2020 catalog, 224 events) for tau in {1, 2, 5, 10}
//! minutes.  The |B|-gradient+rotation baseline is tau-independent and serves
//! as a constant reference line.  Reports P/R/F1 at each tau value (C-1633).
//!
//! HOW:
//!   cargo run --release -p gororoba_cli_physics \
//!     --bin heliosphere -- takens-tau-sweep \
//!     --start-date 2016-08-29 --n-days 7 --probe a \
//!     --staples-catalog data/external/crossing_lists/themis_mp_crossings_v2.txt

use anyhow::{Context, Result};
use chrono::{Datelike, NaiveDate, NaiveDateTime, TimeZone, Utc};
use clap::Args;
use data_core::catalogs::{
    mms::{MmsEventInterval, detect_magnetopause_crossings_filtered},
    themis::{
        ThemisFgmMinuteRecord, parse_staples_crossing_catalog, parse_themis_fgm_hapi_csv_minutes,
    },
};
use serde::Serialize;
use spectral_core::boundary_metrics;
use std::{fs, path::PathBuf};

/// MAD scale factor for the CD sliding-window transition detector.
/// Fixed prior to evaluation on any test data. See plan P6A.S1, task 1.7.
const MAD_SCALE_FACTOR: f64 = 1.5;

// ============================================================================
// CLI
// ============================================================================

#[derive(Args, Debug)]
pub struct Cli {
    /// Start date for FGM data window (YYYY-MM-DD).
    #[arg(long, default_value = "2016-08-29")]
    start_date: String,

    /// Number of days to analyze.
    #[arg(long, default_value_t = 7)]
    n_days: u32,

    /// THEMIS probe letter.
    #[arg(long, default_value = "a")]
    probe: String,

    /// Path to the Staples et al. (2020) catalog .txt file.
    #[arg(
        long,
        default_value = "data/external/crossing_lists/themis_mp_crossings_v2.txt"
    )]
    staples_catalog: PathBuf,

    /// Evaluation padding around each catalog crossing (minutes).
    #[arg(long, default_value_t = 10)]
    pad_minutes: i64,

    /// Embedding dimension (fixed across all tau values).
    #[arg(long, default_value_t = 32)]
    embedding_dim: usize,

    /// Instrument noise floor (nT) -- THEMIS FGS default.
    #[arg(long, default_value_t = 0.5)]
    bmag_noise_floor: f64,

    /// Window size (minutes) for gradient crossing detector.
    #[arg(long, default_value_t = 10)]
    crossing_window_minutes: usize,

    /// |B| gradient threshold (nT) for gradient detector.
    #[arg(long, default_value_t = 5.0)]
    bmag_gradient_threshold: f64,

    /// Rotation threshold (degrees) for gradient detector.
    #[arg(long, default_value_t = 30.0)]
    rotation_threshold_deg: f64,

    /// Max |B| filter (nT) for inner-magnetosphere transit removal.
    #[arg(long, default_value_t = 100.0)]
    max_bmag: f64,

    /// Data cache root directory.
    #[arg(long, default_value = "data/external")]
    data_dir: PathBuf,

    /// Output JSON path.
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/takens_tau_sweep.json"
    )]
    out_json: PathBuf,
}

// ============================================================================
// Output
// ============================================================================

#[derive(Debug, Serialize)]
struct TauResult {
    tau_minutes: usize,
    embedding_window_minutes: usize,
    n_cd_detections: usize,
    n_gap_skipped: usize,
    cd_precision: f64,
    cd_recall: f64,
    cd_f1: f64,
}

#[derive(Debug, Serialize)]
struct TauSweepResults {
    start_date: String,
    n_days: u32,
    probe: String,
    embedding_dim: usize,
    n_catalog_events: usize,
    n_fgm_minutes: usize,
    gradient_precision: f64,
    gradient_recall: f64,
    gradient_f1: f64,
    gradient_n_detections: usize,
    tau_results: Vec<TauResult>,
    ascii_table: String,
}

// ============================================================================
// Shared detection helpers
// ============================================================================

fn detect_crossings_adapted(
    records: &[ThemisFgmMinuteRecord],
    window: usize,
    bmag_threshold: f64,
    rotation_deg: Option<f64>,
) -> Vec<usize> {
    use data_core::catalogs::mms::MmsFgmMinuteRecord;
    let adapted: Vec<MmsFgmMinuteRecord> = records
        .iter()
        .map(|r| MmsFgmMinuteRecord {
            year: r.year,
            doy: r.doy,
            hour: r.hour,
            minute: r.minute,
            elapsed_hours: r.elapsed_hours,
            bx_gse: r.bx_gse,
            by_gse: r.by_gse,
            bz_gse: r.bz_gse,
            b_magnitude: r.b_magnitude,
        })
        .collect();
    detect_magnetopause_crossings_filtered(&adapted, window, bmag_threshold, rotation_deg)
}

fn hours_to_unix(origin: &NaiveDateTime, h: f64) -> i64 {
    Utc.from_utc_datetime(origin).timestamp() + (h * 3600.0) as i64
}

fn event_midpoint_unix(ev: &MmsEventInterval) -> i64 {
    let mid = ev.start + (ev.end - ev.start) / 2;
    Utc.from_utc_datetime(&mid).timestamp()
}

fn eval_detection_hours(
    detection_hours: &[f64],
    reference_midnight: &NaiveDateTime,
    catalog: &[MmsEventInterval],
    window_secs: i64,
) -> (f64, f64, f64) {
    let detection_unix: Vec<i64> = detection_hours
        .iter()
        .map(|&h| hours_to_unix(reference_midnight, h))
        .collect();
    let event_unix: Vec<i64> = catalog.iter().map(event_midpoint_unix).collect();
    boundary_metrics::precision_recall_f1(&detection_unix, &event_unix, window_secs)
}

fn run_cd_for_tau(
    all_minutes: &[ThemisFgmMinuteRecord],
    embedding_dim: usize,
    tau: usize,
    crossing_window: usize,
    noise_floor: f64,
) -> (Vec<f64>, usize) {
    let channels: usize = 4;
    let steps = embedding_dim / channels;
    let window_rows = (steps - 1) * tau + 1;

    if all_minutes.len() < window_rows + 2 {
        return (Vec::new(), 0);
    }

    let expected_span_hours = (steps - 1) as f64 * tau as f64 / 60.0;
    let gap_tolerance_hours = 2.0 * tau as f64 / 60.0;
    let max_window_span_hours = expected_span_hours + gap_tolerance_hours;

    let mut embedded: Vec<Vec<f64>> = Vec::new();
    let mut embed_meta: Vec<usize> = Vec::new();
    let mut n_gap_skipped: usize = 0;

    for w_start in 0..=(all_minutes.len() - window_rows) {
        let sample_indices: Vec<usize> = (0..steps).map(|s| w_start + s * tau).collect();

        let first_h = all_minutes[*sample_indices.first().unwrap()].elapsed_hours;
        let last_h = all_minutes[*sample_indices.last().unwrap()].elapsed_hours;
        if last_h - first_h > max_window_span_hours {
            n_gap_skipped += 1;
            continue;
        }

        let sum_b: f64 = sample_indices
            .iter()
            .map(|&i| all_minutes[i].b_magnitude)
            .sum();
        let local_mean_b = sum_b / steps as f64;
        if local_mean_b <= 0.0 || !local_mean_b.is_finite() {
            continue;
        }
        let denom = local_mean_b.max(noise_floor);
        let mut v = vec![0.0; embedding_dim];
        for (s, &ri) in sample_indices.iter().enumerate() {
            let rec = &all_minutes[ri];
            v[s * channels] = rec.bx_gse / denom;
            v[s * channels + 1] = rec.by_gse / denom;
            v[s * channels + 2] = rec.bz_gse / denom;
            v[s * channels + 3] = (rec.b_magnitude - local_mean_b) / denom;
        }
        embedded.push(v);
        embed_meta.push(*sample_indices.last().unwrap());
    }

    let associators = cd_kernel::batch_sliding_associator_norms_parallel(&embedded, embedding_dim);

    let trans_window = crossing_window.max(5);
    let mut cd_hours: Vec<f64> = Vec::new();

    if associators.len() > trans_window * 2 {
        let global_mean: f64 = associators.iter().sum::<f64>() / associators.len() as f64;
        let global_std: f64 = {
            let var = associators
                .iter()
                .map(|&a| (a - global_mean).powi(2))
                .sum::<f64>()
                / associators.len() as f64;
            var.sqrt()
        };
        let threshold = global_std * MAD_SCALE_FACTOR;
        let half = trans_window;
        let assoc_minute_indices: Vec<usize> =
            (0..associators.len()).map(|k| embed_meta[k + 2]).collect();
        let mut last_trans: Option<usize> = None;
        for i in half..associators.len().saturating_sub(half) {
            let pre_mean: f64 =
                associators[i.saturating_sub(half)..i].iter().sum::<f64>() / half as f64;
            let post_mean: f64 = associators[i..(i + half).min(associators.len())]
                .iter()
                .sum::<f64>()
                / half.min(associators.len() - i) as f64;
            let jump = (post_mean - pre_mean).abs();
            if jump > threshold {
                let dominated =
                    last_trans.is_some_and(|prev| i.saturating_sub(prev) < trans_window);
                if !dominated {
                    cd_hours.push(all_minutes[assoc_minute_indices[i]].elapsed_hours);
                    last_trans = Some(i);
                }
            }
        }
    }

    (cd_hours, n_gap_skipped)
}

fn build_ascii_table(results: &TauSweepResults) -> String {
    let mut s = String::new();
    s.push_str(
        "Takens tau sensitivity sweep (THEMIS E-237, 224 Staples+2020 events)\n\
         ======================================================================\n",
    );
    s.push_str(&format!(
        "{:<12} {:>10} {:>8} {:>8} {:>8} {:>10}\n",
        "Method/tau", "Window(min)", "P", "R", "F1", "Detections"
    ));
    s.push_str(&"-".repeat(60));
    s.push('\n');
    s.push_str(&format!(
        "{:<12} {:>10} {:>8.3} {:>8.3} {:>8.3} {:>10}\n",
        "gradient",
        "--",
        results.gradient_precision,
        results.gradient_recall,
        results.gradient_f1,
        results.gradient_n_detections,
    ));
    for r in &results.tau_results {
        s.push_str(&format!(
            "{:<12} {:>10} {:>8.3} {:>8.3} {:>8.3} {:>10}\n",
            format!("CD tau={}min", r.tau_minutes),
            r.embedding_window_minutes,
            r.cd_precision,
            r.cd_recall,
            r.cd_f1,
            r.n_cd_detections,
        ));
    }
    s.push_str(&"-".repeat(60));
    s.push('\n');
    s
}

// ============================================================================
// main
// ============================================================================

pub fn run(cli: Cli) -> Result<()> {

    let start = NaiveDate::parse_from_str(&cli.start_date, "%Y-%m-%d")
        .with_context(|| format!("bad start_date: {}", cli.start_date))?;
    let probe_char = cli
        .probe
        .trim()
        .to_lowercase()
        .chars()
        .next()
        .context("--probe must be a single letter")?;
    let probe_upper = cli.probe.trim().to_uppercase();
    let spacecraft = format!("TH{probe_upper}");

    println!(
        "=== Takens tau Sensitivity Sweep (E-240) ===\n\
         Window: {} + {} days, Spacecraft: {}\n\
         Embedding: {}D, tau sweep: {{1, 2, 5, 10}} min\n",
        start, cli.n_days, spacecraft, cli.embedding_dim
    );

    // -----------------------------------------------------------------------
    // Load THEMIS FGM data (identical to E-237 -- data must already be cached)
    // -----------------------------------------------------------------------
    println!("[1/3] Loading cached THEMIS FGM data...");

    let themis_dir = cli.data_dir.join("themis");
    let mut all_minutes: Vec<ThemisFgmMinuteRecord> = Vec::new();

    let end = start + chrono::Duration::days(cli.n_days as i64 - 1);
    for day_offset in 0..cli.n_days {
        let date = start + chrono::Duration::days(day_offset as i64);
        let fname = format!(
            "{}_fgm_{:04}_{:03}.csv",
            spacecraft.to_lowercase(),
            date.year(),
            date.ordinal()
        );
        let path = themis_dir.join(&fname);
        if !path.exists() {
            eprintln!(
                "  Warning: {} not found -- run heliosphere-themis-staples-labeled first",
                fname
            );
            continue;
        }
        let content = fs::read_to_string(&path)?;
        let mut records = parse_themis_fgm_hapi_csv_minutes(&content, &spacecraft);
        println!(
            "  {} DOY {}: {} minutes",
            spacecraft,
            date.ordinal(),
            records.len()
        );
        all_minutes.append(&mut records);
    }

    if all_minutes.is_empty() {
        anyhow::bail!(
            "No THEMIS data -- run heliosphere-themis-staples-labeled first to populate cache"
        );
    }

    all_minutes.retain(|r| r.b_magnitude <= cli.max_bmag);
    all_minutes.sort_by(|a, b| {
        a.year
            .cmp(&b.year)
            .then(a.doy.cmp(&b.doy))
            .then(a.hour.cmp(&b.hour))
            .then(a.minute.cmp(&b.minute))
    });

    let (fy, fd, fh, fm) = (
        all_minutes[0].year,
        all_minutes[0].doy,
        all_minutes[0].hour,
        all_minutes[0].minute,
    );
    for rec in &mut all_minutes {
        let day_diff = (rec.year as f64 - fy as f64) * 365.25 + (rec.doy as f64 - fd as f64);
        rec.elapsed_hours = day_diff * 24.0
            + (rec.hour as f64 - fh as f64)
            + (rec.minute as f64 - fm as f64) / 60.0;
    }

    let reference_midnight = NaiveDate::from_yo_opt(fy as i32, fd as u32)
        .and_then(|d| d.and_hms_opt(fh as u32, fm as u32, 0))
        .context("reference midnight")?;

    println!("  Total: {} minutes", all_minutes.len());

    // -----------------------------------------------------------------------
    // Load Staples catalog
    // -----------------------------------------------------------------------
    println!("[2/3] Loading Staples+2020 catalog...");

    let catalog_content = fs::read_to_string(&cli.staples_catalog)
        .with_context(|| format!("reading catalog: {}", cli.staples_catalog.display()))?;

    let catalog =
        parse_staples_crossing_catalog(&catalog_content, probe_char, start, end, cli.pad_minutes);
    println!("  {} catalog intervals", catalog.len());

    // -----------------------------------------------------------------------
    // Gradient baseline (tau-independent)
    // -----------------------------------------------------------------------
    println!("[3/3] Running gradient baseline and tau sweep...");

    let gradient_indices = detect_crossings_adapted(
        &all_minutes,
        cli.crossing_window_minutes,
        cli.bmag_gradient_threshold,
        Some(cli.rotation_threshold_deg),
    );
    let gradient_hours: Vec<f64> = gradient_indices
        .iter()
        .map(|&i| all_minutes[i].elapsed_hours)
        .collect();
    let eval_window_secs = cli.pad_minutes * 60;
    let (grad_p, grad_r, grad_f1) = eval_detection_hours(
        &gradient_hours,
        &reference_midnight,
        &catalog,
        eval_window_secs,
    );
    println!(
        "  Gradient: P={:.3} R={:.3} F1={:.3} ({} detections)",
        grad_p,
        grad_r,
        grad_f1,
        gradient_hours.len()
    );

    // -----------------------------------------------------------------------
    // CD associator tau sweep
    // -----------------------------------------------------------------------
    let tau_values: &[usize] = &[1, 2, 5, 10];
    let mut tau_results: Vec<TauResult> = Vec::new();

    for &tau in tau_values {
        let channels: usize = 4;
        let steps = cli.embedding_dim / channels;
        let window_minutes = (steps - 1) * tau + 1;

        let (cd_hours, n_gap) = run_cd_for_tau(
            &all_minutes,
            cli.embedding_dim,
            tau,
            cli.crossing_window_minutes,
            cli.bmag_noise_floor,
        );
        let (cd_p, cd_r, cd_f1) =
            eval_detection_hours(&cd_hours, &reference_midnight, &catalog, eval_window_secs);

        println!(
            "  tau={:2}min  window={:3}min  P={:.3}  R={:.3}  F1={:.3}  ({} det, {} gap-skipped)",
            tau,
            window_minutes,
            cd_p,
            cd_r,
            cd_f1,
            cd_hours.len(),
            n_gap
        );

        tau_results.push(TauResult {
            tau_minutes: tau,
            embedding_window_minutes: window_minutes,
            n_cd_detections: cd_hours.len(),
            n_gap_skipped: n_gap,
            cd_precision: cd_p,
            cd_recall: cd_r,
            cd_f1,
        });
    }

    let sweep = TauSweepResults {
        start_date: cli.start_date.clone(),
        n_days: cli.n_days,
        probe: cli.probe.clone(),
        embedding_dim: cli.embedding_dim,
        n_catalog_events: catalog.len(),
        n_fgm_minutes: all_minutes.len(),
        gradient_precision: grad_p,
        gradient_recall: grad_r,
        gradient_f1: grad_f1,
        gradient_n_detections: gradient_hours.len(),
        tau_results,
        ascii_table: String::new(), // filled below
    };

    let ascii = build_ascii_table(&sweep);
    println!("\n{}", ascii);

    let mut sweep = sweep;
    sweep.ascii_table = ascii;

    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.out_json, serde_json::to_string_pretty(&sweep)?)?;
    println!("Wrote {}", cli.out_json.display());

    Ok(())
}

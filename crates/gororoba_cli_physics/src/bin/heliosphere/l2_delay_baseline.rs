//! L2-delay baseline for the CD associator dimensional ablation.
//!
//! WHY: The pre-registered ablation (ablation-preregistered-v1, commit 455d4745)
//! requires an L2-delay change-point detector as a lower bound against which the
//! CD associator must exceed F1 by >= 0.05 (Prediction P1).  The L2-delay score
//! is ||v_t - v_{t-lag}||_2, the Euclidean distance between consecutive Takens
//! delay vectors.  It is the simplest possible "something changed" detector and
//! has no algebraic structure beyond what the embedding geometry provides.
//!
//! WHAT: Downloads THEMIS-A FGM for the same default 7-day window as
//! heliosphere-themis-staples-labeled, builds the same 32D delay vectors,
//! computes the L2-delay score series, applies the same MAD_SCALE_FACTOR=1.5
//! threshold, and evaluates F1 against the Staples+2020 catalog.
//!
//! HOW:
//!   cargo run --release -p gororoba_cli_physics \
//!     --bin heliosphere -- l2-delay-baseline \
//!     --start-date 2016-08-29 --n-days 7 \
//!     --staples-catalog data/external/crossing_lists/themis_mp_crossings_v2.txt

use anyhow::{Context, Result};
use chrono::{Datelike, NaiveDate, NaiveDateTime, TimeZone, Utc};
use clap::Args;
use data_core::{
    catalogs::{
        mms::MmsEventInterval,
        themis::{
            ThemisFgmMinuteRecord, parse_staples_crossing_catalog,
            parse_themis_fgm_hapi_csv_minutes,
        },
    },
    download_stack::{DownloadBackend, DownloadStack, TransferRequest},
};
use serde::Serialize;
use spectral_core::boundary_metrics;
use std::{fs, path::PathBuf};

/// MAD scale factor for the L2-delay sliding-window transition detector.
/// Identical to heliosphere-themis-staples-labeled; fixed prior to any evaluation.
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

    /// THEMIS probe: a, b, c, d, e.
    #[arg(long, default_value = "a")]
    probe: String,

    /// Path to the Staples et al. (2020) catalog .txt file.
    #[arg(
        long,
        default_value = "data/external/crossing_lists/themis_mp_crossings_v2.txt"
    )]
    staples_catalog: PathBuf,

    /// Padding around each instantaneous crossing timestamp (minutes).
    #[arg(long, default_value_t = 10)]
    pad_minutes: i64,

    /// Only include catalog crossings with FOM at or above this value.
    #[arg(long, default_value_t = 0.0)]
    min_fom: f64,

    /// Embedding dimension for Takens delay embedding.
    #[arg(long, default_value_t = 32)]
    embedding_dim: usize,

    /// Takens lag in minutes.
    #[arg(long, default_value_t = 1)]
    takens_lag: usize,

    /// Window size (minutes) for transition suppression (same as crossing_window_minutes).
    #[arg(long, default_value_t = 10)]
    crossing_window_minutes: usize,

    /// Maximum |B| (nT).  Filters deep magnetosphere transits.
    #[arg(long, default_value_t = 100.0)]
    max_bmag: f64,

    /// Instrument noise floor (nT) for denominator clamping in embedding normalization.
    #[arg(long, default_value_t = 0.5)]
    bmag_noise_floor: f64,

    /// Data cache root directory.
    #[arg(long, default_value = "data/external")]
    data_dir: PathBuf,

    /// Output JSON path.
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/l2_delay_baseline_eval.json"
    )]
    out_json: PathBuf,
}

// ============================================================================
// Output structures
// ============================================================================

#[derive(Debug, Serialize)]
struct L2DelayResults {
    start_date: String,
    n_days: u32,
    probe: String,
    embedding_dim: usize,
    method: String,
    n_catalog_events: usize,
    n_fgm_minutes: usize,
    n_detections: usize,
    precision: f64,
    recall: f64,
    f1: f64,
    /// Block-bootstrap 95% CI for F1: (mean, ci_lo, ci_hi).
    bootstrap_ci_mean: f64,
    bootstrap_ci_lo: f64,
    bootstrap_ci_hi: f64,
    series_start_unix: i64,
    series_end_unix: i64,
}

// ============================================================================
// Helpers
// ============================================================================

fn hours_to_unix(origin: &NaiveDateTime, h: f64) -> i64 {
    let base = Utc.from_utc_datetime(origin).timestamp();
    base + (h * 3600.0) as i64
}

fn event_midpoint_unix(ev: &MmsEventInterval) -> i64 {
    let mid = ev.start + (ev.end - ev.start) / 2;
    Utc.from_utc_datetime(&mid).timestamp()
}

// ============================================================================
// main
// ============================================================================

pub fn run(cli: Cli) -> Result<()> {

    let start = NaiveDate::parse_from_str(&cli.start_date, "%Y-%m-%d")
        .with_context(|| format!("invalid start_date: {}", cli.start_date))?;
    let end = start + chrono::Duration::days(cli.n_days as i64 - 1);
    let probe_char = cli
        .probe
        .trim()
        .to_lowercase()
        .chars()
        .next()
        .context("--probe must be a single letter (a-e)")?;
    let probe_upper = cli.probe.trim().to_uppercase();
    let spacecraft = format!("TH{probe_upper}");

    println!(
        "=== L2-Delay Baseline (CD Ablation) ===\n\
         Window: {} to {} ({} days)\n\
         Probe: {}  Embedding: {}D, lag={}min\n",
        start, end, cli.n_days, spacecraft, cli.embedding_dim, cli.takens_lag
    );

    // -----------------------------------------------------------------------
    // Step 1: Fetch THEMIS FGM data (identical to heliosphere-themis-staples-labeled)
    // -----------------------------------------------------------------------
    println!("[1/4] Fetching THEMIS-{} FGM data...", probe_upper);

    let themis_dir = cli.data_dir.join("themis");
    fs::create_dir_all(&themis_dir)?;

    let gse_param = format!("{}_fgs_gse", spacecraft.to_lowercase());
    let dataset = format!("{}_L2_FGM@0", spacecraft);

    for day_offset in 0..cli.n_days {
        let date = start + chrono::Duration::days(day_offset as i64);
        let doy = date.ordinal();
        let fname = format!(
            "{}_fgm_{:04}_{:03}.csv",
            spacecraft.to_lowercase(),
            date.year(),
            doy
        );
        let output = themis_dir.join(&fname);
        if output.exists() {
            println!("  DOY {doy}: cached");
            continue;
        }
        let t_min = format!("{}T00:00:00Z", date);
        let t_max = format!("{}T23:59:59Z", date);
        let url = format!(
            "https://cdaweb.gsfc.nasa.gov/hapi/data\
             ?id={dataset}&time.min={t_min}&time.max={t_max}\
             &format=csv&parameters=Time,{gse_param}"
        );
        println!("  DOY {doy}: downloading...");
        let mut request = TransferRequest::download(&url, &output);
        request.backend = DownloadBackend::CurlCli;
        match DownloadStack::default().recover(&request) {
            Ok(_) => {
                let body = fs::read_to_string(&output)?;
                let header = format!("Time,{gse_param}_0,{gse_param}_1,{gse_param}_2");
                fs::write(&output, format!("{header}\n{body}"))?;
                let bytes = fs::metadata(&output)?.len();
                println!("  DOY {doy}: {:.2} MB", bytes as f64 / 1_048_576.0);
            }
            Err(e) => eprintln!("  Warning: DOY {doy} failed: {e}"),
        }
    }

    // -----------------------------------------------------------------------
    // Step 2: Parse to minute-level records
    // -----------------------------------------------------------------------
    println!("[2/4] Parsing FGM to minute-level records...");

    let mut all_minutes: Vec<ThemisFgmMinuteRecord> = Vec::new();

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
            eprintln!("  Warning: {} not found", path.display());
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
        anyhow::bail!("No THEMIS FGM data found.");
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
        .context("failed to build reference midnight")?;

    println!(
        "  Total: {} minutes across {:.1} hours",
        all_minutes.len(),
        all_minutes.last().map(|r| r.elapsed_hours).unwrap_or(0.0)
    );

    // -----------------------------------------------------------------------
    // Step 3: Load Staples catalog
    // -----------------------------------------------------------------------
    println!("[3/4] Loading Staples+2020 catalog...");

    let catalog_content = fs::read_to_string(&cli.staples_catalog)
        .with_context(|| format!("reading catalog: {}", cli.staples_catalog.display()))?;

    let catalog =
        parse_staples_crossing_catalog(&catalog_content, probe_char, start, end, cli.pad_minutes);

    let fom_catalog: Vec<MmsEventInterval> = catalog
        .into_iter()
        .filter(|e| e.fom >= cli.min_fom)
        .collect();

    println!(
        "  Loaded {} catalog intervals (TH{}, {}-{})",
        fom_catalog.len(),
        probe_upper,
        start,
        end
    );

    // -----------------------------------------------------------------------
    // Step 4: L2-delay score series
    //
    // For each consecutive pair of full-window delay vectors v_t and v_{t-1},
    // the L2-delay score is ||v_t - v_{t-1}||_2.  This measures how much the
    // 32D embedding moved in one step -- the simplest change-point proxy.
    // We build delay vectors using the same noise-floor normalization as CD so
    // that the score is comparable in amplitude units.
    // -----------------------------------------------------------------------
    println!("[4/4] Computing L2-delay scores...");

    let channels: usize = 4;
    let steps = cli.embedding_dim / channels;
    let window_rows = (steps - 1) * cli.takens_lag + 1;

    if all_minutes.len() < window_rows + 1 {
        anyhow::bail!(
            "Not enough data: need {} rows, have {}",
            window_rows + 1,
            all_minutes.len()
        );
    }

    let expected_span_hours = (steps - 1) as f64 * cli.takens_lag as f64 / 60.0;
    let gap_tolerance_hours = 2.0 * cli.takens_lag as f64 / 60.0;
    let max_window_span_hours = expected_span_hours + gap_tolerance_hours;

    let mut delay_vectors: Vec<Vec<f64>> = Vec::new();
    let mut embed_meta: Vec<usize> = Vec::new();
    let mut n_gap_skipped: usize = 0;

    for w_start in 0..=(all_minutes.len() - window_rows) {
        let sample_indices: Vec<usize> = (0..steps).map(|s| w_start + s * cli.takens_lag).collect();

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
        let denom = local_mean_b.max(cli.bmag_noise_floor);
        let mut v = vec![0.0_f64; cli.embedding_dim];
        for (s, &ri) in sample_indices.iter().enumerate() {
            let rec = &all_minutes[ri];
            v[s * channels] = rec.bx_gse / denom;
            v[s * channels + 1] = rec.by_gse / denom;
            v[s * channels + 2] = rec.bz_gse / denom;
            v[s * channels + 3] = (rec.b_magnitude - local_mean_b) / denom;
        }
        embed_meta.push(*sample_indices.last().unwrap());
        delay_vectors.push(v);
    }

    // L2-delay score: ||v[k] - v[k-1]||_2 for each consecutive valid pair.
    // Metadata index: use meta[k] (the later window's anchor minute).
    let l2_scores: Vec<f64> = delay_vectors
        .windows(2)
        .map(|w| {
            w[0].iter()
                .zip(w[1].iter())
                .map(|(a, b)| (b - a).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .collect();

    let score_meta: Vec<usize> = embed_meta[1..].to_vec();

    println!(
        "  L2-delay scores: {} values ({} gap-skipped windows)",
        l2_scores.len(),
        n_gap_skipped
    );

    // -----------------------------------------------------------------------
    // Threshold + fire detection (same MAD_SCALE_FACTOR as CD binary)
    // -----------------------------------------------------------------------
    let trans_window = cli.crossing_window_minutes.max(5);
    let mut l2_hours: Vec<f64> = Vec::new();

    if l2_scores.len() > trans_window * 2 {
        let global_mean: f64 = l2_scores.iter().sum::<f64>() / l2_scores.len() as f64;
        let global_std: f64 = {
            let var = l2_scores
                .iter()
                .map(|&a| (a - global_mean).powi(2))
                .sum::<f64>()
                / l2_scores.len() as f64;
            var.sqrt()
        };
        let threshold = global_std * MAD_SCALE_FACTOR;
        let half = trans_window;
        let mut last_trans_idx: Option<usize> = None;
        for i in half..l2_scores.len().saturating_sub(half) {
            let pre_mean: f64 =
                l2_scores[i.saturating_sub(half)..i].iter().sum::<f64>() / half as f64;
            let post_mean: f64 = l2_scores[i..(i + half).min(l2_scores.len())]
                .iter()
                .sum::<f64>()
                / half.min(l2_scores.len() - i) as f64;
            let jump = (post_mean - pre_mean).abs();
            if jump > threshold {
                let dominated =
                    last_trans_idx.is_some_and(|prev| i.saturating_sub(prev) < trans_window);
                if !dominated {
                    let mi = score_meta[i];
                    l2_hours.push(all_minutes[mi].elapsed_hours);
                    last_trans_idx = Some(i);
                }
            }
        }
    }

    println!("  L2-delay detections: {}", l2_hours.len());

    // -----------------------------------------------------------------------
    // Evaluate
    // -----------------------------------------------------------------------
    let eval_window_secs = cli.pad_minutes * 60;

    let detection_unix: Vec<i64> = l2_hours
        .iter()
        .map(|&h| hours_to_unix(&reference_midnight, h))
        .collect();
    let event_unix: Vec<i64> = fom_catalog.iter().map(event_midpoint_unix).collect();

    let (precision, recall, f1) =
        boundary_metrics::precision_recall_f1(&detection_unix, &event_unix, eval_window_secs);

    let series_start_unix = hours_to_unix(&reference_midnight, 0.0);
    let series_end_unix = hours_to_unix(
        &reference_midnight,
        all_minutes.last().map(|r| r.elapsed_hours).unwrap_or(0.0),
    );

    let bootstrap_ci = boundary_metrics::bootstrap_f1_ci_seeded(
        &detection_unix,
        &event_unix,
        series_start_unix,
        series_end_unix,
        1800,
        10_000,
        0.95,
        eval_window_secs,
        42,
    );

    let (bci_mean, bci_lo, bci_hi) = bootstrap_ci;
    println!(
        "  F1={:.3}  P={:.3}  R={:.3}  CI=[{:.3},{:.3}]",
        f1, precision, recall, bci_lo, bci_hi
    );

    // -----------------------------------------------------------------------
    // Write output
    // -----------------------------------------------------------------------
    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }

    let results = L2DelayResults {
        start_date: cli.start_date.clone(),
        n_days: cli.n_days,
        probe: spacecraft.clone(),
        embedding_dim: cli.embedding_dim,
        method: "L2-delay change-point".to_string(),
        n_catalog_events: fom_catalog.len(),
        n_fgm_minutes: all_minutes.len(),
        n_detections: l2_hours.len(),
        precision,
        recall,
        f1,
        bootstrap_ci_mean: bci_mean,
        bootstrap_ci_lo: bci_lo,
        bootstrap_ci_hi: bci_hi,
        series_start_unix,
        series_end_unix,
    };

    let json = serde_json::to_string_pretty(&results)?;
    fs::write(&cli.out_json, &json)?;
    println!("\nResults written to {}", cli.out_json.display());

    Ok(())
}

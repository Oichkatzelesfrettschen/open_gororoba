//! General-interval THEMIS benchmark (plan P6A.S1 task 1.10).
//!
//! WHY: The 7-day window used in Section 3 (2016-08-29 to 2016-09-04) was
//! selected around a confirmed sudden-commencement (SC) event from the Staples
//! et al. (2020) catalog.  SC selection introduces survivor bias: the window
//! is guaranteed to contain sharp, large-amplitude transitions that the CD
//! associator detects well.  A general-interval benchmark measures performance
//! in unselected conditions, including quiet stretches with no SC events.
//! This addresses the Board critique (O-1) that Section 3.2 was missing.
//!
//! WHAT: Downloads THEMIS-A FGM for a user-specified window (default: 30 days
//! in September 2016, matching the 7-day window's year but extending before and
//! after the SC event).  Evaluates CD associator against ALL Staples+2020
//! crossings in that window (not just SC-window events), computing F1, CI, and
//! fire rate.  A second run on 2015-11-01..2015-11-30 (zero Staples entries)
//! serves as the negative control: reports only fire rate per hour.
//!
//! HOW:
//!   # General interval with some Staples entries
//!   cargo run --release -p gororoba_cli_physics \
//!     --bin heliosphere-themis-general-benchmark -- \
//!     --start-date 2016-08-01 --n-days 30 --probe a \
//!     --staples-catalog data/external/crossing_lists/themis_mp_crossings_v2.txt
//!
//!   # Negative control (0 Staples entries)
//!   cargo run --release -p gororoba_cli_physics \
//!     --bin heliosphere-themis-general-benchmark -- \
//!     --start-date 2015-11-01 --n-days 30 --probe a \
//!     --staples-catalog data/external/crossing_lists/themis_mp_crossings_v2.txt

use anyhow::{Context, Result};
use chrono::{Datelike, NaiveDate, NaiveDateTime, TimeZone, Utc};
use clap::Parser;
use data_core::{
    catalogs::{
        mms::{MmsEventInterval, detect_magnetopause_crossings_filtered},
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

/// MAD scale factor (same as staples_labeled binary). Fixed pre-evaluation.
const MAD_SCALE_FACTOR: f64 = 1.5;

// ============================================================================
// CLI
// ============================================================================

#[derive(Parser, Debug)]
#[command(
    name = "heliosphere-themis-general-benchmark",
    about = "General-interval THEMIS CD benchmark (non-SC-selected; plan P6A.S1 task 1.10)"
)]
struct Cli {
    #[arg(long, default_value = "2016-08-01")]
    start_date: String,

    #[arg(long, default_value_t = 30)]
    n_days: u32,

    #[arg(long, default_value = "a")]
    probe: String,

    #[arg(
        long,
        default_value = "data/external/crossing_lists/themis_mp_crossings_v2.txt"
    )]
    staples_catalog: PathBuf,

    /// Padding around each catalog crossing (minutes).
    #[arg(long, default_value_t = 10)]
    pad_minutes: i64,

    #[arg(long, default_value_t = 32)]
    embedding_dim: usize,

    #[arg(long, default_value_t = 1)]
    takens_lag: usize,

    #[arg(long, default_value_t = 10)]
    crossing_window_minutes: usize,

    #[arg(long, default_value_t = 5.0)]
    bmag_gradient_threshold: f64,

    #[arg(long, default_value_t = 30.0)]
    rotation_threshold_deg: f64,

    #[arg(long, default_value_t = 100.0)]
    max_bmag: f64,

    #[arg(long, default_value_t = 0.5)]
    bmag_noise_floor: f64,

    #[arg(long, default_value = "data/external")]
    data_dir: PathBuf,

    #[arg(
        long,
        default_value =
            "data/output/heliosphere/ablations/themis_general_benchmark.json"
    )]
    out_json: PathBuf,
}

// ============================================================================
// Output
// ============================================================================

#[derive(Debug, Serialize)]
struct GeneralBenchmarkResults {
    start_date: String,
    n_days: u32,
    probe: String,
    embedding_dim: usize,
    n_catalog_events: usize,
    n_fgm_minutes: usize,
    cd_fires_total: usize,
    cd_fire_rate_per_hour: f64,
    cd_precision: f64,
    cd_recall: f64,
    cd_f1: f64,
    cd_bootstrap_ci: [f64; 3],
    f1_null_p_value: f64,
    gradient_precision: f64,
    gradient_recall: f64,
    gradient_f1: f64,
    gradient_fires_total: usize,
    ascii_summary: String,
}

// ============================================================================
// Helpers (shared with staples_labeled)
// ============================================================================

fn hours_to_unix(origin: &NaiveDateTime, h: f64) -> i64 {
    Utc.from_utc_datetime(origin).timestamp() + (h * 3600.0) as i64
}

fn event_midpoint_unix(ev: &MmsEventInterval) -> i64 {
    let mid = ev.start + (ev.end - ev.start) / 2;
    Utc.from_utc_datetime(&mid).timestamp()
}

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

// ============================================================================
// main
// ============================================================================

fn main() -> Result<()> {
    let cli = Cli::parse();

    let start = NaiveDate::parse_from_str(&cli.start_date, "%Y-%m-%d")
        .with_context(|| format!("invalid start_date: {}", cli.start_date))?;
    let end = start + chrono::Duration::days(cli.n_days as i64 - 1);
    let probe_upper = cli.probe.trim().to_uppercase();
    let probe_char = cli
        .probe
        .trim()
        .to_lowercase()
        .chars()
        .next()
        .context("--probe must be a single letter")?;
    let spacecraft = format!("TH{probe_upper}");

    println!(
        "=== THEMIS General Benchmark ===\n\
         Window: {} to {} ({} days)\n\
         Probe: {}  Embedding: {}D\n",
        start, end, cli.n_days, spacecraft, cli.embedding_dim
    );

    // -----------------------------------------------------------------------
    // Step 1: Fetch FGM data
    // -----------------------------------------------------------------------
    println!("[1/4] Fetching {spacecraft} FGM data...");
    let themis_dir = cli.data_dir.join("themis");
    fs::create_dir_all(&themis_dir)?;

    let gse_param = format!("{}_fgs_gse", spacecraft.to_lowercase());
    let dataset = format!("{}_L2_FGM@0", spacecraft);

    for day_offset in 0..cli.n_days {
        let date = start + chrono::Duration::days(day_offset as i64);
        let doy = date.ordinal();
        let fname = format!(
            "{}_fgm_{:04}_{:03}.csv",
            probe_upper.to_lowercase(),
            date.year(),
            doy
        );
        let output = themis_dir.join(&fname);
        if output.exists() {
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
            }
            Err(e) => eprintln!("  Warning: DOY {doy} failed: {e}"),
        }
    }

    // -----------------------------------------------------------------------
    // Step 2: Parse and sort FGM records
    // -----------------------------------------------------------------------
    println!("[2/4] Parsing FGM data...");
    let mut all_minutes: Vec<ThemisFgmMinuteRecord> = Vec::new();
    for day_offset in 0..cli.n_days {
        let date = start + chrono::Duration::days(day_offset as i64);
        let fname = format!(
            "{}_fgm_{:04}_{:03}.csv",
            probe_upper.to_lowercase(),
            date.year(),
            date.ordinal()
        );
        let path = themis_dir.join(&fname);
        if !path.exists() {
            continue;
        }
        let content = fs::read_to_string(&path)?;
        let mut recs = parse_themis_fgm_hapi_csv_minutes(&content, &spacecraft);
        all_minutes.append(&mut recs);
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

    println!("  Total: {} minutes", all_minutes.len());

    // -----------------------------------------------------------------------
    // Step 3: Load Staples catalog (all crossings in window, no SC filter)
    // -----------------------------------------------------------------------
    println!("[3/4] Loading Staples+2020 catalog...");
    let catalog_content = fs::read_to_string(&cli.staples_catalog)
        .with_context(|| format!("reading {}", cli.staples_catalog.display()))?;
    let catalog = parse_staples_crossing_catalog(
        &catalog_content,
        probe_char,
        start,
        end,
        cli.pad_minutes,
    );
    println!(
        "  {} catalog events for TH{} in {} to {}",
        catalog.len(),
        probe_upper,
        start,
        end
    );

    // -----------------------------------------------------------------------
    // Step 4: Run detectors
    // -----------------------------------------------------------------------
    println!("[4/4] Running detectors...");

    // Gradient baseline.
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

    // CD associator.
    let channels: usize = 4;
    let steps = cli.embedding_dim / channels;
    let window_rows = (steps - 1) * cli.takens_lag + 1;

    let expected_span_hours = (steps - 1) as f64 * cli.takens_lag as f64 / 60.0;
    let gap_tolerance_hours = 2.0 * cli.takens_lag as f64 / 60.0;
    let max_window_span_hours = expected_span_hours + gap_tolerance_hours;

    let mut embedded_vectors: Vec<Vec<f64>> = Vec::new();
    let mut embed_meta: Vec<usize> = Vec::new();
    let mut n_gap_skipped: usize = 0;

    for w_start in 0..=(all_minutes.len().saturating_sub(window_rows)) {
        let sample_indices: Vec<usize> =
            (0..steps).map(|s| w_start + s * cli.takens_lag).collect();
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
        let mut v = vec![0.0; cli.embedding_dim];
        for (s, &ri) in sample_indices.iter().enumerate() {
            let rec = &all_minutes[ri];
            v[s * channels] = rec.bx_gse / denom;
            v[s * channels + 1] = rec.by_gse / denom;
            v[s * channels + 2] = rec.bz_gse / denom;
            v[s * channels + 3] = (rec.b_magnitude - local_mean_b) / denom;
        }
        embedded_vectors.push(v);
        embed_meta.push(*sample_indices.last().unwrap());
    }

    let associators =
        cd_kernel::batch_sliding_associator_norms_parallel(&embedded_vectors, cli.embedding_dim);
    let assoc_minute_indices: Vec<usize> =
        (0..associators.len()).map(|k| embed_meta[k + 2]).collect();

    let trans_window = cli.crossing_window_minutes.max(5);
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
        let mut last_trans_idx: Option<usize> = None;
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
                    last_trans_idx.is_some_and(|prev| i.saturating_sub(prev) < trans_window);
                if !dominated {
                    cd_hours.push(all_minutes[assoc_minute_indices[i]].elapsed_hours);
                    last_trans_idx = Some(i);
                }
            }
        }
    }

    let total_hours = all_minutes
        .last()
        .map(|r| r.elapsed_hours)
        .unwrap_or(0.0);
    let cd_fire_rate_per_hour = if total_hours > 0.0 {
        cd_hours.len() as f64 / total_hours
    } else {
        0.0
    };

    println!(
        "  CD: {} fires in {:.1} h = {:.3} fires/h ({} gap-skipped)",
        cd_hours.len(),
        total_hours,
        cd_fire_rate_per_hour,
        n_gap_skipped
    );
    println!("  Gradient: {} fires", gradient_hours.len());

    // -----------------------------------------------------------------------
    // Evaluate against catalog
    // -----------------------------------------------------------------------
    let eval_window_secs = cli.pad_minutes * 60;

    let cd_unix: Vec<i64> = cd_hours
        .iter()
        .map(|&h| hours_to_unix(&reference_midnight, h))
        .collect();
    let gr_unix: Vec<i64> = gradient_hours
        .iter()
        .map(|&h| hours_to_unix(&reference_midnight, h))
        .collect();
    let ev_unix: Vec<i64> = catalog.iter().map(event_midpoint_unix).collect();

    let (cd_p, cd_r, cd_f1) =
        boundary_metrics::precision_recall_f1(&cd_unix, &ev_unix, eval_window_secs);
    let (gr_p, gr_r, gr_f1) =
        boundary_metrics::precision_recall_f1(&gr_unix, &ev_unix, eval_window_secs);

    println!(
        "  CD F1 = {:.3} (P={:.3} R={:.3})  |  Gradient F1 = {:.3}",
        cd_f1, cd_p, cd_r, gr_f1
    );

    let series_start_unix = Utc.from_utc_datetime(&reference_midnight).timestamp();
    let series_end_unix =
        series_start_unix + all_minutes.last().map(|r| (r.elapsed_hours * 3600.0) as i64).unwrap_or(0);

    let (bs_mean, bs_lo, bs_hi) = boundary_metrics::bootstrap_f1_ci_seeded(
        &cd_unix,
        &ev_unix,
        series_start_unix,
        series_end_unix,
        1800,
        10_000,
        0.95,
        eval_window_secs,
        42,
    );

    let f1_null_p_value = if ev_unix.is_empty() {
        1.0
    } else {
        boundary_metrics::f1_null_distribution_seeded(
            &cd_unix,
            ev_unix.len(),
            series_start_unix,
            series_end_unix,
            10_000,
            cd_f1,
            eval_window_secs,
            42,
        )
    };

    println!(
        "  CD F1 CI (95%): {:.3} [{:.3}, {:.3}]  null p = {:.4}",
        bs_mean, bs_lo, bs_hi, f1_null_p_value
    );

    let ascii_summary = format!(
        "THEMIS General Benchmark: {} to {} ({} days, {} catalog events)\n\
         CD:       F1={:.3}  P={:.3}  R={:.3}  fires={} ({:.3}/h)\n\
         Gradient: F1={:.3}  P={:.3}  R={:.3}  fires={}\n\
         CD F1 CI (95%): [{:.3}, {:.3}]  null p = {:.4}",
        cli.start_date,
        end,
        cli.n_days,
        catalog.len(),
        cd_f1,
        cd_p,
        cd_r,
        cd_hours.len(),
        cd_fire_rate_per_hour,
        gr_f1,
        gr_p,
        gr_r,
        gradient_hours.len(),
        bs_lo,
        bs_hi,
        f1_null_p_value,
    );
    println!("\n{}", ascii_summary);

    let results = GeneralBenchmarkResults {
        start_date: cli.start_date.clone(),
        n_days: cli.n_days,
        probe: cli.probe.clone(),
        embedding_dim: cli.embedding_dim,
        n_catalog_events: catalog.len(),
        n_fgm_minutes: all_minutes.len(),
        cd_fires_total: cd_hours.len(),
        cd_fire_rate_per_hour,
        cd_precision: cd_p,
        cd_recall: cd_r,
        cd_f1,
        cd_bootstrap_ci: [bs_mean, bs_lo, bs_hi],
        f1_null_p_value,
        gradient_precision: gr_p,
        gradient_recall: gr_r,
        gradient_f1: gr_f1,
        gradient_fires_total: gradient_hours.len(),
        ascii_summary,
    };

    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.out_json, serde_json::to_string_pretty(&results)?)?;
    println!("Wrote {}", cli.out_json.display());

    Ok(())
}

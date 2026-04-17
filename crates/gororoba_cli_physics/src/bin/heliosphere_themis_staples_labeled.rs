//! THEMIS CD associator P/R/F1 evaluated against the Staples et al. (2020)
//! peer-reviewed magnetopause crossing catalog.
//!
//! WHY: A second independent hardened-label evaluation strengthens the methods
//! paper.  Unlike the MMS FPI evaluation (E-236), the ground truth here is an
//! externally published, peer-reviewed catalog (JGR 2020,
//! doi:10.1029/2019JA027190) that was assembled from plasma-moment and
//! magnetic-field criteria by independent researchers -- not derived from the
//! same spacecraft whose FGM data we are evaluating.  This rules out any
//! circular-instrument bias in the comparison.
//!
//! WHAT: Downloads THEMIS-A L2 FGM spin-resolution data for a chosen 7-day
//! window from CDAWeb HAPI, minute-averages it, runs the same 32D Takens
//! embedding / CD associator and |B|-gradient+rotation detector pipeline used
//! in heliosphere-mms-sitl-labeled, and evaluates both against the Staples
//! catalog intervals (+/-pad_minutes around each crossing timestamp).
//!
//! HOW:
//!   cargo run --release -p gororoba_cli_physics \
//!     --bin heliosphere-themis-staples-labeled -- \
//!     --start-date 2016-08-29 --n-days 7 \
//!     --staples-catalog data/external/crossing_lists/themis_mp_crossings_v2.txt

use anyhow::{Context, Result};
use chrono::{Datelike, NaiveDate, NaiveDateTime};
use clap::Parser;
use data_core::{
    catalogs::{
        mms::{MmsEventInterval, detect_magnetopause_crossings_filtered, timestamp_in_sitl_event},
        themis::{
            ThemisFgmMinuteRecord, parse_staples_crossing_catalog,
            parse_themis_fgm_hapi_csv_minutes,
        },
    },
    download_stack::{DownloadBackend, DownloadStack, TransferRequest},
};
use serde::Serialize;
use std::{fs, path::PathBuf};

// ============================================================================
// CLI
// ============================================================================

#[derive(Parser, Debug)]
#[command(
    name = "heliosphere-themis-staples-labeled",
    about = "THEMIS CD associator P/R/F1 against Staples et al. (2020) ground truth"
)]
struct Cli {
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
    /// Converts a point timestamp to an evaluation interval.
    #[arg(long, default_value_t = 10)]
    pad_minutes: i64,

    /// Only include catalog crossings with FOM at or above this value.
    /// All Staples events are FOM=100 (peer reviewed); this arg is kept for
    /// API consistency with the MMS binary.
    #[arg(long, default_value_t = 0.0)]
    min_fom: f64,

    /// Embedding dimension for Takens delay embedding.
    #[arg(long, default_value_t = 32)]
    embedding_dim: usize,

    /// Takens lag in minutes.
    #[arg(long, default_value_t = 1)]
    takens_lag: usize,

    /// Window size (minutes) for gradient crossing detector.
    #[arg(long, default_value_t = 10)]
    crossing_window_minutes: usize,

    /// |B| gradient threshold (nT) for gradient detector.
    #[arg(long, default_value_t = 5.0)]
    bmag_gradient_threshold: f64,

    /// Rotation threshold (degrees) for gradient detector.
    #[arg(long, default_value_t = 30.0)]
    rotation_threshold_deg: f64,

    /// Maximum |B| (nT).  Filters deep magnetosphere transits (|B| > 100 nT).
    #[arg(long, default_value_t = 100.0)]
    max_bmag: f64,

    /// Data cache root directory.
    #[arg(long, default_value = "data/external")]
    data_dir: PathBuf,

    /// Output JSON path.
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/themis_staples_labeled_eval.json"
    )]
    out_json: PathBuf,
}

// ============================================================================
// Output structures
// ============================================================================

#[derive(Debug, Serialize)]
struct CatalogEventSummary {
    timestamp: String,
    interval_start: String,
    interval_end: String,
    gradient_matched: bool,
    cd_matched: bool,
}

#[derive(Debug, Serialize)]
struct DetectorEval {
    method: String,
    n_detections: usize,
    n_catalog_intervals: usize,
    tp_detections: usize,
    tp_catalog: usize,
    precision: f64,
    recall: f64,
    f1: f64,
}

#[derive(Debug, Serialize)]
struct StaplesLabeledResults {
    start_date: String,
    n_days: u32,
    probe: String,
    embedding_dim: usize,
    n_catalog_events: usize,
    n_fgm_minutes: usize,
    gradient_eval: DetectorEval,
    cd_eval: DetectorEval,
    delta_precision: f64,
    delta_recall: f64,
    delta_f1: f64,
    catalog_events: Vec<CatalogEventSummary>,
    ascii_table: String,
}

// ============================================================================
// Helpers
// ============================================================================

/// Adapt ThemisFgmMinuteRecord to the 4-field layout expected by
/// detect_magnetopause_crossings_filtered (which takes &[MmsFgmMinuteRecord]).
/// We replicate the logic directly here to avoid a cross-crate adapter.
fn detect_crossings_from_themis(
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

fn eval_against_catalog(
    detection_hours: &[f64],
    reference_midnight: &NaiveDateTime,
    catalog: &[MmsEventInterval],
    method_label: &str,
) -> DetectorEval {
    let n_catalog = catalog.len();
    let tp_detections = detection_hours
        .iter()
        .filter(|&&h| timestamp_in_sitl_event(h, reference_midnight, catalog))
        .count();

    let tp_catalog = catalog
        .iter()
        .filter(|ev| {
            detection_hours.iter().any(|&h| {
                use chrono::Duration;
                let t = *reference_midnight + Duration::seconds((h * 3600.0) as i64);
                ev.start <= t && t < ev.end
            })
        })
        .count();

    let precision = if detection_hours.is_empty() {
        0.0
    } else {
        tp_detections as f64 / detection_hours.len() as f64
    };
    let recall = if n_catalog == 0 {
        0.0
    } else {
        tp_catalog as f64 / n_catalog as f64
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    DetectorEval {
        method: method_label.to_string(),
        n_detections: detection_hours.len(),
        n_catalog_intervals: n_catalog,
        tp_detections,
        tp_catalog,
        precision,
        recall,
        f1,
    }
}

fn build_ascii_table(grad: &DetectorEval, cd: &DetectorEval) -> String {
    let mut out = String::new();
    out.push_str(
        "THEMIS CD associator vs Staples+2020 peer-reviewed ground truth\n\
         ================================================================\n",
    );
    out.push_str(&format!(
        "{:<28} {:>8} {:>8} {:>8} {:>8} {:>8}\n",
        "Method", "Detects", "TP_det", "TP_cat", "Prec", "Rec"
    ));
    out.push_str(&"-".repeat(72));
    out.push('\n');
    for ev in [grad, cd] {
        out.push_str(&format!(
            "{:<28} {:>8} {:>8} {:>8} {:>8.3} {:>8.3}  F1={:.3}\n",
            ev.method,
            ev.n_detections,
            ev.tp_detections,
            ev.tp_catalog,
            ev.precision,
            ev.recall,
            ev.f1,
        ));
    }
    out.push_str(&"-".repeat(72));
    out.push('\n');
    out.push_str(&format!(
        "Delta (CD - gradient):                                {:>+8.3} {:>+8.3}  dF1={:+.3}\n",
        cd.precision - grad.precision,
        cd.recall - grad.recall,
        cd.f1 - grad.f1,
    ));
    out
}

// ============================================================================
// main
// ============================================================================

fn main() -> Result<()> {
    let cli = Cli::parse();

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
    // CDAWeb HAPI uses full spacecraft IDs: THA, THB, THC, THD, THE.
    // The probe letter "a" maps to spacecraft "THA" for dataset IDs and
    // parameter names (tha_fgs_gse).  probe_upper holds the single letter.
    let spacecraft = format!("TH{probe_upper}");

    println!(
        "=== THEMIS Staples-Labeled Evaluation ===\n\
         Window: {} to {} ({} days)\n\
         Probe: {}  Embedding: {}D, lag={}min\n",
        start, end, cli.n_days, spacecraft, cli.embedding_dim, cli.takens_lag
    );

    // -----------------------------------------------------------------------
    // Step 1: Fetch THEMIS FGM data
    //
    // WHY curl bypass: workspace uses reqwest with `rustls-no-provider`; calling
    // any DownloadStack path with backend=Auto triggers detect_capabilities()
    // which tries to build a reqwest Client and panics ("No provider set").
    // CurlCli backend bypasses that code path entirely.
    //
    // Header synthesis: HAPI CSV is headerless by default.  parse_themis_fgm_hapi_csv_minutes
    // needs "tha_fgs_gse_0/1/2" column names.  We prepend the known header for
    // THA_L2_FGM@0 parameters=[Time, tha_fgs_gse] instead of fetching the info
    // endpoint (which would also trigger reqwest).
    // -----------------------------------------------------------------------
    println!("[1/5] Fetching THEMIS-{} FGM data...", probe_upper);

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
                // Prepend known CSV header: Time, Bx, By, Bz
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
    println!("[2/5] Parsing FGM to minute-level records...");

    let themis_dir = cli.data_dir.join("themis");
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

    // Filter high-|B| inner-magnetosphere transits and sort.
    all_minutes.retain(|r| r.b_magnitude <= cli.max_bmag);
    all_minutes.sort_by(|a, b| {
        a.year
            .cmp(&b.year)
            .then(a.doy.cmp(&b.doy))
            .then(a.hour.cmp(&b.hour))
            .then(a.minute.cmp(&b.minute))
    });

    // Recompute elapsed hours from first record.
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
        "  Total: {} minutes across {:.1} hours (after max_bmag filter)",
        all_minutes.len(),
        all_minutes.last().map(|r| r.elapsed_hours).unwrap_or(0.0)
    );

    // -----------------------------------------------------------------------
    // Step 3: Load Staples catalog
    // -----------------------------------------------------------------------
    println!(
        "[3/5] Loading Staples+2020 catalog ({})...",
        cli.staples_catalog.display()
    );

    let catalog_content = fs::read_to_string(&cli.staples_catalog)
        .with_context(|| format!("reading catalog: {}", cli.staples_catalog.display()))?;

    let catalog =
        parse_staples_crossing_catalog(&catalog_content, probe_char, start, end, cli.pad_minutes);

    let fom_catalog: Vec<MmsEventInterval> = catalog
        .into_iter()
        .filter(|e| e.fom >= cli.min_fom)
        .collect();

    println!(
        "  Loaded {} catalog intervals (probe TH{}, window {} to {}, +/-{}min pad)",
        fom_catalog.len(),
        probe_upper,
        start,
        end,
        cli.pad_minutes
    );

    if fom_catalog.is_empty() {
        eprintln!(
            "  WARNING: no catalog events for TH{} in {} to {}.",
            probe_upper, start, end
        );
    }

    // -----------------------------------------------------------------------
    // Step 4: Gradient detector baseline
    // -----------------------------------------------------------------------
    println!("[4/5] Running gradient detector baseline...");

    let gradient_indices = detect_crossings_from_themis(
        &all_minutes,
        cli.crossing_window_minutes,
        cli.bmag_gradient_threshold,
        Some(cli.rotation_threshold_deg),
    );
    let gradient_hours: Vec<f64> = gradient_indices
        .iter()
        .map(|&i| all_minutes[i].elapsed_hours)
        .collect();

    println!("  Gradient detector: {} crossings", gradient_hours.len());

    // -----------------------------------------------------------------------
    // Step 5: CD associator transitions
    // -----------------------------------------------------------------------
    println!(
        "[5/5] Running {}D Takens embedding + CD associator...",
        cli.embedding_dim
    );

    let channels: usize = 4;
    let steps = cli.embedding_dim / channels;
    let window_rows = (steps - 1) * cli.takens_lag + 1;

    if all_minutes.len() < window_rows + 2 {
        anyhow::bail!(
            "Not enough data for embedding: need {} rows, have {}",
            window_rows + 2,
            all_minutes.len()
        );
    }

    let mut embedded_vectors: Vec<Vec<f64>> = Vec::new();
    let mut embed_meta: Vec<usize> = Vec::new();

    for w_start in 0..=(all_minutes.len() - window_rows) {
        let sample_indices: Vec<usize> = (0..steps).map(|s| w_start + s * cli.takens_lag).collect();
        let sum_b: f64 = sample_indices
            .iter()
            .map(|&i| all_minutes[i].b_magnitude)
            .sum();
        let local_mean_b = sum_b / steps as f64;
        if local_mean_b <= 0.0 || !local_mean_b.is_finite() {
            continue;
        }
        let mut v = vec![0.0; cli.embedding_dim];
        for (s, &ri) in sample_indices.iter().enumerate() {
            let rec = &all_minutes[ri];
            v[s * channels] = rec.bx_gse / local_mean_b;
            v[s * channels + 1] = rec.by_gse / local_mean_b;
            v[s * channels + 2] = rec.bz_gse / local_mean_b;
            v[s * channels + 3] = (rec.b_magnitude - local_mean_b) / local_mean_b;
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
        let threshold = global_std * 1.5;
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
                    let mi = assoc_minute_indices[i];
                    cd_hours.push(all_minutes[mi].elapsed_hours);
                    last_trans_idx = Some(i);
                }
            }
        }
    }

    println!("  CD associator: {} transitions", cd_hours.len());

    // -----------------------------------------------------------------------
    // Evaluate both methods
    // -----------------------------------------------------------------------
    let grad_eval = eval_against_catalog(
        &gradient_hours,
        &reference_midnight,
        &fom_catalog,
        "|B|-gradient+rotation detector",
    );
    let cd_eval = eval_against_catalog(
        &cd_hours,
        &reference_midnight,
        &fom_catalog,
        "CD associator (32D)",
    );

    let catalog_events: Vec<CatalogEventSummary> = fom_catalog
        .iter()
        .map(|ev| {
            let gradient_matched = gradient_hours.iter().any(|&h| {
                use chrono::Duration;
                let t = reference_midnight + Duration::seconds((h * 3600.0) as i64);
                ev.start <= t && t < ev.end
            });
            let cd_matched = cd_hours.iter().any(|&h| {
                use chrono::Duration;
                let t = reference_midnight + Duration::seconds((h * 3600.0) as i64);
                ev.start <= t && t < ev.end
            });
            // Recover approximate original timestamp (midpoint of interval)
            let mid = ev.start + (ev.end - ev.start) / 2;
            CatalogEventSummary {
                timestamp: mid.format("%Y-%m-%dT%H:%M:%S").to_string(),
                interval_start: ev.start.format("%Y-%m-%dT%H:%M:%S").to_string(),
                interval_end: ev.end.format("%Y-%m-%dT%H:%M:%S").to_string(),
                gradient_matched,
                cd_matched,
            }
        })
        .collect();

    let ascii = build_ascii_table(&grad_eval, &cd_eval);
    println!("\n{}", ascii);

    let delta_p = cd_eval.precision - grad_eval.precision;
    let delta_r = cd_eval.recall - grad_eval.recall;
    let delta_f1 = cd_eval.f1 - grad_eval.f1;

    let results = StaplesLabeledResults {
        start_date: cli.start_date.clone(),
        n_days: cli.n_days,
        probe: cli.probe.clone(),
        embedding_dim: cli.embedding_dim,
        n_catalog_events: fom_catalog.len(),
        n_fgm_minutes: all_minutes.len(),
        gradient_eval: grad_eval,
        cd_eval,
        delta_precision: delta_p,
        delta_recall: delta_r,
        delta_f1,
        catalog_events,
        ascii_table: ascii,
    };

    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.out_json, serde_json::to_string_pretty(&results)?)?;
    println!("Wrote {}", cli.out_json.display());

    Ok(())
}

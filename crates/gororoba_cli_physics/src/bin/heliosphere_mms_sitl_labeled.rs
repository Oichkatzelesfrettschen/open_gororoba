//! MMS magnetopause CD associator evaluation against SITL-curated ground truth.
//!
//! WHY: The heliosphere-mms-multiday binary evaluates the CD associator by
//! comparing it to a |B|-gradient pseudo-label detector, not to expert-curated
//! ground truth.  The FAR (false alarm rate) paradox -- compressive events such
//! as pressure pulses and dipolarizations produce |B| jumps without rotating
//! the field -- inflates both the gradient detector's FP count and the CD
//! detector's apparent FA rate.  Using SITL (Scientist-In-The-Loop) burst-mode
//! selections as ground truth strips out compressive false positives and gives
//! a precision/recall that reflects the algorithm's ability to isolate genuine
//! topological boundaries.
//!
//! WHAT: Downloads the MMS SDC GLS/SITL CSV catalog for the analysis window,
//! filters events to magnetopause keywords, then evaluates:
//!   (A) |B|-gradient + rotation detector (old pseudo-label method)
//!   (B) 32D Cayley-Dickson associator transitions (new method)
//! against the SITL catalog, producing per-method P/R/F1 and a comparison table.
//!
//! Evaluation semantics (interval-based):
//!   Precision = transitions that fall inside any SITL interval / total transitions
//!   Recall    = SITL intervals containing at least one transition / total SITL intervals
//!   F1        = 2*P*R / (P+R)
//!
//! HOW:
//!   cargo run --release -p gororoba_cli_physics --bin heliosphere-mms-sitl-labeled -- \
//!     --start-date 2024-01-01 --n-days 7
//!   # Offline / cached catalog:
//!   cargo run --release -p gororoba_cli_physics --bin heliosphere-mms-sitl-labeled -- \
//!     --sitl-csv data/external/mms/mms_sitl_20240101_20240107.csv

use anyhow::{Context, Result};
use chrono::{Datelike, NaiveDate, NaiveDateTime};
use clap::Parser;
use data_core::{
    catalogs::{
        mms::{
            MmsEventInterval, MmsFgmMinuteRecord, detect_magnetopause_crossings_filtered,
            filter_magnetopause_events, parse_mms_fgm_hapi_csv_minutes, parse_mms_sitl_csv,
            timestamp_in_sitl_event,
        },
        mms_fetch::MmsSitlProvider,
    },
    fetcher::{DatasetProvider, FetchConfig, download_hapi_csv_to_file},
};
use serde::Serialize;
use std::{fs, path::PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "heliosphere-mms-sitl-labeled",
    about = "MMS CD associator P/R/F1 against SITL-curated magnetopause ground truth"
)]
struct Cli {
    /// Start date (YYYY-MM-DD).
    #[arg(long, default_value = "2024-01-01")]
    start_date: String,

    /// Number of days to analyze.
    #[arg(long, default_value_t = 7)]
    n_days: u32,

    /// Path to a locally cached SITL/GLS CSV file.  If omitted, the catalog
    /// is downloaded from the MMS SDC public API and cached automatically.
    #[arg(long)]
    sitl_csv: Option<PathBuf>,

    /// Only include SITL events with FOM at or above this threshold.
    /// Range 0-100; default 10 keeps most scientist selections.
    #[arg(long, default_value_t = 10.0)]
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

    /// |B| gradient threshold (nT) for the gradient detector baseline.
    #[arg(long, default_value_t = 5.0)]
    bmag_gradient_threshold: f64,

    /// Rotation threshold (degrees) for the gradient detector baseline.
    /// Recommended: 30.0.  Omit to use gradient-only criterion.
    #[arg(long, default_value_t = 30.0)]
    rotation_threshold_deg: f64,

    /// Maximum |B| (nT) to include.  Filters inner magnetosphere transits.
    #[arg(long, default_value_t = 100.0)]
    max_bmag: f64,

    /// Output JSON path.
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/mms_sitl_labeled_eval.json"
    )]
    out_json: PathBuf,

    /// Data cache directory.
    #[arg(long, default_value = "data/external")]
    data_dir: PathBuf,
}

// ---------------------------------------------------------------------------
// Output structures
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct SitlEventSummary {
    start: String,
    end: String,
    duration_minutes: f64,
    fom: f64,
    discussion: String,
    gradient_matched: bool,
    cd_matched: bool,
}

#[derive(Debug, Serialize)]
struct DetectorEval {
    method: String,
    n_detections: usize,
    n_sitl_intervals: usize,
    tp_detections: usize, // detections inside a SITL interval
    tp_sitl: usize,       // SITL intervals with at least one detection
    precision: f64,
    recall: f64,
    f1: f64,
}

#[derive(Debug, Serialize)]
struct SitlLabeledResults {
    start_date: String,
    n_days: u32,
    embedding_dim: usize,
    n_sitl_events_raw: usize,
    n_sitl_events_filtered: usize,
    n_fgm_minutes: usize,
    gradient_eval: DetectorEval,
    cd_eval: DetectorEval,
    delta_precision: f64,
    delta_recall: f64,
    delta_f1: f64,
    sitl_events: Vec<SitlEventSummary>,
    ascii_table: String,
}

// ---------------------------------------------------------------------------
// Evaluation helpers
// ---------------------------------------------------------------------------

/// Evaluate a set of detected event times (elapsed hours from reference) against
/// SITL event intervals (NaiveDateTime-based).
///
/// Returns `(tp_detections, tp_sitl_intervals, precision, recall, f1)`.
fn eval_against_sitl(
    detection_hours: &[f64],
    reference_midnight: &NaiveDateTime,
    sitl: &[MmsEventInterval],
    method_label: &str,
    n_sitl: usize,
) -> DetectorEval {
    let tp_detections = detection_hours
        .iter()
        .filter(|&&h| timestamp_in_sitl_event(h, reference_midnight, sitl))
        .count();

    let tp_sitl = sitl
        .iter()
        .filter(|ev| {
            // At least one detection falls within this interval.
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

    let recall = if n_sitl == 0 {
        0.0
    } else {
        tp_sitl as f64 / n_sitl as f64
    };

    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    DetectorEval {
        method: method_label.to_string(),
        n_detections: detection_hours.len(),
        n_sitl_intervals: n_sitl,
        tp_detections,
        tp_sitl,
        precision,
        recall,
        f1,
    }
}

fn build_ascii_table(grad: &DetectorEval, cd: &DetectorEval) -> String {
    let mut out = String::new();
    out.push_str("MMS CD associator vs SITL ground truth (hardened label evaluation)\n");
    out.push_str("===================================================================\n");
    out.push_str(&format!(
        "{:<28} {:>8} {:>8} {:>8} {:>8} {:>8}\n",
        "Method", "Detects", "TP_det", "TP_SITL", "Prec", "Rec"
    ));
    out.push_str(&"-".repeat(68));
    out.push('\n');
    for ev in [grad, cd] {
        out.push_str(&format!(
            "{:<28} {:>8} {:>8} {:>8} {:>8.3} {:>8.3}  F1={:.3}\n",
            ev.method,
            ev.n_detections,
            ev.tp_detections,
            ev.tp_sitl,
            ev.precision,
            ev.recall,
            ev.f1,
        ));
    }
    out.push_str(&"-".repeat(68));
    out.push('\n');
    out.push_str(&format!(
        "Delta (CD - gradient):                              {:>+8.3} {:>+8.3}  dF1={:+.3}\n",
        cd.precision - grad.precision,
        cd.recall - grad.recall,
        cd.f1 - grad.f1,
    ));
    out
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let cli = Cli::parse();

    let start = NaiveDate::parse_from_str(&cli.start_date, "%Y-%m-%d")
        .with_context(|| format!("invalid start_date: {}", cli.start_date))?;
    let end = start + chrono::Duration::days(cli.n_days as i64 - 1);

    println!(
        "=== MMS SITL-Labeled Evaluation ===\n\
         Window: {} to {} ({} days)\n\
         Embedding: {}D, lag={}min\n",
        start, end, cli.n_days, cli.embedding_dim, cli.takens_lag,
    );

    // -----------------------------------------------------------------------
    // Step 1: Fetch / read FGM data
    // -----------------------------------------------------------------------
    println!("[1/5] Fetching MMS1 FGM data...");

    let fgm_dir = cli.data_dir.join("mms");
    fs::create_dir_all(&fgm_dir)?;

    let mms_dataset = "MMS1_FGM_SRVY_L2@0";
    let mms_params: &[&str] = &["Time", "mms1_fgm_b_gse_srvy_l2_clean"];

    for day_offset in 0..cli.n_days {
        let date = start + chrono::Duration::days(day_offset as i64);
        let doy = date.ordinal() as u16;
        let y = date.year() as u16;
        let fname = format!("mms1_fgm_srvy_l2_{y}_{doy}_{doy}.csv");
        let output = fgm_dir.join(&fname);
        if output.exists() {
            println!("  DOY {doy}: cached");
            continue;
        }
        let t_min = format!("{}T00:00:00Z", date.format("%Y-%m-%d"));
        let t_max = format!("{}T23:59:59Z", date.format("%Y-%m-%d"));
        println!("  DOY {doy}: downloading...");
        match download_hapi_csv_to_file(mms_dataset, &t_min, &t_max, Some(mms_params), &output) {
            Ok(bytes) => println!("  DOY {doy}: {:.1} MB", bytes as f64 / 1_048_576.0),
            Err(e) => eprintln!("  Warning: DOY {doy} failed: {e}"),
        }
    }

    // -----------------------------------------------------------------------
    // Step 2: Parse FGM to minute-level records
    // -----------------------------------------------------------------------
    println!("[2/5] Parsing FGM to minute-level records...");

    let mut all_minutes: Vec<MmsFgmMinuteRecord> = Vec::new();
    for day_offset in 0..cli.n_days {
        let date = start + chrono::Duration::days(day_offset as i64);
        let doy = date.ordinal() as u16;
        let y = date.year() as u16;
        let path = fgm_dir.join(format!("mms1_fgm_srvy_l2_{y}_{doy}_{doy}.csv"));
        if path.exists() {
            let content = fs::read_to_string(&path)?;
            let mut records = parse_mms_fgm_hapi_csv_minutes(&content);
            println!("  Day {} (DOY {}): {} minutes", date, doy, records.len());
            all_minutes.append(&mut records);
        }
    }

    if all_minutes.is_empty() {
        anyhow::bail!("No MMS FGM data found. Run with --data-dir pointing to cached files.");
    }

    // Filter inner magnetosphere transits and sort.
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

    // Reference midnight for SITL timestamp comparisons.
    let reference_midnight = NaiveDate::from_yo_opt(fy as i32, fd as u32)
        .and_then(|d| d.and_hms_opt(fh as u32, fm as u32, 0))
        .context("failed to build reference midnight")?;

    println!(
        "  Total: {} minutes across {:.1} hours",
        all_minutes.len(),
        all_minutes.last().map(|r| r.elapsed_hours).unwrap_or(0.0)
    );

    // -----------------------------------------------------------------------
    // Step 3: Load SITL catalog
    // -----------------------------------------------------------------------
    println!("[3/5] Loading SITL event catalog...");

    let sitl_content = if let Some(ref path) = cli.sitl_csv {
        println!("  Reading from {}", path.display());
        fs::read_to_string(path).with_context(|| format!("reading SITL CSV: {}", path.display()))?
    } else {
        let provider = MmsSitlProvider {
            start_date: start,
            end_date: end,
        };
        let config = FetchConfig {
            output_dir: cli.data_dir.clone(),
            skip_existing: true,
            verify_checksums: false,
        };
        if !provider.is_cached(&config) {
            provider.fetch(&config)?;
        }
        let fname = format!(
            "mms_sitl_{}_{}.csv",
            start.format("%Y%m%d"),
            end.format("%Y%m%d")
        );
        let path = cli.data_dir.join("mms").join(&fname);
        fs::read_to_string(&path)
            .with_context(|| format!("reading cached SITL CSV: {}", path.display()))?
    };

    let all_sitl = parse_mms_sitl_csv(&sitl_content);
    println!("  Parsed {} SITL events total", all_sitl.len());

    // Apply FOM filter then magnetopause keyword filter.
    let fom_filtered: Vec<MmsEventInterval> = all_sitl
        .iter()
        .filter(|e| e.fom >= cli.min_fom)
        .cloned()
        .collect();
    let mp_sitl = filter_magnetopause_events(&fom_filtered);

    println!(
        "  After FOM>={} filter: {} events; after MP keyword filter: {} events",
        cli.min_fom,
        fom_filtered.len(),
        mp_sitl.len()
    );

    if mp_sitl.is_empty() {
        eprintln!(
            "  WARNING: no SITL magnetopause events found in window {} to {}.\n\
             Recall will be 0 for all methods.  Check --sitl-csv or network access.",
            start, end
        );
    }

    // -----------------------------------------------------------------------
    // Step 4: Gradient detector baseline
    // -----------------------------------------------------------------------
    println!("[4/5] Running gradient detector baseline...");

    let gradient_indices = detect_magnetopause_crossings_filtered(
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

    // Detect transitions (same logic as heliosphere-mms-multiday).
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
    // Evaluate both methods against SITL
    // -----------------------------------------------------------------------
    let n_sitl = mp_sitl.len();

    let grad_eval = eval_against_sitl(
        &gradient_hours,
        &reference_midnight,
        &mp_sitl,
        "|B|-gradient + rotation",
        n_sitl,
    );

    let cd_eval = eval_against_sitl(
        &cd_hours,
        &reference_midnight,
        &mp_sitl,
        "CD associator (32D)",
        n_sitl,
    );

    // Build SITL event summaries.
    let sitl_events: Vec<SitlEventSummary> = mp_sitl
        .iter()
        .map(|ev| {
            let dur_min = ev.end.signed_duration_since(ev.start).num_seconds() as f64 / 60.0;
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
            SitlEventSummary {
                start: ev.start.format("%Y-%m-%dT%H:%M:%S").to_string(),
                end: ev.end.format("%Y-%m-%dT%H:%M:%S").to_string(),
                duration_minutes: dur_min,
                fom: ev.fom,
                discussion: ev.discussion.clone(),
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

    let results = SitlLabeledResults {
        start_date: cli.start_date.clone(),
        n_days: cli.n_days,
        embedding_dim: cli.embedding_dim,
        n_sitl_events_raw: all_sitl.len(),
        n_sitl_events_filtered: n_sitl,
        n_fgm_minutes: all_minutes.len(),
        gradient_eval: grad_eval,
        cd_eval,
        delta_precision: delta_p,
        delta_recall: delta_r,
        delta_f1,
        sitl_events,
        ascii_table: ascii,
    };

    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.out_json, serde_json::to_string_pretty(&results)?)?;
    println!("Wrote {}", cli.out_json.display());

    Ok(())
}

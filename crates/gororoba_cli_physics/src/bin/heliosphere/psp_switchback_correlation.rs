//! PSP micro-switchback correlation experiment (E-271).
//!
//! WHY: E-269 found a 0.311/hr CD associator fire rate in quiet PSP E3 solar
//! wind -- intervals with no Br sign reversal or rotation >= 20 deg in a
//! +-5 min sliding window.  The Discussion in the JGR manuscript hypothesizes
//! that these "quiet" fires detect sub-minute kinetic-scale magnetic topology
//! changes (micro-switchbacks) that are averaged away by the 1-min HAPI
//! decimation and the +-5 min classification window.  This experiment tests
//! that hypothesis by computing the enrichment of quiet-interval CD fires
//! within +-k minutes of large consecutive-minute B-vector rotations that
//! fall below the Alfvenic classifier threshold.
//!
//! WHAT: Uses the same cached PSP E3 data as E-239.  Computes the consecutive-
//! minute rotation angle (angle between B(t) and B(t-1) unit vectors) for
//! every minute.  A "micro-rotation event" is a consecutive rotation exceeding
//! a threshold (default 15 deg, sub-Alfvenic-classifier at 1-min cadence).
//! Reports:
//!   - Fraction of quiet-interval CD fires within +-k min of a micro-rotation
//!     event  ("enrichment rate")
//!   - Fraction of all quiet minutes within +-k min of a micro-rotation event
//!     ("background rate")
//!   - Enrichment factor = enrichment_rate / background_rate
//!
//! HOW:
//!   cargo run --release -p gororoba_cli_physics \
//!     --bin heliosphere -- psp-switchback-correlation \
//!     --start-date 2020-01-20 --end-date 2020-02-10

use anyhow::{Context, Result};
use chrono::NaiveDate;
use clap::Args;
use data_core::{
    catalogs::{
        mms::{MmsFgmMinuteRecord, detect_magnetopause_crossings_filtered},
        psp_fields::{PspFieldsMagMinuteRecord, parse_psp_fields_hapi_csv_minutes},
    },
    download_stack::{DownloadBackend, DownloadStack, TransferRequest},
};
use serde::Serialize;
use std::{fs, path::PathBuf};

// ============================================================================
// CLI
// ============================================================================

#[derive(Args, Debug)]
pub struct Cli {
    /// Start date (YYYY-MM-DD).
    #[arg(long, default_value = "2020-01-20")]
    start_date: String,

    /// End date (YYYY-MM-DD).
    #[arg(long, default_value = "2020-02-10")]
    end_date: String,

    /// Embedding dimension.
    #[arg(long, default_value_t = 32)]
    embedding_dim: usize,

    /// Takens lag in minutes.
    #[arg(long, default_value_t = 1)]
    takens_lag: usize,

    /// Half-width (minutes) of the sliding window used for per-minute
    /// Alfvenic/Compressive/Quiet class labeling.
    #[arg(long, default_value_t = 5)]
    label_half_window: usize,

    /// Relative delta-|B| threshold for Alfvenic class eligibility.
    #[arg(long, default_value_t = 0.25)]
    alfven_compress_max: f64,

    /// Relative delta-|B| threshold for CompressiveBoundary class.
    #[arg(long, default_value_t = 0.30)]
    compress_boundary_min: f64,

    /// Minimum rotation (deg) in Alfvenic classifier window for Alfvenic label.
    #[arg(long, default_value_t = 20.0)]
    alfven_rot_min_deg: f64,

    /// Consecutive-minute B-vector rotation threshold (deg) for a micro-rotation
    /// event.  Default 15 deg: below the Alfvenic 20-deg window-average threshold
    /// but physically significant at 1-min cadence.
    #[arg(long, default_value_t = 15.0)]
    micro_rot_threshold_deg: f64,

    /// Half-width (minutes) of the neighborhood searched around each quiet CD
    /// fire for micro-rotation events.  Also applied to the background rate.
    #[arg(long, default_value_t = 3)]
    correlation_half_window: usize,

    /// Window size (minutes) for the gradient crossing detector.
    #[arg(long, default_value_t = 10)]
    crossing_window_minutes: usize,

    /// |B| gradient threshold (nT) for gradient detector.
    #[arg(long, default_value_t = 5.0)]
    bmag_gradient_threshold: f64,

    /// Rotation threshold (degrees) for gradient detector.
    #[arg(long, default_value_t = 30.0)]
    rotation_threshold_deg: f64,

    /// Instrument noise floor (nT) -- PSP FIELDS fluxgate.
    #[arg(long, default_value_t = 0.01)]
    bmag_noise_floor: f64,

    /// Data cache root directory.
    #[arg(long, default_value = "data/external")]
    data_dir: PathBuf,

    /// Output JSON path.
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/psp_switchback_correlation.json"
    )]
    out_json: PathBuf,
}

// ============================================================================
// Window classification (duplicated from E-239 -- binaries cannot share code)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WindowClass {
    Alfvenic,
    CompressiveBoundary,
    Quiet,
}

fn classify_windows(
    records: &[PspFieldsMagMinuteRecord],
    half_win: usize,
    compress_boundary_min: f64,
    alfven_compress_max: f64,
    alfven_rot_min_deg: f64,
) -> Vec<WindowClass> {
    let n = records.len();
    let mut labels = vec![WindowClass::Quiet; n];

    for (i, _r) in records.iter().enumerate().take(n) {
        let lo = i.saturating_sub(half_win);
        let hi = (i + half_win + 1).min(n);
        let window = &records[lo..hi];

        let b_vals: Vec<f64> = window.iter().map(|r| r.b_magnitude).collect();
        let b_mean = b_vals.iter().sum::<f64>() / b_vals.len() as f64;
        if b_mean <= 0.0 || !b_mean.is_finite() {
            continue;
        }
        let b_min = b_vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let b_max = b_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let rel_delta_b = (b_max - b_min) / b_mean;

        let mid = lo + window.len() / 2;
        let rotation_deg = compute_window_rotation_deg(records, lo, mid, mid, hi);

        let br_min = window.iter().map(|r| r.br).fold(f64::INFINITY, f64::min);
        let br_max = window
            .iter()
            .map(|r| r.br)
            .fold(f64::NEG_INFINITY, f64::max);
        let has_br_reversal = br_min * br_max < 0.0;

        let is_compressive_boundary = rel_delta_b >= compress_boundary_min && rotation_deg >= 15.0;
        let is_alfvenic = (has_br_reversal || rotation_deg >= alfven_rot_min_deg)
            && rel_delta_b < alfven_compress_max;

        labels[i] = if is_compressive_boundary {
            WindowClass::CompressiveBoundary
        } else if is_alfvenic {
            WindowClass::Alfvenic
        } else {
            WindowClass::Quiet
        };
    }
    labels
}

/// Angle (degrees) between mean B-vectors in two index ranges (for window labeling).
fn compute_window_rotation_deg(
    records: &[PspFieldsMagMinuteRecord],
    a_lo: usize,
    a_hi: usize,
    b_lo: usize,
    b_hi: usize,
) -> f64 {
    let mean_unit_b = |lo: usize, hi: usize| -> Option<[f64; 3]> {
        let slice = &records[lo..hi];
        if slice.is_empty() {
            return None;
        }
        let br = slice.iter().map(|r| r.br).sum::<f64>() / slice.len() as f64;
        let bt = slice.iter().map(|r| r.bt).sum::<f64>() / slice.len() as f64;
        let bn = slice.iter().map(|r| r.bn).sum::<f64>() / slice.len() as f64;
        let mag = (br * br + bt * bt + bn * bn).sqrt();
        if mag < 1e-9 {
            return None;
        }
        Some([br / mag, bt / mag, bn / mag])
    };
    let (Some(u), Some(v)) = (mean_unit_b(a_lo, a_hi), mean_unit_b(b_lo, b_hi)) else {
        return 0.0;
    };
    let dot = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]).clamp(-1.0, 1.0);
    dot.acos().to_degrees()
}

// ============================================================================
// Consecutive-minute rotation
// ============================================================================

/// Compute the rotation angle (degrees) between B(i) and B(i-1) unit vectors.
///
/// Returns 0.0 for the first record (no predecessor).  Used to identify
/// micro-rotation events: brief large-angle deflections that are smoothed
/// away by the +-5 min classification window but physically represent
/// kinetic-scale magnetic topology changes.
fn consecutive_rotation_deg_per_minute(records: &[PspFieldsMagMinuteRecord]) -> Vec<f64> {
    let n = records.len();
    let mut out = vec![0.0_f64; n];
    for i in 1..n {
        let unit_b = |r: &PspFieldsMagMinuteRecord| -> Option<[f64; 3]> {
            let mag = (r.br * r.br + r.bt * r.bt + r.bn * r.bn).sqrt();
            if mag < 1e-9 || !mag.is_finite() {
                return None;
            }
            Some([r.br / mag, r.bt / mag, r.bn / mag])
        };
        let (Some(u), Some(v)) = (unit_b(&records[i - 1]), unit_b(&records[i])) else {
            continue;
        };
        let dot = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]).clamp(-1.0, 1.0);
        out[i] = dot.acos().to_degrees();
    }
    out
}

// ============================================================================
// Output
// ============================================================================

#[derive(Debug, Serialize)]
struct SwitchbackCorrelationResults {
    start_date: String,
    end_date: String,
    embedding_dim: usize,
    n_minutes: usize,
    n_gap_skipped: usize,
    /// CD fires in quiet intervals (E-239 equivalence check).
    n_quiet_cd_fires: usize,
    /// Consecutive rotation threshold used to define micro-rotation events.
    micro_rot_threshold_deg: f64,
    /// Half-window (minutes) used for enrichment neighborhood.
    correlation_half_window: usize,
    /// Number of quiet minutes that are within +- correlation_half_window
    /// minutes of at least one micro-rotation event.
    n_quiet_microrot_background: usize,
    /// Total quiet minutes.
    n_quiet_minutes: usize,
    /// Fraction of quiet minutes near a micro-rotation event (background).
    quiet_background_microrot_fraction: f64,
    /// Number of quiet-interval CD fires within +- correlation_half_window
    /// minutes of at least one micro-rotation event.
    n_quiet_cd_fires_microrot_proximate: usize,
    /// Fraction of quiet CD fires near a micro-rotation event (enrichment rate).
    quiet_cd_microrot_fraction: f64,
    /// Enrichment factor: cd_fraction / background_fraction.
    /// Values >> 1 confirm the CD associator preferentially fires near
    /// kinetic-scale topological folds rather than randomly in quiet solar wind.
    enrichment_factor: f64,
    /// Total micro-rotation events in E3 window.
    n_microrot_events_total: usize,
    ascii_summary: String,
}

fn build_ascii_summary(results: &SwitchbackCorrelationResults) -> String {
    let mut s = String::new();
    s.push_str("PSP E3 micro-switchback enrichment analysis (E-271)\n");
    s.push_str("=====================================================\n");
    s.push_str(&format!(
        "  Window: {} to {}  ({} minutes, {} gap-skipped)\n",
        results.start_date, results.end_date, results.n_minutes, results.n_gap_skipped
    ));
    s.push_str(&format!(
        "  Micro-rotation threshold: {:.0} deg consecutive  |  Neighborhood: +- {} min\n",
        results.micro_rot_threshold_deg, results.correlation_half_window
    ));
    s.push_str(&format!(
        "  Micro-rotation events total: {}\n",
        results.n_microrot_events_total
    ));
    s.push('\n');
    s.push_str("  Enrichment analysis (quiet intervals only):\n");
    s.push_str(&format!(
        "    Quiet minutes total:          {:>6}\n",
        results.n_quiet_minutes
    ));
    s.push_str(&format!(
        "    Near micro-rotation (backgrd):{:>6}  ({:.1}%)\n",
        results.n_quiet_microrot_background,
        100.0 * results.quiet_background_microrot_fraction
    ));
    s.push_str(&format!(
        "    Quiet CD fires total:         {:>6}\n",
        results.n_quiet_cd_fires
    ));
    s.push_str(&format!(
        "    CD fires near micro-rotation: {:>6}  ({:.1}%)\n",
        results.n_quiet_cd_fires_microrot_proximate,
        100.0 * results.quiet_cd_microrot_fraction
    ));
    s.push('\n');
    s.push_str(&format!(
        "  Enrichment factor: {:.2}x\n",
        results.enrichment_factor
    ));
    if results.enrichment_factor >= 2.0 {
        s.push_str("  INTERPRETATION: CD associator significantly enriched near micro-rotation\n");
        s.push_str("  events -- consistent with detection of kinetic-scale topological folds\n");
        s.push_str("  (micro-switchbacks) in quiet PSP E3 solar wind (C-1634).\n");
    } else if results.enrichment_factor >= 1.3 {
        s.push_str("  INTERPRETATION: Moderate enrichment -- CD associator partially sensitive\n");
        s.push_str("  to sub-classifier micro-rotation events.\n");
    } else {
        s.push_str("  INTERPRETATION: No significant enrichment -- quiet CD fires not\n");
        s.push_str("  preferentially co-located with micro-rotation events at this threshold.\n");
    }
    s
}

// ============================================================================
// Main
// ============================================================================

pub fn run(cli: Cli) -> Result<()> {
    println!("=== PSP E3 Micro-Switchback Correlation (E-271) ===");
    println!(
        "Window: {} to {}  |  Micro-rot threshold: {:.0} deg  |  Neighborhood: +- {} min",
        cli.start_date, cli.end_date, cli.micro_rot_threshold_deg, cli.correlation_half_window,
    );

    // -----------------------------------------------------------------------
    // Step 1: Load PSP E3 data
    // -----------------------------------------------------------------------
    println!("[1/5] Loading PSP FIELDS L2 MAG RTN 1-min data...");

    let start = NaiveDate::parse_from_str(&cli.start_date, "%Y-%m-%d")
        .with_context(|| format!("bad start date: {}", cli.start_date))?;
    let end = NaiveDate::parse_from_str(&cli.end_date, "%Y-%m-%d")
        .with_context(|| format!("bad end date: {}", cli.end_date))?;

    let mut all_minutes: Vec<PspFieldsMagMinuteRecord> = Vec::new();
    let mut chunk_start = start;

    while chunk_start <= end {
        // Use the same chunk formula as E-239 so this binary reads the same
        // cached files (chunk_end = chunk_start + 7 days, not + 6 days).
        let chunk_end = (chunk_start + chrono::Duration::days(7)).min(end);
        let fname = format!(
            "psp_fields_mag_{}_{}.csv",
            chunk_start.format("%Y%m%d"),
            chunk_end.format("%Y%m%d")
        );
        let path = cli.data_dir.join("psp_fields").join(&fname);

        let content = if path.exists() {
            println!("  {}: cached ({} bytes)", fname, fs::metadata(&path)?.len());
            fs::read_to_string(&path)?
        } else {
            println!("  {}: downloading...", fname);
            let dataset = "PSP_FLD_L2_MAG_RTN_1MIN";
            let t_min = format!("{}T00:00:00Z", chunk_start);
            let t_max = format!("{}T23:59:59Z", chunk_end);
            let url = format!(
                "https://cdaweb.gsfc.nasa.gov/hapi/data\
                 ?id={dataset}&time.min={t_min}&time.max={t_max}\
                 &format=csv&parameters=psp_fld_l2_mag_RTN_1min"
            );
            let mut request = TransferRequest::download(&url, &path);
            request.backend = DownloadBackend::CurlCli;
            match DownloadStack::default().recover(&request) {
                Ok(_) => fs::read_to_string(&path)?,
                Err(e) => {
                    eprintln!("  Warning: {fname} download failed: {e}");
                    chunk_start = chunk_end + chrono::Duration::days(1);
                    continue;
                }
            }
        };

        let mut records = parse_psp_fields_hapi_csv_minutes(&content);
        println!("  {}: {} minute records", fname, records.len());
        all_minutes.append(&mut records);
        chunk_start = chunk_end + chrono::Duration::days(1);
    }

    if all_minutes.is_empty() {
        anyhow::bail!("No PSP data found for {} to {}", start, end);
    }

    all_minutes.sort_by(|a, b| {
        a.year
            .cmp(&b.year)
            .then(a.doy.cmp(&b.doy))
            .then(a.hour.cmp(&b.hour))
            .then(a.minute.cmp(&b.minute))
    });
    all_minutes.dedup_by(|a, b| {
        b.year == a.year && b.doy == a.doy && b.hour == a.hour && b.minute == a.minute
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
    println!(
        "  Total: {} minutes ({:.1} days)",
        all_minutes.len(),
        all_minutes
            .last()
            .map(|r| r.elapsed_hours / 24.0)
            .unwrap_or(0.0),
    );

    // -----------------------------------------------------------------------
    // Step 2: Classify windows
    // -----------------------------------------------------------------------
    println!("[2/5] Classifying windows...");

    let labels = classify_windows(
        &all_minutes,
        cli.label_half_window,
        cli.compress_boundary_min,
        cli.alfven_compress_max,
        cli.alfven_rot_min_deg,
    );
    let n_quiet = labels.iter().filter(|&&c| c == WindowClass::Quiet).count();
    println!(
        "  Quiet: {}  ({:.1}%)",
        n_quiet,
        100.0 * n_quiet as f64 / all_minutes.len() as f64
    );

    // -----------------------------------------------------------------------
    // Step 3: Consecutive-minute rotation events
    // -----------------------------------------------------------------------
    println!(
        "[3/5] Computing consecutive-minute rotations (threshold: {:.0} deg)...",
        cli.micro_rot_threshold_deg
    );

    let consec_rot = consecutive_rotation_deg_per_minute(&all_minutes);
    let microrot_mask: Vec<bool> = consec_rot
        .iter()
        .map(|&r| r >= cli.micro_rot_threshold_deg)
        .collect();
    let n_microrot_events_total = microrot_mask.iter().filter(|&&v| v).count();
    println!(
        "  Micro-rotation events (>= {:.0} deg): {}",
        cli.micro_rot_threshold_deg, n_microrot_events_total
    );

    // Precompute: for each minute index, is there a micro-rotation event within
    // +- correlation_half_window minutes?
    let n = all_minutes.len();
    let half = cli.correlation_half_window;
    let near_microrot: Vec<bool> = (0..n)
        .map(|i| {
            let lo = i.saturating_sub(half);
            let hi = (i + half + 1).min(n);
            microrot_mask[lo..hi].iter().any(|&v| v)
        })
        .collect();

    // -----------------------------------------------------------------------
    // Step 4: CD associator transitions
    // -----------------------------------------------------------------------
    println!(
        "[4/5] Running {}D Takens embedding + CD associator...",
        cli.embedding_dim
    );

    let channels: usize = 4;
    let steps = cli.embedding_dim / channels;
    let window_rows = (steps - 1) * cli.takens_lag + 1;

    if all_minutes.len() < window_rows + 2 {
        anyhow::bail!(
            "Not enough data: need {} rows, have {}",
            window_rows + 2,
            all_minutes.len()
        );
    }

    let expected_span_hours = (steps - 1) as f64 * cli.takens_lag as f64 / 60.0;
    let gap_tolerance_hours = 2.0 * cli.takens_lag as f64 / 60.0;
    let max_window_span_hours = expected_span_hours + gap_tolerance_hours;

    let mut embedded_vectors: Vec<Vec<f64>> = Vec::new();
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
        // Algebraic note: when |B| < noise_floor the denominator locks to a
        // constant, transitioning the normalization from scale-invariant
        // (relative topology) to scale-dependent (absolute amplitude).
        let denom = local_mean_b.max(cli.bmag_noise_floor);

        let mut v = vec![0.0; cli.embedding_dim];
        for (s, &ri) in sample_indices.iter().enumerate() {
            let rec = &all_minutes[ri];
            v[s * channels] = rec.br / denom;
            v[s * channels + 1] = rec.bt / denom;
            v[s * channels + 2] = rec.bn / denom;
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
    let mut cd_minute_indices: Vec<usize> = Vec::new();

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
        let half_t = trans_window;
        let mut last_trans_idx: Option<usize> = None;
        for i in half_t..associators.len().saturating_sub(half_t) {
            let pre_mean: f64 =
                associators[i.saturating_sub(half_t)..i].iter().sum::<f64>() / half_t as f64;
            let post_mean: f64 = associators[i..(i + half_t).min(associators.len())]
                .iter()
                .sum::<f64>()
                / half_t.min(associators.len() - i) as f64;
            let jump = (post_mean - pre_mean).abs();
            if jump > threshold {
                let dominated =
                    last_trans_idx.is_some_and(|prev| i.saturating_sub(prev) < trans_window);
                if !dominated {
                    cd_minute_indices.push(assoc_minute_indices[i]);
                    last_trans_idx = Some(i);
                }
            }
        }
    }

    println!("  CD associator: {} fires total", cd_minute_indices.len());

    // -----------------------------------------------------------------------
    // Step 5: Enrichment analysis
    // -----------------------------------------------------------------------
    println!("[5/5] Computing micro-rotation enrichment...");

    // Quiet-interval CD fires.
    let quiet_cd_indices: Vec<usize> = cd_minute_indices
        .iter()
        .copied()
        .filter(|&idx| idx < labels.len() && labels[idx] == WindowClass::Quiet)
        .collect();
    let n_quiet_cd_fires = quiet_cd_indices.len();
    println!("  Quiet-interval CD fires: {n_quiet_cd_fires}");

    // Background: quiet minutes near a micro-rotation event.
    let n_quiet_microrot_background = labels
        .iter()
        .zip(near_microrot.iter())
        .filter(|&(&cls, &nm)| cls == WindowClass::Quiet && nm)
        .count();
    let quiet_background_microrot_fraction = if n_quiet == 0 {
        0.0
    } else {
        n_quiet_microrot_background as f64 / n_quiet as f64
    };

    // Enrichment: quiet CD fires near a micro-rotation event.
    let n_quiet_cd_fires_microrot_proximate = quiet_cd_indices
        .iter()
        .filter(|&&idx| idx < near_microrot.len() && near_microrot[idx])
        .count();
    let quiet_cd_microrot_fraction = if n_quiet_cd_fires == 0 {
        0.0
    } else {
        n_quiet_cd_fires_microrot_proximate as f64 / n_quiet_cd_fires as f64
    };

    let enrichment_factor = if quiet_background_microrot_fraction < 1e-9 {
        0.0
    } else {
        quiet_cd_microrot_fraction / quiet_background_microrot_fraction
    };

    println!(
        "  Background (quiet near microrot): {}/{} = {:.1}%",
        n_quiet_microrot_background,
        n_quiet,
        100.0 * quiet_background_microrot_fraction,
    );
    println!(
        "  CD fires near microrot: {}/{} = {:.1}%",
        n_quiet_cd_fires_microrot_proximate,
        n_quiet_cd_fires,
        100.0 * quiet_cd_microrot_fraction,
    );
    println!("  Enrichment factor: {enrichment_factor:.2}x");

    // -----------------------------------------------------------------------
    // Gradient baseline validation -- confirm quiet gradient fires are not
    // enriched near micro-rotation events (they should not be: the gradient
    // detector uses a 10-min window, similar scale to the Alfvenic classifier).
    // -----------------------------------------------------------------------
    let adapted: Vec<MmsFgmMinuteRecord> = all_minutes
        .iter()
        .map(|r| MmsFgmMinuteRecord {
            year: r.year,
            doy: r.doy,
            hour: r.hour,
            minute: r.minute,
            elapsed_hours: r.elapsed_hours,
            bx_gse: r.br,
            by_gse: r.bt,
            bz_gse: r.bn,
            b_magnitude: r.b_magnitude,
        })
        .collect();
    let gradient_indices = detect_magnetopause_crossings_filtered(
        &adapted,
        cli.crossing_window_minutes,
        cli.bmag_gradient_threshold,
        Some(cli.rotation_threshold_deg),
    );
    let quiet_grad_indices: Vec<usize> = gradient_indices
        .iter()
        .copied()
        .filter(|&idx| idx < labels.len() && labels[idx] == WindowClass::Quiet)
        .collect();
    let n_quiet_grad_fires_microrot = quiet_grad_indices
        .iter()
        .filter(|&&idx| idx < near_microrot.len() && near_microrot[idx])
        .count();
    let grad_cd_microrot_fraction = if quiet_grad_indices.is_empty() {
        0.0
    } else {
        n_quiet_grad_fires_microrot as f64 / quiet_grad_indices.len() as f64
    };
    let grad_enrichment_factor = if quiet_background_microrot_fraction < 1e-9 {
        0.0
    } else {
        grad_cd_microrot_fraction / quiet_background_microrot_fraction
    };
    println!(
        "  Gradient baseline enrichment: {:.2}x  (CD: {:.2}x)",
        grad_enrichment_factor, enrichment_factor
    );

    // -----------------------------------------------------------------------
    // Output
    // -----------------------------------------------------------------------
    let mut results = SwitchbackCorrelationResults {
        start_date: cli.start_date.clone(),
        end_date: cli.end_date.clone(),
        embedding_dim: cli.embedding_dim,
        n_minutes: all_minutes.len(),
        n_gap_skipped,
        n_quiet_cd_fires,
        micro_rot_threshold_deg: cli.micro_rot_threshold_deg,
        correlation_half_window: cli.correlation_half_window,
        n_quiet_microrot_background,
        n_quiet_minutes: n_quiet,
        quiet_background_microrot_fraction,
        n_quiet_cd_fires_microrot_proximate,
        quiet_cd_microrot_fraction,
        enrichment_factor,
        n_microrot_events_total,
        ascii_summary: String::new(),
    };
    results.ascii_summary = build_ascii_summary(&results);

    println!("\n{}", results.ascii_summary);

    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    let json_str = serde_json::to_string_pretty(&results)?;
    fs::write(&cli.out_json, &json_str)
        .with_context(|| format!("write {}", cli.out_json.display()))?;
    println!("Wrote {}", cli.out_json.display());

    Ok(())
}

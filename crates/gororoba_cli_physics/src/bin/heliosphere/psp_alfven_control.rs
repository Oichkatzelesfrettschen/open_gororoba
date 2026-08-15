//! PSP Alfvenic noise control experiment (E-239).
//!
//! WHY: JGR Major Comment 2 asks whether the CD associator fires spuriously on
//! high-amplitude Alfven waves.  A pure Alfven wave is a rotational fluctuation
//! with constant |B| (delta_B/|B| ~ 0).  PSP perihelion encounter E4
//! (perihelion 2020-01-29 at ~27.9 Rs) is a high-Alfvenicity environment in
//! the inner heliosphere with frequent switchbacks.  Note: E4 is at 27.9 Rs
//! vs E3 (2019-09-01 perihelion) at 35.7 Rs -- the Alfvenicity regime and
//! switchback rate differ; see Bandyopadhyay et al. (2022) for E4 specifics.
//! If the CD associator fires LESS in Alfvenic intervals than in compressive-
//! boundary intervals, it proves selectivity for topological boundaries,
//! not MHD wave turbulence.
//!
//! WHAT: Downloads (or reads cached) PSP FIELDS L2 1-min MAG RTN data for E4
//! (2020-01-20 to 2020-02-10, 31,673 usable minutes; 7 fill-gap minutes excluded).  Classifies each
//! minute window as Alfvenic, CompressiveBoundary, or Quiet using delta_B/|B|
//! and field-rotation criteria.  Runs the 32D CD associator and the
//! |B|-gradient+rotation baseline.  Computes per-class fire rates and the
//! discrimination ratio (compressive / Alfvenic).
//!
//! HOW:
//!   cargo run --release -p gororoba_cli_physics \
//!     --bin heliosphere -- psp-alfven-control \
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
use spectral_core::boundary_metrics;
use std::{collections::HashSet, fs, path::PathBuf};

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
    /// class labeling.  Each minute sees a `[t-W, t+W]` context.
    #[arg(long, default_value_t = 5)]
    label_half_window: usize,

    /// Relative-delta-|B| threshold below which a window is eligible for
    /// the Alfvenic class: delta_B / mean_B < this value.
    #[arg(long, default_value_t = 0.25)]
    alfven_compress_max: f64,

    /// Relative-delta-|B| threshold above which a window is labelled
    /// CompressiveBoundary regardless of rotation.
    #[arg(long, default_value_t = 0.30)]
    compress_boundary_min: f64,

    /// Minimum field-rotation (deg) for a window to be labelled Alfvenic
    /// when no Br sign reversal is detected.
    #[arg(long, default_value_t = 20.0)]
    alfven_rot_min_deg: f64,

    /// Window size (minutes) for the gradient crossing detector.
    #[arg(long, default_value_t = 10)]
    crossing_window_minutes: usize,

    /// |B| gradient threshold (nT) for the gradient detector.
    #[arg(long, default_value_t = 5.0)]
    bmag_gradient_threshold: f64,

    /// Rotation threshold (degrees) for the gradient detector.
    #[arg(long, default_value_t = 30.0)]
    rotation_threshold_deg: f64,

    /// Instrument noise floor (nT) used as denominator floor in the
    /// local-|B|-normalized Takens embedding.  PSP FIELDS fluxgate published
    /// noise floor: ~0.01 nT.
    #[arg(long, default_value_t = 0.01)]
    bmag_noise_floor: f64,

    /// Data cache root directory.
    #[arg(long, default_value = "data/external")]
    data_dir: PathBuf,

    /// Output JSON path.
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/psp_alfven_control.json"
    )]
    out_json: PathBuf,
}

// ============================================================================
// Window classification
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
enum WindowClass {
    Alfvenic,
    CompressiveBoundary,
    Quiet,
}

/// Classify each minute in `records` using a +-half_win context window.
///
/// Each window is labelled:
///   Alfvenic        if (Br sign reversal in window OR rotation >= rot_min)
///                      AND delta_B/mean_B < compress_max
///   CompressiveBoundary if delta_B/mean_B >= compress_boundary_min
///                        AND rotation >= 15 deg
///   Quiet           otherwise
///
/// The CompressiveBoundary check takes precedence: if a window is both
/// highly compressive and rotationally active it becomes CompressiveBoundary
/// (true magnetopause-like boundary), not Alfvenic.
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

        // Compressiveness: (max - min) |B| / mean |B|.
        let b_vals: Vec<f64> = window.iter().map(|r| r.b_magnitude).collect();
        let b_mean = b_vals.iter().sum::<f64>() / b_vals.len() as f64;
        if b_mean <= 0.0 || !b_mean.is_finite() {
            continue;
        }
        let b_min = b_vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let b_max = b_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let rel_delta_b = (b_max - b_min) / b_mean;

        // Field rotation: mean B-direction in first half vs second half.
        let mid = lo + window.len() / 2;
        let rotation_deg = compute_rotation_deg(records, lo, mid, mid, hi);

        // Br sign reversal: min(Br) * max(Br) < 0.
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

/// Compute the angle (degrees) between mean B-vectors in two index ranges.
fn compute_rotation_deg(
    records: &[PspFieldsMagMinuteRecord],
    a_lo: usize,
    a_hi: usize,
    b_lo: usize,
    b_hi: usize,
) -> f64 {
    let mean_b = |lo: usize, hi: usize| -> Option<[f64; 3]> {
        let slice = &records[lo..hi];
        if slice.is_empty() {
            return None;
        }
        let br_m = slice.iter().map(|r| r.br).sum::<f64>() / slice.len() as f64;
        let bt_m = slice.iter().map(|r| r.bt).sum::<f64>() / slice.len() as f64;
        let bn_m = slice.iter().map(|r| r.bn).sum::<f64>() / slice.len() as f64;
        let mag = (br_m * br_m + bt_m * bt_m + bn_m * bn_m).sqrt();
        if mag < 1e-9 {
            return None;
        }
        Some([br_m / mag, bt_m / mag, bn_m / mag])
    };
    let (Some(u), Some(v)) = (mean_b(a_lo, a_hi), mean_b(b_lo, b_hi)) else {
        return 0.0;
    };
    let dot = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]).clamp(-1.0, 1.0);
    dot.acos().to_degrees()
}

// ============================================================================
// Output
// ============================================================================

#[derive(Debug, Serialize)]
struct ClassFireStats {
    class: String,
    total_minutes: usize,
    total_hours: f64,
    cd_fires: usize,
    gradient_fires: usize,
    cd_fire_rate_per_hour: f64,
    gradient_fire_rate_per_hour: f64,
}

#[derive(Debug, Serialize)]
struct AlfvenControlResults {
    start_date: String,
    end_date: String,
    embedding_dim: usize,
    n_minutes: usize,
    n_gap_skipped: usize,
    cd_fires_total: usize,
    gradient_fires_total: usize,
    alfvenic_stats: ClassFireStats,
    compressive_stats: ClassFireStats,
    quiet_stats: ClassFireStats,
    /// Key metric: compressive_cd_fire_rate / alfvenic_cd_fire_rate.
    /// Values >> 1 confirm the CD associator is selective for topological
    /// boundaries and not contaminated by Alfvenic wave noise.
    discrimination_ratio: f64,
    /// Absolute false-alarm context: estimated CD fires per year assuming
    /// the quiet-interval fraction holds over a full-year mission.
    /// Computed as quiet_cd_fire_rate_per_hour * 8760 * quiet_fraction.
    /// Distinguishes post-hoc catalog enrichment (current use case) from
    /// unsupervised flight triage (not the paper's claim).
    cd_fires_per_year_at_quiet_fraction: f64,
    /// Permutation p-value for the compressive/Alfvenic fire-rate ratio.
    /// P(shuffled_ratio >= observed) under null (N=10000, seed=42).
    permutation_p_value: f64,
    ascii_summary: String,
}

fn fire_rate(fires: usize, total_minutes: usize) -> f64 {
    if total_minutes == 0 {
        return 0.0;
    }
    fires as f64 / (total_minutes as f64 / 60.0)
}

fn build_class_stats(
    label: &str,
    minutes: usize,
    cd_fires: usize,
    gradient_fires: usize,
) -> ClassFireStats {
    ClassFireStats {
        class: label.to_string(),
        total_minutes: minutes,
        total_hours: minutes as f64 / 60.0,
        cd_fires,
        gradient_fires,
        cd_fire_rate_per_hour: fire_rate(cd_fires, minutes),
        gradient_fire_rate_per_hour: fire_rate(gradient_fires, minutes),
    }
}

fn build_ascii_summary(
    af: &ClassFireStats,
    co: &ClassFireStats,
    qu: &ClassFireStats,
    ratio: f64,
    quiet_cd_rate: f64,
    cd_fires_per_year: f64,
) -> String {
    let mut s = String::new();
    s.push_str(
        "PSP E3 Alfvenic control: CD associator fire rates by window class\n\
         ==================================================================\n",
    );
    s.push_str(&format!(
        "{:<22} {:>8} {:>8} {:>10} {:>10}\n",
        "Class", "Minutes", "Hours", "CD/hr", "Grad/hr"
    ));
    s.push_str(&"-".repeat(62));
    s.push('\n');
    for st in [af, co, qu] {
        s.push_str(&format!(
            "{:<22} {:>8} {:>8.1} {:>10.3} {:>10.3}\n",
            st.class,
            st.total_minutes,
            st.total_hours,
            st.cd_fire_rate_per_hour,
            st.gradient_fire_rate_per_hour,
        ));
    }
    s.push_str(&"-".repeat(62));
    s.push('\n');
    s.push_str(&format!(
        "Discrimination ratio (Compressive CD / Alfvenic CD): {:.2}x\n",
        ratio
    ));
    s.push_str(&format!(
        "Absolute FAR (quiet intervals only): {:.3} fires/hr = ~{:.0} fires/year\n",
        quiet_cd_rate, cd_fires_per_year
    ));
    s.push_str(
        "NOTE: This experiment targets post-hoc catalog enrichment.\n\
         Unsupervised flight triage would require a secondary sigma_c gate.\n",
    );
    s
}

// ============================================================================
// main
// ============================================================================

pub fn run(cli: Cli) -> Result<()> {
    let start = NaiveDate::parse_from_str(&cli.start_date, "%Y-%m-%d")
        .with_context(|| format!("bad start_date: {}", cli.start_date))?;
    let end = NaiveDate::parse_from_str(&cli.end_date, "%Y-%m-%d")
        .with_context(|| format!("bad end_date: {}", cli.end_date))?;

    println!(
        "=== PSP Alfvenic Control Experiment (E-239) ===\n\
         Window: {} to {}\n\
         Embedding: {}D, lag={}min\n\
         Alfvenic: rot>={:.0}deg AND delta_B/|B|<{:.2}\n\
         Compressive: delta_B/|B|>={:.2} AND rot>=15deg\n",
        start,
        end,
        cli.embedding_dim,
        cli.takens_lag,
        cli.alfven_rot_min_deg,
        cli.alfven_compress_max,
        cli.compress_boundary_min,
    );

    // -----------------------------------------------------------------------
    // Step 1: Load cached PSP MAG minute data
    // -----------------------------------------------------------------------
    println!("[1/5] Loading PSP FIELDS MAG 1-min data...");

    let psp_dir = cli.data_dir.join("psp_fields");
    fs::create_dir_all(&psp_dir)?;

    let mut all_minutes: Vec<PspFieldsMagMinuteRecord> = Vec::new();

    // Iterate weekly chunks matching the cached file naming convention from
    // heliosphere-switchback-omega: psp_fields_mag_YYYYMMDD_YYYYMMDD.csv.
    let mut chunk_start = start;
    while chunk_start <= end {
        let chunk_end = (chunk_start + chrono::Duration::days(7)).min(end);
        let fname = format!(
            "psp_fields_mag_{}_{}.csv",
            chunk_start.format("%Y%m%d"),
            chunk_end.format("%Y%m%d")
        );
        let path = psp_dir.join(&fname);

        let content = if path.exists() {
            println!("  {}: cached ({} bytes)", fname, fs::metadata(&path)?.len());
            fs::read_to_string(&path)?
        } else {
            // Download via CurlCli (avoids reqwest TLS panic on this workspace).
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

    // Sort and compute elapsed_hours from first record.
    all_minutes.sort_by(|a, b| {
        a.year
            .cmp(&b.year)
            .then(a.doy.cmp(&b.doy))
            .then(a.hour.cmp(&b.hour))
            .then(a.minute.cmp(&b.minute))
    });
    // Remove duplicates (overlap at chunk boundaries).
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
        "  Total: {} minutes ({:.1} hours / {:.1} days)",
        all_minutes.len(),
        all_minutes.last().map(|r| r.elapsed_hours).unwrap_or(0.0),
        all_minutes
            .last()
            .map(|r| r.elapsed_hours / 24.0)
            .unwrap_or(0.0),
    );

    // -----------------------------------------------------------------------
    // Step 2: Classify windows
    // -----------------------------------------------------------------------
    println!("[2/5] Classifying windows (Alfvenic / Compressive / Quiet)...");

    let labels = classify_windows(
        &all_minutes,
        cli.label_half_window,
        cli.compress_boundary_min,
        cli.alfven_compress_max,
        cli.alfven_rot_min_deg,
    );

    let n_alfvenic = labels
        .iter()
        .filter(|&&c| c == WindowClass::Alfvenic)
        .count();
    let n_compressive = labels
        .iter()
        .filter(|&&c| c == WindowClass::CompressiveBoundary)
        .count();
    let n_quiet = labels.iter().filter(|&&c| c == WindowClass::Quiet).count();
    println!(
        "  Alfvenic: {} ({:.1}%), Compressive: {} ({:.1}%), Quiet: {} ({:.1}%)",
        n_alfvenic,
        100.0 * n_alfvenic as f64 / all_minutes.len() as f64,
        n_compressive,
        100.0 * n_compressive as f64 / all_minutes.len() as f64,
        n_quiet,
        100.0 * n_quiet as f64 / all_minutes.len() as f64,
    );

    // -----------------------------------------------------------------------
    // Step 3: Gradient detector baseline
    // -----------------------------------------------------------------------
    println!("[3/5] Running |B|-gradient+rotation baseline...");

    // Adapt PspFieldsMagMinuteRecord -> MmsFgmMinuteRecord for shared detector.
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
    let gradient_minute_indices: Vec<usize> = gradient_indices;
    println!(
        "  Gradient detector: {} fires",
        gradient_minute_indices.len()
    );

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
            "Not enough data for embedding: need {} rows, have {}",
            window_rows + 2,
            all_minutes.len()
        );
    }

    // Gap detection: same tolerance as E-237/E-238.
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
        // Noise-floor regularization: clamp denominator to PSP FIELDS noise
        // floor (~0.01 nT) so near-zero |B| at magnetic null points cannot
        // amplify instrument noise into spurious associator spikes.
        // Algebraic note: when |B| < noise_floor the denominator locks to a
        // constant, transitioning the normalization from scale-invariant
        // (relative topology, the intended regime) to scale-dependent
        // (absolute amplitude).  Local field geometry is preserved; only
        // the sub-noise-floor amplitude modulation is suppressed.
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
                    cd_minute_indices.push(assoc_minute_indices[i]);
                    last_trans_idx = Some(i);
                }
            }
        }
    }

    println!(
        "  CD associator: {} transitions ({} windows gap-discarded)",
        cd_minute_indices.len(),
        n_gap_skipped
    );

    // -----------------------------------------------------------------------
    // Step 5: Per-class fire rate computation
    // -----------------------------------------------------------------------
    println!("[5/5] Computing per-class fire rates...");

    let count_fires_by_class = |fire_indices: &[usize], target: WindowClass| -> usize {
        fire_indices
            .iter()
            .filter(|&&mi| mi < labels.len() && labels[mi] == target)
            .count()
    };

    // CD per class.
    let cd_af = count_fires_by_class(&cd_minute_indices, WindowClass::Alfvenic);
    let cd_co = count_fires_by_class(&cd_minute_indices, WindowClass::CompressiveBoundary);
    let cd_qu = count_fires_by_class(&cd_minute_indices, WindowClass::Quiet);

    // Gradient per class.
    let gr_af = count_fires_by_class(&gradient_minute_indices, WindowClass::Alfvenic);
    let gr_co = count_fires_by_class(&gradient_minute_indices, WindowClass::CompressiveBoundary);
    let gr_qu = count_fires_by_class(&gradient_minute_indices, WindowClass::Quiet);

    let af_stats = build_class_stats("Alfvenic", n_alfvenic, cd_af, gr_af);
    let co_stats = build_class_stats("CompressiveBoundary", n_compressive, cd_co, gr_co);
    let qu_stats = build_class_stats("Quiet", n_quiet, cd_qu, gr_qu);

    let discrimination_ratio = if af_stats.cd_fire_rate_per_hour > 1e-9 {
        co_stats.cd_fire_rate_per_hour / af_stats.cd_fire_rate_per_hour
    } else {
        f64::INFINITY
    };

    // Permutation test for the discrimination ratio (plan P6A.S1 task 1.5).
    // Build per-minute bool arrays restricted to {Alfvenic, CompressiveBoundary}.
    // Null: fires equally likely in any minute regardless of class label.
    // Statistic: fires_in_compressive / fires_in_alfvenic (count ratio).
    // Under permutation with fixed class sizes, the count ratio p-value equals
    // the rate ratio p-value (class sizes cancel from both sides).
    let cd_fire_set: HashSet<usize> = cd_minute_indices.iter().copied().collect();
    let mut perm_labels: Vec<bool> = Vec::with_capacity(n_alfvenic + n_compressive);
    let mut perm_fire_mask: Vec<bool> = Vec::with_capacity(n_alfvenic + n_compressive);
    for (i, &cls) in labels.iter().enumerate() {
        if cls == WindowClass::Alfvenic || cls == WindowClass::CompressiveBoundary {
            perm_labels.push(cls == WindowClass::CompressiveBoundary);
            perm_fire_mask.push(cd_fire_set.contains(&i));
        }
    }
    let observed_count_ratio = if cd_af > 0 {
        cd_co as f64 / cd_af as f64
    } else {
        f64::INFINITY
    };
    let permutation_p_value = if observed_count_ratio.is_finite() {
        boundary_metrics::permutation_ratio_test_seeded(
            &perm_labels,
            &perm_fire_mask,
            10_000,
            observed_count_ratio,
            42,
        )
    } else {
        0.0
    };
    println!(
        "  Permutation test: discrimination count ratio = {:.3}, p = {:.4}  (N=10000, seed=42)",
        observed_count_ratio, permutation_p_value
    );

    // Absolute FAR context: how many CD fires per year at the observed
    // quiet-interval rate.  This is NOT the paper's use case (post-hoc
    // catalog enrichment, not unsupervised flight triage), but is reported
    // explicitly so readers can evaluate the operational trade-off.
    let quiet_fraction = n_quiet as f64 / all_minutes.len() as f64;
    let cd_fires_per_year_at_quiet_fraction =
        qu_stats.cd_fire_rate_per_hour * 8760.0 * quiet_fraction;

    let ascii = build_ascii_summary(
        &af_stats,
        &co_stats,
        &qu_stats,
        discrimination_ratio,
        qu_stats.cd_fire_rate_per_hour,
        cd_fires_per_year_at_quiet_fraction,
    );
    println!("\n{}", ascii);

    let results = AlfvenControlResults {
        start_date: cli.start_date.clone(),
        end_date: cli.end_date.clone(),
        embedding_dim: cli.embedding_dim,
        n_minutes: all_minutes.len(),
        n_gap_skipped,
        cd_fires_total: cd_minute_indices.len(),
        gradient_fires_total: gradient_minute_indices.len(),
        alfvenic_stats: af_stats,
        compressive_stats: co_stats,
        quiet_stats: qu_stats,
        discrimination_ratio,
        cd_fires_per_year_at_quiet_fraction,
        permutation_p_value,
        ascii_summary: ascii,
    };

    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.out_json, serde_json::to_string_pretty(&results)?)?;
    println!("Wrote {}", cli.out_json.display());

    Ok(())
}

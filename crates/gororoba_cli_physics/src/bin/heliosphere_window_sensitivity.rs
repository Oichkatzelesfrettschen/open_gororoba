//! Window function sensitivity analysis for the CD delay embedding (Phase 4.9).
//!
//! WHY: The plan (P6A task 4.9, board mandate M-5) requires testing whether the
//! detection F1 depends on the window function applied to the delay vector before
//! the CD associator product.  The paper baseline uses an implicit boxcar (uniform)
//! window.  Hamming and Hann windows down-weight the oldest and newest lags relative
//! to the center of the embedding window, reducing spectral leakage in the delay
//! reconstruction.
//!
//! WHAT: Three runs on the same THEMIS window: boxcar (paper baseline), Hamming
//! (alpha=0.54), and Hann (alpha=0.5) windows applied to the channels of each
//! lag sample before the delay vector is formed.  Report F1 and CI for each.
//! The window is applied per-sample (all 4 channels scaled by `w[s]`) over the
//! n_lags samples in the Takens embedding.
//!
//! HOW:
//!   cargo run --release -p gororoba_cli_physics \
//!     --bin heliosphere-window-sensitivity -- \
//!     --start-date 2016-08-29 --n-days 7

use anyhow::{Context, Result};
use chrono::{Datelike, NaiveDate, NaiveDateTime, TimeZone, Utc};
use clap::Parser;
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

const MAD_SCALE_FACTOR: f64 = 1.5;

/// Window function type applied to each lag sample in the Takens embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WindowFn {
    /// Uniform weights -- the paper baseline.
    Boxcar,
    /// Hamming window: `w[n] = 0.54 - 0.46 * cos(2*pi*n / (N-1))`
    Hamming,
    /// Hann window: `w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))`
    Hann,
}

impl WindowFn {
    fn name(&self) -> &'static str {
        match self {
            WindowFn::Boxcar => "boxcar",
            WindowFn::Hamming => "hamming",
            WindowFn::Hann => "hann",
        }
    }

    /// Compute the window coefficients for `n_lags` samples.
    fn coefficients(&self, n_lags: usize) -> Vec<f64> {
        use std::f64::consts::PI;
        let n = n_lags;
        (0..n)
            .map(|k| match self {
                WindowFn::Boxcar => 1.0,
                WindowFn::Hamming => {
                    0.54 - 0.46 * (2.0 * PI * k as f64 / (n - 1).max(1) as f64).cos()
                }
                WindowFn::Hann => 0.5 * (1.0 - (2.0 * PI * k as f64 / (n - 1).max(1) as f64).cos()),
            })
            .collect()
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "heliosphere-window-sensitivity",
    about = "Window function sensitivity (boxcar/Hamming/Hann) for CD delay embedding (M-5)"
)]
struct Cli {
    #[arg(long, default_value = "2016-08-29")]
    start_date: String,
    #[arg(long, default_value_t = 7)]
    n_days: u32,
    #[arg(long, default_value = "a")]
    probe: String,
    #[arg(
        long,
        default_value = "data/external/crossing_lists/themis_mp_crossings_v2.txt"
    )]
    staples_catalog: PathBuf,
    #[arg(long, default_value_t = 10)]
    pad_minutes: i64,
    #[arg(long, default_value_t = 0.0)]
    min_fom: f64,
    #[arg(long, default_value_t = 32)]
    embedding_dim: usize,
    #[arg(long, default_value_t = 1)]
    takens_lag: usize,
    #[arg(long, default_value_t = 10)]
    crossing_window_minutes: usize,
    #[arg(long, default_value_t = 100.0)]
    max_bmag: f64,
    #[arg(long, default_value_t = 0.5)]
    bmag_noise_floor: f64,
    #[arg(long, default_value = "data/external")]
    data_dir: PathBuf,
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/window_sensitivity_eval.json"
    )]
    out_json: PathBuf,
}

#[derive(Debug, Serialize)]
struct WindowResult {
    window_fn: String,
    n_detections: usize,
    precision: f64,
    recall: f64,
    f1: f64,
    bootstrap_ci_mean: f64,
    bootstrap_ci_lo: f64,
    bootstrap_ci_hi: f64,
}

#[derive(Debug, Serialize)]
struct WindowSensitivityResults {
    start_date: String,
    n_days: u32,
    probe: String,
    embedding_dim: usize,
    n_catalog_events: usize,
    n_fgm_minutes: usize,
    windows: Vec<WindowResult>,
}

fn hours_to_unix(origin: &NaiveDateTime, h: f64) -> i64 {
    let base = Utc.from_utc_datetime(origin).timestamp();
    base + (h * 3600.0) as i64
}

fn event_midpoint_unix(ev: &MmsEventInterval) -> i64 {
    let mid = ev.start + (ev.end - ev.start) / 2;
    Utc.from_utc_datetime(&mid).timestamp()
}

/// Build delay vectors with the specified window function applied per-sample.
fn build_windowed_delay_vectors(
    all_minutes: &[ThemisFgmMinuteRecord],
    embedding_dim: usize,
    takens_lag: usize,
    max_window_span_hours: f64,
    bmag_noise_floor: f64,
    window_coeffs: &[f64],
) -> (Vec<Vec<f64>>, Vec<usize>) {
    let channels: usize = 4;
    let steps = embedding_dim / channels;
    let window_rows = (steps - 1) * takens_lag + 1;
    let mut delay_vectors: Vec<Vec<f64>> = Vec::new();
    let mut embed_meta: Vec<usize> = Vec::new();

    for w_start in 0..=(all_minutes.len().saturating_sub(window_rows)) {
        let sample_indices: Vec<usize> = (0..steps).map(|s| w_start + s * takens_lag).collect();
        let first_h = all_minutes[*sample_indices.first().unwrap()].elapsed_hours;
        let last_h = all_minutes[*sample_indices.last().unwrap()].elapsed_hours;
        if last_h - first_h > max_window_span_hours {
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
        let denom = local_mean_b.max(bmag_noise_floor);
        let mut v = vec![0.0_f64; embedding_dim];
        for (s, &ri) in sample_indices.iter().enumerate() {
            let rec = &all_minutes[ri];
            let w = window_coeffs[s];
            v[s * channels] = w * rec.bx_gse / denom;
            v[s * channels + 1] = w * rec.by_gse / denom;
            v[s * channels + 2] = w * rec.bz_gse / denom;
            v[s * channels + 3] = w * (rec.b_magnitude - local_mean_b) / denom;
        }
        embed_meta.push(*sample_indices.last().unwrap());
        delay_vectors.push(v);
    }
    (delay_vectors, embed_meta)
}

/// Window-sensitivity evaluation context: input data (THEMIS minutes,
/// MMS event intervals), reference midnight anchor, CLI configuration,
/// the window function being tested, and the time-series bounds.
struct WindowVariantInputs<'a> {
    all_minutes: &'a [ThemisFgmMinuteRecord],
    fom_catalog: &'a [MmsEventInterval],
    reference_midnight: &'a NaiveDateTime,
    cli: &'a Cli,
    window_fn: WindowFn,
    eval_window_secs: i64,
    series_start_unix: i64,
    series_end_unix: i64,
}

fn run_window_variant(inputs: WindowVariantInputs<'_>) -> WindowResult {
    let all_minutes = inputs.all_minutes;
    let fom_catalog = inputs.fom_catalog;
    let reference_midnight = inputs.reference_midnight;
    let cli = inputs.cli;
    let window_fn = inputs.window_fn;
    let eval_window_secs = inputs.eval_window_secs;
    let series_start_unix = inputs.series_start_unix;
    let series_end_unix = inputs.series_end_unix;
    let channels: usize = 4;
    let steps = cli.embedding_dim / channels;
    let expected_span_hours = (steps - 1) as f64 * cli.takens_lag as f64 / 60.0;
    let max_window_span_hours = expected_span_hours + 2.0 * cli.takens_lag as f64 / 60.0;
    let window_coeffs = window_fn.coefficients(steps);

    let (delay_vectors, embed_meta) = build_windowed_delay_vectors(
        all_minutes,
        cli.embedding_dim,
        cli.takens_lag,
        max_window_span_hours,
        cli.bmag_noise_floor,
        &window_coeffs,
    );

    let associators =
        cd_kernel::batch_sliding_associator_norms_parallel(&delay_vectors, cli.embedding_dim);
    let assoc_meta: Vec<usize> = embed_meta[2..].to_vec();

    let trans_window = cli.crossing_window_minutes.max(5);
    let mut fire_hours: Vec<f64> = Vec::new();

    if associators.len() > trans_window * 2 {
        let global_mean: f64 = associators.iter().sum::<f64>() / associators.len() as f64;
        let global_std: f64 = (associators
            .iter()
            .map(|&a| (a - global_mean).powi(2))
            .sum::<f64>()
            / associators.len() as f64)
            .sqrt();
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
                    fire_hours.push(all_minutes[assoc_meta[i]].elapsed_hours);
                    last_trans_idx = Some(i);
                }
            }
        }
    }

    let detection_unix: Vec<i64> = fire_hours
        .iter()
        .map(|&h| hours_to_unix(reference_midnight, h))
        .collect();
    let event_unix: Vec<i64> = fom_catalog.iter().map(event_midpoint_unix).collect();

    let (precision, recall, f1) =
        boundary_metrics::precision_recall_f1(&detection_unix, &event_unix, eval_window_secs);

    let (bci_mean, bci_lo, bci_hi) = boundary_metrics::bootstrap_f1_ci_seeded(
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

    println!(
        "  {:8}: detections={:3}  F1={:.3}  P={:.3}  R={:.3}  CI=[{:.3},{:.3}]",
        window_fn.name(),
        fire_hours.len(),
        f1,
        precision,
        recall,
        bci_lo,
        bci_hi
    );

    WindowResult {
        window_fn: window_fn.name().to_string(),
        n_detections: fire_hours.len(),
        precision,
        recall,
        f1,
        bootstrap_ci_mean: bci_mean,
        bootstrap_ci_lo: bci_lo,
        bootstrap_ci_hi: bci_hi,
    }
}

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
    let spacecraft = format!("TH{probe_upper}");

    println!(
        "=== Window Function Sensitivity (CD Ablation, M-5) ===\n\
         Window: {} to {} ({} days), probe: {}, dim={}\n",
        start, end, cli.n_days, spacecraft, cli.embedding_dim
    );

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

    let catalog_content = fs::read_to_string(&cli.staples_catalog)
        .with_context(|| format!("reading catalog: {}", cli.staples_catalog.display()))?;
    let catalog =
        parse_staples_crossing_catalog(&catalog_content, probe_char, start, end, cli.pad_minutes);
    let fom_catalog: Vec<MmsEventInterval> = catalog
        .into_iter()
        .filter(|e| e.fom >= cli.min_fom)
        .collect();

    println!(
        "  Catalog: {} intervals, FGM: {} minutes",
        fom_catalog.len(),
        all_minutes.len()
    );

    let series_start_unix = hours_to_unix(&reference_midnight, 0.0);
    let series_end_unix = hours_to_unix(
        &reference_midnight,
        all_minutes.last().map(|r| r.elapsed_hours).unwrap_or(0.0),
    );
    let eval_window_secs = cli.pad_minutes * 60;

    println!("\nRunning window function variants:");
    let mut windows: Vec<WindowResult> = Vec::new();
    for wfn in [WindowFn::Boxcar, WindowFn::Hamming, WindowFn::Hann] {
        windows.push(run_window_variant(WindowVariantInputs {
            all_minutes: &all_minutes,
            fom_catalog: &fom_catalog,
            reference_midnight: &reference_midnight,
            cli: &cli,
            window_fn: wfn,
            eval_window_secs,
            series_start_unix,
            series_end_unix,
        }));
    }

    let boxcar_f1 = windows
        .iter()
        .find(|w| w.window_fn == "boxcar")
        .map(|w| w.f1)
        .unwrap_or(0.0);
    let max_delta = windows
        .iter()
        .map(|w| (w.f1 - boxcar_f1).abs())
        .fold(0.0_f64, f64::max);
    println!("\n  Max F1 delta from boxcar: {:.3}", max_delta);

    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }

    let results = WindowSensitivityResults {
        start_date: cli.start_date.clone(),
        n_days: cli.n_days,
        probe: spacecraft,
        embedding_dim: cli.embedding_dim,
        n_catalog_events: fom_catalog.len(),
        n_fgm_minutes: all_minutes.len(),
        windows,
    };

    let json = serde_json::to_string_pretty(&results)?;
    fs::write(&cli.out_json, &json)?;
    println!("Results written to {}", cli.out_json.display());

    Ok(())
}

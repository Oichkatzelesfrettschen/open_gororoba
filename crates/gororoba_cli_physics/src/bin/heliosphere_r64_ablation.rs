//! R64 dimensional ablation: 8-channel Takens embedding with half-lag offsets.
//!
//! WHY: Pre-registered Axis A ablation (ablation-preregistered-v1, commit 455d4745).
//! Prediction P5: CD R64 F1 will be WITHIN 0.05 of CD R32 F1 (not significantly better).
//! Rationale: At 1-minute cadence, doubling temporal resolution via half-lag offsets
//! adds noise without proportional signal gain.
//!
//! WHAT: 8-channel embedding = (Bx,By,Bz,|B|) at the full-lag positions plus
//! (Bx,By,Bz,|B|) at half-lag offsets (i.e. lag/2 interleaved samples).
//! 8 channels x 8 lags = 64D.  This is the "doubled temporal resolution" variant.
//!
//! HOW:
//!   cargo run --release -p gororoba_cli_physics \
//!     --bin heliosphere-r64-ablation -- \
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

#[derive(Parser, Debug)]
#[command(
    name = "heliosphere-r64-ablation",
    about = "CD R64 (8-channel half-lag x 8 lags) ablation -- Axis A dim variant"
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
    /// Full lag in minutes (half-lags are inserted between each full-lag step).
    #[arg(long, default_value_t = 2)]
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
        default_value = "data/output/heliosphere/ablations/r64_ablation_eval.json"
    )]
    out_json: PathBuf,
}

#[derive(Debug, Serialize)]
struct AblationResult {
    start_date: String,
    n_days: u32,
    probe: String,
    embedding_dim: usize,
    n_channels: usize,
    n_lags: usize,
    takens_lag_minutes: usize,
    method: String,
    n_catalog_events: usize,
    n_fgm_minutes: usize,
    n_detections: usize,
    precision: f64,
    recall: f64,
    f1: f64,
    bootstrap_ci_mean: f64,
    bootstrap_ci_lo: f64,
    bootstrap_ci_hi: f64,
    series_start_unix: i64,
    series_end_unix: i64,
}

fn hours_to_unix(origin: &NaiveDateTime, h: f64) -> i64 {
    let base = Utc.from_utc_datetime(origin).timestamp();
    base + (h * 3600.0) as i64
}

fn event_midpoint_unix(ev: &MmsEventInterval) -> i64 {
    let mid = ev.start + (ev.end - ev.start) / 2;
    Utc.from_utc_datetime(&mid).timestamp()
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

    // R64: 8 channels (4 full-lag + 4 half-lag interleaved) x 8 lags = 64D
    // The 8 lags are positioned at: 0, half, full, 1.5*full, 2*full, ...
    // Concretely with takens_lag=2: positions at minutes 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16
    // We use: lag_positions = 0, 1, 2, 3, 4, 5, 6, 7 with step = takens_lag/2 = 1 minute
    // This gives 8 samples at 1-minute intervals = same 8-minute window as R32 but at 1-min step.
    // With takens_lag=2 (full) -> half_lag=1.  8 positions at half-lag spacing.
    let half_lag = (cli.takens_lag / 2).max(1);
    let n_lags: usize = 8;
    let channels: usize = 8; // Bx_full, By_full, Bz_full, |B|_full, Bx_half, By_half, Bz_half, |B|_half
    let embedding_dim = channels * n_lags; // 64

    // Sample positions: for lag k, we take a full-lag sample at k*full_lag and a half-lag
    // sample at k*full_lag + half_lag.  This means the window spans (n_lags-1)*full_lag+half_lag.
    // For full_lag=2, half_lag=1: positions are [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] minutes.
    // 8 full-lag anchors at [0,2,4,6,8,10,12,14], 8 half-lag offsets at [1,3,5,7,9,11,13,15].
    // Window span: 15 minutes.  Comparable to the 8-min R32 window -- slightly longer.

    println!(
        "=== CD R64 Ablation ({}D, {}ch x {} lags, half_lag={}min) ===\n\
         Window: {} to {} ({} days), probe: {}\n",
        embedding_dim, channels, n_lags, half_lag, start, end, cli.n_days, spacecraft
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

    // -----------------------------------------------------------------------
    // Build 8-channel delay vectors: full-lag + half-lag interleaved
    // Sample layout for lag k (k=0..7):
    //   channels 0-3: (Bx,By,Bz,|B|) at minute position k * full_lag
    //   channels 4-7: (Bx,By,Bz,|B|) at minute position k * full_lag + half_lag
    // Window spans from minute 0 to minute (n_lags-1)*full_lag + half_lag.
    // -----------------------------------------------------------------------
    let full_lag = cli.takens_lag;
    let window_rows = (n_lags - 1) * full_lag + half_lag + 1;
    if all_minutes.len() < window_rows + 2 {
        anyhow::bail!(
            "Not enough data: need {} rows, have {}",
            window_rows + 2,
            all_minutes.len()
        );
    }

    let expected_span_hours = ((n_lags - 1) * full_lag + half_lag) as f64 / 60.0;
    let max_window_span_hours = expected_span_hours + 2.0 * full_lag as f64 / 60.0;

    let mut embedded_vectors: Vec<Vec<f64>> = Vec::new();
    let mut embed_meta: Vec<usize> = Vec::new();

    for w_start in 0..=(all_minutes.len() - window_rows) {
        // Full-lag positions: w_start + k*full_lag for k=0..7
        // Half-lag positions: w_start + k*full_lag + half_lag for k=0..7
        let full_positions: Vec<usize> = (0..n_lags).map(|k| w_start + k * full_lag).collect();
        let half_positions: Vec<usize> = (0..n_lags)
            .map(|k| w_start + k * full_lag + half_lag)
            .collect();

        let last_pos = half_positions[n_lags - 1];
        if last_pos >= all_minutes.len() {
            continue;
        }

        let first_h = all_minutes[w_start].elapsed_hours;
        let last_h = all_minutes[last_pos].elapsed_hours;
        if last_h - first_h > max_window_span_hours {
            continue;
        }

        // Compute mean |B| over all 16 sample positions for normalization
        let all_pos: Vec<usize> = full_positions
            .iter()
            .chain(half_positions.iter())
            .copied()
            .collect();
        let sum_b: f64 = all_pos.iter().map(|&i| all_minutes[i].b_magnitude).sum();
        let local_mean_b = sum_b / all_pos.len() as f64;
        if local_mean_b <= 0.0 || !local_mean_b.is_finite() {
            continue;
        }
        let denom = local_mean_b.max(cli.bmag_noise_floor);

        let mut v = vec![0.0_f64; embedding_dim];
        for (k, (&fp, &hp)) in full_positions.iter().zip(half_positions.iter()).enumerate() {
            let rf = &all_minutes[fp];
            let rh = &all_minutes[hp];
            // Channels 0-3: full-lag sample
            v[k * channels] = rf.bx_gse / denom;
            v[k * channels + 1] = rf.by_gse / denom;
            v[k * channels + 2] = rf.bz_gse / denom;
            v[k * channels + 3] = (rf.b_magnitude - local_mean_b) / denom;
            // Channels 4-7: half-lag interleaved sample
            v[k * channels + 4] = rh.bx_gse / denom;
            v[k * channels + 5] = rh.by_gse / denom;
            v[k * channels + 6] = rh.bz_gse / denom;
            v[k * channels + 7] = (rh.b_magnitude - local_mean_b) / denom;
        }
        embed_meta.push(last_pos);
        embedded_vectors.push(v);
    }

    println!("  Embedded vectors: {}", embedded_vectors.len());

    // CD associator at dim=64 (supported by batch_sliding_associator_norms_parallel)
    let associators =
        cd_kernel::batch_sliding_associator_norms_parallel(&embedded_vectors, embedding_dim);

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
                    let mi = assoc_minute_indices[i];
                    cd_hours.push(all_minutes[mi].elapsed_hours);
                    last_trans_idx = Some(i);
                }
            }
        }
    }

    println!("  CD R64 detections: {}", cd_hours.len());

    let eval_window_secs = cli.pad_minutes * 60;
    let detection_unix: Vec<i64> = cd_hours
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
        "  F1={:.3}  P={:.3}  R={:.3}  CI=[{:.3},{:.3}]",
        f1, precision, recall, bci_lo, bci_hi
    );

    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }

    let results = AblationResult {
        start_date: cli.start_date.clone(),
        n_days: cli.n_days,
        probe: spacecraft,
        embedding_dim,
        n_channels: channels,
        n_lags,
        takens_lag_minutes: full_lag,
        method: format!(
            "CD R64 (8ch: 4-full+4-half x {} lags, half_lag={}min)",
            n_lags, half_lag
        ),
        n_catalog_events: fom_catalog.len(),
        n_fgm_minutes: all_minutes.len(),
        n_detections: cd_hours.len(),
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

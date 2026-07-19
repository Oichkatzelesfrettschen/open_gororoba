//! PCA variance baseline for the CD associator dimensional ablation.
//!
//! WHY: Pre-registration (ablation-preregistered-v1) requires a PCA baseline to
//! test whether simple linear dimensionality reduction captures boundary crossings
//! as well as the CD associator.  At a magnetopause crossing the field sweeps
//! through a large angle; the fraction of variance in the first principal component
//! of a rolling window peaks during this sweep.  If PCA matches CD performance,
//! the nonlinear algebraic structure adds nothing beyond variance concentration.
//!
//! WHAT: For each position t, compute the covariance of the delay vectors in a
//! rolling window of width `pca_window` (default 15 minutes).  Score at t is the
//! fraction of total variance explained by the first eigenvalue (leading PC ratio).
//! Apply the same MAD_SCALE_FACTOR=1.5 threshold.  Evaluate F1 against Staples+2020.
//!
//! HOW:
//!   cargo run --release -p gororoba_cli_physics \
//!     --bin heliosphere-pca-variance-baseline -- \
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
    name = "heliosphere-pca-variance-baseline",
    about = "PCA leading-PC variance ratio baseline for CD associator ablation (P6A.S2.T2.8)"
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
    /// Rolling window width (in delay vectors) for PCA covariance estimation.
    #[arg(long, default_value_t = 15)]
    pca_window: usize,
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
        default_value = "data/output/heliosphere/ablations/pca_variance_baseline_eval.json"
    )]
    out_json: PathBuf,
}

#[derive(Debug, Serialize)]
struct PcaVarianceResults {
    start_date: String,
    n_days: u32,
    probe: String,
    embedding_dim: usize,
    pca_window: usize,
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

/// Compute the fraction of total variance in the leading principal component
/// of the rows of `window` (each row is a delay vector of length `dim`).
///
/// Uses the power iteration method to find the top eigenvector of the
/// covariance matrix without forming the full dim x dim matrix explicitly.
/// Returns variance_pc1 / variance_total in `[0, 1]`.
fn pca_leading_variance_ratio(window: &[Vec<f64>], dim: usize, n_iter: usize) -> f64 {
    if window.len() < 2 {
        return 0.0;
    }
    let n = window.len();

    // Compute column means.
    let mut mean = vec![0.0_f64; dim];
    for v in window.iter() {
        for (j, &val) in v.iter().enumerate() {
            mean[j] += val;
        }
    }
    for m in mean.iter_mut() {
        *m /= n as f64;
    }

    // Total variance = sum of squared deviations.
    let total_var: f64 = window
        .iter()
        .flat_map(|v| {
            v.iter()
                .enumerate()
                .map(|(j, &val)| (val - mean[j]).powi(2))
        })
        .sum::<f64>();

    if total_var < 1e-30 {
        return 1.0;
    }

    // Power iteration on covariance matrix (no explicit formation).
    // x_{k+1} = C * x_k / ||C * x_k||  where  C*x = (1/n) sum_i (v_i - mu)(v_i - mu)^T x
    let mut pc = vec![1.0_f64; dim];
    for _ in 0..n_iter {
        let mut next = vec![0.0_f64; dim];
        for v in window.iter() {
            // dot_product = (v - mu)^T * pc
            let dot: f64 = v
                .iter()
                .enumerate()
                .map(|(j, &val)| (val - mean[j]) * pc[j])
                .sum();
            // next += dot * (v - mu)
            for (j, val) in next.iter_mut().enumerate() {
                *val += dot * (v[j] - mean[j]);
            }
        }
        let norm: f64 = next.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm < 1e-30 {
            break;
        }
        for val in next.iter_mut() {
            *val /= norm;
        }
        pc = next;
    }

    // Variance explained by leading PC = (1/n) sum_i ((v_i - mu)^T pc)^2
    let var_pc1: f64 = window
        .iter()
        .map(|v| {
            let dot: f64 = v
                .iter()
                .enumerate()
                .map(|(j, &val)| (val - mean[j]) * pc[j])
                .sum();
            dot * dot
        })
        .sum::<f64>()
        / n as f64;

    (var_pc1 / (total_var / n as f64)).min(1.0)
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
        "=== PCA Variance Baseline (CD Ablation) ===\n\
         Window: {} to {} ({} days), probe: {}, dim={}, pca_window={}\n",
        start, end, cli.n_days, spacecraft, cli.embedding_dim, cli.pca_window
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
    let max_window_span_hours = expected_span_hours + 2.0 * cli.takens_lag as f64 / 60.0;

    let mut delay_vectors: Vec<Vec<f64>> = Vec::new();
    let mut embed_meta: Vec<usize> = Vec::new();

    for w_start in 0..=(all_minutes.len() - window_rows) {
        let sample_indices: Vec<usize> = (0..steps).map(|s| w_start + s * cli.takens_lag).collect();
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

    // PCA score: leading PC variance ratio over a rolling window of pca_window vectors.
    // Score at position t = variance_pc1 / variance_total for vectors [t-pca_window+1, t].
    let pca_window = cli.pca_window.max(3);
    let n_iter = 20; // power iterations; adequate for 32D
    let pca_scores: Vec<f64> = (0..delay_vectors.len())
        .map(|t| {
            if t + 1 < pca_window {
                return 0.0;
            }
            let start_idx = t + 1 - pca_window;
            let window = &delay_vectors[start_idx..=t];
            pca_leading_variance_ratio(window, cli.embedding_dim, n_iter)
        })
        .collect();

    let trans_window = cli.crossing_window_minutes.max(5);
    let mut fire_hours: Vec<f64> = Vec::new();

    if pca_scores.len() > trans_window * 2 {
        let global_mean: f64 = pca_scores.iter().sum::<f64>() / pca_scores.len() as f64;
        let global_std: f64 = {
            let var = pca_scores
                .iter()
                .map(|&a| (a - global_mean).powi(2))
                .sum::<f64>()
                / pca_scores.len() as f64;
            var.sqrt()
        };
        let threshold = global_std * MAD_SCALE_FACTOR;
        let half = trans_window;
        let mut last_trans_idx: Option<usize> = None;
        for i in half..pca_scores.len().saturating_sub(half) {
            let pre_mean: f64 =
                pca_scores[i.saturating_sub(half)..i].iter().sum::<f64>() / half as f64;
            let post_mean: f64 = pca_scores[i..(i + half).min(pca_scores.len())]
                .iter()
                .sum::<f64>()
                / half.min(pca_scores.len() - i) as f64;
            let jump = (post_mean - pre_mean).abs();
            if jump > threshold {
                let dominated =
                    last_trans_idx.is_some_and(|prev| i.saturating_sub(prev) < trans_window);
                if !dominated {
                    fire_hours.push(all_minutes[embed_meta[i]].elapsed_hours);
                    last_trans_idx = Some(i);
                }
            }
        }
    }

    let eval_window_secs = cli.pad_minutes * 60;
    let detection_unix: Vec<i64> = fire_hours
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
        "  PCA detections: {}  F1={:.3}  P={:.3}  R={:.3}  CI=[{:.3},{:.3}]",
        fire_hours.len(),
        f1,
        precision,
        recall,
        bci_lo,
        bci_hi
    );

    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }

    let results = PcaVarianceResults {
        start_date: cli.start_date.clone(),
        n_days: cli.n_days,
        probe: spacecraft,
        embedding_dim: cli.embedding_dim,
        pca_window,
        method: format!(
            "PCA leading-PC variance ratio (dim={}, pca_window={}, power_iter={}, MAD_SCALE_FACTOR=1.5)",
            cli.embedding_dim, pca_window, n_iter
        ),
        n_catalog_events: fom_catalog.len(),
        n_fgm_minutes: all_minutes.len(),
        n_detections: fire_hours.len(),
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
    println!("Results written to {}", cli.out_json.display());

    Ok(())
}

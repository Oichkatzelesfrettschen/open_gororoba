//! Commutator baseline for the CD associator dimensional ablation.
//!
//! WHY: Pre-registration (ablation-preregistered-v1) requires a commutator baseline
//! to test whether the three-argument structure of the CD associator `[x,y,z]` adds
//! information beyond the two-argument commutator `[x,y] = x*y - y*x`.
//! The commutator measures non-commutativity directly; for octonions/sedenions it
//! is pure-imaginary and its norm equals the cross-product-like "rotation" between
//! two embedding vectors.  If the commutator matches CD performance, the
//! associator's dependence on the third argument adds nothing.
//!
//! WHAT: Same THEMIS window and Takens delay embedding as CD baseline.
//! Score: ||v_t * v_{t-1} - v_{t-1} * v_t||_2 using CD multiplication at dim=32.
//! Same MAD_SCALE_FACTOR=1.5 threshold and Staples+2020 catalog evaluation.
//!
//! HOW:
//!   cargo run --release -p gororoba_cli_physics \
//!     --bin heliosphere-commutator-baseline -- \
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
    name = "heliosphere-commutator-baseline",
    about = "Commutator ||xy - yx||_2 baseline for CD associator ablation (P6A.S2.T2.7)"
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
        default_value = "data/output/heliosphere/ablations/commutator_baseline_eval.json"
    )]
    out_json: PathBuf,
}

#[derive(Debug, Serialize)]
struct CommutatorResults {
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

    println!(
        "=== Commutator Baseline (CD Ablation) ===\n\
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

    // Commutator score: ||x*y - y*x||_2 for consecutive delay vector pairs.
    let commutator_scores: Vec<f64> = delay_vectors
        .windows(2)
        .map(|w| {
            let x = &w[0];
            let y = &w[1];
            let xy = cd_kernel::cd_multiply(x, y);
            let yx = cd_kernel::cd_multiply(y, x);
            xy.iter()
                .zip(yx.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .collect();

    let score_meta: Vec<usize> = embed_meta[1..].to_vec();

    let trans_window = cli.crossing_window_minutes.max(5);
    let mut fire_hours: Vec<f64> = Vec::new();

    if commutator_scores.len() > trans_window * 2 {
        let global_mean: f64 =
            commutator_scores.iter().sum::<f64>() / commutator_scores.len() as f64;
        let global_std: f64 = {
            let var = commutator_scores
                .iter()
                .map(|&a| (a - global_mean).powi(2))
                .sum::<f64>()
                / commutator_scores.len() as f64;
            var.sqrt()
        };
        let threshold = global_std * MAD_SCALE_FACTOR;
        let half = trans_window;
        let mut last_trans_idx: Option<usize> = None;
        for i in half..commutator_scores.len().saturating_sub(half) {
            let pre_mean: f64 = commutator_scores[i.saturating_sub(half)..i]
                .iter()
                .sum::<f64>()
                / half as f64;
            let post_mean: f64 = commutator_scores[i..(i + half).min(commutator_scores.len())]
                .iter()
                .sum::<f64>()
                / half.min(commutator_scores.len() - i) as f64;
            let jump = (post_mean - pre_mean).abs();
            if jump > threshold {
                let dominated =
                    last_trans_idx.is_some_and(|prev| i.saturating_sub(prev) < trans_window);
                if !dominated {
                    fire_hours.push(all_minutes[score_meta[i]].elapsed_hours);
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
        "  Commutator detections: {}  F1={:.3}  P={:.3}  R={:.3}  CI=[{:.3},{:.3}]",
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

    let results = CommutatorResults {
        start_date: cli.start_date.clone(),
        n_days: cli.n_days,
        probe: spacecraft,
        embedding_dim: cli.embedding_dim,
        method: "commutator ||xy - yx||_2 (CD dim=32, MAD_SCALE_FACTOR=1.5)".to_string(),
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

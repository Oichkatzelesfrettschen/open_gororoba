//! Sparsity-matched random trilinear baseline for the CD associator ablation.
//!
//! WHY: Pre-registration (ablation-preregistered-v1) specifies this as a second
//! random baseline that preserves the CD associator's zero-pattern in T_ijk but
//! randomizes the non-zero entries from N(0, sigma_cd).
//!
//! If CD beats dense-random (P2 satisfied) BUT falls within sparsity-matched-random:
//!   --> The zero-pattern of CD is what matters, not the specific coefficient values.
//! If CD beats sparsity-matched-random by >= 2 sigma:
//!   --> The specific CD coefficient values contribute beyond the zero-pattern.
//!
//! WHAT: Extract the CD T_ijk sparsity pattern by evaluating the associator on
//! standard basis triples.  T_ijk = e_i^T * [e_j, e_0, e_k] where e_j is the
//! j-th standard basis vector.  Entry (i,j,k) is nonzero iff T_ijk != 0.
//! Compute sigma_cd = RMS of the non-zero CD T entries.
//! For each draw: randomize only non-zero entries from N(0, sigma_cd).
//! Run the same change-point detection as the CD baseline.
//!
//! HOW:
//!   cargo run --release -p gororoba_cli_physics \
//!     --bin heliosphere-random-trilinear-sparse -- \
//!     --start-date 2016-08-29 --n-days 7 --n-draws 100

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
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use serde::Serialize;
use spectral_core::boundary_metrics;
use std::{fs, path::PathBuf};

const MAD_SCALE_FACTOR: f64 = 1.5;

#[derive(Parser, Debug)]
#[command(
    name = "heliosphere-random-trilinear-sparse",
    about = "Sparsity-matched random trilinear baseline (same zero-pattern as CD)"
)]
struct Cli {
    #[arg(long, default_value = "2016-08-29")]
    start_date: String,
    #[arg(long, default_value_t = 7)]
    n_days: u32,
    #[arg(long, default_value = "a")]
    probe: String,
    #[arg(long, default_value = "data/external/crossing_lists/themis_mp_crossings_v2.txt")]
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
    #[arg(long, default_value_t = 100)]
    n_draws: usize,
    #[arg(long, default_value_t = 2000)]
    base_seed: u64,
    #[arg(long, default_value = "data/external")]
    data_dir: PathBuf,
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/random_trilinear_sparse_eval.json"
    )]
    out_json: PathBuf,
}

#[derive(Debug, Serialize)]
struct DrawResult {
    draw_index: usize,
    seed: u64,
    n_detections: usize,
    precision: f64,
    recall: f64,
    f1: f64,
}

#[derive(Debug, Serialize)]
struct SparseRandomResults {
    start_date: String,
    n_days: u32,
    probe: String,
    embedding_dim: usize,
    method: String,
    n_catalog_events: usize,
    n_fgm_minutes: usize,
    n_draws: usize,
    base_seed: u64,
    n_nonzero_entries: usize,
    sigma_cd: f64,
    f1_mean: f64,
    f1_std: f64,
    f1_min: f64,
    f1_max: f64,
    f1_mean_plus_2sigma: f64,
    draws: Vec<DrawResult>,
}

fn hours_to_unix(origin: &NaiveDateTime, h: f64) -> i64 {
    let base = Utc.from_utc_datetime(origin).timestamp();
    base + (h * 3600.0) as i64
}

fn event_midpoint_unix(ev: &MmsEventInterval) -> i64 {
    let mid = ev.start + (ev.end - ev.start) / 2;
    Utc.from_utc_datetime(&mid).timestamp()
}

/// Compute the CD trilinear structure tensor for the given dim by probing all basis triples.
///
/// The CD associator [x,y,z]_l = ((xy)z - x(yz))_l is a trilinear form. The structure
/// tensor entry S_{l,i,j,k} = [e_i, e_j, e_k]_l captures the l-th output component when
/// inputs are basis vectors e_i, e_j, e_k.
///
/// Returns: (nonzero_quads, sigma_cd) where nonzero_quads is Vec<(l,i,j,k)> of all index
/// quadruples where S is nonzero, and sigma_cd is the RMS of the nonzero S values.
///
/// For dim=32, there are 32^3=32768 triples to probe; many are nonzero due to zero
/// divisors present in 32D CD algebras. Using e_0 (identity) as any argument always
/// gives zero and must be excluded from meaningful sparsity analysis.
fn build_cd_sparsity_pattern(dim: usize) -> (Vec<(usize, usize, usize, usize)>, f64) {
    let mut nonzero_quads: Vec<(usize, usize, usize, usize)> = Vec::new();
    let mut values_sq_sum = 0.0_f64;
    let mut n_values = 0usize;

    for i in 0..dim {
        let mut ei = vec![0.0_f64; dim];
        ei[i] = 1.0;
        for j in 0..dim {
            let mut ej = vec![0.0_f64; dim];
            ej[j] = 1.0;
            for k in 0..dim {
                let mut ek = vec![0.0_f64; dim];
                ek[k] = 1.0;
                // Quick norm check before computing the full vector
                let assoc_norm = cd_kernel::cd_associator_norm(&ei, &ej, &ek);
                if assoc_norm < 1e-12 { continue; }
                // Compute full associator vector [e_i, e_j, e_k]
                let ab = cd_kernel::cd_multiply(&ei, &ej);
                let abc_left = cd_kernel::cd_multiply(&ab, &ek);
                let bc = cd_kernel::cd_multiply(&ej, &ek);
                let abc_right = cd_kernel::cd_multiply(&ei, &bc);
                let assoc_vec: Vec<f64> = abc_left.iter().zip(&abc_right)
                    .map(|(l, r)| l - r).collect();
                for (l, &val) in assoc_vec.iter().enumerate().take(dim) {
                    if val.abs() > 1e-12 {
                        nonzero_quads.push((l, i, j, k));
                        values_sq_sum += val * val;
                        n_values += 1;
                    }
                }
            }
        }
    }

    let sigma_cd = if n_values > 0 {
        (values_sq_sum / n_values as f64).sqrt()
    } else {
        1.0
    };

    (nonzero_quads, sigma_cd)
}

/// Compute sparse random trilinear scores using the CD sparsity pattern.
///
/// Models the same 3-vector associator structure as `batch_sliding_associator_norms_parallel`:
/// for each consecutive triple (x, y, z), compute ||T[x,y,z]||_2 where T has the same
/// zero-pattern as the CD structure tensor S_{l,i,j,k}.
///
/// Returns n-2 scores for n input delay vectors (same length contract as CD baseline).
fn compute_sparse_random_scores(
    delay_vectors: &[Vec<f64>],
    dim: usize,
    nonzero_quads: &[(usize, usize, usize, usize)],
    sigma_cd: f64,
    seed: u64,
) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let dist = Normal::new(0.0, sigma_cd).expect("valid normal distribution");

    // Sample only the nonzero S entries
    let t_sparse: Vec<f64> = (0..nonzero_quads.len()).map(|_| dist.sample(&mut rng)).collect();

    delay_vectors
        .windows(3)
        .map(|w| {
            let x = &w[0];
            let y = &w[1];
            let z = &w[2];
            let mut result = vec![0.0_f64; dim];
            for (idx, &(l, i, j, k)) in nonzero_quads.iter().enumerate() {
                result[l] += t_sparse[idx] * x[i] * y[j] * z[k];
            }
            result.iter().map(|v| v * v).sum::<f64>().sqrt()
        })
        .collect()
}

fn detect_transitions(
    scores: &[f64],
    score_meta: &[usize],
    all_minutes: &[ThemisFgmMinuteRecord],
    trans_window: usize,
) -> Vec<f64> {
    let mut hours: Vec<f64> = Vec::new();
    if scores.len() <= trans_window * 2 { return hours; }

    let global_mean: f64 = scores.iter().sum::<f64>() / scores.len() as f64;
    let global_std: f64 = {
        let var = scores.iter().map(|&a| (a - global_mean).powi(2)).sum::<f64>()
            / scores.len() as f64;
        var.sqrt()
    };
    let threshold = global_std * MAD_SCALE_FACTOR;
    let half = trans_window;
    let mut last_trans_idx: Option<usize> = None;
    for i in half..scores.len().saturating_sub(half) {
        let pre_mean: f64 = scores[i.saturating_sub(half)..i].iter().sum::<f64>() / half as f64;
        let post_mean: f64 = scores[i..(i + half).min(scores.len())].iter().sum::<f64>()
            / half.min(scores.len() - i) as f64;
        let jump = (post_mean - pre_mean).abs();
        if jump > threshold {
            let dominated = last_trans_idx.is_some_and(
                |prev| i.saturating_sub(prev) < trans_window);
            if !dominated {
                hours.push(all_minutes[score_meta[i]].elapsed_hours);
                last_trans_idx = Some(i);
            }
        }
    }
    hours
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let start = NaiveDate::parse_from_str(&cli.start_date, "%Y-%m-%d")
        .with_context(|| format!("invalid start_date: {}", cli.start_date))?;
    let end = start + chrono::Duration::days(cli.n_days as i64 - 1);
    let probe_char = cli.probe.trim().to_lowercase().chars().next()
        .context("--probe must be a single letter (a-e)")?;
    let probe_upper = cli.probe.trim().to_uppercase();
    let spacecraft = format!("TH{probe_upper}");

    println!(
        "=== Sparsity-Matched Random Trilinear ({} draws) ===\n\
         Window: {} to {} ({} days), probe: {}, dim={}\n",
        cli.n_draws, start, end, cli.n_days, spacecraft, cli.embedding_dim
    );

    // Build CD sparsity pattern FIRST (deterministic, only depends on dim)
    println!("Computing CD T_ijk sparsity pattern (dim={})...", cli.embedding_dim);
    let (nonzero_quads, sigma_cd) = build_cd_sparsity_pattern(cli.embedding_dim);
    // 4-index tensor: dim^4 total entries (l,i,j,k each in [0,dim))
    let total_entries = cli.embedding_dim.pow(4);
    println!(
        "  Nonzero (l,i,j,k) quads: {}/{} ({:.1}% sparse)",
        nonzero_quads.len(),
        total_entries,
        100.0 * (1.0 - nonzero_quads.len() as f64 / total_entries as f64)
    );
    println!("  sigma_cd (RMS of nonzero): {:.6}", sigma_cd);

    let themis_dir = cli.data_dir.join("themis");
    fs::create_dir_all(&themis_dir)?;

    let gse_param = format!("{}_fgs_gse", spacecraft.to_lowercase());
    let dataset = format!("{}_L2_FGM@0", spacecraft);

    for day_offset in 0..cli.n_days {
        let date = start + chrono::Duration::days(day_offset as i64);
        let doy = date.ordinal();
        let fname = format!("{}_fgm_{:04}_{:03}.csv",
            probe_upper.to_lowercase(), date.year(), doy);
        let output = themis_dir.join(&fname);
        if output.exists() { println!("  DOY {doy}: cached"); continue; }
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
        let fname = format!("{}_fgm_{:04}_{:03}.csv",
            probe_upper.to_lowercase(), date.year(), date.ordinal());
        let path = themis_dir.join(&fname);
        if !path.exists() { eprintln!("  Warning: {} not found", path.display()); continue; }
        let content = fs::read_to_string(&path)?;
        let mut records = parse_themis_fgm_hapi_csv_minutes(&content, &spacecraft);
        all_minutes.append(&mut records);
    }

    if all_minutes.is_empty() { anyhow::bail!("No THEMIS FGM data found."); }

    all_minutes.retain(|r| r.b_magnitude <= cli.max_bmag);
    all_minutes.sort_by(|a, b| a.year.cmp(&b.year).then(a.doy.cmp(&b.doy))
        .then(a.hour.cmp(&b.hour)).then(a.minute.cmp(&b.minute)));

    let (fy, fd, fh, fm) = (all_minutes[0].year, all_minutes[0].doy,
                             all_minutes[0].hour, all_minutes[0].minute);
    for rec in &mut all_minutes {
        let day_diff = (rec.year as f64 - fy as f64) * 365.25 + (rec.doy as f64 - fd as f64);
        rec.elapsed_hours = day_diff * 24.0 + (rec.hour as f64 - fh as f64)
            + (rec.minute as f64 - fm as f64) / 60.0;
    }

    let reference_midnight = NaiveDate::from_yo_opt(fy as i32, fd as u32)
        .and_then(|d| d.and_hms_opt(fh as u32, fm as u32, 0))
        .context("failed to build reference midnight")?;

    let catalog_content = fs::read_to_string(&cli.staples_catalog)
        .with_context(|| format!("reading catalog: {}", cli.staples_catalog.display()))?;
    let catalog = parse_staples_crossing_catalog(
        &catalog_content, probe_char, start, end, cli.pad_minutes);
    let fom_catalog: Vec<MmsEventInterval> = catalog.into_iter()
        .filter(|e| e.fom >= cli.min_fom).collect();

    println!("  Catalog: {} intervals, FGM: {} minutes",
             fom_catalog.len(), all_minutes.len());

    // Build delay vectors
    let channels: usize = 4;
    let steps = cli.embedding_dim / channels;
    let window_rows = (steps - 1) * cli.takens_lag + 1;
    if all_minutes.len() < window_rows + 1 {
        anyhow::bail!("Not enough data: need {} rows, have {}", window_rows + 1, all_minutes.len());
    }

    let expected_span_hours = (steps - 1) as f64 * cli.takens_lag as f64 / 60.0;
    let max_window_span_hours = expected_span_hours + 2.0 * cli.takens_lag as f64 / 60.0;

    let mut delay_vectors: Vec<Vec<f64>> = Vec::new();
    let mut embed_meta: Vec<usize> = Vec::new();

    for w_start in 0..=(all_minutes.len() - window_rows) {
        let sample_indices: Vec<usize> = (0..steps).map(|s| w_start + s * cli.takens_lag).collect();
        let first_h = all_minutes[*sample_indices.first().unwrap()].elapsed_hours;
        let last_h = all_minutes[*sample_indices.last().unwrap()].elapsed_hours;
        if last_h - first_h > max_window_span_hours { continue; }
        let sum_b: f64 = sample_indices.iter().map(|&i| all_minutes[i].b_magnitude).sum();
        let local_mean_b = sum_b / steps as f64;
        if local_mean_b <= 0.0 || !local_mean_b.is_finite() { continue; }
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

    // windows(3) produces n-2 scores; align metadata to the LAST index of each triple
    let score_meta: Vec<usize> = embed_meta[2..].to_vec();
    let trans_window = cli.crossing_window_minutes.max(5);
    let eval_window_secs = cli.pad_minutes * 60;

    let event_unix: Vec<i64> = fom_catalog.iter().map(event_midpoint_unix).collect();

    println!("Running {} sparsity-matched random draws...", cli.n_draws);

    let mut draws: Vec<DrawResult> = Vec::with_capacity(cli.n_draws);
    let mut f1_values: Vec<f64> = Vec::with_capacity(cli.n_draws);

    for draw_idx in 0..cli.n_draws {
        let seed = cli.base_seed + draw_idx as u64;
        if draw_idx % 10 == 0 { println!("  Draw {}/{}", draw_idx + 1, cli.n_draws); }

        let scores = compute_sparse_random_scores(
            &delay_vectors, cli.embedding_dim, &nonzero_quads, sigma_cd, seed);
        let fire_hours = detect_transitions(&scores, &score_meta, &all_minutes, trans_window);

        let detection_unix: Vec<i64> = fire_hours.iter()
            .map(|&h| hours_to_unix(&reference_midnight, h)).collect();

        let (precision, recall, f1) =
            boundary_metrics::precision_recall_f1(&detection_unix, &event_unix, eval_window_secs);

        f1_values.push(f1);
        draws.push(DrawResult { draw_index: draw_idx, seed, n_detections: fire_hours.len(),
                                 precision, recall, f1 });
    }

    let n = f1_values.len() as f64;
    let f1_mean = f1_values.iter().sum::<f64>() / n;
    let f1_std = (f1_values.iter().map(|&x| (x - f1_mean).powi(2)).sum::<f64>() / n).sqrt();
    let f1_min = f1_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let f1_max = f1_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let f1_mean_plus_2sigma = f1_mean + 2.0 * f1_std;

    println!(
        "\n  Sparsity-matched F1: mean={:.3} std={:.3} min={:.3} max={:.3}",
        f1_mean, f1_std, f1_min, f1_max
    );
    println!(
        "  mean+2sigma = {:.3}  (CD R32 must exceed this to prove specific coefficients matter)",
        f1_mean_plus_2sigma
    );

    if let Some(parent) = cli.out_json.parent() { fs::create_dir_all(parent)?; }

    let results = SparseRandomResults {
        start_date: cli.start_date.clone(),
        n_days: cli.n_days,
        probe: spacecraft,
        embedding_dim: cli.embedding_dim,
        method: format!("sparsity-matched random (CD zero-pattern, T_nonzero ~ N(0, sigma_cd={:.4}))", sigma_cd),
        n_catalog_events: fom_catalog.len(),
        n_fgm_minutes: all_minutes.len(),
        n_draws: cli.n_draws,
        base_seed: cli.base_seed,
        n_nonzero_entries: nonzero_quads.len(),
        sigma_cd,
        f1_mean, f1_std, f1_min, f1_max, f1_mean_plus_2sigma,
        draws,
    };

    let json = serde_json::to_string_pretty(&results)?;
    fs::write(&cli.out_json, &json)?;
    println!("\nResults written to {}", cli.out_json.display());

    Ok(())
}

//! Dense-random trilinear baseline for the CD associator dimensional ablation.
//!
//! WHY: Pre-registered prediction P2 (ablation-preregistered-v1, commit 455d4745):
//! "CD R32 mean F1 will EXCEED the mean of 100 dense-random-trilinear draws by
//! at least 2 sigma of the dense-random distribution."
//! If true, the specific CD coefficient structure contributes beyond embedding
//! dimension alone.  This binary provides the dense-random reference distribution.
//!
//! WHAT: For each of N_DRAWS random seeds, samples T_ijk ~ N(0, 1/sqrt(dim))
//! for i,j,k in [0,dim-1]^3 and uses the trilinear form
//!   `result_i = sum_{j,k} T[i][j][k] * x[j] * y[k]`
//! as the per-step score (||result||_2).  This is the same tensor contraction
//! structure as the CD associator but with random weights.
//!
//! Note: The trilinear form contracts three consecutive delay vectors
//! (x = v_{t-tau}, y = v_t, z = v_{t+tau}) -- but for the SCORE we only
//! contract x and y via T, then take ||T(x,y)||_2 as the change-point signal.
//! This is the closest random analogue to the CD associator norm.
//!
//! HOW:
//!   cargo run --release -p gororoba_cli_physics \
//!     --bin heliosphere-random-trilinear -- \
//!     --start-date 2016-08-29 --n-days 7 \
//!     --staples-catalog data/external/crossing_lists/themis_mp_crossings_v2.txt \
//!     --n-draws 100

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

/// MAD scale factor for the random trilinear sliding-window detector.
/// Identical to heliosphere-themis-staples-labeled; fixed prior to any evaluation.
const MAD_SCALE_FACTOR: f64 = 1.5;

// ============================================================================
// CLI
// ============================================================================

#[derive(Parser, Debug)]
#[command(
    name = "heliosphere-random-trilinear",
    about = "Dense-random trilinear baseline for CD ablation (P2 pre-registered)"
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
    #[arg(long, default_value_t = 10)]
    pad_minutes: i64,

    /// Only include catalog crossings with FOM at or above this value.
    #[arg(long, default_value_t = 0.0)]
    min_fom: f64,

    /// Embedding dimension for Takens delay embedding.
    #[arg(long, default_value_t = 32)]
    embedding_dim: usize,

    /// Takens lag in minutes.
    #[arg(long, default_value_t = 1)]
    takens_lag: usize,

    /// Window size (minutes) for transition suppression.
    #[arg(long, default_value_t = 10)]
    crossing_window_minutes: usize,

    /// Maximum |B| (nT).
    #[arg(long, default_value_t = 100.0)]
    max_bmag: f64,

    /// Instrument noise floor (nT) for denominator clamping.
    #[arg(long, default_value_t = 0.5)]
    bmag_noise_floor: f64,

    /// Number of independent random trilinear tensor draws.
    /// Pre-registered value: 100.
    #[arg(long, default_value_t = 100)]
    n_draws: usize,

    /// Base RNG seed (each draw i uses seed = base_seed + i for reproducibility).
    #[arg(long, default_value_t = 1000)]
    base_seed: u64,

    /// Data cache root directory.
    #[arg(long, default_value = "data/external")]
    data_dir: PathBuf,

    /// Output JSON path.
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/random_trilinear_dense_eval.json"
    )]
    out_json: PathBuf,
}

// ============================================================================
// Output structures
// ============================================================================

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
struct RandomTrilinearResults {
    start_date: String,
    n_days: u32,
    probe: String,
    embedding_dim: usize,
    method: String,
    n_catalog_events: usize,
    n_fgm_minutes: usize,
    n_draws: usize,
    base_seed: u64,
    f1_mean: f64,
    f1_std: f64,
    f1_min: f64,
    f1_max: f64,
    /// Two-sigma upper bound: mean + 2*std.  CD R32 must exceed this to satisfy P2.
    f1_mean_plus_2sigma: f64,
    draws: Vec<DrawResult>,
}

// ============================================================================
// Helpers
// ============================================================================

fn hours_to_unix(origin: &NaiveDateTime, h: f64) -> i64 {
    let base = Utc.from_utc_datetime(origin).timestamp();
    base + (h * 3600.0) as i64
}

fn event_midpoint_unix(ev: &MmsEventInterval) -> i64 {
    let mid = ev.start + (ev.end - ev.start) / 2;
    Utc.from_utc_datetime(&mid).timestamp()
}

/// Sample a dense random trilinear tensor T_ijk ~ N(0, 1/sqrt(dim)) and compute
/// scores: for consecutive delay vector pairs (x, y), compute
///   `score = ||T(x,y)||_2` where `T(x,y)_i = sum_{j,k} T[i*dim*dim + j*dim + k] * x[j] * y[k]`
/// Returns one score per consecutive pair of delay vectors.
fn compute_random_trilinear_scores(delay_vectors: &[Vec<f64>], dim: usize, seed: u64) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    // T_ijk stored as flat [i][j][k] in row-major order, total dim^3 entries.
    // sigma = 1/sqrt(dim) so that T(x,y) has unit-ish variance when x, y are unit vectors.
    let sigma = 1.0 / (dim as f64).sqrt();
    let dist = Normal::new(0.0, sigma).expect("valid normal distribution");
    let t: Vec<f64> = (0..dim * dim * dim)
        .map(|_| dist.sample(&mut rng))
        .collect();

    delay_vectors
        .windows(2)
        .map(|w| {
            let x = &w[0];
            let y = &w[1];
            // result_i = sum_{j,k} T[i*dim*dim + j*dim + k] * x[j] * y[k]
            let mut norm_sq = 0.0_f64;
            for i in 0..dim {
                let mut result_i = 0.0_f64;
                let base = i * dim * dim;
                for (j, &xj) in x.iter().enumerate().take(dim) {
                    if xj == 0.0 {
                        continue;
                    }
                    let row_base = base + j * dim;
                    for k in 0..dim {
                        result_i += t[row_base + k] * xj * y[k];
                    }
                }
                norm_sq += result_i * result_i;
            }
            norm_sq.sqrt()
        })
        .collect()
}

/// Apply MAD_SCALE_FACTOR change-point detection to a score series.
/// Returns elapsed-hours of detected transitions.
fn detect_transitions(
    scores: &[f64],
    score_meta: &[usize],
    all_minutes: &[ThemisFgmMinuteRecord],
    trans_window: usize,
) -> Vec<f64> {
    let mut hours: Vec<f64> = Vec::new();

    if scores.len() <= trans_window * 2 {
        return hours;
    }

    let global_mean: f64 = scores.iter().sum::<f64>() / scores.len() as f64;
    let global_std: f64 = {
        let var = scores
            .iter()
            .map(|&a| (a - global_mean).powi(2))
            .sum::<f64>()
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
            let dominated =
                last_trans_idx.is_some_and(|prev| i.saturating_sub(prev) < trans_window);
            if !dominated {
                let mi = score_meta[i];
                hours.push(all_minutes[mi].elapsed_hours);
                last_trans_idx = Some(i);
            }
        }
    }
    hours
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
    let spacecraft = format!("TH{probe_upper}");

    println!(
        "=== Dense-Random Trilinear Baseline ({} draws) ===\n\
         Window: {} to {} ({} days)\n\
         Probe: {}  Embedding: {}D, lag={}min\n",
        cli.n_draws, start, end, cli.n_days, spacecraft, cli.embedding_dim, cli.takens_lag
    );

    // -----------------------------------------------------------------------
    // Step 1: Fetch THEMIS FGM data
    // -----------------------------------------------------------------------
    println!("[1/4] Fetching THEMIS-{} FGM data...", probe_upper);

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
        println!("  DOY {doy}: downloading...");
        let mut request = TransferRequest::download(&url, &output);
        request.backend = DownloadBackend::CurlCli;
        match DownloadStack::default().recover(&request) {
            Ok(_) => {
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
    println!("[2/4] Parsing FGM to minute-level records...");

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

    println!(
        "  Total: {} minutes across {:.1} hours",
        all_minutes.len(),
        all_minutes.last().map(|r| r.elapsed_hours).unwrap_or(0.0)
    );

    // -----------------------------------------------------------------------
    // Step 3: Load Staples catalog
    // -----------------------------------------------------------------------
    println!("[3/4] Loading Staples+2020 catalog...");

    let catalog_content = fs::read_to_string(&cli.staples_catalog)
        .with_context(|| format!("reading catalog: {}", cli.staples_catalog.display()))?;

    let catalog =
        parse_staples_crossing_catalog(&catalog_content, probe_char, start, end, cli.pad_minutes);

    let fom_catalog: Vec<MmsEventInterval> = catalog
        .into_iter()
        .filter(|e| e.fom >= cli.min_fom)
        .collect();

    println!(
        "  Loaded {} catalog intervals (TH{}, {}-{})",
        fom_catalog.len(),
        probe_upper,
        start,
        end
    );

    // -----------------------------------------------------------------------
    // Step 4: Build delay vectors (shared across all draws)
    // -----------------------------------------------------------------------
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
    let gap_tolerance_hours = 2.0 * cli.takens_lag as f64 / 60.0;
    let max_window_span_hours = expected_span_hours + gap_tolerance_hours;

    let mut delay_vectors: Vec<Vec<f64>> = Vec::new();
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

    println!(
        "  Delay vectors: {} ({} gap-skipped)",
        delay_vectors.len(),
        n_gap_skipped
    );

    let score_meta: Vec<usize> = embed_meta[1..].to_vec();
    let trans_window = cli.crossing_window_minutes.max(5);
    let eval_window_secs = cli.pad_minutes * 60;

    let event_unix: Vec<i64> = fom_catalog.iter().map(event_midpoint_unix).collect();

    // -----------------------------------------------------------------------
    // Step 4: N_DRAWS random trilinear evaluations
    // -----------------------------------------------------------------------
    println!("[4/4] Running {} random trilinear draws...", cli.n_draws);

    let mut draws: Vec<DrawResult> = Vec::with_capacity(cli.n_draws);
    let mut f1_values: Vec<f64> = Vec::with_capacity(cli.n_draws);

    // Note on complexity: dim=32, n_vectors=~10000, n_draws=100
    // Per draw: dim^3 * n_vectors tensor ops = 32^3 * 10000 = ~327M multiplications.
    // At ~1 GFLOP/s single-thread: ~0.3s/draw, 30s total. Acceptable.
    // If --embedding-dim 64 is used: 64^3 * 10000 = ~2.6B ops, ~2.6s/draw, 260s total.
    for draw_idx in 0..cli.n_draws {
        let seed = cli.base_seed + draw_idx as u64;
        if draw_idx % 10 == 0 {
            println!("  Draw {}/{}", draw_idx + 1, cli.n_draws);
        }

        let scores = compute_random_trilinear_scores(&delay_vectors, cli.embedding_dim, seed);
        let fire_hours = detect_transitions(&scores, &score_meta, &all_minutes, trans_window);

        let detection_unix: Vec<i64> = fire_hours
            .iter()
            .map(|&h| hours_to_unix(&reference_midnight, h))
            .collect();

        let (precision, recall, f1) =
            boundary_metrics::precision_recall_f1(&detection_unix, &event_unix, eval_window_secs);

        f1_values.push(f1);
        draws.push(DrawResult {
            draw_index: draw_idx,
            seed,
            n_detections: fire_hours.len(),
            precision,
            recall,
            f1,
        });
    }

    // -----------------------------------------------------------------------
    // Compute distribution statistics
    // -----------------------------------------------------------------------
    let n = f1_values.len() as f64;
    let f1_mean = f1_values.iter().sum::<f64>() / n;
    let f1_std = {
        let var = f1_values
            .iter()
            .map(|&x| (x - f1_mean).powi(2))
            .sum::<f64>()
            / n;
        var.sqrt()
    };
    let f1_min = f1_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let f1_max = f1_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let f1_mean_plus_2sigma = f1_mean + 2.0 * f1_std;

    println!(
        "\n  Random trilinear F1: mean={:.3} std={:.3} min={:.3} max={:.3}",
        f1_mean, f1_std, f1_min, f1_max
    );
    println!(
        "  mean+2sigma = {:.3}  (CD R32 must exceed this to satisfy P2)",
        f1_mean_plus_2sigma
    );

    // -----------------------------------------------------------------------
    // Write output
    // -----------------------------------------------------------------------
    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }

    let results = RandomTrilinearResults {
        start_date: cli.start_date.clone(),
        n_days: cli.n_days,
        probe: spacecraft,
        embedding_dim: cli.embedding_dim,
        method: "dense-random trilinear T_ijk ~ N(0, 1/sqrt(dim))".to_string(),
        n_catalog_events: fom_catalog.len(),
        n_fgm_minutes: all_minutes.len(),
        n_draws: cli.n_draws,
        base_seed: cli.base_seed,
        f1_mean,
        f1_std,
        f1_min,
        f1_max,
        f1_mean_plus_2sigma,
        draws,
    };

    let json = serde_json::to_string_pretty(&results)?;
    fs::write(&cli.out_json, &json)?;
    println!("\nResults written to {}", cli.out_json.display());

    Ok(())
}

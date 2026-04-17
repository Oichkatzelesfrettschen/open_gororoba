//! Voyager 1 heliopause CD associator analysis (Phase 9A.2-9A.3).
//!
//! WHY: The heliopause (V1 2012-08-25, V2 2018-11-05) is the most extreme magnetic
//! boundary crossing in the solar system, at 121 AU.  This binary tests whether the
//! CD associator detects the heliopause crossing signal from 48-second MAG48 data.
//! Since VLISM |B| ~ 0.1-0.5 nT (vs THEMIS ~20-40 nT), absolute eps_0 = 0.5 nT
//! would suppress all signal.  We use relative normalization: divide each embedding
//! channel by the local-window median |B|, clamped to 1% of that median (Phase 0.5
//! RelativeFraction(0.01) semantics).
//!
//! WHAT: Load V1 MAG48 (48s cadence) for a +-window_days window around the heliopause
//! crossing.  Build 4-channel x n_lags Takens delay embedding at 48s lag.  Run CD
//! associator with relative noise floor.  Report: (a) CD firing pattern, (b) whether
//! CD fires within +-6h of the crossing, (c) enrichment ratio: fire rate in +-24h
//! window vs 30-day quiet background.
//!
//! HOW:
//!   cargo run --release -p gororoba_cli_physics \
//!     --bin heliosphere-voyager-heliopause -- \
//!     --spacecraft v1 --window-days 20

use anyhow::{Context, Result};
use clap::Parser;
use serde::Serialize;
use std::{fs, path::PathBuf};

const MAD_SCALE_FACTOR: f64 = 1.5;

#[derive(Parser, Debug)]
#[command(
    name = "heliosphere-voyager-heliopause",
    about = "CD associator analysis of Voyager heliopause crossings (Phase 9A.2-9A.3)"
)]
struct Cli {
    /// Which Voyager spacecraft: v1 (2012-08-25) or v2 (2018-11-05).
    #[arg(long, default_value = "v1")]
    spacecraft: String,
    /// Days before/after crossing to analyze (total window = 2 * window_days).
    #[arg(long, default_value_t = 20)]
    window_days: u32,
    /// Hours on each side of the crossing to call "boundary window" for detection.
    #[arg(long, default_value_t = 6.0)]
    detection_half_window_hours: f64,
    /// Hours on each side for the enrichment ratio "boundary window".
    #[arg(long, default_value_t = 24.0)]
    enrichment_window_hours: f64,
    /// Hours on each side for the quiet background (used for enrichment denominator).
    #[arg(long, default_value_t = 360.0)]
    background_window_hours: f64,
    /// Number of Takens delay lags.
    #[arg(long, default_value_t = 8)]
    n_lags: usize,
    /// Channels per lag (must divide embedding_dim).
    #[arg(long, default_value_t = 4)]
    channels: usize,
    /// Transition detection window (samples).
    #[arg(long, default_value_t = 10)]
    trans_window: usize,
    /// Relative noise floor fraction (eps = frac * median_B).
    #[arg(long, default_value_t = 0.01)]
    eps_rel_frac: f64,
    /// Root data directory.
    #[arg(long, default_value = "data/external")]
    data_dir: PathBuf,
    /// Output JSON path.
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/voyager_heliopause_eval.json"
    )]
    out_json: PathBuf,
}

/// A single 48-second MAG record from the FLT1 ASCII format.
/// Format: `FLT1\tYR_2DIGIT\tDOY_FRAC\tF1\tBR\tBT\tBN`
/// DOY_FRAC is fractional day of year (1.0 = Jan 1 00:00 UT).
#[derive(Debug, Clone)]
struct Mag48Sample {
    /// Fractional day-of-year (1.0 = Jan 1 00:00 UT).
    doy_frac: f64,
    /// |B| in nT.
    b_mag: f64,
    /// B_R (radial) in nT.
    br: f64,
    /// B_T (tangential) in nT.
    bt: f64,
    /// B_N (normal) in nT.
    bn: f64,
}

fn parse_mag48_flt1(content: &str, full_year: u16) -> Vec<Mag48Sample> {
    let mut records = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || !line.starts_with("FLT") { continue; }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 7 { continue; }
        let doy_frac = match fields[2].parse::<f64>() { Ok(v) => v, Err(_) => continue };
        let b_mag = fields[3].parse::<f64>().unwrap_or(f64::NAN);
        let br = fields[4].parse::<f64>().unwrap_or(f64::NAN);
        let bt = fields[5].parse::<f64>().unwrap_or(f64::NAN);
        let bn = fields[6].parse::<f64>().unwrap_or(f64::NAN);
        // Skip fill values (>99 nT is unphysical for VLISM/heliosheath)
        if b_mag > 99.0 || b_mag.is_nan() { continue; }
        if br.abs() > 99.0 || bt.abs() > 99.0 || bn.abs() > 99.0 { continue; }
        let _ = full_year; // year only used for config metadata
        records.push(Mag48Sample { doy_frac, b_mag, br, bt, bn });
    }
    records
}

/// Crossing configuration for each spacecraft.
struct CrossingConfig {
    full_year: u16,
    crossing_doy_frac: f64,
    mag48_dir: PathBuf,
    label: String,
}

fn get_crossing_config(spacecraft: &str, data_dir: &PathBuf) -> Result<CrossingConfig> {
    match spacecraft.to_lowercase().as_str() {
        "v1" => Ok(CrossingConfig {
            full_year: 2012,
            // 2012-08-25 = DOY 238 (2012 is a leap year: 31+29+31+30+31+30+31+25 = 238)
            crossing_doy_frac: 238.0,
            mag48_dir: data_dir.join("voyager1_mag48"),
            label: "Voyager1_heliopause_2012-08-25".to_string(),
        }),
        "v2" => Ok(CrossingConfig {
            full_year: 2018,
            // 2018-11-05 = DOY 309 (31+28+31+30+31+30+31+31+30+31+5 = 309)
            crossing_doy_frac: 309.0,
            mag48_dir: data_dir.join("voyager2_mag48"),
            label: "Voyager2_heliopause_2018-11-05".to_string(),
        }),
        other => anyhow::bail!("Unknown spacecraft '{}'; use v1 or v2", other),
    }
}

fn load_mag48_window(
    config: &CrossingConfig,
    window_days: u32,
) -> Result<Vec<Mag48Sample>> {
    let doy_lo = config.crossing_doy_frac - window_days as f64;
    let doy_hi = config.crossing_doy_frac + window_days as f64;

    let mut all: Vec<Mag48Sample> = Vec::new();
    let dir = fs::read_dir(&config.mag48_dir)
        .with_context(|| format!("reading {}", config.mag48_dir.display()))?;

    for entry in dir {
        let entry = entry?;
        let path = entry.path();
        if !path.extension().map(|e| e == "dat" || e == "txt").unwrap_or(false) { continue; }
        let content = fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let records = parse_mag48_flt1(&content, config.full_year);
        for r in records {
            if r.doy_frac >= doy_lo && r.doy_frac <= doy_hi {
                all.push(r);
            }
        }
    }

    all.sort_by(|a, b| a.doy_frac.partial_cmp(&b.doy_frac).unwrap_or(std::cmp::Ordering::Equal));
    all.dedup_by(|a, b| (a.doy_frac - b.doy_frac).abs() < 1e-8);
    println!("  Loaded {} MAG48 records in DOY [{:.1}, {:.1}]", all.len(), doy_lo, doy_hi);
    Ok(all)
}

#[derive(Debug, Serialize)]
struct VoyagerHeliopauseResults {
    spacecraft: String,
    crossing_label: String,
    n_records: usize,
    window_days: u32,
    n_lags: usize,
    embedding_dim: usize,
    n_detections: usize,
    detected_within_6h: bool,
    closest_detection_hours: f64,
    enrichment_window_hours: f64,
    enrichment_ratio: f64,
    quiet_fire_rate_per_hour: f64,
    boundary_fire_rate_per_hour: f64,
    detection_doys: Vec<f64>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = get_crossing_config(&cli.spacecraft, &cli.data_dir)?;
    let embedding_dim = cli.channels * cli.n_lags;

    println!(
        "=== Voyager Heliopause CD Associator (Phase 9A) ===\n\
         Spacecraft: {}  Crossing: DOY {:.0}  Window: +-{} days\n\
         Embedding: {}ch x {} lags = {}D  eps_rel={}\n",
        config.label, config.crossing_doy_frac, cli.window_days,
        cli.channels, cli.n_lags, embedding_dim, cli.eps_rel_frac
    );

    let records = load_mag48_window(&config, cli.window_days)?;
    if records.len() < embedding_dim + 2 {
        anyhow::bail!("Insufficient data: {} records, need > {}", records.len(), embedding_dim + 2);
    }

    // Compute global median |B| for relative noise floor.
    let mut b_vals: Vec<f64> = records.iter().map(|r| r.b_mag).filter(|b| b.is_finite()).collect();
    b_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let global_median_b = b_vals[b_vals.len() / 2];
    let eps_floor = (cli.eps_rel_frac * global_median_b).max(1e-6);
    println!("  Global median |B| = {:.4} nT  eps_floor = {:.6} nT", global_median_b, eps_floor);

    // Build delay vectors with relative normalization.
    let n_lags = cli.n_lags;
    let channels = cli.channels;
    let window_rows = n_lags; // lag=1 sample (native 48s cadence)
    let mut delay_vectors: Vec<Vec<f64>> = Vec::new();
    let mut embed_meta_doys: Vec<f64> = Vec::new();

    for w_start in 0..=(records.len() - window_rows) {
        let window = &records[w_start..w_start + window_rows];
        // Check for time continuity: max gap between consecutive samples < 2 * 48s = 96s.
        let max_gap_hours = 96.0 / 3600.0;
        let mut valid = true;
        for pair in window.windows(2) {
            let gap = (pair[1].doy_frac - pair[0].doy_frac) * 24.0;
            if gap > max_gap_hours || gap < 0.0 {
                valid = false;
                break;
            }
        }
        if !valid { continue; }

        // Local window median |B| for relative normalization.
        let mut local_b: Vec<f64> = window.iter().map(|r| r.b_mag).filter(|b| b.is_finite()).collect();
        local_b.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let local_median_b = if local_b.is_empty() { global_median_b } else { local_b[local_b.len() / 2] };
        let denom = local_median_b.max(eps_floor);

        let mut v = vec![0.0_f64; embedding_dim];
        for (s, rec) in window.iter().enumerate() {
            v[s * channels] = rec.br / denom;
            v[s * channels + 1] = rec.bt / denom;
            v[s * channels + 2] = rec.bn / denom;
            v[s * channels + 3] = (rec.b_mag - local_median_b) / denom;
        }
        embed_meta_doys.push(window.last().unwrap().doy_frac);
        delay_vectors.push(v);
    }

    println!("  Built {} delay vectors (embedding_dim={})", delay_vectors.len(), embedding_dim);

    if delay_vectors.len() < cli.trans_window * 2 + 2 {
        anyhow::bail!("Not enough valid delay vectors: {}", delay_vectors.len());
    }

    let associators = cd_kernel::batch_sliding_associator_norms_parallel(&delay_vectors, embedding_dim);
    let assoc_doys: Vec<f64> = embed_meta_doys[2..].to_vec();

    println!("  Computed {} associator scores", associators.len());

    // Threshold and detect.
    let global_mean: f64 = associators.iter().sum::<f64>() / associators.len() as f64;
    let global_std: f64 = (associators.iter().map(|&a| (a - global_mean).powi(2)).sum::<f64>()
        / associators.len() as f64).sqrt();
    let threshold = global_std * MAD_SCALE_FACTOR;

    let half = cli.trans_window;
    let mut fire_doys: Vec<f64> = Vec::new();
    let mut last_fire_idx: Option<usize> = None;

    for i in half..associators.len().saturating_sub(half) {
        let pre_mean: f64 = associators[i.saturating_sub(half)..i].iter().sum::<f64>() / half as f64;
        let post_mean: f64 = associators[i..(i + half).min(associators.len())].iter().sum::<f64>()
            / half.min(associators.len() - i) as f64;
        let jump = (post_mean - pre_mean).abs();
        if jump > threshold {
            let dominated = last_fire_idx.is_some_and(|prev| i.saturating_sub(prev) < half);
            if !dominated {
                fire_doys.push(assoc_doys[i]);
                last_fire_idx = Some(i);
            }
        }
    }

    println!("  CD fires: {}", fire_doys.len());

    // Evaluation metrics.
    let crossing_doy = config.crossing_doy_frac;
    let detection_half_h = cli.detection_half_window_hours;
    let detected_within = fire_doys.iter().any(|&d| (d - crossing_doy).abs() * 24.0 <= detection_half_h);

    let closest_h = fire_doys.iter()
        .map(|&d| (d - crossing_doy).abs() * 24.0)
        .fold(f64::INFINITY, f64::min);

    // Enrichment ratio: fire_rate in boundary_window_hours vs quiet background.
    let enrichment_half_doy = cli.enrichment_window_hours / 24.0;
    let boundary_fires = fire_doys.iter().filter(|&&d| (d - crossing_doy).abs() <= enrichment_half_doy).count();
    let boundary_duration_hours = 2.0 * cli.enrichment_window_hours;
    let boundary_rate = boundary_fires as f64 / boundary_duration_hours;

    let bg_half_doy = cli.background_window_hours / 24.0;
    let quiet_fires = fire_doys.iter().filter(|&&d| {
        (d - crossing_doy).abs() <= bg_half_doy && (d - crossing_doy).abs() > enrichment_half_doy
    }).count();
    let quiet_duration_hours = 2.0 * cli.background_window_hours - 2.0 * cli.enrichment_window_hours;
    let quiet_rate = if quiet_duration_hours > 0.0 { quiet_fires as f64 / quiet_duration_hours } else { 0.0 };

    let enrichment_ratio = if quiet_rate > 0.0 { boundary_rate / quiet_rate } else { f64::INFINITY };

    println!(
        "  Detected within +-{:.0}h: {}  Closest: {:.2}h",
        detection_half_h, detected_within, closest_h
    );
    println!(
        "  Enrichment ratio: {:.2}x (boundary_rate={:.4}/h  quiet_rate={:.4}/h)",
        enrichment_ratio, boundary_rate, quiet_rate
    );

    if let Some(parent) = cli.out_json.parent() { fs::create_dir_all(parent)?; }

    let results = VoyagerHeliopauseResults {
        spacecraft: cli.spacecraft.clone(),
        crossing_label: config.label,
        n_records: records.len(),
        window_days: cli.window_days,
        n_lags: cli.n_lags,
        embedding_dim,
        n_detections: fire_doys.len(),
        detected_within_6h: detected_within,
        closest_detection_hours: closest_h,
        enrichment_window_hours: cli.enrichment_window_hours,
        enrichment_ratio,
        quiet_fire_rate_per_hour: quiet_rate,
        boundary_fire_rate_per_hour: boundary_rate,
        detection_doys: fire_doys,
    };

    let json = serde_json::to_string_pretty(&results)?;
    fs::write(&cli.out_json, &json)?;
    println!("Results written to {}", cli.out_json.display());

    Ok(())
}

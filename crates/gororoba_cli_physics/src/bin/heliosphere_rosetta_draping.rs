//! Rosetta 67P cometary draping analysis with 32D CD associator.
//!
//! Fetches RPC-MAG data from AMDA for the comet escort phase, detects
//! diamagnetic cavity boundaries via |B| drop, and runs the 32D CD
//! associator to test whether it responds to cometary draping transitions.
//!
//! WHY: Cometary magnetic draping is structurally different from the
//! heliopause and magnetopause: the field wraps around a neutral gas coma
//! rather than a magnetized obstacle. If the CD associator detects the
//! draping boundary, it demonstrates universality across boundary types.

use anyhow::{Context, Result};
use clap::Parser;
use data_core::{
    catalogs::rosetta::{
        RosettaMagMinuteRecord, RosettaMagProvider,
        parse_rosetta_amda_mag_csv_minutes,
    },
    fetcher::{DatasetProvider, FetchConfig},
};
use serde::Serialize;
use std::{fs, path::PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "heliosphere-rosetta-draping",
    about = "Rosetta 67P draping analysis with 32D CD associator"
)]
struct Cli {
    /// Year to analyze.
    #[arg(long, default_value_t = 2015)]
    year: u16,

    /// Start month (1-12).
    #[arg(long, default_value_t = 7)]
    month_start: u8,

    /// End month (1-12).
    #[arg(long, default_value_t = 9)]
    month_end: u8,

    /// Embedding dimension.
    #[arg(long, default_value_t = 32)]
    embedding_dim: usize,

    /// Diamagnetic cavity threshold: |B| below this (nT) = inside cavity.
    #[arg(long, default_value_t = 2.0)]
    cavity_threshold: f64,

    /// Window size (minutes) for boundary detection.
    #[arg(long, default_value_t = 10)]
    boundary_window_minutes: usize,

    /// |B| gradient threshold (nT) for boundary detection.
    #[arg(long, default_value_t = 3.0)]
    bmag_gradient_threshold: f64,

    /// Normalization mode: current, raw, direction, clipped.
    /// current = Bx/mean_B (default), raw = unnormalized, direction = unit vectors,
    /// clipped = Bx/max(mean_B, floor).
    #[arg(long, default_value = "current")]
    normalization: String,

    /// Floor for clipped normalization (nT).
    #[arg(long, default_value_t = 1.0)]
    clip_floor: f64,

    /// Output JSON path.
    #[arg(long, default_value = "data/output/heliosphere/ablations/rosetta_draping_analysis.json")]
    out_json: PathBuf,

    /// Data cache directory.
    #[arg(long, default_value = "data/external")]
    data_dir: PathBuf,
}

#[derive(Debug, Serialize)]
struct DrapingResults {
    year: u16,
    month_start: u8,
    month_end: u8,
    embedding_dim: usize,
    normalization: String,
    total_minutes: usize,
    cavity_minutes: usize,
    cavity_fraction: f64,
    n_cavity_boundaries: usize,
    n_associator_transitions: usize,
    detection_rate: f64,
    false_alarm_rate: f64,
    mean_offset_minutes: f64,
    cavity_mean_associator: f64,
    outside_mean_associator: f64,
    ratio_outside_to_cavity: f64,
    hourly_summary: Vec<HourlySummary>,
}

#[derive(Debug, Serialize)]
struct HourlySummary {
    year: u16,
    doy: u16,
    hour: u8,
    mean_bmag: f64,
    mean_associator: f64,
    cavity_fraction: f64,
    n_samples: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    println!(
        "=== Rosetta 67P Draping Analysis ===\n\
         Window: {}-{:02} to {}-{:02}\n\
         Embedding: {}D\n\
         Cavity threshold: {:.1} nT\n",
        cli.year, cli.month_start, cli.year, cli.month_end,
        cli.embedding_dim, cli.cavity_threshold,
    );

    // --- Step 1: Fetch data from AMDA ---
    println!("[1/5] Fetching Rosetta RPC-MAG data from AMDA...");

    let provider = RosettaMagProvider {
        year_start: cli.year,
        year_end: cli.year,
        month_range: Some((cli.month_start, cli.month_end)),
    };

    let config = FetchConfig {
        output_dir: cli.data_dir.clone(),
        skip_existing: true,
        verify_checksums: false,
    };

    let data_dir = provider.fetch(&config)?;

    // --- Step 2: Parse to minute-level records ---
    println!("[2/5] Parsing to minute-level records...");

    let mut all_minutes: Vec<RosettaMagMinuteRecord> = Vec::new();

    for month in cli.month_start..=cli.month_end {
        let fname = format!("rosetta_rpcmag_{:04}_{:02}.csv", cli.year, month);
        let path = data_dir.join(&fname);
        if path.exists() {
            let content = fs::read_to_string(&path)
                .with_context(|| format!("read {}", path.display()))?;
            let mut records = parse_rosetta_amda_mag_csv_minutes(&content);
            println!("  {}-{:02}: {} minutes", cli.year, month, records.len());
            all_minutes.append(&mut records);
        } else {
            println!("  {}-{:02}: no data", cli.year, month);
        }
    }

    if all_minutes.is_empty() {
        anyhow::bail!("No Rosetta data found. AMDA may be unavailable.");
    }

    // Sort and recompute elapsed hours
    all_minutes.sort_by(|a, b| {
        a.year.cmp(&b.year)
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
        rec.elapsed_hours =
            day_diff * 24.0 + (rec.hour as f64 - fh as f64) + (rec.minute as f64 - fm as f64) / 60.0;
    }

    let total = all_minutes.len();
    let cavity_count = all_minutes
        .iter()
        .filter(|r| r.b_magnitude < cli.cavity_threshold)
        .count();

    println!(
        "  Total: {} minutes, cavity fraction: {:.1}%",
        total,
        100.0 * cavity_count as f64 / total as f64
    );

    // --- Step 3: Detect cavity boundaries ---
    println!("[3/5] Detecting diamagnetic cavity boundaries...");

    let half = cli.boundary_window_minutes;
    let mut boundaries: Vec<usize> = Vec::new();

    if all_minutes.len() > half * 2 + 1 {
        for i in half..all_minutes.len().saturating_sub(half) {
            let pre_mean: f64 = all_minutes[i.saturating_sub(half)..i]
                .iter()
                .map(|r| r.b_magnitude)
                .sum::<f64>()
                / half as f64;
            let post_mean: f64 = all_minutes[i..(i + half).min(all_minutes.len())]
                .iter()
                .map(|r| r.b_magnitude)
                .sum::<f64>()
                / half.min(all_minutes.len() - i) as f64;

            let jump = (post_mean - pre_mean).abs();
            if jump > cli.bmag_gradient_threshold {
                let dominated = boundaries
                    .last()
                    .is_some_and(|&prev| i.saturating_sub(prev) < cli.boundary_window_minutes);
                if !dominated {
                    boundaries.push(i);
                }
            }
        }
    }

    println!("  Found {} cavity boundaries", boundaries.len());

    // --- Step 4: Run 32D CD associator ---
    let norm_mode = cli.normalization.as_str();
    println!(
        "[4/5] Computing {}D Takens embedding + CD associator (norm={})...",
        cli.embedding_dim, norm_mode,
    );

    // Always 4 channels for CD power-of-2 dimension requirement.
    // Direction mode sets 4th channel to zero (no magnitude info).
    let channels: usize = 4;
    let steps = cli.embedding_dim / channels;
    let window_rows = (steps - 1) + 1;

    if all_minutes.len() < window_rows + 2 {
        anyhow::bail!(
            "Not enough data: need {} rows, have {}",
            window_rows + 2,
            all_minutes.len()
        );
    }

    let effective_dim = cli.embedding_dim;

    let mut embedded: Vec<Vec<f64>> = Vec::new();
    let mut embed_meta: Vec<usize> = Vec::new();

    for w_start in 0..=(all_minutes.len() - window_rows) {
        let sample_indices: Vec<usize> = (0..steps).map(|s| w_start + s).collect();

        let sum_b: f64 = sample_indices.iter().map(|&i| all_minutes[i].b_magnitude).sum();
        let local_mean_b = sum_b / steps as f64;

        // Skip zero-field windows for all modes except raw
        if norm_mode != "raw" && (local_mean_b <= 0.0 || !local_mean_b.is_finite()) {
            continue;
        }

        let mut v = vec![0.0; effective_dim];
        let mut skip = false;
        for (s, &ri) in sample_indices.iter().enumerate() {
            let rec = &all_minutes[ri];
            match norm_mode {
                "current" => {
                    v[s * channels] = rec.bx_cseq / local_mean_b;
                    v[s * channels + 1] = rec.by_cseq / local_mean_b;
                    v[s * channels + 2] = rec.bz_cseq / local_mean_b;
                    v[s * channels + 3] = (rec.b_magnitude - local_mean_b) / local_mean_b;
                }
                "raw" => {
                    v[s * channels] = rec.bx_cseq;
                    v[s * channels + 1] = rec.by_cseq;
                    v[s * channels + 2] = rec.bz_cseq;
                    v[s * channels + 3] = rec.b_magnitude;
                }
                "direction" => {
                    // Unit vectors: Bx/|B|, By/|B|, Bz/|B|, 0.0
                    let bmag = rec.b_magnitude;
                    if bmag < 1e-12 { skip = true; break; }
                    v[s * channels] = rec.bx_cseq / bmag;
                    v[s * channels + 1] = rec.by_cseq / bmag;
                    v[s * channels + 2] = rec.bz_cseq / bmag;
                    v[s * channels + 3] = 0.0; // no magnitude info
                }
                "l2" => {
                    // L2 energy-preserving: Bx / sqrt(mean(|B|^2))
                    let sum_b2: f64 = sample_indices.iter()
                        .map(|&i| all_minutes[i].b_magnitude.powi(2))
                        .sum();
                    let rms_b = (sum_b2 / steps as f64).sqrt();
                    if rms_b < 1e-12 { skip = true; break; }
                    v[s * channels] = rec.bx_cseq / rms_b;
                    v[s * channels + 1] = rec.by_cseq / rms_b;
                    v[s * channels + 2] = rec.bz_cseq / rms_b;
                    v[s * channels + 3] = (rec.b_magnitude - rms_b) / rms_b;
                }
                "linf" => {
                    // L-inf scale-invariant: Bx / max(|B|)
                    let max_b: f64 = sample_indices.iter()
                        .map(|&i| all_minutes[i].b_magnitude)
                        .fold(0.0f64, f64::max);
                    if max_b < 1e-12 { skip = true; break; }
                    v[s * channels] = rec.bx_cseq / max_b;
                    v[s * channels + 1] = rec.by_cseq / max_b;
                    v[s * channels + 2] = rec.bz_cseq / max_b;
                    v[s * channels + 3] = (rec.b_magnitude - max_b) / max_b;
                }
                "clipped" => {
                    let denom = local_mean_b.max(cli.clip_floor);
                    v[s * channels] = rec.bx_cseq / denom;
                    v[s * channels + 1] = rec.by_cseq / denom;
                    v[s * channels + 2] = rec.bz_cseq / denom;
                    v[s * channels + 3] = (rec.b_magnitude - denom) / denom;
                }
                _ => {
                    anyhow::bail!("Unknown normalization: {}. Use current, raw, direction, clipped, l2, linf.", norm_mode);
                }
            }
        }
        if skip { continue; }
        embedded.push(v);
        embed_meta.push(*sample_indices.last().unwrap());
    }

    println!("  Embedded {} vectors ({}D)", embedded.len(), cli.embedding_dim);

    let associators =
        cd_kernel::batch_sliding_associator_norms_parallel(&embedded, effective_dim);

    println!("  Computed {} associator norms", associators.len());

    let assoc_minute_indices: Vec<usize> = (0..associators.len())
        .map(|k| embed_meta[k + 2])
        .collect();

    // Detect associator transitions
    let trans_window = cli.boundary_window_minutes.max(5);
    let mut transitions: Vec<usize> = Vec::new();

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

        println!(
            "  Associator stats: mean={:.4}, std={:.4}, threshold={:.4}",
            global_mean, global_std, threshold
        );

        for i in trans_window..associators.len().saturating_sub(trans_window) {
            let pre: f64 = associators[i.saturating_sub(trans_window)..i].iter().sum::<f64>()
                / trans_window as f64;
            let post: f64 = associators[i..(i + trans_window).min(associators.len())]
                .iter()
                .sum::<f64>()
                / trans_window.min(associators.len() - i) as f64;
            let jump = (post - pre).abs();
            if jump > threshold {
                let dominated = transitions
                    .last()
                    .is_some_and(|&prev| i.saturating_sub(prev) < trans_window);
                if !dominated {
                    transitions.push(assoc_minute_indices[i]);
                }
            }
        }
    }

    println!("  Found {} associator transitions", transitions.len());

    // --- Step 5: Cross-compare and compute statistics ---
    println!("[5/5] Cross-comparing boundaries vs transitions...");

    let tol = cli.boundary_window_minutes as f64 / 60.0; // hours tolerance
    let mut matched = 0usize;
    let mut offsets: Vec<f64> = Vec::new();

    for &bi in &boundaries {
        let b_hours = all_minutes[bi].elapsed_hours;
        let best = transitions
            .iter()
            .filter(|&&ti| {
                let t_hours = all_minutes[ti].elapsed_hours;
                (t_hours - b_hours).abs() < tol
            })
            .min_by(|&&a, &&b| {
                let da = (all_minutes[a].elapsed_hours - b_hours).abs();
                let db = (all_minutes[b].elapsed_hours - b_hours).abs();
                da.partial_cmp(&db).unwrap()
            });
        if let Some(&ti) = best {
            matched += 1;
            let offset = (all_minutes[ti].elapsed_hours - b_hours).abs() * 60.0;
            offsets.push(offset);
        }
    }

    let matched_transitions = transitions
        .iter()
        .filter(|&&ti| {
            let t_hours = all_minutes[ti].elapsed_hours;
            boundaries.iter().any(|&bi| {
                (all_minutes[bi].elapsed_hours - t_hours).abs() < tol
            })
        })
        .count();

    let detection_rate = if boundaries.is_empty() {
        0.0
    } else {
        matched as f64 / boundaries.len() as f64
    };
    let false_alarm = if transitions.is_empty() {
        0.0
    } else {
        (transitions.len() - matched_transitions) as f64 / transitions.len() as f64
    };
    offsets.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_offset = if offsets.is_empty() {
        0.0
    } else {
        offsets.iter().sum::<f64>() / offsets.len() as f64
    };

    // Cavity vs outside associator
    let mut cavity_assocs: Vec<f64> = Vec::new();
    let mut outside_assocs: Vec<f64> = Vec::new();

    for (k, &norm) in associators.iter().enumerate() {
        let mi = assoc_minute_indices[k];
        if all_minutes[mi].b_magnitude < cli.cavity_threshold {
            cavity_assocs.push(norm);
        } else {
            outside_assocs.push(norm);
        }
    }

    let cavity_mean = if cavity_assocs.is_empty() {
        0.0
    } else {
        cavity_assocs.iter().sum::<f64>() / cavity_assocs.len() as f64
    };
    let outside_mean = if outside_assocs.is_empty() {
        0.0
    } else {
        outside_assocs.iter().sum::<f64>() / outside_assocs.len() as f64
    };
    let ratio = if cavity_mean > 1e-15 {
        outside_mean / cavity_mean
    } else {
        0.0
    };

    println!("\n=== Results ===");
    println!("  Cavity boundaries: {}", boundaries.len());
    println!("  Associator transitions: {}", transitions.len());
    println!("  Detection rate: {:.1}%", detection_rate * 100.0);
    println!("  False alarm rate: {:.1}%", false_alarm * 100.0);
    println!("  Mean offset: {:.1} min", mean_offset);
    println!(
        "  Cavity associator: {:.4} (n={}), Outside: {:.4} (n={})",
        cavity_mean,
        cavity_assocs.len(),
        outside_mean,
        outside_assocs.len()
    );
    println!("  Ratio (outside/cavity): {:.3}", ratio);

    // Build hourly summary
    type HourKey = (u16, u16, u8);
    let mut hourly_map: std::collections::BTreeMap<HourKey, (Vec<f64>, Vec<f64>, usize)> =
        std::collections::BTreeMap::new();

    for (k, &norm) in associators.iter().enumerate() {
        let mi = assoc_minute_indices[k];
        let rec = &all_minutes[mi];
        let entry = hourly_map
            .entry((rec.year, rec.doy, rec.hour))
            .or_insert_with(|| (Vec::new(), Vec::new(), 0));
        entry.0.push(norm);
        entry.1.push(rec.b_magnitude);
        if rec.b_magnitude < cli.cavity_threshold {
            entry.2 += 1;
        }
    }

    let hourly_summary: Vec<HourlySummary> = hourly_map
        .iter()
        .map(|(&(year, doy, hour), (assocs, bmags, cav))| {
            let n = assocs.len();
            HourlySummary {
                year,
                doy,
                hour,
                mean_bmag: bmags.iter().sum::<f64>() / n as f64,
                mean_associator: assocs.iter().sum::<f64>() / n as f64,
                cavity_fraction: *cav as f64 / n as f64,
                n_samples: n,
            }
        })
        .collect();

    let results = DrapingResults {
        year: cli.year,
        month_start: cli.month_start,
        month_end: cli.month_end,
        embedding_dim: cli.embedding_dim,
        normalization: cli.normalization.clone(),
        total_minutes: total,
        cavity_minutes: cavity_count,
        cavity_fraction: cavity_count as f64 / total as f64,
        n_cavity_boundaries: boundaries.len(),
        n_associator_transitions: transitions.len(),
        detection_rate,
        false_alarm_rate: false_alarm,
        mean_offset_minutes: mean_offset,
        cavity_mean_associator: cavity_mean,
        outside_mean_associator: outside_mean,
        ratio_outside_to_cavity: ratio,
        hourly_summary,
    };

    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(&results)?;
    fs::write(&cli.out_json, &json)?;
    println!("\nWrote results to {}", cli.out_json.display());

    Ok(())
}

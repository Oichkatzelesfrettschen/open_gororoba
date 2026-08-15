//! Multi-day MMS magnetopause crossing analysis with 32D CD associator.
//!
//! Fetches multi-day MMS1 FGM Survey L2 data via CDAWeb HAPI, detects
//! magnetopause crossings via |B| gradient + rotation, independently runs
//! the 32D Cayley-Dickson associator on Takens-embedded minute-level data,
//! and cross-compares the two detection methods.
//!
//! WHY: Validates that the CD associator detects the same boundary transitions
//! as the standard |B|-gradient method, establishing it as a general-purpose
//! boundary detector not specific to the heliopause.

use anyhow::{Context, Result};
use chrono::{Datelike, NaiveDate};
use clap::Args;
use data_core::{
    catalogs::mms::{
        MmsFgmMinuteRecord, detect_magnetopause_crossings_filtered, parse_mms_fgm_hapi_csv_minutes,
    },
    fetcher::download_hapi_csv_to_file,
};
use serde::Serialize;
use std::{fs, path::PathBuf};

#[derive(Args, Debug)]
pub struct Cli {
    /// Start date (YYYY-MM-DD).
    #[arg(long, default_value = "2024-01-01")]
    start_date: String,

    /// Number of days to analyze.
    #[arg(long, default_value_t = 7)]
    n_days: u32,

    /// Embedding dimension for Takens delay embedding.
    #[arg(long, default_value_t = 32)]
    embedding_dim: usize,

    /// Takens lag in minutes.
    #[arg(long, default_value_t = 1)]
    takens_lag: usize,

    /// Window size (minutes) for crossing detection sliding window.
    #[arg(long, default_value_t = 10)]
    crossing_window_minutes: usize,

    /// |B| gradient threshold (nT) for crossing detection.
    #[arg(long, default_value_t = 5.0)]
    bmag_gradient_threshold: f64,

    /// Minimum vector rotation (degrees) required between pre- and post-window
    /// mean B vectors to accept a crossing candidate.  Filters compressive
    /// events (pressure pulses, flares) that change |B| without rotating the
    /// field direction.  Recommended: 30.0.  Omit to use gradient-only criterion.
    #[arg(long)]
    rotation_threshold_deg: Option<f64>,

    /// Tolerance (minutes) for matching associator transitions to crossings.
    #[arg(long, default_value_t = 15)]
    match_tolerance_minutes: usize,

    /// Maximum |B| (nT) to include. Filters out inner magnetosphere transits
    /// where |B| >> 100 nT due to dipole field at perigee.
    #[arg(long, default_value_t = 100.0)]
    max_bmag: f64,

    /// Output JSON path.
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/mms_multiday_crossing_analysis.json"
    )]
    out_json: PathBuf,

    /// Data cache directory.
    #[arg(long, default_value = "data/external")]
    data_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize)]
struct CrossingEvent {
    index: usize,
    elapsed_hours: f64,
    year: u16,
    doy: u16,
    hour: u8,
    minute: u8,
    b_magnitude: f64,
}

#[derive(Debug, Clone, Serialize)]
struct AssociatorTransition {
    index: usize,
    elapsed_hours: f64,
    year: u16,
    doy: u16,
    hour: u8,
    minute: u8,
    associator_norm: f64,
    pre_mean: f64,
    post_mean: f64,
    transition_magnitude: f64,
}

#[derive(Debug, Serialize)]
struct CrossingComparison {
    crossing_index: usize,
    crossing_hours: f64,
    matched_transition_index: Option<usize>,
    matched_transition_hours: Option<f64>,
    offset_minutes: Option<f64>,
}

#[derive(Debug, Serialize)]
struct MultiDayResults {
    start_date: String,
    n_days: u32,
    embedding_dim: usize,
    takens_lag: usize,
    bmag_gradient_threshold: f64,
    rotation_threshold_deg: Option<f64>,
    total_minutes: usize,
    n_crossings_detected: usize,
    n_associator_transitions: usize,
    crossings: Vec<CrossingEvent>,
    transitions: Vec<AssociatorTransition>,
    comparisons: Vec<CrossingComparison>,
    detection_rate: f64,
    false_alarm_rate: f64,
    mean_offset_minutes: f64,
    median_offset_minutes: f64,
    hourly_associator_summary: Vec<HourlySummary>,
}

#[derive(Debug, Serialize)]
struct HourlySummary {
    elapsed_hours: f64,
    year: u16,
    doy: u16,
    hour: u8,
    mean_associator: f64,
    max_associator: f64,
    mean_bmag: f64,
    n_samples: usize,
}

pub fn run(cli: Cli) -> Result<()> {
    let start = NaiveDate::parse_from_str(&cli.start_date, "%Y-%m-%d")
        .with_context(|| format!("invalid start_date: {}", cli.start_date))?;

    let end = start + chrono::Duration::days(cli.n_days as i64 - 1);

    println!(
        "=== MMS Multi-Day Crossing Analysis ===\n\
         Window: {} to {} ({} days)\n\
         Embedding: {}D, lag={}min\n\
         Crossing detector: window={}min, |B| threshold={:.1} nT\n",
        start,
        end,
        cli.n_days,
        cli.embedding_dim,
        cli.takens_lag,
        cli.crossing_window_minutes,
        cli.bmag_gradient_threshold,
    );

    // --- Step 1: Fetch data via HAPI (daily chunks to avoid timeout) ---
    println!("[1/5] Fetching MMS1 FGM data (daily chunks)...");

    let mms_dataset = "MMS1_FGM_SRVY_L2@0";
    let mms_params: &[&str] = &["Time", "mms1_fgm_b_gse_srvy_l2_clean"];

    let data_dir = cli.data_dir.join("mms");
    fs::create_dir_all(&data_dir)?;

    for day_offset in 0..cli.n_days {
        let date = start + chrono::Duration::days(day_offset as i64);
        let doy = date.ordinal() as u16;
        let y = date.year() as u16;
        let fname = format!("mms1_fgm_srvy_l2_{y}_{doy}_{doy}.csv");
        let output = data_dir.join(&fname);

        if output.exists() {
            println!("  DOY {doy}: cached");
            continue;
        }

        let t_min = format!("{}T00:00:00Z", date.format("%Y-%m-%d"));
        let t_max = format!("{}T23:59:59Z", date.format("%Y-%m-%d"));

        println!("  DOY {doy}: downloading via file-based path (aria2c fallback)...");
        match download_hapi_csv_to_file(mms_dataset, &t_min, &t_max, Some(mms_params), &output) {
            Ok(bytes) => println!("  DOY {doy}: {:.1} MB", bytes as f64 / 1_048_576.0),
            Err(e) => eprintln!("  Warning: DOY {doy} failed: {e}"),
        }
    }

    // --- Step 2: Parse to minute-level records ---
    println!("[2/5] Parsing to minute-level records...");

    let mut all_minutes: Vec<MmsFgmMinuteRecord> = Vec::new();

    for day_offset in 0..cli.n_days {
        let date = start + chrono::Duration::days(day_offset as i64);
        let doy = date.ordinal() as u16;
        let y = date.year() as u16;
        let path = data_dir.join(format!("mms1_fgm_srvy_l2_{y}_{doy}_{doy}.csv"));
        if path.exists() {
            let content = fs::read_to_string(&path)?;
            let mut records = parse_mms_fgm_hapi_csv_minutes(&content);
            println!("  Day {} (DOY {}): {} minutes", date, doy, records.len());
            all_minutes.append(&mut records);
        } else {
            println!("  Day {} (DOY {}): no data", date, doy);
        }
    }

    if all_minutes.is_empty() {
        anyhow::bail!("No MMS data found. Run fetch-datasets first.");
    }

    // Filter out inner magnetosphere transits (perigee |B| >> 100 nT)
    let pre_filter = all_minutes.len();
    all_minutes.retain(|r| r.b_magnitude <= cli.max_bmag);
    let filtered = pre_filter - all_minutes.len();
    if filtered > 0 {
        println!(
            "  Filtered {} minutes with |B| > {:.0} nT (inner magnetosphere)",
            filtered, cli.max_bmag
        );
    }

    // Sort by elapsed time and recompute elapsed_hours from first record
    all_minutes.sort_by(|a, b| {
        a.year
            .cmp(&b.year)
            .then(a.doy.cmp(&b.doy))
            .then(a.hour.cmp(&b.hour))
            .then(a.minute.cmp(&b.minute))
    });

    // Recompute elapsed hours from the actual first record
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
        "  Total: {} minutes across {:.1} hours",
        all_minutes.len(),
        all_minutes.last().map(|r| r.elapsed_hours).unwrap_or(0.0)
    );

    // --- Step 3: Detect magnetopause crossings via |B| gradient (+optional rotation) ---
    {
        let rot_note = match cli.rotation_threshold_deg {
            Some(r) => format!(", rotation>={:.0} deg", r),
            None => String::new(),
        };
        println!(
            "[3/5] Detecting magnetopause crossings (|B| gradient{})...",
            rot_note
        );
    }

    let crossing_indices = detect_magnetopause_crossings_filtered(
        &all_minutes,
        cli.crossing_window_minutes,
        cli.bmag_gradient_threshold,
        cli.rotation_threshold_deg,
    );

    let crossings: Vec<CrossingEvent> = crossing_indices
        .iter()
        .map(|&i| {
            let r = &all_minutes[i];
            CrossingEvent {
                index: i,
                elapsed_hours: r.elapsed_hours,
                year: r.year,
                doy: r.doy,
                hour: r.hour,
                minute: r.minute,
                b_magnitude: r.b_magnitude,
            }
        })
        .collect();

    println!("  Found {} magnetopause crossings", crossings.len());
    for c in &crossings {
        println!(
            "    DOY {} {:02}:{:02} UT  |B|={:.1} nT  (t={:.1}h)",
            c.doy, c.hour, c.minute, c.b_magnitude, c.elapsed_hours
        );
    }

    // --- Step 4: Run 32D CD associator on Takens-embedded data ---
    println!(
        "[4/5] Computing {}D Takens embedding + CD associator...",
        cli.embedding_dim
    );

    let channels: usize = 4; // Bx, By, Bz, (|B|-mean)/mean
    let steps = cli.embedding_dim / channels;
    let window_rows = (steps - 1) * cli.takens_lag + 1;

    if all_minutes.len() < window_rows + 2 {
        anyhow::bail!(
            "Not enough data: need {} rows for embedding, have {}",
            window_rows + 2,
            all_minutes.len()
        );
    }

    // Build Takens-embedded vectors from minute-level MMS records
    let mut embedded_vectors: Vec<Vec<f64>> = Vec::new();
    let mut embed_meta: Vec<usize> = Vec::new(); // index of last row in each embedding window

    for w_start in 0..=(all_minutes.len() - window_rows) {
        let sample_indices: Vec<usize> = (0..steps).map(|s| w_start + s * cli.takens_lag).collect();

        let sum_b: f64 = sample_indices
            .iter()
            .map(|&i| all_minutes[i].b_magnitude)
            .sum();
        let local_mean_b = sum_b / steps as f64;
        if local_mean_b <= 0.0 || !local_mean_b.is_finite() {
            continue;
        }

        let mut v = vec![0.0; cli.embedding_dim];
        for (s, &ri) in sample_indices.iter().enumerate() {
            let rec = &all_minutes[ri];
            v[s * channels] = rec.bx_gse / local_mean_b;
            v[s * channels + 1] = rec.by_gse / local_mean_b;
            v[s * channels + 2] = rec.bz_gse / local_mean_b;
            v[s * channels + 3] = (rec.b_magnitude - local_mean_b) / local_mean_b;
        }
        embedded_vectors.push(v);
        embed_meta.push(*sample_indices.last().unwrap());
    }

    println!(
        "  Embedded {} vectors ({}D, lag={})",
        embedded_vectors.len(),
        cli.embedding_dim,
        cli.takens_lag
    );

    let associators =
        cd_kernel::batch_sliding_associator_norms_parallel(&embedded_vectors, cli.embedding_dim);

    println!("  Computed {} associator norms", associators.len());

    // Map each associator back to its minute index:
    // associator[k] = triple (emb[k], emb[k+1], emb[k+2])
    // spatial tag from embed_meta[k+2]
    let assoc_minute_indices: Vec<usize> =
        (0..associators.len()).map(|k| embed_meta[k + 2]).collect();

    // --- Step 4b: Detect associator transitions ---
    // A transition is where the local associator mean changes significantly
    let trans_window = cli.crossing_window_minutes.max(5);
    let mut transitions: Vec<AssociatorTransition> = Vec::new();

    if associators.len() > trans_window * 2 {
        // Compute global statistics for threshold calibration
        let global_mean: f64 = associators.iter().sum::<f64>() / associators.len() as f64;
        let global_std: f64 = {
            let var = associators
                .iter()
                .map(|&a| (a - global_mean).powi(2))
                .sum::<f64>()
                / associators.len() as f64;
            var.sqrt()
        };
        let transition_threshold = global_std * 1.5;

        println!(
            "  Associator stats: mean={:.4}, std={:.4}, transition threshold={:.4}",
            global_mean, global_std, transition_threshold
        );

        let half = trans_window;
        for i in half..associators.len().saturating_sub(half) {
            let pre_mean: f64 =
                associators[i.saturating_sub(half)..i].iter().sum::<f64>() / half as f64;
            let post_mean: f64 = associators[i..(i + half).min(associators.len())]
                .iter()
                .sum::<f64>()
                / half.min(associators.len() - i) as f64;

            let jump = (post_mean - pre_mean).abs();
            if jump > transition_threshold {
                // Suppress duplicates
                let dominated = transitions
                    .last()
                    .is_some_and(|prev| i.saturating_sub(prev.index) < trans_window);
                if !dominated {
                    let mi = assoc_minute_indices[i];
                    let rec = &all_minutes[mi];
                    transitions.push(AssociatorTransition {
                        index: mi,
                        elapsed_hours: rec.elapsed_hours,
                        year: rec.year,
                        doy: rec.doy,
                        hour: rec.hour,
                        minute: rec.minute,
                        associator_norm: associators[i],
                        pre_mean,
                        post_mean,
                        transition_magnitude: jump,
                    });
                }
            }
        }
    }

    println!("  Found {} associator transitions", transitions.len());
    for t in &transitions {
        println!(
            "    DOY {} {:02}:{:02} UT  A={:.3}  jump={:.3} (pre={:.3}, post={:.3})",
            t.doy,
            t.hour,
            t.minute,
            t.associator_norm,
            t.transition_magnitude,
            t.pre_mean,
            t.post_mean
        );
    }

    // --- Step 5: Cross-compare crossings vs transitions ---
    println!(
        "[5/5] Cross-comparing {} crossings vs {} transitions...",
        crossings.len(),
        transitions.len()
    );

    let tol_hours = cli.match_tolerance_minutes as f64 / 60.0;
    let mut comparisons: Vec<CrossingComparison> = Vec::new();
    let mut matched_crossings = 0usize;
    let mut offsets: Vec<f64> = Vec::new();

    for c in &crossings {
        let best_match = transitions
            .iter()
            .enumerate()
            .filter(|(_, t)| (t.elapsed_hours - c.elapsed_hours).abs() < tol_hours)
            .min_by(|(_, a), (_, b)| {
                let da = (a.elapsed_hours - c.elapsed_hours).abs();
                let db = (b.elapsed_hours - c.elapsed_hours).abs();
                da.partial_cmp(&db).unwrap()
            });

        if let Some((_idx, t)) = best_match {
            let offset_min = (t.elapsed_hours - c.elapsed_hours) * 60.0;
            offsets.push(offset_min.abs());
            matched_crossings += 1;
            comparisons.push(CrossingComparison {
                crossing_index: c.index,
                crossing_hours: c.elapsed_hours,
                matched_transition_index: Some(t.index),
                matched_transition_hours: Some(t.elapsed_hours),
                offset_minutes: Some(offset_min),
            });
        } else {
            comparisons.push(CrossingComparison {
                crossing_index: c.index,
                crossing_hours: c.elapsed_hours,
                matched_transition_index: None,
                matched_transition_hours: None,
                offset_minutes: None,
            });
        }
    }

    // Count false alarms: transitions not near any crossing
    let matched_transitions: usize = transitions
        .iter()
        .filter(|t| {
            crossings
                .iter()
                .any(|c| (c.elapsed_hours - t.elapsed_hours).abs() < tol_hours)
        })
        .count();
    let false_alarms = transitions.len().saturating_sub(matched_transitions);

    let detection_rate = if crossings.is_empty() {
        0.0
    } else {
        matched_crossings as f64 / crossings.len() as f64
    };

    let false_alarm_rate = if transitions.is_empty() {
        0.0
    } else {
        false_alarms as f64 / transitions.len() as f64
    };

    offsets.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_offset = if offsets.is_empty() {
        0.0
    } else {
        offsets.iter().sum::<f64>() / offsets.len() as f64
    };
    let median_offset = if offsets.is_empty() {
        0.0
    } else {
        offsets[offsets.len() / 2]
    };

    println!("\n=== Results ===");
    println!("  Crossings detected (|B| gradient): {}", crossings.len());
    println!(
        "  Associator transitions:             {}",
        transitions.len()
    );
    println!(
        "  Detection rate:                     {}/{} = {:.1}%",
        matched_crossings,
        crossings.len(),
        detection_rate * 100.0
    );
    println!(
        "  False alarm rate:                   {}/{} = {:.1}%",
        false_alarms,
        transitions.len(),
        false_alarm_rate * 100.0
    );
    println!(
        "  Mean offset:                        {:.1} min",
        mean_offset
    );
    println!(
        "  Median offset:                      {:.1} min",
        median_offset
    );

    // --- Build hourly summary ---
    type HourlyAccum = (Vec<f64>, Vec<f64>);
    let mut hourly_map: std::collections::BTreeMap<(u16, u16, u8), HourlyAccum> =
        std::collections::BTreeMap::new();

    for (k, &norm) in associators.iter().enumerate() {
        let mi = assoc_minute_indices[k];
        let rec = &all_minutes[mi];
        let entry = hourly_map
            .entry((rec.year, rec.doy, rec.hour))
            .or_insert_with(|| (Vec::new(), Vec::new()));
        entry.0.push(norm);
        entry.1.push(rec.b_magnitude);
    }

    let hourly_summary: Vec<HourlySummary> = hourly_map
        .iter()
        .map(|(&(year, doy, hour), (assocs, bmags))| {
            let n = assocs.len();
            let mean_a = assocs.iter().sum::<f64>() / n as f64;
            let max_a = assocs.iter().cloned().fold(0.0f64, f64::max);
            let mean_b = bmags.iter().sum::<f64>() / n as f64;

            // Compute elapsed hours from first record
            let day_diff = (year as f64 - fy as f64) * 365.25 + (doy as f64 - fd as f64);
            let elapsed = day_diff * 24.0 + (hour as f64 - fh as f64);

            HourlySummary {
                elapsed_hours: elapsed,
                year,
                doy,
                hour,
                mean_associator: mean_a,
                max_associator: max_a,
                mean_bmag: mean_b,
                n_samples: n,
            }
        })
        .collect();

    // --- Write results ---
    let results = MultiDayResults {
        start_date: cli.start_date.clone(),
        n_days: cli.n_days,
        embedding_dim: cli.embedding_dim,
        takens_lag: cli.takens_lag,
        bmag_gradient_threshold: cli.bmag_gradient_threshold,
        rotation_threshold_deg: cli.rotation_threshold_deg,
        total_minutes: all_minutes.len(),
        n_crossings_detected: crossings.len(),
        n_associator_transitions: transitions.len(),
        crossings,
        transitions,
        comparisons,
        detection_rate,
        false_alarm_rate,
        mean_offset_minutes: mean_offset,
        median_offset_minutes: median_offset,
        hourly_associator_summary: hourly_summary,
    };

    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(&results)?;
    fs::write(&cli.out_json, &json)?;
    println!("\nWrote results to {}", cli.out_json.display());

    Ok(())
}

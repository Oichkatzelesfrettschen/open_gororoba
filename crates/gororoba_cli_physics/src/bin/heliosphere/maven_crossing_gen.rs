//! Generate a surrogate MAVEN bow shock crossing list from FGM regime transitions.
//!
//! MAVEN orbits Mars with a ~4.5-hour period, crossing the bow shock twice per orbit.
//! This binary detects transitions between the solar wind regime (|B| < SW_THRESH nT)
//! and the magnetosheath regime (|B| > SHEATH_THRESH nT) in minute-averaged FGM data,
//! applying a minimum suppression window to avoid duplicate detections during boundary
//! layer oscillations.
//!
//! Output: MESSENGER-compatible crossing list (YYYY-MM-DD/HH:MM:SS YYYY-MM-DD/HH:MM:SS)
//! readable by `data_core::crossing_lists::parse_crossing_list`.
//!
//! WHY: No public MAVEN bow shock crossing catalog is available in data/external/.
//! A regime-filter-derived list is a physically motivated surrogate: the solar wind
//! (~3-5 nT at 1.52 AU) and Mars magnetosheath (~8-20 nT) are separable by |B|.
//! Compared to pure gradient detection, the regime filter selects genuine outbound/
//! inbound bow shock traversals rather than internal sheath structures.

use anyhow::{Context, Result};
use chrono::{Datelike, Duration, NaiveDate};
use clap::Args;
use data_core::catalogs::maven_mag::parse_maven_mag_hapi_csv_minutes;
use std::{fs, path::PathBuf};

#[derive(Args)]
pub struct Cli {
    /// Start date (YYYY-MM-DD).
    #[arg(long, default_value = "2015-03-01")]
    start_date: String,

    /// End date (YYYY-MM-DD, inclusive).
    #[arg(long, default_value = "2015-03-13")]
    end_date: String,

    /// Directory containing maven_mag_{year}_{doy}.csv files.
    #[arg(long, default_value = "data/external/maven")]
    data_dir: PathBuf,

    /// |B| threshold (nT) for solar wind regime (below this = solar wind).
    #[arg(long, default_value_t = 4.5)]
    sw_threshold: f64,

    /// |B| threshold (nT) for magnetosheath regime (above this = sheath/magnetosphere).
    #[arg(long, default_value_t = 8.0)]
    sheath_threshold: f64,

    /// Rolling window (minutes) for regime classification.
    #[arg(long, default_value_t = 5)]
    rolling_window: usize,

    /// Minimum gap (minutes) between consecutive detected crossings.
    #[arg(long, default_value_t = 30)]
    min_gap_minutes: i64,

    /// Output crossing list file (MESSENGER format).
    #[arg(
        long,
        default_value = "data/external/crossing_lists/maven_bs_2015_regime.txt"
    )]
    out_crossing_list: PathBuf,

    /// Optional JSON summary output.
    #[arg(long)]
    out_json: Option<PathBuf>,
}

pub fn run(cli: Cli) -> Result<()> {
    let start = NaiveDate::parse_from_str(&cli.start_date, "%Y-%m-%d")
        .with_context(|| format!("invalid start_date: {}", cli.start_date))?;
    let end = NaiveDate::parse_from_str(&cli.end_date, "%Y-%m-%d")
        .with_context(|| format!("invalid end_date: {}", cli.end_date))?;

    // Load all minute records.
    let mut all_bmag: Vec<f64> = Vec::new();
    // Store (year, doy, hour, minute) per record for crossing time reconstruction.
    let mut all_keys: Vec<(i32, u32, u8, u8)> = Vec::new();

    let mut date = start;
    while date <= end {
        let fname = format!("maven_mag_{:04}_{:03}.csv", date.year(), date.ordinal());
        let path = cli.data_dir.join(&fname);
        if path.exists() {
            let content =
                fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
            let records = parse_maven_mag_hapi_csv_minutes(&content);
            println!("  DOY {:03}: {} minutes", date.ordinal(), records.len());
            for r in records {
                all_bmag.push(r.b_magnitude);
                all_keys.push((date.year(), date.ordinal(), r.hour, r.minute));
            }
        } else {
            println!("  DOY {:03}: no data ({})", date.ordinal(), path.display());
        }
        date += Duration::days(1);
    }

    if all_bmag.is_empty() {
        anyhow::bail!("No MAVEN data found in {}", cli.data_dir.display());
    }
    println!("Loaded {} minute records total", all_bmag.len());

    // Compute rolling mean |B| with the specified window.
    let hw = cli.rolling_window / 2;
    let n = all_bmag.len();
    let rolling: Vec<f64> = (0..n)
        .map(|i| {
            let lo = i.saturating_sub(hw);
            let hi = (i + hw + 1).min(n);
            let window = &all_bmag[lo..hi];
            window.iter().sum::<f64>() / window.len() as f64
        })
        .collect();

    // Classify each minute: 0=sw, 1=ambiguous, 2=sheath.
    let regime: Vec<u8> = rolling
        .iter()
        .map(|&b| {
            if b < cli.sw_threshold {
                0
            } else if b > cli.sheath_threshold {
                2
            } else {
                1
            }
        })
        .collect();

    // Detect transitions: sw->sheath (inbound) or sheath->sw (outbound).
    // A transition is detected when the regime changes from 0 to 2 (or 2 to 0)
    // looking at consecutive windows (ignoring ambiguous 1-regime minutes).
    let mut crossings: Vec<(i32, u32, u8, u8)> = Vec::new();
    let mut last_stable: u8 = regime[0];
    let mut last_crossing_idx: Option<usize> = None;

    for i in 1..n {
        let r = regime[i];
        if r == 1 {
            continue; // ambiguous, skip
        }
        if r != last_stable {
            // Check minimum gap.
            let too_close = if let Some(prev) = last_crossing_idx {
                (i as i64 - prev as i64) < cli.min_gap_minutes
            } else {
                false
            };
            if !too_close {
                crossings.push(all_keys[i]);
                last_crossing_idx = Some(i);
            }
            last_stable = r;
        }
    }

    println!("Detected {} regime crossings", crossings.len());

    // Write MESSENGER-format crossing list.
    if let Some(parent) = cli.out_crossing_list.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut lines: Vec<String> = Vec::new();
    lines.push("# MAVEN bow shock regime crossings generated from FGM minute data".to_string());
    lines.push(format!(
        "# SW threshold: {:.1} nT, Sheath threshold: {:.1} nT, Gap: {} min",
        cli.sw_threshold, cli.sheath_threshold, cli.min_gap_minutes
    ));
    lines.push(format!(
        "# Window: {} to {}, rolling: {} min",
        start, end, cli.rolling_window
    ));
    for &(year, doy, hour, minute) in &crossings {
        let date = NaiveDate::from_yo_opt(year, doy).unwrap();
        // MESSENGER format: start_time end_time (use 1-min window for crossing duration).
        let start_str = format!("{}/{:02}:{:02}:00", date.format("%Y-%m-%d"), hour, minute);
        let end_str = if minute < 59 {
            format!(
                "{}/{:02}:{:02}:00",
                date.format("%Y-%m-%d"),
                hour,
                minute + 1
            )
        } else {
            format!("{}/{:02}:00:00", date.format("%Y-%m-%d"), hour + 1)
        };
        lines.push(format!("{} {}", start_str, end_str));
    }

    let content = lines.join("\n") + "\n";
    fs::write(&cli.out_crossing_list, &content)?;
    println!(
        "Wrote {} to {}",
        crossings.len(),
        cli.out_crossing_list.display()
    );

    // Optional JSON summary.
    if let Some(ref json_path) = cli.out_json {
        let sw_minutes = regime.iter().filter(|&&r| r == 0).count();
        let sheath_minutes = regime.iter().filter(|&&r| r == 2).count();
        let ambig_minutes = regime.iter().filter(|&&r| r == 1).count();
        let summary = serde_json::json!({
            "start_date": cli.start_date,
            "end_date": cli.end_date,
            "total_minutes": n,
            "sw_minutes": sw_minutes,
            "sheath_minutes": sheath_minutes,
            "ambiguous_minutes": ambig_minutes,
            "sw_fraction": sw_minutes as f64 / n as f64,
            "sheath_fraction": sheath_minutes as f64 / n as f64,
            "n_crossings": crossings.len(),
            "sw_threshold_nt": cli.sw_threshold,
            "sheath_threshold_nt": cli.sheath_threshold,
            "min_gap_minutes": cli.min_gap_minutes,
            "rolling_window_minutes": cli.rolling_window,
        });
        if let Some(parent) = json_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(json_path, serde_json::to_string_pretty(&summary)?)?;
        println!("Wrote JSON summary to {}", json_path.display());
    }

    Ok(())
}

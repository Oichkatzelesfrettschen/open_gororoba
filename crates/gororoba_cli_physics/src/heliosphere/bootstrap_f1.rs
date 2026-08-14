//! Block-bootstrap 95% CI for the CD associator F1 on THEMIS E-237 data.
//!
//! WHY: The paper reports F1 = 0.601 as a scalar.  A confidence interval
//! (CI) is mandatory for a credible statistical claim.  Block bootstrap
//! preserves intra-block temporal autocorrelation (magnetopause crossings
//! cluster in short 10-30 min intervals) while resampling across the 7-day
//! series.  Plan P6A.S1, task 1.4.
//!
//! WHAT: Reads the THEMIS eval JSON produced by heliosphere-themis-staples-labeled,
//! reconstructs detection_unix and event_unix timestamp arrays, then calls
//! boundary_metrics::bootstrap_f1_ci_seeded with 1800s blocks, N=10_000,
//! seed=42.  Reports the 95% CI to stdout and writes a compact JSON result.
//!
//! HOW:
//!   cargo run --release -p gororoba_cli_physics \
//!     --bin heliosphere -- bootstrap-f1 \
//!     --eval-json data/output/heliosphere/ablations/themis_staples_labeled_eval.json

use anyhow::{Context, Result};
use clap::Args;
use serde::{Deserialize, Serialize};
use spectral_core::boundary_metrics;
use std::{fs, path::PathBuf};

#[derive(Args)]
pub struct Cli {
    /// Path to the JSON produced by heliosphere-themis-staples-labeled.
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/themis_staples_labeled_eval.json"
    )]
    eval_json: PathBuf,

    /// Block length in seconds (default 1800 = 30 min).
    #[arg(long, default_value_t = 1800_i64)]
    block_secs: i64,

    /// Number of bootstrap draws.
    #[arg(long, default_value_t = 10_000_usize)]
    n_bootstrap: usize,

    /// CI coverage (0–1).
    #[arg(long, default_value_t = 0.95_f64)]
    coverage: f64,

    /// RNG seed for reproducibility.
    #[arg(long, default_value_t = 42_u64)]
    seed: u64,

    /// Output JSON for CI result.
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/themis_bootstrap_f1_ci.json"
    )]
    out_json: PathBuf,
}

/// Minimal subset of the heliosphere-themis-staples-labeled JSON we need.
#[derive(Deserialize)]
struct EvalJson {
    /// CD firing times as elapsed hours from the series reference midnight.
    cd_fire_hours: Vec<f64>,
    /// Full catalog event summaries (used to derive event midpoint timestamps).
    catalog_events: Vec<CatalogEventJson>,
    /// Unix timestamp of the first minute of the THEMIS series.
    /// Absent in older runs; we compute from the series start_date if missing.
    #[serde(default)]
    series_start_unix: Option<i64>,
    #[serde(default)]
    series_end_unix: Option<i64>,
    /// Bootstrap CI already computed inline by staples_labeled (3-element array).
    #[serde(default)]
    cd_bootstrap_ci: Option<[f64; 3]>,
    start_date: String,
    n_days: u32,
}

#[derive(Deserialize)]
struct CatalogEventJson {
    timestamp: String,
    #[allow(dead_code)]
    interval_start: String,
    #[allow(dead_code)]
    interval_end: String,
}

#[derive(Serialize)]
struct BootstrapCiResult {
    eval_json: String,
    block_secs: i64,
    n_bootstrap: usize,
    coverage: f64,
    seed: u64,
    f1_mean: f64,
    f1_ci_lo: f64,
    f1_ci_hi: f64,
    n_detections: usize,
    n_events: usize,
}

pub fn run(cli: Cli) -> Result<()> {

    let raw = fs::read_to_string(&cli.eval_json)
        .with_context(|| format!("reading {}", cli.eval_json.display()))?;
    let eval: EvalJson = serde_json::from_str(&raw)
        .with_context(|| format!("parsing {}", cli.eval_json.display()))?;

    if let Some(ci) = eval.cd_bootstrap_ci {
        println!(
            "Note: eval JSON already contains inline bootstrap CI: [{:.3}, {:.3}, {:.3}]",
            ci[0], ci[1], ci[2]
        );
        println!("Re-running with explicit parameters for verification.");
    }

    // Derive series start/end Unix timestamps.
    let series_start_unix = if let Some(s) = eval.series_start_unix {
        s
    } else {
        // Parse start_date to UTC midnight.
        let d = chrono::NaiveDate::parse_from_str(&eval.start_date, "%Y-%m-%d")
            .with_context(|| format!("parsing start_date '{}'", eval.start_date))?;
        use chrono::TimeZone;
        chrono::Utc
            .from_utc_datetime(&d.and_hms_opt(0, 0, 0).unwrap())
            .timestamp()
    };

    let series_end_unix = if let Some(e) = eval.series_end_unix {
        e
    } else {
        series_start_unix + eval.n_days as i64 * 86400
    };

    // Build detection Unix timestamps.
    let detection_unix: Vec<i64> = eval
        .cd_fire_hours
        .iter()
        .map(|&h| series_start_unix + (h * 3600.0) as i64)
        .collect();

    // Build event Unix timestamps from catalog midpoints.
    let event_unix: Vec<i64> = eval
        .catalog_events
        .iter()
        .map(|ev| {
            chrono::NaiveDateTime::parse_from_str(&ev.timestamp, "%Y-%m-%dT%H:%M:%S")
                .map(|dt| {
                    use chrono::TimeZone;
                    chrono::Utc.from_utc_datetime(&dt).timestamp()
                })
                .unwrap_or(series_start_unix)
        })
        .collect();

    // Matching window: 10 minutes half-width (matches staples binary default pad_minutes=10).
    let window_secs: i64 = 10 * 60;

    println!(
        "Bootstrap F1 CI\n\
         ===============\n\
         Detections:  {}\n\
         Events:      {}\n\
         Series:      {} s ({:.1} days)\n\
         Block:       {} s ({} min)\n\
         Draws:       {}\n\
         Coverage:    {:.0}%\n\
         Seed:        {}",
        detection_unix.len(),
        event_unix.len(),
        series_end_unix - series_start_unix,
        (series_end_unix - series_start_unix) as f64 / 86400.0,
        cli.block_secs,
        cli.block_secs / 60,
        cli.n_bootstrap,
        cli.coverage * 100.0,
        cli.seed,
    );

    let (f1_mean, f1_lo, f1_hi) = boundary_metrics::bootstrap_f1_ci_seeded(
        &detection_unix,
        &event_unix,
        series_start_unix,
        series_end_unix,
        cli.block_secs,
        cli.n_bootstrap,
        cli.coverage,
        window_secs,
        cli.seed,
    );

    println!(
        "\nCD F1 bootstrap CI ({:.0}%): {:.3} [{:.3}, {:.3}]",
        cli.coverage * 100.0,
        f1_mean,
        f1_lo,
        f1_hi
    );

    let result = BootstrapCiResult {
        eval_json: cli.eval_json.display().to_string(),
        block_secs: cli.block_secs,
        n_bootstrap: cli.n_bootstrap,
        coverage: cli.coverage,
        seed: cli.seed,
        f1_mean,
        f1_ci_lo: f1_lo,
        f1_ci_hi: f1_hi,
        n_detections: detection_unix.len(),
        n_events: event_unix.len(),
    };

    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.out_json, serde_json::to_string_pretty(&result)?)?;
    println!("Wrote {}", cli.out_json.display());

    Ok(())
}

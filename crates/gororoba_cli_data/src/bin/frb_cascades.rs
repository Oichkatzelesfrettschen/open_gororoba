//! Direction 3: Repeating FRB Temporal Cascades
//!
//! Analyzes temporal structure of repeating FRBs to detect SOC
//! (Self-Organized Criticality) patterns and ultrametric hierarchy
//! in waiting-time distributions.
//!
//! # Method
//!
//! For each repeater with >= min_bursts events:
//! 1. Extract burst timestamps and fluences
//! 2. Compute waiting-time statistics
//! 3. Estimate Hurst exponent (persistence in temporal correlations)
//! 4. Build distance matrix in (log_dt, log_energy) space
//! 5. Run ultrametric fraction test
//! 6. Report per-source and aggregate results
//!
//! # Usage
//!
//! frb-cascades --input data/external/chime_frb_cat2.csv \
//!              --min-bursts 5 \
//!              --output data/csv/c071d_frb_cascades_ultrametric.csv

use clap::Parser;
use std::path::PathBuf;

use data_core::catalogs::chime::{extract_repeaters, parse_chime_csv};
use stats_core::claims_gates::Verdict;
use stats_core::ultrametric::temporal::analyze_temporal_cascade;

#[derive(Parser)]
#[command(name = "frb-cascades")]
#[command(about = "Direction 3: Test ultrametric temporal cascades in repeating FRBs")]
struct Cli {
    /// Path to CHIME FRB Cat 2 CSV.
    #[arg(long, default_value = "data/external/chime_frb_cat2.csv")]
    input: PathBuf,

    /// Minimum number of bursts per repeater to include in analysis.
    #[arg(long, default_value = "5")]
    min_bursts: usize,

    /// Number of triples for ultrametric fraction test.
    #[arg(long, default_value = "50000")]
    n_triples: usize,

    /// Number of permutations for null distribution.
    #[arg(long, default_value = "200")]
    n_permutations: usize,

    /// Output CSV path.
    #[arg(long, default_value = "data/csv/c071d_frb_cascades_ultrametric.csv")]
    output: PathBuf,
}

fn main() {
    let cli = Cli::parse();

    eprintln!("=== Direction 3: Repeating FRB Temporal Cascades ===");
    eprintln!("Input: {}", cli.input.display());

    // 1. Load CHIME Cat 2
    let events = parse_chime_csv(&cli.input).unwrap_or_else(|e| {
        eprintln!("Failed to parse CHIME CSV: {}", e);
        std::process::exit(1);
    });

    eprintln!("Loaded {} total events", events.len());

    // 2. Extract repeaters
    let repeaters = extract_repeaters(&events);
    eprintln!("Found {} repeater sources", repeaters.len());

    // Filter to repeaters with enough bursts
    let mut qualified: Vec<&(String, Vec<&data_core::catalogs::chime::FrbEvent>)> = repeaters
        .iter()
        .filter(|(_, bursts)| bursts.len() >= cli.min_bursts)
        .collect();

    // Sort by burst count (descending) for reporting
    qualified.sort_by_key(|b| std::cmp::Reverse(b.1.len()));

    eprintln!(
        "Repeaters with >= {} bursts: {}",
        cli.min_bursts,
        qualified.len()
    );

    if qualified.is_empty() {
        eprintln!("No qualifying repeaters found");
        std::process::exit(1);
    }

    // 3. Analyze each repeater
    if let Some(parent) = cli.output.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let mut wtr = csv::Writer::from_path(&cli.output).unwrap();
    wtr.write_record([
        "source_id",
        "n_bursts",
        "n_intervals",
        "mean_dt_days",
        "median_dt_days",
        "cv",
        "hurst_exponent",
        "ultrametric_fraction",
        "null_fraction_mean",
        "p_value",
        "verdict",
    ])
    .unwrap();

    let mut n_pass = 0;
    let mut n_fail = 0;
    let mut n_analyzed = 0;

    for (source_id, bursts) in &qualified {
        // Extract sorted timestamps (MJD) and fluences
        let mut timestamps: Vec<f64> = bursts
            .iter()
            .filter(|b| !b.mjd_400.is_nan() && b.mjd_400 > 0.0)
            .map(|b| b.mjd_400)
            .collect();

        timestamps.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap());

        if timestamps.len() < 3 {
            continue;
        }

        let fluences: Vec<f64> = bursts
            .iter()
            .filter(|b| !b.mjd_400.is_nan() && b.mjd_400 > 0.0)
            .map(|b| {
                if b.fluence > 0.0 && !b.fluence.is_nan() {
                    b.fluence
                } else {
                    1.0 // Default fluence if missing
                }
            })
            .collect();

        let result = analyze_temporal_cascade(
            source_id,
            &timestamps,
            &fluences,
            cli.n_triples.min(timestamps.len() * timestamps.len()),
            cli.n_permutations,
            42,
        );

        let wt = &result.waiting_time_stats;

        wtr.write_record([
            source_id.as_str(),
            &result.n_bursts.to_string(),
            &wt.n_intervals.to_string(),
            &format!("{:.4}", wt.mean_dt),
            &format!("{:.4}", wt.median_dt),
            &format!("{:.4}", wt.cv),
            &format!("{:.4}", result.hurst_exponent),
            &format!("{:.4}", result.ultrametric_fraction),
            &format!("{:.4}", result.null_fraction_mean),
            &format!("{:.4}", result.p_value),
            &format!("{:?}", result.verdict),
        ])
        .unwrap();

        n_analyzed += 1;

        match result.verdict {
            Verdict::Pass => {
                n_pass += 1;
                eprintln!(
                    "  {} ({} bursts): PASS (H={:.3}, frac={:.3}, p={:.4})",
                    source_id,
                    result.n_bursts,
                    result.hurst_exponent,
                    result.ultrametric_fraction,
                    result.p_value,
                );
            }
            Verdict::Fail => {
                n_fail += 1;
                if result.n_bursts >= 20 {
                    eprintln!(
                        "  {} ({} bursts): FAIL (H={:.3}, frac={:.3}, p={:.4})",
                        source_id,
                        result.n_bursts,
                        result.hurst_exponent,
                        result.ultrametric_fraction,
                        result.p_value,
                    );
                }
            }
            _ => {}
        }
    }

    wtr.flush().unwrap();

    // 4. Summary
    eprintln!("\n=== Summary ===");
    eprintln!("Repeaters analyzed: {}", n_analyzed);
    eprintln!("Pass (ultrametric signal): {}", n_pass);
    eprintln!("Fail (no signal): {}", n_fail);
    eprintln!(
        "Rate: {:.1}% ({}/{})",
        if n_analyzed > 0 {
            100.0 * n_pass as f64 / n_analyzed as f64
        } else {
            0.0
        },
        n_pass,
        n_analyzed,
    );

    // Gate verdict for C-071d: pass if a meaningful fraction show signal
    let gate_verdict = if n_pass > 0 && n_pass as f64 / n_analyzed as f64 > 0.1 {
        Verdict::Pass
    } else {
        Verdict::Fail
    };

    eprintln!("\n=== C-071d Gate: {:?} ===", gate_verdict);
    eprintln!("Results written to {}", cli.output.display());
}

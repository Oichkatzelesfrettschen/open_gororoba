//! GWTC-3 mass clumping analysis via Hartigan dip test.
//!
//! Tests whether the BBH primary mass distribution (mass_1_source) exhibits
//! multimodality, as predicted by pair-instability supernova theory (the
//! "lower mass gap" around 5-15 Msun) and hierarchical merger channels
//! (peak near 35 Msun).
//!
//! # Method
//!
//! 1. Load GWTC events from CSV (combined or GWTC-3 confident).
//! 2. Extract mass_1_source for events with valid (> 0) mass values.
//! 3. Run Hartigan dip test (Hartigan & Hartigan, 1985) with permutation p-value.
//! 4. Optionally repeat for mass_2_source and chirp_mass_source.
//! 5. Output results to CSV.
//!
//! # Pre-registered protocol
//!
//! Per claim C-007, the analysis requires N >= 50 events.  The binary aborts
//! if fewer events are available after filtering.
//!
//! # Usage
//!
//! ```text
//! mass-clumping --input data/external/gwosc_all_events.csv \
//!               --output data/csv/gwtc3_mass_clumping_dip.csv \
//!               --n-permutations 10000 --seed 42
//! ```

use clap::Parser;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::path::PathBuf;

use data_core::catalogs::gwtc::parse_gwtc3_csv;
use stats_core::dip::hartigan_dip_test;

const MIN_EVENTS: usize = 50;

#[derive(Parser)]
#[command(name = "mass-clumping")]
#[command(about = "Test mass distribution multimodality via Hartigan dip test")]
struct Cli {
    /// Path to GWTC events CSV (combined or GWTC-3 confident).
    /// Defaults to the combined catalog if available, otherwise GWTC-3.
    #[arg(long)]
    input: Option<PathBuf>,

    /// Number of permutations for p-value estimation.
    #[arg(long, default_value = "10000")]
    n_permutations: usize,

    /// RNG seed for reproducibility.
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Output CSV path.
    #[arg(long, default_value = "data/csv/gwtc3_mass_clumping_dip.csv")]
    output: PathBuf,
}

/// Extract a named mass column, filtering out zero/NaN values.
fn extract_mass_column(
    events: &[data_core::catalogs::gwtc::GwEvent],
    accessor: fn(&data_core::catalogs::gwtc::GwEvent) -> f64,
    name: &str,
) -> Vec<f64> {
    let values: Vec<f64> = events
        .iter()
        .map(accessor)
        .filter(|&m| m > 0.0 && m.is_finite())
        .collect();
    eprintln!("  {}: {} valid values", name, values.len());
    values
}

fn main() {
    let cli = Cli::parse();

    eprintln!("=== GWTC Mass Clumping: Hartigan Dip Test ===");

    // Resolve input path
    let input = cli.input.unwrap_or_else(|| {
        let combined = PathBuf::from("data/external/gwosc_all_events.csv");
        if combined.exists() {
            eprintln!("Using combined GWOSC catalog");
            combined
        } else {
            eprintln!("Combined catalog not found, falling back to GWTC-3 confident");
            PathBuf::from("data/external/GWTC-3_confident.csv")
        }
    });
    eprintln!("Input: {}", input.display());

    // 1. Load events
    let events = parse_gwtc3_csv(&input).unwrap_or_else(|e| {
        eprintln!("Failed to parse CSV: {}", e);
        std::process::exit(1);
    });
    eprintln!("Loaded {} events", events.len());

    if events.len() < MIN_EVENTS {
        eprintln!(
            "ABORT: Only {} events, need >= {} (pre-registered minimum from C-007 protocol)",
            events.len(),
            MIN_EVENTS,
        );
        std::process::exit(1);
    }

    // 2. Extract mass columns
    let columns: Vec<(&str, Vec<f64>)> = vec![
        ("mass_1_source", extract_mass_column(&events, |e| e.mass_1_source, "mass_1_source")),
        ("mass_2_source", extract_mass_column(&events, |e| e.mass_2_source, "mass_2_source")),
        ("chirp_mass_source", extract_mass_column(&events, |e| e.chirp_mass_source, "chirp_mass_source")),
    ];

    // 3. Run dip tests
    let mut rng = ChaCha8Rng::seed_from_u64(cli.seed);
    let mut results: Vec<(String, usize, f64, f64, usize)> = Vec::new();

    for (name, values) in &columns {
        if values.len() < MIN_EVENTS {
            eprintln!(
                "  SKIP {}: only {} valid values (need >= {})",
                name,
                values.len(),
                MIN_EVENTS,
            );
            continue;
        }

        eprintln!("  Running dip test on {} (N={}, {} permutations)...",
            name, values.len(), cli.n_permutations);

        let result = hartigan_dip_test(values, cli.n_permutations, &mut rng);

        let verdict = if result.p_value < 0.05 {
            "MULTIMODAL"
        } else {
            "UNIMODAL"
        };

        eprintln!(
            "    D_n = {:.6}, p = {:.6} -> {}",
            result.dip_statistic, result.p_value, verdict,
        );

        results.push((
            name.to_string(),
            values.len(),
            result.dip_statistic,
            result.p_value,
            result.n_permutations,
        ));
    }

    // 4. Write output CSV
    if let Some(parent) = cli.output.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let mut wtr = csv::Writer::from_path(&cli.output).unwrap_or_else(|e| {
        eprintln!("Failed to create output CSV: {}", e);
        std::process::exit(1);
    });

    wtr.write_record(["column", "n_events", "dip_statistic", "p_value", "n_permutations", "verdict"])
        .expect("Failed to write CSV header");

    for (name, n, dip, p, n_perm) in &results {
        let verdict = if *p < 0.05 { "MULTIMODAL" } else { "UNIMODAL" };
        wtr.write_record([
            name.as_str(),
            &n.to_string(),
            &format!("{:.8}", dip),
            &format!("{:.8}", p),
            &n_perm.to_string(),
            verdict,
        ])
        .expect("Failed to write CSV row");
    }

    wtr.flush().expect("Failed to flush CSV");
    eprintln!("Output: {}", cli.output.display());
    eprintln!("Done.");
}

//! Direction 1: FRB DM -> Comoving Distance + Local Ultrametricity
//!
//! Transforms FRB dispersion measures into 3D comoving positions and
//! tests for local ultrametric structure using the Bradley (2025) method.
//!
//! # Method
//!
//! 1. Read CHIME FRB catalog, extract DM excess and sky positions
//! 2. Transform: DM -> redshift (Macquart) -> comoving distance
//! 3. Convert (RA, Dec, d_C) to Cartesian (x, y, z) in Mpc
//! 4. Run Bradley local ultrametricity test at multiple epsilon values
//! 5. Run global ultrametric fraction on comoving distances
//! 6. Compare to Poisson null in the same comoving volume
//!
//! # Usage
//!
//! dm-ultrametric --input data/external/chime_frb_cat2.csv \
//!                --output data/csv/c071b_dm_comoving_ultrametric.csv

use clap::Parser;
use std::path::PathBuf;

use cosmology_core::distances::{
    comoving_distance, dm_excess_to_redshift, planck2018, radec_to_cartesian,
};
use data_core::catalogs::chime::parse_chime_csv;
use stats_core::ultrametric::dendrogram::{
    euclidean_distance_matrix_3d, hierarchical_ultrametric_test,
};
use stats_core::ultrametric::local::local_ultrametricity_test;

#[derive(Parser)]
#[command(name = "dm-ultrametric")]
#[command(about = "Direction 1: Test local ultrametric structure in FRB comoving positions")]
struct Cli {
    /// Path to CHIME FRB CSV.
    #[arg(long, default_value = "data/external/chime_frb_cat2.csv")]
    input: PathBuf,

    /// DM column to use for excess (after MW subtraction).
    #[arg(long, default_value = "dm_exc_ne2001")]
    dm_column: String,

    /// Median host galaxy DM contribution to subtract (pc/cm^3).
    #[arg(long, default_value = "50.0")]
    dm_host: f64,

    /// Epsilon values for local ultrametricity test (Mpc, comma-separated).
    #[arg(long, default_value = "50,100,200,500")]
    epsilons: String,

    /// Number of samples per neighborhood for local test.
    #[arg(long, default_value = "1000")]
    n_samples: usize,

    /// Number of permutations for null distribution.
    #[arg(long, default_value = "100")]
    n_permutations: usize,

    /// Output CSV path.
    #[arg(long, default_value = "data/csv/c071b_dm_comoving_ultrametric.csv")]
    output: PathBuf,
}

fn main() {
    let cli = Cli::parse();

    eprintln!("=== Direction 1: FRB DM -> Comoving + Local Ultrametricity ===");
    eprintln!("Input: {}", cli.input.display());

    // 1. Load CHIME events
    let events = parse_chime_csv(&cli.input).unwrap_or_else(|e| {
        eprintln!("Failed to parse CHIME CSV: {}", e);
        std::process::exit(1);
    });

    eprintln!("Loaded {} total FRB events", events.len());

    // 2. Extract valid events with DM excess, RA, Dec
    let omega_m = planck2018::OMEGA_M;
    let omega_b = planck2018::OMEGA_B;
    let h0 = planck2018::H0;

    let mut coords_3d: Vec<(f64, f64, f64)> = Vec::new();
    let mut redshifts: Vec<f64> = Vec::new();
    let mut comoving_dists: Vec<f64> = Vec::new();

    for event in &events {
        let dm_exc = event.dm_exc_ne2001;
        if dm_exc.is_nan() || dm_exc <= 0.0 {
            continue;
        }
        if event.ra.is_nan() || event.dec.is_nan() {
            continue;
        }

        // Subtract host DM
        let dm_cosmic = dm_exc - cli.dm_host;
        if dm_cosmic <= 0.0 {
            continue;
        }

        // DM -> redshift -> comoving distance
        let z = dm_excess_to_redshift(dm_cosmic, omega_m, omega_b, h0);
        let d_c = comoving_distance(z, omega_m, h0);

        // (RA, Dec, d_C) -> (x, y, z) in Mpc
        let (x, y, z_coord) = radec_to_cartesian(event.ra, event.dec, d_c);

        coords_3d.push((x, y, z_coord));
        redshifts.push(z);
        comoving_dists.push(d_c);
    }

    eprintln!("Valid events with DM + position: {}", coords_3d.len());

    if coords_3d.len() < 10 {
        eprintln!("Too few valid events for analysis");
        std::process::exit(1);
    }

    eprintln!(
        "Redshift range: [{:.4}, {:.4}]",
        redshifts.iter().cloned().fold(f64::INFINITY, f64::min),
        redshifts.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );
    eprintln!(
        "Comoving distance range: [{:.0}, {:.0}] Mpc",
        comoving_dists.iter().cloned().fold(f64::INFINITY, f64::min),
        comoving_dists
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max),
    );

    // 3. Parse epsilon values
    let epsilons: Vec<f64> = cli
        .epsilons
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    // 4. Local ultrametricity at each epsilon
    if let Some(parent) = cli.output.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let mut wtr = csv::Writer::from_path(&cli.output).unwrap();
    wtr.write_record([
        "test_type",
        "epsilon_mpc",
        "n_points",
        "n_testable",
        "metric_value",
        "null_mean",
        "p_value",
        "verdict",
    ])
    .unwrap();

    eprintln!("\n--- Local Ultrametricity Tests ---");

    for &eps in &epsilons {
        eprintln!("  epsilon = {} Mpc ...", eps);

        let result =
            local_ultrametricity_test(&coords_3d, eps, cli.n_samples, cli.n_permutations, 42);

        eprintln!(
            "    testable: {}, mean_idx: {:.4}, null: {:.4}, p={:.4}, {:?}",
            result.n_testable,
            result.mean_local_index,
            result.null_mean_index,
            result.p_value,
            result.verdict,
        );

        wtr.write_record([
            "local_ultrametricity",
            &format!("{:.0}", eps),
            &result.n_points.to_string(),
            &result.n_testable.to_string(),
            &format!("{:.6}", result.mean_local_index),
            &format!("{:.6}", result.null_mean_index),
            &format!("{:.6}", result.p_value),
            &format!("{:?}", result.verdict),
        ])
        .unwrap();
    }

    // 5. Global dendrogram analysis (on a subsample if too many points)
    eprintln!("\n--- Dendrogram Analysis ---");

    let max_dend_points = 200; // O(n^2) memory, keep manageable
    let dend_coords: Vec<(f64, f64, f64)> = if coords_3d.len() > max_dend_points {
        // Take evenly spaced subsample
        let step = coords_3d.len() / max_dend_points;
        coords_3d
            .iter()
            .step_by(step)
            .take(max_dend_points)
            .cloned()
            .collect()
    } else {
        coords_3d.clone()
    };

    let dist_matrix = euclidean_distance_matrix_3d(&dend_coords);
    let dend_result =
        hierarchical_ultrametric_test(&dist_matrix, dend_coords.len(), cli.n_permutations, 42);

    eprintln!(
        "  Cophenetic correlation: {:.4} (null: {:.4} +/- {:.4})",
        dend_result.cophenetic_correlation,
        dend_result.null_cophenetic_mean,
        dend_result.null_cophenetic_std,
    );
    eprintln!("  p-value: {:.4}", dend_result.p_value);
    eprintln!("  Verdict: {:?}", dend_result.verdict);

    wtr.write_record([
        "dendrogram_cophenetic",
        "global",
        &dend_coords.len().to_string(),
        &dend_coords.len().to_string(),
        &format!("{:.6}", dend_result.cophenetic_correlation),
        &format!("{:.6}", dend_result.null_cophenetic_mean),
        &format!("{:.6}", dend_result.p_value),
        &format!("{:?}", dend_result.verdict),
    ])
    .unwrap();

    wtr.flush().unwrap();

    eprintln!("\nResults written to {}", cli.output.display());
}

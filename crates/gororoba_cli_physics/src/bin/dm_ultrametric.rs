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

use gororoba_cli_physics::frb_distances::{
    DmExcessColumn, RELABELING_DIAGNOSTIC, load_frb_distances,
};
use stats_core::ultrametric::{
    dendrogram::{euclidean_distance_matrix_3d, hierarchical_ultrametric_test},
    local::local_ultrametricity_test,
};

#[derive(Parser)]
#[command(name = "dm-ultrametric")]
#[command(about = "Direction 1: Test local ultrametric structure in FRB comoving positions")]
struct Cli {
    /// Path to CHIME FRB CSV.
    #[arg(long, default_value = "data/external/chime_frb_cat2.csv")]
    input: PathBuf,

    /// DM column to use for excess (after MW subtraction).
    #[arg(long, value_enum, default_value = "dm_exc_ne2001")]
    dm_column: DmExcessColumn,

    /// Fixed observer-frame host DM to subtract (pc/cm^3), already redshift-diluted.
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

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    eprintln!("=== Direction 1: FRB DM -> Comoving + Local Ultrametricity ===");
    eprintln!("Input: {}", cli.input.display());

    let sample = load_frb_distances(&cli.input, cli.dm_column, cli.dm_host)?;
    eprintln!(
        "DM column: {}; observer-frame host: {} pc/cm^3",
        cli.dm_column.header(),
        cli.dm_host
    );
    eprintln!(
        "Parsed positive-bonsai rows: {}; excluded excess: {}; excluded sky: {}; nonpositive cosmic DM: {}",
        sample.parsed_rows,
        sample.invalid_excess,
        sample.invalid_sky_position,
        sample.nonpositive_cosmic_dm
    );
    let coords_3d: Vec<_> = sample
        .distances
        .iter()
        .map(|point| point.cartesian_mpc)
        .collect();
    let redshifts: Vec<_> = sample
        .distances
        .iter()
        .map(|point| point.redshift)
        .collect();
    let comoving_dists: Vec<_> = sample
        .distances
        .iter()
        .map(|point| point.comoving_mpc)
        .collect();

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
            RELABELING_DIAGNOSTIC,
        );

        wtr.write_record([
            "local_ultrametricity",
            &format!("{:.0}", eps),
            &result.n_points.to_string(),
            &result.n_testable.to_string(),
            &format!("{:.6}", result.mean_local_index),
            &format!("{:.6}", result.null_mean_index),
            &format!("{:.6}", result.p_value),
            RELABELING_DIAGNOSTIC,
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
    eprintln!("  Diagnostic status: {RELABELING_DIAGNOSTIC}");

    wtr.write_record([
        "dendrogram_cophenetic",
        "global",
        &dend_coords.len().to_string(),
        &dend_coords.len().to_string(),
        &format!("{:.6}", dend_result.cophenetic_correlation),
        &format!("{:.6}", dend_result.null_cophenetic_mean),
        &format!("{:.6}", dend_result.p_value),
        RELABELING_DIAGNOSTIC,
    ])
    .unwrap();

    wtr.flush().unwrap();

    eprintln!("\nResults written to {}", cli.output.display());
    Ok(())
}

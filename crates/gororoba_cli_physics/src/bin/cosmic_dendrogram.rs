//! Direction 5: Cross-Dataset Cosmic Dendrogram
//!
//! Tests whether the combined 3D positions of objects from multiple
//! cosmological catalogs exhibit ultrametric (tree-like) structure
//! via hierarchical clustering and cophenetic correlation.
//!
//! # Method
//!
//! 1. Load FRBs (CHIME), GW events (GWTC-3)
//! 2. Assign 3D comoving positions where possible
//! 3. Build unified distance matrix (3D Euclidean in comoving Mpc)
//! 4. Single-linkage hierarchical clustering -> dendrogram
//! 5. Compute cophenetic correlation
//! 6. Run ultrametric fraction test
//! 7. Compare to Poisson process in same comoving volume
//!
//! # Usage
//!
//! cosmic-dendrogram --frbs data/external/chime_frb_cat2.csv \
//!                   --gws data/external/GWTC-3_confident.csv \
//!                   --output data/csv/c071f_cosmic_dendrogram.csv

use clap::Parser;
use std::path::PathBuf;

use cosmology_core::distances::{comoving_distance, planck2018, radec_to_cartesian};
use data_core::catalogs::gwtc::parse_gwtc3_csv;
use gororoba_cli_physics::frb_distances::{
    DmExcessColumn, RELABELING_DIAGNOSTIC, load_frb_distances,
};
use stats_core::ultrametric::dendrogram::{
    euclidean_distance_matrix_3d, hierarchical_ultrametric_test,
};

#[derive(Parser)]
#[command(name = "cosmic-dendrogram")]
#[command(about = "Direction 5: Cross-dataset dendrogram ultrametricity test")]
struct Cli {
    /// Path to CHIME FRB CSV.
    #[arg(long, default_value = "data/external/chime_frb_cat2.csv")]
    frbs: PathBuf,

    /// Path to GWTC-3 confident events CSV.
    #[arg(long, default_value = "data/external/GWTC-3_confident.csv")]
    gws: PathBuf,

    /// Fixed observer-frame host DM to subtract (pc/cm^3), already redshift-diluted.
    #[arg(long, default_value = "50.0")]
    dm_host: f64,

    /// Galactic-model excess column used for the FRB distance conversion.
    #[arg(long, value_enum, default_value = "dm_exc_ne2001")]
    dm_column: DmExcessColumn,

    /// Maximum number of points for dendrogram (O(n^2) memory).
    #[arg(long, default_value = "300")]
    max_points: usize,

    /// Number of permutations for null distribution.
    #[arg(long, default_value = "200")]
    n_permutations: usize,

    /// Output CSV path.
    #[arg(long, default_value = "data/csv/c071f_cosmic_dendrogram.csv")]
    output: PathBuf,
}

/// Object with 3D comoving position and source catalog label.
struct CosmicObject {
    x: f64,
    y: f64,
    z: f64,
    catalog: String,
    _name: String,
    redshift: f64,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    eprintln!("=== Direction 5: Cross-Dataset Cosmic Dendrogram ===");

    let omega_m = planck2018::OMEGA_M;
    let h0 = planck2018::H0;

    let mut objects: Vec<CosmicObject> = Vec::new();

    eprintln!("Loading FRBs...");
    let sample = load_frb_distances(&cli.frbs, cli.dm_column, cli.dm_host)?;
    eprintln!(
        "DM column: {}; parsed positive-bonsai rows: {}; excluded excess: {}; excluded sky: {}; nonpositive cosmic DM: {}",
        cli.dm_column.header(),
        sample.parsed_rows,
        sample.invalid_excess,
        sample.invalid_sky_position,
        sample.nonpositive_cosmic_dm
    );
    for point in sample.distances {
        objects.push(CosmicObject {
            x: point.cartesian_mpc.0,
            y: point.cartesian_mpc.1,
            z: point.cartesian_mpc.2,
            catalog: "FRB".into(),
            _name: point.name,
            redshift: point.redshift,
        });
    }
    eprintln!("  FRBs with comoving positions: {}", objects.len());

    // 2. Load GW events
    // GWTC-3 does not include RA/Dec in the standard confident events release.
    // We place GW events at comoving distance along random directions seeded
    // deterministically from the event name. This preserves the radial
    // distribution while acknowledging we lack sky localization.
    let n_frbs = objects.len();
    eprintln!("Loading GW events...");
    if let Ok(gw_events) = parse_gwtc3_csv(&cli.gws) {
        use rand::{RngExt, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        use std::{
            collections::hash_map::DefaultHasher,
            hash::{Hash, Hasher},
        };

        for event in &gw_events {
            if event.redshift <= 0.0 || event.redshift.is_nan() {
                continue;
            }

            let d_c = comoving_distance(event.redshift, omega_m, h0);

            // Deterministic random sky position from event name
            let mut hasher = DefaultHasher::new();
            event.common_name.hash(&mut hasher);
            let seed = hasher.finish();
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let ra_deg: f64 = rng.random_range(0.0..360.0);
            let dec_deg: f64 = rng.random_range(-90.0..90.0);

            let (x, y, z_c) = radec_to_cartesian(ra_deg, dec_deg, d_c);

            objects.push(CosmicObject {
                x,
                y,
                z: z_c,
                catalog: "GW".into(),
                _name: event.common_name.clone(),
                redshift: event.redshift,
            });
        }
        let n_gws = objects.len() - n_frbs;
        eprintln!(
            "  GW events with comoving positions: {} (random sky positions)",
            n_gws
        );
    }

    eprintln!("Total objects: {}", objects.len());

    if objects.len() < 10 {
        eprintln!("Too few objects for analysis");
        std::process::exit(1);
    }

    // 3. Subsample if needed
    let coords: Vec<(f64, f64, f64)> = if objects.len() > cli.max_points {
        eprintln!("Subsampling to {} points", cli.max_points);
        let step = objects.len() / cli.max_points;
        objects
            .iter()
            .step_by(step)
            .take(cli.max_points)
            .map(|o| (o.x, o.y, o.z))
            .collect()
    } else {
        objects.iter().map(|o| (o.x, o.y, o.z)).collect()
    };

    let n = coords.len();
    eprintln!("Analyzing {} objects", n);

    // 4. Build distance matrix and run dendrogram test
    eprintln!("\nBuilding distance matrix and running dendrogram analysis...");
    let dist_matrix = euclidean_distance_matrix_3d(&coords);

    let result = hierarchical_ultrametric_test(&dist_matrix, n, cli.n_permutations, 42);

    eprintln!(
        "Cophenetic correlation: {:.4} (null: {:.4} +/- {:.4})",
        result.cophenetic_correlation, result.null_cophenetic_mean, result.null_cophenetic_std,
    );
    eprintln!("Ultrametric fraction: {:.4}", result.ultrametric_fraction);
    eprintln!("P-value: {:.4}", result.p_value);
    eprintln!("Diagnostic status: {RELABELING_DIAGNOSTIC}");

    // 5. Catalog breakdown
    let n_frb_final = objects.iter().filter(|o| o.catalog == "FRB").count();
    let n_gw_final = objects.iter().filter(|o| o.catalog == "GW").count();

    let z_range = objects
        .iter()
        .map(|o| o.redshift)
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), z| {
            (lo.min(z), hi.max(z))
        });

    // 6. Write results
    if let Some(parent) = cli.output.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let mut wtr = csv::Writer::from_path(&cli.output).unwrap();
    wtr.write_record([
        "metric",
        "value",
        "null_mean",
        "null_std",
        "p_value",
        "verdict",
    ])
    .unwrap();

    wtr.write_record([
        "cophenetic_correlation",
        &format!("{:.6}", result.cophenetic_correlation),
        &format!("{:.6}", result.null_cophenetic_mean),
        &format!("{:.6}", result.null_cophenetic_std),
        &format!("{:.6}", result.p_value),
        RELABELING_DIAGNOSTIC,
    ])
    .unwrap();

    wtr.write_record([
        "ultrametric_fraction",
        &format!("{:.6}", result.ultrametric_fraction),
        "N/A",
        "N/A",
        "N/A",
        "N/A",
    ])
    .unwrap();

    wtr.write_record([
        "n_frbs",
        &n_frb_final.to_string(),
        "N/A",
        "N/A",
        "N/A",
        "N/A",
    ])
    .unwrap();

    wtr.write_record([
        "n_gw_events",
        &n_gw_final.to_string(),
        "N/A",
        "N/A",
        "N/A",
        "N/A",
    ])
    .unwrap();

    wtr.write_record([
        "redshift_range",
        &format!("[{:.4}, {:.4}]", z_range.0, z_range.1),
        "N/A",
        "N/A",
        "N/A",
        "N/A",
    ])
    .unwrap();

    wtr.flush().unwrap();

    eprintln!("\nResults written to {}", cli.output.display());

    eprintln!("\n=== Summary ===");
    eprintln!("FRBs: {}", n_frb_final);
    eprintln!("GW events: {}", n_gw_final);
    eprintln!("Redshift range: [{:.4}, {:.4}]", z_range.0, z_range.1);
    eprintln!(
        "Cophenetic correlation: {:.4}",
        result.cophenetic_correlation
    );
    eprintln!("Geometry-relabeling diagnostic: {RELABELING_DIAGNOSTIC}");
    Ok(())
}

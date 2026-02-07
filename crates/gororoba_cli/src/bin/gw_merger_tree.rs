//! Direction 4: GW Merger Tree Ultrametricity
//!
//! Tests whether GWTC-3 gravitational wave events exhibit ultrametric
//! structure in mass-redshift space. The hypothesis is that the BBH
//! mass spectrum's bimodal structure (peaks at ~10 and ~35 Msun)
//! reflects a hierarchical merger tree that should appear ultrametric.
//!
//! # Method
//!
//! 1. Load GWTC-3 confident events from CSV
//! 2. Build distance matrix in (mass, redshift) parameter space
//! 3. Run ultrametric fraction test on the distance matrix
//! 4. Run dendrogram analysis (cophenetic correlation)
//! 5. Compare to null: random draws from KDE of mass distribution
//!
//! # Usage
//!
//! gw-merger-tree --input data/external/GWTC-3_confident.csv \
//!                --output data/csv/c071e_gw_merger_ultrametric.csv

use clap::Parser;
use std::path::PathBuf;

use data_core::catalogs::gwtc::{GwEvent, parse_gwtc3_csv};
use stats_core::ultrametric;
use stats_core::ultrametric::dendrogram;

#[derive(Parser)]
#[command(name = "gw-merger-tree")]
#[command(about = "Direction 4: Test ultrametric structure in GW merger mass-redshift space")]
struct Cli {
    /// Path to GWTC events CSV (combined or GWTC-3 confident).
    /// Defaults to the combined catalog if available, otherwise GWTC-3.
    #[arg(long)]
    input: Option<PathBuf>,

    /// Weight for mass dimension in the combined metric.
    /// The distance is: d = sqrt(w_m * (dm/dm_scale)^2 + w_z * (dz/dz_scale)^2)
    #[arg(long, default_value = "1.0")]
    weight_mass: f64,

    /// Weight for redshift dimension.
    #[arg(long, default_value = "1.0")]
    weight_redshift: f64,

    /// Number of triples for ultrametric fraction test.
    #[arg(long, default_value = "100000")]
    n_triples: usize,

    /// Number of permutations for null distribution.
    #[arg(long, default_value = "500")]
    n_permutations: usize,

    /// Output CSV path.
    #[arg(long, default_value = "data/csv/c071e_gw_merger_ultrametric.csv")]
    output: PathBuf,
}

fn main() {
    let cli = Cli::parse();

    eprintln!("=== Direction 4: GW Merger Tree Ultrametricity ===");

    // Resolve input path: prefer combined catalog, fall back to GWTC-3
    let input = cli.input.unwrap_or_else(|| {
        let combined = PathBuf::from("data/external/gwosc_all_events.csv");
        if combined.exists() {
            eprintln!("Using combined GWOSC catalog (O1-O4a, ~219 events)");
            combined
        } else {
            eprintln!("Combined catalog not found, falling back to GWTC-3 confident");
            PathBuf::from("data/external/GWTC-3_confident.csv")
        }
    });
    eprintln!("Input: {}", input.display());

    // 1. Load GWTC events
    let events = parse_gwtc3_csv(&input)
        .unwrap_or_else(|e| {
            eprintln!("Failed to parse GWTC-3 CSV: {}", e);
            std::process::exit(1);
        });

    eprintln!("Loaded {} GW events", events.len());

    // Filter to events with valid mass and redshift
    let valid_events: Vec<&GwEvent> = events
        .iter()
        .filter(|e| {
            e.mass_1_source > 0.0
                && e.mass_2_source > 0.0
                && e.redshift > 0.0
                && !e.mass_1_source.is_nan()
                && !e.redshift.is_nan()
        })
        .collect();

    eprintln!("Valid events with mass + redshift: {}", valid_events.len());

    if valid_events.len() < 3 {
        eprintln!("Too few valid events for analysis");
        std::process::exit(1);
    }

    // 2. Extract parameters
    let n = valid_events.len();

    // Use chirp mass: M_c = (m1*m2)^(3/5) / (m1+m2)^(1/5)
    let chirp_masses: Vec<f64> = valid_events
        .iter()
        .map(|e| {
            let m1 = e.mass_1_source;
            let m2 = e.mass_2_source;
            (m1 * m2).powf(0.6) / (m1 + m2).powf(0.2)
        })
        .collect();

    let redshifts: Vec<f64> = valid_events.iter().map(|e| e.redshift).collect();

    // Compute scale factors (standard deviation for normalization)
    let mean_mc = chirp_masses.iter().sum::<f64>() / n as f64;
    let std_mc = (chirp_masses.iter().map(|m| (m - mean_mc).powi(2)).sum::<f64>() / n as f64).sqrt();

    let mean_z = redshifts.iter().sum::<f64>() / n as f64;
    let std_z = (redshifts.iter().map(|z| (z - mean_z).powi(2)).sum::<f64>() / n as f64).sqrt();

    eprintln!(
        "Chirp mass: mean={:.2} Msun, std={:.2} Msun",
        mean_mc, std_mc
    );
    eprintln!(
        "Redshift: mean={:.4}, std={:.4}",
        mean_z, std_z
    );

    // 3. Build distance matrix
    let n_pairs = n * (n - 1) / 2;
    let mut dist_matrix = Vec::with_capacity(n_pairs);

    let w_m = cli.weight_mass;
    let w_z = cli.weight_redshift;

    for i in 0..n {
        for j in (i + 1)..n {
            let dm = (chirp_masses[i] - chirp_masses[j]) / std_mc;
            let dz = (redshifts[i] - redshifts[j]) / std_z;
            let d = (w_m * dm * dm + w_z * dz * dz).sqrt();
            dist_matrix.push(d);
        }
    }

    // 4. Ultrametric fraction test
    eprintln!("\nRunning ultrametric fraction test...");
    let obs_frac = ultrametric::ultrametric_fraction_from_matrix(
        &dist_matrix, n, cli.n_triples, 42,
    );
    eprintln!("Observed ultrametric fraction: {:.4}", obs_frac);

    // 5. Dendrogram analysis
    eprintln!("Running dendrogram analysis...");
    let dend_result = dendrogram::hierarchical_ultrametric_test(
        &dist_matrix, n, cli.n_permutations, 42,
    );

    eprintln!(
        "Cophenetic correlation: {:.4} (null mean: {:.4} +/- {:.4})",
        dend_result.cophenetic_correlation,
        dend_result.null_cophenetic_mean,
        dend_result.null_cophenetic_std,
    );
    eprintln!("Dendrogram p-value: {:.4}", dend_result.p_value);
    eprintln!("Dendrogram verdict: {:?}", dend_result.verdict);

    // 6. Full ultrametric analysis on chirp masses alone
    eprintln!("\nRunning scalar ultrametric analysis on chirp masses...");
    let scalar_config = ultrametric::UltrametricConfig {
        n_triples: cli.n_triples,
        n_permutations: cli.n_permutations,
        primes: vec![2, 3, 5, 7],
        seed: 42,
    };
    let scalar_analysis = ultrametric::run_ultrametric_analysis(&chirp_masses, &scalar_config);

    eprintln!(
        "Scalar fraction: {:.4} (null: {:.4}, p={:.4})",
        scalar_analysis.fraction_result.ultrametric_fraction,
        scalar_analysis.fraction_result.null_fraction_mean,
        scalar_analysis.fraction_result.p_value,
    );

    // 7. Gate verdict
    let gate = ultrametric::ultrametric_gate(
        "C-071e",
        &scalar_analysis,
        "GWTC-3 chirp mass ultrametric structure",
    );
    eprintln!("\n=== C-071e Gate: {:?} ===", gate.verdict);
    eprintln!("{}", gate.message);

    // 8. Output CSV
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
    ]).unwrap();

    wtr.write_record([
        "chirp_mass_ultrametric_fraction",
        &format!("{:.6}", scalar_analysis.fraction_result.ultrametric_fraction),
        &format!("{:.6}", scalar_analysis.fraction_result.null_fraction_mean),
        &format!("{:.6}", scalar_analysis.fraction_result.null_fraction_std),
        &format!("{:.6}", scalar_analysis.fraction_result.p_value),
        &format!("{:?}", scalar_analysis.verdict),
    ]).unwrap();

    wtr.write_record([
        "mass_redshift_ultrametric_fraction",
        &format!("{:.6}", obs_frac),
        "N/A",
        "N/A",
        "N/A",
        "N/A",
    ]).unwrap();

    wtr.write_record([
        "cophenetic_correlation",
        &format!("{:.6}", dend_result.cophenetic_correlation),
        &format!("{:.6}", dend_result.null_cophenetic_mean),
        &format!("{:.6}", dend_result.null_cophenetic_std),
        &format!("{:.6}", dend_result.p_value),
        &format!("{:?}", dend_result.verdict),
    ]).unwrap();

    wtr.write_record([
        "defect_mean",
        &format!("{:.6}", scalar_analysis.defect_result.mean_defect),
        &format!("{:.6}", scalar_analysis.defect_result.null_mean_defect),
        "N/A",
        &format!("{:.6}", scalar_analysis.defect_result.defect_p_value),
        "N/A",
    ]).unwrap();

    for pr in &scalar_analysis.padic_results {
        wtr.write_record([
            &format!("padic_p{}", pr.prime),
            &format!("{:.6}", pr.padic_ultrametric_fraction),
            &format!("{:.6}", pr.null_ultrametric_fraction),
            "N/A",
            &format!("{:.6}", pr.p_value),
            "N/A",
        ]).unwrap();
    }

    wtr.flush().unwrap();
    eprintln!("\nResults written to {}", cli.output.display());

    // Summary
    eprintln!("\n=== Summary ===");
    eprintln!("Events analyzed: {}", n);
    eprintln!(
        "Chirp mass range: [{:.1}, {:.1}] Msun",
        chirp_masses.iter().cloned().fold(f64::INFINITY, f64::min),
        chirp_masses.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );
    eprintln!(
        "Redshift range: [{:.4}, {:.4}]",
        redshifts.iter().cloned().fold(f64::INFINITY, f64::min),
        redshifts.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );
    eprintln!("Gate C-071e: {:?}", gate.verdict);
}

//! E-028: Lattice Filtration Analyzer
//!
//! Validates Thesis 2 (particle mass from filtration) by:
//! 1. Computing survival depth spectrum for sedenion basis pairs
//! 2. Clustering depths and predicting lepton mass ratios
//! 3. Comparing predictions to PDG experimental values
//! 4. Running collision storm simulations with latency law classification
//!
//! Output: TOML report with survival depths, mass ratio predictions,
//! and latency law analysis.

use clap::Parser;
use lattice_filtration::{
    classify_latency_law_detailed, depth_histogram, pdg_comparison,
    simulate_fibonacci_collision_storm, simulate_sedenion_collision_storm, SurvivalDepthMap,
};
use std::io::Write;

#[derive(Parser, Debug)]
#[command(name = "lattice-filtration-analyzer")]
#[command(about = "E-028: Lattice Filtration Analysis for Thesis 2 Validation")]
struct Args {
    /// Number of collision storm steps
    #[arg(long, default_value = "5000")]
    n_steps: usize,

    /// Output directory for results
    #[arg(long, default_value = "data/evidence")]
    output_dir: String,

    /// Random seed for sedenion storm
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Enable sedenion collision storm (in addition to Fibonacci)
    #[arg(long, default_value = "true")]
    sedenion_storm: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("E-028: Lattice Filtration Analyzer");
    println!("===================================");
    println!("Steps: {}, Seed: {}", args.n_steps, args.seed);
    println!();

    // Phase 1: Survival depth spectrum
    println!("[1/4] Computing survival depth spectrum (16x16 sedenion pairs)...");
    let map = SurvivalDepthMap::compute();
    let hist = depth_histogram(&map);
    let (d_min, d_max) = map.depth_range();
    println!(
        "  256 pairs analyzed: {} distinct depths (range [{}, {}])",
        map.n_distinct_depths(),
        d_min,
        d_max
    );
    println!("  Mean depth: {:.2}", map.mean_depth());

    for bin in &hist {
        println!(
            "  Depth {}: {} pairs ({:.1}%)",
            bin.depth,
            bin.count,
            bin.fraction * 100.0
        );
    }

    // Phase 2: Lepton mass ratio predictions
    println!();
    println!("[2/4] Predicting lepton mass ratios from depth clusters...");
    let comp = pdg_comparison(&map);
    println!("  Found {} depth clusters", comp.n_clusters);

    for pred in &comp.predictions {
        println!(
            "  {:?}: predicted={:.4}, PDG={:.4}, error={:.2}%",
            pred.label,
            pred.predicted_ratio,
            pred.pdg_ratio,
            pred.relative_error * 100.0
        );
    }

    if !comp.predictions.is_empty() {
        println!(
            "  Mean relative error: {:.2}%",
            comp.mean_relative_error * 100.0
        );
    }

    // Phase 3: Fibonacci collision storm
    println!();
    println!(
        "[3/4] Running Fibonacci collision storm ({} steps)...",
        args.n_steps
    );
    let (fib_stats, fib_obs) = simulate_fibonacci_collision_storm(args.n_steps, 251);
    let fib_samples: Vec<(f64, f64)> = fib_obs.iter().map(|o| (o.radius, o.latency)).collect();
    let fib_detail = classify_latency_law_detailed(&fib_samples);
    println!("  Latency law: {:?}", fib_stats.latency_law);
    println!("  R^2 inverse-square: {:.4}", fib_detail.r2_inverse_square);
    println!("  R^2 power-law: {:.4}", fib_detail.r2_power_law);
    println!("  Power-law exponent: {:.4}", fib_detail.power_law_exponent);
    println!("  Mean latency: {:.2}", fib_stats.mean_latency);

    // Phase 4: Sedenion collision storm
    let mut sed_detail = None;
    if args.sedenion_storm {
        println!();
        println!(
            "[4/4] Running sedenion collision storm ({} steps, seed={})...",
            args.n_steps, args.seed
        );
        let (sed_stats, sed_obs) = simulate_sedenion_collision_storm(args.n_steps, 16, args.seed);
        let sed_samples: Vec<(f64, f64)> = sed_obs.iter().map(|o| (o.radius, o.latency)).collect();
        let detail = classify_latency_law_detailed(&sed_samples);
        println!("  Latency law: {:?}", sed_stats.latency_law);
        println!("  R^2 inverse-square: {:.4}", detail.r2_inverse_square);
        println!("  R^2 power-law: {:.4}", detail.r2_power_law);
        println!("  Power-law exponent: {:.4}", detail.power_law_exponent);
        println!("  Mean latency: {:.2}", sed_stats.mean_latency);
        println!("  Total collisions: {}", sed_stats.total_collisions);
        sed_detail = Some((sed_stats, detail));
    }

    // Write TOML output
    println!();
    println!("Writing output...");
    std::fs::create_dir_all(&args.output_dir)?;
    let toml_path = format!("{}/e028_filtration_analysis.toml", args.output_dir);
    let mut f = std::fs::File::create(&toml_path)?;

    writeln!(f, "[metadata]")?;
    writeln!(f, "experiment = \"E-028\"")?;
    writeln!(
        f,
        "title = \"Lepton Mass Ratio Filtration via Patricia Trie\""
    )?;
    writeln!(f, "n_steps = {}", args.n_steps)?;
    writeln!(f, "seed = {}", args.seed)?;
    writeln!(f)?;

    writeln!(f, "[survival_depth]")?;
    writeln!(f, "n_pairs = 256")?;
    writeln!(f, "n_distinct_depths = {}", map.n_distinct_depths())?;
    writeln!(f, "depth_min = {}", d_min)?;
    writeln!(f, "depth_max = {}", d_max)?;
    writeln!(f, "mean_depth = {:.4}", map.mean_depth())?;
    writeln!(f)?;

    writeln!(f, "[mass_ratios]")?;
    writeln!(f, "n_clusters = {}", comp.n_clusters)?;
    if !comp.predictions.is_empty() {
        writeln!(f, "mean_relative_error = {:.6}", comp.mean_relative_error)?;
        writeln!(f, "best_relative_error = {:.6}", comp.best_relative_error)?;
        writeln!(f, "worst_relative_error = {:.6}", comp.worst_relative_error)?;
    }
    writeln!(f)?;

    for pred in &comp.predictions {
        writeln!(f, "[[mass_ratios.prediction]]")?;
        writeln!(f, "label = \"{:?}\"", pred.label)?;
        writeln!(f, "predicted_ratio = {:.6}", pred.predicted_ratio)?;
        writeln!(f, "pdg_ratio = {:.6}", pred.pdg_ratio)?;
        writeln!(f, "relative_error = {:.6}", pred.relative_error)?;
        writeln!(f, "depth_light = {}", pred.depth_light)?;
        writeln!(f, "depth_heavy = {}", pred.depth_heavy)?;
        writeln!(f)?;
    }

    writeln!(f, "[fibonacci_storm]")?;
    writeln!(f, "latency_law = \"{:?}\"", fib_stats.latency_law)?;
    writeln!(f, "r2_inverse_square = {:.6}", fib_detail.r2_inverse_square)?;
    writeln!(f, "r2_power_law = {:.6}", fib_detail.r2_power_law)?;
    writeln!(
        f,
        "power_law_exponent = {:.6}",
        fib_detail.power_law_exponent
    )?;
    writeln!(f, "mean_latency = {:.4}", fib_stats.mean_latency)?;
    writeln!(f, "total_collisions = {}", fib_stats.total_collisions)?;
    writeln!(f)?;

    if let Some((sed_stats, detail)) = &sed_detail {
        writeln!(f, "[sedenion_storm]")?;
        writeln!(f, "latency_law = \"{:?}\"", sed_stats.latency_law)?;
        writeln!(f, "r2_inverse_square = {:.6}", detail.r2_inverse_square)?;
        writeln!(f, "r2_power_law = {:.6}", detail.r2_power_law)?;
        writeln!(f, "power_law_exponent = {:.6}", detail.power_law_exponent)?;
        writeln!(f, "mean_latency = {:.4}", sed_stats.mean_latency)?;
        writeln!(f, "total_collisions = {}", sed_stats.total_collisions)?;
    }

    println!("  TOML: {}", toml_path);
    println!();
    println!("===================================");
    println!("E-028 Analysis Complete");

    Ok(())
}

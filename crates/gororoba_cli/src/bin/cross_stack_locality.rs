//! Cross-stack locality comparison (Experiment A).
//!
//! Compares adjacency-locality metrics across three (optionally four) independent
//! constraint systems under a shared abstract model.

use std::path::PathBuf;

use algebra_core::experimental::algebraic_dynamics::{
    cross_stack_comparison, ConstraintSystem, E10DynkinSystem, EtDmzSystem, SedenionZdSystem,
    TwistNavigationSystem,
};
use clap::Parser;

#[derive(Parser)]
#[command(name = "cross-stack-locality")]
#[command(about = "Compare locality metrics across E10, ET DMZ, and ZD stacks")]
struct Args {
    /// Number of transitions to simulate per system.
    #[arg(long, default_value = "10000")]
    n_bounces: usize,

    /// Number of permutation samples for p-value estimation.
    #[arg(long, default_value = "1000")]
    n_permutations: usize,

    /// RNG seed for reproducibility.
    #[arg(long, default_value = "42")]
    seed: u64,

    /// CD level N used for ET DMZ adjacency graph construction.
    #[arg(long, default_value = "5")]
    et_level: usize,

    /// Strut constant S used for ET DMZ adjacency graph construction.
    #[arg(long, default_value = "9")]
    et_strut: usize,

    /// Include the Twist Navigation stack as a fourth system.
    #[arg(long)]
    include_twist: bool,

    /// Output CSV path.
    #[arg(long, default_value = "data/csv/cross_stack_locality_comparison.csv")]
    output: PathBuf,
}

fn main() {
    let args = Args::parse();

    if args.n_bounces < 2 {
        eprintln!("ERROR: --n-bounces must be >= 2");
        std::process::exit(1);
    }
    if args.n_permutations == 0 {
        eprintln!("ERROR: --n-permutations must be > 0");
        std::process::exit(1);
    }
    if args.et_level < 4 {
        eprintln!("ERROR: --et-level must be >= 4 for ET systems");
        std::process::exit(1);
    }

    let e10 = E10DynkinSystem;
    let et = EtDmzSystem::new(args.et_level, args.et_strut);
    let zd = SedenionZdSystem::new();
    let twist = TwistNavigationSystem::new();

    let mut systems: Vec<&dyn ConstraintSystem> = vec![&e10, &et, &zd];
    if args.include_twist {
        systems.push(&twist);
    }

    let result = cross_stack_comparison(&systems, args.n_bounces, args.n_permutations, args.seed);

    if let Some(parent) = args.output.parent() {
        if let Err(err) = std::fs::create_dir_all(parent) {
            eprintln!(
                "ERROR: failed to create output directory {}: {err}",
                parent.display()
            );
            std::process::exit(1);
        }
    }

    let mut wtr = match csv::Writer::from_path(&args.output) {
        Ok(writer) => writer,
        Err(err) => {
            eprintln!(
                "ERROR: failed to open output CSV {}: {err}",
                args.output.display()
            );
            std::process::exit(1);
        }
    };

    if let Err(err) = wtr.write_record([
        "system_name",
        "n_generators",
        "n_transitions",
        "n_adjacent",
        "walk_r",
        "walk_r_null",
        "walk_mutual_information",
        "null_r",
        "null_mutual_information",
        "p_value",
        "n_bounces",
        "n_permutations",
        "seed",
    ]) {
        eprintln!("ERROR: failed to write CSV header: {err}");
        std::process::exit(1);
    }

    for idx in 0..result.walk_metrics.len() {
        let walk = &result.walk_metrics[idx];
        let null = &result.null_metrics[idx];
        let p_value = result.p_values[idx];

        if let Err(err) = wtr.write_record([
            walk.system_name.clone(),
            walk.n_generators.to_string(),
            walk.n_transitions.to_string(),
            walk.n_adjacent.to_string(),
            format!("{:.10}", walk.r),
            format!("{:.10}", walk.r_null),
            format!("{:.10}", walk.mutual_information),
            format!("{:.10}", null.r),
            format!("{:.10}", null.mutual_information),
            format!("{:.10}", p_value),
            args.n_bounces.to_string(),
            args.n_permutations.to_string(),
            args.seed.to_string(),
        ]) {
            eprintln!("ERROR: failed to write CSV row: {err}");
            std::process::exit(1);
        }
    }

    if let Err(err) = wtr.flush() {
        eprintln!("ERROR: failed to flush output CSV: {err}");
        std::process::exit(1);
    }

    eprintln!(
        "Wrote {} systems to {}",
        result.walk_metrics.len(),
        args.output.display()
    );
}

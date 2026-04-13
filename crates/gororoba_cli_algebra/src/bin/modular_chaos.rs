//! Modular Chaos Simulation.
//!
//! CLI wrapper around spectral_core::modular_chaos::run_modular_chaos.
//! All FFT and phase-evolution logic lives in the domain crate; this binary
//! parses arguments, drives the simulation, and writes the CSV output.

use clap::Parser;
use spectral_core::modular_chaos::run_modular_chaos;
use std::{fs::File, io::Write, path::PathBuf};

#[derive(Parser)]
struct Args {
    /// N (state space size)
    #[arg(long, default_value_t = 256)]
    n: usize,

    /// Steps
    #[arg(long, default_value_t = 100)]
    steps: usize,

    /// Alpha (phase factor)
    #[arg(long, default_value_t = 1.5)]
    alpha: f64,

    /// Output CSV path
    #[arg(long, default_value = "data/csv/modular_chaos_results.csv")]
    output: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("Running Modular Chaos Sim (N={})...", args.n);

    let result = run_modular_chaos(args.n, args.steps, args.alpha);

    let mut file = File::create(&args.output)?;
    writeln!(file, "step,entropy")?;
    for (i, h) in result.entropy_history.iter().enumerate() {
        writeln!(file, "{},{}", i, h)?;
    }

    println!("Wrote results to {}", args.output.display());
    Ok(())
}

//! Modular Chaos Simulation.
//!
//! Ported from src/modular_classical_sim.py to pure Rust.
//! Simulates discrete phase scrambling and Fourier resonance.

use clap::Parser;
use rustfft::{FftPlanner, num_complex::Complex64};
use std::path::PathBuf;
use std::fs::File;
use std::io::Write;

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
    let n = args.n;
    let n_f64 = n as f64;
    let sqrt_n = n_f64.sqrt();

    let mut psi = vec![Complex64::new(1.0 / sqrt_n, 0.0); n];
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);

    let t_op: Vec<Complex64> = (0..n)
        .map(|k| {
            let phase = args.alpha * (k * k) as f64 * std::f64::consts::PI / n_f64;
            Complex64::from_polar(1.0, phase)
        })
        .collect();

    let mut entropy_history = Vec::with_capacity(args.steps);

    println!("Running Modular Chaos Sim (N={})...", n);

    for _ in 0..args.steps {
        // Apply T (Quadratic Phase)
        for (p, t) in psi.iter_mut().zip(t_op.iter()) {
            *p *= t;
        }

        // Apply S (FFT)
        fft.process(&mut psi);
        for c in psi.iter_mut() {
            *c /= sqrt_n;
        }

        // Measure Entropy
        let mut h = 0.0;
        for p in psi.iter().map(|c| c.norm_sqr()) {
            let p_clamped = p.max(1e-15);
            h -= p_clamped * p_clamped.log2();
        }
        entropy_history.push(h);
    }

    let mut file = File::create(&args.output)?;
    writeln!(file, "step,entropy")?;
    for (i, h) in entropy_history.iter().enumerate() {
        writeln!(file, "{},{}", i, h)?;
    }

    println!("Wrote results to {}", args.output.display());
    Ok(())
}

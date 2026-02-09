//! frac-schrodinger: Fractional Schrodinger equation solver.
//!
//! Provides:
//! - Levy free-particle propagator
//! - Split-operator time evolution
//! - Imaginary-time ground state projection
//! - Variational ground state energy
//!
//! Usage:
//!   frac-schrodinger variational --alpha 1.5 --omega 1.0
//!   frac-schrodinger ground-state --alpha 1.5 --n 256 --l 20.0
//!   frac-schrodinger propagator --alpha 1.5 --t 1.0 --n-x 64

use clap::{Parser, Subcommand};
use quantum_core::fractional_schrodinger::{
    imaginary_time_ground_state, propagator_l2_error, variational_ground_state,
};

#[derive(Parser)]
#[command(name = "frac-schrodinger")]
#[command(about = "Fractional Schrodinger equation solver (Levy propagator, split-operator)")]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compute variational ground state energy using Gaussian trial
    Variational {
        /// Levy index alpha in (0, 2]
        #[arg(long, default_value = "2.0")]
        alpha: f64,

        /// Generalized diffusion coefficient D
        #[arg(long, default_value = "0.5")]
        d: f64,

        /// Harmonic oscillator frequency omega
        #[arg(long, default_value = "1.0")]
        omega: f64,

        /// Mass
        #[arg(long, default_value = "1.0")]
        m: f64,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Compute ground state via imaginary-time evolution
    GroundState {
        /// Levy index alpha in (0, 2]
        #[arg(long, default_value = "2.0")]
        alpha: f64,

        /// Generalized diffusion coefficient D
        #[arg(long, default_value = "0.5")]
        d: f64,

        /// Harmonic oscillator frequency omega
        #[arg(long, default_value = "1.0")]
        omega: f64,

        /// Number of grid points
        #[arg(long, default_value = "256")]
        n: usize,

        /// Domain size [-L, L]
        #[arg(long, default_value = "15.0")]
        l: f64,

        /// Imaginary time step
        #[arg(long, default_value = "0.01")]
        tau: f64,

        /// Number of time steps
        #[arg(long, default_value = "3000")]
        steps: usize,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Compare Levy propagator to Gaussian at alpha=2
    PropagatorBenchmark {
        /// Levy index alpha
        #[arg(long, default_value = "2.0")]
        alpha: f64,

        /// Generalized diffusion coefficient D
        #[arg(long, default_value = "0.5")]
        d: f64,

        /// Time
        #[arg(long, default_value = "1.0")]
        t: f64,

        /// Number of spatial points
        #[arg(long, default_value = "64")]
        n_x: usize,

        /// Domain size
        #[arg(long, default_value = "10.0")]
        l: f64,

        /// Number of k-space quadrature points
        #[arg(long, default_value = "4096")]
        n_k: usize,

        /// k-space cutoff
        #[arg(long, default_value = "40.0")]
        k_max: f64,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Sweep alpha values and compute ground state energies
    AlphaSweep {
        /// Starting alpha
        #[arg(long, default_value = "0.5")]
        alpha_start: f64,

        /// Ending alpha
        #[arg(long, default_value = "2.0")]
        alpha_end: f64,

        /// Number of alpha values
        #[arg(long, default_value = "10")]
        n_alpha: usize,

        /// Use variational (faster) or imaginary-time (more accurate)
        #[arg(long, default_value = "variational")]
        method: String,

        /// Output CSV file
        #[arg(short, long)]
        output: Option<String>,
    },
}

fn main() {
    let args = Args::parse();

    match args.command {
        Commands::Variational {
            alpha,
            d,
            omega,
            m,
            json,
        } => {
            run_variational(alpha, d, omega, m, json);
        }
        Commands::GroundState {
            alpha,
            d,
            omega,
            n,
            l,
            tau,
            steps,
            json,
        } => {
            run_ground_state(alpha, d, omega, n, l, tau, steps, json);
        }
        Commands::PropagatorBenchmark {
            alpha,
            d,
            t,
            n_x,
            l,
            n_k,
            k_max,
            json,
        } => {
            run_propagator_benchmark(alpha, d, t, n_x, l, n_k, k_max, json);
        }
        Commands::AlphaSweep {
            alpha_start,
            alpha_end,
            n_alpha,
            method,
            output,
        } => {
            run_alpha_sweep(alpha_start, alpha_end, n_alpha, &method, output);
        }
    }
}

fn run_variational(alpha: f64, d: f64, omega: f64, m: f64, json: bool) {
    eprintln!(
        "Variational ground state: alpha={}, D={}, omega={}, m={}",
        alpha, d, omega, m
    );

    let result = variational_ground_state(alpha, d, omega, m);

    if json {
        println!("{{");
        println!("  \"alpha\": {},", alpha);
        println!("  \"D\": {},", d);
        println!("  \"omega\": {},", omega);
        println!("  \"m\": {},", m);
        println!("  \"energy\": {},", result.energy);
        println!("  \"beta_opt\": {},", result.beta_opt);
        println!("  \"kinetic\": {},", result.kinetic);
        println!("  \"potential\": {}", result.potential);
        println!("}}");
    } else {
        println!("Fractional Harmonic Oscillator (Variational)");
        println!("  H = D * |k|^alpha + (1/2) * m * omega^2 * x^2");
        println!("  alpha = {}", alpha);
        println!();
        println!("Results:");
        println!("  Ground state energy:  {:.10}", result.energy);
        println!("  Optimal beta:         {:.10}", result.beta_opt);
        println!("  Kinetic energy <T>:   {:.10}", result.kinetic);
        println!("  Potential energy <V>: {:.10}", result.potential);

        if (alpha - 2.0).abs() < 0.01 {
            let exact = omega / 2.0;
            let rel_err = (result.energy - exact).abs() / exact;
            println!();
            println!("  Exact (alpha=2):      {:.10}", exact);
            println!("  Relative error:       {:.2e}", rel_err);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_ground_state(
    alpha: f64,
    d: f64,
    omega: f64,
    n: usize,
    l: f64,
    tau: f64,
    steps: usize,
    json: bool,
) {
    eprintln!(
        "Imaginary-time ground state: alpha={}, D={}, omega={}, n={}, L={}",
        alpha, d, omega, n, l
    );
    eprintln!("Evolution: tau={}, steps={}", tau, steps);

    let dx = 2.0 * l / n as f64;
    let x: Vec<f64> = (0..n).map(|i| -l + i as f64 * dx).collect();
    let v: Vec<f64> = x.iter().map(|&xi| 0.5 * omega * omega * xi * xi).collect();

    let (psi, energy) = imaginary_time_ground_state(&x, &v, alpha, d, tau, steps);

    if json {
        println!("{{");
        println!("  \"alpha\": {},", alpha);
        println!("  \"D\": {},", d);
        println!("  \"omega\": {},", omega);
        println!("  \"n\": {},", n);
        println!("  \"L\": {},", l);
        println!("  \"energy\": {},", energy);
        println!(
            "  \"psi_max\": {}",
            psi.iter().cloned().fold(0.0_f64, f64::max)
        );
        println!("}}");
    } else {
        println!("Fractional Harmonic Oscillator (Imaginary-Time)");
        println!("  H = D * |k|^alpha + (1/2) * omega^2 * x^2");
        println!("  alpha = {}", alpha);
        println!();
        println!("Results:");
        println!("  Ground state energy: {:.10}", energy);
        println!(
            "  Max |psi|:           {:.10}",
            psi.iter().cloned().fold(0.0_f64, f64::max)
        );

        if (alpha - 2.0).abs() < 0.01 {
            let exact = omega / 2.0;
            let rel_err = (energy - exact).abs() / exact;
            println!();
            println!("  Exact (alpha=2):     {:.10}", exact);
            println!("  Relative error:      {:.2e}", rel_err);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_propagator_benchmark(
    alpha: f64,
    d: f64,
    t: f64,
    n_x: usize,
    l: f64,
    n_k: usize,
    k_max: f64,
    json: bool,
) {
    eprintln!("Propagator L2 error: alpha={}, D={}, t={}", alpha, d, t);

    let l2_err = propagator_l2_error(alpha, d, t, n_x, l, n_k, k_max);

    if json {
        println!("{{");
        println!("  \"alpha\": {},", alpha);
        println!("  \"D\": {},", d);
        println!("  \"t\": {},", t);
        println!("  \"n_x\": {},", n_x);
        println!("  \"L\": {},", l);
        println!("  \"n_k\": {},", n_k);
        println!("  \"k_max\": {},", k_max);
        println!("  \"l2_error\": {}", l2_err);
        println!("}}");
    } else {
        println!("Levy Propagator vs Gaussian Benchmark");
        println!("  alpha = {}", alpha);
        println!("  L2 error = {:.6e}", l2_err);
        if (alpha - 2.0).abs() < 0.01 {
            println!("  (At alpha=2, error should be < 0.05 if Levy matches Gaussian)");
        }
    }
}

fn run_alpha_sweep(
    alpha_start: f64,
    alpha_end: f64,
    n_alpha: usize,
    method: &str,
    output: Option<String>,
) {
    eprintln!(
        "Alpha sweep: [{} -> {}], {} values, method={}",
        alpha_start, alpha_end, n_alpha, method
    );

    let alpha_values: Vec<f64> = (0..n_alpha)
        .map(|i| alpha_start + (alpha_end - alpha_start) * i as f64 / (n_alpha - 1) as f64)
        .collect();

    let energies: Vec<f64> = alpha_values
        .iter()
        .map(|&alpha| {
            if method == "variational" {
                variational_ground_state(alpha, 0.5, 1.0, 1.0).energy
            } else {
                let n = 256;
                let l = 15.0;
                let dx = 2.0 * l / n as f64;
                let x: Vec<f64> = (0..n).map(|i| -l + i as f64 * dx).collect();
                let v: Vec<f64> = x.iter().map(|&xi| 0.5 * xi * xi).collect();
                let (_, e) = imaginary_time_ground_state(&x, &v, alpha, 0.5, 0.01, 2000);
                e
            }
        })
        .collect();

    if let Some(path) = output {
        let mut wtr = csv::Writer::from_path(&path).expect("Failed to create CSV");
        wtr.write_record(["alpha", "energy"]).unwrap();
        for (alpha, e) in alpha_values.iter().zip(energies.iter()) {
            wtr.write_record(&[alpha.to_string(), e.to_string()])
                .unwrap();
        }
        wtr.flush().unwrap();
        println!("Wrote {} records to {}", n_alpha, path);
    } else {
        println!("alpha,energy");
        for (alpha, e) in alpha_values.iter().zip(energies.iter()) {
            println!("{:.4},{:.10}", alpha, e);
        }
    }
}

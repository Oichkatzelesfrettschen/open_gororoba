//! neg-dim-eigen: Negative-dimension PDE eigenvalue solver.
//!
//! Computes eigenvalues of H = T(k) + V(x) where:
//! - T(k) = (|k| + epsilon)^alpha (regularized fractional kinetic operator)
//! - V(x) = 0.5 * x^2 (harmonic potential)
//!
//! Usage: neg-dim-eigen eigenvalues --alpha -1.5 --epsilon 0.1 --n-eig 5
//!        neg-dim-eigen sweep --alpha -1.5 --eps-start 0.5 --eps-end 0.01 --eps-steps 10
//!        neg-dim-eigen caffarelli-silvestre --s 0.5 --n-eig 3

use clap::{Parser, Subcommand};
use spectral_core::neg_dim::{
    caffarelli_silvestre_eigenvalues, eigenvalues_imaginary_time, epsilon_convergence_sweep,
};

#[derive(Parser)]
#[command(name = "neg-dim-eigen")]
#[command(about = "Negative-dimension PDE eigenvalue solver via imaginary-time evolution")]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compute eigenvalues for H = T(k) + V(x)
    Eigenvalues {
        /// Fractional exponent (alpha < 0 for negative-dimension)
        #[arg(long, default_value = "-1.5", allow_hyphen_values = true)]
        alpha: f64,

        /// Regularization parameter (epsilon > 0)
        #[arg(long, default_value = "0.1")]
        epsilon: f64,

        /// Number of grid points
        #[arg(long, default_value = "128")]
        n: usize,

        /// Domain size [-L/2, L/2]
        #[arg(long, default_value = "10.0")]
        domain: f64,

        /// Number of eigenvalues to compute
        #[arg(long, default_value = "5")]
        n_eig: usize,

        /// Imaginary time step
        #[arg(long, default_value = "0.005")]
        dt: f64,

        /// Number of time steps
        #[arg(long, default_value = "5000")]
        n_steps: usize,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Sweep epsilon and track eigenvalue convergence
    Sweep {
        /// Fractional exponent
        #[arg(long, default_value = "-1.5", allow_hyphen_values = true)]
        alpha: f64,

        /// Starting epsilon (largest)
        #[arg(long, default_value = "0.5")]
        eps_start: f64,

        /// Ending epsilon (smallest)
        #[arg(long, default_value = "0.01")]
        eps_end: f64,

        /// Number of epsilon values
        #[arg(long, default_value = "10")]
        eps_steps: usize,

        /// Number of grid points
        #[arg(long, default_value = "128")]
        n: usize,

        /// Domain size
        #[arg(long, default_value = "10.0")]
        domain: f64,

        /// Number of eigenvalues to track
        #[arg(long, default_value = "3")]
        n_eig: usize,

        /// Output CSV file
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Compute Caffarelli-Silvestre fractional Laplacian eigenvalues
    CaffarelliSilvestre {
        /// Fractional order s in (0,1)
        #[arg(long, default_value = "0.5")]
        s: f64,

        /// Number of grid points
        #[arg(long, default_value = "128")]
        n: usize,

        /// Domain size
        #[arg(long, default_value = "10.0")]
        domain: f64,

        /// Number of eigenvalues
        #[arg(long, default_value = "5")]
        n_eig: usize,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
}

fn main() {
    let args = Args::parse();

    match args.command {
        Commands::Eigenvalues {
            alpha,
            epsilon,
            n,
            domain,
            n_eig,
            dt,
            n_steps,
            json,
        } => {
            run_eigenvalues(alpha, epsilon, n, domain, n_eig, dt, n_steps, json);
        }
        Commands::Sweep {
            alpha,
            eps_start,
            eps_end,
            eps_steps,
            n,
            domain,
            n_eig,
            output,
        } => {
            run_sweep(
                alpha, eps_start, eps_end, eps_steps, n, domain, n_eig, output,
            );
        }
        Commands::CaffarelliSilvestre {
            s,
            n,
            domain,
            n_eig,
            json,
        } => {
            run_caffarelli_silvestre(s, n, domain, n_eig, json);
        }
    }
}
#[allow(clippy::too_many_arguments)]
fn run_eigenvalues(
    alpha: f64,
    epsilon: f64,
    n: usize,
    domain: f64,
    n_eig: usize,
    dt: f64,
    n_steps: usize,
    json: bool,
) {
    eprintln!(
        "Negative-dimension eigenvalues: alpha={}, epsilon={}, n={}, L={}",
        alpha, epsilon, n, domain
    );
    eprintln!("Imaginary-time evolution: dt={}, steps={}", dt, n_steps);

    let result = eigenvalues_imaginary_time(alpha, epsilon, n, domain, n_eig, dt, n_steps);

    if json {
        println!("{{");
        println!("  \"alpha\": {},", alpha);
        println!("  \"epsilon\": {},", epsilon);
        println!("  \"n\": {},", n);
        println!("  \"domain\": {},", domain);
        println!("  \"eigenvalues\": [");
        for (i, &e) in result.eigenvalues.iter().enumerate() {
            let comma = if i < result.eigenvalues.len() - 1 {
                ","
            } else {
                ""
            };
            println!("    {}{}", e, comma);
        }
        println!("  ]");
        println!("}}");
    } else {
        println!("Eigenvalues of H = T(k) + V(x):");
        println!("  T(k) = (|k| + {})^{}", epsilon, alpha);
        println!("  V(x) = 0.5 * x^2");
        println!();
        for (i, &e) in result.eigenvalues.iter().enumerate() {
            println!("  E_{} = {:.10}", i, e);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_sweep(
    alpha: f64,
    eps_start: f64,
    eps_end: f64,
    eps_steps: usize,
    n: usize,
    domain: f64,
    n_eig: usize,
    output: Option<String>,
) {
    eprintln!(
        "Epsilon convergence sweep: alpha={}, eps=[{} -> {}], {} steps",
        alpha, eps_start, eps_end, eps_steps
    );

    // Generate logarithmically-spaced epsilon values
    let log_start = eps_start.ln();
    let log_end = eps_end.ln();
    let eps_values: Vec<f64> = (0..eps_steps)
        .map(|i| {
            let t = i as f64 / (eps_steps - 1) as f64;
            (log_start + t * (log_end - log_start)).exp()
        })
        .collect();

    let results = epsilon_convergence_sweep(alpha, &eps_values, n, domain, n_eig, 0.005, 3000);

    if let Some(path) = output {
        let mut wtr = csv::Writer::from_path(&path).expect("Failed to create CSV");
        wtr.write_record(["epsilon", "index", "eigenvalue", "rel_change"])
            .unwrap();
        for r in &results {
            wtr.write_record(&[
                r.epsilon.to_string(),
                r.index.to_string(),
                r.value.to_string(),
                r.rel_change.to_string(),
            ])
            .unwrap();
        }
        wtr.flush().unwrap();
        println!("Wrote {} records to {}", results.len(), path);
    } else {
        println!("epsilon,index,eigenvalue,rel_change");
        for r in &results {
            println!("{},{},{},{}", r.epsilon, r.index, r.value, r.rel_change);
        }
    }
}

fn run_caffarelli_silvestre(s: f64, n: usize, domain: f64, n_eig: usize, json: bool) {
    eprintln!(
        "Caffarelli-Silvestre fractional Laplacian: s={}, n={}, L={}",
        s, n, domain
    );

    if s <= 0.0 || s >= 1.0 {
        eprintln!("Warning: s should be in (0, 1) for Caffarelli-Silvestre extension");
    }

    let eigs = caffarelli_silvestre_eigenvalues(s, n, domain, n_eig);

    if json {
        println!("{{");
        println!("  \"s\": {},", s);
        println!("  \"n\": {},", n);
        println!("  \"domain\": {},", domain);
        println!("  \"eigenvalues\": [");
        for (i, &e) in eigs.iter().enumerate() {
            let comma = if i < eigs.len() - 1 { "," } else { "" };
            println!("    {}{}", e, comma);
        }
        println!("  ]");
        println!("}}");
    } else {
        println!("Caffarelli-Silvestre eigenvalues:");
        println!("  H = |k|^{{2s}} + V(x), s = {}", s);
        println!("  V(x) = 0.5 * x^2");
        println!();
        for (i, &e) in eigs.iter().enumerate() {
            println!("  E_{} = {:.10}", i, e);
        }
    }
}

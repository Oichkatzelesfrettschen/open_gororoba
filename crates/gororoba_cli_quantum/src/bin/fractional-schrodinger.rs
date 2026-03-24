use anyhow::Result;
use clap::Parser;
use num_complex::Complex64;
use quantum_core::fractional_schrodinger::split_operator_evolve;
use std::fs::File;
use std::io::Write;

#[derive(Parser, Debug)]
#[command(author, version, about = "Fractional Schrodinger Equation Solver (Spectral/Strang)")]
struct Args {
    #[arg(short, long, default_value_t = 1024)]
    n: usize,

    #[arg(short, long, default_value_t = 100.0)]
    l: f64,

    #[arg(short, long, default_value_t = 0.5)]
    d: f64,

    #[arg(long, default_value_t = 0.05)]
    dt: f64,

    #[arg(short, long, default_value_t = 40.0)]
    t: f64,

    #[arg(short, long, default_value = "2.0,1.5,1.2")]
    alphas: String,

    #[arg(short, long, default_value = "data/artifacts/fractional_schrodinger_results.csv")]
    output: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let n = args.n;
    let l = args.l;
    let dx = (2.0 * l) / n as f64;
    let x: Vec<f64> = (0..n).map(|i| -l + i as f64 * dx).collect();

    // Potential: Optical Lattice + Harmonic Trap
    // V = V0 * (1.0 - cos(Klat * x)) + 0.5 * Omega^2 * x^2
    let v0 = 2.0;
    let k_lat = 1.0;
    let omega = 0.05;
    let v: Vec<f64> = x
        .iter()
        .map(|&xi| v0 * (1.0 - (k_lat * xi).cos()) + 0.5 * omega * omega * xi * xi)
        .collect();

    // Initial State: Gaussian
    let x0 = -20.0;
    let k0 = 1.5;
    let sigma = 4.0;
    let mut psi: Vec<Complex64> = x
        .iter()
        .map(|&xi| {
            let amp = (-(xi - x0).powi(2) / (2.0 * sigma * sigma)).exp();
            let phase = k0 * xi;
            Complex64::new(amp * phase.cos(), amp * phase.sin())
        })
        .collect();

    // Normalize
    let norm_sq: f64 = psi.iter().map(|p| p.norm_sqr()).sum();
    let norm = (norm_sq * dx).sqrt();
    for p in &mut psi {
        *p /= norm;
    }

    let alpha_vals: Vec<f64> = args
        .alphas
        .split(',')
        .map(|s| s.trim().parse::<f64>().expect("Invalid alpha"))
        .collect();

    let steps = (args.t / args.dt).ceil() as usize;

    let mut csv_file = File::create(&args.output)?;
    writeln!(csv_file, "x,alpha,density")?;

    for &alpha in &alpha_vals {
        println!("Evolving alpha = {:.2}...", alpha);
        let result = split_operator_evolve(&psi, &x, &v, alpha, args.d, args.dt, steps, true);

        for (xi, &dens) in x.iter().zip(result.density.iter()) {
            writeln!(csv_file, "{:.6},{:.2},{:.6e}", xi, alpha, dens)?;
        }
    }

    println!("Saved results to {}", args.output);

    Ok(())
}

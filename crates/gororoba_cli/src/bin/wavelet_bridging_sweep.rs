//! Thesis 1: Negative-d wavelet reservoir reduces terminal RelErr >= 5%.
//!
//! Sweeps rho in {0.2, 0.5, 1.0, 1.5, 2.0} and eps in {1e-3, 3e-3, 1e-2, 3e-2}
//! on the forced-diffusion PDE surrogate. For each (rho, eps) pair, compares:
//!   - Plain model: hard-threshold at each step, no memory
//!   - Bridge model: negative-d reservoir accumulates residual, reinjected
//!     after R consecutive steps with concurrency below M0
//!
//! PASS criterion: median improvement across all (rho, eps) pairs >= 5%.

use clap::Parser;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use spectral_core::{
    pde_surrogates::ForcedDiffusion,
    wavelet::{WaveletBridge, concurrency, haar_dwt, haar_idwt, hard_threshold},
};
use stats_core::helpers::median;
use std::fs;
use std::path::Path;

#[derive(Parser, Debug)]
#[command(author, version, about = "Thesis 1: wavelet bridging accuracy sweep")]
struct Args {
    /// Grid size (power of 2)
    #[arg(long, default_value_t = 128)]
    n: usize,
    /// Number of time steps
    #[arg(long, default_value_t = 300)]
    steps: usize,
    /// Time step size
    #[arg(long, default_value_t = 2e-3)]
    dt: f64,
    /// Diffusion coefficient
    #[arg(long, default_value_t = 0.01)]
    nu: f64,
    /// Exponential decay rate for bridge alpha = 1 - exp(-kappa*dt)
    #[arg(long, default_value_t = 6.0)]
    kappa: f64,
    /// Recovery window: steps below M0 before reinjecting
    #[arg(long, default_value_t = 5)]
    recovery_window: usize,
    /// Target concurrency M0 for bridge reinjection trigger
    #[arg(long, default_value_t = 64)]
    m0: usize,
    /// Output directory
    #[arg(long, default_value = "data/wavelet_bridging")]
    output_dir: String,
    /// RNG seed
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn l2_rel_err(u: &[f64], u_ref: &[f64]) -> f64 {
    let num: f64 = u
        .iter()
        .zip(u_ref.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    let den: f64 = u_ref.iter().map(|x| x * x).sum::<f64>().sqrt();
    if den < 1e-300 { 0.0 } else { num / den }
}

fn make_initial(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 0.1).unwrap();
    (0..n).map(|_| normal.sample(&mut rng)).collect()
}

fn main() {
    let args = Args::parse();

    let rho_vals = [0.2_f64, 0.5, 1.0, 1.5, 2.0];
    let eps_vals = [1e-3_f64, 3e-3, 1e-2, 3e-2];
    let alpha = 1.0 - (-args.kappa * args.dt).exp();

    println!("Wavelet Bridging Accuracy Sweep");
    println!(
        "  N={} steps={} dt={} nu={} kappa={} alpha={:.4} R={}",
        args.n, args.steps, args.dt, args.nu, args.kappa, alpha, args.recovery_window
    );
    println!();
    println!(
        "  {:>6}  {:>8}  {:>12}  {:>12}  {:>12}  {:>8}",
        "rho", "eps", "plain_err", "bridge_err", "improve_%", "conc"
    );

    let mut improvements: Vec<f64> = Vec::with_capacity(rho_vals.len() * eps_vals.len());
    let mut rows: Vec<String> = Vec::new();

    for (ri, &rho) in rho_vals.iter().enumerate() {
        let pde = ForcedDiffusion::new(args.n, args.dt, args.nu, rho, 1);
        for (ei, &eps) in eps_vals.iter().enumerate() {
            // Unique per-cell seed: XOR seed with hash of (ri, ei)
            let cell_seed = args.seed
                ^ ((ri as u64).wrapping_mul(0xDEAD_BEEF))
                ^ ((ei as u64).wrapping_mul(0xCAFE_BABE));
            let u0 = make_initial(args.n, cell_seed);

            // Full reference solution (no compression)
            let u_ref = pde.run(&u0, args.steps);

            // --- Plain model (advance PDE, then hard-threshold the state) ---
            let mut u_plain = u0.clone();
            let mut total_conc: usize = 0;
            for _ in 0..args.steps {
                u_plain = pde.step(&u_plain);
                let c = haar_dwt(&u_plain);
                let ct = hard_threshold(&c, eps);
                total_conc += concurrency(&c, eps);
                u_plain = haar_idwt(&ct);
            }
            let plain_err = l2_rel_err(&u_plain, &u_ref);
            let mean_conc = total_conc as f64 / args.steps as f64;

            // --- Bridge model (advance PDE + threshold + negative-d reservoir) ---
            let mut u_bridge = u0.clone();
            let mut bridge = WaveletBridge::new(args.n);
            let mut recovery_counter: usize = 0;
            for _ in 0..args.steps {
                u_bridge = pde.step(&u_bridge);
                let c = haar_dwt(&u_bridge);
                let c_thresh = hard_threshold(&c, eps);
                // residual = c - c_thresh
                let residual: Vec<f64> = c
                    .iter()
                    .zip(c_thresh.iter())
                    .map(|(&a, &b)| a - b)
                    .collect();
                bridge.accumulate(&residual);

                let m = concurrency(&c_thresh, eps);
                if m < args.m0 {
                    recovery_counter += 1;
                } else {
                    recovery_counter = 0;
                }
                let mut ct_mutable = c_thresh;
                if recovery_counter >= args.recovery_window {
                    bridge.reinject(&mut ct_mutable, alpha);
                }
                u_bridge = haar_idwt(&ct_mutable);
            }
            let bridge_err = l2_rel_err(&u_bridge, &u_ref);

            let improve_pct = if plain_err > 1e-14 {
                (plain_err - bridge_err) / plain_err * 100.0
            } else {
                0.0
            };
            improvements.push(improve_pct);

            println!(
                "  {:>6.2}  {:>8.0e}  {:>12.5e}  {:>12.5e}  {:>12.2}  {:>8.1}",
                rho, eps, plain_err, bridge_err, improve_pct, mean_conc
            );

            rows.push(format!(
                "[[result]]\nrho = {rho}\neps = {eps:.0e}\nplain_err = {plain_err:.6e}\nbridge_err = {bridge_err:.6e}\nimprovement_pct = {improve_pct:.3}\nmean_conc = {mean_conc:.1}"
            ));
        }
    }

    let mut improvements_sorted = improvements.clone();
    let med_improvement = median(&mut improvements_sorted);

    println!();
    println!("  Median improvement: {med_improvement:.2}%");
    let pass = med_improvement >= 5.0;
    println!("  Verdict: {}", if pass { "PASS" } else { "FAIL" });

    // Write TOML output
    let dir = Path::new(&args.output_dir);
    fs::create_dir_all(dir).expect("failed to create output dir");
    let toml_path = dir.join("results.toml");
    let header = format!(
        "# Wavelet Bridging Accuracy Sweep results\n# N={} steps={} dt={} nu={} kappa={} alpha={:.4} R={}\n# median_improvement_pct = {med_improvement:.3}\n# verdict = {}\n\n",
        args.n,
        args.steps,
        args.dt,
        args.nu,
        args.kappa,
        alpha,
        args.recovery_window,
        if pass { "PASS" } else { "FAIL" }
    );
    let body = rows.join("\n\n");
    fs::write(&toml_path, format!("{header}{body}\n")).expect("failed to write results.toml");
    println!("  Results written to {}", toml_path.display());

    if !pass {
        std::process::exit(1);
    }
}

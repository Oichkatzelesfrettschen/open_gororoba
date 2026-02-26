//! Thesis 2: Meltdown gating reduces p95 concurrency >= 10% vs fixed-eps baseline.
//!
//! Adaptive epsilon controller drives concurrency toward target M0 via
//! proportional threshold adjustment. Sweeps rho in {0.5, 1.0, 1.5, 2.0}.
//!
//! PASS criterion: ALL rho values show reduction_pct >= 10%.
//! rel_err_adaptive (deviation of adaptive from fixed surrogate) is reported as informational;
//! the two surrogates operate in different compression regimes, so their terminal states
//! differ substantially -- the relevant metric is the concurrency reduction, not accuracy.

use clap::Parser;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use spectral_core::{
    pde_surrogates::ForcedDiffusion,
    wavelet::{concurrency, haar_dwt, haar_idwt, hard_threshold},
};
use std::fs;
use std::path::Path;

#[derive(Parser, Debug)]
#[command(author, version, about = "Thesis 2: meltdown gating concurrency sweep")]
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
    /// Fixed baseline threshold
    #[arg(long, default_value_t = 1e-3)]
    eps0: f64,
    /// Target concurrency
    #[arg(long, default_value_t = 64)]
    m0: usize,
    /// Low watermark for adaptive decrease
    #[arg(long, default_value_t = 32)]
    m_low: usize,
    /// Multiplier when concurrency exceeds M0 (increase eps to sparsify)
    #[arg(long, default_value_t = 1.35)]
    gamma_up: f64,
    /// Multiplier when concurrency below Mlow (decrease eps to retain more)
    #[arg(long, default_value_t = 0.92)]
    gamma_down: f64,
    /// Minimum adaptive epsilon
    #[arg(long, default_value_t = 1e-4)]
    eps_min: f64,
    /// Maximum adaptive epsilon
    #[arg(long, default_value_t = 0.1)]
    eps_max: f64,
    /// Output directory
    #[arg(long, default_value = "data/meltdown_gating")]
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

fn percentile95(vals: &mut [f64]) -> f64 {
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((vals.len() as f64) * 0.95).ceil() as usize;
    vals[idx.min(vals.len() - 1)]
}

fn make_initial(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 0.1).unwrap();
    (0..n).map(|_| normal.sample(&mut rng)).collect()
}

fn main() {
    let args = Args::parse();

    let rho_vals = [0.5_f64, 1.0, 1.5, 2.0];

    println!("Meltdown Gating Concurrency Sweep");
    println!(
        "  N={} steps={} dt={} nu={}",
        args.n, args.steps, args.dt, args.nu
    );
    println!(
        "  eps0={:.0e} M0={} Mlow={} gamma_up={} gamma_down={}",
        args.eps0, args.m0, args.m_low, args.gamma_up, args.gamma_down
    );
    println!();
    println!(
        "  {:>6}  {:>10}  {:>10}  {:>12}  {:>10}",
        "rho", "p95_fixed", "p95_adapt", "reduc_%", "rel_err_a"
    );

    let mut all_pass = true;
    let mut rows: Vec<String> = Vec::new();

    for (ri, &rho) in rho_vals.iter().enumerate() {
        let pde = ForcedDiffusion::new(args.n, args.dt, args.nu, rho, 1);
        let cell_seed = args.seed ^ ((ri as u64).wrapping_mul(0xABCD_EF01));
        let u0 = make_initial(args.n, cell_seed);

        // --- Fixed-eps baseline (advance PDE, then threshold state) ---
        let mut u_fixed = u0.clone();
        let mut conc_fixed: Vec<f64> = Vec::with_capacity(args.steps);
        for _ in 0..args.steps {
            u_fixed = pde.step(&u_fixed);
            let c = haar_dwt(&u_fixed);
            conc_fixed.push(concurrency(&c, args.eps0) as f64);
            let ct = hard_threshold(&c, args.eps0);
            u_fixed = haar_idwt(&ct);
        }
        let p95_fixed = percentile95(&mut conc_fixed);

        // --- Adaptive gating (advance PDE, then adaptive threshold) ---
        let mut u_adapt = u0.clone();
        let mut eps_n = args.eps0;
        let mut conc_adapt: Vec<f64> = Vec::with_capacity(args.steps);
        for _ in 0..args.steps {
            u_adapt = pde.step(&u_adapt);
            let c = haar_dwt(&u_adapt);
            let m = concurrency(&c, eps_n);
            conc_adapt.push(m as f64);
            // Adjust eps for next step
            if m > args.m0 {
                eps_n = (eps_n * args.gamma_up).min(args.eps_max);
            } else if m < args.m_low {
                eps_n = (eps_n * args.gamma_down).max(args.eps_min);
            }
            let ct = hard_threshold(&c, eps_n);
            u_adapt = haar_idwt(&ct);
        }
        let p95_adapt = percentile95(&mut conc_adapt);
        // Measure how much the adaptive surrogate deviates from the fixed surrogate
        // (not from the full uncompressed PDE, since both surrogates have compression error).
        let rel_err_adapt = l2_rel_err(&u_adapt, &u_fixed);
        let reduction_pct = (p95_fixed - p95_adapt) / p95_fixed * 100.0;

        let pass_row = reduction_pct >= 10.0;
        if !pass_row {
            all_pass = false;
        }

        println!(
            "  {:>6.2}  {:>10.1}  {:>10.1}  {:>12.2}  {:>10.4}  {}",
            rho,
            p95_fixed,
            p95_adapt,
            reduction_pct,
            rel_err_adapt,
            if pass_row { "OK" } else { "FAIL" }
        );

        rows.push(format!(
            "[[result]]\nrho = {rho}\np95_fixed = {p95_fixed:.1}\np95_adaptive = {p95_adapt:.1}\nreduction_pct = {reduction_pct:.3}\nrel_err_adaptive = {rel_err_adapt:.6e}\npass = {pass_row}"
        ));
    }

    println!();
    println!("  Verdict: {}", if all_pass { "PASS" } else { "FAIL" });

    let dir = Path::new(&args.output_dir);
    fs::create_dir_all(dir).expect("failed to create output dir");
    let toml_path = dir.join("results.toml");
    let header = format!(
        "# Meltdown Gating Concurrency results\n# N={} steps={} eps0={:.0e} M0={} Mlow={}\n# verdict = {}\n\n",
        args.n,
        args.steps,
        args.eps0,
        args.m0,
        args.m_low,
        if all_pass { "PASS" } else { "FAIL" }
    );
    fs::write(&toml_path, format!("{}{}\n", header, rows.join("\n\n")))
        .expect("failed to write results.toml");
    println!("  Results written to {}", toml_path.display());

    if !all_pass {
        std::process::exit(1);
    }
}

//! Reframed Thesis 4: Pulsed vs steady entropy injection comparison.
//!
//! Tests whether burst-quench cycles accumulate less persistent entropy
//! than steady injection at equal total forcing energy.  Quench phases
//! allow partial relaxation via pure diffusion.
//!
//! Three protocols at matched total forcing energy:
//!   A) Steady:     rho constant for N steps
//!   B) Pulsed 50%: rho*2 for N/2 steps (on), 0 for N/2 steps (off)
//!   C) Pulsed 25%: rho*4 for N/4 steps (on), 0 for 3N/4 steps (off)
//!
//! PASS: pulsed p95 entropy < steady p95 entropy by >10%.
//! FAIL: pulsed p95 entropy >= steady p95 entropy (within 5% tolerance).

use clap::Parser;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use spectral_core::{
    pde_surrogates::ForcedDiffusion,
    wavelet::{concurrency, haar_dwt, hard_threshold},
};
use std::fs;
use std::path::Path;

#[derive(Parser, Debug)]
#[command(author, version, about = "Reframed T4: pulsed vs steady entropy injection")]
struct Args {
    /// Grid size (power of 2, must satisfy CFL: nu*dt*N^2 <= 0.695)
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
    /// Base forcing amplitude (steady protocol)
    #[arg(long, default_value_t = 1.0)]
    rho_base: f64,
    /// Wavelet threshold levels to test
    #[arg(long, default_values_t = vec![0.01, 0.1, 1.0])]
    eps: Vec<f64>,
    /// Output directory
    #[arg(long, default_value = "data/meltdown_pulsed")]
    output_dir: String,
    /// RNG seed
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn percentile95(vals: &mut [f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((vals.len() as f64) * 0.95).ceil() as usize;
    vals[idx.min(vals.len() - 1)]
}

fn make_initial(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 0.1).unwrap();
    (0..n).map(|_| normal.sample(&mut rng)).collect()
}

/// Run a simulation and collect wavelet concurrency at each step.
/// Returns vector of concurrency values at each step.
fn run_protocol(
    pde: &ForcedDiffusion,
    u0: &[f64],
    schedule: &[bool],
    eps: f64,
) -> Vec<f64> {
    let mut u = u0.to_vec();
    let mut conc_trace = Vec::with_capacity(schedule.len());
    for &active in schedule {
        u = pde.step_pulsed(&u, active);
        let c = haar_dwt(&u);
        conc_trace.push(concurrency(&c, eps) as f64);
        let ct = hard_threshold(&c, eps);
        // Reconstruct from compressed representation
        u = spectral_core::wavelet::haar_idwt(&ct);
    }
    conc_trace
}

struct ProtocolResult {
    name: String,
    duty_cycle: f64,
    rho_effective: f64,
    p95_by_eps: Vec<(f64, f64)>,
}

fn main() {
    let args = Args::parse();

    // CFL stability check
    let r = args.nu * args.dt * (args.n as f64) * (args.n as f64);
    assert!(
        r <= 0.695,
        "CFL violation: r = nu*dt*N^2 = {:.3} > 0.695. Reduce N, dt, or nu.",
        r
    );

    println!("Meltdown Pulsed Gate Experiment");
    println!("  N={} steps={} dt={} nu={} rho_base={}", args.n, args.steps, args.dt, args.nu, args.rho_base);
    println!("  CFL parameter r = {:.3}", r);
    println!();

    let u0 = make_initial(args.n, args.seed);

    // Define three protocols with equal total forcing energy:
    // Total energy ~ rho * active_steps
    // Steady:     rho_base * N
    // Pulsed 50%: (rho_base*2) * (N/2) = rho_base * N
    // Pulsed 25%: (rho_base*4) * (N/4) = rho_base * N
    let protocols: Vec<(String, f64, f64)> = vec![
        ("Steady".to_string(), 1.0, args.rho_base),
        ("Pulsed_50".to_string(), 0.5, args.rho_base * 2.0),
        ("Pulsed_25".to_string(), 0.25, args.rho_base * 4.0),
    ];

    let mut results: Vec<ProtocolResult> = Vec::new();

    for (name, duty, rho_eff) in &protocols {
        let schedule = if *duty >= 1.0 {
            vec![true; args.steps]
        } else {
            ForcedDiffusion::pulsed_schedule(args.steps, *duty)
        };

        let pde = ForcedDiffusion::new(args.n, args.dt, args.nu, *rho_eff, 1);

        let mut p95_by_eps = Vec::new();
        for &eps in &args.eps {
            let mut conc = run_protocol(&pde, &u0, &schedule, eps);
            let p95 = percentile95(&mut conc);
            p95_by_eps.push((eps, p95));
        }

        results.push(ProtocolResult {
            name: name.clone(),
            duty_cycle: *duty,
            rho_effective: *rho_eff,
            p95_by_eps,
        });
    }

    // Print results table
    println!("  {:>12}  {:>6}  {:>10}", "protocol", "eps", "p95_conc");
    println!("  {:->12}  {:->6}  {:->10}", "", "", "");
    for r in &results {
        for (eps, p95) in &r.p95_by_eps {
            println!("  {:>12}  {:>6.0e}  {:>10.1}", r.name, eps, p95);
        }
    }

    // Compute reduction percentages: pulsed vs steady
    println!();
    println!("  Reduction vs Steady:");
    println!("  {:>12}  {:>6}  {:>10}  {:>8}", "protocol", "eps", "reduc_%", "status");
    println!("  {:->12}  {:->6}  {:->10}  {:->8}", "", "", "", "");

    let steady = &results[0];
    let mut any_pass = false;
    let mut any_fail = false;
    let mut gate_rows: Vec<String> = Vec::new();

    for pulsed in &results[1..] {
        for (i, (eps, p95_pulsed)) in pulsed.p95_by_eps.iter().enumerate() {
            let p95_steady = steady.p95_by_eps[i].1;
            let reduction = if p95_steady > 0.0 {
                (p95_steady - p95_pulsed) / p95_steady * 100.0
            } else {
                0.0
            };
            // PASS if reduction > 10%, FAIL if reduction < -5% (pulsed is worse)
            let status = if reduction > 10.0 {
                any_pass = true;
                "PASS"
            } else if reduction < -5.0 {
                any_fail = true;
                "FAIL"
            } else {
                "INCONC"
            };
            println!(
                "  {:>12}  {:>6.0e}  {:>10.2}  {:>8}",
                pulsed.name, eps, reduction, status
            );
            gate_rows.push(format!(
                "[[gate]]\nprotocol = \"{}\"\neps = {:.0e}\np95_steady = {:.1}\np95_pulsed = {:.1}\nreduction_pct = {:.3}\nstatus = \"{}\"",
                pulsed.name, eps, p95_steady, p95_pulsed, reduction, status
            ));
        }
    }

    let verdict = if any_pass {
        "PASS"
    } else if any_fail {
        "FAIL"
    } else {
        "INCONCLUSIVE"
    };
    println!();
    println!("  Verdict: {}", verdict);

    // Write results
    let dir = Path::new(&args.output_dir);
    fs::create_dir_all(dir).expect("failed to create output dir");
    let toml_path = dir.join("results.toml");
    let header = format!(
        "# Meltdown Pulsed Gate results\n# N={} steps={} rho_base={} nu={}\n# verdict = {}\n\n",
        args.n, args.steps, args.rho_base, args.nu, verdict
    );
    let body: String = results
        .iter()
        .map(|r| {
            let eps_str: Vec<String> = r.p95_by_eps.iter().map(|(e, p)| format!("  {{ eps = {:.0e}, p95 = {:.1} }}", e, p)).collect();
            format!(
                "[[protocol]]\nname = \"{}\"\nduty_cycle = {}\nrho_effective = {}\np95 = [\n{}\n]",
                r.name, r.duty_cycle, r.rho_effective, eps_str.join(",\n")
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    let gates = gate_rows.join("\n\n");
    fs::write(&toml_path, format!("{}{}\n\n{}\n", header, body, gates))
        .expect("failed to write results.toml");
    println!("  Results written to {}", toml_path.display());

    if any_fail {
        std::process::exit(1);
    }
}

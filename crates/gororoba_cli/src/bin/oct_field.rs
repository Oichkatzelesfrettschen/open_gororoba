//! oct-field: Octonionic scalar field simulator.
//!
//! Simulates an octonion-valued scalar field on a 1D periodic lattice
//! using a Stormer-Verlet symplectic integrator. Demonstrates:
//! - Energy conservation (symplectic property)
//! - Noether charge conservation (internal U(1)^7 symmetry)
//! - Correct dispersion relation (omega^2 = k^2 + m^2)
//!
//! Usage:
//!   oct-field evolve --n 64 --t-final 10.0 --mass 1.0 --coupling 0.0
//!   oct-field dispersion --n 128 --n-modes 5 --mass 1.0
//!   oct-field sweep --n-min 32 --n-max 256 --mass 1.0

use clap::{Parser, Subcommand};
use algebra_core::{
    FieldParams, evolve, gaussian_wave_packet, measure_dispersion,
};
use std::f64::consts::PI;

#[derive(Parser)]
#[command(name = "oct-field")]
#[command(about = "Octonionic scalar field simulator")]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Evolve a Gaussian wave packet and report energy/charge conservation
    Evolve {
        /// Number of lattice sites
        #[arg(long, default_value = "64")]
        n: usize,

        /// Domain length
        #[arg(long, default_value = "6.283185307179586")]
        domain: f64,

        /// Final time
        #[arg(long, default_value = "10.0")]
        t_final: f64,

        /// Scalar mass
        #[arg(long, default_value = "1.0")]
        mass: f64,

        /// Quartic coupling (lambda)
        #[arg(long, default_value = "0.0")]
        coupling: f64,

        /// Time step
        #[arg(long, default_value = "0.01")]
        dt: f64,

        /// Output CSV file
        #[arg(short, long)]
        output: Option<String>,

        /// JSON output
        #[arg(long)]
        json: bool,
    },

    /// Measure dispersion relation for lowest modes
    Dispersion {
        /// Number of lattice sites
        #[arg(long, default_value = "128")]
        n: usize,

        /// Domain length
        #[arg(long, default_value = "6.283185307179586")]
        domain: f64,

        /// Number of modes to measure
        #[arg(long, default_value = "5")]
        n_modes: usize,

        /// Scalar mass
        #[arg(long, default_value = "1.0")]
        mass: f64,

        /// Output CSV file
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Sweep lattice sizes for convergence analysis
    Sweep {
        /// Minimum lattice size
        #[arg(long, default_value = "32")]
        n_min: usize,

        /// Maximum lattice size
        #[arg(long, default_value = "256")]
        n_max: usize,

        /// Number of lattice sizes to test
        #[arg(long, default_value = "5")]
        n_samples: usize,

        /// Scalar mass
        #[arg(long, default_value = "1.0")]
        mass: f64,

        /// Output CSV file
        #[arg(short, long)]
        output: Option<String>,
    },
}

fn main() {
    let args = Args::parse();

    match args.command {
        Commands::Evolve { n, domain, t_final, mass, coupling, dt, output, json } => {
            run_evolve(n, domain, t_final, mass, coupling, dt, output, json);
        }
        Commands::Dispersion { n, domain, n_modes, mass, output } => {
            run_dispersion(n, domain, n_modes, mass, output);
        }
        Commands::Sweep { n_min, n_max, n_samples, mass, output } => {
            run_sweep(n_min, n_max, n_samples, mass, output);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_evolve(
    n: usize,
    domain: f64,
    t_final: f64,
    mass: f64,
    coupling: f64,
    dt: f64,
    output: Option<String>,
    json: bool,
) {
    let params = FieldParams {
        n,
        l: domain,
        mass,
        coupling,
        dt,
    };

    let n_steps = (t_final / dt) as usize;
    let (phi0, pi0) = gaussian_wave_packet(&params);
    let result = evolve(&phi0, &pi0, &params, n_steps);

    let e0 = result.energies[0];
    let e_final = result.energies.last().copied().unwrap_or(e0);
    let max_drift: f64 = result.energies.iter()
        .map(|&e| (e - e0).abs() / e0)
        .fold(0.0_f64, f64::max);

    let charge_drifts: Vec<f64> = result.charges_initial.iter()
        .zip(result.charges_final.iter())
        .map(|(q0, qf)| {
            let scale = q0.abs().max(1e-10);
            (qf - q0).abs() / scale
        })
        .collect();
    let max_charge_drift: f64 = charge_drifts.iter().copied().fold(0.0_f64, f64::max);

    if json {
        println!("{{");
        println!("  \"n_sites\": {},", n);
        println!("  \"domain\": {},", domain);
        println!("  \"mass\": {},", mass);
        println!("  \"coupling\": {},", coupling);
        println!("  \"dt\": {},", dt);
        println!("  \"n_steps\": {},", n_steps);
        println!("  \"t_final\": {},", t_final);
        println!("  \"e_initial\": {},", e0);
        println!("  \"e_final\": {},", e_final);
        println!("  \"max_energy_drift\": {},", max_drift);
        println!("  \"charges_initial\": {:?},", result.charges_initial);
        println!("  \"charges_final\": {:?},", result.charges_final);
        println!("  \"max_charge_drift\": {}", max_charge_drift);
        println!("}}");
    } else {
        println!("Octonionic Scalar Field Evolution");
        println!("==================================");
        println!("  N = {} sites, L = {:.4}", n, domain);
        println!("  mass = {}, coupling = {}", mass, coupling);
        println!("  dt = {}, steps = {}, T = {:.4}", dt, n_steps, t_final);
        println!();
        println!("Energy:");
        println!("  E(0) = {:.6}", e0);
        println!("  E(T) = {:.6}", e_final);
        println!("  max|dE/E| = {:.2e}", max_drift);
        println!();
        println!("Noether Charges Q_k (k=1..7):");
        println!("  Initial: {:?}", result.charges_initial.map(|x| format!("{:.4}", x)));
        println!("  Final:   {:?}", result.charges_final.map(|x| format!("{:.4}", x)));
        println!("  max|dQ/Q| = {:.2e}", max_charge_drift);
    }

    if let Some(path) = output {
        let mut wtr = csv::Writer::from_path(&path).expect("Failed to create CSV");
        wtr.write_record(["step", "time", "energy"]).unwrap();
        for (i, &e) in result.energies.iter().enumerate() {
            wtr.write_record(&[
                i.to_string(),
                (i as f64 * dt).to_string(),
                e.to_string(),
            ]).unwrap();
        }
        wtr.flush().unwrap();
        eprintln!("Wrote {} energy samples to {}", result.energies.len(), path);
    }
}

fn run_dispersion(n: usize, domain: f64, n_modes: usize, mass: f64, output: Option<String>) {
    let params = FieldParams {
        n,
        l: domain,
        mass,
        coupling: 0.0,
        dt: 0.01,
    };

    println!("Measuring Dispersion Relation");
    println!("=============================");
    println!("  N = {} sites, L = {:.4}, mass = {}", n, domain, mass);
    println!();

    let results = measure_dispersion(&params, n_modes);

    println!("mode  k          omega_exact  omega_meas   rel_error");
    println!("----  ---------  -----------  -----------  ---------");
    for r in &results {
        println!("{:4}  {:9.4}  {:11.6}  {:11.6}  {:9.2e}",
            r.mode, r.k, r.omega_exact, r.omega_measured, r.rel_error);
    }

    let avg_error: f64 = results.iter().map(|r| r.rel_error).sum::<f64>() / results.len() as f64;
    println!();
    println!("Average relative error: {:.2e}", avg_error);

    if let Some(path) = output {
        let mut wtr = csv::Writer::from_path(&path).expect("Failed to create CSV");
        wtr.write_record(["mode", "k", "omega_exact", "omega_measured", "rel_error"]).unwrap();
        for r in &results {
            wtr.write_record(&[
                r.mode.to_string(),
                r.k.to_string(),
                r.omega_exact.to_string(),
                r.omega_measured.to_string(),
                r.rel_error.to_string(),
            ]).unwrap();
        }
        wtr.flush().unwrap();
        eprintln!("Wrote {} modes to {}", results.len(), path);
    }
}

fn run_sweep(n_min: usize, n_max: usize, n_samples: usize, mass: f64, output: Option<String>) {
    println!("Lattice Size Convergence Sweep");
    println!("==============================");
    println!("  N = {} to {}, samples = {}, mass = {}", n_min, n_max, n_samples, mass);
    println!();

    let sizes: Vec<usize> = (0..n_samples)
        .map(|i| {
            let frac = i as f64 / (n_samples - 1).max(1) as f64;
            (n_min as f64 + frac * (n_max - n_min) as f64) as usize
        })
        .collect();

    println!("N       mode1_error   mode2_error   energy_drift");
    println!("------  -----------   -----------   ------------");

    let mut records = Vec::new();

    for &n in &sizes {
        let params = FieldParams {
            n,
            l: 2.0 * PI,
            mass,
            coupling: 0.0,
            dt: 0.01,
        };

        // Measure dispersion for first 2 modes
        let disp = measure_dispersion(&params, 2);
        let err1 = disp.first().map(|r| r.rel_error).unwrap_or(f64::NAN);
        let err2 = disp.get(1).map(|r| r.rel_error).unwrap_or(f64::NAN);

        // Measure energy drift
        let (phi0, pi0) = gaussian_wave_packet(&params);
        let result = evolve(&phi0, &pi0, &params, 100);
        let e0 = result.energies[0];
        let max_drift: f64 = result.energies.iter()
            .map(|&e| (e - e0).abs() / e0)
            .fold(0.0_f64, f64::max);

        println!("{:6}  {:11.2e}   {:11.2e}   {:12.2e}", n, err1, err2, max_drift);
        records.push((n, err1, err2, max_drift));
    }

    if let Some(path) = output {
        let mut wtr = csv::Writer::from_path(&path).expect("Failed to create CSV");
        wtr.write_record(["n_sites", "mode1_error", "mode2_error", "energy_drift"]).unwrap();
        for (n, e1, e2, drift) in &records {
            wtr.write_record(&[
                n.to_string(),
                e1.to_string(),
                e2.to_string(),
                drift.to_string(),
            ]).unwrap();
        }
        wtr.flush().unwrap();
        eprintln!("Wrote {} records to {}", records.len(), path);
    }
}

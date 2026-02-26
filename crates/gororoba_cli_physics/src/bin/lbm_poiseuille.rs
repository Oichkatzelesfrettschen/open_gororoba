//! lbm-poiseuille: D2Q9 Lattice Boltzmann Poiseuille flow simulation.
//!
//! Usage: lbm-poiseuille --ny 41 --tau 0.8 --steps 10000 --output profile.csv

use clap::Parser;
use lbm_core::simulate_poiseuille;

#[derive(Parser)]
#[command(name = "lbm-poiseuille")]
#[command(about = "Simulate Poiseuille flow using D2Q9 Lattice Boltzmann Method")]
struct Args {
    /// Grid size in x (periodic direction)
    #[arg(long, default_value = "3")]
    nx: usize,

    /// Grid size in y (wall-bounded direction)
    #[arg(long, default_value = "41")]
    ny: usize,

    /// BGK relaxation time (must be > 0.5 for stability)
    #[arg(long, default_value = "0.8")]
    tau: f64,

    /// Body force in x-direction
    #[arg(long, default_value = "1e-5")]
    fx: f64,

    /// Number of LBM time steps
    #[arg(short, long, default_value = "10000")]
    steps: usize,

    /// Output CSV file for velocity profile
    #[arg(short, long)]
    output: Option<String>,

    /// Output as JSON summary
    #[arg(long)]
    json: bool,
}

fn main() {
    let args = Args::parse();

    if args.tau <= 0.5 {
        eprintln!("Warning: tau={} <= 0.5 may cause instability", args.tau);
    }

    eprintln!(
        "LBM Poiseuille: {}x{} grid, tau={}, fx={}, {} steps",
        args.nx, args.ny, args.tau, args.fx, args.steps
    );

    let result = simulate_poiseuille(args.nx, args.ny, args.tau, args.fx, args.steps);

    // Compute statistics
    let nu = (args.tau - 0.5) / 3.0;
    let max_numerical: f64 = result.ux_numerical[1..args.ny - 1]
        .iter()
        .cloned()
        .fold(0.0, f64::max);
    let max_analytical: f64 = result.ux_analytical[1..args.ny - 1]
        .iter()
        .cloned()
        .fold(0.0, f64::max);

    eprintln!("Kinematic viscosity: nu = {:.6e}", nu);
    eprintln!("Max numerical velocity: {:.6e}", max_numerical);
    eprintln!("Max analytical velocity: {:.6e}", max_analytical);
    eprintln!("Max relative error: {:.4}%", result.max_rel_error * 100.0);

    // Check mass conservation
    let mass_initial = result.mass_history[0];
    let mass_final = *result.mass_history.last().unwrap();
    let mass_drift = (mass_final - mass_initial).abs() / mass_initial;
    eprintln!("Mass drift: {:.6e}", mass_drift);

    if args.json {
        println!("{{");
        println!("  \"nx\": {},", args.nx);
        println!("  \"ny\": {},", args.ny);
        println!("  \"tau\": {},", args.tau);
        println!("  \"nu\": {},", nu);
        println!("  \"fx\": {},", args.fx);
        println!("  \"steps\": {},", args.steps);
        println!("  \"max_numerical\": {},", max_numerical);
        println!("  \"max_analytical\": {},", max_analytical);
        println!("  \"max_rel_error\": {},", result.max_rel_error);
        println!("  \"mass_drift\": {}", mass_drift);
        println!("}}");
    } else if let Some(path) = &args.output {
        let mut wtr = csv::Writer::from_path(path).expect("Failed to create CSV");
        wtr.write_record(["y", "ux_numerical", "ux_analytical", "error"])
            .unwrap();
        for i in 0..args.ny {
            let error = result.ux_numerical[i] - result.ux_analytical[i];
            wtr.write_record(&[
                result.y[i].to_string(),
                result.ux_numerical[i].to_string(),
                result.ux_analytical[i].to_string(),
                error.to_string(),
            ])
            .unwrap();
        }
        wtr.flush().unwrap();
        println!("Wrote {} points to {}", args.ny, path);
    } else {
        println!("y,ux_numerical,ux_analytical");
        for i in 0..args.ny {
            println!(
                "{},{},{}",
                result.y[i], result.ux_numerical[i], result.ux_analytical[i]
            );
        }
    }
}

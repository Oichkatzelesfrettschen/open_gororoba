//! Demonstration: E-027 with REAL materials properties
//!
//! This binary shows how to ground the frustration-viscosity experiment in
//! real physical materials instead of arbitrary parameters.
//!
//! Example usage:
//! ```bash
//! cargo run --bin demo-real-materials -- --material He4_normal --grid-size 16
//! cargo run --bin demo-real-materials -- --material water_20C --length-scale 1e-5
//! cargo run --bin demo-real-materials -- --material air_20C --length-scale 0.01
//! ```

use clap::Parser;
use materials_core::{get_viscosity_material, list_viscosity_materials, reynolds_number, to_lattice_units};

#[derive(Parser)]
#[command(name = "demo-real-materials")]
#[command(about = "Demonstrate E-027 with real materials properties")]
struct Args {
    /// Material ID from registry (use --list to see available)
    #[arg(long)]
    material: Option<String>,

    /// List all available materials
    #[arg(long, default_value = "false")]
    list: bool,

    /// Grid size (cubic domain)
    #[arg(long, default_value = "16")]
    grid_size: usize,

    /// Physical length scale: 1 grid cell = ? meters
    #[arg(long, default_value = "1e-5")]
    length_scale: f64,

    /// Physical time scale: 1 LBM timestep = ? seconds
    #[arg(long, default_value = "1e-7")]
    time_scale: f64,

    /// Characteristic velocity (m/s) for Reynolds number
    #[arg(long, default_value = "0.1")]
    velocity: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    if args.list {
        println!("Available materials:");
        for id in list_viscosity_materials() {
            println!("  {}", id);
        }
        return Ok(());
    }

    let material_id = args.material.as_ref().ok_or("Must specify --material or use --list")?;
    let material = get_viscosity_material(material_id)
        .ok_or_else(|| format!("Material '{}' not found", material_id))?;

    println!("========================================");
    println!("Material: {} ({})", material.name, material.id);
    println!("========================================");
    println!("Formula: {}", material.formula);
    println!("Phase: {}", material.phase);
    println!("Temperature: {:.2} K", material.temperature_K);
    println!("Pressure: {:.2e} Pa", material.pressure_Pa);
    println!("Density: {:.2} kg/m^3", material.density_kg_m3);

    if let Some(nu) = material.kinematic_viscosity_m2_s {
        println!("\nViscosity Properties:");
        println!("  Kinematic: {:.6e} m^2/s", nu);
        if let Some(eta) = material.dynamic_viscosity_Pa_s {
            println!("  Dynamic: {:.6e} Pa*s", eta);
        }

        // Convert to lattice units
        let nu_lattice = to_lattice_units(nu, args.length_scale, args.time_scale);
        let tau = 3.0 * nu_lattice + 0.5;

        println!("\nLBM Parameters:");
        println!("  Length scale (dx): {:.2e} m", args.length_scale);
        println!("  Time scale (dt): {:.2e} s", args.time_scale);
        println!("  Grid size: {}", args.grid_size);
        println!("  Physical domain: {:.2e} m", args.grid_size as f64 * args.length_scale);
        println!("  Kinematic viscosity (lattice): {:.6e}", nu_lattice);
        println!("  Relaxation time (tau): {:.6}", tau);

        if tau < 0.5 {
            println!("  WARNING: tau < 0.5 is unstable! Increase length or time scale.");
        } else if tau > 2.0 {
            println!("  NOTE: tau > 2.0 may be overly diffusive. Consider decreasing scales.");
        }

        // Compute Reynolds number
        let length_physical = args.grid_size as f64 * args.length_scale;
        let re = reynolds_number(args.velocity, length_physical, nu);

        println!("\nFlow Regime:");
        println!("  Characteristic velocity: {:.2e} m/s", args.velocity);
        println!("  Reynolds number: {:.2}", re);
        if re < 2000.0 {
            println!("  Regime: LAMINAR");
        } else if re < 4000.0 {
            println!("  Regime: TRANSITIONAL");
        } else {
            println!("  Regime: TURBULENT");
        }

        // Estimate evolution time
        let n_timesteps = 1000;
        let physical_time = n_timesteps as f64 * args.time_scale;
        let flow_time = length_physical / args.velocity;

        println!("\nTiming Estimates:");
        println!("  {} LBM steps = {:.2e} seconds", n_timesteps, physical_time);
        println!("  Flow-through time: {:.2e} seconds", flow_time);
        println!("  Steps per flow-through: {:.0}", flow_time / args.time_scale);
    } else {
        println!("\nNo viscosity data (solid phase)");
        if let Some(creep) = material.creep_viscosity_Pa_s {
            println!("Creep viscosity: {:.2e} Pa*s", creep);
        }
    }

    if let Some(ref notes) = material.notes {
        println!("\nNotes: {}", notes);
    }

    if let Some(ref reference) = material.reference {
        println!("Reference: {}", reference);
    }

    println!("\n========================================");
    println!("Next Steps:");
    println!("========================================");
    println!("1. Run E-027 with these parameters:");
    println!("   cargo run --bin percolation-experiment --");
    println!("     --grid-size {} \\", args.grid_size);
    println!("     --nu-base {:.6e} \\", nu_lattice);
    println!("     --lambda <derived from theory> \\");
    println!("     --lbm-steps {} \\", (flow_time / args.time_scale) as usize);
    println!("     --forcing-mode <appropriate for Re={:.0}>", re);
    println!();
    println!("2. Derive lambda from statistical mechanics:");
    println!("   lambda ~ (structural disorder energy) / (thermal energy)");
    println!("   For frustration: lambda ~ E_frustration / k_B T");
    println!();
    println!("3. Compare to experimental data:");
    println!("   Validate percolation threshold predictions against");
    println!("   microfluidic experiments or DNS simulations");

    Ok(())
}

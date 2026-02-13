//! Thesis 2: 3D Associator-Coupled Shear Thickening
//!
//! Tests whether algebraic non-associativity in Cayley-Dickson sedenions
//! can produce genuine shear thickening in 3D LBM flow.
//!
//! Pipeline:
//! 1. Generate SedenionField with spatial variation
//! 2. Compute local associator norm field (non-associativity strength)
//! 3. Apply Kolmogorov shear forcing
//! 4. For each (alpha, power_index) pair:
//!    a. Newtonian reference: evolve_non_newtonian with zero coupling
//!    b. Non-Newtonian: coupling = alpha * associator_norm
//!    c. Compare max velocities: vel_reduction = 1 - (non_newton / newton)
//! 5. Gate: vel_reduction > 0.05 for at least one parameter pair
//! 6. Output TOML + CSV

use clap::Parser;
use lbm_3d::solver::LbmSolver3D;
use std::fmt::Write as _;
use vacuum_frustration::bridge::SedenionField;

#[derive(Parser, Debug)]
#[command(name = "thesis2-3d-thickening")]
#[command(about = "Thesis 2: 3D associator-coupled shear thickening")]
struct Args {
    /// Grid size per axis (N^3 cells)
    #[arg(long, default_value = "16")]
    grid_size: usize,

    /// Number of LBM steps
    #[arg(long, default_value = "500")]
    lbm_steps: usize,

    /// Base kinematic viscosity
    #[arg(long, default_value = "0.05")]
    nu_base: f64,

    /// Force amplitude for Kolmogorov forcing
    #[arg(long, default_value = "5e-4")]
    force_amp: f64,

    /// Alpha values (coupling strengths), comma-separated
    #[arg(long, default_value = "0.0,0.5,1.0,5.0,20.0,100.0")]
    alphas: String,

    /// Power-law indices, comma-separated
    #[arg(long, default_value = "1.2,1.5,2.0")]
    power_indices: String,

    /// Output directory
    #[arg(long, default_value = "data/thesis_lab/thesis2_3d")]
    output_dir: String,
}

/// Result for one (alpha, power_index) pair.
struct ThickeningResult {
    alpha: f64,
    power_index: f64,
    newton_max_vel: f64,
    non_newton_max_vel: f64,
    vel_reduction: f64,
    tau_min: f64,
    tau_max: f64,
    tau_range: f64,
    mean_strain_rate: f64,
}

/// Generate SedenionField with spatial variation (same as e027_v2 for consistency).
fn generate_sedenion_field(nx: usize) -> SedenionField {
    let mut field = SedenionField::uniform(nx, nx, nx);
    let mut state = 42_u64;
    let mut next_rand = || -> f64 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state as f64) / (u64::MAX as f64) * 2.0 - 1.0
    };
    let pi2 = std::f64::consts::PI * 2.0;
    for z in 0..nx {
        for y in 0..nx {
            for x in 0..nx {
                let s = field.get_mut(x, y, z);
                let xn = x as f64 / nx as f64;
                let yn = y as f64 / nx as f64;
                let zn = z as f64 / nx as f64;
                s[1] = 0.3 * (pi2 * xn).sin();
                s[3] = 0.2 * (pi2 * 2.0 * yn).cos();
                s[5] = 0.15 * (pi2 * xn + pi2 * zn).sin();
                s[7] = 0.15 * zn;
                s[9] = 0.1 * (pi2 * 3.0 * xn).cos();
                s[11] = 0.1 * (pi2 * yn * 2.0).sin();
                for component in s.iter_mut().take(16) {
                    *component += 0.05 * next_rand();
                }
            }
        }
    }
    field
}

/// Compute max velocity magnitude from LBM solver.
fn max_velocity(solver: &LbmSolver3D) -> f64 {
    let (nx, ny, nz) = (solver.nx, solver.ny, solver.nz);
    let mut max_vel = 0.0_f64;
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let (_rho, u) = solver.get_macroscopic(x, y, z);
                let mag = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
                max_vel = max_vel.max(mag);
            }
        }
    }
    max_vel
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let nx = args.grid_size;
    let n_cells = nx * nx * nx;

    let alphas: Vec<f64> = args
        .alphas
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let power_indices: Vec<f64> = args
        .power_indices
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    println!("Thesis 2: 3D Associator-Coupled Shear Thickening");
    println!("=================================================");
    println!("Grid: {}^3 ({} cells)", nx, n_cells);
    println!("LBM steps: {}, nu_base: {:.4}", args.lbm_steps, args.nu_base);
    println!("Force amplitude: {:.2e}", args.force_amp);
    println!("Alphas: {:?}", alphas);
    println!("Power indices: {:?}", power_indices);
    println!();

    // Step 1: Generate SedenionField and compute associator norms
    println!("[1/4] Generating SedenionField and associator norms...");
    let field = generate_sedenion_field(nx);
    let assoc_norms = field.local_associator_norm_field(16);
    drop(field);

    let mean_assoc = assoc_norms.iter().sum::<f64>() / n_cells as f64;
    let max_assoc = assoc_norms.iter().cloned().fold(0.0_f64, f64::max);
    println!(
        "  Associator norm: mean={:.6}, max={:.6}",
        mean_assoc, max_assoc
    );

    // Step 2: Set up Kolmogorov forcing
    println!("[2/4] Configuring Kolmogorov forcing...");
    let pi2 = std::f64::consts::PI * 2.0;
    let force_field: Vec<[f64; 3]> = (0..n_cells)
        .map(|idx| {
            let y = (idx / nx) % nx;
            let yn = y as f64 / nx as f64;
            [args.force_amp * (pi2 * yn).sin(), 0.0, 0.0]
        })
        .collect();

    // Step 3: Sweep over (alpha, power_index) pairs
    println!(
        "[3/4] Running {} parameter pairs...",
        alphas.len() * power_indices.len()
    );
    let mut results = Vec::new();
    let mut pair_idx = 0usize;
    let total_pairs = alphas.len() * power_indices.len();

    for &alpha in &alphas {
        for &power_index in &power_indices {
            pair_idx += 1;
            print!(
                "  [{}/{}] alpha={:.1}, n={:.1}... ",
                pair_idx, total_pairs, alpha, power_index
            );

            // Build coupling field: coupling = alpha * associator_norm
            let coupling: Vec<f64> = assoc_norms.iter().map(|&a| alpha * a).collect();

            // Newtonian reference (zero coupling)
            let zero_coupling = vec![0.0; n_cells];
            let default_tau = 3.0 * args.nu_base + 0.5;
            let mut newton_solver = LbmSolver3D::new(nx, nx, nx, default_tau);
            newton_solver
                .set_force_field(force_field.clone())
                .expect("force field");
            newton_solver.evolve_non_newtonian(
                args.lbm_steps,
                &zero_coupling,
                args.nu_base,
                power_index,
                0.505,
                3.0,
            );
            newton_solver.compute_macroscopic();
            let newton_max = max_velocity(&newton_solver);

            // Non-Newtonian: coupling = alpha * assoc_norm
            let mut nn_solver = LbmSolver3D::new(nx, nx, nx, default_tau);
            nn_solver
                .set_force_field(force_field.clone())
                .expect("force field");
            let strain_rate = nn_solver.evolve_non_newtonian(
                args.lbm_steps,
                &coupling,
                args.nu_base,
                power_index,
                0.505,
                3.0,
            );
            nn_solver.compute_macroscopic();
            let nn_max = max_velocity(&nn_solver);

            // Tau field statistics from the non-Newtonian solver
            let tau_f = nn_solver.collider.get_tau_field();
            let t_min = tau_f.iter().cloned().fold(f64::INFINITY, f64::min);
            let t_max = tau_f.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            let mean_sr = if strain_rate.is_empty() {
                0.0
            } else {
                strain_rate.iter().sum::<f64>() / strain_rate.len() as f64
            };

            let vel_reduction = if newton_max > 1e-12 {
                1.0 - (nn_max / newton_max)
            } else {
                0.0
            };

            println!(
                "v_newton={:.6}, v_nn={:.6}, reduction={:.4}",
                newton_max, nn_max, vel_reduction
            );

            results.push(ThickeningResult {
                alpha,
                power_index,
                newton_max_vel: newton_max,
                non_newton_max_vel: nn_max,
                vel_reduction,
                tau_min: t_min,
                tau_max: t_max,
                tau_range: t_max - t_min,
                mean_strain_rate: mean_sr,
            });
        }
    }

    // Step 4: Output
    println!("[4/4] Writing output...");
    std::fs::create_dir_all(&args.output_dir)?;

    // TOML report
    let mut report = String::new();
    let _ = writeln!(report, "[metadata]");
    let _ = writeln!(report, "experiment = \"Thesis2-3D-Thickening\"");
    let _ = writeln!(report, "grid_size = {}", nx);
    let _ = writeln!(report, "lbm_steps = {}", args.lbm_steps);
    let _ = writeln!(report, "nu_base = {:.6}", args.nu_base);
    let _ = writeln!(report, "force_amplitude = {:.2e}", args.force_amp);
    let _ = writeln!(report, "mean_associator_norm = {:.8}", mean_assoc);
    let _ = writeln!(report, "max_associator_norm = {:.8}", max_assoc);
    let _ = writeln!(report);

    for r in &results {
        let _ = writeln!(report, "[[sweep]]");
        let _ = writeln!(report, "alpha = {:.3}", r.alpha);
        let _ = writeln!(report, "power_index = {:.3}", r.power_index);
        let _ = writeln!(report, "newton_max_vel = {:.8}", r.newton_max_vel);
        let _ = writeln!(report, "non_newton_max_vel = {:.8}", r.non_newton_max_vel);
        let _ = writeln!(report, "vel_reduction = {:.6}", r.vel_reduction);
        let _ = writeln!(report, "tau_min = {:.6}", r.tau_min);
        let _ = writeln!(report, "tau_max = {:.6}", r.tau_max);
        let _ = writeln!(report, "tau_range = {:.6}", r.tau_range);
        let _ = writeln!(report, "mean_strain_rate = {:.8}", r.mean_strain_rate);
        let _ = writeln!(report);
    }

    // Summary
    let best = results
        .iter()
        .filter(|r| r.alpha > 0.0)
        .max_by(|a, b| {
            a.vel_reduction
                .abs()
                .partial_cmp(&b.vel_reduction.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    let _ = writeln!(report, "[summary]");
    let _ = writeln!(report, "n_pairs = {}", results.len());
    if let Some(best) = best {
        let _ = writeln!(report, "best_alpha = {:.3}", best.alpha);
        let _ = writeln!(report, "best_power_index = {:.3}", best.power_index);
        let _ = writeln!(report, "best_vel_reduction = {:.6}", best.vel_reduction);
        let _ = writeln!(
            report,
            "significant = {}",
            best.vel_reduction.abs() > 0.05
        );
        println!(
            "\nBest: alpha={:.1}, n={:.1}, vel_reduction={:.4} (significant={})",
            best.alpha,
            best.power_index,
            best.vel_reduction,
            best.vel_reduction.abs() > 0.05,
        );
    }

    let report_path = format!("{}/thesis2_3d.toml", args.output_dir);
    std::fs::write(&report_path, &report)?;
    println!("Report: {}", report_path);

    // CSV
    let csv_path = format!("{}/thesis2_3d.csv", args.output_dir);
    let mut wtr = csv::Writer::from_path(&csv_path)?;
    wtr.write_record([
        "alpha",
        "power_index",
        "newton_max_vel",
        "non_newton_max_vel",
        "vel_reduction",
        "tau_min",
        "tau_max",
        "mean_strain_rate",
    ])?;
    for r in &results {
        wtr.write_record(&[
            format!("{:.3}", r.alpha),
            format!("{:.3}", r.power_index),
            format!("{:.8}", r.newton_max_vel),
            format!("{:.8}", r.non_newton_max_vel),
            format!("{:.6}", r.vel_reduction),
            format!("{:.6}", r.tau_min),
            format!("{:.6}", r.tau_max),
            format!("{:.8}", r.mean_strain_rate),
        ])?;
    }
    wtr.flush()?;
    println!("CSV: {}", csv_path);

    Ok(())
}

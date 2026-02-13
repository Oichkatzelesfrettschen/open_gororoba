//! Thesis 2 Kolmogorov Validation: 2D non-Newtonian power-law LBM
//!
//! Validates that power-law viscosity coupling produces measurable
//! non-Newtonian effects in 2D Kolmogorov flow before attempting 3D.
//!
//! Compares Newtonian vs non-Newtonian enstrophy across a sweep of
//! Reynolds numbers (via tau) and coupling strengths.
//!
//! Success criterion: enstrophy ratio (non-Newtonian / Newtonian) deviates
//! from 1.0 by more than 5% for at least one parameter combination.

use clap::Parser;
use lbm_core::{equilibrium, macroscopic, stream, viscosity_with_power_law_associator, CX, W};
use ndarray::Array2;
use std::f64::consts::PI;
use std::fmt::Write as _;

#[derive(Parser, Debug)]
#[command(name = "thesis2-kolmogorov-2d")]
#[command(about = "2D Kolmogorov flow validation for non-Newtonian power-law viscosity")]
struct Args {
    /// Grid size (NxN)
    #[arg(long, default_value = "64")]
    grid_size: usize,

    /// Number of LBM timesteps
    #[arg(long, default_value = "5000")]
    steps: usize,

    /// Force amplitude
    #[arg(long, default_value = "1e-4")]
    force_amp: f64,

    /// Force wavenumber (1 = fundamental)
    #[arg(long, default_value = "1")]
    force_mode: usize,

    /// Power-law index (n>1 = shear-thickening)
    #[arg(long, default_value = "1.5")]
    power_index: f64,

    /// Coupling strength sweep (comma-separated)
    #[arg(long, default_value = "0.0,1.0,5.0,20.0,50.0,100.0")]
    couplings: String,

    /// Tau sweep (comma-separated)
    #[arg(long, default_value = "0.55,0.6,0.7,0.8,1.0,1.5")]
    taus: String,

    /// Output file
    #[arg(long, default_value = "data/thesis_lab/thesis2_kolmogorov_2d.toml")]
    output: String,
}

/// 2D strain rate magnitude from velocity field.
/// |gamma_dot| = sqrt(2 * (e_xx^2 + e_yy^2 + 2*e_xy^2))
/// where e_ab = (du_a/dx_b + du_b/dx_a) / 2
fn compute_strain_rate_2d(ux: &Array2<f64>, uy: &Array2<f64>) -> Array2<f64> {
    let (nx, ny) = ux.dim();
    let mut gamma_dot = Array2::zeros((nx, ny));

    for x in 0..nx {
        for y in 0..ny {
            let xp = (x + 1) % nx;
            let xm = (x + nx - 1) % nx;
            let yp = (y + 1) % ny;
            let ym = (y + ny - 1) % ny;

            // Central differences (periodic)
            let dux_dx = (ux[[xp, y]] - ux[[xm, y]]) / 2.0;
            let dux_dy = (ux[[x, yp]] - ux[[x, ym]]) / 2.0;
            let duy_dx = (uy[[xp, y]] - uy[[xm, y]]) / 2.0;
            let duy_dy = (uy[[x, yp]] - uy[[x, ym]]) / 2.0;

            let e_xx = dux_dx;
            let e_yy = duy_dy;
            let e_xy = (dux_dy + duy_dx) / 2.0;

            gamma_dot[[x, y]] = (2.0 * (e_xx * e_xx + e_yy * e_yy + 2.0 * e_xy * e_xy)).sqrt();
        }
    }
    gamma_dot
}

/// Mean enstrophy from velocity fields.
fn compute_enstrophy(ux: &Array2<f64>, uy: &Array2<f64>) -> f64 {
    let (nx, ny) = ux.dim();
    let mut total = 0.0;
    for x in 0..nx {
        for y in 0..ny {
            let xp = (x + 1) % nx;
            let xm = (x + nx - 1) % nx;
            let yp = (y + 1) % ny;
            let ym = (y + ny - 1) % ny;
            let duy_dx = (uy[[xp, y]] - uy[[xm, y]]) / 2.0;
            let dux_dy = (ux[[x, yp]] - ux[[x, ym]]) / 2.0;
            let omega = duy_dx - dux_dy;
            total += omega * omega;
        }
    }
    total / (nx * ny) as f64
}

/// Result from one simulation run.
struct RunResult {
    nu_base: f64,
    enstrophy: f64,
    max_velocity: f64,
    mean_strain_rate: f64,
    mean_tau_eff: f64,
    mass_drift: f64,
}

/// Configuration for a single Kolmogorov simulation.
struct KolmogorovConfig {
    nx: usize,
    ny: usize,
    tau: f64,
    force_amp: f64,
    force_mode: usize,
    n_steps: usize,
    coupling: f64,
    power_index: f64,
}

/// Simulate 2D Kolmogorov flow with optional non-Newtonian power-law viscosity.
///
/// When coupling > 0, tau is updated at each step based on local strain rate:
///   nu_eff = nu_base * (1 + coupling * |gamma_dot|^(power_index - 1))
///   tau_eff = 3 * nu_eff + 0.5
fn simulate_kolmogorov_power_law(cfg: &KolmogorovConfig) -> RunResult {
    let KolmogorovConfig {
        nx,
        ny,
        tau,
        force_amp,
        force_mode,
        n_steps,
        coupling,
        power_index,
    } = *cfg;
    let nu_base = (tau - 0.5) / 3.0;

    // Initialize equilibrium at rest
    let rho_init = Array2::ones((nx, ny));
    let ux_init = Array2::zeros((nx, ny));
    let uy_init = Array2::zeros((nx, ny));
    let mut f = equilibrium(&rho_init, &ux_init, &uy_init);

    let initial_mass = f.sum();
    let is_non_newtonian = coupling > 0.0 && power_index > 1.0;

    // Per-node tau field (only used when non-Newtonian)
    let mut tau_field = Array2::from_elem((nx, ny), tau);
    let mut last_strain_rate = Array2::zeros((nx, ny));

    for step in 0..n_steps {
        let (rho, ux, uy) = macroscopic(&f);
        let feq = equilibrium(&rho, &ux, &uy);

        // Update tau field from strain rate (after initial transient)
        if is_non_newtonian && step >= 10 {
            let gamma_dot = compute_strain_rate_2d(&ux, &uy);
            for x in 0..nx {
                for y in 0..ny {
                    let sr = gamma_dot[[x, y]];
                    let nu_eff = viscosity_with_power_law_associator(
                        nu_base,
                        coupling,
                        1.0,
                        1.0,
                        sr,
                        power_index,
                    );
                    tau_field[[x, y]] = (3.0 * nu_eff + 0.5).clamp(0.505, 3.0);
                }
            }
            last_strain_rate = gamma_dot;
        }

        // BGK collision with per-node tau
        for i in 0..9 {
            for x in 0..nx {
                for y in 0..ny {
                    let omega = 1.0 / tau_field[[x, y]];
                    f[[i, x, y]] += omega * (feq[[i, x, y]] - f[[i, x, y]]);
                }
            }
        }

        // Kolmogorov forcing: fx(y) = A * sin(2*pi*n*y/ny)
        let (rho2, _, _) = macroscopic(&f);
        for i in 0..9 {
            let cx = CX[i] as f64;
            for x in 0..nx {
                for y in 0..ny {
                    let fy_force =
                        force_amp * (2.0 * PI * force_mode as f64 * y as f64 / ny as f64).sin();
                    f[[i, x, y]] += 3.0 * W[i] * cx * fy_force * rho2[[x, y]];
                }
            }
        }

        // Streaming (periodic)
        f = stream(&f);
    }

    let (rho_final, ux_final, uy_final) = macroscopic(&f);
    let enstrophy = compute_enstrophy(&ux_final, &uy_final);
    let final_mass = f.sum();

    // Max velocity
    let mut max_vel = 0.0_f64;
    for x in 0..nx {
        for y in 0..ny {
            let v = (ux_final[[x, y]].powi(2) + uy_final[[x, y]].powi(2)).sqrt();
            max_vel = max_vel.max(v);
        }
    }

    // Mean strain rate and effective tau
    let mean_sr = last_strain_rate.sum() / (nx * ny) as f64;
    let mean_tau = tau_field.sum() / (nx * ny) as f64;

    // Mass conservation check
    let _ = rho_final; // used via enstrophy

    RunResult {
        nu_base,
        enstrophy,
        max_velocity: max_vel,
        mean_strain_rate: mean_sr,
        mean_tau_eff: mean_tau,
        mass_drift: (final_mass - initial_mass).abs() / initial_mass,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let nx = args.grid_size;
    let ny = args.grid_size;

    let taus: Vec<f64> = args
        .taus
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let couplings: Vec<f64> = args
        .couplings
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    println!("Thesis 2 Kolmogorov Validation (2D)");
    println!("====================================");
    println!("Grid: {}x{}", nx, ny);
    println!("Steps: {}", args.steps);
    println!("Force: A={:.2e}, mode={}", args.force_amp, args.force_mode);
    println!("Power index: {:.2}", args.power_index);
    println!("Tau values: {:?}", taus);
    println!("Coupling values: {:?}", couplings);
    println!();

    let mut report = String::new();
    let _ = writeln!(report, "[metadata]");
    let _ = writeln!(report, "experiment = \"thesis2_kolmogorov_2d\"");
    let _ = writeln!(report, "grid_size = {}", nx);
    let _ = writeln!(report, "steps = {}", args.steps);
    let _ = writeln!(report, "force_amp = {:.2e}", args.force_amp);
    let _ = writeln!(report, "force_mode = {}", args.force_mode);
    let _ = writeln!(report, "power_index = {:.2}", args.power_index);
    let _ = writeln!(report);

    let mut all_results: Vec<RunResult> = Vec::new();

    for &tau in &taus {
        let re = (nx as f64) * args.force_amp / ((tau - 0.5) / 3.0).powi(2);
        println!(
            "--- tau={:.2} (nu={:.4}, Re~{:.1}) ---",
            tau,
            (tau - 0.5) / 3.0,
            re
        );

        for &coupling in &couplings {
            let kcfg = KolmogorovConfig {
                nx,
                ny,
                tau,
                force_amp: args.force_amp,
                force_mode: args.force_mode,
                n_steps: args.steps,
                coupling,
                power_index: args.power_index,
            };
            let result = simulate_kolmogorov_power_law(&kcfg);

            let label = if coupling == 0.0 {
                "Newtonian"
            } else {
                "PowerLaw"
            };
            println!(
                "  coupling={:6.1}: enstrophy={:.6e}, v_max={:.6}, <gamma_dot>={:.6}, <tau_eff>={:.4}, mass_drift={:.2e} [{}]",
                coupling, result.enstrophy, result.max_velocity,
                result.mean_strain_rate, result.mean_tau_eff, result.mass_drift, label,
            );

            let _ = writeln!(report, "[[result]]");
            let _ = writeln!(report, "tau = {:.4}", tau);
            let _ = writeln!(report, "nu_base = {:.6}", result.nu_base);
            let _ = writeln!(report, "coupling = {:.2}", coupling);
            let _ = writeln!(report, "enstrophy = {:.8e}", result.enstrophy);
            let _ = writeln!(report, "max_velocity = {:.8}", result.max_velocity);
            let _ = writeln!(report, "mean_strain_rate = {:.8}", result.mean_strain_rate);
            let _ = writeln!(report, "mean_tau_eff = {:.6}", result.mean_tau_eff);
            let _ = writeln!(report, "mass_drift = {:.2e}", result.mass_drift);
            let _ = writeln!(report);

            all_results.push(result);
        }
        println!();
    }

    // Enstrophy ratio analysis: for each tau, compute ratio non-Newtonian / Newtonian
    let n_couplings = couplings.len();
    let _ = writeln!(report, "[enstrophy_ratios]");
    let _ = writeln!(
        report,
        "note = \"ratio = enstrophy(coupling) / enstrophy(coupling=0)\""
    );
    let mut max_deviation = 0.0_f64;
    let mut max_deviation_params = String::new();

    println!("=== Enstrophy Ratio Analysis ===");
    for (ti, &tau) in taus.iter().enumerate() {
        let base_idx = ti * n_couplings;
        let newtonian_enstrophy = all_results[base_idx].enstrophy;

        if newtonian_enstrophy < 1e-20 {
            println!("  tau={:.2}: SKIP (zero Newtonian enstrophy)", tau);
            continue;
        }

        print!("  tau={:.2}:", tau);
        let mut ratios = Vec::new();
        for (ci, &coupling) in couplings.iter().enumerate() {
            let ratio = all_results[base_idx + ci].enstrophy / newtonian_enstrophy;
            let deviation = (ratio - 1.0).abs();
            if deviation > max_deviation {
                max_deviation = deviation;
                max_deviation_params = format!("tau={:.2}, coupling={:.1}", tau, coupling);
            }
            print!(" c={:.0}:{:.4}", coupling, ratio);
            ratios.push(format!("{:.6}", ratio));
        }
        println!();

        let _ = writeln!(
            report,
            "tau_{} = [{}]",
            format!("{:.2}", tau).replace('.', "_"),
            ratios.join(", ")
        );
    }
    println!();

    let success = max_deviation > 0.05;
    println!(
        "Max enstrophy deviation: {:.4} ({:.1}%) at {}",
        max_deviation,
        max_deviation * 100.0,
        max_deviation_params
    );
    println!(
        "Thesis 2 validation: {}",
        if success {
            "PASS (>5% deviation)"
        } else {
            "FAIL (<5% deviation)"
        }
    );

    let _ = writeln!(report, "max_deviation = {:.6}", max_deviation);
    let _ = writeln!(
        report,
        "max_deviation_params = \"{}\"",
        max_deviation_params
    );
    let _ = writeln!(report, "validation_pass = {}", success);

    if let Some(parent) = std::path::Path::new(&args.output).parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.output, &report)?;
    println!("Report written to: {}", args.output);

    Ok(())
}

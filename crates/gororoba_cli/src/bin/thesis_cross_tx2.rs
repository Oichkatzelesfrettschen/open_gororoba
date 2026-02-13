//! TX-2: Cross-Thesis T2 x T4 -- Viscosity-to-Filtration Loop
//!
//! Tests whether frustration-derived viscosity -> LBM flow -> filtration
//! produces a meaningful latency law. This closes the loop:
//! algebraic frustration -> viscosity -> flow -> filtration -> latency law.
//!
//! Pipeline:
//! 1. Generate SedenionField, compute frustration density
//! 2. For each lambda (coupling strength):
//!    a. Convert frustration -> viscosity -> tau
//!    b. Run LBM evolution with Kolmogorov forcing
//!    c. Extract velocity field from LBM
//!    d. Feed velocity into filtration_from_velocity_field()
//!    e. Classify latency law from filtration spectrum
//! 3. Gate: at least one lambda produces a non-Undetermined law

use clap::Parser;
use lattice_filtration::{
    classify_latency_law_detailed, filtration_from_velocity_field, LatencyLawDetail,
};
use lbm_3d::solver::LbmSolver3D;
use std::fmt::Write as _;
use vacuum_frustration::bridge::{FrustrationViscosityBridge, SedenionField};

#[derive(Parser, Debug)]
#[command(name = "thesis-cross-tx2")]
#[command(about = "TX-2: Viscosity-to-filtration loop (T2 x T4)")]
struct Args {
    /// Grid size per axis (N^3 cells)
    #[arg(long, default_value = "16")]
    grid_size: usize,

    /// Number of LBM evolution steps
    #[arg(long, default_value = "500")]
    lbm_steps: usize,

    /// Base kinematic viscosity
    #[arg(long, default_value = "0.1")]
    nu_base: f64,

    /// Force amplitude for Kolmogorov forcing
    #[arg(long, default_value = "5e-4")]
    force_amp: f64,

    /// Lambda values for coupling sweep (comma-separated)
    #[arg(long, default_value = "0.5,1.0,2.0,5.0,10.0")]
    lambdas: String,

    /// Lattice projection scale for filtration
    #[arg(long, default_value = "10.0")]
    projection_scale: f64,

    /// Number of radial bins for spectrum analysis
    #[arg(long, default_value = "12")]
    n_bins: usize,

    /// Output directory
    #[arg(long, default_value = "data/thesis_lab/tx2")]
    output_dir: String,
}

/// Result for one lambda value.
struct Tx2Result {
    lambda: f64,
    mean_viscosity: f64,
    lbm_max_velocity: f64,
    n_active_cells: usize,
    n_total_cells: usize,
    latency_law: String,
    detail: LatencyLawDetail,
    n_spectrum_bins: usize,
}

/// Generate SedenionField with spatial variation (consistent with other binaries).
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

/// Format a LatencyLaw variant as string.
fn latency_law_label(law: &lattice_filtration::LatencyLaw) -> &'static str {
    match law {
        lattice_filtration::LatencyLaw::InverseSquare => "InverseSquare",
        lattice_filtration::LatencyLaw::PowerLaw => "PowerLaw",
        lattice_filtration::LatencyLaw::Linear => "Linear",
        lattice_filtration::LatencyLaw::Exponential => "Exponential",
        lattice_filtration::LatencyLaw::Uniform => "Uniform",
        lattice_filtration::LatencyLaw::Undetermined => "Undetermined",
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let nx = args.grid_size;
    let n_cells = nx * nx * nx;

    let lambdas: Vec<f64> = args
        .lambdas
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    println!("TX-2: Viscosity-to-Filtration Loop (T2 x T4)");
    println!("=============================================");
    println!("Grid: {}^3 ({} cells)", nx, n_cells);
    println!("LBM steps: {}, nu_base: {:.4}", args.lbm_steps, args.nu_base);
    println!("Force amplitude: {:.2e}", args.force_amp);
    println!("Lambda sweep: {:?}", lambdas);
    println!();

    // Step 1: Generate frustration field
    println!("[1/3] Generating SedenionField and frustration density...");
    let field = generate_sedenion_field(nx);
    let frustration = field.local_frustration_density(16);
    drop(field);

    let mean_f = frustration.iter().sum::<f64>() / n_cells as f64;
    println!("  Mean frustration: {:.6}", mean_f);

    // Step 2: Lambda sweep
    println!("[2/3] Lambda sweep ({} values)...", lambdas.len());
    let bridge = FrustrationViscosityBridge::new(16);
    let pi2 = std::f64::consts::PI * 2.0;
    let mut results = Vec::new();

    // Kolmogorov forcing
    let force_field: Vec<[f64; 3]> = (0..n_cells)
        .map(|idx| {
            let y = (idx / nx) % nx;
            let yn = y as f64 / nx as f64;
            [args.force_amp * (pi2 * yn).sin(), 0.0, 0.0]
        })
        .collect();

    for (i, &lambda) in lambdas.iter().enumerate() {
        print!("  [{}/{}] lambda={:.1}... ", i + 1, lambdas.len(), lambda);

        // Frustration -> viscosity -> tau
        let viscosity = bridge.frustration_to_viscosity(&frustration, args.nu_base, lambda);
        let mean_viscosity = viscosity.iter().sum::<f64>() / viscosity.len() as f64;
        let tau_field: Vec<f64> = viscosity.iter().map(|&nu| 3.0 * nu + 0.5).collect();

        // LBM evolution
        let default_tau = 3.0 * args.nu_base + 0.5;
        let mut solver = LbmSolver3D::new(nx, nx, nx, default_tau);
        solver
            .set_viscosity_field(tau_field)
            .expect("viscosity field");
        solver
            .set_force_field(force_field.clone())
            .expect("force field");
        solver.evolve(args.lbm_steps);
        solver.compute_macroscopic();

        // Extract velocity field
        let mut vel_field = Vec::with_capacity(n_cells);
        let mut max_vel = 0.0_f64;
        for z in 0..nx {
            for y in 0..nx {
                for x in 0..nx {
                    let (_rho, u) = solver.get_macroscopic(x, y, z);
                    let mag = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
                    max_vel = max_vel.max(mag);
                    vel_field.push(u);
                }
            }
        }

        // Feed velocity into filtration pipeline
        let filt = filtration_from_velocity_field(
            &vel_field,
            nx,
            nx,
            nx,
            args.projection_scale,
            args.n_bins,
        );

        // Build samples for detailed classification
        let samples: Vec<(f64, f64)> = filt
            .spectrum_bins
            .iter()
            .filter(|b| b.n_samples > 0)
            .map(|b| (b.radius, b.mean_latency))
            .collect();
        let detail = classify_latency_law_detailed(&samples);
        let law_label = latency_law_label(&filt.latency_law);

        println!(
            "max_vel={:.6}, active={}/{}, law={}, gamma={:.4}",
            max_vel,
            filt.n_active_cells,
            filt.n_total_cells,
            law_label,
            detail.power_law_exponent,
        );

        results.push(Tx2Result {
            lambda,
            mean_viscosity,
            lbm_max_velocity: max_vel,
            n_active_cells: filt.n_active_cells,
            n_total_cells: filt.n_total_cells,
            latency_law: law_label.to_string(),
            detail,
            n_spectrum_bins: filt.spectrum_bins.len(),
        });
    }

    // Step 3: Output
    println!("[3/3] Writing output...");
    std::fs::create_dir_all(&args.output_dir)?;

    let mut report = String::new();
    let _ = writeln!(report, "[metadata]");
    let _ = writeln!(report, "experiment = \"TX-2\"");
    let _ = writeln!(
        report,
        "description = \"Viscosity-to-filtration loop (T2 x T4)\""
    );
    let _ = writeln!(report, "grid_size = {}", nx);
    let _ = writeln!(report, "lbm_steps = {}", args.lbm_steps);
    let _ = writeln!(report, "nu_base = {:.6}", args.nu_base);
    let _ = writeln!(report, "force_amplitude = {:.2e}", args.force_amp);
    let _ = writeln!(report, "projection_scale = {:.2}", args.projection_scale);
    let _ = writeln!(report, "n_bins = {}", args.n_bins);
    let _ = writeln!(report, "mean_frustration = {:.8}", mean_f);
    let _ = writeln!(report);

    for r in &results {
        let _ = writeln!(report, "[[lambda_sweep]]");
        let _ = writeln!(report, "lambda = {:.3}", r.lambda);
        let _ = writeln!(report, "mean_viscosity = {:.8}", r.mean_viscosity);
        let _ = writeln!(report, "lbm_max_velocity = {:.8}", r.lbm_max_velocity);
        let _ = writeln!(report, "n_active_cells = {}", r.n_active_cells);
        let _ = writeln!(report, "n_total_cells = {}", r.n_total_cells);
        let _ = writeln!(report, "latency_law = \"{}\"", r.latency_law);
        let _ = writeln!(
            report,
            "power_law_exponent = {:.6}",
            r.detail.power_law_exponent
        );
        let _ = writeln!(report, "r2_power_law = {:.6}", r.detail.r2_power_law);
        let _ = writeln!(
            report,
            "r2_inverse_square = {:.6}",
            r.detail.r2_inverse_square
        );
        let _ = writeln!(report, "r2_linear = {:.6}", r.detail.r2_linear);
        let _ = writeln!(report, "r2_exponential = {:.6}", r.detail.r2_exponential);
        let _ = writeln!(report, "n_spectrum_bins = {}", r.n_spectrum_bins);
        let _ = writeln!(report);
    }

    // Summary
    let n_determined = results
        .iter()
        .filter(|r| r.latency_law != "Undetermined")
        .count();
    let _ = writeln!(report, "[summary]");
    let _ = writeln!(report, "n_lambdas = {}", results.len());
    let _ = writeln!(report, "n_determined = {}", n_determined);
    let _ = writeln!(report, "significant = {}", n_determined > 0);

    // Check if coupling changes the law
    let laws: Vec<&str> = results.iter().map(|r| r.latency_law.as_str()).collect();
    let all_same = laws.windows(2).all(|w| w[0] == w[1]);
    let _ = writeln!(report, "coupling_changes_law = {}", !all_same);

    let report_path = format!("{}/tx2_report.toml", args.output_dir);
    std::fs::write(&report_path, &report)?;
    println!("Report: {}", report_path);

    println!();
    println!("=============================================");
    println!(
        "TX-2 Summary: {}/{} determined, coupling_changes_law={}",
        n_determined,
        results.len(),
        !all_same,
    );

    Ok(())
}

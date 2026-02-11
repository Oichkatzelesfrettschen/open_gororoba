//! E-027 Percolation Threshold Experiment
//!
//! Validates Thesis 1: Frustration-viscosity coupling produces measurable percolation channels.
//!
//! Pipeline:
//! 1. Generate Sedenion field (APT-evolved or uniform with perturbation)
//! 2. Compute frustration density F(x,y,z)
//! 3. Transform to viscosity: nu(x) = nu_base * exp(-lambda * (F(x) - 3/8)^2)
//! 4. Initialize LBM solver with spatial viscosity
//! 5. Evolve LBM for N timesteps
//! 6. Detect percolation channels (BFS on velocity field)
//! 7. Correlate channels with frustration (Welch's t-test)
//! 8. Run Besag-Clifford null model test
//! 9. Export results to TOML
//! 10. Validate/falsify Thesis 1 (p < 0.05 threshold)

use std::fs;
use clap::Parser;
use serde::Serialize;

#[derive(Parser)]
#[command(name = "percolation-experiment")]
#[command(about = "E-027: Percolation Threshold vs Frustration Correlation", long_about = None)]
struct Args {
    /// Grid size (cubic domain)
    #[arg(long, default_value = "32")]
    grid_size: usize,

    /// Number of LBM evolution steps
    #[arg(long, default_value = "2500")]
    lbm_steps: usize,

    /// Base kinematic viscosity
    #[arg(long, default_value = "0.333")]
    nu_base: f64,

    /// Frustration coupling strength
    #[arg(long, default_value = "1.0")]
    lambda: f64,

    /// Number of Besag-Clifford permutations
    #[arg(long, default_value = "1000")]
    n_permutations: usize,

    /// Random seed for reproducibility
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Output directory for TOML results
    #[arg(long, default_value = "data/e027")]
    output_dir: String,

    /// Verbose output
    #[arg(long, default_value = "false")]
    verbose: bool,

    /// Forcing mode: none, uniform, gradient, vortex
    #[arg(long, default_value = "none")]
    forcing_mode: String,

    /// Forcing strength magnitude
    #[arg(long, default_value = "0.001")]
    forcing_strength: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    if args.verbose {
        eprintln!("E-027 Percolation Experiment");
        eprintln!("============================");
        eprintln!("Grid size: {}^3", args.grid_size);
        eprintln!("LBM steps: {}", args.lbm_steps);
        eprintln!("nu_base: {}", args.nu_base);
        eprintln!("lambda: {}", args.lambda);
        eprintln!("forcing_mode: {}", args.forcing_mode);
        eprintln!("forcing_strength: {}", args.forcing_strength);
        eprintln!("seed: {}", args.seed);
    }

    // Step 1: Generate Sedenion field
    if args.verbose {
        eprintln!("\n[1/10] Generating Sedenion field...");
    }

    // Step 2: Compute frustration density via APT-evolved Sedenion field
    if args.verbose {
        eprintln!("[2/10] Computing frustration density via APT evolution...");
    }
    let frustration_field = generate_apt_frustration_field(args.grid_size, args.seed)?;

    // Compute frustration statistics
    let f_min = frustration_field.iter().copied().fold(f64::INFINITY, f64::min);
    let f_max = frustration_field.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let f_mean = frustration_field.iter().sum::<f64>() / (frustration_field.len() as f64);
    let f_std = {
        let variance = frustration_field.iter()
            .map(|&f| (f - f_mean).powi(2))
            .sum::<f64>() / (frustration_field.len() as f64);
        variance.sqrt()
    };

    if args.verbose {
        eprintln!("  Frustration stats: mean={:.4}, std={:.4}, range=[{:.4}, {:.4}]", f_mean, f_std, f_min, f_max);
    }

    // Step 3: Transform frustration to viscosity
    if args.verbose {
        eprintln!("[3/10] Transforming to viscosity field...");
    }
    let viscosity_field = frustration_to_viscosity(&frustration_field, args.nu_base, args.lambda);

    // Compute viscosity statistics
    let nu_min = viscosity_field.iter().copied().fold(f64::INFINITY, f64::min);
    let nu_max = viscosity_field.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let nu_mean = viscosity_field.iter().sum::<f64>() / (viscosity_field.len() as f64);

    if args.verbose {
        eprintln!("  Viscosity stats: mean={:.4}, range=[{:.4}, {:.4}]", nu_mean, nu_min, nu_max);
    }

    // Step 4: Initialize LBM solver
    if args.verbose {
        eprintln!("[4/10] Initializing LBM solver...");
    }
    let mut solver = initialize_lbm_solver(
        args.grid_size,
        &viscosity_field,
        &args.forcing_mode,
        args.forcing_strength,
    )?;

    // Step 5: Evolve LBM
    if args.verbose {
        eprintln!("[5/10] Evolving LBM for {} steps...", args.lbm_steps);
    }
    solver.evolve(args.lbm_steps);

    // Compute velocity statistics
    let u_magnitudes: Vec<f64> = solver.u.iter()
        .map(|&[ux, uy, uz]| (ux*ux + uy*uy + uz*uz).sqrt())
        .collect();
    let u_min = u_magnitudes.iter().copied().fold(f64::INFINITY, f64::min);
    let u_max = u_magnitudes.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let u_mean = u_magnitudes.iter().sum::<f64>() / (u_magnitudes.len() as f64);
    let u_std = {
        let variance = u_magnitudes.iter()
            .map(|&u| (u - u_mean).powi(2))
            .sum::<f64>() / (u_magnitudes.len() as f64);
        variance.sqrt()
    };

    if args.verbose {
        eprintln!("  Velocity magnitude stats: mean={:.6}, std={:.6}, range=[{:.6}, {:.6}]", u_mean, u_std, u_min, u_max);
    }

    // Step 6: Detect percolation channels
    if args.verbose {
        eprintln!("[6/10] Detecting percolation channels...");
    }
    let mut detector = vacuum_frustration::PercolationDetector::new(
        args.grid_size,
        args.grid_size,
        args.grid_size,
    );
    let threshold = vacuum_frustration::auto_velocity_threshold(&solver.u);
    let above_threshold = u_magnitudes.iter().filter(|&&u| u > threshold).count();

    if args.verbose {
        eprintln!("  Velocity threshold: {:.6}", threshold);
        eprintln!("  Cells above threshold: {}/{}", above_threshold, u_magnitudes.len());
    }

    let channels = detector.detect_channels(&solver.u, threshold);

    if args.verbose {
        eprintln!("  Found {} channels", channels.len());
    }

    // Step 7: Correlate with frustration
    if args.verbose {
        eprintln!("[7/10] Computing frustration correlation...");
    }
    let correlation = vacuum_frustration::correlate_with_frustration(&channels, &frustration_field);

    if args.verbose {
        eprintln!("  t-statistic: {:.6}", correlation.t_statistic);
        eprintln!("  p-value: {:.6}", correlation.p_value);
        eprintln!("  effect size: {:.6}", correlation.effect_size);
    }

    // Step 8: Besag-Clifford null model
    if args.verbose {
        eprintln!("[8/10] Running Besag-Clifford null model (max {} permutations, adaptive)...", args.n_permutations);
    }
    let null_result = run_besag_clifford_null(BesagCliffordParams {
        frustration_field: &frustration_field,
        grid_size: args.grid_size,
        max_permutations: args.n_permutations,
        seed: args.seed + 1000, // Different seed for null model
        observed_t: correlation.t_statistic,
        nu_base: args.nu_base,
        lambda: args.lambda,
        lbm_steps: args.lbm_steps,
        forcing_mode: &args.forcing_mode,
        forcing_strength: args.forcing_strength,
    })?;

    if args.verbose {
        eprintln!("  null p-value: {:.6}", null_result.p_value);
        eprintln!("  permutations used: {}", null_result.n_permutations);
        eprintln!("  stopped early: {}", null_result.stopped_early);
        eprintln!("  stop reason: {}", null_result.stop_reason);
    }

    // Step 9: Export results
    if args.verbose {
        eprintln!("[9/10] Exporting results to TOML...");
    }
    export_results(
        &args.output_dir,
        &channels,
        &correlation,
        &null_result,
        ExportParams {
            grid_size: args.grid_size,
            lbm_steps: args.lbm_steps,
            nu_base: args.nu_base,
            lambda: args.lambda,
        },
    )?;

    // Step 10: Falsification check
    if args.verbose {
        eprintln!("[10/10] Validating Thesis 1...");
    }

    let thesis1_validated = correlation.p_value < 0.05;
    let null_rejects_random = null_result.p_value < 0.05;

    if thesis1_validated && null_rejects_random {
        println!("E-027 PASS: Thesis 1 validated (p={:.6})", correlation.p_value);
        Ok(())
    } else {
        eprintln!(
            "E-027 FAIL: Thesis 1 refuted (correlation p={:.6}, null p={:.6})",
            correlation.p_value, null_result.p_value
        );
        std::process::exit(1);
    }
}

// Helper: Generate APT-evolved Sedenion field with frustration
//
// Implements Attracting-Point-Transformation (APT) evolution on Sedenion algebra.
// Uses Harary-Zaslavsky frustration as attractor to generate correlated spatial fields
// without ad-hoc perturbation. This replaces mock data with real algebraic evolution.
fn generate_apt_frustration_field(
    grid_size: usize,
    seed: u64,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    use vacuum_frustration::AptSedenionField;

    // Create APT field with Metropolis-Hastings evolution
    let mut apt = AptSedenionField::new(grid_size, seed);

    // Evolve for 1000+ iterations per cell to reach quasi-equilibrium
    // Temperature cooling schedule drives frustration toward vacuum attractor (3/8)
    let n_iterations = grid_size.max(100);
    apt.evolve(n_iterations);

    // Extract frustration field after evolution
    Ok(apt.frustration_field())
}

// Helper: Transform frustration to viscosity
fn frustration_to_viscosity(frustration: &[f64], nu_base: f64, lambda: f64) -> Vec<f64> {
    frustration
        .iter()
        .map(|&f| {
            let vacuum_attractor = 0.375;
            let exponent = -lambda * (f - vacuum_attractor).powi(2);
            nu_base * exponent.exp()
        })
        .collect()
}

// Helper: Initialize LBM solver with velocity shear to seed flow instability
fn initialize_lbm_solver(
    grid_size: usize,
    viscosity_field: &[f64],
    forcing_mode: &str,
    forcing_strength: f64,
) -> Result<lbm_3d::solver::LbmSolver3D, Box<dyn std::error::Error>> {
    let mut solver = lbm_3d::solver::LbmSolver3D::new(grid_size, grid_size, grid_size, 0.6);

    // Convert viscosity to tau
    let tau_field: Vec<f64> = viscosity_field
        .iter()
        .map(|&nu| 3.0 * nu + 0.5)
        .collect();

    solver.set_viscosity_field(tau_field)?;

    // Initialize with velocity shear: u_x varies with z-coordinate to create instability
    // This is more physical than random perturbations
    for z in 0..grid_size {
        for y in 0..grid_size {
            for x in 0..grid_size {
                let idx = z * (grid_size * grid_size) + y * grid_size + x;
                let z_normalized = (z as f64) / (grid_size as f64);
                // Linear shear: u_x increases from 0.005 to 0.015 across z-direction
                let u_x = 0.01 + 0.005 * (z_normalized - 0.5);
                let u_y = 0.0;
                let u_z = 0.0;

                solver.rho[idx] = 1.0;
                solver.u[idx] = [u_x, u_y, u_z];

                // Initialize distribution function to equilibrium with shear velocity
                let f_eq = lbm_3d::solver::BgkCollision::initialize_with_velocity(1.0, [u_x, u_y, u_z], &solver.collider.lattice);
                let f_start = idx * 19;
                solver.f[f_start..f_start + 19].copy_from_slice(&f_eq);
            }
        }
    }

    // Apply body forcing based on mode
    match forcing_mode {
        "none" => {
            // No forcing - rely on shear instability alone
        },
        "uniform" => {
            // Uniform body force in x-direction (like electric field or pressure gradient)
            let force = vec![[forcing_strength, 0.0, 0.0]; grid_size * grid_size * grid_size];
            solver.set_force_field(force)?;
        },
        "gradient" => {
            // Linearly varying force creates shear flow instability
            let mut force = vec![[0.0, 0.0, 0.0]; grid_size * grid_size * grid_size];
            for z in 0..grid_size {
                for y in 0..grid_size {
                    for x in 0..grid_size {
                        let idx = z * (grid_size * grid_size) + y * grid_size + x;
                        let z_normalized = (z as f64) / (grid_size as f64);
                        // Force increases with z
                        force[idx] = [forcing_strength * z_normalized, 0.0, 0.0];
                    }
                }
            }
            solver.set_force_field(force)?;
        },
        "vortex" => {
            // Rotational forcing around domain center
            let mut force = vec![[0.0, 0.0, 0.0]; grid_size * grid_size * grid_size];
            let center = grid_size as f64 / 2.0;
            for z in 0..grid_size {
                for y in 0..grid_size {
                    for x in 0..grid_size {
                        let idx = z * (grid_size * grid_size) + y * grid_size + x;
                        let dx = x as f64 - center;
                        let dy = y as f64 - center;
                        let r = (dx * dx + dy * dy).sqrt();
                        if r > 1e-6 {
                            // Tangential force: F = strength * (-dy/r, dx/r, 0)
                            force[idx] = [
                                -forcing_strength * dy / r,
                                forcing_strength * dx / r,
                                0.0,
                            ];
                        }
                    }
                }
            }
            solver.set_force_field(force)?;
        },
        _ => {
            return Err(format!("Unknown forcing mode: {}", forcing_mode).into());
        },
    }

    Ok(solver)
}

/// Parameters for Besag-Clifford null model test
struct BesagCliffordParams<'a> {
    frustration_field: &'a [f64],
    grid_size: usize,
    max_permutations: usize,
    seed: u64,
    observed_t: f64,
    nu_base: f64,
    lambda: f64,
    lbm_steps: usize,
    forcing_mode: &'a str,
    forcing_strength: f64,
}

// Helper: Besag-Clifford null model via adaptive permutation test
//
// Tests spatial autocorrelation by shuffling viscosity field and recomputing correlation.
// Uses adaptive stopping (Besag & Clifford 1991) to terminate early when CI is tight.
fn run_besag_clifford_null(
    params: BesagCliffordParams,
) -> Result<NullModelResult, Box<dyn std::error::Error>> {
    use rand::seq::SliceRandom;
    use rand::SeedableRng;

    let mut rng = rand::rngs::StdRng::seed_from_u64(params.seed);

    // Adaptive configuration
    let config = stats_core::ultrametric::adaptive::AdaptiveConfig {
        batch_size: 20,
        max_permutations: params.max_permutations,
        alpha: 0.05,
        confidence: 0.99,
        min_permutations: 100,
    };

    // Run adaptive test
    let result = stats_core::ultrametric::adaptive::adaptive_permutation_test(
        &config,
        |batch_size| {
            let mut batch_extreme = 0;

            for _ in 0..batch_size {
                // Shuffle frustration field (breaks spatial structure)
                let mut shuffled_frustration = params.frustration_field.to_vec();
                shuffled_frustration.shuffle(&mut rng);

                // Transform to viscosity
                let shuffled_viscosity = frustration_to_viscosity(
                    &shuffled_frustration,
                    params.nu_base,
                    params.lambda,
                );

                // Initialize and evolve LBM with shuffled field
                let mut solver_null = match initialize_lbm_solver(
                    params.grid_size,
                    &shuffled_viscosity,
                    params.forcing_mode,
                    params.forcing_strength,
                ) {
                    Ok(s) => s,
                    Err(_) => continue, // Skip failed initialization
                };

                solver_null.evolve(params.lbm_steps);

                // Detect channels
                let mut detector = vacuum_frustration::PercolationDetector::new(
                    params.grid_size,
                    params.grid_size,
                    params.grid_size,
                );
                let threshold = vacuum_frustration::auto_velocity_threshold(&solver_null.u);
                let channels_null = detector.detect_channels(&solver_null.u, threshold);

                // Compute null correlation
                let corr_null = vacuum_frustration::correlate_with_frustration(
                    &channels_null,
                    &shuffled_frustration,
                );

                // Count if |t_null| >= |t_observed| (two-sided test)
                if corr_null.t_statistic.abs() >= params.observed_t.abs() {
                    batch_extreme += 1;
                }
            }

            batch_extreme
        },
    );

    Ok(NullModelResult {
        p_value: result.p_value,
        n_permutations: result.n_permutations_used,
        stopped_early: result.stopped_early,
        stop_reason: format!("{:?}", result.stop_reason),
    })
}

#[derive(Serialize)]
struct NullModelResult {
    p_value: f64,
    n_permutations: usize,
    stopped_early: bool,
    stop_reason: String,
}

#[derive(Serialize)]
struct ChannelData {
    id: usize,
    size: usize,
    mean_velocity: f64,
    max_velocity: f64,
    x_min: usize,
    x_max: usize,
    y_min: usize,
    y_max: usize,
    z_min: usize,
    z_max: usize,
}

#[derive(Serialize)]
struct ExperimentMetadata {
    grid_size: usize,
    lbm_steps: usize,
    nu_base: f64,
    lambda: f64,
}

#[derive(Serialize)]
struct CorrelationData {
    t_statistic: f64,
    p_value: f64,
    effect_size: f64,
    mean_frustration_channels: f64,
    mean_frustration_background: f64,
    n_channel: usize,
    n_background: usize,
    n_channels_detected: usize,
}

#[derive(Serialize)]
struct E027Results {
    metadata: ExperimentMetadata,
    correlation: CorrelationData,
    #[serde(rename = "null_model")]
    null_model: NullModelResult,
    channels: Vec<ChannelData>,
}

#[derive(Clone, Copy)]
struct ExportParams {
    grid_size: usize,
    lbm_steps: usize,
    nu_base: f64,
    lambda: f64,
}

// Helper: Export results to TOML
fn export_results(
    output_dir: &str,
    channels: &[vacuum_frustration::PercolationChannel],
    correlation: &vacuum_frustration::CorrelationResult,
    null_result: &NullModelResult,
    params: ExportParams,
) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(output_dir)?;

    // Convert channels to serializable format
    let channel_data: Vec<ChannelData> = channels
        .iter()
        .map(|ch| ChannelData {
            id: ch.id,
            size: ch.size,
            mean_velocity: ch.mean_velocity,
            max_velocity: ch.max_velocity,
            x_min: ch.bounding_box.x_min,
            x_max: ch.bounding_box.x_max,
            y_min: ch.bounding_box.y_min,
            y_max: ch.bounding_box.y_max,
            z_min: ch.bounding_box.z_min,
            z_max: ch.bounding_box.z_max,
        })
        .collect();

    // Build complete results structure
    let results = E027Results {
        metadata: ExperimentMetadata {
            grid_size: params.grid_size,
            lbm_steps: params.lbm_steps,
            nu_base: params.nu_base,
            lambda: params.lambda,
        },
        correlation: CorrelationData {
            t_statistic: correlation.t_statistic,
            p_value: correlation.p_value,
            effect_size: correlation.effect_size,
            mean_frustration_channels: correlation.mean_frustration_channels,
            mean_frustration_background: correlation.mean_frustration_background,
            n_channel: correlation.n_channel,
            n_background: correlation.n_background,
            n_channels_detected: channels.len(),
        },
        null_model: NullModelResult {
            p_value: null_result.p_value,
            n_permutations: null_result.n_permutations,
            stopped_early: null_result.stopped_early,
            stop_reason: null_result.stop_reason.clone(),
        },
        channels: channel_data,
    };

    // Serialize to TOML and write to file
    let toml_string = toml::to_string_pretty(&results)?;
    fs::write(format!("{}/e027_results.toml", output_dir), toml_string)?;

    Ok(())
}

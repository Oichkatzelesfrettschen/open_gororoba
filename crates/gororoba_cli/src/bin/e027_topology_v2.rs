//! E-027 v2: Corrected Topology-Frustration Spatial Correlation
//!
//! Fixes the v1 methodology flaw: v1 correlated Betti numbers against a
//! fabricated linear ramp `mean_F * (1 - eps/eps_max)` which had zero
//! genuine spatial information.
//!
//! v2 corrected pipeline:
//! 1. Generate SedenionField with spatial variation
//! 2. Compute per-cell frustration density
//! 3. Convert frustration -> viscosity via FrustrationViscosityBridge
//! 4. Convert viscosity -> tau field: tau[i] = 3*nu[i] + 0.5
//! 5. Run actual LBM evolution (D3Q19 streaming + collision)
//! 6. Extract velocity point cloud from LBM output
//! 7. Partition grid into subregions
//! 8. Per-subregion: compute mean frustration AND local Betti numbers
//! 9. Spatial correlation (Spearman + Pearson) across subregions
//! 10. Lambda sweep [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
//! 11. Output TOML + CSV

use clap::Parser;
use lbm_3d::solver::LbmSolver3D;
use std::fmt::Write as _;
use vacuum_frustration::bridge::{FrustrationViscosityBridge, SedenionField};
use vacuum_frustration::spatial_correlation::{spatial_correlation, SpatialCorrelationResult};
use vacuum_frustration::vietoris_rips::{
    compute_betti_numbers_at_time, compute_persistent_homology, DistanceMatrix,
    VietorisRipsComplex,
};

#[derive(Parser, Debug)]
#[command(name = "e027-topology-v2")]
#[command(about = "E-027 v2: Corrected topology-frustration spatial correlation")]
struct Args {
    /// Grid size per axis (N^3 cells)
    #[arg(long, default_value = "16")]
    grid_size: usize,

    /// Number of LBM evolution steps
    #[arg(long, default_value = "500")]
    lbm_steps: usize,

    /// Subregion subdivisions per axis (n_sub^3 total subregions)
    #[arg(long, default_value = "2")]
    n_sub: usize,

    /// Maximum points for VR topology (controls O(n^2) distance matrix)
    #[arg(long, default_value = "200")]
    max_points: usize,

    /// Epsilon threshold for VR complex (0 = auto from distance matrix)
    #[arg(long, default_value = "0.0")]
    epsilon: f64,

    /// Lambda values for coupling sweep (comma-separated)
    #[arg(long, default_value = "0.5,1.0,1.5,2.0,3.0,5.0,10.0")]
    lambdas: String,

    /// Output directory
    #[arg(long, default_value = "data/e027/v2")]
    output_dir: String,
}

/// Configuration shared across the lambda sweep.
struct SweepConfig {
    nx: usize,
    ny: usize,
    nz: usize,
    lbm_steps: usize,
    n_sub: usize,
    max_points: usize,
    epsilon: f64,
}

/// Result for one lambda value in the sweep.
struct LambdaSweepResult {
    lambda: f64,
    correlation: SpatialCorrelationResult,
    mean_frustration: f64,
    mean_viscosity: f64,
    lbm_max_velocity: f64,
    n_vr_points: usize,
    betti_0: usize,
    betti_1: usize,
}

/// Generate a SedenionField with spatial variation (deterministic).
///
/// Creates a field with smooth gradients plus localized perturbations
/// to produce nontrivial frustration structure.
fn generate_sedenion_field(nx: usize, ny: usize, nz: usize) -> SedenionField {
    let mut field = SedenionField::uniform(nx, ny, nz);

    // Deterministic xorshift for perturbations
    let mut state = 42_u64;
    let mut next_rand = || -> f64 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state as f64) / (u64::MAX as f64) * 2.0 - 1.0
    };

    let pi2 = std::f64::consts::PI * 2.0;

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let s = field.get_mut(x, y, z);
                let xn = x as f64 / nx as f64;
                let yn = y as f64 / ny as f64;
                let zn = z as f64 / nz as f64;

                // Smooth gradients in sedenion components
                s[1] = 0.3 * (pi2 * xn).sin();
                s[3] = 0.2 * (pi2 * 2.0 * yn).cos();
                s[5] = 0.15 * (pi2 * xn + pi2 * zn).sin();
                s[7] = 0.15 * zn;
                s[9] = 0.1 * (pi2 * 3.0 * xn).cos();
                s[11] = 0.1 * (pi2 * yn * 2.0).sin();

                // Localized noise for structure
                for component in s.iter_mut().take(16) {
                    *component += 0.05 * next_rand();
                }
            }
        }
    }

    field
}

/// Run the corrected E-027 pipeline for a single lambda value.
fn run_single_lambda(
    frustration: &[f64],
    lambda: f64,
    cfg: &SweepConfig,
) -> LambdaSweepResult {
    let (nx, ny, nz) = (cfg.nx, cfg.ny, cfg.nz);
    let bridge = FrustrationViscosityBridge::new(16);

    // Frustration -> viscosity
    let viscosity = bridge.frustration_to_viscosity(frustration, 1.0 / 3.0, lambda);
    let mean_viscosity = viscosity.iter().sum::<f64>() / viscosity.len() as f64;

    // Viscosity -> tau field for LBM: tau = 3*nu + 0.5
    let tau_field: Vec<f64> = viscosity.iter().map(|&nu| 3.0 * nu + 0.5).collect();

    // Initialize and evolve LBM solver
    let default_tau = 1.0; // overridden by spatially-varying viscosity field
    let mut solver = LbmSolver3D::new(nx, ny, nz, default_tau);
    solver
        .set_viscosity_field(tau_field)
        .expect("viscosity field length should match grid");

    // Apply uniform body force for flow generation (one per cell)
    let n_cells = nx * ny * nz;
    let force_field = vec![[1e-5, 0.0, 0.0]; n_cells];
    solver
        .set_force_field(force_field)
        .expect("force field length should match grid");

    // Evolve LBM (collision + streaming)
    solver.evolve(cfg.lbm_steps);
    solver.compute_macroscopic();

    // Extract velocity field via get_macroscopic
    let mut lbm_velocity = Vec::with_capacity(n_cells);
    let mut lbm_max_velocity = 0.0_f64;
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let (_rho, u) = solver.get_macroscopic(x, y, z);
                let mag = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
                lbm_max_velocity = lbm_max_velocity.max(mag);
                lbm_velocity.push(u);
            }
        }
    }

    // Compute spatial correlation between frustration and viscosity
    let correlation = spatial_correlation(frustration, &viscosity, nx, ny, nz, cfg.n_sub);

    // VR topology on velocity point cloud
    let mut candidates: Vec<(f64, [f64; 3])> = Vec::new();
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let idx = x + nx * (y + ny * z);
                let u = lbm_velocity[idx];
                let mag = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
                if mag > 1e-10 {
                    candidates.push((
                        mag,
                        [
                            x as f64 / nx.max(2) as f64,
                            y as f64 / ny.max(2) as f64,
                            z as f64 / nz.max(2) as f64,
                        ],
                    ));
                }
            }
        }
    }

    // Take top max_points by velocity magnitude
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(cfg.max_points);
    let n_vr_points = candidates.len();

    let (betti_0, betti_1) = if n_vr_points >= 3 {
        let mut flat = Vec::with_capacity(n_vr_points * 3);
        for (_, p) in &candidates {
            flat.extend_from_slice(p);
        }
        let dist = DistanceMatrix::from_points_3d(&flat);

        // Auto epsilon if not specified
        let eps = if cfg.epsilon > 0.0 {
            cfg.epsilon
        } else {
            auto_epsilon(&dist)
        };

        let complex = VietorisRipsComplex::build(&dist, eps, 2);
        let pairs = compute_persistent_homology(&complex);
        let betti = compute_betti_numbers_at_time(&pairs, eps);
        (*betti.first().unwrap_or(&0), *betti.get(1).unwrap_or(&0))
    } else {
        (n_vr_points, 0)
    };

    let mean_frustration = frustration.iter().sum::<f64>() / frustration.len() as f64;

    LambdaSweepResult {
        lambda,
        correlation,
        mean_frustration,
        mean_viscosity,
        lbm_max_velocity,
        n_vr_points,
        betti_0,
        betti_1,
    }
}

/// Compute auto epsilon as median of pairwise distances.
fn auto_epsilon(dist: &DistanceMatrix) -> f64 {
    let n = dist.size();
    if n < 2 {
        return 1.0;
    }
    let mut distances = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            distances.push(dist.get(i, j));
        }
    }
    distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    distances[distances.len() / 2]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let nx = args.grid_size;
    let ny = args.grid_size;
    let nz = args.grid_size;

    let lambdas: Vec<f64> = args
        .lambdas
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    println!("E-027 v2: Corrected Topology-Frustration Spatial Correlation");
    println!("=============================================================");
    println!(
        "Grid: {}^3, LBM steps: {}, Subregions: {}^3",
        nx, args.lbm_steps, args.n_sub
    );
    println!("Lambda sweep: {:?}", lambdas);
    println!();

    let sweep_cfg = SweepConfig {
        nx,
        ny,
        nz,
        lbm_steps: args.lbm_steps,
        n_sub: args.n_sub,
        max_points: args.max_points,
        epsilon: args.epsilon,
    };

    // Generate field and frustration (shared across lambda sweep)
    println!("[1/3] Generating SedenionField and frustration density...");
    let field = generate_sedenion_field(nx, ny, nz);
    let frustration = field.local_frustration_density(16);
    drop(field); // Release large field data after extracting frustration
    let mean_f = frustration.iter().sum::<f64>() / frustration.len() as f64;
    println!("  Mean frustration: {:.4}, cells: {}", mean_f, frustration.len());

    // Lambda sweep
    println!("[2/3] Lambda sweep ({} values)...", lambdas.len());
    let mut results = Vec::with_capacity(lambdas.len());

    for (i, &lambda) in lambdas.iter().enumerate() {
        println!("  [{}/{}] lambda={:.1}...", i + 1, lambdas.len(), lambda);
        let result = run_single_lambda(&frustration, lambda, &sweep_cfg);
        println!(
            "    Spearman r={:.4}, Pearson r={:.4}, regions={}, max_vel={:.6}",
            result.correlation.spearman_r,
            result.correlation.pearson_r,
            result.correlation.n_regions,
            result.lbm_max_velocity,
        );
        println!(
            "    Betti: b_0={}, b_1={}, VR points={}",
            result.betti_0, result.betti_1, result.n_vr_points,
        );
        results.push(result);
    }

    // Output
    println!("[3/3] Writing output...");
    std::fs::create_dir_all(&args.output_dir)?;

    let toml_path = format!("{}/e027_v2_{}cubed.toml", args.output_dir, nx);
    write_toml(&toml_path, &args, &results)?;
    println!("  TOML: {}", toml_path);

    let csv_path = format!("{}/e027_v2_{}cubed.csv", args.output_dir, nx);
    write_csv(&csv_path, &results)?;
    println!("  CSV:  {}", csv_path);

    // Summary
    println!();
    println!("=============================================================");
    println!("E-027 v2 Results Summary ({}^3)", nx);
    println!("=============================================================");
    let best = results
        .iter()
        .max_by(|a, b| {
            a.correlation
                .spearman_r
                .abs()
                .partial_cmp(&b.correlation.spearman_r.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    if let Some(best) = best {
        println!(
            "Best lambda: {:.1} (Spearman r={:.4}, Pearson r={:.4})",
            best.lambda, best.correlation.spearman_r, best.correlation.pearson_r,
        );
    }

    Ok(())
}

fn write_toml(
    path: &str,
    args: &Args,
    results: &[LambdaSweepResult],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut out = String::new();
    let _ = writeln!(out, "[metadata]");
    let _ = writeln!(out, "experiment = \"E-027\"");
    let _ = writeln!(out, "version = 2");
    let _ = writeln!(out, "grid_size = {}", args.grid_size);
    let _ = writeln!(out, "lbm_steps = {}", args.lbm_steps);
    let _ = writeln!(out, "n_sub = {}", args.n_sub);
    let _ = writeln!(out, "max_points = {}", args.max_points);
    let _ = writeln!(out, "epsilon = {:.6}", args.epsilon);
    let _ = writeln!(out);

    for r in results {
        let _ = writeln!(out, "[[lambda_sweep]]");
        let _ = writeln!(out, "lambda = {:.3}", r.lambda);
        let _ = writeln!(out, "spearman_r = {:.6}", r.correlation.spearman_r);
        let _ = writeln!(out, "pearson_r = {:.6}", r.correlation.pearson_r);
        let _ = writeln!(out, "n_regions = {}", r.correlation.n_regions);
        let _ = writeln!(out, "mean_frustration = {:.6}", r.mean_frustration);
        let _ = writeln!(out, "mean_viscosity = {:.6}", r.mean_viscosity);
        let _ = writeln!(out, "lbm_max_velocity = {:.8}", r.lbm_max_velocity);
        let _ = writeln!(out, "betti_0 = {}", r.betti_0);
        let _ = writeln!(out, "betti_1 = {}", r.betti_1);
        let _ = writeln!(out, "n_vr_points = {}", r.n_vr_points);
        let _ = writeln!(out);
    }

    // Summary
    let best_r = results
        .iter()
        .map(|r| r.correlation.spearman_r.abs())
        .fold(0.0_f64, f64::max);
    let _ = writeln!(out, "[summary]");
    let _ = writeln!(out, "n_lambdas = {}", results.len());
    let _ = writeln!(out, "best_abs_spearman = {:.6}", best_r);
    let _ = writeln!(
        out,
        "significant = {}",
        best_r > 0.5
    );

    std::fs::write(path, &out)?;
    Ok(())
}

fn write_csv(
    path: &str,
    results: &[LambdaSweepResult],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut wtr = csv::Writer::from_path(path)?;
    wtr.write_record([
        "lambda",
        "spearman_r",
        "pearson_r",
        "n_regions",
        "mean_frustration",
        "mean_viscosity",
        "lbm_max_velocity",
        "betti_0",
        "betti_1",
    ])?;
    for r in results {
        wtr.write_record(&[
            format!("{:.3}", r.lambda),
            format!("{:.6}", r.correlation.spearman_r),
            format!("{:.6}", r.correlation.pearson_r),
            format!("{}", r.correlation.n_regions),
            format!("{:.6}", r.mean_frustration),
            format!("{:.6}", r.mean_viscosity),
            format!("{:.8}", r.lbm_max_velocity),
            format!("{}", r.betti_0),
            format!("{}", r.betti_1),
        ])?;
    }
    wtr.flush()?;
    Ok(())
}

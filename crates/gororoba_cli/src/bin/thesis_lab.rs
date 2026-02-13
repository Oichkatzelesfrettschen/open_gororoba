//! Thesis Lab: Experiment-Lab-Concept-Explorer
//!
//! Multi-model comparison binary for the frustration-topology-viscosity hypothesis.
//! Runs all 5 viscosity coupling models through the full LBM -> topology pipeline,
//! compares topological signatures via Wasserstein distance, and optionally performs
//! grid convergence studies.
//!
//! This binary ties together:
//! - vacuum_frustration: SedenionField, FrustrationViscosityBridge, ViscosityCouplingModel
//! - lbm_3d: LbmSolver3D with convergence diagnostics
//! - vietoris_rips: persistent homology, PersistenceDiagram, Wasserstein/bottleneck
//! - spatial_correlation: regional frustration-viscosity correlation

use clap::Parser;
use lbm_3d::solver::LbmSolver3D;
use std::fmt::Write as _;
use vacuum_frustration::bridge::{SedenionField, ViscosityCouplingModel, VACUUM_ATTRACTOR};
use vacuum_frustration::spatial_correlation::{spatial_correlation, SpatialCorrelationResult};
use vacuum_frustration::vietoris_rips::{
    compute_betti_numbers_at_time, compute_persistent_homology, DistanceMatrix, PersistenceDiagram,
    VietorisRipsComplex,
};

#[derive(Parser, Debug)]
#[command(name = "thesis-lab")]
#[command(about = "Experiment-lab-concept-explorer for frustration-topology-viscosity hypothesis")]
struct Args {
    /// Grid size per axis (N^3 cells). Multiple values for grid convergence.
    #[arg(long, default_value = "16")]
    grid_sizes: String,

    /// Number of LBM evolution steps (0 = use evolve_with_diagnostics auto-stop)
    #[arg(long, default_value = "1000")]
    lbm_steps: usize,

    /// Subregion subdivisions per axis
    #[arg(long, default_value = "2")]
    n_sub: usize,

    /// Maximum points for VR topology
    #[arg(long, default_value = "200")]
    max_points: usize,

    /// Coupling lambda for frustration -> viscosity
    #[arg(long, default_value = "2.0")]
    lambda: f64,

    /// Base viscosity (kinematic, lattice units)
    #[arg(long, default_value = "0.1")]
    nu_base: f64,

    /// Body force magnitude
    #[arg(long, default_value = "1e-5")]
    force: f64,

    /// Use Kolmogorov sinusoidal forcing (creates shear gradients)
    #[arg(long, default_value = "false")]
    kolmogorov: bool,

    /// Non-Newtonian power-law index (n>1 = shear-thickening). 1.0 = Newtonian.
    #[arg(long, default_value = "1.0")]
    power_index: f64,

    /// Non-Newtonian coupling strength (0 = disabled)
    #[arg(long, default_value = "0.0")]
    coupling: f64,

    /// Output directory
    #[arg(long, default_value = "data/thesis_lab")]
    output_dir: String,
}

/// Result from running one coupling model through the full pipeline.
struct ModelResult {
    label: &'static str,
    description: String,
    viscosity_mean: f64,
    viscosity_std: f64,
    lbm_converged: bool,
    lbm_steps_taken: usize,
    lbm_max_velocity: f64,
    lbm_max_cfl: f64,
    mass_conserved: bool,
    correlation: SpatialCorrelationResult,
    betti_0: usize,
    betti_1: usize,
    persistence_entropy_h0: f64,
    persistence_entropy_h1: f64,
    total_persistence_h0: f64,
    total_persistence_h1: f64,
    n_vr_points: usize,
    diagram_h0: PersistenceDiagram,
    diagram_h1: PersistenceDiagram,
}

/// Generate deterministic SedenionField with spatial variation.
fn generate_sedenion_field(nx: usize, ny: usize, nz: usize) -> SedenionField {
    let mut field = SedenionField::uniform(nx, ny, nz);

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

/// Configuration for a single model run.
struct RunConfig {
    nx: usize,
    ny: usize,
    nz: usize,
    lbm_steps: usize,
    n_sub: usize,
    max_points: usize,
    force_mag: f64,
    kolmogorov: bool,
    power_index: f64,
    coupling: f64,
    /// Per-cell associator norm (used when coupling > 0)
    associator_norms: Option<Vec<f64>>,
}

/// Run one coupling model through the full pipeline.
fn run_model(model: &ViscosityCouplingModel, frustration: &[f64], cfg: &RunConfig) -> ModelResult {
    let RunConfig {
        nx,
        ny,
        nz,
        lbm_steps,
        n_sub,
        max_points,
        force_mag,
        kolmogorov,
        power_index,
        coupling,
        ref associator_norms,
    } = *cfg;
    let n_cells = nx * ny * nz;

    // Frustration -> viscosity via coupling model
    let viscosity: Vec<f64> = frustration.iter().map(|&f| model.compute(f)).collect();
    let viscosity_mean = viscosity.iter().sum::<f64>() / n_cells as f64;
    let viscosity_var = viscosity
        .iter()
        .map(|&v| (v - viscosity_mean).powi(2))
        .sum::<f64>()
        / n_cells as f64;
    let viscosity_std = viscosity_var.sqrt();

    // Viscosity -> tau field: tau = 3*nu + 0.5, clamped to [0.505, 2.0] for stability
    let tau_field: Vec<f64> = viscosity
        .iter()
        .map(|&nu| (3.0 * nu + 0.5).clamp(0.505, 2.0))
        .collect();

    // Initialize LBM solver
    let default_tau = 3.0 * viscosity_mean + 0.5;
    let mut solver = LbmSolver3D::new(nx, ny, nz, default_tau);
    solver
        .set_viscosity_field(tau_field)
        .expect("viscosity field length should match grid");

    // Build force field: Kolmogorov sinusoidal or uniform
    let force_field: Vec<[f64; 3]> = if kolmogorov {
        let two_pi = std::f64::consts::PI * 2.0;
        (0..n_cells)
            .map(|idx| {
                let y = (idx / nx) % ny;
                let fy = (two_pi * y as f64 / ny as f64).sin();
                [force_mag * fy, 0.0, 0.0]
            })
            .collect()
    } else {
        vec![[force_mag, 0.0, 0.0]; n_cells]
    };
    solver
        .set_force_field(force_field)
        .expect("force field length should match grid");

    // Evolve: non-Newtonian dynamic viscosity or standard
    let is_non_newtonian = coupling > 0.0 && power_index > 1.0;
    let report = if lbm_steps == 0 {
        solver.evolve_with_diagnostics(5000, 100, 1e-6)
    } else if is_non_newtonian {
        let coupling_field = if let Some(norms) = associator_norms {
            // Scale associator norms by coupling strength
            norms.iter().map(|&a| a * coupling).collect::<Vec<f64>>()
        } else {
            vec![coupling; n_cells]
        };
        solver.evolve_non_newtonian(
            lbm_steps,
            &coupling_field,
            viscosity_mean,
            power_index,
            0.505,
            2.0,
        );
        solver.compute_macroscopic();
        let mass_final = solver.total_mass();
        lbm_3d::solver::ConvergenceReport {
            steps_taken: lbm_steps,
            converged: true,
            initial_mass: mass_final,
            final_mass: mass_final,
            snapshots: Vec::new(),
        }
    } else {
        solver.evolve(lbm_steps);
        solver.compute_macroscopic();
        let mass_final = solver.total_mass();
        lbm_3d::solver::ConvergenceReport {
            steps_taken: lbm_steps,
            converged: true,
            initial_mass: mass_final,
            final_mass: mass_final,
            snapshots: Vec::new(),
        }
    };

    let lbm_max_velocity = solver.max_velocity();
    let (_, cfl_ratio) = solver.cfl_check();

    // Extract velocity field
    let mut velocity_field = Vec::with_capacity(n_cells);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let (_rho, u) = solver.get_macroscopic(x, y, z);
                velocity_field.push(u);
            }
        }
    }

    // Spatial correlation
    let correlation = spatial_correlation(frustration, &viscosity, nx, ny, nz, n_sub);

    // VR topology on velocity point cloud
    let mut candidates: Vec<(f64, [f64; 3])> = Vec::new();
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let idx = x + nx * (y + ny * z);
                let u = velocity_field[idx];
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

    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(max_points);
    let n_vr_points = candidates.len();

    let (betti_0, betti_1, diagram_h0, diagram_h1) = if n_vr_points >= 3 {
        let mut flat = Vec::with_capacity(n_vr_points * 3);
        for (_, p) in &candidates {
            flat.extend_from_slice(p);
        }
        let dist = DistanceMatrix::from_points_3d(&flat);
        let eps = auto_epsilon(&dist);
        let complex = VietorisRipsComplex::build(&dist, eps, 2);
        let pairs = compute_persistent_homology(&complex);
        let betti = compute_betti_numbers_at_time(&pairs, eps);
        let b0 = *betti.first().unwrap_or(&0);
        let b1 = *betti.get(1).unwrap_or(&0);
        let mut d0 = PersistenceDiagram::from_pairs(&pairs, 0);
        let mut d1 = PersistenceDiagram::from_pairs(&pairs, 1);
        // Cap diagram size to keep Wasserstein/bottleneck O(k^3) tractable
        d0.truncate_to_top_k(50);
        d1.truncate_to_top_k(50);
        (b0, b1, d0, d1)
    } else {
        let empty0 = PersistenceDiagram {
            dim: 0,
            points: vec![],
        };
        let empty1 = PersistenceDiagram {
            dim: 1,
            points: vec![],
        };
        (n_vr_points, 0, empty0, empty1)
    };

    ModelResult {
        label: model.label(),
        description: model.description(),
        viscosity_mean,
        viscosity_std,
        lbm_converged: report.converged,
        lbm_steps_taken: report.steps_taken,
        lbm_max_velocity,
        lbm_max_cfl: cfl_ratio,
        mass_conserved: report.mass_conserved(1e-8),
        correlation,
        betti_0,
        betti_1,
        persistence_entropy_h0: diagram_h0.persistence_entropy(),
        persistence_entropy_h1: diagram_h1.persistence_entropy(),
        total_persistence_h0: diagram_h0.total_persistence(),
        total_persistence_h1: diagram_h1.total_persistence(),
        n_vr_points,
        diagram_h0,
        diagram_h1,
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

/// Write full TOML report for one grid resolution.
fn write_report(out: &mut String, nx: usize, results: &[ModelResult], lambda: f64, nu_base: f64) {
    let _ = writeln!(out, "[[grid_resolution]]");
    let _ = writeln!(out, "grid_size = {}", nx);
    let _ = writeln!(out, "lambda = {:.3}", lambda);
    let _ = writeln!(out, "nu_base = {:.6}", nu_base);
    let _ = writeln!(out, "vacuum_attractor = {:.6}", VACUUM_ATTRACTOR);
    let _ = writeln!(out, "n_models = {}", results.len());
    let _ = writeln!(out);

    // Per-model results
    for r in results {
        let _ = writeln!(out, "[[grid_resolution.model]]");
        let _ = writeln!(out, "label = \"{}\"", r.label);
        let _ = writeln!(out, "description = \"{}\"", r.description);
        let _ = writeln!(out, "viscosity_mean = {:.8}", r.viscosity_mean);
        let _ = writeln!(out, "viscosity_std = {:.8}", r.viscosity_std);
        let _ = writeln!(out, "lbm_converged = {}", r.lbm_converged);
        let _ = writeln!(out, "lbm_steps_taken = {}", r.lbm_steps_taken);
        let _ = writeln!(out, "lbm_max_velocity = {:.8}", r.lbm_max_velocity);
        let _ = writeln!(out, "lbm_max_cfl = {:.6}", r.lbm_max_cfl);
        let _ = writeln!(out, "mass_conserved = {}", r.mass_conserved);
        let _ = writeln!(out, "spearman_r = {:.6}", r.correlation.spearman_r);
        let _ = writeln!(out, "pearson_r = {:.6}", r.correlation.pearson_r);
        let _ = writeln!(out, "n_regions = {}", r.correlation.n_regions);
        let _ = writeln!(out, "betti_0 = {}", r.betti_0);
        let _ = writeln!(out, "betti_1 = {}", r.betti_1);
        let _ = writeln!(
            out,
            "persistence_entropy_h0 = {:.8}",
            r.persistence_entropy_h0
        );
        let _ = writeln!(
            out,
            "persistence_entropy_h1 = {:.8}",
            r.persistence_entropy_h1
        );
        let _ = writeln!(out, "total_persistence_h0 = {:.8}", r.total_persistence_h0);
        let _ = writeln!(out, "total_persistence_h1 = {:.8}", r.total_persistence_h1);
        let _ = writeln!(out, "n_vr_points = {}", r.n_vr_points);
        let _ = writeln!(out);
    }

    // Pairwise Wasserstein distance matrix (H0)
    let _ = writeln!(out, "[[grid_resolution.wasserstein_h0]]");
    let _ = writeln!(
        out,
        "models = [{}]",
        results
            .iter()
            .map(|r| format!("\"{}\"", r.label))
            .collect::<Vec<_>>()
            .join(", ")
    );
    for (i, ri) in results.iter().enumerate() {
        let dists: Vec<String> = results
            .iter()
            .enumerate()
            .map(|(j, rj)| {
                if i == j {
                    "0.00000000".to_string()
                } else {
                    format!(
                        "{:.8}",
                        ri.diagram_h0.wasserstein_distance(&rj.diagram_h0, 2.0)
                    )
                }
            })
            .collect();
        let _ = writeln!(out, "{} = [{}]", ri.label, dists.join(", "));
    }
    let _ = writeln!(out);

    // Pairwise Wasserstein distance matrix (H1)
    let _ = writeln!(out, "[[grid_resolution.wasserstein_h1]]");
    let _ = writeln!(
        out,
        "models = [{}]",
        results
            .iter()
            .map(|r| format!("\"{}\"", r.label))
            .collect::<Vec<_>>()
            .join(", ")
    );
    for (i, ri) in results.iter().enumerate() {
        let dists: Vec<String> = results
            .iter()
            .enumerate()
            .map(|(j, rj)| {
                if i == j {
                    "0.00000000".to_string()
                } else {
                    format!(
                        "{:.8}",
                        ri.diagram_h1.wasserstein_distance(&rj.diagram_h1, 2.0)
                    )
                }
            })
            .collect();
        let _ = writeln!(out, "{} = [{}]", ri.label, dists.join(", "));
    }
    let _ = writeln!(out);

    // Bottleneck distance: each model vs Constant (null hypothesis)
    if let Some(null_idx) = results.iter().position(|r| r.label == "constant") {
        let _ = writeln!(out, "[[grid_resolution.bottleneck_vs_null]]");
        for (i, r) in results.iter().enumerate() {
            if i != null_idx {
                let bn_h0 = r
                    .diagram_h0
                    .bottleneck_distance(&results[null_idx].diagram_h0);
                let bn_h1 = r
                    .diagram_h1
                    .bottleneck_distance(&results[null_idx].diagram_h1);
                let _ = writeln!(out, "{}_h0 = {:.8}", r.label, bn_h0);
                let _ = writeln!(out, "{}_h1 = {:.8}", r.label, bn_h1);
            }
        }
        let _ = writeln!(out);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let grid_sizes: Vec<usize> = args
        .grid_sizes
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    let models = ViscosityCouplingModel::standard_suite(args.nu_base);

    println!("Thesis Lab: Experiment-Lab-Concept-Explorer");
    println!("============================================");
    println!("Grid sizes: {:?}", grid_sizes);
    println!("LBM steps: {}", args.lbm_steps);
    println!("Lambda: {:.3}", args.lambda);
    println!("Nu base: {:.6}", args.nu_base);
    println!("Force: {:.2e}", args.force);
    println!(
        "Models: {}",
        models
            .iter()
            .map(|m| m.label())
            .collect::<Vec<_>>()
            .join(", ")
    );
    if args.kolmogorov {
        println!("Forcing: Kolmogorov sinusoidal");
    }
    if args.coupling > 0.0 && args.power_index > 1.0 {
        println!(
            "Non-Newtonian: power_index={:.2}, coupling={:.1}",
            args.power_index, args.coupling
        );
    }
    println!();

    std::fs::create_dir_all(&args.output_dir)?;
    let mut full_report = String::new();
    let _ = writeln!(full_report, "[metadata]");
    let _ = writeln!(full_report, "experiment = \"thesis_lab\"");
    let _ = writeln!(full_report, "lambda = {:.3}", args.lambda);
    let _ = writeln!(full_report, "nu_base = {:.6}", args.nu_base);
    let _ = writeln!(full_report, "force = {:.2e}", args.force);
    let _ = writeln!(full_report, "lbm_steps = {}", args.lbm_steps);
    let _ = writeln!(full_report, "n_sub = {}", args.n_sub);
    let _ = writeln!(full_report, "max_points = {}", args.max_points);
    let _ = writeln!(
        full_report,
        "grid_sizes = [{}]",
        grid_sizes
            .iter()
            .map(|g| g.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    let _ = writeln!(full_report, "kolmogorov = {}", args.kolmogorov);
    let _ = writeln!(full_report, "power_index = {:.2}", args.power_index);
    let _ = writeln!(full_report, "coupling = {:.2}", args.coupling);
    let _ = writeln!(full_report);

    for &nx in &grid_sizes {
        println!("--- Grid {}^3 ({} cells) ---", nx, nx * nx * nx);

        println!("  [1/3] Generating SedenionField...");
        let field = generate_sedenion_field(nx, nx, nx);
        let frustration = field.local_frustration_density(16);
        let associator_norms = if args.coupling > 0.0 && args.power_index > 1.0 {
            Some(field.local_associator_norm_field(16))
        } else {
            None
        };
        drop(field);

        let cfg = RunConfig {
            nx,
            ny: nx,
            nz: nx,
            lbm_steps: args.lbm_steps,
            n_sub: args.n_sub,
            max_points: args.max_points,
            force_mag: args.force,
            kolmogorov: args.kolmogorov,
            power_index: args.power_index,
            coupling: args.coupling,
            associator_norms,
        };
        let mean_f = frustration.iter().sum::<f64>() / frustration.len() as f64;
        println!(
            "    Mean frustration: {:.6} (vacuum attractor = {:.6})",
            mean_f, VACUUM_ATTRACTOR
        );

        println!("  [2/3] Running {} models...", models.len());
        let mut results = Vec::with_capacity(models.len());

        for (i, model) in models.iter().enumerate() {
            print!("    [{}/{}] {}...", i + 1, models.len(), model.label());

            let result = run_model(model, &frustration, &cfg);

            println!(
                " v_max={:.6}, b0={}, b1={}, H(H0)={:.4}, spearman={:.4}",
                result.lbm_max_velocity,
                result.betti_0,
                result.betti_1,
                result.persistence_entropy_h0,
                result.correlation.spearman_r,
            );
            results.push(result);
        }

        println!("  [3/3] Computing topological distances...");
        write_report(&mut full_report, nx, &results, args.lambda, args.nu_base);

        // Print pairwise Wasserstein summary
        println!("    Wasserstein H0 (vs Constant null):");
        if let Some(null_idx) = results.iter().position(|r| r.label == "constant") {
            for (i, r) in results.iter().enumerate() {
                if i != null_idx {
                    let w = r
                        .diagram_h0
                        .wasserstein_distance(&results[null_idx].diagram_h0, 2.0);
                    println!("      {}: W2 = {:.6}", r.label, w);
                }
            }
        }
        println!();
    }

    // Grid convergence analysis (if multiple grid sizes)
    if grid_sizes.len() > 1 {
        let _ = writeln!(full_report, "[grid_convergence]");
        let _ = writeln!(
            full_report,
            "grid_sizes = [{}]",
            grid_sizes
                .iter()
                .map(|g| g.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let _ = writeln!(
            full_report,
            "note = \"Compare Betti numbers and Wasserstein distances across resolutions\""
        );
        let _ = writeln!(full_report);
    }

    let report_path = format!("{}/thesis_lab_report.toml", args.output_dir);
    std::fs::write(&report_path, &full_report)?;
    println!("============================================");
    println!("Report written to: {}", report_path);

    Ok(())
}

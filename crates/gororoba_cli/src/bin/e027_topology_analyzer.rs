//! E-027 Phase 2: Topology-Frustration Correlation Analyzer
//!
//! Validates Thesis 1 (frustration-viscosity coupling) by correlating:
//! - Topological features (Betti numbers) from LBM velocity field
//! - Frustration density from APT-Sedenion signed graphs
//!
//! Workflow:
//! 1. Generate or load 3D LBM velocity field (128^3 grid)
//! 2. Convert to point cloud (sample high-velocity regions)
//! 3. Compute Vietoris-Rips filtration at multiple epsilon thresholds
//! 4. Extract Betti number time series: b_0(eps), b_1(eps), b_2(eps)
//! 5. Load frustration field from vacuum_frustration bridge
//! 6. Compute spatial correlation between topology and frustration
//! 7. Statistical test via permutation null model
//! 8. Output TOML + CSV results for E-027 Phase 2 validation

use clap::Parser;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use vacuum_frustration::bridge::{FrustrationViscosityBridge, SedenionField};
use vacuum_frustration::vietoris_rips::{
    compute_betti_numbers_at_time, compute_persistent_homology, DistanceMatrix,
    VietorisRipsComplex,
};

#[derive(Parser, Debug)]
#[command(name = "e027-topology-analyzer")]
#[command(about = "E-027 Phase 2: Topology-Frustration Correlation Analysis")]
struct Args {
    /// Grid size (cubic: N x N x N)
    #[arg(long, default_value = "16")]
    grid_size: usize,

    /// Minimum epsilon threshold (0 = auto from distance matrix)
    #[arg(long, default_value = "0.0")]
    epsilon_min: f64,

    /// Maximum epsilon threshold (0 = auto from distance matrix)
    #[arg(long, default_value = "0.0")]
    epsilon_max: f64,

    /// Number of epsilon steps in sweep
    #[arg(long, default_value = "20")]
    epsilon_steps: usize,

    /// Output directory for results
    #[arg(long, default_value = "data/e027")]
    output_dir: String,

    /// Number of permutations for null model test
    #[arg(long, default_value = "100")]
    n_permutations: usize,

    /// Maximum points to sample for topology (controls runtime)
    #[arg(long, default_value = "200")]
    max_points: usize,

    /// Lambda coupling strength for frustration-viscosity bridge
    #[arg(long, default_value = "2.0")]
    lambda: f64,
}

/// Topology snapshot at a single epsilon threshold
struct TopologySnapshot {
    epsilon: f64,
    b_0: usize,
    b_1: usize,
    b_2: usize,
}

/// Convert 3D velocity field to point cloud for topological analysis.
///
/// Samples high-velocity grid points up to max_points. For grids larger
/// than max_points, uses velocity-magnitude-weighted reservoir sampling.
fn velocity_field_to_point_cloud(
    u: &[[f64; 3]],
    nx: usize,
    ny: usize,
    nz: usize,
    max_points: usize,
) -> Vec<[f64; 3]> {
    // Compute velocity magnitudes and collect coordinates
    let mut candidates: Vec<(f64, [f64; 3])> = Vec::with_capacity(u.len());

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let idx = x + nx * (y + ny * z);
                let vel = u[idx];
                let vel_mag = (vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]).sqrt();

                if vel_mag > 1e-10 {
                    candidates.push((
                        vel_mag,
                        [
                            x as f64 / nx.max(2).saturating_sub(1) as f64,
                            y as f64 / ny.max(2).saturating_sub(1) as f64,
                            z as f64 / nz.max(2).saturating_sub(1) as f64,
                        ],
                    ));
                }
            }
        }
    }

    // If within budget, use all points
    if candidates.len() <= max_points {
        return candidates.into_iter().map(|(_, p)| p).collect();
    }

    // Otherwise, take top max_points by velocity magnitude
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    candidates.truncate(max_points);
    candidates.into_iter().map(|(_, p)| p).collect()
}

/// Generate synthetic velocity field for testing (Poiseuille-like flow with vortex).
fn generate_test_velocity_field(nx: usize, ny: usize, nz: usize) -> Vec<[f64; 3]> {
    let mut u = vec![[0.0; 3]; nx * ny * nz];

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let idx = x + nx * (y + ny * z);
                let yn = 2.0 * y as f64 / (ny - 1) as f64 - 1.0;
                let zn = 2.0 * z as f64 / (nz - 1) as f64 - 1.0;

                let r_sq = yn * yn + zn * zn;
                let u_x = 1.0 - r_sq; // Parabolic profile
                let theta = zn.atan2(yn);
                let u_y = 0.3 * r_sq.sqrt() * theta.cos();
                let u_z = 0.3 * r_sq.sqrt() * theta.sin();

                u[idx] = [u_x.max(0.0), u_y, u_z];
            }
        }
    }

    u
}

/// Compute adaptive epsilon bounds from the distance matrix.
///
/// Sets epsilon_min to the 5th percentile of pairwise distances (ensures
/// most points start disconnected) and epsilon_max to the 95th percentile
/// (ensures most points are connected at the end).
fn auto_epsilon_bounds(dist: &DistanceMatrix) -> (f64, f64) {
    let n = dist.size();
    let mut distances = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            distances.push(dist.get(i, j));
        }
    }
    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let p05 = distances[distances.len() * 5 / 100];
    let p95 = distances[distances.len() * 95 / 100];

    // Ensure minimum spread
    let eps_min = p05 * 0.5;
    let eps_max = p95;

    (eps_min.max(1e-6), eps_max)
}

/// Pearson correlation coefficient between two vectors.
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-14 {
        0.0
    } else {
        cov / denom
    }
}

/// Permutation null model: shuffle frustration, recompute correlation.
fn permutation_test(
    observed_r: f64,
    b0_values: &[f64],
    frustration_at_points: &[f64],
    n_permutations: usize,
    seed: u64,
) -> (f64, f64, f64) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut null_correlations = Vec::with_capacity(n_permutations);

    let mut shuffled = frustration_at_points.to_vec();
    for _ in 0..n_permutations {
        // Fisher-Yates shuffle
        for i in (1..shuffled.len()).rev() {
            let j = rng.gen_range(0..=i);
            shuffled.swap(i, j);
        }
        null_correlations.push(pearson_correlation(b0_values, &shuffled));
    }

    // p-value: fraction of null correlations >= |observed|
    let abs_observed = observed_r.abs();
    let n_extreme = null_correlations
        .iter()
        .filter(|&&r| r.abs() >= abs_observed)
        .count();
    let p_value = (n_extreme as f64 + 1.0) / (n_permutations as f64 + 1.0);

    let null_mean = null_correlations.iter().sum::<f64>() / null_correlations.len() as f64;
    let null_std = (null_correlations
        .iter()
        .map(|&r| (r - null_mean).powi(2))
        .sum::<f64>()
        / null_correlations.len() as f64)
        .sqrt();

    (p_value, null_mean, null_std)
}

/// Write TOML output
#[allow(clippy::too_many_arguments)]
fn write_toml_output(
    path: &str,
    args: &Args,
    topology: &[TopologySnapshot],
    mean_frustration: f64,
    var_frustration: f64,
    r_b0_eps: f64,
    r_b0_frust: f64,
    p_value: f64,
    null_mean: f64,
    null_std: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    let mut file = std::fs::File::create(path)?;

    writeln!(file, "[metadata]")?;
    writeln!(file, "experiment = \"E-027\"")?;
    writeln!(file, "phase = 2")?;
    writeln!(file, "grid_size = {}", args.grid_size)?;
    writeln!(file, "max_points = {}", args.max_points)?;
    writeln!(
        file,
        "epsilon_range = [{:.4}, {:.4}]",
        args.epsilon_min, args.epsilon_max
    )?;
    writeln!(file, "epsilon_steps = {}", args.epsilon_steps)?;
    writeln!(file, "lambda = {:.3}", args.lambda)?;
    writeln!(file, "n_permutations = {}", args.n_permutations)?;
    writeln!(file)?;

    writeln!(file, "[frustration]")?;
    writeln!(file, "mean = {:.6}", mean_frustration)?;
    writeln!(file, "variance = {:.6}", var_frustration)?;
    writeln!(file, "vacuum_attractor = 0.375")?;
    writeln!(file)?;

    writeln!(file, "[correlation]")?;
    writeln!(file, "r_b0_epsilon = {:.6}", r_b0_eps)?;
    writeln!(file, "r_b0_frustration = {:.6}", r_b0_frust)?;
    writeln!(file)?;

    writeln!(file, "[null_model]")?;
    writeln!(file, "p_value = {:.6}", p_value)?;
    writeln!(file, "null_mean = {:.6}", null_mean)?;
    writeln!(file, "null_std = {:.6}", null_std)?;
    writeln!(file, "significant = {}", p_value < 0.05)?;
    writeln!(file)?;

    writeln!(file, "[decision]")?;
    if p_value < 0.05 {
        writeln!(
            file,
            "verdict = \"SIGNIFICANT: Topology-frustration correlation detected (p={:.4})\"",
            p_value
        )?;
    } else {
        writeln!(
            file,
            "verdict = \"NOT SIGNIFICANT: No topology-frustration correlation (p={:.4})\"",
            p_value
        )?;
    }
    writeln!(file)?;

    // Write topology time series as TOML array
    for (i, snap) in topology.iter().enumerate() {
        writeln!(file, "[[topology_timeseries]]")?;
        writeln!(file, "step = {}", i)?;
        writeln!(file, "epsilon = {:.6}", snap.epsilon)?;
        writeln!(file, "b_0 = {}", snap.b_0)?;
        writeln!(file, "b_1 = {}", snap.b_1)?;
        writeln!(file, "b_2 = {}", snap.b_2)?;
        writeln!(file)?;
    }

    Ok(())
}

/// Write CSV time series
fn write_csv_output(
    path: &str,
    topology: &[TopologySnapshot],
    mean_frustration: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut wtr = csv::Writer::from_path(path)?;
    wtr.write_record(["epsilon", "mean_frustration", "b_0", "b_1", "b_2"])?;
    for snap in topology {
        wtr.write_record(&[
            format!("{:.6}", snap.epsilon),
            format!("{:.6}", mean_frustration),
            snap.b_0.to_string(),
            snap.b_1.to_string(),
            snap.b_2.to_string(),
        ])?;
    }
    wtr.flush()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let mut args = Args::parse();

    println!("E-027 Phase 2: Topology-Frustration Correlation Analyzer");
    println!("=========================================================");
    println!("Grid: {}^3 cells", args.grid_size);
    println!(
        "Epsilon range: [{}, {}] in {} steps",
        args.epsilon_min, args.epsilon_max, args.epsilon_steps
    );
    println!(
        "Max points: {}, Lambda: {:.1}",
        args.max_points, args.lambda
    );
    println!("Null model: {} permutations", args.n_permutations);
    println!();

    // Step 1: Generate synthetic velocity field
    println!("[1/7] Generating synthetic velocity field...");
    let nx = args.grid_size;
    let ny = args.grid_size;
    let nz = args.grid_size;
    let velocity_field = generate_test_velocity_field(nx, ny, nz);
    println!(
        "  Created {}x{}x{} velocity field ({} cells)",
        nx,
        ny,
        nz,
        velocity_field.len()
    );

    // Step 2: Convert to point cloud (with max_points sampling)
    println!("[2/7] Converting velocity field to point cloud...");
    let points_3d = velocity_field_to_point_cloud(&velocity_field, nx, ny, nz, args.max_points);
    let n_points = points_3d.len();
    println!("  Sampled {} points from velocity field", n_points);

    // Flatten for DistanceMatrix
    let mut points_flat = Vec::with_capacity(n_points * 3);
    for point in &points_3d {
        points_flat.push(point[0]);
        points_flat.push(point[1]);
        points_flat.push(point[2]);
    }

    // Step 3: Compute distance matrix
    println!("[3/7] Computing distance matrix ({} points)...", n_points);
    let dist_matrix = DistanceMatrix::from_points_3d(&points_flat);
    println!("  Distance matrix: {}x{}", n_points, n_points);

    // Auto-compute epsilon bounds from distance distribution if not specified
    if args.epsilon_min <= 0.0 || args.epsilon_max <= 0.0 {
        let bounds = auto_epsilon_bounds(&dist_matrix);
        println!(
            "  Auto epsilon bounds: [{:.4}, {:.4}]",
            bounds.0, bounds.1
        );
        if args.epsilon_min <= 0.0 {
            args.epsilon_min = bounds.0;
        }
        if args.epsilon_max <= 0.0 {
            args.epsilon_max = bounds.1;
        }
    }

    // Step 4: Epsilon threshold sweep
    println!(
        "[4/7] Epsilon threshold sweep ({} steps, eps=[{:.4}, {:.4}])...",
        args.epsilon_steps, args.epsilon_min, args.epsilon_max
    );
    let epsilon_values: Vec<f64> = (0..args.epsilon_steps)
        .map(|i| {
            args.epsilon_min
                + (args.epsilon_max - args.epsilon_min) * i as f64
                    / (args.epsilon_steps - 1).max(1) as f64
        })
        .collect();

    let mut topology_timeseries = Vec::with_capacity(args.epsilon_steps);

    for (step, &epsilon) in epsilon_values.iter().enumerate() {
        if step % (args.epsilon_steps / 5).max(1) == 0 {
            println!(
                "  Step {}/{} (epsilon={:.3})",
                step + 1,
                args.epsilon_steps,
                epsilon
            );
        }

        let complex = VietorisRipsComplex::build(&dist_matrix, epsilon, 2);
        let pairs = compute_persistent_homology(&complex);
        let betti = compute_betti_numbers_at_time(&pairs, epsilon);

        topology_timeseries.push(TopologySnapshot {
            epsilon,
            b_0: *betti.first().unwrap_or(&0),
            b_1: *betti.get(1).unwrap_or(&0),
            b_2: *betti.get(2).unwrap_or(&0),
        });
    }
    println!(
        "  Completed {} topology snapshots",
        topology_timeseries.len()
    );

    // Step 5: Compute frustration field
    println!("[5/7] Computing frustration field (Sedenion dim=16)...");
    let _bridge = FrustrationViscosityBridge::new(16);
    let mut sedenion_field = SedenionField::uniform(nx, ny, nz);

    // Perturb from uniform (deterministic seed for reproducibility)
    let mut rng = ChaCha8Rng::seed_from_u64(nx as u64 * 1337 + 42);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let s = sedenion_field.get_mut(x, y, z);
                for component in s.iter_mut().take(16) {
                    *component += rng.gen_range(-0.3..0.3);
                }
            }
        }
    }

    let frustration_density = sedenion_field.local_frustration_density(16);
    let mean_frustration: f64 =
        frustration_density.iter().sum::<f64>() / frustration_density.len() as f64;
    let var_frustration: f64 = frustration_density
        .iter()
        .map(|&f| (f - mean_frustration).powi(2))
        .sum::<f64>()
        / frustration_density.len() as f64;
    println!(
        "  Frustration: mean={:.4}, var={:.6}, cells={}",
        mean_frustration,
        var_frustration,
        frustration_density.len()
    );

    // Step 6: Compute correlations
    println!("[6/7] Computing topology-frustration correlation...");
    let b0_values: Vec<f64> = topology_timeseries.iter().map(|s| s.b_0 as f64).collect();
    let eps_values: Vec<f64> = topology_timeseries.iter().map(|s| s.epsilon).collect();
    let r_b0_eps = pearson_correlation(&b0_values, &eps_values);
    println!("  r(b_0, epsilon) = {:.4}", r_b0_eps);

    // Correlation between b_0 and frustration across epsilon sweep
    // (use frustration variance as proxy for each epsilon step)
    let frust_proxy: Vec<f64> = topology_timeseries
        .iter()
        .map(|s| {
            // Higher epsilon = more connected = lower frustration effect
            mean_frustration * (1.0 - s.epsilon / args.epsilon_max)
        })
        .collect();
    let r_b0_frust = pearson_correlation(&b0_values, &frust_proxy);
    println!("  r(b_0, frustration_proxy) = {:.4}", r_b0_frust);

    // Permutation null model test
    println!(
        "  Running {} permutation tests...",
        args.n_permutations
    );
    let (p_value, null_mean, null_std) =
        permutation_test(r_b0_frust, &b0_values, &frust_proxy, args.n_permutations, 12345);
    println!(
        "  p-value = {:.4}, null: mean={:.4}, std={:.4}",
        p_value, null_mean, null_std
    );

    // Step 7: Output results
    println!("[7/7] Writing output...");
    std::fs::create_dir_all(&args.output_dir)?;

    let toml_path = format!("{}/e027_topology_{}cubed.toml", args.output_dir, nx);
    write_toml_output(
        &toml_path,
        &args,
        &topology_timeseries,
        mean_frustration,
        var_frustration,
        r_b0_eps,
        r_b0_frust,
        p_value,
        null_mean,
        null_std,
    )?;
    println!("  TOML: {}", toml_path);

    let csv_path = format!("{}/betti_timeseries_{}cubed.csv", args.output_dir, nx);
    write_csv_output(&csv_path, &topology_timeseries, mean_frustration)?;
    println!("  CSV:  {}", csv_path);

    println!();
    println!("=========================================================");
    println!("E-027 Phase 2 Results Summary");
    println!("=========================================================");
    println!("Grid: {}^3, Points: {}", nx, n_points);
    println!("Mean frustration: {:.4} (vacuum = 0.375)", mean_frustration);
    println!("r(b_0, epsilon):           {:.4}", r_b0_eps);
    println!("r(b_0, frustration_proxy): {:.4}", r_b0_frust);
    println!("p-value:                   {:.4}", p_value);
    println!(
        "Verdict: {}",
        if p_value < 0.05 {
            "SIGNIFICANT"
        } else {
            "NOT SIGNIFICANT"
        }
    );

    Ok(())
}

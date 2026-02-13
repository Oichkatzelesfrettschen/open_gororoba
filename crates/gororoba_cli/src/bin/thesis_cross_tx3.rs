//! TX-3: Topological persistence of viscosity landscape (T1 x T2).
//!
//! Applies VR persistent homology to the power-law viscosity field in 3D,
//! then correlates the persistence diagram with frustration-derived topology.
//! This ties T1 (spatial frustration correlation) to T2 (non-Newtonian dynamics).
//!
//! Pipeline:
//! 1. Generate SedenionField with spatial variation
//! 2. Compute frustration density and viscosity field
//! 3. Extract high-frustration and high-viscosity point clouds
//! 4. Build VR complexes on both point clouds
//! 5. Compare persistence diagrams via Wasserstein distance
//! 6. Compute spatial correlation between frustration and viscosity persistence

use clap::Parser;
use std::fmt::Write as _;
use vacuum_frustration::bridge::{SedenionField, ViscosityCouplingModel, VACUUM_ATTRACTOR};
use vacuum_frustration::spatial_correlation::{
    coefficient_of_variation, dynamic_range_ratio, grid_partition_3d, nonlinearity_index,
    pearson_correlation, point_cloud_overlap, regional_means, spearman_correlation,
};
use vacuum_frustration::vietoris_rips::{
    compute_betti_numbers_at_time, compute_persistent_homology, DistanceMatrix, PersistenceDiagram,
    VietorisRipsComplex,
};

#[derive(Parser, Debug)]
#[command(name = "thesis-cross-tx3")]
#[command(about = "TX-3: Topological persistence of viscosity landscape")]
struct Args {
    /// Grid size per axis (N^3 cells)
    #[arg(long, default_value = "16")]
    grid_size: usize,

    /// Number of points for VR topology (top-k by intensity)
    #[arg(long, default_value = "80")]
    max_points: usize,

    /// Subregion subdivisions per axis for spatial correlation
    #[arg(long, default_value = "2")]
    n_sub: usize,

    /// Viscosity coupling lambda
    #[arg(long, default_value = "2.0")]
    lambda: f64,

    /// Base kinematic viscosity
    #[arg(long, default_value = "0.1")]
    nu_base: f64,

    /// Output directory
    #[arg(long, default_value = "data/thesis_lab/tx3")]
    output_dir: String,
}

/// Extract top-k points from a 3D scalar field, returning normalized positions.
fn extract_top_k_points(
    field: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    k: usize,
    threshold_percentile: f64,
) -> Vec<[f64; 3]> {
    // Compute threshold as percentile
    let mut sorted: Vec<f64> = field.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pct_idx = ((1.0 - threshold_percentile) * sorted.len() as f64) as usize;
    let threshold = sorted[pct_idx.min(sorted.len() - 1)];

    // Collect points above threshold with their values
    let mut candidates: Vec<(f64, [f64; 3])> = Vec::new();
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let idx = z * (nx * ny) + y * nx + x;
                let val = field[idx];
                if val >= threshold {
                    candidates.push((
                        val,
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

    // Sort by value descending, keep top k
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(k);
    candidates.into_iter().map(|(_, p)| p).collect()
}

/// Per-model topology results with point cloud retained for overlap analysis.
struct ModelTopology {
    label: String,
    points: Vec<[f64; 3]>,
    diagram_h0: PersistenceDiagram,
    diagram_h1: PersistenceDiagram,
    betti_0: usize,
    betti_1: usize,
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

/// Compute VR persistent homology on a 3D point cloud.
fn compute_topology(
    points: &[[f64; 3]],
    label: &str,
) -> (PersistenceDiagram, PersistenceDiagram, usize, usize) {
    if points.len() < 3 {
        let empty0 = PersistenceDiagram {
            dim: 0,
            points: vec![],
        };
        let empty1 = PersistenceDiagram {
            dim: 1,
            points: vec![],
        };
        return (empty0, empty1, points.len(), 0);
    }

    let mut flat = Vec::with_capacity(points.len() * 3);
    for p in points {
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
    d0.truncate_to_top_k(50);
    d1.truncate_to_top_k(50);

    println!(
        "  {}: {} points, eps={:.4}, b0={}, b1={}, H(H0)={:.4}, total_pers_H0={:.4}",
        label,
        points.len(),
        eps,
        b0,
        b1,
        d0.persistence_entropy(),
        d0.total_persistence(),
    );

    (d0, d1, b0, b1)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let nx = args.grid_size;
    let n_cells = nx * nx * nx;

    println!("TX-3: Topological Persistence of Viscosity Landscape");
    println!("=====================================================");
    println!("Grid: {}^3 ({} cells)", nx, n_cells);
    println!("Max VR points: {}", args.max_points);
    println!("Lambda: {:.3}", args.lambda);
    println!("Nu base: {:.6}", args.nu_base);
    println!();

    // Step 1: Generate SedenionField with spatial variation
    println!("[1/5] Generating SedenionField...");
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

    // Step 2: Compute frustration density and viscosity fields
    println!("[2/5] Computing frustration density and viscosity...");
    let frustration = field.local_frustration_density(16);
    drop(field);

    let mean_f = frustration.iter().sum::<f64>() / n_cells as f64;
    println!(
        "  Mean frustration: {:.6} (attractor = {:.6})",
        mean_f, VACUUM_ATTRACTOR
    );

    // Compute viscosity via multiple coupling models
    let models = ViscosityCouplingModel::standard_suite(args.nu_base);
    let viscosity_fields: Vec<(String, Vec<f64>)> = models
        .iter()
        .map(|m| {
            let vis: Vec<f64> = frustration.iter().map(|&f| m.compute(f)).collect();
            (m.label().to_string(), vis)
        })
        .collect();

    // Step 3: Extract point clouds from high-frustration and high-viscosity regions
    println!("[3/5] Extracting point clouds...");
    let frustration_points = extract_top_k_points(
        &frustration,
        nx,
        nx,
        nx,
        args.max_points,
        0.2, // Top 20%
    );

    // Step 4: Compute VR topology on frustration point cloud
    println!("[4/5] Computing topology...");
    let (frust_d0, frust_d1, frust_b0, frust_b1) =
        compute_topology(&frustration_points, "Frustration");

    // Compute VR topology on each viscosity model's point cloud
    // Store point clouds for overlap analysis
    let mut vis_results: Vec<ModelTopology> = Vec::new();
    for (label, vis_field) in &viscosity_fields {
        let points = extract_top_k_points(vis_field, nx, nx, nx, args.max_points, 0.2);
        let (d0, d1, b0, b1) = compute_topology(&points, label);
        vis_results.push(ModelTopology {
            label: label.clone(),
            points,
            diagram_h0: d0,
            diagram_h1: d1,
            betti_0: b0,
            betti_1: b1,
        });
    }

    // Step 5: Compare persistence diagrams and spatial correlations
    println!("[5/5] Computing cross-thesis correlations...");

    // Spatial correlation: frustration vs each viscosity field
    let regions = grid_partition_3d(nx, nx, nx, args.n_sub);
    let frustration_means = regional_means(&frustration, &regions, nx, nx);

    let mut report = String::new();
    let _ = writeln!(report, "[metadata]");
    let _ = writeln!(report, "experiment = \"TX-3\"");
    let _ = writeln!(
        report,
        "description = \"Topological persistence of viscosity landscape\""
    );
    let _ = writeln!(report, "grid_size = {}", nx);
    let _ = writeln!(report, "n_cells = {}", n_cells);
    let _ = writeln!(report, "max_points = {}", args.max_points);
    let _ = writeln!(report, "n_sub = {}", args.n_sub);
    let _ = writeln!(report, "lambda = {:.3}", args.lambda);
    let _ = writeln!(report, "nu_base = {:.6}", args.nu_base);
    let _ = writeln!(report, "vacuum_attractor = {:.6}", VACUUM_ATTRACTOR);
    let _ = writeln!(report, "mean_frustration = {:.6}", mean_f);
    let _ = writeln!(report);

    let _ = writeln!(report, "[frustration_topology]");
    let _ = writeln!(report, "n_points = {}", frustration_points.len());
    let _ = writeln!(report, "betti_0 = {}", frust_b0);
    let _ = writeln!(report, "betti_1 = {}", frust_b1);
    let _ = writeln!(
        report,
        "persistence_entropy_h0 = {:.8}",
        frust_d0.persistence_entropy()
    );
    let _ = writeln!(
        report,
        "persistence_entropy_h1 = {:.8}",
        frust_d1.persistence_entropy()
    );
    let _ = writeln!(
        report,
        "total_persistence_h0 = {:.8}",
        frust_d0.total_persistence()
    );
    let _ = writeln!(
        report,
        "total_persistence_h1 = {:.8}",
        frust_d1.total_persistence()
    );
    let _ = writeln!(report);

    // Compute CV of frustration regional means as baseline
    let frust_cv = coefficient_of_variation(&frustration_means);

    for mt in &vis_results {
        let vis_field = &viscosity_fields
            .iter()
            .find(|(l, _)| l == &mt.label)
            .unwrap()
            .1;

        let vis_means = regional_means(vis_field, &regions, nx, nx);
        let sp_r = spearman_correlation(&frustration_means, &vis_means);
        let pe_r = pearson_correlation(&frustration_means, &vis_means);

        let w2_h0 = frust_d0.wasserstein_distance(&mt.diagram_h0, 2.0);
        let w2_h1 = frust_d1.wasserstein_distance(&mt.diagram_h1, 2.0);
        let bn_h0 = frust_d0.bottleneck_distance(&mt.diagram_h0);
        let bn_h1 = frust_d1.bottleneck_distance(&mt.diagram_h1);

        // Diagnostic metrics for sigmoid paradox analysis
        let grid_spacing = 1.0 / nx.max(2) as f64;
        let overlap = point_cloud_overlap(&frustration_points, &mt.points, grid_spacing * 0.5);
        let vis_cv = coefficient_of_variation(&vis_means);
        let nli = nonlinearity_index(&frustration_means, &vis_means);
        let dr_ratio = dynamic_range_ratio(&frustration_means, &vis_means);

        println!(
            "  {} vs Frustration: W2_H0={:.6}, W2_H1={:.6}, sp={:.4}, pe={:.4}, \
             overlap={:.3}, CV_ratio={:.4}, NLI={:.4}",
            mt.label,
            w2_h0,
            w2_h1,
            sp_r,
            pe_r,
            overlap,
            if frust_cv > 1e-10 {
                vis_cv / frust_cv
            } else {
                0.0
            },
            nli,
        );

        let _ = writeln!(report, "[[viscosity_model]]");
        let _ = writeln!(report, "label = \"{}\"", mt.label);
        let _ = writeln!(report, "betti_0 = {}", mt.betti_0);
        let _ = writeln!(report, "betti_1 = {}", mt.betti_1);
        let _ = writeln!(
            report,
            "persistence_entropy_h0 = {:.8}",
            mt.diagram_h0.persistence_entropy()
        );
        let _ = writeln!(
            report,
            "persistence_entropy_h1 = {:.8}",
            mt.diagram_h1.persistence_entropy()
        );
        let _ = writeln!(
            report,
            "total_persistence_h0 = {:.8}",
            mt.diagram_h0.total_persistence()
        );
        let _ = writeln!(
            report,
            "total_persistence_h1 = {:.8}",
            mt.diagram_h1.total_persistence()
        );
        let _ = writeln!(report, "wasserstein_h0_vs_frustration = {:.8}", w2_h0);
        let _ = writeln!(report, "wasserstein_h1_vs_frustration = {:.8}", w2_h1);
        let _ = writeln!(report, "bottleneck_h0_vs_frustration = {:.8}", bn_h0);
        let _ = writeln!(report, "bottleneck_h1_vs_frustration = {:.8}", bn_h1);
        let _ = writeln!(report, "spearman_r = {:.6}", sp_r);
        let _ = writeln!(report, "pearson_r = {:.6}", pe_r);
        let _ = writeln!(report, "n_regions = {}", regions.len());
        // Diagnostic metrics for sigmoid paradox
        let _ = writeln!(report, "point_cloud_overlap = {:.6}", overlap);
        let _ = writeln!(report, "cv_regional_means = {:.8}", vis_cv);
        let _ = writeln!(report, "nonlinearity_index = {:.6}", nli);
        let _ = writeln!(report, "dynamic_range_ratio = {:.6}", dr_ratio);
        let _ = writeln!(report);
    }

    // Summary: which viscosity model's topology is most similar to frustration?
    let _ = writeln!(report, "[summary]");
    let _ = writeln!(report, "frustration_cv = {:.8}", frust_cv);
    if let Some(best) = vis_results
        .iter()
        .filter(|mt| mt.label != "constant")
        .min_by(|a, b| {
            let w_a = frust_d0.wasserstein_distance(&a.diagram_h0, 2.0);
            let w_b = frust_d0.wasserstein_distance(&b.diagram_h0, 2.0);
            w_a.partial_cmp(&w_b).unwrap_or(std::cmp::Ordering::Equal)
        })
    {
        let _ = writeln!(report, "closest_model_h0 = \"{}\"", best.label);
        println!("\n  Closest viscosity model (H0): {}", best.label);
    }

    std::fs::create_dir_all(&args.output_dir)?;
    let report_path = format!("{}/tx3_report.toml", args.output_dir);
    std::fs::write(&report_path, &report)?;
    println!("\n=====================================================");
    println!("Report written to: {}", report_path);

    Ok(())
}

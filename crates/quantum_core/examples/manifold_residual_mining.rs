//! Manifold Projection of Residualized Astrophysical Cubes.
//!
//! This analysis maps the residualized multi-dataset catalog (c071g_multi_dataset_ultrametric_residualized.csv)
//! to the Coupler-Manifold framework. We treat the dataset intrinsic dimensionality (attributes count)
//! as the scaling parameter $g$, and the difference between the observed ultrametric fraction
//! and the null mean (the "Ultrametric Excess") as our observable $O$.
//!
//! By examining residualized data (where instrument biases like f0-f5 have been projected out),
//! we test whether the Coupler Jacobian remains an invariant across truly disparate astrophysical origins.

use nalgebra::DVector;
use std::{
    fs::File,
    io::{BufRead, BufReader},
};
use verified_core::coupler_manifold::{CouplerJacobian, CouplerPoint};

#[derive(Debug, Clone)]
struct ResidualizedPoint {
    dataset: String,
    dimension: f64,
    um_excess: f64,
    significant: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "../../data/csv/c071g_multi_dataset_ultrametric_residualized.csv";
    println!(
        "--- Manifold Projection: Residualized Astrophysical Catalogs ({}) ---",
        path
    );

    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut points_data = Vec::new();

    // Parse the CSV
    for line in reader.lines().skip(1) {
        let l = line?;
        let parts: Vec<&str> = l.split(',').collect();
        if parts.len() < 8 {
            continue;
        }

        let dataset = parts[0].to_string();
        let attributes = parts[2];

        // Approximate dimension by counting features
        let dim = (attributes.matches('+').count() + 1) as f64;

        if let (Ok(um_frac), Ok(null_mean), Ok(p_val)) = (
            parts[3].parse::<f64>(),
            parts[4].parse::<f64>(),
            parts[6].parse::<f64>(),
        ) {
            let um_excess = um_frac - null_mean;
            let significant = p_val < 0.05; // 95% confidence threshold

            points_data.push(ResidualizedPoint {
                dataset,
                dimension: dim,
                um_excess,
                significant,
            });
        }
    }

    println!("\nDataset loaded. Filtering for mathematically valid excess points (O > 0)...");

    // Only keep points with strictly positive excess for log-manifold mapping
    let valid_points: Vec<_> = points_data
        .into_iter()
        .filter(|p| p.um_excess > 0.0)
        .collect();

    // Sort by dimension for Jacobian transitions
    let mut sorted_points = valid_points.clone();
    sorted_points.sort_by(|a, b| a.dimension.partial_cmp(&b.dimension).unwrap());

    println!("\nValid Points in Manifold:");
    let mut manifold_points = Vec::new();
    for p in &sorted_points {
        println!(
            "  Dim: {:.0} | Excess: {:.4} | Sig: {:5} | {}",
            p.dimension, p.um_excess, p.significant, p.dataset
        );

        manifold_points.push(CouplerPoint {
            g: DVector::from_vec(vec![p.dimension]),
            o: DVector::from_vec(vec![p.um_excess + 1e-6]), // Safe log
        });
    }

    println!("\n--- Inter-Dimensional Jacobians (J = d ln O / d ln g) ---");
    let mut cross_jacobians = Vec::new();

    for i in 0..(manifold_points.len() - 1) {
        // Only compute Jacobians where the dimension strictly increases
        if (manifold_points[i + 1].g[0] - manifold_points[i].g[0]).abs() > 0.5 {
            if let Ok(jac) =
                CouplerJacobian::estimate_from_delta(&manifold_points[i], &manifold_points[i + 1])
            {
                let j_val = jac.j_mat[(0, 0)];
                println!(
                    "  Transition: {} -> {}",
                    sorted_points[i].dataset,
                    sorted_points[i + 1].dataset
                );
                println!(
                    "  g: {:.0} -> {:.0}",
                    manifold_points[i].g[0],
                    manifold_points[i + 1].g[0]
                );
                println!("  J: {:.4}\n", j_val);
                cross_jacobians.push(j_val);
            }
        }
    }

    if !cross_jacobians.is_empty() {
        let mean_j = cross_jacobians.iter().sum::<f64>() / cross_jacobians.len() as f64;
        let variance = cross_jacobians
            .iter()
            .map(|j| (j - mean_j).powi(2))
            .sum::<f64>()
            / cross_jacobians.len() as f64;

        println!("--- Universal Residualized Invariant ---");
        println!("Mean Jacobian <J>: {:.4}", mean_j);
        println!("Variance: {:.4}", variance);

        if variance > 1.0 {
            println!(
                "!! INSIGHT: High variance indicates that removing instrument bias shatters the 'universal' scaling."
            );
            println!(
                "The previous tight variance may have been an artifact of shared systemic measurement noise (the confounder) rather than deep physical universality."
            );
        } else {
            println!(
                "!! INSIGHT: Tight variance persists even after residualizing instrument bias."
            );
            println!(
                "This strengthens the thesis that information-theoretic bounds are fundamentally geometric."
            );
        }
    } else {
        println!("Not enough dimensional separation to compute smooth Jacobians.");
    }

    Ok(())
}

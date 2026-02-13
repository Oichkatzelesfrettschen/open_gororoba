//! TDA crate API validation - toy 2D circle example
//!
//! Tests the tda crate's Vietoris-Rips persistent homology on a known topology:
//! A circle in R^2 should have:
//! - b_0 = 1 (one connected component)
//! - b_1 = 1 (one loop/hole)
//!
//! This validates the API before integrating with E-027 LBM velocity fields.

use nalgebra::DMatrix;
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("TDA API Validation: 2D Circle Point Cloud");
    println!("==========================================\n");

    // Generate 20 points on unit circle
    let n = 20;
    let mut points = Vec::with_capacity(n * 2);
    for i in 0..n {
        let theta = 2.0 * PI * (i as f64) / (n as f64);
        points.push(theta.cos());
        points.push(theta.sin());
    }

    println!("Generated {} points on unit circle", n);

    // Create nalgebra DMatrix (n x 2)
    let point_matrix = DMatrix::from_row_slice(n, 2, &points);
    println!(
        "Point matrix shape: {}x{}\n",
        point_matrix.nrows(),
        point_matrix.ncols()
    );

    // Compute Euclidean distance matrix
    println!("Computing distance matrix...");
    let dist = tda::euclidean_distance_matrix(&point_matrix)?;
    println!("Distance matrix shape: {}x{}\n", dist.nrows(), dist.ncols());

    // Compute persistent homology up to dimension 1
    println!("Computing persistent homology (max_dim=1)...");
    let threshold = 3.0; // Large enough to capture full circle topology
    let max_dim = 1;
    let pairs = tda::persistent_homology_rips(&dist, threshold, max_dim)?;

    println!("Found {} persistence pairs:\n", pairs.len());

    // Count Betti numbers
    let mut betti_0 = 0;
    let mut betti_1 = 0;

    for (i, pair) in pairs.iter().enumerate() {
        println!(
            "  Pair {}: dim={}, birth={:.4}, death={:.4}, persistence={:.4}",
            i,
            pair.dimension,
            pair.birth,
            pair.death,
            pair.death - pair.birth
        );

        // Count long-lived features (persistence > 0.1)
        if pair.death - pair.birth > 0.1 {
            match pair.dimension {
                0 => betti_0 += 1,
                1 => betti_1 += 1,
                _ => {}
            }
        }
    }

    println!("\nBetti numbers (long-lived features, persistence > 0.1):");
    println!("  b_0 (connected components): {}", betti_0);
    println!("  b_1 (loops): {}", betti_1);

    // Validate expected topology
    println!("\nValidation:");
    if betti_0 == 1 && betti_1 == 1 {
        println!("  [PASS] Circle topology detected correctly (b_0=1, b_1=1)");
        Ok(())
    } else {
        println!(
            "  [FAIL] Expected b_0=1, b_1=1, got b_0={}, b_1={}",
            betti_0, betti_1
        );
        Err("Topology validation failed".into())
    }
}

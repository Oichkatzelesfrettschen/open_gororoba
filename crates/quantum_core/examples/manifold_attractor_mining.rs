//! Advanced Manifold Mining: Algebraic Attractors and Final Synthesis.
//!
//! This script adds:
//! 1. Attractor Ratio Scaling (c590): How the algebraic frustration ratio
//!    approaches the 3/8 limit across dimensions.
//!
//! This completes the Coupler-Manifold atlas for the current repo datasets.

use nalgebra::DVector;
use std::{
    fs::File,
    io::{BufRead, BufReader},
};
use verified_core::coupler_manifold::{CouplerJacobian, CouplerPoint};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Domain 6: Algebraic Attractor Scaling (c590_attractor_ratio_sweep.csv) ---");
    let path = "../../data/csv/c590_attractor_ratio_sweep.csv";
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut points = Vec::new();
    for line in reader.lines().skip(1) {
        let l = line?;
        let parts: Vec<&str> = l.split(',').collect();
        if parts.len() < 3 {
            continue;
        }

        if let (Ok(dim), Ok(ratio)) = (parts[1].parse::<f64>(), parts[2].parse::<f64>()) {
            // Observable O is the delta to the theoretical limit (0.375)
            // We want to see how this 'gap' closes.
            let gap = (0.375 - ratio).abs();
            points.push(CouplerPoint {
                g: DVector::from_vec(vec![dim]),
                o: DVector::from_vec(vec![gap + 1e-12]),
            });
            println!("  Dim (g): {:.0} | Frustration Gap (O): {:.4e}", dim, gap);
        }
    }

    if points.len() >= 2 {
        println!("\nAttractor Convergence Jacobians:");
        for i in 0..(points.len() - 1) {
            if let Ok(jac) = CouplerJacobian::estimate_from_delta(&points[i], &points[i + 1]) {
                println!(
                    "    g range [{:.0}, {:.0}]: J = {:.4}",
                    points[i].g[0],
                    points[i + 1].g[0],
                    jac.j_mat[(0, 0)]
                );
            }
        }
    }

    Ok(())
}

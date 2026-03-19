//! Real-world "Lattice Mining" for the Coupler-Manifold.
//!
//! This example streams the 46MB `topological_voids.csv`, 
//! maps LBM fluid density (g) to algebraic imbalance syndromes (O),
//! and extracts the local Jacobian from the real dataset.

use nalgebra::DVector;
use std::fs::File;
use std::io::{BufRead, BufReader};
use quantum_core::coupler_manifold::{CouplerPoint, CouplerJacobian};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "../../data/topological_voids.csv";
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    // We'll perform a sweep over the density threshold to define "syndromes"
    let rho_thresholds = vec![0.05, 0.1, 0.2, 0.4, 0.6, 0.8];
    let mut syndrome_counts = Vec::new();

    println!("Scanning {} for coupler manifold mining...", path);

    for &thr in &rho_thresholds {
        let mut count = 0;
        let mut total = 0;
        
        // Re-read file for each threshold (not the most efficient, but simple)
        let f = File::open(path)?;
        let r = BufReader::new(f);
        
        for line in r.lines().skip(1) {
            let l = line?;
            let parts: Vec<&str> = l.split(',').collect();
            if parts.len() < 5 { continue; }
            
            let rho: f64 = parts[3].parse()?;
            let imbalance: f64 = parts[4].parse()?;
            
            // Define a syndrome: Density is BELOW threshold AND imbalance is high
            if rho < thr && imbalance > 0.4 {
                count += 1;
            }
            total += 1;
        }
        
        println!("Threshold g={:.2}: Syndrome Count O={}", thr, count);
        syndrome_counts.push(count as f64);
    }

    // Now compute Jacobians between consecutive points
    println!("\n--- Jacobian Extraction (Elasticity J = d ln O / d ln g) ---");
    
    for i in 0..(rho_thresholds.len() - 1) {
        let p1 = CouplerPoint {
            g: DVector::from_vec(vec![rho_thresholds[i]]),
            o: DVector::from_vec(vec![syndrome_counts[i] + 1e-6]),
        };
        let p2 = CouplerPoint {
            g: DVector::from_vec(vec![rho_thresholds[i+1]]),
            o: DVector::from_vec(vec![syndrome_counts[i+1] + 1e-6]),
        };
        
        if let Ok(jac) = CouplerJacobian::estimate_from_delta(&p1, &p2) {
            println!("g range [{:.2}, {:.2}]: J = {:.2}", 
                     rho_thresholds[i], rho_thresholds[i+1], jac.j_mat[(0,0)]);
        }
    }

    Ok(())
}

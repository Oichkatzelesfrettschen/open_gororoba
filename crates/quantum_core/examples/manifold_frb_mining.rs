//! Manifold Mining for Real-World FRB Cascade Data.
//!
//! This example streams `c071d_frb_cascades_ultrametric.csv`,
//! separating repeating Fast Radio Bursts into two populations:
//! 1. "Smooth" continuous scaling (Hurst exponent near 0.5)
//! 2. "Burst/Cascade" highly clustered scaling (Hurst > 0.7)
//! 
//! We then construct a Coupler Jacobian to measure how the 
//! ultrametric fraction (O) scales against the Hurst exponent (g).

use nalgebra::DVector;
use std::fs::File;
use std::io::{BufRead, BufReader};
use verified_core::coupler_manifold::{CouplerPoint, CouplerJacobian};

#[derive(Debug)]
struct FrbSource {
    id: String,
    hurst: f64,
    ultrametric_frac: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "../../data/csv/c071d_frb_cascades_ultrametric.csv";
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut sources = Vec::new();

    // Parse the CSV, skip header
    for line in reader.lines().skip(1) {
        let l = line?;
        let parts: Vec<&str> = l.split(',').collect();
        if parts.len() < 10 { continue; }

        let id = parts[0].to_string();
        let hurst: f64 = parts[6].parse().unwrap_or(0.5);
        let ultrametric_frac: f64 = parts[7].parse().unwrap_or(0.0);

        sources.push(FrbSource { id, hurst, ultrametric_frac });
    }

    println!("Loaded {} FRB repeater sources.", sources.len());

    // Sort by Hurst exponent to create a smooth control parameter axis g
    sources.sort_by(|a, b| a.hurst.partial_cmp(&b.hurst).unwrap());

    // Group into bins to compute a stable average CouplerPoint
    let bin_size = 4;
    let mut points = Vec::new();

    for chunk in sources.chunks(bin_size) {
        if chunk.len() < bin_size { break; } // skip incomplete tail
        
        let avg_hurst: f64 = chunk.iter().map(|s| s.hurst).sum::<f64>() / bin_size as f64;
        let avg_ultra: f64 = chunk.iter().map(|s| s.ultrametric_frac).sum::<f64>() / bin_size as f64;
        
        // Hurst parameter acts as our scaling control g
        // Ultrametricity fraction is our observable O
        points.push(CouplerPoint {
            g: DVector::from_vec(vec![avg_hurst + 1e-6]), // avoid exact zero
            o: DVector::from_vec(vec![avg_ultra + 1e-6]),
        });
    }

    println!("\n--- Coupler-Manifold Trajectory for FRB Cascades ---");
    for (i, p) in points.iter().enumerate() {
        println!("Bin {}: Hurst (g) = {:.3}, Ultrametricity (O) = {:.3}", 
                 i, p.g[0], p.o[0]);
    }

    println!("\n--- Jacobian Elasticity (J = d ln O / d ln g) ---");
    for i in 0..(points.len() - 1) {
        if let Ok(jac) = CouplerJacobian::estimate_from_delta(&points[i], &points[i+1]) {
            println!("Transition Bin {} -> {}: J = {:.3}", 
                     i, i+1, jac.j_mat[(0,0)]);
        }
    }

    // Two-Sector separation: Compare low-Hurst (Brownian) vs high-Hurst (Persistent Cascade)
    if points.len() >= 4 {
        let j_brownian = CouplerJacobian::estimate_from_delta(&points[0], &points[1]).unwrap();
        let j_persistent = CouplerJacobian::estimate_from_delta(&points[points.len()-2], &points[points.len()-1]).unwrap();
        
        println!("\n--- Sector Separation Analysis ---");
        println!("Brownian Sector (Low Hurst) Jacobian J: {:.3}", j_brownian.j_mat[(0,0)]);
        println!("Persistent Sector (High Hurst) Jacobian J: {:.3}", j_persistent.j_mat[(0,0)]);
        
        let delta_j = (j_brownian.j_mat[(0,0)] - j_persistent.j_mat[(0,0)]).abs();
        println!("Delta J: {:.3}", delta_j);
        if delta_j > 1.0 {
            println!("CONCLUSION: Strong multi-sector behavior. The hierarchy of time-cascades fundamentally shifts the scaling manifold.");
        } else {
            println!("CONCLUSION: Uniform scaling behavior across all temporal clustering regimes.");
        }
    }

    Ok(())
}

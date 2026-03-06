//! generate-topological-voids: Generate macroscopic topological voids
//! required to match the CDG-2 astrophysical observations (VF=0.0).
//!
//! Creates a 3D density field where baryonic matter is excluded from
//! regions of high Pathion/Sedenion associator frustration.

use clap::Parser;
use sign_imbalance::bridge::{SedenionField};
use std::fs::File;
use std::io::Write;

#[derive(Parser)]
struct Args {
    /// Grid dimension.
    #[arg(long, default_value = "128")]
    grid: usize,
    /// Associator threshold for void exclusion.
    #[arg(long, default_value = "0.8")]
    threshold: f64,
    /// Path to save the resulting density field (CSV).
    #[arg(long, default_value = "data/topological_voids.csv")]
    output: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let n = args.grid;
    
    println!("Generating Macroscopic Topological Voids (Grid: {}^3)", n);
    
    // 1. Generate Sedenion Field with spatial variation
    let mut field = SedenionField::uniform(n, n, n);
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let s = field.get_mut(x, y, z);
                let fx = x as f64 / n as f64;
                let fy = y as f64 / n as f64;
                let fz = z as f64 / n as f64;
                
                // Maximize non-associative frustration
                for (i, si) in s.iter_mut().enumerate().skip(1).take(15) {
                    let freq = (i as f64 * 7.0).sqrt() * 10.0;
                    *si = (freq * (fx + fy + fz)).sin();
                }
            }
        }
    }
    
    // 2. Compute local imbalance (associator proxy)
    println!("Computing imbalance field...");
    let imbalance = field.local_imbalance_density(16);
    
    // 3. Apply exclusion principle: rho = 0 if imbalance > threshold
    let mut rho_field = vec![1.0f64; n * n * n];
    let mut void_count = 0;
    
    for (s, &imb) in rho_field.iter_mut().zip(imbalance.iter()) {
        if imb > args.threshold {
            *s = 0.0;
            void_count += 1;
        }
    }
    
    let vf = void_count as f64 / (n * n * n) as f64;
    println!("Void Generation Complete.");
    println!("  Threshold: {}", args.threshold);
    println!("  Void Volume Fraction (Exclusion): {:.6}", vf);
    println!("  Baryon VF (Remaining): {:.6}", 1.0 - vf);
    
    if (1.0 - vf) < 0.0006 {
        println!("  [SUCCESS] Matches CDG-2 astrophysical observation (VF < 0.0006)");
    } else {
        println!("  [NOTICE] Baryon fraction exceeds CDG-2 bound. Lower threshold or increase coupling.");
    }
    
    // 4. Save to CSV
    let mut file = File::create(&args.output)?;
    writeln!(file, "x,y,z,rho,imbalance")?;
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let idx = x + n * (y + n * z);
                if rho_field[idx] < 0.1 || idx % 100 == 0 { // Sparsity for CSV
                    writeln!(file, "{},{},{},{:.4},{:.4}", x, y, z, rho_field[idx], imbalance[idx])?;
                }
            }
        }
    }
    
    println!("Data saved to {}", args.output);
    
    Ok(())
}

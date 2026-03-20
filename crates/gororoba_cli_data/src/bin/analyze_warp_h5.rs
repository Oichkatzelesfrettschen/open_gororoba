//! HDF5 Warp Simulation Analysis Utility.
//!
//! Inspects HDF5 structure, trace data stability (NaN/Inf check),
//! and mass drift metrics for warp field simulations.
//!
//! Migrated from bin/analyze_warp_h5.py.

use anyhow::{Result, Context};
use hdf5::File;
use ndarray::Array1;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        println!("Usage: analyze-warp-h5 <path_to_h5>");
        return Ok(());
    }

    let file_path = &args[0];
    let file = File::open(file_path).with_context(|| format!("Failed to open {}", file_path))?;

    println!("=== Analysis of {} ===", file_path);

    // 1. Structural inspection (Simplified: check for key groups)
    if let Ok(sim_group) = file.group("simulation")
        && let Ok(trace_group) = sim_group.group("trace") {
            println!("\nTime-Series Summary:");
            
            if let Ok(time_ds) = trace_group.dataset("time") {
                let time: Array1<f64> = time_ds.read_1d()?;
                println!("  Steps recorded: {}", time.len());
                
                if let Ok(rho_ds) = trace_group.dataset("rho_mean") {
                    let rho: Array1<f64> = rho_ds.read_1d()?;
                    if !rho.is_empty() {
                        let initial = rho[0];
                        let final_rho = rho[rho.len() - 1];
                        println!("  Initial Mean Density: {:.6}", initial);
                        println!("  Final Mean Density: {:.6}", final_rho);
                        
                        let mut nan_found = false;
                        for &v in &rho {
                            if !v.is_finite() { nan_found = true; break; }
                        }
                        if nan_found {
                            println!("  CRITICAL: Instability (NaN/Inf) detected in density trace!");
                        } else if (final_rho - 1.0).abs() > 0.1 {
                            println!("  WARNING: High mass drift: {:+.4}", final_rho - 1.0);
                        }
                    }
                }

                if let Ok(ens_ds) = trace_group.dataset("enstrophy") {
                    let ens: Array1<f64> = ens_ds.read_1d()?;
                    if !ens.is_empty() {
                        println!("  Final Enstrophy: {:.6e}", ens[ens.len() - 1]);
                    }
                }
            }
        }

    Ok(())
}

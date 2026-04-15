//! HDF5 Field Analysis Utility.
//!
//! Computes basic statistics (NaN count, grid resolution, kinetic energy)
//! for 3D velocity field snapshots.
//!
//! Migrated from bin/analyze_field_h5.py.

use anyhow::{Context, Result};
use hdf5::File;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        println!("Usage: analyze-field-h5 <path_to_h5>");
        return Ok(());
    }

    let file_path = &args[0];
    let file = File::open(file_path).with_context(|| format!("Failed to open {}", file_path))?;

    println!("=== Field Snapshot Analysis: {} ===", file_path);

    if let Ok(dataset) = file.dataset("velocity") {
        let u_flat: Vec<f64> = dataset.read_1d::<f64>()?.into_raw_vec_and_offset().0;
        println!("  Raw data size: {}", u_flat.len());

        let mut nan_count = 0;
        let mut sum_sq: f64 = 0.0;
        let mut max_sq: f64 = 0.0;

        for &val in &u_flat {
            if val.is_nan() {
                nan_count += 1;
            } else {
                let sq = val * val;
                sum_sq += sq;
                if sq > max_sq {
                    max_sq = sq;
                }
            }
        }

        if nan_count > 0 {
            println!("  CRITICAL: Found {} NaNs in velocity field!", nan_count);
        }

        let n_total = u_flat.len() / 3;
        let res = (n_total as f64).powf(1.0 / 3.0).round() as usize;

        if res * res * res == n_total {
            println!("  Detected Grid: {}^3", res);
            println!("  Max Velocity: {:.6e}", max_sq.sqrt());
            println!(
                "  Mean Velocity: {:.6e}",
                (sum_sq / u_flat.len() as f64).sqrt()
            );
            println!("  Total Kinetic Energy: {:.6e}", 0.5 * sum_sq);
        } else {
            println!("  Unknown grid layout. Total elements: {}", u_flat.len());
        }
    } else {
        println!("  Dataset 'velocity' not found.");
    }

    Ok(())
}

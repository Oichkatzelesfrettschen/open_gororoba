//! Benchmark for GPU-accelerated Algebraic Lensing.
//! Shoots millions of rays through a Sedenion Black Hole.

use optics_core::{AlgebraicLensingGpu, GpuVec3};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    #[cfg(not(feature = "gpu"))]
    {
        println!("GPU feature not enabled. Run with --features gpu");
        return Ok(());
    }

    #[cfg(feature = "gpu")]
    {
        println!("Initializing Algebraic Lensing GPU (Ada Lovelace Optimized)...");
        let tracer = AlgebraicLensingGpu::new()?;

        // 1. Setup a "Sedenion Black Hole" field (64x64x64)
        let nx = 64;
        let ny = 64;
        let nz = 64;
        let mut density = vec![0.375f32; nx * ny * nz]; // Vacuum background

        // Create a high-frustration core (the "Black Hole")
        let cx = nx as f32 / 2.0;
        let cy = ny as f32 / 2.0;
        let cz = nz as f32 / 2.0;
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let dist = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2) + (z as f32 - cz).powi(2)).sqrt();
                    if dist < 10.0 {
                        let idx = z * nx * ny + y * nx + x;
                        density[idx] = 0.8; // High frustration core
                    }
                }
            }
        }

        // 2. Prepare Rays (High Resolution: 1024x1024)
        let res_x = 1024;
        let res_y = 1024;
        let mut initial_pos = Vec::with_capacity(res_x * res_y);
        let mut initial_dir = Vec::with_capacity(res_x * res_y);

        for y in 0..res_y {
            for x in 0..res_x {
                // Ray starting on a plane behind the BH, looking through it
                initial_pos.push(GpuVec3 { 
                    x: (x as f32 / res_x as f32) * nx as f32, 
                    y: (y as f32 / res_y as f32) * ny as f32, 
                    z: 0.0 
                });
                initial_dir.push(GpuVec3 { x: 0.0, y: 0.0, z: 1.0 });
            }
        }

        println!("Tracing {} million rays...", (initial_pos.len() as f32 / 1e6));
        
        let alpha = 10.0; // Strong lensing
        let dt = 0.1;
        let max_steps = 1000;

        let start = Instant::now();
        let (final_pos, _final_dir) = tracer.trace_rays(
            &density, nx, ny, nz,
            &initial_pos, &initial_dir,
            alpha, dt, max_steps
        )?;
        let duration = start.elapsed();

        println!("Done! Time: {:?}, Rays/sec: {:.2} million", 
            duration, 
            (initial_pos.len() as f64 / 1e6) / duration.as_secs_f64()
        );

        // 3. Crude ASCII Visualization of the "Einstein Ring"
        println!("\nProjection of Ray Deflection (Algebraic Shadow):");
        let mut grid = vec![0; 80 * 40];
        let mut max_deflection = 0.0f32;
        
        for (i, p) in final_pos.iter().enumerate() {
            let start = initial_pos[i];
            let dx = p.x - start.x;
            let dy = p.y - start.y;
            let def = (dx*dx + dy*dy).sqrt();
            if def > max_deflection { max_deflection = def; }

            let gx = ((p.x / nx as f32) * 79.0) as usize;
            let gy = ((p.y / ny as f32) * 39.0) as usize;
            if gx < 80 && gy < 40 {
                grid[gy * 80 + gx] += 1;
            }
        }

        println!("Max Ray Deflection: {:.4} units", max_deflection);

        let max_count = *grid.iter().max().unwrap_or(&1) as f32;
        for y in 0..40 {
            for x in 0..80 {
                let val = grid[y * 80 + x] as f32 / max_count;
                let char = if val < 0.1 { ' ' } 
                          else if val < 0.3 { '.' } 
                          else if val < 0.6 { 'o' } 
                          else if val < 0.9 { '0' }
                          else { '#' };
                print!("{}", char);
            }
            println!();
        }
    }

    Ok(())
}

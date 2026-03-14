//! GPU scaling benchmarks for D3Q19 LBM solver tailored for Ada Lovelace (SM 8.9).
//!
//! Comprehensive scaling study: 8^3, 16^3, 32^3, 64^3, 128^3, 256^3, 512^3, 1024^3.
//! Sweeps across precisions: FP32, BF16, FP64.

use lbm_3d_cuda::{LbmSolver3DCuda, Precision, probe_cuda_available};
use std::time::Instant;

/// Benchmark configuration
#[derive(Clone)]
struct BenchConfig {
    grid_size: usize,
    steps: usize,
    precision: Precision,
    tau: f64,
    rho_init: f64,
    u_init: [f64; 3],
}

impl BenchConfig {
    fn name(&self) -> String {
        let prec_str = match self.precision {
            Precision::FP32 => "FP32",
            Precision::BF16 => "BF16",
            Precision::FP64 => "FP64",
        };
        format!("{}^3 x {} steps [{}]", self.grid_size, self.steps, prec_str)
    }

    fn memory_gb(&self) -> f64 {
        let n_cells = self.grid_size.pow(3) as f64;
        let bytes_per_float = match self.precision {
            Precision::FP32 => 4.0,
            Precision::BF16 => 2.0,
            Precision::FP64 => 8.0,
        };
        // f: 19 * n_cells
        // rho: n_cells
        // u: 3 * n_cells
        // tau: n_cells
        let total_bytes = n_cells * (19.0 + 1.0 + 3.0 + 1.0) * bytes_per_float;
        // Times 2 for f_tmp (ping-pong buffers in standard solver)
        let total_vram = total_bytes + (n_cells * 19.0 * bytes_per_float);
        total_vram / (1024.0 * 1024.0 * 1024.0)
    }

    fn throughput_mcells_per_sec(&self, elapsed_secs: f64) -> f64 {
        let n_cells = self.grid_size.pow(3) as f64;
        (n_cells * self.steps as f64) / elapsed_secs / 1e6
    }
}

/// Run GPU benchmark
fn bench_gpu(config: &BenchConfig) -> Result<f64, String> {
    // Check if VRAM requirement is too large (e.g., > 20GB for a typical Ada card)
    if config.memory_gb() > 20.0 {
        return Err(format!("Skipping due to excessive VRAM requirement: {:.2} GB", config.memory_gb()));
    }

    let mut solver = LbmSolver3DCuda::new(
        config.grid_size,
        config.grid_size,
        config.grid_size,
        config.tau,
        config.precision,
    )
    .map_err(|e| format!("GPU initialization failed: {:?}", e))?;

    solver
        .initialize_uniform(
            config.rho_init as f32,
            [
                config.u_init[0] as f32,
                config.u_init[1] as f32,
                config.u_init[2] as f32,
            ],
        )
        .map_err(|e| format!("GPU initialization failed: {:?}", e))?;

    // Warmup
    solver.evolve(10).unwrap_or_default();

    let start = Instant::now();
    solver
        .evolve(config.steps)
        .map_err(|e| format!("GPU evolution failed: {:?}", e))?;
    
    let elapsed = start.elapsed();
    Ok(elapsed.as_secs_f64())
}

fn main() {
    println!("\n{}", "#".repeat(80));
    println!("# ADA LOVELACE LBM SCALING BENCHMARK (8^3 to 1024^3)");
    println!("# Profiling memory budget, datatypes, and accuracy");
    println!("{}", "#".repeat(80));

    if !probe_cuda_available() {
        println!("CUDA not detected. Exiting benchmark.");
        return;
    }

    let grid_sizes = vec![8, 16, 32, 64, 128, 256, 512, 1024];
    let precisions = vec![Precision::BF16, Precision::FP32, Precision::FP64];
    
    println!(
        "{:<25} {:>10} {:>10} {:>12} {:>15}",
        "Benchmark", "VRAM (GB)", "Steps", "Time (s)", "Throughput (MLUPS)"
    );
    println!("{}", "-".repeat(80));

    for &grid_size in &grid_sizes {
        // Reduce steps for large grids to keep runtimes reasonable
        let steps = if grid_size >= 512 {
            50
        } else if grid_size >= 256 {
            500
        } else if grid_size >= 128 {
            2000
        } else {
            10000
        };

        for &precision in &precisions {
            let config = BenchConfig {
                grid_size,
                steps,
                precision,
                tau: 1.0,
                rho_init: 1.0,
                u_init: [0.01, 0.0, 0.0],
            };

            let name = config.name();
            let vram = config.memory_gb();
            
            if vram > 20.0 {
                println!(
                    "{:<25} {:>10.2} {:>10} {:>12} {:>15}",
                    name, vram, steps, "OOM", "-"
                );
                continue;
            }

            match bench_gpu(&config) {
                Ok(time) => {
                    let throughput = config.throughput_mcells_per_sec(time);
                    println!(
                        "{:<25} {:>10.2} {:>10} {:>12.4} {:>15.2}",
                        name, vram, steps, time, throughput
                    );
                }
                Err(e) => {
                    println!(
                        "{:<25} {:>10.2} {:>10} {:>12} {:>15}",
                        name, vram, steps, "ERR", "-"
                    );
                    eprintln!("  Error: {}", e);
                }
            }
        }
    }
    println!("{}", "#".repeat(80));
}

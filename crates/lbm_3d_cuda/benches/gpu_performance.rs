//! GPU performance benchmarks for D3Q19 LBM solver.
//!
//! Comprehensive scaling study: 32^3 to 256^3, 2500 to 10000 steps.
//! Includes ALL overheads: memory transfer, kernel compilation, synchronization.
//!
//! Target hardware: NVIDIA RTX 4070 Ti (7680 CUDA cores, 504 GB/s bandwidth)
//! vs CPU: 12 cores (parallel LBM)

use lbm_3d::solver::LbmSolver3D;
use lbm_3d_cuda::LbmSolver3DCuda;
use std::time::Instant;

/// Benchmark configuration
#[derive(Clone)]
struct BenchConfig {
    name: String,
    grid_size: usize,
    steps: usize,
    tau: f64,
    rho_init: f64,
    u_init: [f64; 3],
}

impl BenchConfig {
    fn memory_mb(&self) -> f64 {
        let n_cells = self.grid_size.pow(3);
        // f: 19 * n_cells * 8 bytes
        // rho: n_cells * 8 bytes
        // u: 3 * n_cells * 8 bytes
        // tau: n_cells * 8 bytes
        let total_bytes = n_cells * (19 + 1 + 3 + 1) * 8;
        total_bytes as f64 / (1024.0 * 1024.0)
    }

    fn throughput_mcells_per_sec(&self, elapsed_secs: f64) -> f64 {
        let n_cells = self.grid_size.pow(3) as f64;
        (n_cells * self.steps as f64) / elapsed_secs / 1e6
    }
}

/// Benchmark result
struct BenchResult {
    config: BenchConfig,
    cpu_time: f64,
    gpu_time: f64,
    speedup: f64,
    cpu_throughput: f64,
    gpu_throughput: f64,
}

impl BenchResult {
    fn new(config: BenchConfig, cpu_time: f64, gpu_time: f64) -> Self {
        let speedup = cpu_time / gpu_time;
        let cpu_throughput = config.throughput_mcells_per_sec(cpu_time);
        let gpu_throughput = config.throughput_mcells_per_sec(gpu_time);
        BenchResult {
            config,
            cpu_time,
            gpu_time,
            speedup,
            cpu_throughput,
            gpu_throughput,
        }
    }

    fn print(&self) {
        println!("\n{}", "=".repeat(80));
        println!("Benchmark: {}", self.config.name);
        println!("{}", "=".repeat(80));
        println!("Grid size:       {}^3 = {} cells", self.config.grid_size, self.config.grid_size.pow(3));
        println!("Steps:           {}", self.config.steps);
        println!("Memory (GPU):    {:.2} MB", self.config.memory_mb());
        println!();
        println!("CPU time:        {:.3} s", self.cpu_time);
        println!("GPU time:        {:.3} s", self.gpu_time);
        println!("Speedup:         {:.2}x", self.speedup);
        println!();
        println!("CPU throughput:  {:.2} Mcells/s", self.cpu_throughput);
        println!("GPU throughput:  {:.2} Mcells/s", self.gpu_throughput);
        println!("Throughput gain: {:.2}x", self.gpu_throughput / self.cpu_throughput);
        println!("{}", "=".repeat(80));
    }
}

/// Run CPU benchmark
fn bench_cpu(config: &BenchConfig) -> Result<f64, String> {
    let mut solver = LbmSolver3D::new(config.grid_size, config.grid_size, config.grid_size, config.tau);
    solver.initialize_uniform(config.rho_init, config.u_init);

    let start = Instant::now();
    solver.evolve(config.steps);
    let elapsed = start.elapsed();

    Ok(elapsed.as_secs_f64())
}

/// Run GPU benchmark (includes ALL overheads)
fn bench_gpu(config: &BenchConfig) -> Result<f64, String> {
    let start_total = Instant::now();

    let mut solver = LbmSolver3DCuda::new(
        config.grid_size,
        config.grid_size,
        config.grid_size,
        config.tau,
    ).map_err(|e| format!("GPU initialization failed: {:?}", e))?;

    solver.initialize_uniform(config.rho_init, config.u_init)
        .map_err(|e| format!("GPU initialization failed: {:?}", e))?;

    solver.evolve(config.steps)
        .map_err(|e| format!("GPU evolution failed: {:?}", e))?;

    let elapsed = start_total.elapsed();
    Ok(elapsed.as_secs_f64())
}

/// Run single benchmark (CPU vs GPU)
fn run_single_benchmark(config: &BenchConfig) -> Option<BenchResult> {
    println!("\nRunning: {}", config.name);

    println!("  CPU benchmark...");
    let cpu_time = match bench_cpu(config) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("  CPU benchmark failed: {}", e);
            return None;
        }
    };

    println!("  GPU benchmark...");
    let gpu_time = match bench_gpu(config) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("  GPU not available or benchmark failed: {}", e);
            return None;
        }
    };

    Some(BenchResult::new(config.clone(), cpu_time, gpu_time))
}

/// Run multiple GPU-only benchmarks and average (for 256^3 production validation)
fn run_gpu_only_multi(config: &BenchConfig, n_runs: usize) -> Option<f64> {
    println!("\nRunning GPU-only: {} ({} runs for averaging)", config.name, n_runs);

    let mut times = Vec::new();
    for i in 1..=n_runs {
        print!("  Run {}/{}...", i, n_runs);
        let gpu_time = match bench_gpu(config) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("\n  GPU benchmark failed: {}", e);
                return None;
            }
        };
        println!(" {:.3} s", gpu_time);
        times.push(gpu_time);
    }

    let avg_time = times.iter().sum::<f64>() / times.len() as f64;
    let stddev = (times.iter().map(|t| (t - avg_time).powi(2)).sum::<f64>() / times.len() as f64).sqrt();

    println!("\n  Average GPU time: {:.3} s +/- {:.3} s", avg_time, stddev);
    println!("  Throughput:       {:.2} Mcells/s", config.throughput_mcells_per_sec(avg_time));

    Some(avg_time)
}

/// Print comprehensive summary table
fn print_summary(results: &[BenchResult]) {
    if results.is_empty() {
        return;
    }

    println!("\n{}", "#".repeat(80));
    println!("# COMPREHENSIVE SUMMARY TABLE");
    println!("{}", "#".repeat(80));
    println!();
    println!("{:<20} {:>10} {:>12} {:>12} {:>10} {:>12}",
             "Benchmark", "Grid", "Steps", "GPU (s)", "Speedup", "Throughput");
    println!("{}", "-".repeat(80));

    for result in results {
        println!("{:<20} {:>10} {:>12} {:>12.3} {:>9.2}x {:>10.2} M",
                 result.config.name,
                 format!("{}^3", result.config.grid_size),
                 result.config.steps,
                 result.gpu_time,
                 result.speedup,
                 result.gpu_throughput);
    }

    println!("{}", "-".repeat(80));

    let avg_speedup: f64 = results.iter().map(|r| r.speedup).sum::<f64>() / results.len() as f64;
    let max_speedup = results.iter().map(|r| r.speedup).fold(f64::NEG_INFINITY, f64::max);
    let min_speedup = results.iter().map(|r| r.speedup).fold(f64::INFINITY, f64::min);

    println!("Average speedup: {:.2}x", avg_speedup);
    println!("Speedup range:   {:.2}x - {:.2}x", min_speedup, max_speedup);
    println!("{}", "#".repeat(80));
}

/// Generate comprehensive benchmark configurations
fn generate_configs() -> Vec<BenchConfig> {
    let grid_sizes = vec![32, 64, 128, 256];
    let step_counts = vec![2500, 5000, 7500, 10000];

    let mut configs = Vec::new();

    for &grid_size in &grid_sizes {
        for &steps in &step_counts {
            configs.push(BenchConfig {
                name: format!("{}^3 x {} steps", grid_size, steps),
                grid_size,
                steps,
                tau: 1.0,
                rho_init: 1.0,
                u_init: [0.01, 0.0, 0.0],
            });
        }
    }

    configs
}

fn main() {
    println!("\n{}", "#".repeat(80));
    println!("# GPU PERFORMANCE BENCHMARK SUITE - COMPREHENSIVE SCALING STUDY");
    println!("# Hardware: NVIDIA RTX 4070 Ti (7680 CUDA cores, 504 GB/s)");
    println!("# CPU: 12 cores (parallel LBM)");
    println!("# Scaling: 32^3 to 256^3, 2500 to 10000 steps");
    println!("{}", "#".repeat(80));

    let configs = generate_configs();
    let mut results = Vec::new();

    // Run all benchmarks except 256^3 x 10K (reserved for 5x averaging)
    for config in &configs {
        // Skip CPU for 256^3 (too slow) and the 256^3 x 10K special case
        if config.grid_size == 256 {
            if config.steps == 10000 {
                // Special case: 5x averaging for production validation
                println!("\n{}", "=".repeat(80));
                println!("PRODUCTION VALIDATION: 256^3 x 10000 steps (5x averaged)");
                println!("{}", "=".repeat(80));

                if let Some(avg_gpu_time) = run_gpu_only_multi(config, 5) {
                    println!("\n>>> Average GPU time: {:.3} s", avg_gpu_time);

                    if avg_gpu_time < 25.0 {
                        println!(">>> FAST ENOUGH FOR TESTING! (<25s threshold)");
                        println!(">>> Recommend: Add as regression test in test_gpu_cpu_equivalence.rs");
                    } else {
                        println!(">>> TOO SLOW FOR TESTING (>=25s threshold)");
                        println!(">>> Recommend: Keep for production runs only");
                    }
                }
            } else {
                // GPU-only for other 256^3 configs (CPU too slow)
                println!("\nRunning GPU-only: {} (CPU too slow at this scale)", config.name);
                if let Some(gpu_time) = run_gpu_only_multi(config, 1) {
                    println!("  GPU time: {:.3} s", gpu_time);
                    println!("  Throughput: {:.2} Mcells/s", config.throughput_mcells_per_sec(gpu_time));
                }
            }
        } else {
            // CPU vs GPU comparison for grids up to 128^3
            if let Some(result) = run_single_benchmark(config) {
                result.print();
                results.push(result);
            }
        }
    }

    print_summary(&results);

    println!("\n{}", "#".repeat(80));
    println!("# BENCHMARK SUITE COMPLETE");
    println!("{}", "#".repeat(80));
}

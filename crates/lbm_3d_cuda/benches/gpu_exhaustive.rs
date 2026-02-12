//! GPU-only exhaustive performance benchmarks for D3Q19 LBM solver.
//!
//! Tests large grids (128^3, 256^3, 384^3) with extensive step counts
//! where CPU is impractical. Includes memory scaling and convergence analysis.
//!
//! Target hardware: NVIDIA RTX 4070 Ti (7680 CUDA cores, 504 GB/s bandwidth)

use lbm_3d_cuda::LbmSolver3DCuda;
use std::time::Instant;

/// GPU benchmark configuration
#[derive(Clone)]
struct GpuBenchConfig {
    name: String,
    grid_size: usize,
    steps: usize,
    tau: f64,
    rho_init: f64,
    u_init: [f64; 3],
}

impl GpuBenchConfig {
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

/// Run GPU-only benchmark
fn bench_gpu(config: &GpuBenchConfig) -> Result<f64, String> {
    let start_total = Instant::now();

    let mut solver = LbmSolver3DCuda::new(
        config.grid_size,
        config.grid_size,
        config.grid_size,
        config.tau,
    )
    .map_err(|e| format!("GPU initialization failed: {:?}", e))?;

    solver
        .initialize_uniform(config.rho_init, config.u_init)
        .map_err(|e| format!("GPU initialization failed: {:?}", e))?;

    solver
        .evolve(config.steps)
        .map_err(|e| format!("GPU evolution failed: {:?}", e))?;

    let elapsed = start_total.elapsed();
    Ok(elapsed.as_secs_f64())
}

/// Run multiple benchmarks and compute statistics
fn bench_gpu_multi(config: &GpuBenchConfig, n_runs: usize) -> Option<(f64, f64, f64, f64)> {
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

    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let variance = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64;
    let stddev = variance.sqrt();
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    Some((mean, stddev, min, max))
}

/// Print result for single run
fn print_result_single(config: &GpuBenchConfig, time_s: f64) {
    println!("\n{}", "=".repeat(80));
    println!("Benchmark: {}", config.name);
    println!("{}", "=".repeat(80));
    println!(
        "Grid size:       {}^3 = {} cells",
        config.grid_size,
        config.grid_size.pow(3)
    );
    println!("Steps:           {}", config.steps);
    println!("Memory (GPU):    {:.2} MB", config.memory_mb());
    println!();
    println!("GPU time:        {:.3} s", time_s);
    println!(
        "Throughput:      {:.2} Mcells/s",
        config.throughput_mcells_per_sec(time_s)
    );
    println!("{}", "=".repeat(80));
}

/// Print result for multiple runs with statistics
fn print_result_multi(config: &GpuBenchConfig, mean: f64, stddev: f64, min: f64, max: f64) {
    println!("\n{}", "=".repeat(80));
    println!("Benchmark: {} (averaged)", config.name);
    println!("{}", "=".repeat(80));
    println!(
        "Grid size:       {}^3 = {} cells",
        config.grid_size,
        config.grid_size.pow(3)
    );
    println!("Steps:           {}", config.steps);
    println!("Memory (GPU):    {:.2} MB", config.memory_mb());
    println!();
    println!("Mean GPU time:   {:.3} +/- {:.3} s", mean, stddev);
    println!("Range:           {:.3} - {:.3} s", min, max);
    println!(
        "Throughput:      {:.2} Mcells/s",
        config.throughput_mcells_per_sec(mean)
    );
    println!();

    // 25-second test threshold check
    if mean < 25.0 {
        println!(">>> FAST ENOUGH FOR TESTING! (<25s threshold)");
        println!(">>> Recommend: Add as regression test");
    } else {
        println!(">>> TOO SLOW FOR TESTING (>=25s threshold)");
        println!(">>> Recommend: Production runs only");
    }
    println!("{}", "=".repeat(80));
}

/// Generate exhaustive GPU-only configurations
fn generate_gpu_configs() -> Vec<(GpuBenchConfig, usize)> {
    let mut configs = Vec::new();

    // 128^3: Comprehensive step coverage (2500 to 20000 in 2500 increments)
    for &steps in &[2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000] {
        configs.push((
            GpuBenchConfig {
                name: format!("128^3 x {} steps", steps),
                grid_size: 128,
                steps,
                tau: 1.0,
                rho_init: 1.0,
                u_init: [0.01, 0.0, 0.0],
            },
            1, // Single run for most configs
        ));
    }

    // 256^3: Medium step coverage (2500 to 20000 in 2500 increments)
    for &steps in &[2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000] {
        let n_runs = if steps == 10000 { 5 } else { 1 }; // 5x averaging for 10K
        configs.push((
            GpuBenchConfig {
                name: format!("256^3 x {} steps", steps),
                grid_size: 256,
                steps,
                tau: 1.0,
                rho_init: 1.0,
                u_init: [0.01, 0.0, 0.0],
            },
            n_runs,
        ));
    }

    // 384^3: Conservative step coverage (memory limit test)
    for &steps in &[2500, 5000, 7500, 10000] {
        configs.push((
            GpuBenchConfig {
                name: format!("384^3 x {} steps", steps),
                grid_size: 384,
                steps,
                tau: 1.0,
                rho_init: 1.0,
                u_init: [0.01, 0.0, 0.0],
            },
            1,
        ));
    }

    configs
}

fn main() {
    println!("\n{}", "#".repeat(80));
    println!("# GPU-ONLY EXHAUSTIVE PERFORMANCE BENCHMARK SUITE");
    println!("# Hardware: NVIDIA RTX 4070 Ti (7680 CUDA cores, 504 GB/s)");
    println!("# Grids: 128^3 (8 configs), 256^3 (8 configs), 384^3 (4 configs)");
    println!("# Total: 20 configurations");
    println!("{}", "#".repeat(80));

    let configs = generate_gpu_configs();
    let mut all_results = Vec::new();

    for (config, n_runs) in &configs {
        println!("\nRunning: {} ({} runs)", config.name, n_runs);

        if *n_runs == 1 {
            // Single run
            match bench_gpu(config) {
                Ok(time_s) => {
                    print_result_single(config, time_s);
                    all_results.push((config.clone(), time_s, 0.0));
                }
                Err(e) => {
                    eprintln!("GPU benchmark failed: {}", e);
                }
            }
        } else {
            // Multiple runs with statistics
            if let Some((mean, stddev, min, max)) = bench_gpu_multi(config, *n_runs) {
                print_result_multi(config, mean, stddev, min, max);
                all_results.push((config.clone(), mean, stddev));
            }
        }
    }

    // Summary table
    if !all_results.is_empty() {
        println!("\n{}", "#".repeat(80));
        println!("# COMPREHENSIVE SUMMARY TABLE - GPU-ONLY");
        println!("{}", "#".repeat(80));
        println!();
        println!(
            "{:<25} {:>10} {:>12} {:>12} {:>12}",
            "Benchmark", "Grid", "Steps", "GPU (s)", "Throughput"
        );
        println!("{}", "-".repeat(80));

        for (config, mean, stddev) in &all_results {
            if *stddev > 0.0 {
                println!(
                    "{:<25} {:>10} {:>12} {:>9.3}+/-{:.3} {:>10.2} M",
                    config.name,
                    format!("{}^3", config.grid_size),
                    config.steps,
                    mean,
                    stddev,
                    config.throughput_mcells_per_sec(*mean)
                );
            } else {
                println!(
                    "{:<25} {:>10} {:>12} {:>12.3} {:>10.2} M",
                    config.name,
                    format!("{}^3", config.grid_size),
                    config.steps,
                    mean,
                    config.throughput_mcells_per_sec(*mean)
                );
            }
        }

        println!("{}", "-".repeat(80));
        println!("Total configurations tested: {}", all_results.len());
        println!("{}", "#".repeat(80));
    }

    println!("\n{}", "#".repeat(80));
    println!("# GPU-ONLY EXHAUSTIVE BENCHMARK COMPLETE");
    println!("{}", "#".repeat(80));
}

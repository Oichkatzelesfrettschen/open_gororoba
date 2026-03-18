//! CPU LBM Benchmark -- cross-backend parity baseline.
//!
//! Benchmarks the CPU D3Q19 solver (BGK and MRT collision modes) at FP64
//! precision across grid sizes 32^3, 64^3, and 128^3. Writes CSV output
//! with the unified schema (backend, workload, precision, variant, grid_size,
//! ..., mlups, bandwidth_gbs) compatible with CUDA and Vulkan benchmark CSVs.
//!
//! # Schema Alignment
//!
//! The CSV uses the same column layout as `cuda-precision-bench` to enable
//! direct comparison via `parity-report`. Key columns:
//! - `backend`: always "cpu"
//! - `workload`: always "lbm_sweep"
//! - `precision`: "FP64" (CPU solver is f64-native)
//! - `variant`: "standard" (BGK) or "mrt" (MRT)
//!
//! # Example
//!
//! ```bash
//! cargo run --release -p gororoba_cli_physics --bin cpu-lbm-bench \
//!     --no-default-features -- --grids 32,64,128 --workloads all
//! ```

use anyhow::Result;
use clap::Parser;
use lbm_3d::solver::LbmSolver3D;
use std::{io::Write as _, time::Instant};

// D3Q19 bandwidth model: 19 read + 19 write + 8 macroscopic = 46 scalars per cell.
const D3Q19_SCALARS: usize = 46;

#[derive(Parser, Debug)]
#[command(name = "cpu-lbm-bench", version)]
struct BenchConfig {
    /// Comma-separated grid side lengths.
    #[arg(long, default_value = "32,64,128")]
    grids: String,

    /// Warmup steps.
    #[arg(long, default_value_t = 10)]
    warmup: usize,

    /// Timing steps for 32/64 grids.
    #[arg(long, default_value_t = 100)]
    steps_small: usize,

    /// Timing steps for 128 grids.
    #[arg(long, default_value_t = 50)]
    steps_mid: usize,

    /// Timing steps for 256+ grids.
    #[arg(long, default_value_t = 20)]
    steps_large: usize,

    /// Workloads (comma-separated: bgk,mrt,all).
    #[arg(long, default_value = "all")]
    workloads: String,

    /// Output CSV path.
    #[arg(long, default_value = "data/benchmarks/cpu_lbm_baseline.csv")]
    output: String,
}

impl BenchConfig {
    fn grids(&self) -> Vec<usize> {
        self.grids
            .split(',')
            .filter_map(|p| p.trim().parse::<usize>().ok())
            .collect()
    }

    fn n_steps_for(&self, n: usize) -> usize {
        if n <= 64 {
            self.steps_small
        } else if n <= 128 {
            self.steps_mid
        } else {
            self.steps_large
        }
    }

    fn run_workload(&self, name: &str) -> bool {
        if self.workloads.contains("all") {
            return true;
        }
        self.workloads.split(',').any(|w| w.trim() == name)
    }
}

#[derive(Debug, Clone)]
struct BenchRow {
    backend: &'static str,
    workload: &'static str,
    precision: String,
    variant: String,
    grid_size: usize,
    param2: String,
    n_steps: usize,
    warmup: usize,
    elapsed_ms: f64,
    mlups: Option<f64>,
    bandwidth_gbs: Option<f64>,
    bandwidth_pct_peak: Option<f64>,
    vram_dist_mb: Option<usize>,
    notes: String,
}

fn mlups(n_cells: usize, steps: usize, elapsed_ms: f64) -> f64 {
    n_cells as f64 * steps as f64 / elapsed_ms / 1e3
}

fn bandwidth_gbs(n_cells: usize, steps: usize, elapsed_ms: f64) -> f64 {
    let bytes = D3Q19_SCALARS as f64 * n_cells as f64 * steps as f64 * 8.0; // f64
    bytes / (elapsed_ms * 1e-3) / 1e9
}

fn write_csv(path: &str, rows: &[BenchRow]) -> Result<()> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = std::fs::File::create(path)?;
    writeln!(
        f,
        "backend,workload,precision,variant,grid_size,param2,n_steps,warmup,\
         elapsed_ms,mlups,bandwidth_gbs,bandwidth_pct_peak,vram_dist_mb,notes"
    )?;
    for r in rows {
        writeln!(
            f,
            "{},{},{},{},{},{},{},{},{:.3},{},{},{},{},\"{}\"",
            r.backend,
            r.workload,
            r.precision,
            r.variant,
            r.grid_size,
            r.param2,
            r.n_steps,
            r.warmup,
            r.elapsed_ms,
            r.mlups.map_or(String::new(), |v| format!("{v:.2}")),
            r.bandwidth_gbs.map_or(String::new(), |v| format!("{v:.2}")),
            r.bandwidth_pct_peak
                .map_or(String::new(), |v| format!("{v:.1}")),
            r.vram_dist_mb.map_or(String::new(), |v| format!("{v}")),
            r.notes,
        )?;
    }
    Ok(())
}

fn print_table(rows: &[BenchRow]) {
    println!(
        "\n{:<24} {:<6} {:<10} {:>8} {:>8} {:>7}",
        "variant", "grid", "precision", "MLUPS", "BW_GBS", "ms"
    );
    println!("{:-<70}", "");
    for r in rows {
        println!(
            "{:<24} {:<6} {:<10} {:>8} {:>8} {:>7.1}",
            r.variant,
            r.grid_size,
            r.precision,
            r.mlups.map_or("--".to_string(), |v| format!("{v:.1}")),
            r.bandwidth_gbs
                .map_or("--".to_string(), |v| format!("{v:.1}")),
            r.elapsed_ms,
        );
    }
}

fn bench_bgk(cfg: &BenchConfig, rows: &mut Vec<BenchRow>) {
    if !cfg.run_workload("bgk") {
        return;
    }
    for &n in &cfg.grids() {
        let steps = cfg.n_steps_for(n);
        let n_cells = n * n * n;
        let tau = 0.7;

        let mut solver = LbmSolver3D::new(n, n, n, tau);
        // Warmup
        for _ in 0..cfg.warmup {
            solver.evolve_one_step();
        }

        let t0 = Instant::now();
        for _ in 0..steps {
            solver.evolve_one_step();
        }
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;

        let m = mlups(n_cells, steps, elapsed_ms);
        let bw = bandwidth_gbs(n_cells, steps, elapsed_ms);
        rows.push(BenchRow {
            backend: "cpu",
            workload: "lbm_sweep",
            precision: "FP64".to_string(),
            variant: "standard".to_string(),
            grid_size: n,
            param2: format!("tau={tau}"),
            n_steps: steps,
            warmup: cfg.warmup,
            elapsed_ms,
            mlups: Some(m),
            bandwidth_gbs: Some(bw),
            bandwidth_pct_peak: None,
            vram_dist_mb: Some(n_cells * 19 * 8 * 2 / (1024 * 1024)),
            notes: "CPU BGK f64".to_string(),
        });
    }
}

fn bench_mrt(cfg: &BenchConfig, rows: &mut Vec<BenchRow>) {
    if !cfg.run_workload("mrt") {
        return;
    }
    for &n in &cfg.grids() {
        let steps = cfg.n_steps_for(n);
        let n_cells = n * n * n;
        let tau = 0.7;

        let mut solver = LbmSolver3D::new_mrt(n, n, n, tau);
        for _ in 0..cfg.warmup {
            solver.evolve_one_step();
        }

        let t0 = Instant::now();
        for _ in 0..steps {
            solver.evolve_one_step();
        }
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;

        let m = mlups(n_cells, steps, elapsed_ms);
        let bw = bandwidth_gbs(n_cells, steps, elapsed_ms);
        rows.push(BenchRow {
            backend: "cpu",
            workload: "lbm_sweep",
            precision: "FP64".to_string(),
            variant: "mrt".to_string(),
            grid_size: n,
            param2: format!("tau={tau}"),
            n_steps: steps,
            warmup: cfg.warmup,
            elapsed_ms,
            mlups: Some(m),
            bandwidth_gbs: Some(bw),
            bandwidth_pct_peak: None,
            vram_dist_mb: Some(n_cells * 19 * 8 * 2 / (1024 * 1024)),
            notes: "CPU MRT f64".to_string(),
        });
    }
}

fn main() -> Result<()> {
    let cfg = BenchConfig::parse();
    println!("CPU LBM Benchmark");
    println!("Grids: {:?}", cfg.grids());
    println!("Workloads: {}", cfg.workloads);

    let mut rows = Vec::new();
    bench_bgk(&cfg, &mut rows);
    bench_mrt(&cfg, &mut rows);

    print_table(&rows);
    write_csv(&cfg.output, &rows)?;
    println!("\nCSV written to: {}", cfg.output);
    Ok(())
}

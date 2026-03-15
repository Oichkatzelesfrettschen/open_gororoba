//! CUDA Kernel Baseline Benchmark
//!
//! Sweeps the full kernel decision space across six workload groups,
//! computes MLUPS and effective bandwidth, and writes an ASCII summary
//! table (stdout) plus a CSV file for agent consumption.
//!
//! This is NOT a CI gate. It is a reference tool for design decisions.
//! See: plans/golden-splashing-boole.md for the full specification.

use anyhow::Result;
use clap::Parser;
use std::io::Write as _;

// RTX 4070 Ti theoretical peak memory bandwidth.
const PEAK_BW_GBS: f64 = 504.0;

// D3Q19 bandwidth model: ping-pong scalars per cell per step.
// 19f_read + 19f_write + rho + tau + 3u + 3force = 46 scalars
const D3Q19_SCALARS_PER_CELL: usize = 46;

// ============================================================================
// CLI
// ============================================================================

/// CUDA kernel baseline benchmark.
///
/// Sweeps kernel variants across precisions and grid sizes.
/// Writes ASCII table to stdout and CSV to --output path.
#[derive(Parser, Debug)]
#[command(name = "cuda-precision-bench", version)]
struct BenchConfig {
    /// Comma-separated grid side lengths to benchmark (e.g. "32,64,128,256").
    #[arg(long, default_value = "32,64,128,256")]
    grids: String,

    /// Warmup steps for LBM workloads.
    #[arg(long, default_value_t = 20)]
    warmup: usize,

    /// Timing steps for 32^3 and 64^3 grids.
    #[arg(long, default_value_t = 200)]
    steps_small: usize,

    /// Timing steps for 128^3 grids.
    #[arg(long, default_value_t = 100)]
    steps_mid: usize,

    /// Timing steps for 256^3+ grids.
    #[arg(long, default_value_t = 50)]
    steps_large: usize,

    /// Workloads to run (comma-separated: lbm,box,dark_halo,sparse,voudon,all).
    #[arg(long, default_value = "all")]
    workloads: String,

    /// Output CSV path.
    #[arg(long, default_value = "data/benchmarks/cuda_kernel_baseline.csv")]
    output: String,
}

impl BenchConfig {
    fn grids(&self) -> Vec<usize> {
        parse_grids(&self.grids)
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

fn parse_grids(s: &str) -> Vec<usize> {
    s.split(',')
        .filter_map(|p| p.trim().parse::<usize>().ok())
        .collect()
}

// ============================================================================
// Result row
// ============================================================================

#[derive(Debug, Clone)]
struct BenchRow {
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
    notes: String,
}

// ============================================================================
// Metric helpers
// ============================================================================

fn mlups(n_cells: usize, steps: usize, elapsed_ms: f64) -> f64 {
    n_cells as f64 * steps as f64 / elapsed_ms / 1e3
}

fn bandwidth_gbs(n_cells: usize, steps: usize, elapsed_ms: f64, elem_bytes: usize) -> f64 {
    let bytes = D3Q19_SCALARS_PER_CELL as f64 * n_cells as f64 * steps as f64
        * elem_bytes as f64;
    bytes / (elapsed_ms * 1e-3) / 1e9
}

// ============================================================================
// Output
// ============================================================================

fn write_csv(path: &str, rows: &[BenchRow]) -> Result<()> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = std::fs::File::create(path)?;
    writeln!(
        f,
        "workload,precision,variant,grid_size,param2,n_steps,warmup,\
         elapsed_ms,mlups,bandwidth_gbs,bandwidth_pct_peak,notes"
    )?;
    for r in rows {
        writeln!(
            f,
            "{},{},{},{},{},{},{},{:.3},{},{},{},\"{}\"",
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
            r.bandwidth_pct_peak.map_or(String::new(), |v| format!("{v:.1}")),
            r.notes,
        )?;
    }
    Ok(())
}

fn print_table(rows: &[BenchRow]) {
    let mut groups: Vec<&'static str> = Vec::new();
    for r in rows {
        if !groups.contains(&r.workload) {
            groups.push(r.workload);
        }
    }

    for group in groups {
        let subset: Vec<&BenchRow> = rows.iter().filter(|r| r.workload == group).collect();
        println!();
        println!("Workload: {group}");
        println!("{:-<80}", "");
        println!(
            "{:<22} {:<6} {:<18} {:>8} {:>8} {:>8} {:>7}",
            "variant", "grid", "precision", "MLUPS", "BW_GBS", "BW_PCT", "ms"
        );
        println!("{:-<80}", "");
        for r in &subset {
            println!(
                "{:<22} {:>5}^3 {:<18} {:>8} {:>8} {:>7} {:>7.1}",
                r.variant,
                r.grid_size,
                r.precision,
                r.mlups.map_or("-".to_string(), |v| format!("{v:.1}")),
                r.bandwidth_gbs.map_or("-".to_string(), |v| format!("{v:.1}")),
                r.bandwidth_pct_peak.map_or("-".to_string(), |v| format!("{v:.1}%")),
                r.elapsed_ms,
            );
        }
        println!("{:-<80}", "");
    }
}

fn print_precision_guide() {
    println!();
    println!("Ada SM 8.9 precision tiers for LBM scalar workloads:");
    println!("  FP32 : ~40 TFLOPS scalar; memory bottleneck at 128^3+. Primary production tier.");
    println!(
        "  BF16 : Same scalar speed as FP32 (no BF16 scalar FFMA acceleration on Ada);"
    );
    println!(
        "         benefit is 2x bandwidth reduction. Risk: 8-bit mantissa -> LBM instability"
    );
    println!("         for tau < 0.55 or at shear boundaries.");
    println!(
        "  FP64 : ~0.6 TFLOPS (64:1 ratio on gaming SKU). Use only for validation, not perf."
    );
    println!(
        "  TF32/FP16/FP8/INT8/INT4: Tensor Core formats; only fast via cuBLAS/cuDNN matrix"
    );
    println!("         paths. Not applicable to custom scalar LBM or AVT kernels.");
    println!();
    println!("Kernel variant decision table (FP32 SoA):");
    println!(
        "  grid <= 32^3  : coarsened (instruction pressure, not bandwidth, is limit)"
    );
    println!(
        "  grid = 64^3   : mrt_tiled or coarsened (test both; L2 saturation varies)"
    );
    println!(
        "  grid = 128^3  : mrt_tiled (target >= 60% peak bandwidth per C-1304)"
    );
    println!(
        "  grid >= 256^3 : aa (A-A halves VRAM; stays compute-bound at same MLUPS)"
    );
    println!("  VRAM-critical : aa + bf16 (4x reduction vs fp64 ping-pong)");
    println!();
    println!("YSU lessons embedded in benchmark design:");
    println!(
        "  float2 widening   : measured as standard vs coarsened MLUPS delta"
    );
    println!(
        "  ILP (MRT chains)  : measured as bgk vs mrt MLUPS delta (MRT should win at 128^3+)"
    );
    println!(
        "  Tiling LDS vs L2  : measured as standard vs tiled delta (target 15-25%)"
    );
    println!(
        "  A-A VRAM halving  : measured as aa vs standard at 256^3 (same MLUPS, half VRAM)"
    );
    println!();
}

// ============================================================================
// GPU workloads
// ============================================================================

#[cfg(feature = "gpu")]
mod gpu {
    use super::{BenchRow, BenchConfig, PEAK_BW_GBS, bandwidth_gbs, mlups};
    use anyhow::Result;
    use lbm_3d_cuda::{
        DarkHaloCudaSolver, LbmSolver3DCuda, Precision,
        box_counting_gpu::GpuBoxCounter,
        probe_cuda_device_props,
        sparse::{SparseBrickMap, SparseLbmSolver},
    };
    use std::time::Instant;

    // -------------------------------------------------------------------------
    // Workload A: LBM sweep
    // -------------------------------------------------------------------------

    struct Fp32Variant {
        id: &'static str,
        use_mrt: bool,
        use_pull: bool,
        use_tiling: bool,
        use_coarsening: bool,
        use_aa: bool,
        use_smagorinsky: bool,
        notes: &'static str,
    }

    const FP32_VARIANTS: &[Fp32Variant] = &[
        Fp32Variant {
            id: "standard",
            use_mrt: false, use_pull: false, use_tiling: false,
            use_coarsening: false, use_aa: false, use_smagorinsky: false,
            notes: "baseline BGK SoA ping-pong",
        },
        Fp32Variant {
            id: "mrt",
            use_mrt: true, use_pull: false, use_tiling: false,
            use_coarsening: false, use_aa: false, use_smagorinsky: false,
            notes: "d'Humieres MRT collision",
        },
        Fp32Variant {
            id: "pull",
            use_mrt: false, use_pull: true, use_tiling: false,
            use_coarsening: false, use_aa: false, use_smagorinsky: false,
            notes: "pull-scheme coalesced writes",
        },
        Fp32Variant {
            id: "mrt_pull",
            use_mrt: true, use_pull: true, use_tiling: false,
            use_coarsening: false, use_aa: false, use_smagorinsky: false,
            notes: "MRT + pull-scheme",
        },
        Fp32Variant {
            id: "tiled",
            use_mrt: false, use_pull: false, use_tiling: true,
            use_coarsening: false, use_aa: false, use_smagorinsky: false,
            notes: "8x8x4 shared-mem tile + 1-cell halo",
        },
        Fp32Variant {
            id: "mrt_tiled",
            use_mrt: true, use_pull: false, use_tiling: true,
            use_coarsening: false, use_aa: false, use_smagorinsky: false,
            notes: "MRT + 8x8x4 tile (target 60%+ peak bw)",
        },
        Fp32Variant {
            id: "coarsened",
            use_mrt: false, use_pull: false, use_tiling: false,
            use_coarsening: true, use_aa: false, use_smagorinsky: false,
            notes: "thread-coarsened float2/float4 ILP",
        },
        Fp32Variant {
            id: "mrt_coarsened",
            use_mrt: true, use_pull: false, use_tiling: false,
            use_coarsening: true, use_aa: false, use_smagorinsky: false,
            notes: "MRT + coarsened float2/float4",
        },
        Fp32Variant {
            id: "aa",
            use_mrt: false, use_pull: false, use_tiling: false,
            use_coarsening: false, use_aa: true, use_smagorinsky: false,
            notes: "A-A single-buffer half-VRAM",
        },
        Fp32Variant {
            id: "mrt_aa",
            use_mrt: true, use_pull: false, use_tiling: false,
            use_coarsening: false, use_aa: true, use_smagorinsky: false,
            notes: "MRT + A-A streaming",
        },
        Fp32Variant {
            id: "smagorinsky_mrt_tiled",
            use_mrt: true, use_pull: false, use_tiling: true,
            use_coarsening: false, use_aa: false, use_smagorinsky: true,
            notes: "LES Smagorinsky + MRT tiled; tau set once before timing",
        },
    ];

    fn make_fp32_solver(n: usize, v: &Fp32Variant) -> Result<LbmSolver3DCuda> {
        let mut solver = if v.use_mrt {
            LbmSolver3DCuda::new_mrt(n, n, n, 0.7, Precision::FP32)?
        } else {
            LbmSolver3DCuda::new(n, n, n, 0.7, Precision::FP32)?
        };
        if v.use_pull {
            solver.set_pull_streaming(true);
        }
        if v.use_tiling {
            solver.set_tiling(true);
        }
        if v.use_coarsening {
            solver.set_coarsening(true);
        }
        if v.use_aa {
            solver.set_aa_streaming(true);
        }
        Ok(solver)
    }

    pub fn run_lbm_sweep(cfg: &BenchConfig) -> Result<Vec<BenchRow>> {
        let mut rows = Vec::new();
        let grids = cfg.grids();

        // FP32 SoA variants
        for &n in &grids {
            let n_steps = cfg.n_steps_for(n);
            let n_cells = n * n * n;
            let elem_bytes = 4usize;

            for v in FP32_VARIANTS {
                eprint!(
                    "  [A] FP32/{:<22} {}^3 ({} steps)... ",
                    v.id, n, n_steps
                );
                let mut solver = match make_fp32_solver(n, v) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("SKIP: {e}");
                        continue;
                    }
                };
                // Smagorinsky: set tau field once before timing
                if v.use_smagorinsky {
                    let dx = 1.0 / n as f64;
                    if let Err(e) = solver.update_smagorinsky_tau(0.1, dx, 0.7) {
                        eprintln!("SKIP smagorinsky setup: {e}");
                        continue;
                    }
                }
                // Warmup (captures graph on first pair call)
                if let Err(e) = solver.step_n(cfg.warmup) {
                    eprintln!("SKIP warmup: {e}");
                    continue;
                }
                if let Err(e) = solver.context().synchronize() {
                    eprintln!("SKIP sync: {e}");
                    continue;
                }
                // Timing
                let t0 = Instant::now();
                if let Err(e) = solver.step_n(n_steps) {
                    eprintln!("SKIP step: {e}");
                    continue;
                }
                if let Err(e) = solver.context().synchronize() {
                    eprintln!("SKIP sync: {e}");
                    continue;
                }
                let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
                let ml = mlups(n_cells, n_steps, elapsed_ms);
                let bw = bandwidth_gbs(n_cells, n_steps, elapsed_ms, elem_bytes);
                let bw_pct = bw / PEAK_BW_GBS * 100.0;
                eprintln!("{ml:.1} MLUPS  {bw:.1} GB/s  ({bw_pct:.1}%)");
                rows.push(BenchRow {
                    workload: "lbm_sweep",
                    precision: "FP32".to_string(),
                    variant: v.id.to_string(),
                    grid_size: n,
                    param2: String::new(),
                    n_steps,
                    warmup: cfg.warmup,
                    elapsed_ms,
                    mlups: Some(ml),
                    bandwidth_gbs: Some(bw),
                    bandwidth_pct_peak: Some(bw_pct),
                    notes: v.notes.to_string(),
                });
            }
        }

        // BF16 standard
        for &n in &grids {
            let n_steps = cfg.n_steps_for(n);
            let n_cells = n * n * n;
            let elem_bytes = 2usize; // bf16 distribution; approx (macro fields are f32)
            eprint!("  [A] BF16/standard            {}^3 ({} steps)... ", n, n_steps);
            let mut solver = match LbmSolver3DCuda::new(n, n, n, 0.7, Precision::BF16) {
                Ok(s) => s,
                Err(e) => { eprintln!("SKIP: {e}"); continue; }
            };
            if solver.step_n(cfg.warmup).is_err() { eprintln!("SKIP warmup"); continue; }
            let _ = solver.context().synchronize();
            let t0 = Instant::now();
            if solver.step_n(n_steps).is_err() { eprintln!("SKIP step"); continue; }
            let _ = solver.context().synchronize();
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
            let ml = mlups(n_cells, n_steps, elapsed_ms);
            let bw = bandwidth_gbs(n_cells, n_steps, elapsed_ms, elem_bytes);
            let bw_pct = bw / PEAK_BW_GBS * 100.0;
            eprintln!("{ml:.1} MLUPS  {bw:.1} GB/s  ({bw_pct:.1}%)");
            rows.push(BenchRow {
                workload: "lbm_sweep",
                precision: "BF16".to_string(),
                variant: "standard".to_string(),
                grid_size: n,
                param2: String::new(),
                n_steps,
                warmup: cfg.warmup,
                elapsed_ms,
                mlups: Some(ml),
                bandwidth_gbs: Some(bw),
                bandwidth_pct_peak: Some(bw_pct),
                notes: "BF16 AoS; same MLUPS as FP32 (no scalar FFMA accel on Ada); 2x bw reduction".to_string(),
            });
        }

        // FP64 standard
        for &n in &grids {
            let n_steps = cfg.n_steps_for(n);
            let n_cells = n * n * n;
            let elem_bytes = 8usize;
            eprint!("  [A] FP64/standard            {}^3 ({} steps)... ", n, n_steps);
            let mut solver = match LbmSolver3DCuda::new(n, n, n, 0.7, Precision::FP64) {
                Ok(s) => s,
                Err(e) => { eprintln!("SKIP: {e}"); continue; }
            };
            if solver.step_n(cfg.warmup).is_err() { eprintln!("SKIP warmup"); continue; }
            let _ = solver.context().synchronize();
            let t0 = Instant::now();
            if solver.step_n(n_steps).is_err() { eprintln!("SKIP step"); continue; }
            let _ = solver.context().synchronize();
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
            let ml = mlups(n_cells, n_steps, elapsed_ms);
            let bw = bandwidth_gbs(n_cells, n_steps, elapsed_ms, elem_bytes);
            let bw_pct = bw / PEAK_BW_GBS * 100.0;
            eprintln!("{ml:.1} MLUPS  {bw:.1} GB/s  ({bw_pct:.1}%)");
            rows.push(BenchRow {
                workload: "lbm_sweep",
                precision: "FP64".to_string(),
                variant: "standard".to_string(),
                grid_size: n,
                param2: String::new(),
                n_steps,
                warmup: cfg.warmup,
                elapsed_ms,
                mlups: Some(ml),
                bandwidth_gbs: Some(bw),
                bandwidth_pct_peak: Some(bw_pct),
                notes: "FP64 AoS; ~0.6 TFLOPS on Ada gaming SKU; validation tier only".to_string(),
            });
        }

        Ok(rows)
    }

    // -------------------------------------------------------------------------
    // Workload B: GPU box-counting fractal dimension
    // -------------------------------------------------------------------------

    pub fn run_box_counting(cfg: &BenchConfig) -> Result<Vec<BenchRow>> {
        let mut rows = Vec::new();
        let grids = cfg.grids();

        for &n in &grids {
            eprint!("  [B] box_counting {}^3... ", n);
            // Bootstrap a small solver to get a GPU context and stream,
            // then run LBM briefly to populate a realistic rho field.
            let mut solver = match LbmSolver3DCuda::new(n, n, n, 0.7, Precision::FP32) {
                Ok(s) => s,
                Err(e) => { eprintln!("SKIP (solver init): {e}"); continue; }
            };
            // Warm up 20 LBM steps to get a non-trivial rho field.
            if let Err(e) = solver.step_n(20) {
                eprintln!("SKIP (LBM warmup): {e}");
                continue;
            }
            if let Err(e) = solver.context().synchronize() {
                eprintln!("SKIP (sync): {e}");
                continue;
            }

            let mut counter = match GpuBoxCounter::new(solver.context()) {
                Ok(c) => c,
                Err(e) => { eprintln!("SKIP (counter init): {e}"); continue; }
            };

            let t0 = Instant::now();
            let result = match counter.fractal_dimension_device_auto(
                solver.d_rho_bytes(), n, n, n,
            ) {
                Ok(r) => r,
                Err(e) => { eprintln!("SKIP (box-count): {e}"); continue; }
            };
            let _ = solver.context().synchronize();
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;

            eprintln!(
                "D_f={:.4}  R^2={:.4}  {:.1} ms",
                result.d_f, result.r_squared, elapsed_ms
            );
            rows.push(BenchRow {
                workload: "box_counting",
                precision: "FP32".to_string(),
                variant: "gpu_otsu".to_string(),
                grid_size: n,
                param2: format!("{}_scales", result.box_sizes.len()),
                n_steps: 0,
                warmup: 20,
                elapsed_ms,
                mlups: None,
                bandwidth_gbs: None,
                bandwidth_pct_peak: None,
                notes: format!(
                    "D_f={:.4}; R2={:.4}; warp_ballot 32x fewer atomics",
                    result.d_f, result.r_squared
                ),
            });
        }

        Ok(rows)
    }

    // -------------------------------------------------------------------------
    // Workload C: Chingon AVT contraction (skip -- requires PackedAvt construction)
    // -------------------------------------------------------------------------

    pub fn run_chingon_skip() -> Vec<BenchRow> {
        vec![BenchRow {
            workload: "chingon_avt",
            precision: "N/A".to_string(),
            variant: "skipped".to_string(),
            grid_size: 0,
            param2: String::new(),
            n_steps: 0,
            warmup: 0,
            elapsed_ms: 0.0,
            mlups: None,
            bandwidth_gbs: None,
            bandwidth_pct_peak: None,
            notes: "Chingon requires PackedAvt from gororoba_algebra; not wired in this binary".to_string(),
        }]
    }

    // -------------------------------------------------------------------------
    // Workload D: Dark halo fused kernel
    // -------------------------------------------------------------------------

    pub fn run_dark_halo(cfg: &BenchConfig) -> Result<Vec<BenchRow>> {
        let mut rows = Vec::new();
        // Dark halo is memory-intensive; cap at 128^3
        let bench_grids: Vec<usize> = cfg.grids().into_iter().filter(|&n| n <= 128).collect();

        for n in bench_grids {
            let n_cells = n * n * n;
            eprint!("  [D] dark_halo {}^3... ", n);

            let mut solver = match DarkHaloCudaSolver::new(n, n, n) {
                Ok(s) => s,
                Err(e) => { eprintln!("SKIP: {e}"); continue; }
            };

            // Warmup: small run
            let warmup_steps = cfg.warmup.min(20) as u32;
            if let Err(e) = solver.run_k_value(
                256, warmup_steps, 0.7, 0.1, 1.5, 1e-4, 0.0, 0,
            ) {
                eprintln!("SKIP (warmup): {e}");
                continue;
            }

            let n_steps = cfg.n_steps_for(n).min(50) as u32;
            let t0 = Instant::now();
            if let Err(e) = solver.run_k_value(
                256, n_steps, 0.7, 0.1, 1.5, 1e-4, 0.0, 0,
            ) {
                eprintln!("SKIP (bench): {e}");
                continue;
            }
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;

            // MLUPS from the LBM step portion (approximate; pipeline includes ZD setup)
            let ml = mlups(n_cells, n_steps as usize, elapsed_ms);
            eprintln!("{ml:.1} MLUPS  {elapsed_ms:.1} ms  (includes ZD setup + halo detection)");

            rows.push(BenchRow {
                workload: "dark_halo",
                precision: "FP32".to_string(),
                variant: "fused_soa_zd_detect".to_string(),
                grid_size: n,
                param2: String::new(),
                n_steps: n_steps as usize,
                warmup: cfg.warmup,
                elapsed_ms,
                mlups: Some(ml),
                bandwidth_gbs: None,
                bandwidth_pct_peak: None,
                notes: "fused lbm_step_soa + ZD viscosity modulation + dark_halo_detector; k=256".to_string(),
            });
        }

        Ok(rows)
    }

    // -------------------------------------------------------------------------
    // Workload E: Sparse brick-map LBM
    // -------------------------------------------------------------------------

    pub fn run_sparse_lbm(cfg: &BenchConfig) -> Result<Vec<BenchRow>> {
        let mut rows = Vec::new();
        // Only benchmark 256^3; 1024^3 requires 1 GB CPU mask -- skip by default
        let sparse_grids = [256usize];
        let occupancies = [0.10f64, 0.25, 0.50];

        for &n in &sparse_grids {
            let n_cells_total = n * n * n;

            // Check if device has enough VRAM for 256^3 sparse
            if let Some(props) = probe_cuda_device_props() {
                let vram_gb = props.l2_bytes as f64 / (1 << 30) as f64;
                // l2_bytes is L2 cache size, not VRAM -- just use it as a proxy sanity check
                // A realistic VRAM check would use cuMemGetInfo, but we keep it simple here
                let _ = vram_gb;
            }

            for &occ in &occupancies {
                eprint!("  [E] sparse_lbm {}^3 occ={:.0}%... ", n, occ * 100.0);

                // Bootstrap a tiny LBM solver just to obtain Arc<CudaContext> and Arc<CudaStream>.
                // We drop the solver after extracting the Arcs; the context stays alive via clone.
                let bootstrap = match LbmSolver3DCuda::new(8, 8, 8, 0.7, Precision::FP32) {
                    Ok(s) => s,
                    Err(e) => { eprintln!("SKIP (bootstrap): {e}"); continue; }
                };
                let ctx = bootstrap.context().clone();
                let stream = bootstrap.stream().clone();
                drop(bootstrap);

                // Build geometry mask at cell level: mark first occ*n_cells as occupied
                let n_occupied = (occ * n_cells_total as f64).round() as usize;
                let mut mask_host = vec![0u8; n_cells_total];
                for v in mask_host.iter_mut().take(n_occupied) {
                    *v = 1;
                }

                let d_mask = match stream.clone_htod(&mask_host) {
                    Ok(m) => m,
                    Err(e) => { eprintln!("SKIP (mask upload): {e}"); continue; }
                };
                drop(mask_host); // free 16 MB CPU allocation

                let bm = match SparseBrickMap::new_from_geometry(
                    ctx.clone(), stream.clone(), n, n, n, &d_mask,
                ) {
                    Ok(b) => b,
                    Err(e) => { eprintln!("SKIP (brick map): {e}"); continue; }
                };
                drop(d_mask);

                let n_active = bm.n_active_bricks;
                let n_active_cells = n_active * 512;
                let vram_mb = (n_active_cells * 19 * 4 * 2 + n_active_cells * 4 * 5)
                    / (1024 * 1024);

                let mut sparse_solver = match SparseLbmSolver::new(bm) {
                    Ok(s) => s,
                    Err(e) => { eprintln!("SKIP (sparse solver): {e}"); continue; }
                };

                // Warmup
                let warmup_steps = cfg.warmup;
                if let Err(e) = sparse_solver.evolve(warmup_steps) {
                    eprintln!("SKIP (warmup): {e}");
                    continue;
                }

                let n_steps = cfg.n_steps_for(n);
                let t0 = Instant::now();
                if let Err(e) = sparse_solver.evolve(n_steps) {
                    eprintln!("SKIP (bench): {e}");
                    continue;
                }
                // SparseLbmSolver::evolve() includes ctx.synchronize() at end
                let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;

                let ml = mlups(n_active_cells, n_steps, elapsed_ms);
                let sparse_overhead_pct = (n_cells_total as f64 - n_active_cells as f64)
                    / n_cells_total as f64 * 100.0;
                eprintln!(
                    "{ml:.1} MLUPS  {n_active} active bricks  {vram_mb} MB VRAM"
                );

                rows.push(BenchRow {
                    workload: "sparse_lbm",
                    precision: "FP32".to_string(),
                    variant: "sparse_brick_map".to_string(),
                    grid_size: n,
                    param2: format!("occ{:.0}pct", occ * 100.0),
                    n_steps,
                    warmup: warmup_steps,
                    elapsed_ms,
                    mlups: Some(ml),
                    bandwidth_gbs: None,
                    bandwidth_pct_peak: None,
                    notes: format!(
                        "{n_active} active bricks; {vram_mb} MB VRAM; \
                         {sparse_overhead_pct:.0}% skipped cells; 8x8x8 bricks A-A pattern"
                    ),
                });
            }
        }

        Ok(rows)
    }

    // -------------------------------------------------------------------------
    // Workload F: Voudon frustration modulation
    // -------------------------------------------------------------------------

    pub fn run_voudon(cfg: &BenchConfig) -> Result<Vec<BenchRow>> {
        let mut rows = Vec::new();
        let voudon_grids: Vec<usize> = cfg.grids().into_iter().filter(|&n| n <= 64).collect();

        for n in voudon_grids {
            let n_cells = n * n * n;
            eprint!("  [F] voudon {}^3... ", n);

            let mut solver = match LbmSolver3DCuda::new(n, n, n, 0.7, Precision::FP32) {
                Ok(s) => s,
                Err(e) => { eprintln!("SKIP: {e}"); continue; }
            };

            // Allocate frustration field (uniform frustration = 0.5 everywhere)
            let frustration_host = vec![0.5f32; n_cells];
            let d_frustration = match solver.stream().clone_htod(&frustration_host) {
                Ok(d) => d,
                Err(e) => { eprintln!("SKIP (alloc): {e}"); continue; }
            };
            drop(frustration_host);

            // Warmup calls
            for _ in 0..cfg.warmup {
                if let Err(e) = solver.update_tau_from_voudon(&d_frustration, 0.7, 0.05) {
                    eprintln!("SKIP (warmup): {e}");
                    break;
                }
            }
            if let Err(e) = solver.context().synchronize() {
                eprintln!("SKIP (sync): {e}");
                continue;
            }

            // Timing: 200 individual kernel calls (measures launch overhead)
            let n_calls = 200usize;
            let t0 = Instant::now();
            for _ in 0..n_calls {
                if let Err(e) = solver.update_tau_from_voudon(&d_frustration, 0.7, 0.05) {
                    eprintln!("SKIP (bench call): {e}");
                    break;
                }
            }
            if let Err(e) = solver.context().synchronize() {
                eprintln!("SKIP (final sync): {e}");
                continue;
            }
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
            let per_call_us = elapsed_ms / n_calls as f64 * 1e3;
            eprintln!("{per_call_us:.1} us/call  total {elapsed_ms:.1} ms");

            rows.push(BenchRow {
                workload: "voudon_modulate",
                precision: "FP32".to_string(),
                variant: "update_tau_from_voudon".to_string(),
                grid_size: n,
                param2: String::new(),
                n_steps: n_calls,
                warmup: cfg.warmup,
                elapsed_ms,
                mlups: None,
                bandwidth_gbs: None,
                bandwidth_pct_peak: None,
                notes: format!(
                    "{per_call_us:.1} us/call; {n_calls} calls; first live caller removes dead_code"
                ),
            });
        }

        Ok(rows)
    }

    // -------------------------------------------------------------------------
    // Workload G: cuFFT round-trip (conditional)
    // -------------------------------------------------------------------------

    #[cfg(feature = "cufft")]
    pub fn run_cufft(cfg: &BenchConfig) -> Result<Vec<BenchRow>> {
        use std::time::Instant;
        let mut rows = Vec::new();
        let fft_grids: Vec<usize> = cfg.grids().into_iter().filter(|&n| n <= 128).collect();

        for n in fft_grids {
            eprint!("  [G] cufft {}^3... ", n);
            let mut solver = match LbmSolver3DCuda::new(n, n, n, 0.7, Precision::FP32) {
                Ok(s) => s,
                Err(e) => { eprintln!("SKIP: {e}"); continue; }
            };

            // Forward FFT
            let t0 = Instant::now();
            if let Err(e) = solver.fft_forward() {
                eprintln!("SKIP (forward): {e}");
                continue;
            }
            let _ = solver.context().synchronize();
            let forward_ms = t0.elapsed().as_secs_f64() * 1e3;

            // Inverse FFT
            let t1 = Instant::now();
            if let Err(e) = solver.fft_inverse() {
                eprintln!("SKIP (inverse): {e}");
                continue;
            }
            let _ = solver.context().synchronize();
            let inverse_ms = t1.elapsed().as_secs_f64() * 1e3;

            let round_trip_ms = forward_ms + inverse_ms;
            let n_cells = n * n * n;
            // Effective bandwidth: 2 passes (forward + inverse) * n_cells * 4 bytes * 2 (complex)
            let bw = 2.0 * n_cells as f64 * 8.0 / (round_trip_ms * 1e-3) / 1e9;
            eprintln!("fwd={forward_ms:.1} ms  inv={inverse_ms:.1} ms  bw={bw:.1} GB/s");

            rows.push(BenchRow {
                workload: "cufft",
                precision: "FP32".to_string(),
                variant: "round_trip".to_string(),
                grid_size: n,
                param2: String::new(),
                n_steps: 1,
                warmup: 0,
                elapsed_ms: round_trip_ms,
                mlups: None,
                bandwidth_gbs: Some(bw),
                bandwidth_pct_peak: Some(bw / super::PEAK_BW_GBS * 100.0),
                notes: format!(
                    "fwd={forward_ms:.1} ms; inv={inverse_ms:.1} ms; complex f32 in-place"
                ),
            });
        }

        Ok(rows)
    }

    // -------------------------------------------------------------------------
    // Orchestrator
    // -------------------------------------------------------------------------

    pub fn run(cfg: &BenchConfig) -> Result<Vec<BenchRow>> {
        let mut rows: Vec<BenchRow> = Vec::new();

        if cfg.run_workload("lbm") {
            eprintln!("[A] LBM sweep...");
            match run_lbm_sweep(cfg) {
                Ok(mut r) => rows.append(&mut r),
                Err(e) => eprintln!("  Workload A failed: {e}"),
            }
        }

        if cfg.run_workload("box") {
            eprintln!("[B] GPU box-counting...");
            match run_box_counting(cfg) {
                Ok(mut r) => rows.append(&mut r),
                Err(e) => eprintln!("  Workload B failed: {e}"),
            }
        }

        // Workload C: Chingon -- skip (PackedAvt requires full algebra construction)
        if cfg.run_workload("chingon") {
            eprintln!("[C] Chingon AVT -- skipped (see notes)");
            rows.append(&mut run_chingon_skip());
        }

        if cfg.run_workload("dark_halo") {
            eprintln!("[D] Dark halo fused kernel...");
            match run_dark_halo(cfg) {
                Ok(mut r) => rows.append(&mut r),
                Err(e) => eprintln!("  Workload D failed: {e}"),
            }
        }

        if cfg.run_workload("sparse") {
            eprintln!("[E] Sparse brick-map LBM...");
            match run_sparse_lbm(cfg) {
                Ok(mut r) => rows.append(&mut r),
                Err(e) => eprintln!("  Workload E failed: {e}"),
            }
        }

        if cfg.run_workload("voudon") {
            eprintln!("[F] Voudon frustration modulation...");
            match run_voudon(cfg) {
                Ok(mut r) => rows.append(&mut r),
                Err(e) => eprintln!("  Workload F failed: {e}"),
            }
        }

        #[cfg(feature = "cufft")]
        if cfg.run_workload("cufft") {
            eprintln!("[G] cuFFT round-trip...");
            match run_cufft(cfg) {
                Ok(mut r) => rows.append(&mut r),
                Err(e) => eprintln!("  Workload G failed: {e}"),
            }
        }

        Ok(rows)
    }
}

// ============================================================================
// Entry point
// ============================================================================

fn main() -> Result<()> {
    let cfg = BenchConfig::parse();

    print_precision_guide();

    #[cfg(feature = "gpu")]
    {
        eprintln!("cuda-precision-bench starting...");
        eprintln!("  grids    : {}", cfg.grids);
        eprintln!("  workloads: {}", cfg.workloads);
        eprintln!("  output   : {}", cfg.output);
        eprintln!();

        let rows = gpu::run(&cfg)?;

        if rows.is_empty() {
            eprintln!("WARNING: no benchmark rows produced (is CUDA available?)");
        } else {
            print_table(&rows);
            write_csv(&cfg.output, &rows)?;
            eprintln!();
            eprintln!("Benchmark complete. {} rows written to {}", rows.len(), cfg.output);
        }
    }

    #[cfg(not(feature = "gpu"))]
    {
        eprintln!("ERROR: compile with --features gpu to run CUDA benchmarks");
        std::process::exit(1);
    }

    Ok(())
}

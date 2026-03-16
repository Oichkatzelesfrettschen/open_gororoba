//! CUDA Kernel Baseline Benchmark
//!
//! Sweeps the full kernel decision space across six workload groups plus the
//! full precision tier ladder (DD/FP64/FP32/BF16/FP16/FP8/INT8 + TC WMMA proxy),
//! computes MLUPS, effective bandwidth, and VRAM footprint, then writes an ASCII
//! summary table (stdout) plus a CSV file for agent consumption.
//!
//! This is NOT a CI gate. It is a reference tool for design decisions.
//! See: plans/golden-splashing-boole.md for the full specification.

use anyhow::Result;
use clap::Parser;
use std::io::Write as _;

// RTX 4070 Ti theoretical peak memory bandwidth.
const PEAK_BW_GBS: f64 = 504.0;

// D3Q19 bandwidth model: ping-pong scalars per cell per step.
// Non-padded (FP32/BF16/FP64/DD/sparse): 19f_read + 19f_write + 8 macro = 46 scalars.
// Padded AoS (FP16/FP8/INT8): 20f_read + 20f_write + 8 macro = 48 scalars.
// Macroscopic fields (rho/u/tau/force = 8 scalars) are always FP32;
// treating them as elem_bytes here is an approximation consistent across all tiers.
const D3Q19_SCALARS_NON_PADDED: usize = 46;
const D3Q19_SCALARS_PADDED: usize = 48;

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

    /// Workloads to run (comma-separated: lbm,box,dark_halo,sparse,voudon,tc,all).
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
    vram_dist_mb: Option<usize>,
    notes: String,
}

// ============================================================================
// Metric helpers
// ============================================================================

fn mlups(n_cells: usize, steps: usize, elapsed_ms: f64) -> f64 {
    n_cells as f64 * steps as f64 / elapsed_ms / 1e3
}

/// Effective bandwidth model for D3Q19 LBM.
/// `dist_stride`: actual per-cell distribution count (19 for non-padded, 20 for AoS-padded).
fn bandwidth_gbs(
    n_cells: usize,
    steps: usize,
    elapsed_ms: f64,
    elem_bytes: usize,
    dist_stride: usize,
) -> f64 {
    let scalars = if dist_stride == 20 {
        D3Q19_SCALARS_PADDED
    } else {
        D3Q19_SCALARS_NON_PADDED
    };
    let bytes = scalars as f64 * n_cells as f64 * steps as f64 * elem_bytes as f64;
    bytes / (elapsed_ms * 1e-3) / 1e9
}

/// Ping-pong distribution buffer footprint in MiB.
/// `dist_stride`: 19 for non-padded layouts, 20 for AoS-padded (FP16/FP8/INT8).
fn vram_ping_pong_mb(n_cells: usize, elem_bytes: usize, dist_stride: usize) -> usize {
    n_cells * dist_stride * elem_bytes * 2 / (1024 * 1024)
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
         elapsed_ms,mlups,bandwidth_gbs,bandwidth_pct_peak,vram_dist_mb,notes"
    )?;
    for r in rows {
        writeln!(
            f,
            "{},{},{},{},{},{},{},{:.3},{},{},{},{},\"{}\"",
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
        println!("{:-<88}", "");
        println!(
            "{:<22} {:<6} {:<16} {:>8} {:>8} {:>7} {:>7} {:>7}",
            "variant", "grid", "precision", "MLUPS", "BW_GBS", "BW_PCT", "VRAM_MB", "ms"
        );
        println!("{:-<88}", "");
        for r in &subset {
            println!(
                "{:<22} {:>5}^3 {:<16} {:>8} {:>8} {:>7} {:>7} {:>7.1}",
                r.variant,
                r.grid_size,
                r.precision,
                r.mlups.map_or("-".to_string(), |v| format!("{v:.1}")),
                r.bandwidth_gbs
                    .map_or("-".to_string(), |v| format!("{v:.1}")),
                r.bandwidth_pct_peak
                    .map_or("-".to_string(), |v| format!("{v:.1}%")),
                r.vram_dist_mb.map_or("-".to_string(), |v| format!("{v}")),
                r.elapsed_ms,
            );
        }
        println!("{:-<88}", "");
    }
}

fn print_precision_guide() {
    println!();
    println!(
        "Ada SM 8.9 precision tiers -- D3Q19 LBM scalar kernels (RTX 4070 Ti, 504 GB/s peak):"
    );
    println!();
    println!("  Tier         Bytes/dist  Mantissa  VRAM vs FP32  Notes");
    println!(
        "  -----------  ----------  --------  ------------  -----------------------------------------"
    );
    println!(
        "  DD_FP128           16      ~106-bit      16x     two f64 (hi+lo); Knuth 2-sum + Veltkamp FMA;"
    );
    println!(
        "                                                    4 bufs (hi/lo ping/pong); research tier only"
    );
    println!(
        "  FP64                8       53-bit        4x     IEEE double; ~0.6 TFLOPS on Ada gaming SKU;"
    );
    println!(
        "                                                    validation only; 64:1 speed penalty vs FP32"
    );
    println!(
        "  FP32                4       24-bit        1x     primary production tier; ~40 TFLOPS scalar;"
    );
    println!("                                                    memory bottleneck at 128^3+");
    println!(
        "  BF16                2        8-bit        0.5x   2x bandwidth save; same scalar speed as FP32"
    );
    println!(
        "                                                    (no BF16 FFMA accel on Ada); 8-bit exponent"
    );
    println!(
        "                                                    risk: instability for tau < 0.55"
    );
    println!(
        "  FP16                2       10-bit        0.5x   2x bandwidth save; half2 paired loads;"
    );
    println!(
        "                                                    narrower range than BF16 (5-bit exponent);"
    );
    println!(
        "                                                    better mantissa (10-bit vs 7-bit)"
    );
    println!(
        "  FP8_e4m3            1        4-bit        0.25x  4x bandwidth save; uchar4 vectorized loads;"
    );
    println!(
        "                                                    SM 8.9+ only; significant accuracy risk"
    );
    println!(
        "  FP8_e5m2            1        3-bit        0.25x  4x bandwidth save; wider range than e4m3"
    );
    println!(
        "                                                    (5-bit exp vs 4-bit); lower precision;"
    );
    println!(
        "                                                    SM 8.9+ only; Ada nearest to FP4 concept"
    );
    println!(
        "  INT8                1   fixed(64)         0.25x  4x bandwidth save; __dp4a momentum accel;"
    );
    println!(
        "                                                    DIST_SCALE=64; overflow risk at high density"
    );
    println!(
        "  INT4             0.5   fixed(14)         0.125x  BANDWIDTH CEILING ONLY -- NOT physics-viable;"
    );
    println!(
        "                                                    edge weights (1/36*14=0) collapse to zero;"
    );
    println!(
        "                                                    i-major SoA; 2 cells/thread (RMW race avoid)"
    );
    println!();
    println!("  SoA variants (i-major, pull-scheme, coalesced writes -- no AoS padding):");
    println!(
        "  FP16_SoA            2       10-bit       0.5x   stride=19 (vs AoS stride=20); 5% less VRAM;"
    );
    println!(
        "                                                    pull gather-reads + coalesced scatter-writes;"
    );
    println!(
        "                                                    expected 4x MLUPS vs FP32 baseline"
    );
    println!(
        "  FP8_SoA             1        4-bit       0.25x  stride=19 non-padded; best bandwidth ratio;"
    );
    println!(
        "                                                    pull-scheme eliminates AoS scatter penalty"
    );
    println!(
        "  INT8_SoA            1   fixed(64)        0.25x  stride=19 non-padded; int accumulation"
    );
    println!(
        "                                                    for momentum (no dp4a for scattered SoA loads)"
    );
    println!();
    println!("  TF32/FP16/BF16/INT8/INT4 (Tensor Core WMMA proxy -- NOT LBM kernels):");
    println!("    TF32:  M=16 N=16 K=8   ~165 TFLOPS (Ada theoretical)  FP32 accumulator");
    println!("    FP16:  M=16 N=16 K=16  ~330 TFLOPS                    FP32 accumulator");
    println!("    BF16:  M=16 N=16 K=16  ~330 TFLOPS  (same as FP16)   FP32 accumulator");
    println!("    INT8:  M=16 N=16 K=16  ~330 TOPS                      INT32 accumulator");
    println!("    INT4:  M=8  N=8  K=32  ~660 TOPS  (experimental)      INT32 accumulator");
    println!("    These paths only activate via cuBLAS/cuDNN matrix routes.");
    println!("    The WMMA benchmark quantifies the headroom over bandwidth-bound LBM.");
    println!();
    println!("Kernel variant decision table (FP32 SoA):");
    println!("  grid <= 32^3  : coarsened (instruction pressure, not bandwidth, is limit)");
    println!("  grid = 64^3   : mrt_tiled or coarsened (test both; L2 saturation varies)");
    println!("  grid = 128^3  : mrt_tiled (target >= 60% peak bandwidth per C-1304)");
    println!("  grid >= 256^3 : aa (A-A halves VRAM; stays compute-bound at same MLUPS)");
    println!("  VRAM-critical : aa + bf16 or fp16 (4x reduction vs fp64 ping-pong)");
    println!();
    println!("YSU lessons embedded in benchmark design:");
    println!("  float2/float4 widening  : standard vs coarsened MLUPS delta");
    println!("  ILP (MRT chains)        : bgk vs mrt MLUPS delta (MRT wins at 128^3+)");
    println!("  Tiling LDS vs L2        : standard vs tiled delta (target 15-25%)");
    println!("  A-A VRAM halving        : aa vs standard at 256^3 (same MLUPS, half VRAM)");
    println!("  half2 vectorized loads  : fp16 AoS vs fp32 AoS bandwidth ratio");
    println!("  __dp4a momentum accel   : int8 AoS vs fp32 AoS at same MLUPS");
    println!("  Knuth DD arithmetic     : dd vs fp64 MLUPS ratio (measures DD overhead)");
    println!();
}

// ============================================================================
// GPU workloads
// ============================================================================

#[cfg(feature = "gpu")]
mod gpu {
    use super::{BenchConfig, BenchRow, PEAK_BW_GBS, bandwidth_gbs, mlups, vram_ping_pong_mb};
    use anyhow::Result;
    use lbm_3d_cuda::{
        DarkHaloCudaSolver, LbmSolver3DCuda, Precision,
        bench_kernels::{
            BenchKernelRunner, DdBenchSolver, Fp4BenchRunner, Int4BenchRunner, SoaBenchRunner,
            TensorCoreProbe,
        },
        box_counting_gpu::GpuBoxCounter,
        probe_cuda_device_props,
        sparse::{SparseBrickMap, SparseLbmSolver},
    };
    use std::time::Instant;

    // -------------------------------------------------------------------------
    // Workload A: LBM sweep (FP32 variants, BF16, FP64)
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
            use_mrt: false,
            use_pull: false,
            use_tiling: false,
            use_coarsening: false,
            use_aa: false,
            use_smagorinsky: false,
            notes: "baseline BGK SoA ping-pong",
        },
        Fp32Variant {
            id: "mrt",
            use_mrt: true,
            use_pull: false,
            use_tiling: false,
            use_coarsening: false,
            use_aa: false,
            use_smagorinsky: false,
            notes: "d'Humieres MRT collision",
        },
        Fp32Variant {
            id: "pull",
            use_mrt: false,
            use_pull: true,
            use_tiling: false,
            use_coarsening: false,
            use_aa: false,
            use_smagorinsky: false,
            notes: "pull-scheme coalesced writes",
        },
        Fp32Variant {
            id: "mrt_pull",
            use_mrt: true,
            use_pull: true,
            use_tiling: false,
            use_coarsening: false,
            use_aa: false,
            use_smagorinsky: false,
            notes: "MRT + pull-scheme",
        },
        Fp32Variant {
            id: "tiled",
            use_mrt: false,
            use_pull: false,
            use_tiling: true,
            use_coarsening: false,
            use_aa: false,
            use_smagorinsky: false,
            notes: "8x8x4 shared-mem tile + 1-cell halo",
        },
        Fp32Variant {
            id: "mrt_tiled",
            use_mrt: true,
            use_pull: false,
            use_tiling: true,
            use_coarsening: false,
            use_aa: false,
            use_smagorinsky: false,
            notes: "MRT + 8x8x4 tile (target 60%+ peak bw)",
        },
        Fp32Variant {
            id: "coarsened",
            use_mrt: false,
            use_pull: false,
            use_tiling: false,
            use_coarsening: true,
            use_aa: false,
            use_smagorinsky: false,
            notes: "thread-coarsened float2/float4 ILP",
        },
        Fp32Variant {
            id: "mrt_coarsened",
            use_mrt: true,
            use_pull: false,
            use_tiling: false,
            use_coarsening: true,
            use_aa: false,
            use_smagorinsky: false,
            notes: "MRT + coarsened float2/float4",
        },
        Fp32Variant {
            id: "aa",
            use_mrt: false,
            use_pull: false,
            use_tiling: false,
            use_coarsening: false,
            use_aa: true,
            use_smagorinsky: false,
            notes: "A-A single-buffer half-VRAM",
        },
        Fp32Variant {
            id: "mrt_aa",
            use_mrt: true,
            use_pull: false,
            use_tiling: false,
            use_coarsening: false,
            use_aa: true,
            use_smagorinsky: false,
            notes: "MRT + A-A streaming",
        },
        Fp32Variant {
            id: "smagorinsky_mrt_tiled",
            use_mrt: true,
            use_pull: false,
            use_tiling: true,
            use_coarsening: false,
            use_aa: false,
            use_smagorinsky: true,
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
                eprint!("  [A] FP32/{:<22} {}^3 ({} steps)... ", v.id, n, n_steps);
                let mut solver = match make_fp32_solver(n, v) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("SKIP: {e}");
                        continue;
                    }
                };
                if v.use_smagorinsky {
                    let dx = 1.0 / n as f64;
                    if let Err(e) = solver.update_smagorinsky_tau(0.1, dx, 0.7) {
                        eprintln!("SKIP smagorinsky setup: {e}");
                        continue;
                    }
                }
                // Use step() loop instead of step_n() to bypass CUDA Graph capture path.
                // step_n() routes SoA variants through step_graph_pair() which calls
                // begin_capture(GLOBAL), incompatible with the benchmark's host timing model.
                let warmup_err = (0..cfg.warmup).try_for_each(|_| solver.step());
                if let Err(e) = warmup_err {
                    eprintln!("SKIP warmup: {e}");
                    continue;
                }
                if let Err(e) = solver.context().synchronize() {
                    eprintln!("SKIP sync: {e}");
                    continue;
                }
                let t0 = Instant::now();
                let step_err = (0..n_steps).try_for_each(|_| solver.step());
                if let Err(e) = step_err {
                    eprintln!("SKIP step: {e}");
                    continue;
                }
                if let Err(e) = solver.context().synchronize() {
                    eprintln!("SKIP sync: {e}");
                    continue;
                }
                let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
                let ml = mlups(n_cells, n_steps, elapsed_ms);
                let bw = bandwidth_gbs(n_cells, n_steps, elapsed_ms, elem_bytes, 19);
                let bw_pct = bw / PEAK_BW_GBS * 100.0;
                let vram_mb = vram_ping_pong_mb(n_cells, elem_bytes, 19);
                eprintln!("{ml:.1} MLUPS  {bw:.1} GB/s  ({bw_pct:.1}%)  VRAM={vram_mb} MB");
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
                    vram_dist_mb: Some(vram_mb),
                    notes: v.notes.to_string(),
                });
            }
        }

        // BF16 standard
        for &n in &grids {
            let n_steps = cfg.n_steps_for(n);
            let n_cells = n * n * n;
            let elem_bytes = 2usize;
            eprint!(
                "  [A] BF16/standard            {}^3 ({} steps)... ",
                n, n_steps
            );
            let mut solver = match LbmSolver3DCuda::new(n, n, n, 0.7, Precision::BF16) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("SKIP: {e}");
                    continue;
                }
            };
            if solver.step_n(cfg.warmup).is_err() {
                eprintln!("SKIP warmup");
                continue;
            }
            let _ = solver.context().synchronize();
            let t0 = Instant::now();
            if solver.step_n(n_steps).is_err() {
                eprintln!("SKIP step");
                continue;
            }
            let _ = solver.context().synchronize();
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
            let ml = mlups(n_cells, n_steps, elapsed_ms);
            let bw = bandwidth_gbs(n_cells, n_steps, elapsed_ms, elem_bytes, 19);
            let bw_pct = bw / PEAK_BW_GBS * 100.0;
            let vram_mb = vram_ping_pong_mb(n_cells, elem_bytes, 19);
            eprintln!("{ml:.1} MLUPS  {bw:.1} GB/s  ({bw_pct:.1}%)  VRAM={vram_mb} MB");
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
                vram_dist_mb: Some(vram_mb),
                notes:
                    "BF16 AoS; same MLUPS as FP32 (no scalar FFMA accel on Ada); 2x bw reduction"
                        .to_string(),
            });
        }

        // FP64 standard
        for &n in &grids {
            let n_steps = cfg.n_steps_for(n);
            let n_cells = n * n * n;
            let elem_bytes = 8usize;
            eprint!(
                "  [A] FP64/standard            {}^3 ({} steps)... ",
                n, n_steps
            );
            let mut solver = match LbmSolver3DCuda::new(n, n, n, 0.7, Precision::FP64) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("SKIP: {e}");
                    continue;
                }
            };
            if solver.step_n(cfg.warmup).is_err() {
                eprintln!("SKIP warmup");
                continue;
            }
            let _ = solver.context().synchronize();
            let t0 = Instant::now();
            if solver.step_n(n_steps).is_err() {
                eprintln!("SKIP step");
                continue;
            }
            let _ = solver.context().synchronize();
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
            let ml = mlups(n_cells, n_steps, elapsed_ms);
            let bw = bandwidth_gbs(n_cells, n_steps, elapsed_ms, elem_bytes, 19);
            let bw_pct = bw / PEAK_BW_GBS * 100.0;
            let vram_mb = vram_ping_pong_mb(n_cells, elem_bytes, 19);
            eprintln!("{ml:.1} MLUPS  {bw:.1} GB/s  ({bw_pct:.1}%)  VRAM={vram_mb} MB");
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
                vram_dist_mb: Some(vram_mb),
                notes: "FP64 AoS; ~0.6 TFLOPS on Ada gaming SKU; validation tier only".to_string(),
            });
        }

        Ok(rows)
    }

    // -------------------------------------------------------------------------
    // Workload H: FP16 LBM (NVRTC runtime-compiled kernel)
    // -------------------------------------------------------------------------

    pub fn run_lbm_fp16(cfg: &BenchConfig) -> Result<Vec<BenchRow>> {
        let mut rows = Vec::new();
        for &n in &cfg.grids() {
            let n_steps = cfg.n_steps_for(n);
            let n_cells = n * n * n;
            let elem_bytes = 2usize;
            eprint!(
                "  [H] FP16/standard            {}^3 ({} steps)... ",
                n, n_steps
            );
            let mut runner = match BenchKernelRunner::new_fp16(n, n, n) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("SKIP: {e}");
                    continue;
                }
            };
            if let Err(e) = runner.step_n(cfg.warmup) {
                eprintln!("SKIP warmup: {e}");
                continue;
            }
            let _ = runner.context().synchronize();
            let t0 = Instant::now();
            if let Err(e) = runner.step_n(n_steps) {
                eprintln!("SKIP step: {e}");
                continue;
            }
            let _ = runner.context().synchronize();
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
            let ml = mlups(n_cells, n_steps, elapsed_ms);
            let bw = bandwidth_gbs(n_cells, n_steps, elapsed_ms, elem_bytes, 20);
            let bw_pct = bw / PEAK_BW_GBS * 100.0;
            let vram_mb = runner.vram_dist_bytes() / (1024 * 1024);
            eprintln!("{ml:.1} MLUPS  {bw:.1} GB/s  ({bw_pct:.1}%)  VRAM={vram_mb} MB");
            rows.push(BenchRow {
                workload: "lbm_sweep",
                precision: "FP16".to_string(),
                variant: "standard".to_string(),
                grid_size: n,
                param2: String::new(),
                n_steps,
                warmup: cfg.warmup,
                elapsed_ms,
                mlups: Some(ml),
                bandwidth_gbs: Some(bw),
                bandwidth_pct_peak: Some(bw_pct),
                vram_dist_mb: Some(vram_mb),
                notes: "FP16 AoS stride=20 (padded); 10x half2 vectorized loads; FP32 compute; 2 bytes/dist; 160 MB VRAM at 128^3".to_string(),
            });
        }
        Ok(rows)
    }

    // -------------------------------------------------------------------------
    // Workload I: FP8 e4m3 LBM (SM 8.9+ only)
    // -------------------------------------------------------------------------

    pub fn run_lbm_fp8(cfg: &BenchConfig) -> Result<Vec<BenchRow>> {
        let mut rows = Vec::new();
        for &n in &cfg.grids() {
            let n_steps = cfg.n_steps_for(n);
            let n_cells = n * n * n;
            let elem_bytes = 1usize;
            eprint!(
                "  [I] FP8_e4m3/standard        {}^3 ({} steps)... ",
                n, n_steps
            );
            let mut runner = match BenchKernelRunner::new_fp8(n, n, n) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("SKIP: {e}");
                    continue;
                }
            };
            if let Err(e) = runner.step_n(cfg.warmup) {
                eprintln!("SKIP warmup: {e}");
                continue;
            }
            let _ = runner.context().synchronize();
            let t0 = Instant::now();
            if let Err(e) = runner.step_n(n_steps) {
                eprintln!("SKIP step: {e}");
                continue;
            }
            let _ = runner.context().synchronize();
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
            let ml = mlups(n_cells, n_steps, elapsed_ms);
            let bw = bandwidth_gbs(n_cells, n_steps, elapsed_ms, elem_bytes, 20);
            let bw_pct = bw / PEAK_BW_GBS * 100.0;
            let vram_mb = runner.vram_dist_bytes() / (1024 * 1024);
            eprintln!("{ml:.1} MLUPS  {bw:.1} GB/s  ({bw_pct:.1}%)  VRAM={vram_mb} MB");
            rows.push(BenchRow {
                workload: "lbm_sweep",
                precision: "FP8_e4m3".to_string(),
                variant: "standard".to_string(),
                grid_size: n,
                param2: String::new(),
                n_steps,
                warmup: cfg.warmup,
                elapsed_ms,
                mlups: Some(ml),
                bandwidth_gbs: Some(bw),
                bandwidth_pct_peak: Some(bw_pct),
                vram_dist_mb: Some(vram_mb),
                notes: "FP8 e4m3 AoS stride=20 (padded); 5x uchar4 vectorized loads; FP32 compute; SM 8.9+ only; 1 byte/dist; 80 MB VRAM at 128^3".to_string(),
            });
        }
        Ok(rows)
    }

    // -------------------------------------------------------------------------
    // Workload J: INT8 fixed-point LBM
    // -------------------------------------------------------------------------

    pub fn run_lbm_int8(cfg: &BenchConfig) -> Result<Vec<BenchRow>> {
        let mut rows = Vec::new();
        for &n in &cfg.grids() {
            let n_steps = cfg.n_steps_for(n);
            let n_cells = n * n * n;
            let elem_bytes = 1usize;
            eprint!(
                "  [J] INT8/standard            {}^3 ({} steps)... ",
                n, n_steps
            );
            let mut runner = match BenchKernelRunner::new_int8(n, n, n) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("SKIP: {e}");
                    continue;
                }
            };
            if let Err(e) = runner.step_n(cfg.warmup) {
                eprintln!("SKIP warmup: {e}");
                continue;
            }
            let _ = runner.context().synchronize();
            let t0 = Instant::now();
            if let Err(e) = runner.step_n(n_steps) {
                eprintln!("SKIP step: {e}");
                continue;
            }
            let _ = runner.context().synchronize();
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
            let ml = mlups(n_cells, n_steps, elapsed_ms);
            let bw = bandwidth_gbs(n_cells, n_steps, elapsed_ms, elem_bytes, 20);
            let bw_pct = bw / PEAK_BW_GBS * 100.0;
            let vram_mb = runner.vram_dist_bytes() / (1024 * 1024);
            eprintln!("{ml:.1} MLUPS  {bw:.1} GB/s  ({bw_pct:.1}%)  VRAM={vram_mb} MB");
            rows.push(BenchRow {
                workload: "lbm_sweep",
                precision: "INT8".to_string(),
                variant: "standard".to_string(),
                grid_size: n,
                param2: String::new(),
                n_steps,
                warmup: cfg.warmup,
                elapsed_ms,
                mlups: Some(ml),
                bandwidth_gbs: Some(bw),
                bandwidth_pct_peak: Some(bw_pct),
                vram_dist_mb: Some(vram_mb),
                notes: "INT8 fixed-point AoS stride=20 (padded); DIST_SCALE=64; 5x __dp4a momentum groups; FP32 BGK; 1 byte/dist; 80 MB VRAM at 128^3".to_string(),
            });
        }
        Ok(rows)
    }

    // -------------------------------------------------------------------------
    // Workload I-2: FP8 e5m2 LBM (SM 8.9+ only; wider range, lower precision than e4m3)
    // -------------------------------------------------------------------------

    pub fn run_lbm_fp8_e5m2(cfg: &BenchConfig) -> Result<Vec<BenchRow>> {
        let mut rows = Vec::new();
        for &n in &cfg.grids() {
            let n_steps = cfg.n_steps_for(n);
            let n_cells = n * n * n;
            let elem_bytes = 1usize;
            eprint!(
                "  [I2] FP8_e5m2/standard       {}^3 ({} steps)... ",
                n, n_steps
            );
            let mut runner = match BenchKernelRunner::new_fp8_e5m2(n, n, n) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("SKIP: {e}");
                    continue;
                }
            };
            if let Err(e) = runner.step_n(cfg.warmup) {
                eprintln!("SKIP warmup: {e}");
                continue;
            }
            let _ = runner.context().synchronize();
            let t0 = std::time::Instant::now();
            if let Err(e) = runner.step_n(n_steps) {
                eprintln!("SKIP step: {e}");
                continue;
            }
            let _ = runner.context().synchronize();
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
            let ml = mlups(n_cells, n_steps, elapsed_ms);
            // AoS stride=20 (padded, same as e4m3)
            let bw = bandwidth_gbs(n_cells, n_steps, elapsed_ms, elem_bytes, 20);
            let bw_pct = bw / PEAK_BW_GBS * 100.0;
            let vram_mb = runner.vram_dist_bytes() / (1024 * 1024);
            eprintln!("{ml:.1} MLUPS  {bw:.1} GB/s  ({bw_pct:.1}%)  VRAM={vram_mb} MB");
            rows.push(BenchRow {
                workload: "lbm_sweep",
                precision: "FP8_e5m2".to_string(),
                variant: "standard".to_string(),
                grid_size: n,
                param2: String::new(),
                n_steps,
                warmup: cfg.warmup,
                elapsed_ms,
                mlups: Some(ml),
                bandwidth_gbs: Some(bw),
                bandwidth_pct_peak: Some(bw_pct),
                vram_dist_mb: Some(vram_mb),
                notes: "FP8 e5m2 AoS stride=20; 5-bit exponent (wider range than e4m3); 2-bit mantissa (lower precision); SM 8.9+ only".to_string(),
            });
        }
        Ok(rows)
    }

    // -------------------------------------------------------------------------
    // Workload H-2: FP16 i-major SoA pull-scheme LBM
    // SoA layout eliminates AoS scatter-write non-coalescing for diagonal dirs.
    // Expected: ~4x MLUPS vs FP32 baseline (2x from bandwidth, 2x from coalescing fix).
    // -------------------------------------------------------------------------

    pub fn run_lbm_fp16_soa(cfg: &BenchConfig) -> Result<Vec<BenchRow>> {
        let mut rows = Vec::new();
        for &n in &cfg.grids() {
            let n_steps = cfg.n_steps_for(n);
            let n_cells = n * n * n;
            let elem_bytes = 2usize;
            eprint!(
                "  [H2] FP16_SoA/pull           {}^3 ({} steps)... ",
                n, n_steps
            );
            let mut runner = match SoaBenchRunner::new_fp16_soa(n, n, n) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("SKIP: {e}");
                    continue;
                }
            };
            if let Err(e) = runner.step_n(cfg.warmup) {
                eprintln!("SKIP warmup: {e}");
                continue;
            }
            let _ = runner.context().synchronize();
            let t0 = std::time::Instant::now();
            if let Err(e) = runner.step_n(n_steps) {
                eprintln!("SKIP step: {e}");
                continue;
            }
            let _ = runner.context().synchronize();
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
            let ml = mlups(n_cells, n_steps, elapsed_ms);
            // SoA stride=19 (non-padded): no AoS padding byte
            let bw = bandwidth_gbs(n_cells, n_steps, elapsed_ms, elem_bytes, 19);
            let bw_pct = bw / PEAK_BW_GBS * 100.0;
            let vram_mb = runner.vram_dist_bytes() / (1024 * 1024);
            eprintln!("{ml:.1} MLUPS  {bw:.1} GB/s  ({bw_pct:.1}%)  VRAM={vram_mb} MB");
            rows.push(BenchRow {
                workload: "lbm_sweep",
                precision: "FP16_SoA".to_string(),
                variant: "pull".to_string(),
                grid_size: n,
                param2: String::new(),
                n_steps,
                warmup: cfg.warmup,
                elapsed_ms,
                mlups: Some(ml),
                bandwidth_gbs: Some(bw),
                bandwidth_pct_peak: Some(bw_pct),
                vram_dist_mb: Some(vram_mb),
                notes: "FP16 i-major SoA pull-scheme; stride=19 non-padded; coalesced gather+scatter; 2 bytes/dist; ~152 MB VRAM at 128^3".to_string(),
            });
        }
        Ok(rows)
    }

    // -------------------------------------------------------------------------
    // Workload I-3: FP8 e4m3 i-major SoA pull-scheme LBM
    // -------------------------------------------------------------------------

    pub fn run_lbm_fp8_soa(cfg: &BenchConfig) -> Result<Vec<BenchRow>> {
        let mut rows = Vec::new();
        for &n in &cfg.grids() {
            let n_steps = cfg.n_steps_for(n);
            let n_cells = n * n * n;
            let elem_bytes = 1usize;
            eprint!(
                "  [I3] FP8_SoA/pull            {}^3 ({} steps)... ",
                n, n_steps
            );
            let mut runner = match SoaBenchRunner::new_fp8_soa(n, n, n) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("SKIP: {e}");
                    continue;
                }
            };
            if let Err(e) = runner.step_n(cfg.warmup) {
                eprintln!("SKIP warmup: {e}");
                continue;
            }
            let _ = runner.context().synchronize();
            let t0 = std::time::Instant::now();
            if let Err(e) = runner.step_n(n_steps) {
                eprintln!("SKIP step: {e}");
                continue;
            }
            let _ = runner.context().synchronize();
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
            let ml = mlups(n_cells, n_steps, elapsed_ms);
            let bw = bandwidth_gbs(n_cells, n_steps, elapsed_ms, elem_bytes, 19);
            let bw_pct = bw / PEAK_BW_GBS * 100.0;
            let vram_mb = runner.vram_dist_bytes() / (1024 * 1024);
            eprintln!("{ml:.1} MLUPS  {bw:.1} GB/s  ({bw_pct:.1}%)  VRAM={vram_mb} MB");
            rows.push(BenchRow {
                workload: "lbm_sweep",
                precision: "FP8_e4m3_SoA".to_string(),
                variant: "pull".to_string(),
                grid_size: n,
                param2: String::new(),
                n_steps,
                warmup: cfg.warmup,
                elapsed_ms,
                mlups: Some(ml),
                bandwidth_gbs: Some(bw),
                bandwidth_pct_peak: Some(bw_pct),
                vram_dist_mb: Some(vram_mb),
                notes: "FP8 e4m3 i-major SoA pull-scheme; stride=19 non-padded; SM 8.9+ only; 1 byte/dist; ~76 MB VRAM at 128^3".to_string(),
            });
        }
        Ok(rows)
    }

    // -------------------------------------------------------------------------
    // Workload J-2: INT8 i-major SoA pull-scheme LBM
    // -------------------------------------------------------------------------

    pub fn run_lbm_int8_soa(cfg: &BenchConfig) -> Result<Vec<BenchRow>> {
        let mut rows = Vec::new();
        for &n in &cfg.grids() {
            let n_steps = cfg.n_steps_for(n);
            let n_cells = n * n * n;
            let elem_bytes = 1usize;
            eprint!(
                "  [J2] INT8_SoA/pull           {}^3 ({} steps)... ",
                n, n_steps
            );
            let mut runner = match SoaBenchRunner::new_int8_soa(n, n, n) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("SKIP: {e}");
                    continue;
                }
            };
            if let Err(e) = runner.step_n(cfg.warmup) {
                eprintln!("SKIP warmup: {e}");
                continue;
            }
            let _ = runner.context().synchronize();
            let t0 = std::time::Instant::now();
            if let Err(e) = runner.step_n(n_steps) {
                eprintln!("SKIP step: {e}");
                continue;
            }
            let _ = runner.context().synchronize();
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
            let ml = mlups(n_cells, n_steps, elapsed_ms);
            let bw = bandwidth_gbs(n_cells, n_steps, elapsed_ms, elem_bytes, 19);
            let bw_pct = bw / PEAK_BW_GBS * 100.0;
            let vram_mb = runner.vram_dist_bytes() / (1024 * 1024);
            eprintln!("{ml:.1} MLUPS  {bw:.1} GB/s  ({bw_pct:.1}%)  VRAM={vram_mb} MB");
            rows.push(BenchRow {
                workload: "lbm_sweep",
                precision: "INT8_SoA".to_string(),
                variant: "pull".to_string(),
                grid_size: n,
                param2: String::new(),
                n_steps,
                warmup: cfg.warmup,
                elapsed_ms,
                mlups: Some(ml),
                bandwidth_gbs: Some(bw),
                bandwidth_pct_peak: Some(bw_pct),
                vram_dist_mb: Some(vram_mb),
                notes: "INT8 i-major SoA pull-scheme; stride=19 non-padded; DIST_SCALE=64; int momentum accumulation; 1 byte/dist; ~76 MB VRAM at 128^3".to_string(),
            });
        }
        Ok(rows)
    }

    // -------------------------------------------------------------------------
    // Workload L: INT4 nibble-packed LBM -- BANDWIDTH CEILING ONLY
    // Physics NOT viable (edge weights quantize to zero); reports max achievable
    // MLUPS when distributions are 2x denser than INT8.
    // -------------------------------------------------------------------------

    pub fn run_lbm_int4(cfg: &BenchConfig) -> Result<Vec<BenchRow>> {
        let mut rows = Vec::new();
        for &n in &cfg.grids() {
            let n_steps = cfg.n_steps_for(n);
            let n_cells = n * n * n;
            let half_cells = n_cells / 2;
            eprint!(
                "  [L]  INT4/bw_ceil            {}^3 ({} steps)... ",
                n, n_steps
            );
            let mut runner = match Int4BenchRunner::new(n, n, n) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("SKIP: {e}");
                    continue;
                }
            };
            if let Err(e) = runner.step_n(cfg.warmup) {
                eprintln!("SKIP warmup: {e}");
                continue;
            }
            let _ = runner.context().synchronize();
            let t0 = std::time::Instant::now();
            if let Err(e) = runner.step_n(n_steps) {
                eprintln!("SKIP step: {e}");
                continue;
            }
            let _ = runner.context().synchronize();
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
            let ml = mlups(n_cells, n_steps, elapsed_ms);
            // INT4 bandwidth: 19 directions * half_cells bytes/step (ping-pong each)
            // plus 8 macro fields (rho,ux,uy,uz,tau,fx,fy,fz) * n_cells * 4 bytes
            let bytes_per_step = (19.0 * half_cells as f64 * 2.0  // dist read + write
                                  + 8.0 * n_cells as f64 * 4.0)   // macro FP32 fields
                                 * n_steps as f64;
            let bw = bytes_per_step / (elapsed_ms * 1e-3) / 1e9;
            let bw_pct = bw / PEAK_BW_GBS * 100.0;
            let vram_mb = runner.vram_dist_bytes() / (1024 * 1024);
            eprintln!("{ml:.1} MLUPS  {bw:.1} GB/s  ({bw_pct:.1}%)  VRAM={vram_mb} MB");
            rows.push(BenchRow {
                workload: "lbm_sweep",
                precision: "INT4".to_string(),
                variant: "bw_ceiling".to_string(),
                grid_size: n,
                param2: String::new(),
                n_steps,
                warmup: cfg.warmup,
                elapsed_ms,
                mlups: Some(ml),
                bandwidth_gbs: Some(bw),
                bandwidth_pct_peak: Some(bw_pct),
                vram_dist_mb: Some(vram_mb),
                notes: "BANDWIDTH CEILING ONLY -- physics broken (edge weights=0); nibble-packed i-major SoA; 2 cells/thread; 0.5 bytes/dist; ~38 MB VRAM at 128^3".to_string(),
            });
        }
        Ok(rows)
    }

    // -------------------------------------------------------------------------
    // Workload J-3: INT16 AoS stride-20 LBM
    // 2 bytes/dist; DIST_SCALE=16384 -> LSB=6.1e-5 (vs INT8 LSB=0.016).
    // Better precision than INT8 for moderate-Re flows; same VRAM cost as FP16.
    // -------------------------------------------------------------------------

    pub fn run_lbm_int16(cfg: &BenchConfig) -> Result<Vec<BenchRow>> {
        let mut rows = Vec::new();
        for &n in &cfg.grids() {
            let n_steps = cfg.n_steps_for(n);
            let n_cells = n * n * n;
            let elem_bytes = 2usize;
            eprint!(
                "  [J3] INT16/standard          {}^3 ({} steps)... ",
                n, n_steps
            );
            let mut runner = match BenchKernelRunner::new_int16(n, n, n) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("SKIP: {e}");
                    continue;
                }
            };
            if let Err(e) = runner.step_n(cfg.warmup) {
                eprintln!("SKIP warmup: {e}");
                continue;
            }
            let _ = runner.context().synchronize();
            let t0 = std::time::Instant::now();
            if let Err(e) = runner.step_n(n_steps) {
                eprintln!("SKIP step: {e}");
                continue;
            }
            let _ = runner.context().synchronize();
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
            let ml = mlups(n_cells, n_steps, elapsed_ms);
            let bw = bandwidth_gbs(n_cells, n_steps, elapsed_ms, elem_bytes, 20);
            let bw_pct = bw / PEAK_BW_GBS * 100.0;
            let vram_mb = runner.vram_dist_bytes() / (1024 * 1024);
            eprintln!("{ml:.1} MLUPS  {bw:.1} GB/s  ({bw_pct:.1}%)  VRAM={vram_mb} MB");
            rows.push(BenchRow {
                workload: "lbm_sweep",
                precision: "INT16".to_string(),
                variant: "standard".to_string(),
                grid_size: n,
                param2: String::new(),
                n_steps,
                warmup: cfg.warmup,
                elapsed_ms,
                mlups: Some(ml),
                bandwidth_gbs: Some(bw),
                bandwidth_pct_peak: Some(bw_pct),
                vram_dist_mb: Some(vram_mb),
                notes: "INT16 AoS stride=20; DIST_SCALE=16384; LSB=6.1e-5; better precision than INT8 for moderate-Re; 2 bytes/dist".to_string(),
            });
        }
        Ok(rows)
    }

    // -------------------------------------------------------------------------
    // Workload J-4: INT16 i-major SoA pull-scheme LBM
    // -------------------------------------------------------------------------

    pub fn run_lbm_int16_soa(cfg: &BenchConfig) -> Result<Vec<BenchRow>> {
        let mut rows = Vec::new();
        for &n in &cfg.grids() {
            let n_steps = cfg.n_steps_for(n);
            let n_cells = n * n * n;
            let elem_bytes = 2usize;
            eprint!(
                "  [J4] INT16_SoA/pull          {}^3 ({} steps)... ",
                n, n_steps
            );
            let mut runner = match SoaBenchRunner::new_int16_soa(n, n, n) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("SKIP: {e}");
                    continue;
                }
            };
            if let Err(e) = runner.step_n(cfg.warmup) {
                eprintln!("SKIP warmup: {e}");
                continue;
            }
            let _ = runner.context().synchronize();
            let t0 = std::time::Instant::now();
            if let Err(e) = runner.step_n(n_steps) {
                eprintln!("SKIP step: {e}");
                continue;
            }
            let _ = runner.context().synchronize();
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
            let ml = mlups(n_cells, n_steps, elapsed_ms);
            let bw = bandwidth_gbs(n_cells, n_steps, elapsed_ms, elem_bytes, 19);
            let bw_pct = bw / PEAK_BW_GBS * 100.0;
            let vram_mb = runner.vram_dist_bytes() / (1024 * 1024);
            eprintln!("{ml:.1} MLUPS  {bw:.1} GB/s  ({bw_pct:.1}%)  VRAM={vram_mb} MB");
            rows.push(BenchRow {
                workload: "lbm_sweep",
                precision: "INT16_SoA".to_string(),
                variant: "pull".to_string(),
                grid_size: n,
                param2: String::new(),
                n_steps,
                warmup: cfg.warmup,
                elapsed_ms,
                mlups: Some(ml),
                bandwidth_gbs: Some(bw),
                bandwidth_pct_peak: Some(bw_pct),
                vram_dist_mb: Some(vram_mb),
                notes: "INT16 i-major SoA pull-scheme; stride=19 non-padded; int momentum accumulation; 2 bytes/dist".to_string(),
            });
        }
        Ok(rows)
    }

    // -------------------------------------------------------------------------
    // Workload H-3: BF16 i-major SoA pull-scheme LBM (SM 8.0+)
    // Same scalar speed as FP32 on Ada (no BF16 FFMA accel); 2x BW reduction.
    // 8-bit exponent (same range as FP32) vs FP16 5-bit exponent.
    // -------------------------------------------------------------------------

    pub fn run_lbm_bf16_soa(cfg: &BenchConfig) -> Result<Vec<BenchRow>> {
        let mut rows = Vec::new();
        for &n in &cfg.grids() {
            let n_steps = cfg.n_steps_for(n);
            let n_cells = n * n * n;
            let elem_bytes = 2usize;
            eprint!(
                "  [H3] BF16_SoA/pull           {}^3 ({} steps)... ",
                n, n_steps
            );
            let mut runner = match SoaBenchRunner::new_bf16_soa(n, n, n) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("SKIP: {e}");
                    continue;
                }
            };
            if let Err(e) = runner.step_n(cfg.warmup) {
                eprintln!("SKIP warmup: {e}");
                continue;
            }
            let _ = runner.context().synchronize();
            let t0 = std::time::Instant::now();
            if let Err(e) = runner.step_n(n_steps) {
                eprintln!("SKIP step: {e}");
                continue;
            }
            let _ = runner.context().synchronize();
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
            let ml = mlups(n_cells, n_steps, elapsed_ms);
            let bw = bandwidth_gbs(n_cells, n_steps, elapsed_ms, elem_bytes, 19);
            let bw_pct = bw / PEAK_BW_GBS * 100.0;
            let vram_mb = runner.vram_dist_bytes() / (1024 * 1024);
            eprintln!("{ml:.1} MLUPS  {bw:.1} GB/s  ({bw_pct:.1}%)  VRAM={vram_mb} MB");
            rows.push(BenchRow {
                workload: "lbm_sweep",
                precision: "BF16_SoA".to_string(),
                variant: "pull".to_string(),
                grid_size: n,
                param2: String::new(),
                n_steps,
                warmup: cfg.warmup,
                elapsed_ms,
                mlups: Some(ml),
                bandwidth_gbs: Some(bw),
                bandwidth_pct_peak: Some(bw_pct),
                vram_dist_mb: Some(vram_mb),
                notes: "BF16 i-major SoA pull; stride=19; 8-bit exp (FP32 range); 7-bit mantissa; SM 8.0+; 2 bytes/dist; no BF16 FFMA accel on Ada".to_string(),
            });
        }
        Ok(rows)
    }

    // -------------------------------------------------------------------------
    // Workload I-4: FP8 e5m2 i-major SoA pull-scheme LBM (SM 8.9+)
    // -------------------------------------------------------------------------

    pub fn run_lbm_fp8_e5m2_soa(cfg: &BenchConfig) -> Result<Vec<BenchRow>> {
        let mut rows = Vec::new();
        for &n in &cfg.grids() {
            let n_steps = cfg.n_steps_for(n);
            let n_cells = n * n * n;
            let elem_bytes = 1usize;
            eprint!(
                "  [I4] FP8_e5m2_SoA/pull       {}^3 ({} steps)... ",
                n, n_steps
            );
            let mut runner = match SoaBenchRunner::new_fp8_e5m2_soa(n, n, n) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("SKIP: {e}");
                    continue;
                }
            };
            if let Err(e) = runner.step_n(cfg.warmup) {
                eprintln!("SKIP warmup: {e}");
                continue;
            }
            let _ = runner.context().synchronize();
            let t0 = std::time::Instant::now();
            if let Err(e) = runner.step_n(n_steps) {
                eprintln!("SKIP step: {e}");
                continue;
            }
            let _ = runner.context().synchronize();
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
            let ml = mlups(n_cells, n_steps, elapsed_ms);
            let bw = bandwidth_gbs(n_cells, n_steps, elapsed_ms, elem_bytes, 19);
            let bw_pct = bw / PEAK_BW_GBS * 100.0;
            let vram_mb = runner.vram_dist_bytes() / (1024 * 1024);
            eprintln!("{ml:.1} MLUPS  {bw:.1} GB/s  ({bw_pct:.1}%)  VRAM={vram_mb} MB");
            rows.push(BenchRow {
                workload: "lbm_sweep",
                precision: "FP8_e5m2_SoA".to_string(),
                variant: "pull".to_string(),
                grid_size: n,
                param2: String::new(),
                n_steps,
                warmup: cfg.warmup,
                elapsed_ms,
                mlups: Some(ml),
                bandwidth_gbs: Some(bw),
                bandwidth_pct_peak: Some(bw_pct),
                vram_dist_mb: Some(vram_mb),
                notes: "FP8 e5m2 i-major SoA pull; stride=19; 5-bit exp (wider range than e4m3); 2-bit mantissa; SM 8.9+ only; 1 byte/dist; ~38 MB VRAM at 128^3".to_string(),
            });
        }
        Ok(rows)
    }

    // -------------------------------------------------------------------------
    // Workload A-2: FP64 i-major SoA pull-scheme LBM
    // Primary use: numerical validation. Compute-bound on Ada gaming SKU.
    // Capped at 128^3: 128^3 ping-pong = ~636 MB; 256^3 = ~5 GB.
    // -------------------------------------------------------------------------

    pub fn run_lbm_fp64_soa(cfg: &BenchConfig) -> Result<Vec<BenchRow>> {
        let mut rows = Vec::new();
        // 256^3 would require 5 GB just for distribution buffers; cap at 128^3.
        let grids: Vec<usize> = cfg.grids().into_iter().filter(|&n| n <= 128).collect();
        for n in grids {
            let n_steps = cfg.n_steps_for(n).min(30); // FP64 is very slow on Ada
            let n_cells = n * n * n;
            let elem_bytes = 8usize;
            eprint!(
                "  [A2] FP64_SoA/pull           {}^3 ({} steps)... ",
                n, n_steps
            );
            let mut runner = match SoaBenchRunner::new_fp64_soa(n, n, n) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("SKIP: {e}");
                    continue;
                }
            };
            if let Err(e) = runner.step_n(cfg.warmup.min(5)) {
                eprintln!("SKIP warmup: {e}");
                continue;
            }
            let _ = runner.context().synchronize();
            let t0 = std::time::Instant::now();
            if let Err(e) = runner.step_n(n_steps) {
                eprintln!("SKIP step: {e}");
                continue;
            }
            let _ = runner.context().synchronize();
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
            let ml = mlups(n_cells, n_steps, elapsed_ms);
            let bw = bandwidth_gbs(n_cells, n_steps, elapsed_ms, elem_bytes, 19);
            let bw_pct = bw / PEAK_BW_GBS * 100.0;
            let vram_mb = runner.vram_dist_bytes() / (1024 * 1024);
            eprintln!("{ml:.1} MLUPS  {bw:.1} GB/s  ({bw_pct:.1}%)  VRAM={vram_mb} MB");
            rows.push(BenchRow {
                workload: "lbm_sweep",
                precision: "FP64_SoA".to_string(),
                variant: "pull".to_string(),
                grid_size: n,
                param2: String::new(),
                n_steps,
                warmup: cfg.warmup.min(5),
                elapsed_ms,
                mlups: Some(ml),
                bandwidth_gbs: Some(bw),
                bandwidth_pct_peak: Some(bw_pct),
                vram_dist_mb: Some(vram_mb),
                notes: "FP64 i-major SoA pull; stride=19; 53-bit mantissa; ~0.6 TFLOPS on Ada gaming SKU; compute-bound; validation tier only; 8 bytes/dist".to_string(),
            });
        }
        Ok(rows)
    }

    // -------------------------------------------------------------------------
    // Workload A-3: FP32 i-major SoA with Ada cache-streaming stores (__stcs)
    // Ping reads: __ldg() (read-only path). Pong writes: __stcs() (L2 evict-first).
    // Expected 5-15% improvement vs FP32_SoA standard at 128^3 on Ada SM 8.9.
    // -------------------------------------------------------------------------

    pub fn run_lbm_fp32_cs(cfg: &BenchConfig) -> Result<Vec<BenchRow>> {
        let mut rows = Vec::new();
        for &n in &cfg.grids() {
            let n_steps = cfg.n_steps_for(n);
            let n_cells = n * n * n;
            let elem_bytes = 4usize;
            eprint!(
                "  [A3] FP32_SoA_CS/pull        {}^3 ({} steps)... ",
                n, n_steps
            );
            let mut runner = match SoaBenchRunner::new_fp32_cs(n, n, n) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("SKIP: {e}");
                    continue;
                }
            };
            if let Err(e) = runner.step_n(cfg.warmup) {
                eprintln!("SKIP warmup: {e}");
                continue;
            }
            let _ = runner.context().synchronize();
            let t0 = std::time::Instant::now();
            if let Err(e) = runner.step_n(n_steps) {
                eprintln!("SKIP step: {e}");
                continue;
            }
            let _ = runner.context().synchronize();
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
            let ml = mlups(n_cells, n_steps, elapsed_ms);
            let bw = bandwidth_gbs(n_cells, n_steps, elapsed_ms, elem_bytes, 19);
            let bw_pct = bw / PEAK_BW_GBS * 100.0;
            let vram_mb = runner.vram_dist_bytes() / (1024 * 1024);
            eprintln!("{ml:.1} MLUPS  {bw:.1} GB/s  ({bw_pct:.1}%)  VRAM={vram_mb} MB");
            rows.push(BenchRow {
                workload: "lbm_sweep",
                precision: "FP32_SoA_CS".to_string(),
                variant: "pull_cs".to_string(),
                grid_size: n,
                param2: String::new(),
                n_steps,
                warmup: cfg.warmup,
                elapsed_ms,
                mlups: Some(ml),
                bandwidth_gbs: Some(bw),
                bandwidth_pct_peak: Some(bw_pct),
                vram_dist_mb: Some(vram_mb),
                notes: "FP32 SoA pull + __stcs pong writes (L2 evict-first); __ldg ping reads; pong does not evict ping from L2; Ada SM 8.9 specific; expected +5-15% vs baseline".to_string(),
            });
        }
        Ok(rows)
    }

    // -------------------------------------------------------------------------
    // Workload H-4: FP16 i-major SoA with half2 ILP (2 cells per thread)
    // Uses __half2 for velocity moment accumulation (2 FP16 FMAs per instruction).
    // Halves thread count; BGK collision remains in FP32 for stability.
    // -------------------------------------------------------------------------

    pub fn run_lbm_fp16_half2(cfg: &BenchConfig) -> Result<Vec<BenchRow>> {
        let mut rows = Vec::new();
        for &n in &cfg.grids() {
            let n_steps = cfg.n_steps_for(n);
            let n_cells = n * n * n;
            let elem_bytes = 2usize;
            eprint!(
                "  [H4] FP16_SoA_H2/half2_ilp   {}^3 ({} steps)... ",
                n, n_steps
            );
            let mut runner = match SoaBenchRunner::new_fp16_half2(n, n, n) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("SKIP: {e}");
                    continue;
                }
            };
            if let Err(e) = runner.step_n(cfg.warmup) {
                eprintln!("SKIP warmup: {e}");
                continue;
            }
            let _ = runner.context().synchronize();
            let t0 = std::time::Instant::now();
            if let Err(e) = runner.step_n(n_steps) {
                eprintln!("SKIP step: {e}");
                continue;
            }
            let _ = runner.context().synchronize();
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
            let ml = mlups(n_cells, n_steps, elapsed_ms);
            let bw = bandwidth_gbs(n_cells, n_steps, elapsed_ms, elem_bytes, 19);
            let bw_pct = bw / PEAK_BW_GBS * 100.0;
            let vram_mb = runner.vram_dist_bytes() / (1024 * 1024);
            eprintln!("{ml:.1} MLUPS  {bw:.1} GB/s  ({bw_pct:.1}%)  VRAM={vram_mb} MB");
            rows.push(BenchRow {
                workload: "lbm_sweep",
                precision: "FP16_SoA_H2".to_string(),
                variant: "half2_ilp".to_string(),
                grid_size: n,
                param2: String::new(),
                n_steps,
                warmup: cfg.warmup,
                elapsed_ms,
                mlups: Some(ml),
                bandwidth_gbs: Some(bw),
                bandwidth_pct_peak: Some(bw_pct),
                vram_dist_mb: Some(vram_mb),
                notes: "FP16 SoA half2 ILP; 2 cells/thread; __half2 moment accum (2x FMA/insn); FP32 BGK collision; 50% thread count vs standard FP16_SoA".to_string(),
            });
        }
        Ok(rows)
    }

    // -------------------------------------------------------------------------
    // Workload L-2: FP4 E2M1 nibble-packed LBM -- BANDWIDTH CEILING ONLY
    // FP4 hardware is Blackwell SM 10.0+ only. On Ada: emulated via nibble pack
    // (same layout as INT4 but FP4 E2M1 decode table). Physics NOT viable.
    // -------------------------------------------------------------------------

    pub fn run_lbm_fp4(cfg: &BenchConfig) -> Result<Vec<BenchRow>> {
        let mut rows = Vec::new();
        for &n in &cfg.grids() {
            let n_steps = cfg.n_steps_for(n);
            let n_cells = n * n * n;
            let half_cells = n_cells.div_ceil(2);
            eprint!(
                "  [L2] FP4/bw_ceil             {}^3 ({} steps)... ",
                n, n_steps
            );
            let mut runner = match Fp4BenchRunner::new(n, n, n) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("SKIP: {e}");
                    continue;
                }
            };
            if let Err(e) = runner.step_n(cfg.warmup) {
                eprintln!("SKIP warmup: {e}");
                continue;
            }
            let _ = runner.context().synchronize();
            let t0 = std::time::Instant::now();
            if let Err(e) = runner.step_n(n_steps) {
                eprintln!("SKIP step: {e}");
                continue;
            }
            let _ = runner.context().synchronize();
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
            let ml = mlups(n_cells, n_steps, elapsed_ms);
            // FP4: same nibble model as INT4 (19 bytes per cell-pair, ping+pong)
            let bytes_per_step =
                (19.0 * half_cells as f64 * 2.0 + 8.0 * n_cells as f64 * 4.0) * n_steps as f64;
            let bw = bytes_per_step / (elapsed_ms * 1e-3) / 1e9;
            let bw_pct = bw / PEAK_BW_GBS * 100.0;
            let vram_mb = runner.vram_dist_bytes() / (1024 * 1024);
            eprintln!("{ml:.1} MLUPS  {bw:.1} GB/s  ({bw_pct:.1}%)  VRAM={vram_mb} MB");
            rows.push(BenchRow {
                workload: "lbm_sweep",
                precision: "FP4_E2M1".to_string(),
                variant: "bw_ceiling".to_string(),
                grid_size: n,
                param2: String::new(),
                n_steps,
                warmup: cfg.warmup,
                elapsed_ms,
                mlups: Some(ml),
                bandwidth_gbs: Some(bw),
                bandwidth_pct_peak: Some(bw_pct),
                vram_dist_mb: Some(vram_mb),
                notes: "BANDWIDTH CEILING ONLY -- physics broken (rest weight 1/3 -> 0.5, 50% error); FP4 E2M1 emulated on Ada (Blackwell HW only); nibble-packed i-major SoA; 2 cells/thread; ~38 MB VRAM at 128^3".to_string(),
            });
        }
        Ok(rows)
    }

    // -------------------------------------------------------------------------
    // Workload K: Double-Double FP128 LBM (capped at 64^3)
    // -------------------------------------------------------------------------

    pub fn run_lbm_dd(cfg: &BenchConfig) -> Result<Vec<BenchRow>> {
        let mut rows = Vec::new();
        // DD is FP64 compute -- extremely slow on Ada gaming SKU. Cap grid size.
        let dd_grids: Vec<usize> = cfg.grids().into_iter().filter(|&n| n <= 64).collect();
        for n in dd_grids {
            // Use shorter warmup and step counts to avoid multi-minute waits.
            let n_steps = cfg.n_steps_for(n).min(20);
            let warmup = cfg.warmup.min(5);
            let n_cells = n * n * n;
            // Each distribution value is 16 bytes (8-byte hi + 8-byte lo).
            // Ping-pong uses 4 buffers (hi_a, lo_a, hi_b, lo_b).
            let elem_bytes = 16usize;
            eprint!(
                "  [K] DD_FP128/standard        {}^3 ({} steps)... ",
                n, n_steps
            );
            let mut solver = match DdBenchSolver::new(n, n, n) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("SKIP: {e}");
                    continue;
                }
            };
            if let Err(e) = solver.step_n(warmup) {
                eprintln!("SKIP warmup: {e}");
                continue;
            }
            let _ = solver.context().synchronize();
            let t0 = Instant::now();
            if let Err(e) = solver.step_n(n_steps) {
                eprintln!("SKIP step: {e}");
                continue;
            }
            let _ = solver.context().synchronize();
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
            let ml = mlups(n_cells, n_steps, elapsed_ms);
            let bw = bandwidth_gbs(n_cells, n_steps, elapsed_ms, elem_bytes, 19);
            let bw_pct = bw / PEAK_BW_GBS * 100.0;
            // DD uses 4 i-major SoA buffers (hi_a, lo_a, hi_b, lo_b), each 19*n_cells*8 bytes.
            let vram_mb = solver.vram_dist_bytes() / (1024 * 1024);
            eprintln!("{ml:.1} MLUPS  {bw:.1} GB/s  ({bw_pct:.1}%)  VRAM={vram_mb} MB");
            rows.push(BenchRow {
                workload: "lbm_sweep",
                precision: "DD_FP128".to_string(),
                variant: "standard".to_string(),
                grid_size: n,
                param2: String::new(),
                n_steps,
                warmup,
                elapsed_ms,
                mlups: Some(ml),
                bandwidth_gbs: Some(bw),
                bandwidth_pct_peak: Some(bw_pct),
                vram_dist_mb: Some(vram_mb),
                notes: "Double-double FP128 emulation; Knuth 2-sum + Veltkamp/Dekker FMA; i-major SoA (coalesced); 4 bufs; 16 bytes/dist".to_string(),
            });
        }
        Ok(rows)
    }

    // -------------------------------------------------------------------------
    // Workload TC: Tensor Core WMMA proxy (measures headroom vs LBM bandwidth)
    // -------------------------------------------------------------------------

    pub fn run_tensor_core(_cfg: &BenchConfig) -> Result<Vec<BenchRow>> {
        let mut rows = Vec::new();
        eprint!("  [TC] TensorCore WMMA proxy... ");

        let mut probe = match TensorCoreProbe::new() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("SKIP: {e}");
                return Ok(rows);
            }
        };

        // Warmup: brief run of each available tier (not timed).
        let _ = probe.run_tf32(100);
        let _ = probe.run_fp16(100);
        let _ = probe.run_int8(100);
        let _ = probe.run_int4(100);
        let _ = probe.context().synchronize();

        let n_iters = 10_000_i32;

        // TF32 (M=16, N=16, K=8)
        if probe.has_tf32() {
            let t0 = Instant::now();
            if let Ok(flops) = probe.run_tf32(n_iters) {
                let _ = probe.context().synchronize();
                let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
                let gflops = flops / (elapsed_ms * 1e-3) / 1e9;
                eprintln!("TF32={gflops:.0} GFLOPS");
                rows.push(BenchRow {
                    workload: "tensor_core",
                    precision: "TF32".to_string(),
                    variant: "wmma_16x16x8".to_string(),
                    grid_size: 0,
                    param2: format!("n_iters={n_iters}"),
                    n_steps: n_iters as usize,
                    warmup: 100,
                    elapsed_ms,
                    mlups: None,
                    bandwidth_gbs: None,
                    bandwidth_pct_peak: None,
                    vram_dist_mb: None,
                    notes: format!("{gflops:.0} GFLOPS; M=16 N=16 K=8 FP32 accum; Ada ~165 TFLOPS peak; NOT an LBM kernel"),
                });
            }
        }

        // FP16 (M=16, N=16, K=16)
        if probe.has_fp16() {
            eprint!("  [TC] FP16 ");
            let t0 = Instant::now();
            if let Ok(flops) = probe.run_fp16(n_iters) {
                let _ = probe.context().synchronize();
                let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
                let gflops = flops / (elapsed_ms * 1e-3) / 1e9;
                eprintln!("{gflops:.0} GFLOPS");
                rows.push(BenchRow {
                    workload: "tensor_core",
                    precision: "FP16".to_string(),
                    variant: "wmma_16x16x16".to_string(),
                    grid_size: 0,
                    param2: format!("n_iters={n_iters}"),
                    n_steps: n_iters as usize,
                    warmup: 100,
                    elapsed_ms,
                    mlups: None,
                    bandwidth_gbs: None,
                    bandwidth_pct_peak: None,
                    vram_dist_mb: None,
                    notes: format!("{gflops:.0} GFLOPS; M=16 N=16 K=16 FP32 accum; Ada ~330 TFLOPS peak; NOT an LBM kernel"),
                });
            }
        }

        // INT8 (M=16, N=16, K=16)
        if probe.has_int8() {
            eprint!("  [TC] INT8 ");
            let t0 = Instant::now();
            if let Ok(ops) = probe.run_int8(n_iters) {
                let _ = probe.context().synchronize();
                let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
                let gops = ops / (elapsed_ms * 1e-3) / 1e9;
                eprintln!("{gops:.0} GOPS");
                rows.push(BenchRow {
                    workload: "tensor_core",
                    precision: "INT8".to_string(),
                    variant: "wmma_16x16x16".to_string(),
                    grid_size: 0,
                    param2: format!("n_iters={n_iters}"),
                    n_steps: n_iters as usize,
                    warmup: 100,
                    elapsed_ms,
                    mlups: None,
                    bandwidth_gbs: None,
                    bandwidth_pct_peak: None,
                    vram_dist_mb: None,
                    notes: format!("{gops:.0} GOPS; M=16 N=16 K=16 INT32 accum; Ada ~330 TOPS peak; NOT an LBM kernel"),
                });
            }
        }

        // INT4 (M=8, N=8, K=32, experimental)
        if probe.has_int4() {
            eprint!("  [TC] INT4 ");
            let t0 = Instant::now();
            if let Ok(ops) = probe.run_int4(n_iters) {
                let _ = probe.context().synchronize();
                let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
                let gops = ops / (elapsed_ms * 1e-3) / 1e9;
                eprintln!("{gops:.0} GOPS");
                rows.push(BenchRow {
                    workload: "tensor_core",
                    precision: "INT4".to_string(),
                    variant: "wmma_8x8x32".to_string(),
                    grid_size: 0,
                    param2: format!("n_iters={n_iters}"),
                    n_steps: n_iters as usize,
                    warmup: 100,
                    elapsed_ms,
                    mlups: None,
                    bandwidth_gbs: None,
                    bandwidth_pct_peak: None,
                    vram_dist_mb: None,
                    notes: format!("{gops:.0} GOPS; M=8 N=8 K=32 INT32 accum; Ada ~660 TOPS peak experimental; NOT an LBM kernel"),
                });
            }
        }

        // BF16 (M=16, N=16, K=16) -- same shape as FP16; Ada parity ~330 TFLOPS
        if probe.has_bf16() {
            eprint!("  [TC] BF16 ");
            let t0 = Instant::now();
            if let Ok(flops) = probe.run_bf16(n_iters) {
                let _ = probe.context().synchronize();
                let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
                let gflops = flops / (elapsed_ms * 1e-3) / 1e9;
                eprintln!("{gflops:.0} GFLOPS");
                rows.push(BenchRow {
                    workload: "tensor_core",
                    precision: "BF16".to_string(),
                    variant: "wmma_16x16x16".to_string(),
                    grid_size: 0,
                    param2: format!("n_iters={n_iters}"),
                    n_steps: n_iters as usize,
                    warmup: 100,
                    elapsed_ms,
                    mlups: None,
                    bandwidth_gbs: None,
                    bandwidth_pct_peak: None,
                    vram_dist_mb: None,
                    notes: format!("{gflops:.0} GFLOPS; M=16 N=16 K=16 FP32 accum; Ada ~330 TFLOPS (same as FP16 TC); NOT an LBM kernel"),
                });
            }
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
            let mut solver = match LbmSolver3DCuda::new(n, n, n, 0.7, Precision::FP32) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("SKIP (solver init): {e}");
                    continue;
                }
            };
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
                Err(e) => {
                    eprintln!("SKIP (counter init): {e}");
                    continue;
                }
            };

            let t0 = Instant::now();
            let result = match counter.fractal_dimension_device_auto(solver.d_rho_bytes(), n, n, n)
            {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("SKIP (box-count): {e}");
                    continue;
                }
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
                vram_dist_mb: None,
                notes: format!(
                    "D_f={:.4}; R2={:.4}; warp_ballot 32x fewer atomics",
                    result.d_f, result.r_squared
                ),
            });
        }

        Ok(rows)
    }

    // -------------------------------------------------------------------------
    // Workload C: Chingon AVT contraction (skip -- requires PackedAvt)
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
            vram_dist_mb: None,
            notes: "Chingon requires PackedAvt from gororoba_algebra; not wired in this binary"
                .to_string(),
        }]
    }

    // -------------------------------------------------------------------------
    // Workload D: Dark halo fused kernel
    // -------------------------------------------------------------------------

    pub fn run_dark_halo(cfg: &BenchConfig) -> Result<Vec<BenchRow>> {
        let mut rows = Vec::new();
        let bench_grids: Vec<usize> = cfg.grids().into_iter().filter(|&n| n <= 128).collect();

        for n in bench_grids {
            let n_cells = n * n * n;
            eprint!("  [D] dark_halo {}^3... ", n);

            let mut solver = match DarkHaloCudaSolver::new(n, n, n) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("SKIP: {e}");
                    continue;
                }
            };

            let warmup_steps = cfg.warmup.min(20) as u32;
            if let Err(e) = solver.run_k_value(256, warmup_steps, 0.7, 0.1, 1.5, 1e-4, 0.0, 0) {
                eprintln!("SKIP (warmup): {e}");
                continue;
            }

            let n_steps = cfg.n_steps_for(n).min(50) as u32;
            let t0 = Instant::now();
            if let Err(e) = solver.run_k_value(256, n_steps, 0.7, 0.1, 1.5, 1e-4, 0.0, 0) {
                eprintln!("SKIP (bench): {e}");
                continue;
            }
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;

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
                vram_dist_mb: None,
                notes: "fused lbm_step_soa + ZD viscosity modulation + dark_halo_detector; k=256"
                    .to_string(),
            });
        }

        Ok(rows)
    }

    // -------------------------------------------------------------------------
    // Workload E: Sparse brick-map LBM
    // -------------------------------------------------------------------------

    pub fn run_sparse_lbm(cfg: &BenchConfig) -> Result<Vec<BenchRow>> {
        let mut rows = Vec::new();
        let sparse_grids = [256usize];
        let occupancies = [0.10f64, 0.25, 0.50];

        for &n in &sparse_grids {
            let n_cells_total = n * n * n;

            if let Some(props) = probe_cuda_device_props() {
                let _ = props.l2_bytes;
            }

            for &occ in &occupancies {
                eprint!("  [E] sparse_lbm {}^3 occ={:.0}%... ", n, occ * 100.0);

                let bootstrap = match LbmSolver3DCuda::new(8, 8, 8, 0.7, Precision::FP32) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("SKIP (bootstrap): {e}");
                        continue;
                    }
                };
                let ctx = bootstrap.context().clone();
                let stream = bootstrap.stream().clone();
                drop(bootstrap);

                let n_occupied = (occ * n_cells_total as f64).round() as usize;
                let mut mask_host = vec![0u8; n_cells_total];
                for v in mask_host.iter_mut().take(n_occupied) {
                    *v = 1;
                }

                let d_mask = match stream.clone_htod(&mask_host) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("SKIP (mask upload): {e}");
                        continue;
                    }
                };
                drop(mask_host);

                let bm = match SparseBrickMap::new_from_geometry(
                    ctx.clone(),
                    stream.clone(),
                    n,
                    n,
                    n,
                    &d_mask,
                ) {
                    Ok(b) => b,
                    Err(e) => {
                        eprintln!("SKIP (brick map): {e}");
                        continue;
                    }
                };
                drop(d_mask);

                let n_active = bm.n_active_bricks;
                let n_active_cells = n_active * 512;
                let vram_mb =
                    (n_active_cells * 19 * 4 * 2 + n_active_cells * 4 * 5) / (1024 * 1024);

                let mut sparse_solver = match SparseLbmSolver::new(bm) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("SKIP (sparse solver): {e}");
                        continue;
                    }
                };

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
                let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;

                let ml = mlups(n_active_cells, n_steps, elapsed_ms);
                let sparse_overhead_pct =
                    (n_cells_total as f64 - n_active_cells as f64) / n_cells_total as f64 * 100.0;
                eprintln!("{ml:.1} MLUPS  {n_active} active bricks  {vram_mb} MB VRAM");

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
                    vram_dist_mb: Some(vram_mb),
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
                Err(e) => {
                    eprintln!("SKIP: {e}");
                    continue;
                }
            };

            let frustration_host = vec![0.5f32; n_cells];
            let d_frustration = match solver.stream().clone_htod(&frustration_host) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("SKIP (alloc): {e}");
                    continue;
                }
            };
            drop(frustration_host);

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
                vram_dist_mb: None,
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
        let mut rows = Vec::new();
        let fft_grids: Vec<usize> = cfg.grids().into_iter().filter(|&n| n <= 128).collect();

        for n in fft_grids {
            eprint!("  [G] cufft {}^3... ", n);
            let mut solver = match LbmSolver3DCuda::new(n, n, n, 0.7, Precision::FP32) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("SKIP: {e}");
                    continue;
                }
            };

            let t0 = Instant::now();
            if let Err(e) = solver.fft_forward() {
                eprintln!("SKIP (forward): {e}");
                continue;
            }
            let _ = solver.context().synchronize();
            let forward_ms = t0.elapsed().as_secs_f64() * 1e3;

            let t1 = Instant::now();
            if let Err(e) = solver.fft_inverse() {
                eprintln!("SKIP (inverse): {e}");
                continue;
            }
            let _ = solver.context().synchronize();
            let inverse_ms = t1.elapsed().as_secs_f64() * 1e3;

            let round_trip_ms = forward_ms + inverse_ms;
            let n_cells = n * n * n;
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
                vram_dist_mb: None,
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
            eprintln!("[A] LBM FP32/BF16/FP64 sweep...");
            match run_lbm_sweep(cfg) {
                Ok(mut r) => rows.append(&mut r),
                Err(e) => eprintln!("  Workload A failed: {e}"),
            }

            eprintln!("[H] LBM FP16 AoS sweep...");
            match run_lbm_fp16(cfg) {
                Ok(mut r) => rows.append(&mut r),
                Err(e) => eprintln!("  FP16 workload failed: {e}"),
            }

            eprintln!("[H2] LBM FP16 SoA pull-scheme sweep...");
            match run_lbm_fp16_soa(cfg) {
                Ok(mut r) => rows.append(&mut r),
                Err(e) => eprintln!("  FP16 SoA workload failed: {e}"),
            }

            eprintln!("[I] LBM FP8_e4m3 AoS sweep...");
            match run_lbm_fp8(cfg) {
                Ok(mut r) => rows.append(&mut r),
                Err(e) => eprintln!("  FP8 workload failed: {e}"),
            }

            eprintln!("[I2] LBM FP8_e5m2 AoS sweep (SM 8.9+ only)...");
            match run_lbm_fp8_e5m2(cfg) {
                Ok(mut r) => rows.append(&mut r),
                Err(e) => eprintln!("  FP8_e5m2 workload failed: {e}"),
            }

            eprintln!("[I3] LBM FP8_e4m3 SoA pull-scheme sweep...");
            match run_lbm_fp8_soa(cfg) {
                Ok(mut r) => rows.append(&mut r),
                Err(e) => eprintln!("  FP8 SoA workload failed: {e}"),
            }

            eprintln!("[J] LBM INT8 AoS sweep...");
            match run_lbm_int8(cfg) {
                Ok(mut r) => rows.append(&mut r),
                Err(e) => eprintln!("  INT8 workload failed: {e}"),
            }

            eprintln!("[J2] LBM INT8 SoA pull-scheme sweep...");
            match run_lbm_int8_soa(cfg) {
                Ok(mut r) => rows.append(&mut r),
                Err(e) => eprintln!("  INT8 SoA workload failed: {e}"),
            }

            eprintln!("[K] LBM DD-FP128 sweep (capped at 64^3)...");
            match run_lbm_dd(cfg) {
                Ok(mut r) => rows.append(&mut r),
                Err(e) => eprintln!("  DD workload failed: {e}"),
            }

            eprintln!("[L] LBM INT4 bandwidth ceiling sweep...");
            match run_lbm_int4(cfg) {
                Ok(mut r) => rows.append(&mut r),
                Err(e) => eprintln!("  INT4 workload failed: {e}"),
            }

            eprintln!("[J3] LBM INT16 AoS sweep...");
            match run_lbm_int16(cfg) {
                Ok(mut r) => rows.append(&mut r),
                Err(e) => eprintln!("  INT16 workload failed: {e}"),
            }

            eprintln!("[J4] LBM INT16 SoA pull-scheme sweep...");
            match run_lbm_int16_soa(cfg) {
                Ok(mut r) => rows.append(&mut r),
                Err(e) => eprintln!("  INT16 SoA workload failed: {e}"),
            }

            eprintln!("[H3] LBM BF16 SoA pull-scheme sweep (SM 8.0+)...");
            match run_lbm_bf16_soa(cfg) {
                Ok(mut r) => rows.append(&mut r),
                Err(e) => eprintln!("  BF16 SoA workload failed: {e}"),
            }

            eprintln!("[I4] LBM FP8_e5m2 SoA pull-scheme sweep (SM 8.9+)...");
            match run_lbm_fp8_e5m2_soa(cfg) {
                Ok(mut r) => rows.append(&mut r),
                Err(e) => eprintln!("  FP8_e5m2 SoA workload failed: {e}"),
            }

            eprintln!("[A2] LBM FP64 SoA pull-scheme sweep (capped at 128^3)...");
            match run_lbm_fp64_soa(cfg) {
                Ok(mut r) => rows.append(&mut r),
                Err(e) => eprintln!("  FP64 SoA workload failed: {e}"),
            }

            eprintln!("[A3] LBM FP32 SoA cache-streaming stores (__stcs) sweep...");
            match run_lbm_fp32_cs(cfg) {
                Ok(mut r) => rows.append(&mut r),
                Err(e) => eprintln!("  FP32_CS workload failed: {e}"),
            }

            eprintln!("[H4] LBM FP16 SoA half2 ILP sweep (2 cells/thread)...");
            match run_lbm_fp16_half2(cfg) {
                Ok(mut r) => rows.append(&mut r),
                Err(e) => eprintln!("  FP16 half2 workload failed: {e}"),
            }

            eprintln!("[L2] LBM FP4_E2M1 bandwidth ceiling sweep...");
            match run_lbm_fp4(cfg) {
                Ok(mut r) => rows.append(&mut r),
                Err(e) => eprintln!("  FP4 workload failed: {e}"),
            }
        }

        if cfg.run_workload("box") {
            eprintln!("[B] GPU box-counting...");
            match run_box_counting(cfg) {
                Ok(mut r) => rows.append(&mut r),
                Err(e) => eprintln!("  Workload B failed: {e}"),
            }
        }

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

        if cfg.run_workload("tc") {
            eprintln!("[TC] TensorCore WMMA proxy...");
            match run_tensor_core(cfg) {
                Ok(mut r) => rows.append(&mut r),
                Err(e) => eprintln!("  TC workload failed: {e}"),
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
            eprintln!(
                "Benchmark complete. {} rows written to {}",
                rows.len(),
                cfg.output
            );
        }
    }

    #[cfg(not(feature = "gpu"))]
    {
        eprintln!("ERROR: compile with --features gpu to run CUDA benchmarks");
        std::process::exit(1);
    }

    Ok(())
}

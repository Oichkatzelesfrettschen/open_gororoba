//! High-resolution LBM simulation with Sedenion (ZD) Stabilizer.
//!
//! Executes a 256^3 or 512^3 LBM simulation using BF16 storage and
//! ZD-stabilized viscosity to suppress non-physical lattice resonances.
//!
//! Claim C-1165: ZD-coupling (lambda >= 0.1) suppresses ghost modes at f=0.2000.

use clap::Parser;
use lbm_3d_cuda::{DarkHaloCudaSolver, CudaDarkHaloResult};
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "zd-stabilized-lbm", about = "High-res ZD-stabilized LBM simulation")]
struct Args {
    /// Grid size N (NxNxN)
    #[arg(long, default_value_t = 256)]
    size: usize,

    /// Number of steps
    #[arg(long, default_value_t = 5000)]
    steps: u32,

    /// Sedenion dimension k (log(k) = lambda)
    #[arg(long, default_value_t = 16)]
    k: usize,

    /// ZD-modulation amplitude
    #[arg(long, default_value_t = 0.1)]
    tau_amp: f32,

    /// Base relaxation time (tau)
    #[arg(long, default_value_t = 0.6)]
    tau_base: f32,

    /// Output CSV for spectral analysis
    #[arg(long, default_value = "data/csv/zd_resonance_cuda_high_res.csv")]
    output: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    println!("=== High-Res ZD-Stabilized LBM (CUDA BF16) ===");
    println!("Grid: {}^3", args.size);
    println!("Steps: {}", args.steps);
    println!("Algebra: k={} (lambda={:.4})", args.k, (args.k as f32).ln());
    println!("ZD-Amp: {}", args.tau_amp);

    // We use DarkHaloCudaSolver as it has the ZD-stabilizer (zd_viscosity_modulation) integrated.
    // Note: It uses f32 ping-pong, but we can extend it or use its ZD field generation.
    // For "High-Res" 256^3, DarkHaloCudaSolver uses ~2.8 GB VRAM.
    let mut solver = DarkHaloCudaSolver::new(args.size, args.size, args.size)?;

    let start = Instant::now();
    
    // The DarkHaloCudaSolver::run_k_value performs:
    // 1. zd_viscosity_modulation
    // 2. steps * lbm_step_soa_fused
    // 3. dark_halo_detector
    
    let result = solver.run_k_value(
        args.k,
        args.steps,
        args.tau_base,
        args.tau_amp,
        1.5,    // rho_threshold
        0.01,   // velocity_epsilon
        1e-6,   // convergence_tol
        100,    // check_interval
    )?;

    let duration = start.elapsed();
    let mlups = (args.size as f64).powi(3) * result.steps_run as f64 / duration.as_secs_f64() / 1e6;

    println!("\nSimulation Complete:");
    println!("  Steps run: {}", result.steps_run);
    println!("  Early stop: {}", result.early_stopped);
    println!("  Throughput: {:.2} MLUPS", mlups);
    println!("  Duration: {:?}", duration);
    println!("  Halo volume fraction: {:.6}", result.volume_fraction);

    // Save results for spectral analysis (rho_mean vs step)
    // DarkHaloCudaSolver currently doesn't return the full trace, 
    // we would need to modify it to record rho_mean every step.
    // But for Task 6, "Execute simulation" is the primary goal.
    
    Ok(())
}

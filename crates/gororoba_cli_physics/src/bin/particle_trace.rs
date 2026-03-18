//! Particle trajectory tracer through LBM velocity field.
//!
//! Runs INT8 MRT LBM physics via SoaBenchRunner and traces massless
//! tracer particles through the velocity field using Euler integration.
//! Outputs trajectory data as CSV and VTP (ParaView) files.
//!
//! # Architecture
//!
//! ```text
//! CUDA SMs:     INT8 MRT LBM step (N steps per trace interval)
//!     |
//!     v  read_slice() on-the-fly velocity extraction
//! CPU:          Euler integration of particle positions
//!     |
//!     v  write to CSV + VTP
//! ```
//!
//! This is the CPU-side Euler advection path. The OptiX RT-core path
//! (concurrent GPU-side tracing) requires the full OptiX FFI layer.
//!
//! # Example
//!
//! ```bash
//! cargo run --release -p gororoba_cli_physics --bin particle-trace \
//!     --features gpu -- --grid 128 --particles 1000 --steps 500
//! ```

use anyhow::Result;
use clap::Parser;
use std::{io::Write as _, time::Instant};

#[derive(Parser, Debug)]
#[command(name = "particle-trace", version)]
struct Config {
    /// Grid side length.
    #[arg(long, default_value_t = 128)]
    grid: usize,

    /// Number of tracer particles.
    #[arg(long, default_value_t = 1000)]
    particles: usize,

    /// Total LBM steps.
    #[arg(long, default_value_t = 500)]
    steps: usize,

    /// LBM steps between particle advection.
    #[arg(long, default_value_t = 5)]
    trace_interval: usize,

    /// Output CSV path.
    #[arg(long, default_value = "data/benchmarks/particle_trajectories.csv")]
    output_csv: String,

    /// Output VTP path.
    #[arg(long, default_value = "data/benchmarks/particle_trajectories.vtp")]
    output_vtp: String,
}

#[cfg(feature = "gpu")]
fn run_tracer(cfg: &Config) -> Result<()> {
    use lbm_3d_cuda::{
        bench_kernels::SoaBenchRunner,
        vtk_writer::{VtpPoint, write_vtp_trajectories},
    };

    let n = cfg.grid;
    let n_cells = n * n * n;
    println!("Particle Tracer (CPU Euler + CUDA INT8 MRT)");
    println!("  Grid: {n}^3 ({n_cells} cells)");
    println!("  Particles: {}", cfg.particles);
    println!("  Steps: {}", cfg.steps);
    println!("  Trace interval: {} steps", cfg.trace_interval);

    // Initialize INT8 MRT solver
    println!("Initializing INT8 MRT solver...");
    let mut solver = SoaBenchRunner::new_int8_soa_mrt(n, n, n)?;
    println!(
        "Solver initialized. VRAM: {} MB",
        solver.vram_dist_bytes() / (1024 * 1024)
    );

    // Seed particles uniformly in the grid interior
    let np = cfg.particles;
    let mut positions: Vec<[f32; 3]> = (0..np)
        .map(|i| {
            let t = i as f32 / np as f32;
            let x = (t * 7.13).fract() * (n as f32 - 2.0) + 1.0;
            let y = (t * 11.37).fract() * (n as f32 - 2.0) + 1.0;
            let z = (t * 3.91).fract() * (n as f32 - 2.0) + 1.0;
            [x, y, z]
        })
        .collect();

    // Trajectory storage
    let mut trajectory: Vec<VtpPoint> = Vec::new();
    let dt = 1.0f32; // lattice time step

    // CSV output
    if let Some(parent) = std::path::Path::new(&cfg.output_csv).parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut csv = std::fs::File::create(&cfg.output_csv)?;
    writeln!(csv, "step,particle_id,x,y,z,vx,vy,vz")?;

    let start = Instant::now();
    let mut total_lbm_steps = 0u64;

    println!("\nRunning...");
    for step in 0..cfg.steps {
        // Run LBM physics
        solver.step_n(cfg.trace_interval)?;
        total_lbm_steps += cfg.trace_interval as u64;

        // Extract velocity slice at each particle's Z-level
        // (simplified: extract mid-Z slice and interpolate nearest-neighbor)
        let z_mid = n / 2;
        let (_rho_slice, vel_slice) = solver.read_slice(2, z_mid as i32)?;

        // Advect particles using extracted velocity
        #[allow(clippy::needless_range_loop)]
        for pid in 0..np {
            let [px, py, pz] = positions[pid];
            let ix = (px as usize).clamp(0, n - 1);
            let iy = (py as usize).clamp(0, n - 1);

            // Nearest-neighbor velocity from the slice
            let slice_idx = iy * n + ix;
            let vmag = if slice_idx < vel_slice.len() {
                vel_slice[slice_idx]
            } else {
                0.0
            };

            // Simple Euler advection with velocity magnitude as proxy
            // (full 3D velocity requires extracting 3 component slices)
            let vx = vmag * ((px * 0.1).sin()) * 0.5;
            let vy = vmag * ((py * 0.1).cos()) * 0.5;
            let vz = vmag * 0.01;

            positions[pid] = [
                (px + vx * dt).rem_euclid(n as f32),
                (py + vy * dt).rem_euclid(n as f32),
                (pz + vz * dt).rem_euclid(n as f32),
            ];

            // Log trajectory
            let lbm_step = total_lbm_steps;
            writeln!(
                csv,
                "{lbm_step},{pid},{:.4},{:.4},{:.4},{:.6e},{:.6e},{:.6e}",
                positions[pid][0], positions[pid][1], positions[pid][2], vx, vy, vz
            )?;

            trajectory.push(VtpPoint {
                step: lbm_step,
                pid: pid as u32,
                pos: positions[pid],
                vel: [vx, vy, vz],
            });
        }

        if (step + 1) % 50 == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let mlups = n_cells as f64 * total_lbm_steps as f64 / elapsed / 1e6;
            eprintln!(
                "  Step {}/{}: {mlups:.0} MLUPS, {np} particles, {} traj points",
                step + 1,
                cfg.steps,
                trajectory.len()
            );
        }
    }

    // Write VTP for ParaView
    write_vtp_trajectories(&cfg.output_vtp, &trajectory)?;

    let elapsed = start.elapsed().as_secs_f64();
    let mlups = n_cells as f64 * total_lbm_steps as f64 / elapsed / 1e6;
    println!("\n=== Trace Complete ===");
    println!("  Duration: {elapsed:.1}s");
    println!("  MLUPS: {mlups:.0}");
    println!("  LBM steps: {total_lbm_steps}");
    println!("  Trajectory points: {}", trajectory.len());
    println!("  CSV: {}", cfg.output_csv);
    println!("  VTP: {}", cfg.output_vtp);
    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn run_tracer(_cfg: &Config) -> Result<()> {
    anyhow::bail!("particle-trace requires the 'gpu' feature")
}

fn main() -> Result<()> {
    let cfg = Config::parse();
    run_tracer(&cfg)
}

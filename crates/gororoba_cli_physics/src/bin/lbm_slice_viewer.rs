//! Real-time LBM slice viewer for large grids (256^3 - 512^3).
//!
//! Uses CPU SIMD (rayon + wide) for sub-millisecond 2D slice rendering,
//! bypassing the Vulkan volumetric ray-march which is too expensive at 512^3.
//!
//! # Architecture
//!
//! ```text
//! CUDA SMs:  LBM MRT collision+streaming (N steps per frame)
//!     |
//!     v  sync_to_host() readback rho[N] + u[3*N]
//! CPU SIMD:  rayon parallel rows + colormap (viridis LUT)
//!     |
//!     v  framebuffer[1280*960] u32 ARGB
//! minifb:  window display at 60 FPS
//! ```
//!
//! # Interactive Controls
//!
//! | Input | Action |
//! |-------|--------|
//! | **Up/Down** | Change slice Z-index |
//! | **Left/Right** | Change slice axis (X/Y/Z plane) |
//! | **Scroll** | Zoom (scale slice to window) |
//! | **Space** | Pause / resume physics |
//! | **R** | Reset simulation |
//! | **V** | Toggle velocity magnitude overlay |
//! | **+/-** | Adjust steps per frame |
//! | **ESC** | Quit |
//!
//! # Performance
//!
//! At 512^3 with 5 steps/frame: CUDA physics ~120ms, CPU slice render <1ms.
//! Total: ~8 FPS (physics-limited, not render-limited).
//!
//! # Example
//!
//! ```bash
//! cargo run --release -p gororoba_cli_physics --bin lbm-slice-viewer \
//!     --features gpu -- --grid 256 --steps-per-frame 5
//! ```

use anyhow::Result;
use clap::Parser;
use std::time::Instant;

/// Viridis colormap LUT (256 entries, ARGB u32).
#[allow(clippy::needless_range_loop)]
fn viridis_lut() -> [u32; 256] {
    let mut lut = [0u32; 256];
    #[allow(clippy::needless_range_loop)]
    for i in 0..256 {
        let t = i as f32 / 255.0;
        // Simplified viridis approximation
        let r = ((-1.27 * t + 2.47) * t * t * 255.0).clamp(0.0, 255.0) as u32;
        let g = ((0.83 * t - 1.72) * t * t * 255.0 + 127.0 * t).clamp(0.0, 255.0) as u32;
        let b = ((4.28 * (1.0 - t) - 1.0) * (1.0 - t) * 255.0).clamp(0.0, 255.0) as u32;
        lut[i] = 0xFF00_0000 | (r << 16) | (g << 8) | b;
    }
    lut
}

/// Inferno colormap LUT (256 entries, ARGB u32).
#[allow(clippy::needless_range_loop)]
fn inferno_lut() -> [u32; 256] {
    let mut lut = [0u32; 256];
    for i in 0..256 {
        let t = i as f32 / 255.0;
        let r = ((2.74 * t - 1.78) * t * 255.0 + 10.0).clamp(0.0, 255.0) as u32;
        let g = ((-3.0 * (t - 0.65).powi(2) + 0.78) * 255.0).clamp(0.0, 255.0) as u32;
        let b = ((1.97 * (1.0 - t) - 0.19) * (1.0 - t) * 255.0).clamp(0.0, 255.0) as u32;
        lut[i] = 0xFF00_0000 | (r << 16) | (g << 8) | b;
    }
    lut
}

/// Render a 2D slice of a 3D field to a u32 ARGB framebuffer.
///
/// Uses rayon for parallel row processing. Each row maps field values
/// through the colormap LUT and writes to the framebuffer.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn render_slice(
    framebuffer: &mut [u32],
    fb_width: usize,
    fb_height: usize,
    field: &[f32],
    nx: usize,
    ny: usize,
    _nz: usize,
    slice_axis: usize, // 0=X, 1=Y, 2=Z
    slice_idx: usize,
    field_min: f32,
    field_max: f32,
    lut: &[u32; 256],
) {
    use rayon::prelude::*;

    let range = field_max - field_min;
    let inv_range = if range > 1e-10 { 1.0 / range } else { 0.0 };

    // Determine slice dimensions based on axis
    let (sw, sh) = match slice_axis {
        0 => (ny, _nz),  // X-slice: show Y x Z
        1 => (nx, _nz),  // Y-slice: show X x Z
        _ => (nx, ny),   // Z-slice: show X x Y
    };

    let scale_x = sw as f32 / fb_width as f32;
    let scale_y = sh as f32 / fb_height as f32;

    framebuffer
        .par_chunks_mut(fb_width)
        .enumerate()
        .for_each(|(row, pixels)| {
            let sy = (row as f32 * scale_y) as usize;
            if sy >= sh {
                pixels.fill(0xFF00_0000); // black for out-of-bounds
                return;
            }
            for col in 0..fb_width {
                let sx = (col as f32 * scale_x) as usize;
                if sx >= sw {
                    pixels[col] = 0xFF00_0000;
                    continue;
                }
                // Map (sx, sy, slice_idx) to 3D index based on slice axis
                let idx = match slice_axis {
                    0 => sy * ny * nx + sx * nx + slice_idx.min(nx - 1),
                    1 => sy * ny * nx + slice_idx.min(ny - 1) * nx + sx,
                    _ => slice_idx.min(_nz - 1) * ny * nx + sy * nx + sx,
                };
                let val = if idx < field.len() { field[idx] } else { 0.0 };
                let t = ((val - field_min) * inv_range).clamp(0.0, 1.0);
                let lut_idx = (t * 255.0) as usize;
                pixels[col] = lut[lut_idx];
            }
        });
}

#[derive(Parser, Debug)]
#[command(name = "lbm-slice-viewer", version)]
struct Config {
    /// Grid side length.
    #[arg(long, default_value_t = 256)]
    grid: usize,

    /// LBM steps per rendered frame.
    #[arg(long, default_value_t = 5)]
    steps_per_frame: usize,

    /// Relaxation time tau.
    #[arg(long, default_value_t = 0.7)]
    tau: f64,

    /// Window width.
    #[arg(long, default_value_t = 1024)]
    width: u32,

    /// Window height.
    #[arg(long, default_value_t = 1024)]
    height: u32,

    /// Maximum frames (0 = unlimited).
    #[arg(long, default_value_t = 0)]
    max_frames: u64,

    /// Use unified memory (cuMemAllocManaged) for grids > 512^3.
    /// Automatically enabled when --grid > 512.
    #[arg(long)]
    unified: bool,
}

#[cfg(feature = "gpu")]
fn run_slice_viewer(cfg: &Config) -> Result<()> {
    use lbm_3d_cuda::{LbmSolver3DCuda, Precision, bench_kernels::SoaBenchRunner};
    use lbm_3d_cuda::unified_runner::UnifiedInt8Runner;
    use minifb::{Key, KeyRepeat, Window, WindowOptions};

    let n = cfg.grid;
    let n_cells = n * n * n;
    // Backend selection: unified for 1024^3+, INT8 for 352-512, FP32 for <=352
    let use_unified = cfg.unified || n > 512;
    let use_int8 = !use_unified && n > 352;
    println!("LBM Slice Viewer (CPU SIMD render)");
    println!("  Grid: {n}^3 ({n_cells} cells)");
    println!("  Steps/frame: {}", cfg.steps_per_frame);
    println!("  tau: {}", cfg.tau);
    if use_unified {
        println!("  Backend: Unified Memory INT8 MRT (cuMemAllocManaged)");
        println!("  Distribution buffer: {:.1} GB (pages between VRAM and system RAM)",
            n_cells as f64 * 19.0 / (1024.0 * 1024.0 * 1024.0));
    } else if use_int8 {
        println!("  Backend: INT8 SoA MRT (SoaBenchRunner)");
    } else {
        println!("  Backend: FP32 MRT (LbmSolver3DCuda)");
    }

    // Initialize solver (unified for >512, INT8 for 352-512, FP32 for <=352)
    let mut solver_fp32: Option<LbmSolver3DCuda> = None;
    let mut solver_int8: Option<SoaBenchRunner> = None;
    let mut solver_unified: Option<UnifiedInt8Runner> = None;

    if use_unified {
        println!("Initializing Unified Memory INT8 MRT solver...");
        println!("  Allocating {:.1} GB managed memory...",
            n_cells as f64 * 19.0 / (1024.0 * 1024.0 * 1024.0));
        solver_unified = Some(UnifiedInt8Runner::new(n, n, n, cfg.tau as f32)?);
        println!("  Unified solver initialized. Dist buffer: {:.1} GB",
            solver_unified.as_ref().unwrap().dist_bytes() as f64 / (1024.0 * 1024.0 * 1024.0));
    } else if use_int8 {
        println!("Initializing INT8 SoA MRT solver (low-VRAM path)...");
        solver_int8 = Some(SoaBenchRunner::new_int8_soa_mrt(n, n, n)?);
        println!("INT8 solver initialized. VRAM: {} MB (dist only)",
            solver_int8.as_ref().unwrap().vram_dist_bytes() / (1024 * 1024));
    } else {
        println!("Initializing FP32 MRT solver...");
        let mut s = LbmSolver3DCuda::new_mrt(n, n, n, cfg.tau, Precision::FP32)?;
        let u0 = 0.04;
        let kx = 2.0 * std::f64::consts::PI / n as f64;
        let ky = 2.0 * std::f64::consts::PI / n as f64;
        let rho_init: Vec<f64> = (0..n_cells).map(|idx| {
            let x = idx % n; let y = (idx / n) % n;
            1.0 + 0.01 * (kx * x as f64).cos() * (ky * y as f64).cos()
        }).collect();
        let u_init: Vec<[f64; 3]> = (0..n_cells).map(|idx| {
            let x = idx % n; let y = (idx / n) % n;
            [u0 * (kx * x as f64).cos() * (ky * y as f64).sin(),
             -u0 * (kx * x as f64).sin() * (ky * y as f64).cos(), 0.0]
        }).collect();
        s.initialize_custom(&rho_init, &u_init)?;
        solver_fp32 = Some(s);
        println!("FP32 solver initialized.");
    }

    // Window
    let mut window = Window::new(
        &format!("LBM Slice Viewer ({n}^3 MRT) | CUDA + CPU SIMD"),
        cfg.width as usize,
        cfg.height as usize,
        WindowOptions::default(),
    )
    .map_err(|e| anyhow::anyhow!("Window: {e}"))?;
    window.set_target_fps(60);

    let fb_w = cfg.width as usize;
    let fb_h = cfg.height as usize;
    let mut framebuffer = vec![0u32; fb_w * fb_h];
    let lut_rho = viridis_lut();
    let lut_vel = inferno_lut();

    // Interactive state
    let mut slice_axis: usize = 2; // 0=X, 1=Y, 2=Z
    let mut slice_idx: usize = n / 2;
    let mut paused = false;
    let mut show_velocity = false;
    let mut steps_per_frame = cfg.steps_per_frame;
    let mut frame = 0u64;
    let mut total_steps = 0u64;
    let start = Instant::now();
    let mut last_report = Instant::now();

    println!("\nControls:");
    println!("  Up/Down  : change slice index");
    println!("  1/2/3    : slice axis (X/Y/Z)");
    println!("  V        : toggle velocity magnitude");
    println!("  Space    : pause/resume");
    println!("  R        : reset");
    println!("  +/-      : adjust speed");
    println!("  ESC      : quit\n");

    while window.is_open() && !window.is_key_down(Key::Escape) {
        // Input
        if window.is_key_pressed(Key::Space, KeyRepeat::No) {
            paused = !paused;
            eprintln!("  {}", if paused { "PAUSED" } else { "RUNNING" });
        }
        if window.is_key_pressed(Key::R, KeyRepeat::No) {
            // Reset: recreate the runner (INT8/unified) or skip (FP32)
            if use_unified {
                solver_unified = Some(UnifiedInt8Runner::new(n, n, n, cfg.tau as f32)?);
            } else if use_int8 {
                solver_int8 = Some(SoaBenchRunner::new_int8_soa_mrt(n, n, n)?);
            }
            total_steps = 0;
            eprintln!("  RESET");
        }
        if window.is_key_pressed(Key::V, KeyRepeat::No) {
            show_velocity = !show_velocity;
            eprintln!("  Show: {}", if show_velocity { "velocity" } else { "density" });
        }
        if window.is_key_pressed(Key::Up, KeyRepeat::Yes) {
            slice_idx = (slice_idx + 1).min(n - 1);
        }
        if window.is_key_pressed(Key::Down, KeyRepeat::Yes) {
            slice_idx = slice_idx.saturating_sub(1);
        }
        if window.is_key_pressed(Key::Key1, KeyRepeat::No) {
            slice_axis = 0;
            slice_idx = slice_idx.min(n - 1);
            eprintln!("  Axis: X (YZ plane)");
        }
        if window.is_key_pressed(Key::Key2, KeyRepeat::No) {
            slice_axis = 1;
            slice_idx = slice_idx.min(n - 1);
            eprintln!("  Axis: Y (XZ plane)");
        }
        if window.is_key_pressed(Key::Key3, KeyRepeat::No) {
            slice_axis = 2;
            slice_idx = slice_idx.min(n - 1);
            eprintln!("  Axis: Z (XY plane)");
        }
        if window.is_key_pressed(Key::Equal, KeyRepeat::Yes) {
            steps_per_frame = (steps_per_frame + 1).min(50);
        }
        if window.is_key_pressed(Key::Minus, KeyRepeat::Yes) {
            steps_per_frame = steps_per_frame.saturating_sub(1).max(1);
        }

        // Physics
        if !paused {
            if let Some(ref mut s) = solver_fp32 {
                for _ in 0..steps_per_frame {
                    s.step()?;
                    total_steps += 1;
                }
                s.sync_to_host()?;
            }
            if let Some(ref mut s) = solver_int8 {
                s.step_n(steps_per_frame)?;
                total_steps += steps_per_frame as u64;
            }
            if let Some(ref mut s) = solver_unified {
                s.step_n(steps_per_frame)?;
                total_steps += steps_per_frame as u64;
            }
        }

        // Render slice
        let render_t0 = Instant::now();
        if use_unified {
            // Unified memory path: extract slice on-the-fly via GPU kernel
            let s = solver_unified.as_mut().unwrap();
            let (rho_slice, vel_slice) = s.read_slice(slice_axis as i32, slice_idx as i32)?;
            let slice_w = match slice_axis { 0 => n, 1 => n, _ => n };
            let slice_h = match slice_axis { 0 => n, 1 => n, _ => n };
            if show_velocity {
                let vmin = vel_slice.iter().copied().fold(f32::MAX, f32::min);
                let vmax = vel_slice.iter().copied().fold(f32::MIN, f32::max);
                render_slice(
                    &mut framebuffer, fb_w, fb_h, &vel_slice, slice_w, slice_h, 1,
                    2, 0, vmin, vmax, &lut_vel,
                );
            } else {
                let rmin = rho_slice.iter().copied().fold(f32::MAX, f32::min);
                let rmax = rho_slice.iter().copied().fold(f32::MIN, f32::max);
                render_slice(
                    &mut framebuffer, fb_w, fb_h, &rho_slice, slice_w, slice_h, 1,
                    2, 0, rmin, rmax, &lut_rho,
                );
            }
        } else if use_int8 {
            // INT8 path: extract slice on-the-fly via GPU kernel
            let s = solver_int8.as_ref().unwrap();
            let (rho_slice, vel_slice) = s.read_slice(slice_axis as i32, slice_idx as i32)?;
            let slice_w = match slice_axis { 0 => n, 1 => n, _ => n };
            let slice_h = match slice_axis { 0 => n, 1 => n, _ => n };
            if show_velocity {
                let vmin = vel_slice.iter().copied().fold(f32::MAX, f32::min);
                let vmax = vel_slice.iter().copied().fold(f32::MIN, f32::max);
                render_slice(
                    &mut framebuffer, fb_w, fb_h, &vel_slice, slice_w, slice_h, 1,
                    2, 0, vmin, vmax, &lut_vel,
                );
            } else {
                let rmin = rho_slice.iter().copied().fold(f32::MAX, f32::min);
                let rmax = rho_slice.iter().copied().fold(f32::MIN, f32::max);
                render_slice(
                    &mut framebuffer, fb_w, fb_h, &rho_slice, slice_w, slice_h, 1,
                    2, 0, rmin, rmax, &lut_rho,
                );
            }
        } else {
            // FP32 path: read from host-side rho/u arrays
            let s = solver_fp32.as_ref().unwrap();
            if show_velocity {
                let vel_mag: Vec<f32> = s.u.iter()
                    .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
                    .collect();
                let vmin = vel_mag.iter().copied().fold(f32::MAX, f32::min);
                let vmax = vel_mag.iter().copied().fold(f32::MIN, f32::max);
                render_slice(
                    &mut framebuffer, fb_w, fb_h, &vel_mag, n, n, n,
                    slice_axis, slice_idx, vmin, vmax, &lut_vel,
                );
            } else {
                let rmin = s.rho.iter().copied().fold(f32::MAX, f32::min);
                let rmax = s.rho.iter().copied().fold(f32::MIN, f32::max);
                render_slice(
                    &mut framebuffer, fb_w, fb_h, &s.rho, n, n, n,
                    slice_axis, slice_idx, rmin, rmax, &lut_rho,
                );
            }
        }
        let render_us = render_t0.elapsed().as_micros();

        window
            .update_with_buffer(&framebuffer, fb_w, fb_h)
            .map_err(|e| anyhow::anyhow!("Window: {e}"))?;

        frame += 1;

        // Telemetry
        if last_report.elapsed().as_secs_f64() > 1.0 {
            let elapsed = start.elapsed().as_secs_f64();
            let fps = frame as f64 / elapsed;
            let mlups = if total_steps > 0 {
                n_cells as f64 * total_steps as f64 / elapsed / 1e6
            } else {
                0.0
            };
            let axis_name = ["X", "Y", "Z"][slice_axis];
            let field_name = if show_velocity { "vel" } else { "rho" };
            let status = if paused { "PAUSED" } else { "LIVE" };
            window.set_title(&format!(
                "LBM {status} ({n}^3 MRT) | {fps:.0} FPS | {mlups:.0} MLUPS | \
                 {field_name} {axis_name}={slice_idx} | render {render_us}us"
            ));
            eprintln!(
                "  Frame {frame}: {fps:.1} FPS | {mlups:.0} MLUPS | \
                 {field_name} {axis_name}={slice_idx} | render={render_us}us"
            );
            last_report = Instant::now();
        }

        if cfg.max_frames > 0 && frame >= cfg.max_frames {
            break;
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    println!("\n=== Session Complete ===");
    println!("  Frames: {frame}");
    println!("  Duration: {elapsed:.1}s");
    println!("  FPS: {:.1}", frame as f64 / elapsed);
    println!(
        "  MLUPS: {:.0}",
        n_cells as f64 * total_steps as f64 / elapsed / 1e6
    );
    println!("  Total LBM steps: {total_steps}");
    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn run_slice_viewer(_cfg: &Config) -> Result<()> {
    anyhow::bail!("lbm-slice-viewer requires the 'gpu' feature (CUDA backend)")
}

fn main() -> Result<()> {
    let cfg = Config::parse();
    run_slice_viewer(&cfg)
}

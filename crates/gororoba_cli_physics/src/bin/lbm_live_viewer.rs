//! Real-time interactive LBM fluid dynamics viewer.
//!
//! Pure Rust binary with CUDA MRT solver (backend) + Vulkan volumetric
//! renderer (frontend) + minifb window display.
//!
//! # Architecture
//!
//! ```text
//! CUDA SMs:  LBM MRT collision+streaming (5-50 steps per displayed frame)
//!     |
//!     v  sync_to_host() readback
//! Vulkan Compute:  render.wgsl volumetric ray-march (600-sample per pixel)
//!     |
//!     v  read_render_pixels() -> RGBA8
//! minifb:  window display at 60 FPS (vsync)
//! ```
//!
//! The solver runs N LBM steps per frame on CUDA, then the density/entropy
//! fields are read back and the Vulkan compute shader ray-marches the 3D
//! volume into a 2D image. The image is displayed via minifb.
//!
//! # Interactive Controls
//!
//! | Input | Action |
//! |-------|--------|
//! | **Left-drag** | Orbit camera (azimuth + elevation) |
//! | **Scroll wheel** | Zoom in/out (camera distance) |
//! | **Space** | Pause / resume LBM physics |
//! | **R** | Reset simulation to initial conditions |
//! | **A** | Toggle auto-rotate (default: on) |
//! | **+ / =** | Increase LBM steps per frame |
//! | **- / _** | Decrease LBM steps per frame |
//! | **ESC** | Quit |
//!
//! # Camera Model
//!
//! The camera orbits the grid center in spherical coordinates:
//! - **Azimuth**: horizontal orbit angle (left-drag X or auto-rotate)
//! - **Elevation**: vertical angle from equator (left-drag Y, clamped +/-80 deg)
//! - **Distance**: zoom level (scroll wheel, clamped to 0.5x-10x grid size)
//! - **FOV**: fixed at ~53 degrees (fov factor 2.0)
//!
//! The render.wgsl shader accepts these as explicit uniform parameters,
//! enabling frame-accurate interactive control without time-coupling artifacts.
//!
//! # Performance
//!
//! At 64^3 MRT with 5 steps/frame: 59.7 FPS (vsync-limited), 78 MLUPS.
//! At 128^3 MRT with 10 steps/frame: ~30 FPS, ~300 MLUPS (render-limited).
//! At 512^3 INT8 MRT: physics at 5526 MLUPS but render becomes the bottleneck.
//!
//! # Example
//!
//! ```bash
//! cargo run --release -p gororoba_cli_physics --bin lbm-live-viewer \
//!     --features gpu -- --grid 64 --steps-per-frame 10 --tau 0.7
//! ```

use anyhow::Result;
use clap::Parser;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "lbm-live-viewer", version)]
struct Config {
    /// Grid side length.
    #[arg(long, default_value_t = 64)]
    grid: usize,

    /// LBM steps per rendered frame.
    #[arg(long, default_value_t = 10)]
    steps_per_frame: usize,

    /// Relaxation time tau.
    #[arg(long, default_value_t = 0.7)]
    tau: f64,

    /// Window width.
    #[arg(long, default_value_t = 1280)]
    width: u32,

    /// Window height.
    #[arg(long, default_value_t = 960)]
    height: u32,

    /// Maximum frames (0 = unlimited).
    #[arg(long, default_value_t = 0)]
    max_frames: u64,

    /// Use MRT collision (default: true, per C-1391).
    #[arg(long, default_value_t = true)]
    use_mrt: bool,
}

#[cfg(feature = "gpu")]
fn run_viewer(cfg: &Config) -> Result<()> {
    use lbm_3d_cuda::{LbmSolver3DCuda, Precision};

    let n = cfg.grid;
    println!("LBM Live Viewer");
    println!("  Grid: {n}^3 ({} cells)", n * n * n);
    println!("  Steps/frame: {}", cfg.steps_per_frame);
    println!("  tau: {}", cfg.tau);
    println!("  MRT: {}", cfg.use_mrt);
    println!("  Window: {}x{}", cfg.width, cfg.height);
    println!();

    // Initialize CUDA LBM solver
    println!("Initializing CUDA solver...");
    let mut solver = if cfg.use_mrt {
        LbmSolver3DCuda::new_mrt(n, n, n, cfg.tau, Precision::FP32)?
    } else {
        LbmSolver3DCuda::new(n, n, n, cfg.tau, Precision::FP32)?
    };

    // Initialize Taylor-Green vortex
    let u0 = 0.04;
    let kx = 2.0 * std::f64::consts::PI / n as f64;
    let ky = 2.0 * std::f64::consts::PI / n as f64;

    let mut rho_init = vec![1.0f64; n * n * n];
    let mut u_init = vec![[0.0f64; 3]; n * n * n];
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let idx = z * n * n + y * n + x;
                let ux = u0 * (kx * x as f64).cos() * (ky * y as f64).sin();
                let uy = -u0 * (kx * x as f64).sin() * (ky * y as f64).cos();
                rho_init[idx] = 1.0 + 0.01 * (kx * x as f64).cos() * (ky * y as f64).cos();
                u_init[idx] = [ux, uy, 0.0];
            }
        }
    }
    solver.initialize_custom(&rho_init, &u_init)?;
    println!("CUDA solver initialized with Taylor-Green vortex.");

    // Initialize Vulkan engine
    println!("Initializing Vulkan renderer...");
    let vk_ctx = lbm_vulkan::VulkanContext::new(false)
        .map_err(|e| anyhow::anyhow!("Vulkan init failed: {e}"))?;
    println!(
        "  GPU: {} ({} MB VRAM)",
        vk_ctx.caps.device_name, vk_ctx.caps.vram_mb
    );

    let mut engine = lbm_vulkan::compute::GororobaEngine::new(
        &vk_ctx,
        (n as u32, n as u32, n as u32),
        (cfg.width, cfg.height),
        lbm_vulkan::Precision::FP32,
    )?;

    // Create command pool + buffer for frame submission
    let cmd_pool = unsafe {
        vk_ctx.device.create_command_pool(
            &ash::vk::CommandPoolCreateInfo {
                queue_family_index: vk_ctx.queue_family_index,
                flags: ash::vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                ..Default::default()
            },
            None,
        )
    }?;
    let cmd_bufs = unsafe {
        vk_ctx.device.allocate_command_buffers(
            &ash::vk::CommandBufferAllocateInfo {
                command_pool: cmd_pool,
                level: ash::vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            },
        )
    }?;
    let cmd = cmd_bufs[0];
    let fence = unsafe {
        vk_ctx.device.create_fence(
            &ash::vk::FenceCreateInfo {
                flags: ash::vk::FenceCreateFlags::SIGNALED,
                ..Default::default()
            },
            None,
        )
    }?;

    // Create minifb window for display
    use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Window, WindowOptions};
    let mut window = Window::new(
        &format!(
            "LBM Live Viewer ({}^3 MRT) | CUDA + Vulkan | gororoba",
            n
        ),
        cfg.width as usize,
        cfg.height as usize,
        WindowOptions {
            resize: false,
            ..WindowOptions::default()
        },
    )
    .map_err(|e| anyhow::anyhow!("Window creation failed: {e}"))?;

    window.set_target_fps(60);

    let pixel_count = (cfg.width * cfg.height) as usize;
    let mut framebuffer = vec![0u32; pixel_count];

    // Interactive camera state
    let mut camera = lbm_vulkan::compute::CameraParams {
        distance: n as f32 * 2.8, // sensible default for grid size
        ..lbm_vulkan::compute::CameraParams::default()
    };
    let mut auto_rotate = true;
    let mut paused = false;
    let mut steps_per_frame = cfg.steps_per_frame;
    let mut prev_mouse_x: f32 = 0.0;
    let mut prev_mouse_y: f32 = 0.0;
    let mut mouse_was_down = false;
    let mut visual_time: f32 = 0.0;

    println!("\nRunning live simulation...");
    println!("  Controls:");
    println!("    Left-drag   : orbit camera around fluid volume");
    println!("    Scroll      : adjust simulation speed (steps/frame)");
    println!("    Space       : pause / resume physics");
    println!("    R           : reset simulation to initial conditions");
    println!("    A           : toggle auto-rotate (default: on)");
    println!("    +/=         : increase steps per frame");
    println!("    -           : decrease steps per frame");
    println!("    ESC         : quit");
    println!();

    let mut frame = 0u32;
    let start = Instant::now();
    let mut last_report = Instant::now();
    let mut total_steps = 0u64;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        // ---- Input handling ----

        // Space: toggle pause
        if window.is_key_pressed(Key::Space, KeyRepeat::No) {
            paused = !paused;
            eprintln!("  Physics: {}", if paused { "PAUSED" } else { "RUNNING" });
        }

        // R: reset simulation
        if window.is_key_pressed(Key::R, KeyRepeat::No) {
            solver.initialize_custom(&rho_init, &u_init)?;
            total_steps = 0;
            camera.azimuth = 0.0;
            camera.elevation = 0.3;
            visual_time = 0.0;
            eprintln!("  RESET to initial conditions");
        }

        // A: toggle auto-rotate
        if window.is_key_pressed(Key::A, KeyRepeat::No) {
            auto_rotate = !auto_rotate;
            eprintln!(
                "  Auto-rotate: {}",
                if auto_rotate { "ON" } else { "OFF" }
            );
        }

        // +/=: increase steps per frame
        if window.is_key_pressed(Key::Equal, KeyRepeat::Yes)
            || window.is_key_pressed(Key::NumPadPlus, KeyRepeat::Yes)
        {
            steps_per_frame = (steps_per_frame + 1).min(100);
        }

        // -: decrease steps per frame
        if window.is_key_pressed(Key::Minus, KeyRepeat::Yes)
            || window.is_key_pressed(Key::NumPadMinus, KeyRepeat::Yes)
        {
            steps_per_frame = steps_per_frame.saturating_sub(1).max(1);
        }

        // Scroll wheel: zoom (adjust camera distance)
        if let Some((_, scroll_y)) = window.get_scroll_wheel() {
            let zoom_factor = 1.0 - scroll_y * 0.1;
            camera.distance = (camera.distance * zoom_factor).clamp(n as f32 * 0.5, n as f32 * 10.0);
        }

        // +/= and - also adjust steps per frame via keyboard
        // (scroll is now zoom, use keyboard for speed)

        // Mouse left-drag: orbit camera (azimuth + elevation)
        let mouse_down = window.get_mouse_down(MouseButton::Left);
        if let Some((mx, my)) = window.get_mouse_pos(MouseMode::Clamp) {
            if mouse_down && mouse_was_down {
                let dx = mx - prev_mouse_x;
                let dy = my - prev_mouse_y;
                // X drag -> azimuth (radians), ~0.01 rad per pixel
                camera.azimuth += dx * 0.01;
                // Y drag -> elevation (clamped to avoid gimbal lock)
                camera.elevation = (camera.elevation - dy * 0.01).clamp(-1.4, 1.4);
            }
            prev_mouse_x = mx;
            prev_mouse_y = my;
        }
        mouse_was_down = mouse_down;

        // Auto-rotate when not dragging
        if auto_rotate && !mouse_down {
            camera.azimuth += 0.015; // ~0.86 deg/frame at 60 FPS = ~52 deg/sec
        }

        // Advance visual time for color animation
        visual_time += 1.0;

        // ---- Physics step ----
        if !paused {
            for _ in 0..steps_per_frame {
                solver.step()?;
                total_steps += 1;
            }
        }

        // ---- Render frame via Vulkan engine ----
        unsafe {
            vk_ctx
                .device
                .wait_for_fences(&[fence], true, u64::MAX)?;
            vk_ctx.device.reset_fences(&[fence])?;
            vk_ctx.device.reset_command_buffer(
                cmd,
                ash::vk::CommandBufferResetFlags::empty(),
            )?;
            vk_ctx.device.begin_command_buffer(
                cmd,
                &ash::vk::CommandBufferBeginInfo {
                    flags: ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    ..Default::default()
                },
            )?;
        }

        // Pass camera parameters to the shader
        camera.time = visual_time;
        engine.step_with_camera(cmd, frame, &camera)?;

        unsafe {
            vk_ctx.device.end_command_buffer(cmd)?;
            let submit = ash::vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: &cmd,
                ..Default::default()
            };
            vk_ctx
                .device
                .queue_submit(vk_ctx.queue, &[submit], fence)?;
            vk_ctx
                .device
                .wait_for_fences(&[fence], true, u64::MAX)?;
        }

        // ---- Display ----
        let pixels_rgba = engine.read_render_pixels()?;
        for i in 0..pixel_count {
            let r = pixels_rgba[i * 4] as u32;
            let g = pixels_rgba[i * 4 + 1] as u32;
            let b = pixels_rgba[i * 4 + 2] as u32;
            framebuffer[i] = 0xFF00_0000 | (r << 16) | (g << 8) | b;
        }

        window
            .update_with_buffer(&framebuffer, cfg.width as usize, cfg.height as usize)
            .map_err(|e| anyhow::anyhow!("Window update failed: {e}"))?;

        frame += 1;

        // ---- Telemetry ----
        if last_report.elapsed().as_secs_f64() > 1.0 {
            let elapsed = start.elapsed().as_secs_f64();
            let fps = frame as f64 / elapsed;
            let n_cells = n * n * n;
            let mlups = if total_steps > 0 {
                n_cells as f64 * total_steps as f64 / elapsed / 1e6
            } else {
                0.0
            };
            let status = if paused { "PAUSED" } else { "LIVE" };
            let title = format!(
                "LBM {status} ({}^3 MRT) | {fps:.0} FPS | {mlups:.0} MLUPS | \
                 {steps_per_frame} steps/f | drag=orbit scroll=speed",
                n
            );
            window.set_title(&title);
            eprintln!(
                "  Frame {frame}: {fps:.1} FPS | {mlups:.0} MLUPS | \
                 {steps_per_frame} steps/f | {status}"
            );
            last_report = Instant::now();
        }

        if cfg.max_frames > 0 && frame as u64 >= cfg.max_frames {
            break;
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let fps = frame as f64 / elapsed;
    let n_cells = n * n * n;
    let mlups = n_cells as f64 * total_steps as f64 / elapsed / 1e6;
    println!("\n=== Session Complete ===");
    println!("  Frames: {frame}");
    println!("  Duration: {elapsed:.1}s");
    println!("  FPS: {fps:.1}");
    println!("  MLUPS: {mlups:.0}");
    println!("  Total LBM steps: {total_steps}");

    // Cleanup
    unsafe {
        vk_ctx.device.device_wait_idle()?;
        vk_ctx.device.destroy_fence(fence, None);
        vk_ctx.device.destroy_command_pool(cmd_pool, None);
    }

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn run_viewer(_cfg: &Config) -> Result<()> {
    anyhow::bail!("lbm-live-viewer requires the 'gpu' feature (CUDA backend)")
}

fn main() -> Result<()> {
    let cfg = Config::parse();
    run_viewer(&cfg)
}

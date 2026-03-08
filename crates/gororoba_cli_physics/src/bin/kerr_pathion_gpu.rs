//! Extreme Kerr Pathion Sink GPU Port (Priority #9)
//!
//! Validates that the Vulkan-accelerated Pathion heat sink can stabilize
//! an extreme Kerr (a=0.999) acoustic horizon simulation that would
//! otherwise rapidly diverge.

use ash::vk;
use lbm_vulkan::{Precision, VulkanContext, compute::GororobaEngine};
use std::time::Instant;

fn generate_kerr_field(n: usize, mass: f32) -> (Vec<f32>, Vec<f32>) {
    let total = n * n * n;
    let mut force = vec![0.0; total * 3];
    let mut f_init = vec![0.0; total * 19];
    let wf = [
        0.33333333, 0.05555556, 0.05555556, 0.05555556, 0.05555556, 0.05555556, 0.05555556,
        0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778,
        0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778,
    ];
    let center = n as f32 / 2.0;

    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let idx = x + n * (y + n * z);
                let dx = x as f32 - center;
                let dy = y as f32 - center;
                let dz = z as f32 - center;
                let r_sq = dx * dx + dy * dy + dz * dz;
                let r = r_sq.sqrt().max(0.1);

                // Extremely strong inward acceleration to simulate Kerr near-singularity
                // Normal LBM solvers crash when u > c_s (0.577).
                let mag = (50.0 * mass) / r_sq;

                force[idx] = -mag * dx / r;
                force[total + idx] = -mag * dy / r;
                force[2 * total + idx] = -mag * dz / r;

                for (q, &w) in wf.iter().enumerate() {
                    f_init[q * total + idx] = w;
                }
            }
        }
    }
    (f_init, force)
}

fn run_simulation(
    ctx: &VulkanContext,
    grid: u32,
    mass: f32,
    spin: f32,
    coupling: f32,
    damping: f32,
    steps: u32,
) -> Result<(f32, f64), Box<dyn std::error::Error>> {
    let grid_dim = (grid, grid, grid);
    let mut engine = GororobaEngine::new(ctx, grid_dim, (64, 64), Precision::FP32)?;

    let (f_init, force) = generate_kerr_field(grid as usize, mass);
    engine.upload_initial_state(ctx, &f_init, &force)?;

    let pool = unsafe {
        ctx.device.create_command_pool(
            &vk::CommandPoolCreateInfo {
                flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                ..Default::default()
            },
            None,
        )
    }?;
    let cmd = unsafe {
        ctx.device
            .allocate_command_buffers(&vk::CommandBufferAllocateInfo {
                command_pool: pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            })
    }?[0];
    let fence = unsafe {
        ctx.device.create_fence(
            &vk::FenceCreateInfo {
                flags: vk::FenceCreateFlags::SIGNALED,
                ..Default::default()
            },
            None,
        )
    }?;

    println!("    Running {} steps...", steps);
    let start = Instant::now();
    let mut max_rho_final = 0.0;

    for frame in 0..steps {
        unsafe {
            ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
            ctx.device.reset_fences(&[fence])?;
            ctx.device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo {
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    ..Default::default()
                },
            )?;
        }

        engine.step_with_sink(cmd, frame, mass, spin, coupling, damping)?;

        unsafe {
            ctx.device.end_command_buffer(cmd)?;
            ctx.device.queue_submit(
                ctx.queue,
                &[vk::SubmitInfo::default().command_buffers(&[cmd])],
                fence,
            )?;
        }

        if frame % 50 == 0 || frame == steps - 1 {
            unsafe { ctx.device.wait_for_fences(&[fence], true, u64::MAX)? };
            let (_, max_rho) = engine.get_diagnostics()?;
            if max_rho.is_nan() || max_rho > 1000.0 {
                println!("    [!] EXPLOSION at step {}: max_rho = {}", frame, max_rho);
                max_rho_final = max_rho;
                break;
            }
            max_rho_final = max_rho;
        }
    }

    unsafe { ctx.device.wait_for_fences(&[fence], true, u64::MAX)? };
    let total_accum = engine.read_accumulated_energy()?;
    let elapsed = start.elapsed();

    println!(
        "    Finished in {:.2?} (max_rho={:.4}, accum={:.4e})",
        elapsed, max_rho_final, total_accum
    );

    unsafe {
        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_command_pool(pool, None);
    }

    Ok((max_rho_final, total_accum))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    println!("=== GPU Pathion Heat Sink: Extreme Kerr (a=0.999) ===");

    let ctx = VulkanContext::new(true)?;
    let grid = 64;
    let steps = 1000;
    let mass = 1500.0; // M in grid units
    let spin = 0.999 * mass; // Extreme Kerr

    println!("\n[1] Control Run: NO Pathion Sink (damping=0.0)");
    let (rho_ctrl, _accum_ctrl) = run_simulation(&ctx, grid, mass, spin, 0.0, 0.0, steps)?;

    println!("\n[2] Stabilized Run: WITH Pathion Sink (coupling=1.0, damping=50.0)");
    let (rho_sink, accum_sink) = run_simulation(&ctx, grid, mass, spin, 1.0, 50.0, steps)?;

    println!("\n=== Verdict ===");
    println!("Control max rho: {}", rho_ctrl);
    println!("Sink max rho:    {}", rho_sink);
    println!("Sink Energy Abs: {:.4e}", accum_sink);

    let is_ctrl_exploded = rho_ctrl.is_nan() || rho_ctrl > 1000.0;
    let is_sink_stable = !rho_sink.is_nan() && rho_sink < 1000.0;

    if is_ctrl_exploded && is_sink_stable {
        println!("==> PASS (Sink stabilized an otherwise exploding extreme Kerr simulation)");
    } else if !is_ctrl_exploded {
        println!("==> INCONCLUSIVE (Control didn't explode. Increase mass or steps)");
    } else {
        println!("==> FAIL (Sink failed to stabilize the simulation)");
    }

    Ok(())
}

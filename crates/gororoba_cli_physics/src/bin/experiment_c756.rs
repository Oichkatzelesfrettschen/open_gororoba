//! Experiment C-756: Production Algebraic Synthesis.
//! Unifying Sedenion Associator Tension with LBM Fluid Dynamics.

use ash::vk;
use lbm_vulkan::{Precision, VulkanContext, compute::GororobaEngine};
use std::time::Instant;

fn generate_physics_data(n: usize) -> (Vec<f32>, Vec<f32>) {
    let total = n * n * n;
    // SoA layout: force[comp * total + idx], f_init[q * total + idx]
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
                let r_sq = dx * dx + dy * dy + dz * dz + 1.0;
                let mag = 200.0 / r_sq;
                force[idx] = -mag * dx / r_sq.sqrt();
                force[total + idx] = -mag * dy / r_sq.sqrt();
                force[2 * total + idx] = -mag * dz / r_sq.sqrt();
                for (q, &w) in wf.iter().enumerate() {
                    f_init[q * total + idx] = w;
                }
            }
        }
    }
    (f_init, force)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    println!("=== Experiment C-756: Production Synthesis ===");

    // Resolution: 160^3
    let grid_dim = (160, 160, 160);
    let ctx = VulkanContext::new(true)?;
    let mut engine = GororobaEngine::new(&ctx, grid_dim, (1280, 720), Precision::FP32)?;

    let (f_init, force) = generate_physics_data(160);
    engine.upload_initial_state(&ctx, &f_init, &force)?;

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

    std::fs::create_dir_all("data/artifacts/c756_frames")?;
    println!("Synthesizing 600 frames at high fidelity...");
    let start = Instant::now();

    for frame in 0..600 {
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
            engine.step(cmd, frame as u32)?;
            ctx.device.end_command_buffer(cmd)?;
            ctx.device.queue_submit(
                ctx.queue,
                &[vk::SubmitInfo {
                    command_buffer_count: 1,
                    p_command_buffers: &cmd,
                    ..Default::default()
                }],
                fence,
            )?;
            ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
            engine.save_frame(&format!(
                "data/artifacts/c756_frames/frame_{:04}.png",
                frame
            ))?;
        }
        if frame % 50 == 0 {
            println!("  Fidelity Progress: {}/600", frame);
        }
    }

    println!("Finalizing Video...");
    let _ = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-framerate",
            "30",
            "-i",
            "data/artifacts/c756_frames/frame_%04d.png",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "15",
            "data/artifacts/c756_production.mp4",
        ])
        .output();

    println!("Total Time: {:.2?}", start.elapsed());
    Ok(())
}

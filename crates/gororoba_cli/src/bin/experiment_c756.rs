//! Experiment C-756: Unified Holo-Algebraic Synthesis.
//!
//! Using the GororobaEngine to simulate Sedenion-modulated fluid dynamics
//! and render the result via dual-channel holographic projection.

use lbm_vulkan::{VulkanContext, compute::GororobaEngine};
use std::time::Instant;
use ash::vk;

fn generate_physics_data(n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut force = vec![0.0; n * n * n * 3];
    let mut f_init = vec![0.0; n * n * n * 19];
    let wf = [0.33333333, 0.05555556, 0.05555556, 0.05555556, 0.05555556, 0.05555556, 0.05555556, 0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778];
    let center = n as f32 / 2.0;

    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let idx_3 = (x + n * (y + n * z)) * 3;
                let idx_19 = (x + n * (y + n * z)) * 19;
                
                // 1. Kerr-Newman-ish Force
                let dx = x as f32 - center;
                let dy = y as f32 - center;
                let dz = z as f32 - center;
                let r_sq = dx*dx + dy*dy + dz*dz + 1.0;
                let mag = 500.0 / r_sq;
                force[idx_3] = -mag * dx / r_sq.sqrt();
                force[idx_3+1] = -mag * dy / r_sq.sqrt();
                force[idx_3+2] = -mag * dz / r_sq.sqrt();

                // 2. Initial Equilibrium State
                f_init[idx_19..idx_19 + 19].copy_from_slice(&wf);
            }
        }
    }
    (f_init, force)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    println!("=== Experiment C-756: Holo-Algebraic Synthesis ===");

    // 1. Initialize Unified Engine
    let ctx = VulkanContext::new(true)?;
    let params = ctx.get_scaling_parameters();
    let mut engine = GororobaEngine::new(&ctx, params.grid_dim, (1280, 720))?;

    // 2. Upload Physics
    let (f_init, force) = generate_physics_data(params.grid_dim.0 as usize);
    engine.upload_initial_state(&ctx, &f_init, &force)?;

    // 3. Command Pool for Loop
    let pool = unsafe { ctx.device.create_command_pool(&vk::CommandPoolCreateInfo { flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER, ..Default::default() }, None) }?;
    let cmd = unsafe { ctx.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo { command_pool: pool, level: vk::CommandBufferLevel::PRIMARY, command_buffer_count: 1, ..Default::default() }) }?[0];
    let fence = unsafe { ctx.device.create_fence(&vk::FenceCreateInfo { flags: vk::FenceCreateFlags::SIGNALED, ..Default::default() }, None) }?;

    // 4. Run Synthesis
    std::fs::create_dir_all("data/artifacts/c756_frames")?;
    println!("Synthesizing 300 frames...");
    let start = Instant::now();

    for frame in 0..300 {
        unsafe {
            ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
            ctx.device.reset_fences(&[fence])?;
            ctx.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo { flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT, ..Default::default() })?;

            // The engine now handles all internal steps (ZD -> LBM -> Render)
            engine.step(cmd, frame as u32);

            ctx.device.end_command_buffer(cmd)?;
            ctx.device.queue_submit(ctx.queue, &[vk::SubmitInfo { command_buffer_count: 1, p_command_buffers: &cmd, ..Default::default() }], fence)?;
            
            ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
            engine.save_frame(&format!("data/artifacts/c756_frames/frame_{:04}.png", frame))?;
        }
        if frame % 25 == 0 { println!("  Progress: {}/300", frame); }
    }

    println!("Synthesis Complete. Time: {:.2?}", start.elapsed());
    println!("Encoding data/artifacts/c756_simulation.mp4 ...");
    let _ = std::process::Command::new("ffmpeg")
        .args(["-y", "-framerate", "30", "-i", "data/artifacts/c756_frames/frame_%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "data/artifacts/c756_simulation.mp4"])
        .output();

    Ok(())
}

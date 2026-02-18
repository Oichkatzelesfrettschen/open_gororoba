//! Experiment C-756: Topological Entropy Locking (Video Export).
//!
//! Hypothesis: Spatially-modulated viscosity from Sedenion ZDs induces
//! stationary entropy features in a Kerr-Newman potential.
//!
//! This version exports a sequence of PNG frames for video encoding.

use lbm_vulkan::{VulkanContext, compute::{LbmComputePipeline, ZdGenPipeline, VulkanRenderer}};
use std::time::Instant;
use ash::vk;

fn generate_kerr_newman_force(n: usize) -> Vec<f32> {
    let mut force = vec![0.0; n * n * n * 3];
    let center = n as f32 / 2.0;
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let dx = x as f32 - center;
                let dy = y as f32 - center;
                let dz = z as f32 - center;
                let r = (dx*dx + dy*dy + dz*dz + 1.0).sqrt();
                let mag = 1000.0 / (r * r * r); 
                let idx = (x + n * (y + n * z)) * 3;
                force[idx] = -mag * dx;
                force[idx + 1] = -mag * dy;
                force[idx + 2] = -mag * dz;
            }
        }
    }
    force
}

fn generate_initial_state(n: usize) -> Vec<f32> {
    let mut f_init = vec![0.0; n * n * n * 19];
    let wf = [
        0.33333333,
        0.05555556, 0.05555556, 0.05555556, 0.05555556, 0.05555556, 0.05555556,
        0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778,
        0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778
    ];
    
    // Initialize with rho = 1.0, u = 0.0 (Equilibrium)
    // f_i = w_i * rho
    for i in 0..(n*n*n) {
        for q in 0..19 {
            f_init[i * 19 + q] = wf[q] * 1.0; 
        }
    }
    f_init
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    println!("=== Experiment C-756: Video Export ===");

    // 1. Initialize Vulkan
    let ctx = VulkanContext::new(true)?;
    let params = ctx.get_scaling_parameters();
    let n = params.grid_dim.0; 
    let width = 1280;
    let height = 720;

    // 2. Pipelines
    let mut lbm = LbmComputePipeline::new(&ctx, params.grid_dim)?;
    let zd_gen = ZdGenPipeline::new(&ctx, params.grid_dim, lbm.tau_buffer.as_ref().unwrap().0)?;
    
    // Renderer targets the entropy buffer
    let renderer = VulkanRenderer::new(&ctx, width, height, lbm.entropy_buffer.as_ref().unwrap().0)?;

    // 3. Setup Folders
    std::fs::create_dir_all("data/artifacts/c756_frames")?;

    // 4. Init Fields
    println!("Initializing Fields...");
    lbm.write_inputs(&[], &generate_kerr_newman_force(n as usize))?;
    lbm.write_state(&ctx, &generate_initial_state(n as usize))?;

    // 5. Command Infrastructure
    let pool_info = vk::CommandPoolCreateInfo { s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO, queue_family_index: ctx.queue_family_index, flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER, ..Default::default() };
    let pool = unsafe { ctx.device.create_command_pool(&pool_info, None) }?;
    let alloc_info = vk::CommandBufferAllocateInfo { s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO, command_pool: pool, level: vk::CommandBufferLevel::PRIMARY, command_buffer_count: 1, ..Default::default() };
    let cmd = unsafe { ctx.device.allocate_command_buffers(&alloc_info) }?[0];
    let fence = unsafe { ctx.device.create_fence(&vk::FenceCreateInfo { s_type: vk::StructureType::FENCE_CREATE_INFO, flags: vk::FenceCreateFlags::SIGNALED, ..Default::default() }, None) }?;

    // 6. Run Simulation & Export Frames
    println!("Running 60 frames...");
    let start = Instant::now();

    for frame in 0..60 {
        unsafe {
            ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
            ctx.device.reset_fences(&[fence])?;
            
            ctx.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo { s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO, flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT, ..Default::default() })?;

            // A. Update Viscosity (once)
            if frame == 0 { zd_gen.record_command_buffer(cmd); }

            // B. Sim Step (5 iterations per frame for smoothness)
            for _ in 0..5 { lbm.record_command_buffer(cmd); }

            // C. Render
            renderer.record_command_buffer(cmd, n, n, n, frame as f32);

            ctx.device.end_command_buffer(cmd)?;
            ctx.device.queue_submit(ctx.queue, &[vk::SubmitInfo { s_type: vk::StructureType::SUBMIT_INFO, command_buffer_count: 1, p_command_buffers: &cmd, ..Default::default() }], fence)?;
            
            // Wait for GPU to finish frame so we can save PNG
            ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
            renderer.save_frame(&format!("data/artifacts/c756_frames/frame_{:04}.png", frame))?;
        }
        if frame % 10 == 0 { 
            let entropy = lbm.read_entropy()?;
            let max_e = entropy.iter().cloned().fold(f32::NAN, f32::max);
            let avg_e: f32 = entropy.iter().sum::<f32>() / entropy.len() as f32;
            println!("  Frame {}/60 | Max Entropy: {:.6e} | Avg Entropy: {:.6e}", frame, max_e, avg_e);
        }
    }

    println!("Export Complete. Total Time: {:.2?}", start.elapsed());
    println!("To encode video, run:");
    println!("  ffmpeg -y -framerate 30 -i data/artifacts/c756_frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p data/artifacts/c756_simulation.mp4");

    Ok(())
}

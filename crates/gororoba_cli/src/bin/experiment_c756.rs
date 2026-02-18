//! Experiment C-756: Topological Entropy Locking (GPU Accelerated).
//!
//! Hypothesis: A spatially-modulated viscosity field derived from the Sedenion
//! Zero-Divisor topology (distance to ZD manifold) coupled to a Kerr-Newman
//! gravitational force field will induce non-trivial entropy production features.
//!
//! Pipeline:
//! 1. Initialize Vulkan Context.
//! 2. Initialize LBM Pipeline (allocates buffers).
//! 3. Initialize ZD Generator Pipeline (targets LBM's tau buffer).
//! 4. Execute ZD Generation (GPU Compute).
//! 5. Execute LBM Simulation (GPU Compute).
//! 6. Extract Entropy.

use lbm_vulkan::{VulkanContext, compute::{LbmComputePipeline, ZdGenPipeline}};
use std::time::Instant;
use std::path::Path;
use ash::vk;

fn generate_kerr_newman_force(n: usize) -> Vec<f32> {
    // Pseudo-Newtonian potential for Kerr-Newman
    let mut force = vec![0.0; n * n * n * 3];
    let center = n as f32 / 2.0;
    
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let dx = x as f32 - center;
                let dy = y as f32 - center;
                let dz = z as f32 - center;
                let r2 = dx*dx + dy*dy + dz*dz + 1.0; 
                let r = r2.sqrt();
                let mag = 1000.0 / (r * r * r); 
                
                let idx = (x + n * (y + n * z)) * 3;
                force[idx + 0] = -mag * dx;
                force[idx + 1] = -mag * dy;
                force[idx + 2] = -mag * dz;
            }
        }
    }
    force
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    println!("=== Experiment C-756: Topological Entropy Locking (GPU) ===");

    // 1. Initialize Vulkan
    let ctx = VulkanContext::new(true)?;
    println!("GPU: {} ({} MB)", ctx.caps.device_name, ctx.caps.vram_mb);
    
    let params = ctx.get_scaling_parameters();
    let n = params.grid_dim.0; 
    println!("Grid: {}^3", n);

    // 2. Init LBM Pipeline
    let mut lbm = LbmComputePipeline::new(&ctx, params.grid_dim)?;

    // 3. Init ZD Generator (Algebraic Field)
    // We borrow the tau buffer from LBM to write into it.
    let tau_buffer_handle = lbm.tau_buffer.as_ref().unwrap().0;
    let zd_gen = ZdGenPipeline::new(&ctx, params.grid_dim, tau_buffer_handle)?;

    // 4. Generate Forces (Host)
    println!("Generating Kerr-Newman Force Field (Host)...");
    let force_field = generate_kerr_newman_force(n as usize);
    // Write forces. Note: write_inputs only writes provided slices.
    // We only provide force, leaving tau untouched (GPU will write it).
    // We pass a dummy empty slice for tau to write_inputs if needed, or update write_inputs api.
    // My previous implementation of write_inputs checks `if let` for both.
    // I can pass an empty slice for tau.
    lbm.write_inputs(&[], &force_field)?;

    // 5. Run Pipelines
    let pool_info = vk::CommandPoolCreateInfo {
        s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
        queue_family_index: ctx.queue_family_index,
        flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        ..Default::default()
    };
    let pool = unsafe { ctx.device.create_command_pool(&pool_info, None) }?;
    let alloc_info = vk::CommandBufferAllocateInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
        command_pool: pool,
        level: vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: 1,
        ..Default::default()
    };
    let cmd = unsafe { ctx.device.allocate_command_buffers(&alloc_info) }?[0];
    
    let fence_info = vk::FenceCreateInfo {
        s_type: vk::StructureType::FENCE_CREATE_INFO,
        flags: vk::FenceCreateFlags::SIGNALED,
        ..Default::default()
    };
    let fence = unsafe { ctx.device.create_fence(&fence_info, None) }?;

    println!("Running GPU Tasks...");
    let start = Instant::now();

    unsafe {
        ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
        ctx.device.reset_fences(&[fence])?;
        
        let begin_info = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };
        ctx.device.begin_command_buffer(cmd, &begin_info)?;

        // A. Generate Algebraic Viscosity Field
        // Barrier handled inside record_command_buffer? No, I added one at the end of zd_gen.
        zd_gen.record_command_buffer(cmd);

        // B. Run LBM Steps
        for _ in 0..100 {
            lbm.record_command_buffer(cmd);
        }

        ctx.device.end_command_buffer(cmd)?;
        
        let submit = vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            command_buffer_count: 1,
            p_command_buffers: &cmd,
            ..Default::default()
        };
        ctx.device.queue_submit(ctx.queue, &[submit], fence)?;
    }
    unsafe { ctx.device.wait_for_fences(&[fence], true, u64::MAX)?; }
    
    let duration = start.elapsed();
    println!("Total GPU Time: {:.2?}", duration);

    // 6. Extract Entropy
    println!("Extracting Entropy Production...");
    let entropy = lbm.read_entropy()?;
    let mean_entropy: f32 = entropy.iter().sum::<f32>() / entropy.len() as f32;
    let max_entropy = entropy.iter().fold(0.0f32, |a: f32, &b| a.max(b));
    
    println!("Mean Entropy Production: {:.6e}", mean_entropy);
    println!("Max Entropy Production:  {:.6e}", max_entropy);

    // Save
    let out_path = Path::new("data/csv/c756_entropy_locking_gpu.csv");
    use std::io::Write;
    let mut file = std::fs::File::create(out_path)?;
    writeln!(file, "x,y,entropy")?;
    let z_slice = n / 2;
    for y in 0..n {
        for x in 0..n {
            let idx = x + n * (y + n * z_slice);
            writeln!(file, "{},{},{:.6e}", x, y, entropy[idx as usize])?;
        }
    }
    println!("Saved center slice to {}", out_path.display());

    Ok(())
}

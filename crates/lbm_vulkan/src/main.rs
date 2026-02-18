use lbm_vulkan::VulkanContext;
use lbm_vulkan::compute::LbmComputePipeline;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    println!("Initializing Vulkan Context...");
    let ctx = VulkanContext::new(true)?; // Validation enabled for safety
    
    println!("GPU Detected: {}", ctx.caps.device_name);
    println!("VRAM: {} MB", ctx.caps.vram_mb);
    println!("Tier: {:?}", ctx.caps.tier);

    let params = ctx.get_scaling_parameters();
    println!("Dynamic Scaling Parameters:");
    println!("  Grid: {}x{}x{}", params.grid_dim.0, params.grid_dim.1, params.grid_dim.2);
    println!("  Precision: {:?}", params.precision);
    println!("  Math Complexity: {:?}", params.math_complexity);
    println!("  Render Scale: {:.2}", params.render_resolution_scale);

    let mut pipeline = LbmComputePipeline::new(&ctx, params.grid_dim)?;
    
    // Command Buffer recording
    let pool_info = ash::vk::CommandPoolCreateInfo {
        s_type: ash::vk::StructureType::COMMAND_POOL_CREATE_INFO,
        queue_family_index: ctx.queue_family_index,
        flags: ash::vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        ..Default::default()
    };
    let command_pool = unsafe { ctx.device.create_command_pool(&pool_info, None) }?;
    
    let alloc_info = ash::vk::CommandBufferAllocateInfo {
        s_type: ash::vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
        command_pool,
        level: ash::vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: 1,
        ..Default::default()
    };
    let command_buffer = unsafe { ctx.device.allocate_command_buffers(&alloc_info) }?[0];

    // Record once
    let begin_info = ash::vk::CommandBufferBeginInfo {
        s_type: ash::vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
        flags: ash::vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,
        ..Default::default()
    };
    
    unsafe {
        ctx.device.begin_command_buffer(command_buffer, &begin_info)?;
        // Record compute dispatch
        pipeline.record_command_buffer(command_buffer);
        ctx.device.end_command_buffer(command_buffer)?;
    }

    // Fence for sync
    let fence_info = ash::vk::FenceCreateInfo {
        s_type: ash::vk::StructureType::FENCE_CREATE_INFO,
        flags: ash::vk::FenceCreateFlags::SIGNALED,
        ..Default::default()
    };
    let fence = unsafe { ctx.device.create_fence(&fence_info, None) }?;

    println!("Starting Simulation Loop (1000 steps)...");
    let start = Instant::now();
    let steps = 1000;
    
    for _ in 0..steps {
        unsafe {
            ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
            ctx.device.reset_fences(&[fence])?;
            
            let submit_info = ash::vk::SubmitInfo {
                s_type: ash::vk::StructureType::SUBMIT_INFO,
                command_buffer_count: 1,
                p_command_buffers: &command_buffer,
                ..Default::default()
            };
            
            ctx.device.queue_submit(ctx.queue, &[submit_info], fence)?;
        }
    }
    
    // Wait for last
    unsafe { ctx.device.wait_for_fences(&[fence], true, u64::MAX)? };
    
    let duration = start.elapsed();
    let total_cells = (params.grid_dim.0 * params.grid_dim.1 * params.grid_dim.2) as u64;
    let total_updates = total_cells * steps;
    let mlups = (total_updates as f64 / duration.as_secs_f64()) / 1_000_000.0;
    
    println!("Simulation Complete.");
    println!("Time: {:.2?}", duration);
    println!("MLUPS: {:.2}", mlups);
    
    Ok(())
}

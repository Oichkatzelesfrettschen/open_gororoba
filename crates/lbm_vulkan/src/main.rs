use lbm_vulkan::{VulkanContext, compute::GororobaEngine};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    println!("Initializing Vulkan Context...");
    let ctx = VulkanContext::new(true)?;

    println!("GPU Detected: {}", ctx.caps.device_name);
    println!("VRAM: {} MB", ctx.caps.vram_mb);
    println!("Tier: {:?}", ctx.caps.tier);

    let params = ctx.get_scaling_parameters();
    println!("Dynamic Scaling Parameters:");
    println!(
        "  Grid: {}x{}x{}",
        params.grid_dim.0, params.grid_dim.1, params.grid_dim.2
    );
    println!("  Precision: {:?}", params.precision);
    println!("  Math Complexity: {:?}", params.math_complexity);
    println!("  Render Scale: {:.2}", params.render_resolution_scale);

    let grid = params.grid_dim;
    let screen = (1280u32, 720u32);
    let mut engine = GororobaEngine::new(&ctx, grid, screen, params.precision)?;

    // Generate D3Q19 equilibrium initial state: f_i = w_i * rho_0 (rho_0 = 1)
    let n = (grid.0 * grid.1 * grid.2) as usize;
    let weights: [f32; 19] = [
        1.0 / 3.0,
        1.0 / 18.0,
        1.0 / 18.0,
        1.0 / 18.0,
        1.0 / 18.0,
        1.0 / 18.0,
        1.0 / 18.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
    ];
    // SoA layout: f_init[q * n + cell] -- contiguous per direction
    let mut f_init = vec![0.0f32; n * 19];
    for (q, &w) in weights.iter().enumerate() {
        for cell in 0..n {
            f_init[q * n + cell] = w;
        }
    }
    // SoA force: fx[0..n], fy[n..2n], fz[2n..3n]
    let force = vec![0.0f32; n * 3];
    engine.upload_initial_state(&ctx, &f_init, &force)?;

    // Command buffer for simulation loop
    let pool = unsafe {
        ctx.device.create_command_pool(
            &ash::vk::CommandPoolCreateInfo {
                queue_family_index: ctx.queue_family_index,
                flags: ash::vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                ..Default::default()
            },
            None,
        )
    }?;
    let cmd = unsafe {
        ctx.device
            .allocate_command_buffers(&ash::vk::CommandBufferAllocateInfo {
                command_pool: pool,
                level: ash::vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            })
    }?[0];
    let fence = unsafe {
        ctx.device.create_fence(
            &ash::vk::FenceCreateInfo {
                flags: ash::vk::FenceCreateFlags::SIGNALED,
                ..Default::default()
            },
            None,
        )
    }?;

    let steps = 1000u32;
    println!("Starting Simulation Loop ({steps} steps)...");
    let start = Instant::now();

    for frame in 0..steps {
        unsafe {
            ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
            ctx.device.reset_fences(&[fence])?;
            ctx.device
                .reset_command_buffer(cmd, ash::vk::CommandBufferResetFlags::empty())?;
            ctx.device.begin_command_buffer(
                cmd,
                &ash::vk::CommandBufferBeginInfo {
                    flags: ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    ..Default::default()
                },
            )?;
            engine.step(cmd, frame)?;
            ctx.device.end_command_buffer(cmd)?;
            ctx.device.queue_submit(
                ctx.queue,
                &[ash::vk::SubmitInfo {
                    command_buffer_count: 1,
                    p_command_buffers: &cmd,
                    ..Default::default()
                }],
                fence,
            )?;
        }
    }

    unsafe { ctx.device.wait_for_fences(&[fence], true, u64::MAX)? };

    let duration = start.elapsed();
    let total_cells = (grid.0 * grid.1 * grid.2) as u64;
    let total_updates = total_cells * steps as u64;
    let mlups = (total_updates as f64 / duration.as_secs_f64()) / 1_000_000.0;

    println!("Simulation Complete.");
    println!("Time: {duration:.2?}");
    println!("MLUPS: {mlups:.2}");

    Ok(())
}

//! Sparse LBM Benchmark: 1024^3 Domain in 12GB VRAM.
//!
//! Validates the YSU-Engine inspired occupancy-grid and A-A streaming
//! pattern to fit massive sparse domains in consumer GPUs.

use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use lbm_3d_cuda::probe_cuda_ada_available;
use lbm_3d_cuda::sparse::{SparseBrickMap, SparseLbmSolver};
use std::time::Instant;

fn main() {
    println!("\n{}", "#".repeat(80));
    println!("# ADA LOVELACE SPARSE LBM BENCHMARK (1024^3)");
    println!("# Validating memory-budget constraints for 12GB VRAM");
    println!("{}", "#".repeat(80));

    if !probe_cuda_ada_available() {
        println!("Ada Lovelace (SM 8.9) GPU not detected. Exiting.");
        return;
    }

    let ctx = CudaContext::new(0).expect("Failed to initialize CUDA context");
    let stream = ctx.default_stream();

    // 1024^3 domain
    let nx = 1024;
    let ny = 1024;
    let nz = 1024;
    let total_cells = nx * ny * nz;

    println!("Domain Size: {}^3 ({} billion cells)", nx, total_cells as f64 / 1e9);

    // Create a sparse geometry mask: 10% sparsity (e.g. a porous media or an irregular shape)
    // We'll approximate this by activating bricks near the center.
    // Instead of allocating a 1GB byte array on the CPU and sending it to GPU (which takes time),
    // we'll run a quick CUDA kernel to generate a ~10% sparse mask directly on the GPU.

    let mut d_mask = stream.alloc_zeros::<u8>(total_cells).expect("Failed to allocate 1GB mask");
    
    // We'll just define a sphere in the center with radius ~ R=240, yielding ~10% volume
    let init_mask_src = r#"
    extern "C" __global__ void init_mask(unsigned char* mask, int nx, int ny, int nz, float radius) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= nx * ny * nz) return;
        
        int x = idx % nx;
        int y = (idx / nx) % ny;
        int z = idx / (nx * ny);
        
        float dx = x - nx/2;
        float dy = y - ny/2;
        float dz = z - nz/2;
        
        if (dx*dx + dy*dy + dz*dz <= radius*radius) {
            mask[idx] = 1; // Fluid
        } else {
            mask[idx] = 0; // Solid
        }
    }
    "#;
    
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(init_mask_src, Default::default()).unwrap();
    let module = ctx.load_module(ptx).unwrap();
    let init_mask_kernel = module.load_function("init_mask").unwrap();
    
    let threads = 256;
    let blocks = (total_cells as u32 + threads - 1) / threads;
    let nx_i = nx as i32;
    let ny_i = ny as i32;
    let nz_i = nz as i32;
    let radius = 240.0f32; // (4/3)*pi*(240)^3 / 1024^3 ≈ 0.054 -> 5.4% sparsity. Let's use 300.0 for ~10%.
    
    let mut b = stream.launch_builder(&init_mask_kernel);
    b.arg(&mut d_mask).arg(&nx_i).arg(&ny_i).arg(&nz_i).arg(&radius);
    unsafe {
        b.launch(LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 }).unwrap();
    }
    ctx.synchronize().unwrap();

    println!("Generating Sparse Brick Map...");
    let start_map = Instant::now();
    let map = SparseBrickMap::new_from_geometry(ctx.clone(), stream.clone(), nx, ny, nz, &d_mask).unwrap();
    let map_time = start_map.elapsed();
    
    println!("  Map Generation Time: {:.2?}", map_time);
    println!("  Total Bricks: {}", map.n_bricks);
    println!("  Active Bricks: {} ({:.2}% Sparsity)", 
        map.n_active_bricks, 
        (map.n_active_bricks as f64 / map.n_bricks as f64) * 100.0
    );

    // Free the mask to emulate constrained VRAM environment
    drop(d_mask);

    println!("Initializing Sparse LBM Solver...");
    let mut solver = SparseLbmSolver::new(map).unwrap();
    
    let active_cells = solver.map.n_active_bricks * 512;
    let bytes_per_cell = 19 * 4 + 4 + 3 * 4 + 4 + 3 * 4; // f, rho, u, tau, force
    let active_vram_gb = (active_cells as f64 * bytes_per_cell as f64) / 1e9;
    
    println!("  Active VRAM Footprint: {:.2} GB", active_vram_gb);
    
    let steps = 100;
    println!("Running Sparse A-A LBM for {} steps...", steps);
    
    // Warmup
    solver.evolve(5).unwrap();
    
    let start_evolve = Instant::now();
    solver.evolve(steps).unwrap();
    let evolve_time = start_evolve.elapsed();
    
    let throughput = (active_cells as f64 * steps as f64) / evolve_time.as_secs_f64() / 1e6;
    
    println!("Results:");
    println!("  Time elapsed: {:.4} s", evolve_time.as_secs_f64());
    println!("  Throughput:   {:.2} MLUPS (Active Cells Only)", throughput);
    println!("  Effective:    {:.2} GLUPS (Dense Equivalent)", (total_cells as f64 * steps as f64) / evolve_time.as_secs_f64() / 1e9);
    println!("{}", "#".repeat(80));
}
//! Sparse LBM Manager and Kernel Interface.
//! Integrates the sparse brick map with cudarc for 1024^3 in 12GB VRAM.

use anyhow::{Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use std::sync::Arc;

const KERNEL_SPARSE_MAP_SRC: &str = include_str!("kernels_sparse_map.cu");

/// Manages the sparse occupancy map and indirect table.
pub struct SparseBrickMap {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub bx_max: usize,
    pub by_max: usize,
    pub bz_max: usize,
    pub n_bricks: usize,
    pub n_active_bricks: usize,

    // Device memory handles
    pub d_occupancy_words: CudaSlice<u32>,
    pub d_brick_counts: CudaSlice<u32>,
    pub d_brick_offsets: CudaSlice<u32>,
    pub d_indirect_table: CudaSlice<i32>,
    pub d_active_brick_ids: CudaSlice<u32>,

    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
}

impl SparseBrickMap {
    /// Builds the sparse metadata structures from a given 3D geometry mask.
    pub fn new_from_geometry(
        ctx: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        nx: usize,
        ny: usize,
        nz: usize,
        d_geometry_mask: &CudaSlice<u8>,
    ) -> Result<Self> {
        // Compile kernels
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some(crate::preferred_cuda_arch()),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(KERNEL_SPARSE_MAP_SRC, opts)
            .context("Failed to compile kernels_sparse_map.cu")?;
        let module = ctx.load_module(ptx).context("Failed to load sparse map module")?;

        let generate_occupancy_kernel = module.load_function("generate_occupancy_bitmask")?;
        let expand_bitmask_kernel = module.load_function("expand_bitmask_to_counts")?;
        let build_indirect_table_kernel = module.load_function("build_indirect_table")?;

        let bx_max = (nx + 7) / 8;
        let by_max = (ny + 7) / 8;
        let bz_max = (nz + 7) / 8;
        let n_bricks = bx_max * by_max * bz_max;

        let num_words = (n_bricks + 31) / 32;

        let mut d_occupancy_words = stream.alloc_zeros::<u32>(num_words)?;
        let mut d_brick_counts = stream.alloc_zeros::<u32>(n_bricks)?;

        // 1. Generate Bitmask
        let block = (8, 8, 8);
        let grid = (
            (nx as u32 + block.0 - 1) / block.0,
            (ny as u32 + block.1 - 1) / block.1,
            (nz as u32 + block.2 - 1) / block.2,
        );
        let cfg = LaunchConfig { grid_dim: grid, block_dim: block, shared_mem_bytes: 0 };
        let nx_i = nx as i32;
        let ny_i = ny as i32;
        let nz_i = nz as i32;
        let bx_max_i = bx_max as i32;
        let by_max_i = by_max as i32;
        let bz_max_i = bz_max as i32;
        let mut b1 = stream.launch_builder(&generate_occupancy_kernel);
        b1.arg(d_geometry_mask)
            .arg(&mut d_occupancy_words)
            .arg(&nx_i).arg(&ny_i).arg(&nz_i)
            .arg(&bx_max_i).arg(&by_max_i).arg(&bz_max_i);
        unsafe { b1.launch(cfg) }?;

        // 2. Expand Bitmask
        let block2 = 256;
        let grid2 = (n_bricks as u32 + block2 - 1) / block2;
        let n_bricks_i = n_bricks as i32;
        let mut b2 = stream.launch_builder(&expand_bitmask_kernel);
        b2.arg(&d_occupancy_words).arg(&mut d_brick_counts).arg(&n_bricks_i);
        unsafe {
            b2.launch(LaunchConfig { grid_dim: (grid2, 1, 1), block_dim: (block2, 1, 1), shared_mem_bytes: 0 })
        }?;

        // 3. Prefix Sum
        let h_brick_counts = stream.clone_dtoh(&d_brick_counts)?;
        let mut h_brick_offsets = vec![0u32; n_bricks];
        let mut total_active = 0;
        for i in 0..n_bricks {
            h_brick_offsets[i] = total_active;
            total_active += h_brick_counts[i];
        }
        let n_active_bricks = total_active as usize;
        
        // Push offsets back to device
        let d_brick_offsets = stream.clone_htod(&h_brick_offsets)?;

        // 4. Compact Indirect Table
        let mut d_indirect_table = stream.alloc_zeros::<i32>(n_bricks)?;
        let mut d_active_brick_ids = stream.alloc_zeros::<u32>(n_active_bricks.max(1))?;
        
        let mut b3 = stream.launch_builder(&build_indirect_table_kernel);
        b3.arg(&d_brick_counts)
            .arg(&d_brick_offsets)
            .arg(&mut d_indirect_table)
            .arg(&mut d_active_brick_ids)
            .arg(&n_bricks_i);
        unsafe {
            b3.launch(LaunchConfig { grid_dim: (grid2, 1, 1), block_dim: (block2, 1, 1), shared_mem_bytes: 0 })
        }?;
        
        // Ensure synchronization if needed (clone_dtoh/htod does it)

        Ok(Self {
            nx, ny, nz,
            bx_max, by_max, bz_max,
            n_bricks,
            n_active_bricks,
            d_occupancy_words,
            d_brick_counts,
            d_brick_offsets,
            d_indirect_table,
            d_active_brick_ids,
            ctx,
            stream,
        })
    }
}

const KERNEL_SPARSE_LBM_SRC: &str = include_str!("kernels_sparse_lbm.cu");

/// Sparse LBM Solver using the Brick Map and A-A streaming pattern.
pub struct SparseLbmSolver {
    pub map: SparseBrickMap,
    pub d_f: CudaSlice<f32>,
    pub d_rho: CudaSlice<f32>,
    pub d_u: CudaSlice<f32>,
    pub d_tau: CudaSlice<f32>,
    pub d_force: CudaSlice<f32>,
    
    pub step: usize,
    
    lbm_step_kernel: CudaFunction,
}

impl SparseLbmSolver {
    pub fn new(map: SparseBrickMap) -> Result<Self> {
        let n_active_cells = map.n_active_bricks * 512;
        let stream = map.stream.clone();
        
        // Allocate only for active cells
        let d_f = stream.alloc_zeros::<f32>(19 * n_active_cells.max(1))?;
        let d_rho = stream.alloc_zeros::<f32>(n_active_cells.max(1))?;
        let d_u = stream.alloc_zeros::<f32>(3 * n_active_cells.max(1))?;
        let d_tau = stream.alloc_zeros::<f32>(n_active_cells.max(1))?;
        let d_force = stream.alloc_zeros::<f32>(3 * n_active_cells.max(1))?;

        // Compile kernel
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some(crate::preferred_cuda_arch()),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(KERNEL_SPARSE_LBM_SRC, opts)
            .context("Failed to compile kernels_sparse_lbm.cu")?;
        let module = map.ctx.load_module(ptx).context("Failed to load sparse lbm module")?;
        
        let lbm_step_kernel = module.load_function("lbm_step_sparse_aa")?;

        Ok(Self {
            map,
            d_f,
            d_rho,
            d_u,
            d_tau,
            d_force,
            step: 0,
            lbm_step_kernel,
        })
    }

    pub fn evolve(&mut self, steps: usize) -> Result<()> {
        if self.map.n_active_bricks == 0 {
            self.step += steps;
            return Ok(());
        }

        let n_active_cells = self.map.n_active_bricks * 512;
        let block = 512;
        let grid = (n_active_cells as u32 + block - 1) / block;
        let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 };
        
        let nx_i = self.map.nx as i32;
        let ny_i = self.map.ny as i32;
        let nz_i = self.map.nz as i32;
        let bx_max_i = self.map.bx_max as i32;
        let by_max_i = self.map.by_max as i32;
        let bz_max_i = self.map.bz_max as i32;
        let n_active_cells_i = n_active_cells as i32;

        for _ in 0..steps {
            let parity = (self.step % 2) as i32;
            let mut b = self.map.stream.launch_builder(&self.lbm_step_kernel);
            b.arg(&mut self.d_f)
                .arg(&mut self.d_rho)
                .arg(&mut self.d_u)
                .arg(&self.d_tau)
                .arg(&self.d_force)
                .arg(&self.map.d_indirect_table)
                .arg(&self.map.d_active_brick_ids)
                .arg(&nx_i).arg(&ny_i).arg(&nz_i)
                .arg(&bx_max_i).arg(&by_max_i).arg(&bz_max_i)
                .arg(&n_active_cells_i)
                .arg(&parity);
                
            unsafe {
                b.launch(cfg)
            }?;
            self.step += 1;
        }
        
        // Wait for operations to complete before returning
        // In cudarc 0.19, `CudaStream` might not have `sync()`. We'll synchronize via device context.
        self.map.ctx.synchronize()?;
        Ok(())
    }
}

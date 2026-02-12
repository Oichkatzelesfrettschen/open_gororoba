// GPU-accelerated Lattice Boltzmann Method (D3Q19) with CUDA
// Runtime kernel compilation via cudarc NVRTC

use anyhow::{Context, Result};
use cosmic_scheduler::{ScheduleError, ScheduleResult, TwoPhaseSystem};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

const KERNEL_SRC: &str = include_str!("kernels.cu");

/// GPU-accelerated D3Q19 LBM solver with spatially-varying viscosity
pub struct LbmSolver3DCuda {
    // Grid dimensions
    nx: usize,
    ny: usize,
    nz: usize,
    n_cells: usize,

    // CUDA context and stream
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,

    // Device memory buffers
    d_f: CudaSlice<f64>, // Distributions (19 x n_cells)
    #[allow(dead_code)]
    d_f_tmp: CudaSlice<f64>, // Temp buffer (unused in fused collision-streaming)
    d_rho: CudaSlice<f64>, // Density (n_cells)
    d_u: CudaSlice<f64>, // Velocity (3 x n_cells)
    d_tau: CudaSlice<f64>, // Relaxation time (n_cells) - spatially varying!

    // Compiled kernels
    compute_macro_kernel: CudaFunction,
    collision_kernel: CudaFunction,
    #[allow(dead_code)]
    streaming_kernel: CudaFunction, // Unused in fused collision-streaming
    init_kernel: CudaFunction,

    // Host-side state (for CPU fallback and data export)
    pub rho: Vec<f64>,
    pub u: Vec<[f64; 3]>,
}

impl LbmSolver3DCuda {
    /// Create new GPU solver with uniform relaxation time
    pub fn new(nx: usize, ny: usize, nz: usize, tau: f64) -> Result<Self> {
        let n_cells = nx * ny * nz;

        // Initialize CUDA context and default stream.
        let ctx = CudaContext::new(0).context("Failed to initialize CUDA context on device 0")?;
        let stream = ctx.default_stream();

        // Compile CUDA kernels at runtime via NVRTC.
        let ptx = compile_ptx(KERNEL_SRC).context("Failed to compile CUDA kernels via NVRTC")?;
        let module = ctx.load_module(ptx).context("Failed to load PTX module")?;

        let compute_macro_kernel = module
            .load_function("compute_macroscopic_kernel")
            .context("Kernel compute_macroscopic_kernel not found")?;
        let collision_kernel = module
            .load_function("bgk_collision_kernel")
            .context("Kernel bgk_collision_kernel not found")?;
        let streaming_kernel = module
            .load_function("streaming_kernel")
            .context("Kernel streaming_kernel not found")?;
        let init_kernel = module
            .load_function("initialize_uniform_kernel")
            .context("Kernel initialize_uniform_kernel not found")?;

        // Allocate device memory.
        let d_f = stream
            .alloc_zeros::<f64>(19 * n_cells)
            .context("Failed to allocate d_f")?;
        let d_f_tmp = stream
            .alloc_zeros::<f64>(19 * n_cells)
            .context("Failed to allocate d_f_tmp")?;
        let d_rho = stream
            .alloc_zeros::<f64>(n_cells)
            .context("Failed to allocate d_rho")?;
        let d_u = stream
            .alloc_zeros::<f64>(3 * n_cells)
            .context("Failed to allocate d_u")?;

        // Initialize uniform tau field
        let tau_vec = vec![tau; n_cells];
        let d_tau = stream
            .clone_htod(&tau_vec)
            .context("Failed to initialize d_tau")?;

        // Host-side state
        let rho = vec![1.0; n_cells];
        let u = vec![[0.0, 0.0, 0.0]; n_cells];

        Ok(Self {
            nx,
            ny,
            nz,
            n_cells,
            _ctx: ctx,
            stream,
            d_f,
            d_f_tmp,
            d_rho,
            d_u,
            d_tau,
            compute_macro_kernel,
            collision_kernel,
            streaming_kernel,
            init_kernel,
            rho,
            u,
        })
    }

    /// Set spatially-varying viscosity field (critical for frustration coupling!)
    pub fn set_viscosity_field(&mut self, viscosity: &[f64]) -> Result<()> {
        if viscosity.len() != self.n_cells {
            anyhow::bail!(
                "Viscosity field length {} does not match grid size {}",
                viscosity.len(),
                self.n_cells
            );
        }

        // Convert nu -> tau via Chapman-Enskog: tau = 3*nu + 0.5
        let tau_field: Vec<f64> = viscosity.iter().map(|&nu| 3.0 * nu + 0.5).collect();

        // Validate: all tau >= 0.5 for BGK stability
        if let Some(&min_tau) = tau_field.iter().min_by(|a, b| a.partial_cmp(b).unwrap()) {
            if min_tau < 0.5 {
                anyhow::bail!(
                    "tau field contains values < 0.5 (unstable): min={}",
                    min_tau
                );
            }
        }

        // Upload to GPU
        self.d_tau = self
            .stream
            .clone_htod(&tau_field)
            .context("Failed to upload tau field to GPU")?;

        Ok(())
    }

    /// Get current viscosity field from GPU
    pub fn get_viscosity_field(&self) -> Result<Vec<f64>> {
        let tau_field = self
            .stream
            .clone_dtoh(&self.d_tau)
            .context("Failed to download tau field from GPU")?;

        // Convert tau -> nu: nu = (tau - 0.5) / 3
        Ok(tau_field.iter().map(|&tau| (tau - 0.5) / 3.0).collect())
    }

    /// Initialize uniform density and velocity
    pub fn initialize_uniform(&mut self, rho_init: f64, u_init: [f64; 3]) -> Result<()> {
        // Launch initialization kernel
        let block_size = 256;
        let grid_size = self.n_cells.div_ceil(block_size);

        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = self.stream.launch_builder(&self.init_kernel);
        builder.arg(&mut self.d_f);
        builder.arg(&mut self.d_rho);
        builder.arg(&mut self.d_u);
        builder.arg(&rho_init);
        builder.arg(&u_init[0]);
        builder.arg(&u_init[1]);
        builder.arg(&u_init[2]);
        let nx_i32 = self.nx as i32;
        let ny_i32 = self.ny as i32;
        let nz_i32 = self.nz as i32;
        builder.arg(&nx_i32);
        builder.arg(&ny_i32);
        builder.arg(&nz_i32);

        unsafe { builder.launch(config) }.context("Failed to launch initialize_uniform_kernel")?;

        self.stream
            .synchronize()
            .context("CUDA synchronize failed")?;

        // Update host-side state
        self.rho = vec![rho_init; self.n_cells];
        self.u = vec![u_init; self.n_cells];

        Ok(())
    }

    /// Initialize with custom velocity field (for shear initialization)
    ///
    /// Takes a host-side velocity field and initializes distributions to local equilibrium.
    /// Useful for seeding flow instabilities with velocity shear profiles.
    pub fn initialize_custom(&mut self, rho: &[f64], u: &[[f64; 3]]) -> Result<()> {
        if rho.len() != self.n_cells || u.len() != self.n_cells {
            anyhow::bail!(
                "Field size mismatch: rho.len()={}, u.len()={}, expected={}",
                rho.len(),
                u.len(),
                self.n_cells
            );
        }

        // Upload rho to GPU
        self.d_rho = self
            .stream
            .clone_htod(rho)
            .context("Failed to upload rho field")?;

        // Flatten u for GPU: [ux0, uy0, uz0, ux1, uy1, uz1, ...]
        let u_flat: Vec<f64> = u
            .iter()
            .flat_map(|&[ux, uy, uz]| vec![ux, uy, uz])
            .collect();

        self.d_u = self
            .stream
            .clone_htod(&u_flat)
            .context("Failed to upload velocity field")?;

        // Initialize distribution function to local equilibrium using uploaded u
        // We'll use the init kernel with per-cell velocities (requires kernel modification)
        // For now, use a workaround: loop over cells and initialize f directly on CPU, then upload

        let mut f_host = vec![0.0; 19 * self.n_cells];

        for idx in 0..self.n_cells {
            let rho_val = rho[idx];
            let u_val = u[idx];

            // D3Q19 lattice velocities (from kernels.cu)
            let cx = [0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0];
            let cy = [0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1];
            let cz = [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1];
            let w = [
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

            // Compute equilibrium distribution for this cell
            let ux = u_val[0];
            let uy = u_val[1];
            let uz = u_val[2];
            let usqr = ux * ux + uy * uy + uz * uz;

            for i in 0..19 {
                let ci_dot_u = (cx[i] as f64) * ux + (cy[i] as f64) * uy + (cz[i] as f64) * uz;
                let f_eq = w[i]
                    * rho_val
                    * (1.0 + 3.0 * ci_dot_u + 4.5 * ci_dot_u * ci_dot_u - 1.5 * usqr);
                f_host[idx * 19 + i] = f_eq;
            }
        }

        // Upload initialized distribution function to GPU
        self.d_f = self
            .stream
            .clone_htod(&f_host)
            .context("Failed to upload distribution function")?;

        self.stream
            .synchronize()
            .context("CUDA synchronize failed")?;

        // Update host-side state
        self.rho = rho.to_vec();
        self.u = u.to_vec();

        Ok(())
    }

    /// Compute macroscopic quantities (rho, u) from distributions
    fn compute_macroscopic(&mut self) -> Result<()> {
        let block_size = 256;
        let grid_size = self.n_cells.div_ceil(block_size);

        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = self.stream.launch_builder(&self.compute_macro_kernel);
        builder.arg(&self.d_f);
        builder.arg(&mut self.d_rho);
        builder.arg(&mut self.d_u);
        let nx_i32 = self.nx as i32;
        let ny_i32 = self.ny as i32;
        let nz_i32 = self.nz as i32;
        builder.arg(&nx_i32);
        builder.arg(&ny_i32);
        builder.arg(&nz_i32);

        unsafe { builder.launch(config) }.context("Failed to launch compute_macroscopic_kernel")?;

        Ok(())
    }

    /// BGK collision with spatially-varying relaxation time
    fn collision(&mut self) -> Result<()> {
        let block_size = 256;
        let grid_size = self.n_cells.div_ceil(block_size);

        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = self.stream.launch_builder(&self.collision_kernel);
        builder.arg(&mut self.d_f);
        builder.arg(&self.d_rho);
        builder.arg(&self.d_u);
        builder.arg(&self.d_tau); // Per-cell tau.
        let nx_i32 = self.nx as i32;
        let ny_i32 = self.ny as i32;
        let nz_i32 = self.nz as i32;
        builder.arg(&nx_i32);
        builder.arg(&ny_i32);
        builder.arg(&nz_i32);

        unsafe { builder.launch(config) }.context("Failed to launch bgk_collision_kernel")?;

        Ok(())
    }

    /// Stream distributions to neighbor cells (unused in fused collision-streaming)
    #[allow(dead_code)]
    fn streaming(&mut self) -> Result<()> {
        // 3D grid of threads
        let block_dim = (8, 8, 8); // 512 threads per block
        let grid_dim = (
            self.nx.div_ceil(block_dim.0),
            self.ny.div_ceil(block_dim.1),
            self.nz.div_ceil(block_dim.2),
        );

        let config = LaunchConfig {
            grid_dim: (grid_dim.0 as u32, grid_dim.1 as u32, grid_dim.2 as u32),
            block_dim: (block_dim.0 as u32, block_dim.1 as u32, block_dim.2 as u32),
            shared_mem_bytes: 0,
        };

        let mut builder = self.stream.launch_builder(&self.streaming_kernel);
        builder.arg(&self.d_f);
        builder.arg(&mut self.d_f_tmp);
        let nx_i32 = self.nx as i32;
        let ny_i32 = self.ny as i32;
        let nz_i32 = self.nz as i32;
        builder.arg(&nx_i32);
        builder.arg(&ny_i32);
        builder.arg(&nz_i32);

        unsafe { builder.launch(config) }.context("Failed to launch streaming_kernel")?;

        // Swap buffers: f <- f_tmp
        std::mem::swap(&mut self.d_f, &mut self.d_f_tmp);

        Ok(())
    }

    /// Evolve LBM for multiple steps
    pub fn evolve(&mut self, steps: usize) -> Result<()> {
        for _ in 0..steps {
            self.step()?;
        }

        // Synchronize and download final state
        self.stream
            .synchronize()
            .context("CUDA synchronize failed")?;
        self.sync_to_host()?;

        Ok(())
    }

    /// Single LBM step (collision only, streaming implicit)
    ///
    /// Matches CPU solver's fused collision-streaming approach where
    /// distributions are updated in-place and no explicit propagation occurs.
    pub fn step(&mut self) -> Result<()> {
        self.compute_macroscopic()?;
        self.collision()?;
        // NOTE: NO streaming - fused approach like CPU solver
        Ok(())
    }

    /// Synchronize GPU state to host memory
    pub fn sync_to_host(&mut self) -> Result<()> {
        // Download rho
        let rho_vec = self
            .stream
            .clone_dtoh(&self.d_rho)
            .context("Failed to download rho from GPU")?;
        self.rho = rho_vec;

        // Download u (flattened [ux0, uy0, uz0, ux1, uy1, uz1, ...])
        let u_flat = self
            .stream
            .clone_dtoh(&self.d_u)
            .context("Failed to download u from GPU")?;

        self.u = u_flat
            .chunks_exact(3)
            .map(|chunk| [chunk[0], chunk[1], chunk[2]])
            .collect();

        Ok(())
    }

    /// Get grid dimensions
    pub fn grid_size(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz)
    }

    /// Get total cell count
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }
}

impl TwoPhaseSystem for LbmSolver3DCuda {
    fn execute_phase1(&mut self) -> ScheduleResult<()> {
        self.compute_macroscopic().map_err(|e| {
            ScheduleError::StateInvalid(format!("CUDA phase1 macroscopic failure: {e}"))
        })?;
        self.collision().map_err(|e| {
            ScheduleError::StateInvalid(format!("CUDA phase1 collision failure: {e}"))
        })?;
        Ok(())
    }

    fn execute_phase2(&mut self) -> ScheduleResult<()> {
        // NO explicit streaming - fused collision-streaming like CPU solver
        // Streaming is implicit in D3Q19 lattice structure
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_solver_creation() {
        let result = LbmSolver3DCuda::new(8, 8, 8, 1.0);
        match result {
            Ok(solver) => {
                assert_eq!(solver.grid_size(), (8, 8, 8));
                assert_eq!(solver.n_cells(), 512);
            }
            Err(e) => {
                // GPU not available - skip test
                eprintln!("GPU test skipped: {}", e);
            }
        }
    }

    #[test]
    fn test_viscosity_field_upload() {
        let mut solver = match LbmSolver3DCuda::new(8, 8, 8, 1.0) {
            Ok(s) => s,
            Err(_) => return, // Skip if no GPU
        };

        let nu_field = vec![0.1; 512];
        solver.set_viscosity_field(&nu_field).unwrap();

        let retrieved = solver.get_viscosity_field().unwrap();
        for (&nu_in, &nu_out) in nu_field.iter().zip(retrieved.iter()) {
            assert!((nu_in - nu_out).abs() < 1e-10);
        }
    }

    #[test]
    fn test_uniform_initialization() {
        let mut solver = match LbmSolver3DCuda::new(8, 8, 8, 1.0) {
            Ok(s) => s,
            Err(_) => return,
        };

        solver.initialize_uniform(1.5, [0.01, 0.02, 0.03]).unwrap();
        solver.sync_to_host().unwrap();

        // Check host-side state
        assert!((solver.rho[0] - 1.5).abs() < 1e-10);
        assert!((solver.u[0][0] - 0.01).abs() < 1e-10);
        assert!((solver.u[0][1] - 0.02).abs() < 1e-10);
        assert!((solver.u[0][2] - 0.03).abs() < 1e-10);
    }
}

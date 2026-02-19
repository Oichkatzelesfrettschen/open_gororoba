//! GPU-accelerated Algebraic Lensing using CUDA.
//!
//! Bridges the GRIN RK4 solver to custom CUDA kernels for massive throughput.
//! Optimized for Ada Lovelace (SM89) hardware.

#[cfg(feature = "gpu")]
use cudarc::driver::{
    CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg, DeviceRepr, ValidAsZeroBits,
};
#[cfg(feature = "gpu")]
use std::sync::Arc;

#[cfg(feature = "gpu")]
const KERNEL_SRC: &str = include_str!("algebraic_lensing_grin.cu");

/// GPU-compatible 3D vector.
#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuVec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[cfg(feature = "gpu")]
unsafe impl DeviceRepr for GpuVec3 {}
#[cfg(feature = "gpu")]
unsafe impl ValidAsZeroBits for GpuVec3 {}

/// GPU-accelerated Algebraic Lensing simulation.
#[cfg(feature = "gpu")]
pub struct AlgebraicLensingGpu {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    trace_kernel: CudaFunction,
}

#[cfg(feature = "gpu")]
impl AlgebraicLensingGpu {
    /// Initialize CUDA context and compile kernels.
    pub fn new() -> anyhow::Result<Self> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();

        // Compile kernel at runtime
        use cudarc::nvrtc::CompileOptions;
        let opts = CompileOptions {
            arch: Some("sm_89"), // Ada Lovelace
            include_paths: vec!["/opt/cuda/include".to_string()],
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(KERNEL_SRC, opts)?;
        let module = ctx.load_module(ptx)?;
        let trace_kernel = module.load_function("trace_rays_kernel")?;

        Ok(Self {
            _ctx: ctx,
            stream,
            trace_kernel,
        })
    }

    /// Trace a batch of rays through the algebraic vacuum.
    ///
    /// # Arguments
    /// * `density_field` - Precomputed frustration density (nx * ny * nz)
    /// * `nx, ny, nz` - Grid dimensions
    /// * `initial_pos` - Starting positions
    /// * `initial_dir` - Starting unit directions
    /// * `alpha` - Coupling strength
    /// * `dt` - Step size
    /// * `max_steps` - Max iterations per ray
    pub fn trace_rays(
        &self,
        density_field: &[f32],
        nx: usize, ny: usize, nz: usize,
        initial_pos: &[GpuVec3],
        initial_dir: &[GpuVec3],
        alpha: f32,
        dt: f32,
        max_steps: usize,
    ) -> anyhow::Result<(Vec<GpuVec3>, Vec<GpuVec3>)> {
        let n_rays = initial_pos.len();
        
        // Upload data to GPU
        let d_density = self.stream.clone_htod(density_field)?;
        let d_initial_pos = self.stream.clone_htod(initial_pos)?;
        let d_initial_dir = self.stream.clone_htod(initial_dir)?;
        let mut d_final_pos = self.stream.alloc_zeros::<GpuVec3>(n_rays)?;
        let mut d_final_dir = self.stream.alloc_zeros::<GpuVec3>(n_rays)?;

        let eps_grad = 0.1f32;
        let threads_per_block = 256;
        let blocks = (n_rays as u32 + threads_per_block - 1) / threads_per_block;
        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };
        
        let nx_i = nx as i32;
        let ny_i = ny as i32;
        let nz_i = nz as i32;
        let n_rays_i = n_rays as i32;
        let max_steps_i = max_steps as i32;

        let mut builder = self.stream.launch_builder(&self.trace_kernel);
        builder.arg(&d_density)
               .arg(&nx_i).arg(&ny_i).arg(&nz_i)
               .arg(&d_initial_pos)
               .arg(&d_initial_dir)
               .arg(&mut d_final_pos)
               .arg(&mut d_final_dir)
               .arg(&n_rays_i)
               .arg(&alpha)
               .arg(&dt)
               .arg(&max_steps_i)
               .arg(&eps_grad);

        unsafe { builder.launch(config) }?;

        // Download results
        Ok((
            self.stream.clone_dtoh(&d_final_pos)?,
            self.stream.clone_dtoh(&d_final_dir)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_initialization() {
        if CudaContext::new(0).is_err() {
            eprintln!("GPU not available, skipping test");
            return;
        }
        let tracer = AlgebraicLensingGpu::new();
        assert!(tracer.is_ok());
    }

    #[test]
    fn test_gpu_trace_smoke() {
        if CudaContext::new(0).is_err() {
            eprintln!("GPU not available, skipping test");
            return;
        }
        let tracer = AlgebraicLensingGpu::new().unwrap();
        let density = vec![0.375f32; 8*8*8];
        let pos = vec![GpuVec3 { x: 4.0, y: 4.0, z: 0.0 }];
        let dir = vec![GpuVec3 { x: 0.0, y: 0.0, z: 1.0 }];
        
        let result = tracer.trace_rays(&density, 8, 8, 8, &pos, &dir, 1.0, 0.1, 10);
        assert!(result.is_ok());
        let (f_pos, _) = result.unwrap();
        assert_eq!(f_pos.len(), 1);
        // Position should have advanced in Z
        assert!(f_pos[0].z > 0.0);
    }
}

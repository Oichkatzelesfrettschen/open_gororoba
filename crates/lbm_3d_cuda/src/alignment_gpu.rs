use anyhow::Result;
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::compile_ptx,
};
use std::sync::Arc;

/// GPU-accelerated Box-Kite alignment engine.
pub struct GpuBoxKiteAlignmentEngine {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    kernel: CudaFunction,
}

impl GpuBoxKiteAlignmentEngine {
    pub fn try_new() -> Option<Self> {
        let ctx = CudaContext::new(0).ok()?;
        let stream = ctx.default_stream();

        // Load kernel source
        let ptx = compile_ptx(include_str!("kernels_alignment.cu")).ok()?;
        let module = ctx.load_module(ptx).ok()?;
        let kernel = module.load_function("box_kite_alignment_scan").ok()?;

        Some(Self {
            _ctx: ctx,
            stream,
            kernel,
        })
    }

    pub fn run_alignment_scan(
        &self,
        vectors: &[f64],     // [n * 16]
        orientations: &[u8], // [168 * 16]
        bk_indices: &[u8],   // [7 * 12]
    ) -> Result<(Vec<f64>, Vec<u32>)> {
        let n_vectors = vectors.len() / 16;
        let n_orientations = orientations.len() / 16;

        let v_dev = self.stream.clone_htod(vectors)?;
        let o_dev = self.stream.clone_htod(orientations)?;
        let bk_dev = self.stream.clone_htod(bk_indices)?;

        let mut out_max_dev = self.stream.alloc_zeros::<f64>(n_vectors)?;
        let mut out_best_dev = self.stream.alloc_zeros::<u32>(n_vectors)?;

        let block_size = 256_u32;
        let grid_size = (n_vectors as u32).div_ceil(block_size);
        let cfg = LaunchConfig {
            block_dim: (block_size, 1, 1),
            grid_dim: (grid_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = self.stream.launch_builder(&self.kernel);
        let n_vectors_u32 = n_vectors as u32;
        let n_orientations_u32 = n_orientations as u32;

        builder.arg(&v_dev);
        builder.arg(&o_dev);
        builder.arg(&bk_dev);
        builder.arg(&mut out_max_dev);
        builder.arg(&mut out_best_dev);
        builder.arg(&n_vectors_u32);
        builder.arg(&n_orientations_u32);

        unsafe { builder.launch(cfg) }?;

        let out_max = self.stream.clone_dtoh(&out_max_dev)?;
        let out_best = self.stream.clone_dtoh(&out_best_dev)?;

        Ok((out_max, out_best))
    }
}

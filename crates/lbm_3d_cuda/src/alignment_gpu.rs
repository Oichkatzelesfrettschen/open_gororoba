use anyhow::Result;
use cudarc::driver::{CudaStream, PushKernelArg};
use gororoba_gpu_cuda::{
    Buffer, CompileOptions, Context as CudaContextHelper, KernelHandle, LaunchConfig,
    ModuleRegistry,
};
use std::sync::Arc;

/// GPU-accelerated Box-Kite alignment engine.
pub struct GpuBoxKiteAlignmentEngine {
    _ctx: CudaContextHelper,
    _module_registry: ModuleRegistry,
    stream: Arc<CudaStream>,
    kernel: KernelHandle,
}

impl GpuBoxKiteAlignmentEngine {
    pub fn try_new() -> Option<Self> {
        let ctx = CudaContextHelper::with_default_device().ok()?;
        let stream = ctx.default_stream();

        let opts = CompileOptions::empty();
        let module_registry = ModuleRegistry::compile_and_load(
            ctx.raw(),
            include_str!("kernels_alignment.cu"),
            &opts,
            &["box_kite_alignment_scan"],
        )
        .ok()?;
        let kernel = module_registry.get("box_kite_alignment_scan").ok()?;

        Some(Self {
            _ctx: ctx,
            _module_registry: module_registry,
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

        let v_dev = Buffer::<f64>::htod(&self.stream, vectors)?;
        let o_dev = Buffer::<u8>::htod(&self.stream, orientations)?;
        let bk_dev = Buffer::<u8>::htod(&self.stream, bk_indices)?;

        let mut out_max_dev = Buffer::<f64>::alloc_zeros(&self.stream, n_vectors)?;
        let mut out_best_dev = Buffer::<u32>::alloc_zeros(&self.stream, n_vectors)?;

        let cfg = LaunchConfig::launch_1d(n_vectors as u32);

        let mut builder = self.stream.launch_builder(&self.kernel);
        let n_vectors_u32 = n_vectors as u32;
        let n_orientations_u32 = n_orientations as u32;

        builder.arg(v_dev.raw());
        builder.arg(o_dev.raw());
        builder.arg(bk_dev.raw());
        builder.arg(out_max_dev.raw_mut());
        builder.arg(out_best_dev.raw_mut());
        builder.arg(&n_vectors_u32);
        builder.arg(&n_orientations_u32);

        unsafe { builder.launch(cfg) }?;

        let out_max = out_max_dev.dtoh_vec()?;
        let out_best = out_best_dev.dtoh_vec()?;

        Ok((out_max, out_best))
    }
}

//! CUDA kernel launch wrappers for TurboQuant.
//!
//! Context acquisition, PTX module load, launch-shape helpers, and
//! stream-attached buffer ownership route through `gororoba_gpu_cuda`.
//! The per-call launch path still uses cudarc's raw launch builder because
//! `gororoba_gpu_cuda` intentionally does not wrap every kernel argument
//! specialization.

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
#[allow(deprecated)]
use cudarc::driver::{CudaStream, PushKernelArg};
#[cfg(feature = "cuda")]
use gororoba_gpu_cuda::{Buffer, KernelHandle, ModuleRegistry};

/// Compiled TurboQuant CUDA kernel handles.
#[cfg(feature = "cuda")]
#[allow(deprecated, dead_code)]
pub struct TurboQuantCudaKernels {
    _ctx: gororoba_gpu_cuda::Context,
    _module_registry: ModuleRegistry,
    stream: Arc<CudaStream>,
    quantize_fn: KernelHandle,
    dequant_dot_fn: KernelHandle,
    fast_jl_fn: KernelHandle,
    sign_dot_fn: KernelHandle,
    dequant_dot_q16_fn: KernelHandle,
}

#[cfg(feature = "cuda")]
#[allow(deprecated)]
impl TurboQuantCudaKernels {
    /// Initialize: probe device, NVRTC compile, load all 5 kernel functions.
    ///
    /// Routes context acquisition through
    /// `gororoba_gpu_cuda::Context::with_default_device`. The JIT layer
    /// returns PTX so tests and launch code can share one compiled artifact;
    /// this constructor loads that PTX through `ModuleRegistry`.
    pub fn new() -> Result<Self, String> {
        let props =
            super::device::probe_device().ok_or_else(|| "No CUDA device available".to_string())?;

        let ctx = gororoba_gpu_cuda::Context::with_default_device()
            .map_err(|e| format!("CUDA context: {}", e))?;
        let stream = ctx.default_stream();

        let ptx = super::jit::compile_kernels(props.major, props.minor)?;
        use super::jit::kernel_names;
        let registry = gororoba_gpu_cuda::ModuleRegistry::load(
            ctx.raw(),
            ptx,
            &[
                kernel_names::QUANTIZE_BOUNDARY,
                kernel_names::DEQUANT_DOT,
                kernel_names::FAST_JL_ROTATE,
                kernel_names::SIGN_DOT,
                kernel_names::DEQUANT_DOT_Q16,
            ],
        )
        .map_err(|e| format!("Module load: {}", e))?;

        let quantize_fn = registry
            .get(kernel_names::QUANTIZE_BOUNDARY)
            .map_err(|e| format!("Load {}: {}", kernel_names::QUANTIZE_BOUNDARY, e))?;
        let dequant_dot_fn = registry
            .get(kernel_names::DEQUANT_DOT)
            .map_err(|e| format!("Load {}: {}", kernel_names::DEQUANT_DOT, e))?;
        let fast_jl_fn = registry
            .get(kernel_names::FAST_JL_ROTATE)
            .map_err(|e| format!("Load {}: {}", kernel_names::FAST_JL_ROTATE, e))?;
        let sign_dot_fn = registry
            .get(kernel_names::SIGN_DOT)
            .map_err(|e| format!("Load {}: {}", kernel_names::SIGN_DOT, e))?;
        let dequant_dot_q16_fn = registry
            .get(kernel_names::DEQUANT_DOT_Q16)
            .map_err(|e| format!("Load {}: {}", kernel_names::DEQUANT_DOT_Q16, e))?;

        Ok(TurboQuantCudaKernels {
            _ctx: ctx,
            _module_registry: registry,
            stream,
            quantize_fn,
            dequant_dot_fn,
            fast_jl_fn,
            sign_dot_fn,
            dequant_dot_q16_fn,
        })
    }

    /// Quantize a batch of f32 values on GPU.
    ///
    /// Returns u8 indices.
    pub fn quantize_batch(&self, values: &[f32], boundaries: &[f32]) -> Result<Vec<u8>, String> {
        let n = values.len();
        let n_boundaries = boundaries.len() as i32;
        let n_i32 = n as i32;

        let d_values = Buffer::<f32>::htod(&self.stream, values)
            .map_err(|e| format!("memcpy values: {}", e))?;
        let d_boundaries = Buffer::<f32>::htod(&self.stream, boundaries)
            .map_err(|e| format!("memcpy boundaries: {}", e))?;
        let mut d_indices = Buffer::<u8>::alloc_zeros(&self.stream, n)
            .map_err(|e| format!("alloc indices: {}", e))?;

        let config = gororoba_gpu_cuda::LaunchConfig::launch_1d(n as u32);

        let mut b = self.stream.launch_builder(&self.quantize_fn);
        b.arg(d_boundaries.raw())
            .arg(d_values.raw())
            .arg(d_indices.raw_mut())
            .arg(&n_boundaries)
            .arg(&n_i32);
        unsafe { b.launch(config) }.map_err(|e| format!("quantize launch: {}", e))?;

        d_indices.dtoh_vec().map_err(|e| format!("readback: {}", e))
    }

    /// Dequantize + dot product on GPU.
    ///
    /// Returns attention scores for each key.
    pub fn dequant_dot_batch(
        &self,
        queries: &[f32],
        key_indices: &[u8],
        centroids: &[f32],
        key_norms: &[f32],
        d: usize,
    ) -> Result<Vec<f32>, String> {
        if d == 0 {
            return Err("d must be > 0".to_string());
        }
        if !queries.len().is_multiple_of(d) {
            return Err(format!(
                "queries length mismatch: got {}, expected multiple of d={}",
                queries.len(),
                d
            ));
        }
        let n_queries = queries.len() / d;
        let n_keys = key_norms.len();
        let validated = super::super::dequant_contract::validate_dequant_dot_contract(
            queries.len(),
            key_indices.len(),
            centroids.len(),
            key_norms.len(),
            n_queries,
            n_keys,
            d,
        )?;
        if validated.expected_scores == 0 {
            return Ok(vec![]);
        }
        let dims = validated.kernel_dims_i32()?;

        let d_query = Buffer::<f32>::htod(&self.stream, queries)
            .map_err(|e| format!("memcpy query: {}", e))?;
        let d_key_indices = Buffer::<u8>::htod(&self.stream, key_indices)
            .map_err(|e| format!("memcpy keys: {}", e))?;
        let d_centroids = Buffer::<f32>::htod(&self.stream, centroids)
            .map_err(|e| format!("memcpy centroids: {}", e))?;
        let d_key_norms = Buffer::<f32>::htod(&self.stream, key_norms)
            .map_err(|e| format!("memcpy key_norms: {}", e))?;
        let mut d_scores = Buffer::<f32>::alloc_zeros(&self.stream, validated.expected_scores)
            .map_err(|e| format!("alloc scores: {}", e))?;

        let block = 256u32;
        let grid_x = (dims.n_keys as u32).div_ceil(block);
        let config = gororoba_gpu_cuda::LaunchConfig::launch_blocks_2d(
            grid_x,
            dims.n_queries as u32,
            block,
            1,
        );

        let mut b = self.stream.launch_builder(&self.dequant_dot_fn);
        b.arg(d_query.raw())
            .arg(d_key_indices.raw())
            .arg(d_centroids.raw())
            .arg(d_key_norms.raw())
            .arg(d_scores.raw_mut())
            .arg(&dims.d)
            .arg(&dims.n_queries)
            .arg(&dims.n_keys)
            .arg(&dims.n_levels);
        unsafe { b.launch(config) }.map_err(|e| format!("dequant_dot launch: {}", e))?;

        d_scores.dtoh_vec().map_err(|e| format!("readback: {}", e))
    }
}

/// Stub for non-CUDA builds.
#[cfg(not(feature = "cuda"))]
pub struct TurboQuantCudaKernels {
    _private: (),
}

#[cfg(not(feature = "cuda"))]
impl TurboQuantCudaKernels {
    pub fn new() -> Result<Self, String> {
        Err("CUDA feature not enabled".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_kernels_available() {
        match TurboQuantCudaKernels::new() {
            Ok(_) => println!("CUDA kernels initialized"),
            Err(e) => println!("CUDA not available: {}", e),
        }
    }
}

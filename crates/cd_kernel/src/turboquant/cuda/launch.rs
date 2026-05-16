//! CUDA kernel launch wrappers for TurboQuant.
//!
//! Pattern proven in lbm_3d_cuda/src/lib.rs (steinmarder SoA design):
//!   CudaContext::new -> compile_ptx -> load_module -> load_function -> launch
//!
//! Wave C2.3 migrated the context acquisition, PTX module load, and
//! launch-config helpers to `gororoba_gpu_cuda`. The
//! `TurboQuantCudaKernels` struct still owns the resolved `CudaFunction`
//! handles directly so the per-call launch path stays a thin builder
//! against cudarc::driver (which gpu_cuda does not yet wrap).

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
#[allow(deprecated)]
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

/// Compiled TurboQuant CUDA kernel handles.
#[cfg(feature = "cuda")]
#[allow(deprecated, dead_code)]
pub struct TurboQuantCudaKernels {
    _ctx: gororoba_gpu_cuda::Context,
    stream: Arc<CudaStream>,
    quantize_fn: CudaFunction,
    dequant_dot_fn: CudaFunction,
    fast_jl_fn: CudaFunction,
    sign_dot_fn: CudaFunction,
    dequant_dot_q16_fn: CudaFunction,
}

#[cfg(feature = "cuda")]
#[allow(deprecated)]
impl TurboQuantCudaKernels {
    /// Initialize: probe device, NVRTC compile, load all 5 kernel functions.
    ///
    /// Routes through `gororoba_gpu_cuda::Context::with_default_device` and
    /// `gororoba_gpu_cuda::ModuleRegistry::load` so the cudarc-init
    /// boilerplate stays in one place. The function preserves the
    /// `Result<Self, String>` shape so the `BackendQuantizer::try_quantize`
    /// call site keeps compiling without an import change.
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
            stream,
            quantize_fn,
            dequant_dot_fn,
            fast_jl_fn,
            sign_dot_fn,
            dequant_dot_q16_fn,
        })
    }

    /// 1D launch config: thin wrapper around
    /// `gororoba_gpu_cuda::LaunchConfig::launch_1d` so the call site
    /// reads the same as before.
    fn launch_config_1d(n: u32) -> LaunchConfig {
        gororoba_gpu_cuda::LaunchConfig::launch_1d(n)
    }

    /// 2D launch config: cd_kernel turboquant uses a 1D block of 256
    /// threads paired with `n_y` as the y-axis grid dimension (one row
    /// of queries per slice). `gpu_cuda::LaunchConfig::launch_2d`'s
    /// 16x16 block is a different convention, so this fn keeps the
    /// hand-rolled shape rather than delegate.
    fn launch_config_2d(n_x: u32, n_y: u32) -> LaunchConfig {
        let block = 256u32;
        let grid_x = n_x.div_ceil(block);
        LaunchConfig {
            grid_dim: (grid_x, n_y, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Quantize a batch of f32 values on GPU.
    ///
    /// Returns u8 indices.
    pub fn quantize_batch(&self, values: &[f32], boundaries: &[f32]) -> Result<Vec<u8>, String> {
        let n = values.len();
        let n_boundaries = boundaries.len() as i32;
        let n_i32 = n as i32;

        let d_values: CudaSlice<f32> = self
            .stream
            .memcpy_stod(values)
            .map_err(|e| format!("memcpy values: {}", e))?;
        let d_boundaries: CudaSlice<f32> = self
            .stream
            .memcpy_stod(boundaries)
            .map_err(|e| format!("memcpy boundaries: {}", e))?;
        let mut d_indices: CudaSlice<u8> = self
            .stream
            .alloc_zeros(n)
            .map_err(|e| format!("alloc indices: {}", e))?;

        let config = Self::launch_config_1d(n as u32);

        let mut b = self.stream.launch_builder(&self.quantize_fn);
        b.arg(&d_boundaries)
            .arg(&d_values)
            .arg(&mut d_indices)
            .arg(&n_boundaries)
            .arg(&n_i32);
        unsafe { b.launch(config) }.map_err(|e| format!("quantize launch: {}", e))?;

        let mut host_indices = vec![0u8; n];
        self.stream
            .memcpy_dtoh(&d_indices, &mut host_indices)
            .map_err(|e| format!("readback: {}", e))?;
        Ok(host_indices)
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

        let d_query: CudaSlice<f32> = self
            .stream
            .memcpy_stod(queries)
            .map_err(|e| format!("memcpy query: {}", e))?;
        let d_key_indices: CudaSlice<u8> = self
            .stream
            .memcpy_stod(key_indices)
            .map_err(|e| format!("memcpy keys: {}", e))?;
        let d_centroids: CudaSlice<f32> = self
            .stream
            .memcpy_stod(centroids)
            .map_err(|e| format!("memcpy centroids: {}", e))?;
        let d_key_norms: CudaSlice<f32> = self
            .stream
            .memcpy_stod(key_norms)
            .map_err(|e| format!("memcpy key_norms: {}", e))?;
        let mut d_scores: CudaSlice<f32> = self
            .stream
            .alloc_zeros(validated.expected_scores)
            .map_err(|e| format!("alloc scores: {}", e))?;

        let config = Self::launch_config_2d(dims.n_keys as u32, dims.n_queries as u32);

        let mut b = self.stream.launch_builder(&self.dequant_dot_fn);
        b.arg(&d_query)
            .arg(&d_key_indices)
            .arg(&d_centroids)
            .arg(&d_key_norms)
            .arg(&mut d_scores)
            .arg(&dims.d)
            .arg(&dims.n_queries)
            .arg(&dims.n_keys)
            .arg(&dims.n_levels);
        unsafe { b.launch(config) }.map_err(|e| format!("dequant_dot launch: {}", e))?;

        let mut host_scores = vec![0.0f32; validated.expected_scores];
        self.stream
            .memcpy_dtoh(&d_scores, &mut host_scores)
            .map_err(|e| format!("readback: {}", e))?;
        Ok(host_scores)
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

//! CUDA kernel launch wrappers for TurboQuant.
//!
//! Pattern proven in lbm_3d_cuda/src/lib.rs (steinmarder SoA design):
//!   CudaContext::new -> compile_ptx -> load_module -> load_function -> launch

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
#[allow(deprecated)]
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};

/// Compiled TurboQuant CUDA kernel handles.
#[cfg(feature = "cuda")]
#[allow(deprecated, dead_code)]
pub struct TurboQuantCudaKernels {
    _ctx: Arc<CudaContext>,
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
    pub fn new() -> Result<Self, String> {
        let props = super::device::probe_device()
            .ok_or_else(|| "No CUDA device available".to_string())?;
        let arch = props.compile_arch();

        let ctx = CudaContext::new(0)
            .map_err(|e| format!("CUDA context: {}", e))?;
        let stream = ctx.default_stream();

        let ptx = super::jit::compile_kernels(arch)?;
        let module = ctx.load_module(ptx)
            .map_err(|e| format!("Module load: {}", e))?;

        use super::jit::kernel_names;
        let quantize_fn = module.load_function(kernel_names::QUANTIZE_BOUNDARY)
            .map_err(|e| format!("Load {}: {}", kernel_names::QUANTIZE_BOUNDARY, e))?;
        let dequant_dot_fn = module.load_function(kernel_names::DEQUANT_DOT)
            .map_err(|e| format!("Load {}: {}", kernel_names::DEQUANT_DOT, e))?;
        let fast_jl_fn = module.load_function(kernel_names::FAST_JL_ROTATE)
            .map_err(|e| format!("Load {}: {}", kernel_names::FAST_JL_ROTATE, e))?;
        let sign_dot_fn = module.load_function(kernel_names::SIGN_DOT)
            .map_err(|e| format!("Load {}: {}", kernel_names::SIGN_DOT, e))?;
        let dequant_dot_q16_fn = module.load_function(kernel_names::DEQUANT_DOT_Q16)
            .map_err(|e| format!("Load {}: {}", kernel_names::DEQUANT_DOT_Q16, e))?;

        Ok(TurboQuantCudaKernels {
            _ctx: ctx, stream, quantize_fn, dequant_dot_fn,
            fast_jl_fn, sign_dot_fn, dequant_dot_q16_fn,
        })
    }

    fn launch_config_1d(n: u32) -> LaunchConfig {
        let block = 256u32;
        let grid = (n + block - 1) / block;
        LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 }
    }

    /// Quantize a batch of f32 values on GPU.
    ///
    /// Returns u8 indices.
    pub fn quantize_batch(
        &self,
        values: &[f32],
        boundaries: &[f32],
    ) -> Result<Vec<u8>, String> {
        let n = values.len();
        let n_boundaries = boundaries.len() as i32;
        let n_i32 = n as i32;

        let mut d_values: CudaSlice<f32> = self.stream.memcpy_stod(values)
            .map_err(|e| format!("memcpy values: {}", e))?;
        let d_boundaries: CudaSlice<f32> = self.stream.memcpy_stod(boundaries)
            .map_err(|e| format!("memcpy boundaries: {}", e))?;
        let mut d_indices: CudaSlice<u8> = self.stream.alloc_zeros(n)
            .map_err(|e| format!("alloc indices: {}", e))?;

        let config = Self::launch_config_1d(n as u32);

        let mut b = self.stream.launch_builder(&self.quantize_fn);
        b.arg(&d_boundaries)
            .arg(&mut d_values)
            .arg(&mut d_indices)
            .arg(&n_boundaries)
            .arg(&n_i32);
        unsafe { b.launch(config) }
            .map_err(|e| format!("quantize launch: {}", e))?;

        let mut host_indices = vec![0u8; n];
        self.stream.memcpy_dtoh(&d_indices, &mut host_indices)
            .map_err(|e| format!("readback: {}", e))?;
        Ok(host_indices)
    }

    /// Dequantize + dot product on GPU.
    ///
    /// Returns attention scores for each key.
    pub fn dequant_dot_batch(
        &self,
        query: &[f32],
        key_indices: &[u8],
        centroids: &[f32],
        n_keys: usize,
        d: usize,
    ) -> Result<Vec<f32>, String> {
        let d_i32 = d as i32;
        let n_keys_i32 = n_keys as i32;

        let d_query: CudaSlice<f32> = self.stream.memcpy_stod(query)
            .map_err(|e| format!("memcpy query: {}", e))?;
        let d_key_indices: CudaSlice<u8> = self.stream.memcpy_stod(key_indices)
            .map_err(|e| format!("memcpy keys: {}", e))?;
        let d_centroids: CudaSlice<f32> = self.stream.memcpy_stod(centroids)
            .map_err(|e| format!("memcpy centroids: {}", e))?;
        let mut d_scores: CudaSlice<f32> = self.stream.alloc_zeros(n_keys)
            .map_err(|e| format!("alloc scores: {}", e))?;

        let config = Self::launch_config_1d(n_keys as u32);

        let mut b = self.stream.launch_builder(&self.dequant_dot_fn);
        b.arg(&d_query)
            .arg(&d_key_indices)
            .arg(&d_centroids)
            .arg(&mut d_scores)
            .arg(&d_i32)
            .arg(&n_keys_i32);
        unsafe { b.launch(config) }
            .map_err(|e| format!("dequant_dot launch: {}", e))?;

        let mut host_scores = vec![0.0f32; n_keys];
        self.stream.memcpy_dtoh(&d_scores, &mut host_scores)
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

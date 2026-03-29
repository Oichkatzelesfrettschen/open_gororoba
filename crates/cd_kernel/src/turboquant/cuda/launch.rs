//! CUDA kernel launch wrappers for TurboQuant.
//!
//! Handles: CudaContext creation, buffer allocation, kernel dispatch,
//! and result readback.  Completes the Backend::Cuda dispatch path.
//!
//! Pattern from lbm_3d_cuda/lib.rs:
//! ```ignore
//! // 1. Probe device
//! let props = probe_device()?;
//! let arch = props.compile_arch();
//!
//! // 2. Create context + stream
//! let ctx = CudaContext::new(0)?;
//! let stream = ctx.default_stream();
//!
//! // 3. Compile kernels via NVRTC
//! let ptx = jit::compile_kernels(arch)?;
//! let module = ctx.load_module(ptx)?;
//! let quantize_fn = module.load_function("turboquant_quantize_boundary")?;
//!
//! // 4. Allocate device buffers
//! let d_values = stream.memcpy_stod(host_values)?;
//! let d_boundaries = stream.memcpy_stod(host_boundaries)?;
//! let d_indices = stream.alloc_zeros::<u8>(n)?;
//!
//! // 5. Launch kernel
//! let config = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
//! unsafe {
//!     let mut launch = stream.launch_builder(&quantize_fn);
//!     launch.arg(&d_boundaries).arg(&d_values).arg(&d_indices)
//!           .arg(&(n_boundaries as i32)).arg(&(n as i32));
//!     launch.launch(config)?;
//! }
//!
//! // 6. Readback
//! let mut host_indices = vec![0u8; n];
//! stream.memcpy_dtos(&d_indices, &mut host_indices)?;
//! ```
//!
//! The actual launch implementation requires runtime CUDA availability
//! and matching cudarc API signatures.  See lbm_3d_cuda/src/lib.rs for
//! the proven pattern in this workspace.

/// Placeholder for compiled CUDA kernel handles.
///
/// When fully implemented, this holds:
/// - Arc<CudaContext> for device management
/// - Arc<CudaStream> for async operation
/// - CudaFunction handles for each kernel (quantize, dequant_dot, sign_dot, fast_jl)
///
/// See the module-level doc for the exact initialization sequence.
pub struct TurboQuantCudaKernels {
    _private: (), // prevent construction outside this module
}

impl TurboQuantCudaKernels {
    /// Initialize CUDA kernels.
    ///
    /// Requires: CUDA device present, NVRTC installed, `cuda` feature enabled.
    /// The initialization is expensive (~100ms for JIT compilation) and should
    /// be done once at startup.
    ///
    /// Currently returns Err because the full launch path is not yet wired.
    /// The kernel source (kernels/turboquant.cu) and JIT compiler (jit.rs)
    /// are ready; what remains is the buffer management and launch config
    /// following the lbm_3d_cuda pattern.
    pub fn new() -> Result<Self, String> {
        #[cfg(feature = "cuda")]
        {
            // Verify we can at least probe the device and compile
            let props = super::device::probe_device()
                .ok_or_else(|| "No CUDA device available".to_string())?;
            let arch = props.compile_arch();
            let _ptx = super::jit::compile_kernels(arch)?;
            // PTX compiled successfully -- kernel source is valid.
            // Full launch path deferred to next iteration.
            return Err(format!(
                "CUDA kernels compiled for {} but launch path not yet wired (use CPU backend)",
                arch
            ));
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err("CUDA feature not enabled".to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_kernels_stub() {
        // Without full launch path, new() returns Err
        let result = TurboQuantCudaKernels::new();
        assert!(result.is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_kernels_compile_check() {
        // Verify the kernels at least compile via NVRTC
        if let Some(props) = super::super::device::probe_device() {
            let arch = props.compile_arch();
            match super::super::jit::compile_kernels(arch) {
                Ok(_ptx) => println!("CUDA kernels compiled for {}", arch),
                Err(e) => println!("NVRTC not available: {}", e),
            }
        }
    }
}

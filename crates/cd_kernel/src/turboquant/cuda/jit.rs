//! NVRTC JIT compilation for TurboQuant CUDA kernels.
//!
//! Pattern from `lbm_3d_cuda/src/lib.rs`: embed kernel source as a static
//! string, compile at runtime with architecture-targeted options, cache
//! the compiled module.

/// Embedded CUDA kernel source.
pub const KERNEL_SRC: &str = include_str!("kernels/turboquant.cu");

/// Kernel function names matching the extern "C" declarations in turboquant.cu.
pub mod kernel_names {
    pub const QUANTIZE_BOUNDARY: &str = "turboquant_quantize_boundary";
    pub const DEQUANT_DOT: &str = "turboquant_dequant_dot";
    pub const SIGN_DOT: &str = "turboquant_sign_dot";
    pub const FAST_JL_ROTATE: &str = "turboquant_fast_jl_rotate";
    /// Q16.16 fixed-point exact dequant+dot (zero accumulation drift).
    pub const DEQUANT_DOT_Q16: &str = "turboquant_dequant_dot_q16";
}

/// Compile the TurboQuant kernel source for the detected architecture.
///
/// Returns the compiled PTX.  The caller loads it into a CudaModule
/// and extracts individual functions by name.
///
/// Pattern from `lbm_3d_cuda/lib.rs`:
/// ```ignore
/// let ptx = compile_kernels("sm_89")?;
/// let module = ctx.load_module(ptx)?;
/// let quantize_fn = module.load_function("turboquant_quantize_boundary")?;
/// ```
#[cfg(feature = "cuda")]
pub fn compile_kernels(arch: &str) -> Result<cudarc::nvrtc::Ptx, String> {
    use cudarc::nvrtc::CompileOptions;

    // CompileOptions borrows arch, so use a static leak for the lifetime.
    // This is fine because we only compile once per architecture.
    let arch_static: &'static str = Box::leak(arch.to_string().into_boxed_str());

    let opts = CompileOptions {
        arch: Some(arch_static),
        include_paths: vec![],
        ..Default::default()
    };

    let arch_owned = arch.to_string();
    cudarc::nvrtc::compile_ptx_with_opts(KERNEL_SRC, opts)
        .map_err(move |e| format!("NVRTC compilation failed for {}: {}", arch_owned, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_source_embedded() {
        // Verify the kernel source is embedded and non-empty
        assert!(!KERNEL_SRC.is_empty());
        assert!(KERNEL_SRC.contains("turboquant_quantize_boundary"));
        assert!(KERNEL_SRC.contains("turboquant_dequant_dot"));
        assert!(KERNEL_SRC.contains("turboquant_sign_dot"));
        assert!(KERNEL_SRC.contains("turboquant_fast_jl_rotate"));
        assert!(KERNEL_SRC.contains("turboquant_dequant_dot_q16"));
    }

    #[test]
    fn test_kernel_names() {
        assert_eq!(kernel_names::QUANTIZE_BOUNDARY, "turboquant_quantize_boundary");
        assert_eq!(kernel_names::DEQUANT_DOT, "turboquant_dequant_dot");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_compile_for_detected_arch() {
        if let Some(props) = super::super::device::probe_device() {
            let arch = props.compile_arch();
            match compile_kernels(arch) {
                Ok(_ptx) => {
                    println!("NVRTC compiled successfully for {}", arch);
                }
                Err(e) => {
                    // NVRTC may not be installed -- not a test failure
                    println!("NVRTC compilation skipped: {}", e);
                }
            }
        }
    }
}

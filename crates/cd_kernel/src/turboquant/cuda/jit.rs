//! NVRTC JIT compilation for TurboQuant CUDA kernels.
//!
//! TurboQuant keeps compilation in the JIT layer and module loading in
//! the launch layer so tests can validate the generated PTX boundary.

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

/// Compile the TurboQuant kernel source for the given compute capability.
///
/// Returns the compiled PTX. The launch layer owns module loading
/// because TurboQuant tests and dispatch code share the same JIT output.
///
/// Delegates to `gororoba_gpu_cuda::CompileOptions::for_arch` which
/// owns the canonical `(major, minor) -> "sm_XX"` mapping and avoids
/// the `Box::leak(arch.to_string())` static-lifetime trick that prior
/// versions of this fn used.
///
/// TurboQuant intentionally keeps the PTX handoff visible:
/// ```ignore
/// let probe = gororoba_gpu_cuda::DeviceProbe::query()?;
/// let ptx = compile_kernels(probe.major, probe.minor)?;
/// let registry = gororoba_gpu_cuda::ModuleRegistry::load(
///     ctx.raw(), ptx, &[kernel_names::QUANTIZE_BOUNDARY])?;
/// ```
#[cfg(feature = "cuda")]
pub fn compile_kernels(major: u32, minor: u32) -> Result<gororoba_gpu_cuda::Ptx, String> {
    let opts = gororoba_gpu_cuda::CompileOptions::for_arch(major, minor);
    gororoba_gpu_cuda::CompileOptions::compile_ptx(KERNEL_SRC, &opts)
        .map_err(|e| format!("NVRTC compilation failed for sm_{}{}: {}", major, minor, e))
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
        assert_eq!(
            kernel_names::QUANTIZE_BOUNDARY,
            "turboquant_quantize_boundary"
        );
        assert_eq!(kernel_names::DEQUANT_DOT, "turboquant_dequant_dot");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_compile_for_detected_arch() {
        if let Some(props) = super::super::device::probe_device() {
            let arch = props.compile_arch();
            match compile_kernels(props.major, props.minor) {
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

//! Unified backend dispatch for TurboQuant quantization.
//!
//! Auto-detects the best available compute backend (CPU SIMD, CUDA, Vulkan)
//! and provides a single API surface for quantize/dequantize/inner_product.
//!
//! Detection priority: CUDA > Vulkan > CPU (highest throughput first).
//! All backends produce bit-identical quantization results -- the dispatch
//! only affects throughput, not quality.

use super::dispatch::{DispatchedQuantizer, SimdLevel, detect_simd_level};

/// Compute backend selector.
#[derive(Clone, Debug)]
pub enum Backend {
    /// CPU with SIMD acceleration.
    Cpu(SimdLevel),
    /// NVIDIA CUDA with architecture-specific kernels.
    #[cfg(feature = "cuda")]
    Cuda(super::cuda::device::KernelTier),
    /// Vulkan compute shaders.
    #[cfg(feature = "vulkan")]
    Vulkan(super::vulkan::context::VulkanShaderTier),
    /// cubecl unified GPU backend (CUDA + Vulkan + WebGPU + Metal).
    #[cfg(feature = "cubecl")]
    CubeCL,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Backend::Cpu(level) => write!(f, "cpu-{}", level),
            #[cfg(feature = "cuda")]
            Backend::Cuda(tier) => write!(f, "cuda-{}", tier),
            #[cfg(feature = "vulkan")]
            Backend::Vulkan(tier) => write!(f, "vulkan-{}", tier),
            #[cfg(feature = "cubecl")]
            Backend::CubeCL => write!(f, "cubecl"),
        }
    }
}

/// Detect the best available backend.
///
/// Probes in order: CUDA -> Vulkan -> CPU.
/// Returns the highest-throughput backend that is available and functional.
pub fn detect_best_backend() -> Backend {
    // Try cubecl first (unified, multi-platform)
    #[cfg(feature = "cubecl")]
    {
        if super::cubecl_backend::probe_cubecl().is_some() {
            return Backend::CubeCL;
        }
    }

    // Try CUDA next (highest throughput for NVIDIA)
    #[cfg(feature = "cuda")]
    {
        if let Some(props) = super::cuda::device::probe_device() {
            return Backend::Cuda(props.recommended_tier());
        }
    }

    // Try Vulkan next
    #[cfg(feature = "vulkan")]
    {
        if let Some(caps) = super::vulkan::context::probe_vulkan() {
            if caps.is_compute_capable() {
                return Backend::Vulkan(caps.recommended_shader_tier());
            }
        }
    }

    // Fall back to CPU SIMD
    Backend::Cpu(detect_simd_level())
}

/// Detect all available backends (for benchmark comparison).
pub fn detect_all_backends() -> Vec<Backend> {
    // CPU is always available; mut needed when cuda/vulkan features add GPU backends
    #[allow(unused_mut)]
    let mut backends = vec![Backend::Cpu(detect_simd_level())];

    #[cfg(feature = "cubecl")]
    {
        if super::cubecl_backend::probe_cubecl().is_some() {
            backends.push(Backend::CubeCL);
        }
    }

    #[cfg(feature = "cuda")]
    {
        if let Some(props) = super::cuda::device::probe_device() {
            backends.push(Backend::Cuda(props.recommended_tier()));
        }
    }

    #[cfg(feature = "vulkan")]
    {
        if let Some(caps) = super::vulkan::context::probe_vulkan() {
            if caps.is_compute_capable() {
                backends.push(Backend::Vulkan(caps.recommended_shader_tier()));
            }
        }
    }

    backends
}

/// Backend-dispatched TurboQuant quantizer.
///
/// Wraps the CPU DispatchedQuantizer and will dispatch to GPU kernels
/// when CUDA or Vulkan backends are selected.  Currently, GPU backends
/// fall through to CPU -- the GPU kernel implementations are stubs that
/// will be filled in as the CUDA .cu and Vulkan .comp files are written.
pub struct BackendQuantizer {
    cpu: DispatchedQuantizer,
    backend: Backend,
}

impl BackendQuantizer {
    /// Create a backend-dispatched quantizer.
    ///
    /// Uses the best available backend by default.
    pub fn new(codebook: &crate::lloyd_max::LloydMaxCodebook, bits: u32) -> Self {
        let backend = detect_best_backend();
        let cpu = DispatchedQuantizer::new(codebook, bits);
        BackendQuantizer { cpu, backend }
    }

    /// Create with an explicit backend.
    pub fn with_backend(
        codebook: &crate::lloyd_max::LloydMaxCodebook,
        bits: u32,
        backend: Backend,
    ) -> Self {
        let cpu = DispatchedQuantizer::new(codebook, bits);
        BackendQuantizer { cpu, backend }
    }

    /// Quantize a batch of values.
    ///
    /// Dispatches to the appropriate backend.  GPU backends currently
    /// fall through to CPU (TODO: CUDA/Vulkan kernel launch).
    pub fn quantize(&self, values: &[f32], out: &mut [u8]) {
        match &self.backend {
            Backend::Cpu(_) => self.cpu.quantize(values, out),
            #[cfg(feature = "cuda")]
            Backend::Cuda(_tier) => {
                // TODO: launch CUDA kernel via NVRTC
                // For now, fall through to CPU
                self.cpu.quantize(values, out);
            }
            #[cfg(feature = "vulkan")]
            Backend::Vulkan(_tier) => {
                // TODO: dispatch Vulkan compute shader
                self.cpu.quantize(values, out);
            }
            #[cfg(feature = "cubecl")]
            Backend::CubeCL => {
                // TODO: dispatch cubecl unified kernel
                // Falls through to CPU until cubecl kernels are implemented
                self.cpu.quantize(values, out);
            }
        }
    }

    /// Dequantize a batch of indices.
    pub fn dequantize(&self, indices: &[u8], out: &mut [f32]) {
        // Dequantize is a simple table lookup -- CPU is fast enough
        // that GPU dispatch overhead is not worth it for this alone.
        // GPU dequant happens inside fused dequant+dot kernels.
        self.cpu.dequantize(indices, out);
    }

    /// Get the active backend.
    pub fn backend(&self) -> &Backend {
        &self.backend
    }

    /// Get the CPU quantizer (for direct access when needed).
    pub fn cpu_quantizer(&self) -> &DispatchedQuantizer {
        &self.cpu
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lloyd_max;

    #[test]
    fn test_detect_best_backend() {
        let backend = detect_best_backend();
        // Should return something without panicking
        println!("Best backend: {}", backend);
        match &backend {
            Backend::Cpu(level) => {
                println!("  CPU SIMD level: {}", level);
            }
            #[cfg(feature = "cuda")]
            Backend::Cuda(tier) => {
                println!("  CUDA tier: {}", tier);
            }
            #[cfg(feature = "vulkan")]
            Backend::Vulkan(tier) => {
                println!("  Vulkan tier: {}", tier);
            }
        }
    }

    #[test]
    fn test_detect_all_backends() {
        let backends = detect_all_backends();
        assert!(!backends.is_empty(), "Should have at least CPU backend");
        for b in &backends {
            println!("Available: {}", b);
        }
    }

    #[test]
    fn test_backend_quantizer_correctness() {
        let cb = lloyd_max::get_codebook(128, 3);

        // CPU explicit
        let cpu_q = BackendQuantizer::with_backend(&cb, 3, Backend::Cpu(SimdLevel::Scalar));
        // Auto-detect
        let auto_q = BackendQuantizer::new(&cb, 3);

        let sigma = 1.0 / (128.0f32).sqrt();
        let values: Vec<f32> = (0..256)
            .map(|i| ((i as f32 * 0.618) % 1.0 - 0.5) * 7.0 * sigma)
            .collect();

        let mut cpu_out = vec![0u8; 256];
        let mut auto_out = vec![0u8; 256];
        cpu_q.quantize(&values, &mut cpu_out);
        auto_q.quantize(&values, &mut auto_out);

        // All backends should produce identical results
        assert_eq!(cpu_out, auto_out, "CPU scalar and auto-detect differ");
    }

    #[test]
    fn test_backend_display() {
        let cpu = Backend::Cpu(SimdLevel::Avx2);
        assert_eq!(format!("{}", cpu), "cpu-avx2");
    }
}

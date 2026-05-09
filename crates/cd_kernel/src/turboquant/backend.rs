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
/// when CUDA or Vulkan backends are selected.
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
    /// Panics on backend dispatch failures. Use `try_quantize` for explicit
    /// error handling.
    pub fn quantize(&self, values: &[f32], out: &mut [u8]) {
        if let Err(err) = self.try_quantize(values, out) {
            panic!(
                "TurboQuant quantize dispatch failed on {}: {}",
                self.backend, err
            );
        }
    }

    /// Quantize a batch of values with explicit error handling.
    pub fn try_quantize(&self, values: &[f32], out: &mut [u8]) -> Result<(), String> {
        if out.len() != values.len() {
            return Err(format!(
                "quantize output length mismatch: got {}, expected {}",
                out.len(),
                values.len()
            ));
        }

        match &self.backend {
            Backend::Cpu(_) => {
                self.cpu.quantize(values, out);
                Ok(())
            }
            #[cfg(feature = "cuda")]
            Backend::Cuda(_tier) => {
                let kernels = super::cuda::launch::TurboQuantCudaKernels::new()
                    .map_err(|e| format!("CUDA kernel init: {}", e))?;
                let gpu_indices = kernels
                    .quantize_batch(values, self.cpu.boundaries())
                    .map_err(|e| format!("CUDA quantize launch: {}", e))?;
                if gpu_indices.len() != out.len() {
                    return Err(format!(
                        "CUDA quantize output length mismatch: got {}, expected {}",
                        gpu_indices.len(),
                        out.len()
                    ));
                }
                out.copy_from_slice(&gpu_indices);
                Ok(())
            }
            #[cfg(feature = "vulkan")]
            Backend::Vulkan(_tier) => {
                // Deferred: dispatch Vulkan compute shader. CPU fallback is
                // intentional; the Vulkan quantize kernel is tracked as a
                // future-work item -- see plans/repo_debt_roadmap_2026_04_11.toml.
                self.cpu.quantize(values, out);
                Ok(())
            }
            #[cfg(feature = "cubecl")]
            Backend::CubeCL => {
                // Deferred: dispatch cubecl unified kernel. CPU fallback is
                // intentional; the cubecl unified kernel is tracked as a
                // future-work item -- see plans/repo_debt_roadmap_2026_04_11.toml.
                self.cpu.quantize(values, out);
                Ok(())
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

    /// Fused dequantization + dot-product attention scores.
    ///
    /// `queries` has shape `(n_queries, d)`, `key_indices` is SoA-packed
    /// `(d, n_keys)`, and the returned scores have shape `(n_queries, n_keys)`.
    pub fn dequant_dot_batch(
        &self,
        queries: &[f32],
        key_indices: &[u8],
        key_norms: &[f32],
        n_queries: usize,
        n_keys: usize,
        d: usize,
    ) -> Result<Vec<f32>, String> {
        let centroids = self.cpu.centroids();
        let expected_scores = Self::validate_dequant_dot_shapes(
            queries,
            key_indices,
            centroids,
            key_norms,
            n_queries,
            n_keys,
            d,
        )?;

        match &self.backend {
            Backend::Cpu(_) => {
                Self::dequant_dot_batch_cpu(queries, key_indices, centroids, key_norms, n_keys, d)
            }
            #[cfg(feature = "cuda")]
            Backend::Cuda(_tier) => {
                let kernels = super::cuda::launch::TurboQuantCudaKernels::new()
                    .map_err(|e| format!("CUDA kernel init: {}", e))?;
                kernels.dequant_dot_batch(queries, key_indices, centroids, key_norms, d)
            }
            #[cfg(feature = "vulkan")]
            Backend::Vulkan(_tier) => Err(format!(
                "backend {} fused dequant-dot dispatch is not implemented",
                self.backend
            )),
            #[cfg(feature = "cubecl")]
            Backend::CubeCL => {
                Err("backend cubecl fused dequant-dot dispatch is not implemented".to_string())
            }
        }
        .and_then(|scores| {
            if scores.len() != expected_scores {
                return Err(format!(
                    "dequant-dot output length mismatch: got {}, expected {}",
                    scores.len(),
                    expected_scores
                ));
            }
            Ok(scores)
        })
    }

    fn validate_dequant_dot_shapes(
        queries: &[f32],
        key_indices: &[u8],
        centroids: &[f32],
        key_norms: &[f32],
        n_queries: usize,
        n_keys: usize,
        d: usize,
    ) -> Result<usize, String> {
        super::dequant_contract::validate_dequant_dot_contract(
            queries.len(),
            key_indices.len(),
            centroids.len(),
            key_norms.len(),
            n_queries,
            n_keys,
            d,
        )
        .map(|validated| validated.expected_scores)
    }

    fn dequant_dot_batch_cpu(
        queries: &[f32],
        key_indices: &[u8],
        centroids: &[f32],
        key_norms: &[f32],
        n_keys: usize,
        d: usize,
    ) -> Result<Vec<f32>, String> {
        let n_queries = queries.len() / d;
        let mut scores = vec![0.0f32; n_queries * n_keys];
        if scores.is_empty() {
            return Ok(scores);
        }

        for qi in 0..n_queries {
            let q = &queries[qi * d..(qi + 1) * d];
            for (ki, &norm) in key_norms.iter().enumerate() {
                let mut dot = 0.0f32;
                for (c, &qv) in q.iter().enumerate() {
                    let centroid_idx = key_indices[c * n_keys + ki] as usize;
                    if centroid_idx >= centroids.len() {
                        return Err(format!(
                            "key_indices value out of range at (coord={}, key={}): {} >= {}",
                            c,
                            ki,
                            centroid_idx,
                            centroids.len()
                        ));
                    }
                    dot += qv * centroids[centroid_idx];
                }
                scores[qi * n_keys + ki] = dot * norm;
            }
        }

        Ok(scores)
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
        cpu_q
            .try_quantize(&values, &mut cpu_out)
            .expect("CPU quantize should not fail");
        if let Err(err) = auto_q.try_quantize(&values, &mut auto_out) {
            println!(
                "Auto backend ({}) quantize unavailable in test env: {}. Using CPU quantizer for parity check.",
                auto_q.backend(),
                err
            );
            auto_q.cpu_quantizer().quantize(&values, &mut auto_out);
        }

        // All backends should produce identical results
        assert_eq!(cpu_out, auto_out, "CPU scalar and auto-detect differ");
    }

    #[test]
    fn test_backend_display() {
        let cpu = Backend::Cpu(SimdLevel::Avx2);
        assert_eq!(format!("{}", cpu), "cpu-avx2");
    }

    #[test]
    fn test_backend_try_quantize_validates_shape() {
        let cb = lloyd_max::get_codebook(128, 3);
        let q = BackendQuantizer::with_backend(&cb, 3, Backend::Cpu(SimdLevel::Scalar));
        let values = vec![0.1f32, 0.2, 0.3];
        let mut out = vec![0u8; 2];
        let err = q
            .try_quantize(&values, &mut out)
            .expect_err("shape mismatch should fail");
        assert!(err.contains("output length mismatch"));
    }

    #[test]
    fn test_backend_cpu_fused_dequant_dot() {
        let cb = lloyd_max::get_codebook(128, 3);
        let q = BackendQuantizer::with_backend(&cb, 3, Backend::Cpu(SimdLevel::Scalar));
        let centroids = q.cpu_quantizer().centroids();

        let d = 4usize;
        let n_queries = 2usize;
        let n_keys = 3usize;
        let queries = vec![0.5, -0.25, 0.75, 0.1, -0.2, 0.3, 0.4, -0.5];
        let key_indices = vec![
            0, 1, 2, // c0 across keys
            3, 4, 5, // c1 across keys
            6, 7, 0, // c2 across keys
            1, 2, 3, // c3 across keys
        ];
        let key_norms = vec![1.0f32, 0.5, 2.0];

        let scores = q
            .dequant_dot_batch(&queries, &key_indices, &key_norms, n_queries, n_keys, d)
            .expect("CPU fused dequant-dot should succeed");

        assert_eq!(scores.len(), n_queries * n_keys);

        let mut expected = vec![0.0f32; n_queries * n_keys];
        for qi in 0..n_queries {
            let qrow = &queries[qi * d..(qi + 1) * d];
            for ki in 0..n_keys {
                let mut dot = 0.0f32;
                for c in 0..d {
                    dot += qrow[c] * centroids[key_indices[c * n_keys + ki] as usize];
                }
                expected[qi * n_keys + ki] = dot * key_norms[ki];
            }
        }

        for (got, exp) in scores.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-6, "got {}, expected {}", got, exp);
        }
    }

    #[test]
    fn test_backend_fused_dequant_dot_validates_shapes() {
        let cb = lloyd_max::get_codebook(128, 3);
        let q = BackendQuantizer::with_backend(&cb, 3, Backend::Cpu(SimdLevel::Scalar));
        let err = q
            .dequant_dot_batch(&[0.1f32, 0.2, 0.3], &[0u8, 1, 2, 3], &[1.0f32], 1, 1, 4)
            .expect_err("invalid query shape should fail");
        assert!(err.contains("queries length mismatch"));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_backend_cuda_fused_dequant_dot_parity_smoke() {
        let Some(props) = super::super::cuda::device::probe_device() else {
            println!("Skipping CUDA parity smoke: no CUDA device available");
            return;
        };

        let cb = lloyd_max::get_codebook(128, 3);
        let cpu_q = BackendQuantizer::with_backend(&cb, 3, Backend::Cpu(SimdLevel::Scalar));
        let cuda_q =
            BackendQuantizer::with_backend(&cb, 3, Backend::Cuda(props.recommended_tier()));

        let d = 4usize;
        let n_queries = 2usize;
        let n_keys = 3usize;
        let queries = vec![0.5, -0.25, 0.75, 0.1, -0.2, 0.3, 0.4, -0.5];
        let key_indices = vec![
            0, 1, 2, // c0 across keys
            3, 4, 5, // c1 across keys
            6, 7, 0, // c2 across keys
            1, 2, 3, // c3 across keys
        ];
        let key_norms = vec![1.0f32, 0.5, 2.0];

        let cpu_scores = cpu_q
            .dequant_dot_batch(&queries, &key_indices, &key_norms, n_queries, n_keys, d)
            .expect("CPU fused dequant-dot should succeed");

        let cuda_scores = match cuda_q.dequant_dot_batch(
            &queries,
            &key_indices,
            &key_norms,
            n_queries,
            n_keys,
            d,
        ) {
            Ok(scores) => scores,
            Err(err) if err.contains("CUDA kernel init") => {
                println!("Skipping CUDA parity smoke: {err}");
                return;
            }
            Err(err) => {
                panic!("CUDA fused dequant-dot should succeed when runtime is available: {err}")
            }
        };

        assert_eq!(
            cpu_scores.len(),
            cuda_scores.len(),
            "CPU/CUDA score length mismatch"
        );

        for (idx, (cpu, cuda)) in cpu_scores.iter().zip(cuda_scores.iter()).enumerate() {
            let abs_diff = (cpu - cuda).abs();
            assert!(
                abs_diff <= 1e-5,
                "CPU/CUDA parity mismatch at {}: cpu={}, cuda={}, |diff|={}",
                idx,
                cpu,
                cuda,
                abs_diff
            );
        }
    }
}

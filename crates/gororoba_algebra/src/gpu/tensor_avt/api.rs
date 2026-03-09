use gororoba_gpu_bridge::ComputeBackend;

#[cfg(feature = "gpu")]
use super::cuda;
use super::{
    cpu::{TensorAvtCpuMulSession, TensorAvtCpuNormSession},
    policy,
    sessions::{
        TensorAvtMulSession, TensorAvtMulSessionInner, TensorAvtNormSession,
        TensorAvtNormSessionInner,
    },
    vulkan,
};

/// Tensor Core AVT contraction engine.
///
/// For dense CD multiplication, constructs the Left-Multiplication Matrix
/// `L_a` (dim x dim) and decomposes it into (dim/16)^2 WMMA tiles:
/// - dim=256: 256 tiles (16x16 each), 128 KB in FP16
/// - dim=512: 1024 tiles, 512 KB in FP16
/// - dim=1024: 4096 tiles, 2 MB in FP16 (requires tiled streaming)
///
/// For basis-element-only AVT sampling, uses the sign-table kernel
/// (no `L_a` construction needed -- basis elements have single component).
pub struct TensorAVT {
    /// CD algebra dimension (must be power of 2, >= 256)
    pub dim: usize,
    /// Number of 16x16 tiles per dimension
    pub tile_count: usize,
}

impl TensorAVT {
    /// Create a new Tensor Core AVT engine for the given dimension.
    ///
    /// # Panics
    /// Panics if dim is not a power of 2 or is less than 16.
    pub fn new(dim: usize) -> Self {
        assert!(
            dim >= 16 && dim.is_power_of_two(),
            "TensorAVT requires dim >= 16 and power of 2, got {dim}"
        );
        Self {
            dim,
            tile_count: dim / 16,
        }
    }

    fn validate_backend_request(&self, backend: ComputeBackend) -> Result<ComputeBackend, String> {
        match backend {
            ComputeBackend::CpuScalar => Ok(backend),
            ComputeBackend::CpuSimd => {
                if policy::simd_available() {
                    Ok(backend)
                } else {
                    Err("TensorAVT CPU SIMD backend unavailable on this machine".into())
                }
            }
            ComputeBackend::Vulkan => Err(vulkan::tensor_avt_vulkan_error()),
            ComputeBackend::Cuda => {
                #[cfg(feature = "gpu")]
                {
                    if cuda::tensor_avt_cuda_available() {
                        Ok(backend)
                    } else {
                        Err("TensorAVT CUDA backend unavailable on this machine".into())
                    }
                }
                #[cfg(not(feature = "gpu"))]
                {
                    let _ = self;
                    Err(
                        "TensorAVT CUDA backend requires building gororoba_algebra with --features gpu"
                            .into(),
                    )
                }
            }
        }
    }

    pub fn new_mul_session(
        &self,
        backend: ComputeBackend,
        max_batch_size: usize,
    ) -> Result<TensorAvtMulSession, String> {
        if max_batch_size == 0 {
            return Err("max_batch_size must be > 0".into());
        }
        let backend = self.validate_backend_request(backend)?;
        let inner = match backend {
            ComputeBackend::CpuScalar | ComputeBackend::CpuSimd => {
                TensorAvtMulSessionInner::Cpu(TensorAvtCpuMulSession::new(self.dim, max_batch_size))
            }
            #[cfg(feature = "gpu")]
            ComputeBackend::Cuda => {
                TensorAvtMulSessionInner::Cuda(self.new_gpu_mul_workspace(max_batch_size)?)
            }
            ComputeBackend::Vulkan => unreachable!("validated unsupported Vulkan above"),
            #[cfg(not(feature = "gpu"))]
            ComputeBackend::Cuda => unreachable!("validated unavailable CUDA above"),
        };
        Ok(TensorAvtMulSession {
            backend,
            dim: self.dim,
            max_batch_size,
            inner,
        })
    }

    pub fn new_norm_session(
        &self,
        backend: ComputeBackend,
        max_vectors: usize,
    ) -> Result<TensorAvtNormSession, String> {
        if max_vectors == 0 {
            return Err("max_vectors must be > 0".into());
        }
        let backend = self.validate_backend_request(backend)?;
        let inner = match backend {
            ComputeBackend::CpuScalar | ComputeBackend::CpuSimd => {
                TensorAvtNormSessionInner::Cpu(TensorAvtCpuNormSession::new(self.dim, max_vectors))
            }
            #[cfg(feature = "gpu")]
            ComputeBackend::Cuda => {
                TensorAvtNormSessionInner::Cuda(self.new_gpu_norm_workspace(max_vectors)?)
            }
            ComputeBackend::Vulkan => unreachable!("validated unsupported Vulkan above"),
            #[cfg(not(feature = "gpu"))]
            ComputeBackend::Cuda => unreachable!("validated unavailable CUDA above"),
        };
        Ok(TensorAvtNormSession {
            backend,
            dim: self.dim,
            max_vectors,
            inner,
        })
    }
}

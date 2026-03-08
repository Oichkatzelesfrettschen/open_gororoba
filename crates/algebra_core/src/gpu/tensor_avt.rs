//! Tensor Core AVT contraction for high-dimensional CD algebras.
//!
//! CD multiplication is NOT standard matrix multiplication. For any fixed
//! vector `a`, the map `x -> a*x` is linear and represented by the
//! Left-Multiplication Matrix `L_a` (dim x dim), where:
//!   `L_a[i][j] = a_{i XOR j} * gamma(i XOR j, j)`
//!
//! For dim=256, `L_a` is 256x256 = 128 KB in FP16, fitting in SM shared
//! memory. It decomposes into (dim/16)^2 WMMA 16x16 tiles on the RTX 4070 Ti
//! (SM 8.9, 4th-gen Tensor Cores).
//!
//! FP16 storage + FP32 accumulate: `wmma::mma_sync` does
//! A(FP16) x B(FP16) + C(FP32) -> D(FP32).
//!
//! For CD basis elements (entries are exactly 0 or +/-1), FP16 is EXACT.
//! The FP32 accumulator preserves full single-precision for dense vectors.

#[path = "tensor_avt_cpu.rs"]
mod cpu;
#[path = "tensor_avt_cuda.rs"]
mod cuda;
#[path = "tensor_avt_policy.rs"]
mod policy;
#[path = "tensor_avt_vulkan.rs"]
mod vulkan;

use self::cpu::{TensorAvtCpuMulSession, TensorAvtCpuNormSession};
use gororoba_gpu_bridge::ComputeBackend;

#[cfg(feature = "gpu")]
pub use self::cuda::{TensorAvtMulGpuWorkspace, TensorAvtNormGpuWorkspace};
pub use self::policy::{
    TensorAvtAutoConfig, TensorAvtAutoResult, TensorAvtCalibrationMode, TensorAvtThresholdOverrides,
};

enum TensorAvtMulSessionInner {
    Cpu(TensorAvtCpuMulSession),
    #[cfg(feature = "gpu")]
    Cuda(TensorAvtMulGpuWorkspace),
}

pub struct TensorAvtMulSession {
    backend: ComputeBackend,
    dim: usize,
    max_batch_size: usize,
    inner: TensorAvtMulSessionInner,
}

impl TensorAvtMulSession {
    pub fn backend(&self) -> ComputeBackend {
        self.backend
    }

    pub fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    pub fn load_left(&mut self, a: &[f32]) -> Result<(), String> {
        if a.len() != self.dim {
            return Err(format!(
                "left input length {} must equal dim {}",
                a.len(),
                self.dim
            ));
        }
        match &mut self.inner {
            TensorAvtMulSessionInner::Cpu(session) => {
                session.left.copy_from_slice(a);
                Ok(())
            }
            #[cfg(feature = "gpu")]
            TensorAvtMulSessionInner::Cuda(workspace) => workspace.upload_a(a),
        }
    }

    pub fn load_right(&mut self, x: &[f32], batch_size: usize) -> Result<(), String> {
        if batch_size == 0 || batch_size > self.max_batch_size {
            return Err(format!(
                "batch_size must be in 1..={}, got {}",
                self.max_batch_size, batch_size
            ));
        }
        let expected = batch_size * self.dim;
        if x.len() != expected {
            return Err(format!(
                "right input length {} must equal batch_size * dim {}",
                x.len(),
                expected
            ));
        }
        match &mut self.inner {
            TensorAvtMulSessionInner::Cpu(session) => {
                session.right[..expected].copy_from_slice(x);
                Ok(())
            }
            #[cfg(feature = "gpu")]
            TensorAvtMulSessionInner::Cuda(workspace) => {
                workspace.upload_x(x, batch_size, self.dim)
            }
        }
    }

    pub fn run_single(&mut self, avt: &TensorAVT) -> Result<(), String> {
        match &mut self.inner {
            TensorAvtMulSessionInner::Cpu(session) => {
                session.output =
                    avt.compute_cd_mul_cpu(&session.left, &session.right[..self.dim])?;
                Ok(())
            }
            #[cfg(feature = "gpu")]
            TensorAvtMulSessionInner::Cuda(workspace) => {
                avt.launch_cd_mul_with_workspace(workspace)
            }
        }
    }

    pub fn run_batch(&mut self, avt: &TensorAVT, batch_size: usize) -> Result<(), String> {
        if batch_size == 0 || batch_size > self.max_batch_size {
            return Err(format!(
                "batch_size must be in 1..={}, got {}",
                self.max_batch_size, batch_size
            ));
        }
        let expected = batch_size * self.dim;
        match &mut self.inner {
            TensorAvtMulSessionInner::Cpu(session) => {
                session.output = avt.compute_cd_mul_batch_cpu(
                    &session.left,
                    &session.right[..expected],
                    batch_size,
                )?;
                Ok(())
            }
            #[cfg(feature = "gpu")]
            TensorAvtMulSessionInner::Cuda(workspace) => {
                avt.launch_cd_mul_batch_with_workspace(batch_size, workspace)
            }
        }
    }

    pub fn download_output(&mut self, len: usize) -> Result<Vec<f32>, String> {
        if len > self.dim * self.max_batch_size {
            return Err(format!(
                "output length {} exceeds session capacity {}",
                len,
                self.dim * self.max_batch_size
            ));
        }
        match &mut self.inner {
            TensorAvtMulSessionInner::Cpu(session) => {
                if len > session.output.len() {
                    return Err(format!(
                        "requested output length {} exceeds computed output length {}",
                        len,
                        session.output.len()
                    ));
                }
                Ok(session.output[..len].to_vec())
            }
            #[cfg(feature = "gpu")]
            TensorAvtMulSessionInner::Cuda(workspace) => workspace.download_y(len),
        }
    }
}

enum TensorAvtNormSessionInner {
    Cpu(TensorAvtCpuNormSession),
    #[cfg(feature = "gpu")]
    Cuda(TensorAvtNormGpuWorkspace),
}

pub struct TensorAvtNormSession {
    backend: ComputeBackend,
    dim: usize,
    max_vectors: usize,
    inner: TensorAvtNormSessionInner,
}

impl TensorAvtNormSession {
    pub fn backend(&self) -> ComputeBackend {
        self.backend
    }

    pub fn max_vectors(&self) -> usize {
        self.max_vectors
    }

    pub fn load_vectors(&mut self, vectors: &[f32], n_vectors: usize) -> Result<(), String> {
        if n_vectors == 0 || n_vectors > self.max_vectors {
            return Err(format!(
                "n_vectors must be in 1..={}, got {}",
                self.max_vectors, n_vectors
            ));
        }
        let expected = n_vectors * self.dim;
        if vectors.len() != expected {
            return Err(format!(
                "vectors length {} must equal n_vectors * dim {}",
                vectors.len(),
                expected
            ));
        }
        match &mut self.inner {
            TensorAvtNormSessionInner::Cpu(session) => {
                session.vectors[..expected].copy_from_slice(vectors);
                Ok(())
            }
            #[cfg(feature = "gpu")]
            TensorAvtNormSessionInner::Cuda(workspace) => {
                workspace.upload_vectors(vectors, n_vectors, self.dim)
            }
        }
    }

    pub fn run_norms(&mut self, avt: &TensorAVT, n_vectors: usize) -> Result<(), String> {
        if n_vectors == 0 || n_vectors > self.max_vectors {
            return Err(format!(
                "n_vectors must be in 1..={}, got {}",
                self.max_vectors, n_vectors
            ));
        }
        let expected = n_vectors * self.dim;
        match &mut self.inner {
            TensorAvtNormSessionInner::Cpu(session) => {
                session.norms =
                    avt.compute_norm_sq_batch_cpu(&session.vectors[..expected], n_vectors)?;
                Ok(())
            }
            #[cfg(feature = "gpu")]
            TensorAvtNormSessionInner::Cuda(workspace) => {
                avt.launch_norm_sq_batch_with_workspace(n_vectors, workspace)
            }
        }
    }

    pub fn download_norms(&mut self, n_vectors: usize) -> Result<Vec<f32>, String> {
        if n_vectors == 0 || n_vectors > self.max_vectors {
            return Err(format!(
                "n_vectors must be in 1..={}, got {}",
                self.max_vectors, n_vectors
            ));
        }
        match &mut self.inner {
            TensorAvtNormSessionInner::Cpu(session) => {
                if n_vectors > session.norms.len() {
                    return Err(format!(
                        "requested norms length {} exceeds computed length {}",
                        n_vectors,
                        session.norms.len()
                    ));
                }
                Ok(session.norms[..n_vectors].to_vec())
            }
            #[cfg(feature = "gpu")]
            TensorAvtNormSessionInner::Cuda(workspace) => workspace.download_norms(n_vectors),
        }
    }
}

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
                        "TensorAVT CUDA backend requires building algebra_core with --features gpu"
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

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_vec_close(got: &[f32], expected: &[f32], tol: f32, label: &str) {
        assert_eq!(
            got.len(),
            expected.len(),
            "{label}: length mismatch {} != {}",
            got.len(),
            expected.len()
        );
        for (idx, (lhs, rhs)) in got.iter().zip(expected.iter()).enumerate() {
            let diff = (*lhs - *rhs).abs();
            assert!(
                diff <= tol,
                "{label}: mismatch at [{idx}] got {lhs}, expected {rhs}, diff {diff}"
            );
        }
    }

    #[test]
    fn test_tensor_avt_creation() {
        let avt = TensorAVT::new(256);
        assert_eq!(avt.dim, 256);
        assert_eq!(avt.tile_count, 16);

        let avt1024 = TensorAVT::new(1024);
        assert_eq!(avt1024.tile_count, 64);
    }

    #[test]
    #[should_panic(expected = "TensorAVT requires dim >= 16")]
    fn test_tensor_avt_invalid_dim() {
        let _avt = TensorAVT::new(8);
    }

    #[test]
    fn test_norm_sq_batch_cpu_fallback() {
        let avt = TensorAVT::new(16);
        let mut vec = vec![0.0f32; 16];
        vec[0] = 1.0;
        let norms = avt.compute_norm_sq_batch_cpu(&vec, 1).unwrap();
        assert!((norms[0] - 1.0).abs() < 1e-6);

        let uniform: Vec<f32> = vec![0.25; 16];
        let norms2 = avt.compute_norm_sq_batch_cpu(&uniform, 1).unwrap();
        assert!((norms2[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cd_mul_cpu_basis_identity() {
        let avt = TensorAVT::new(16);
        let mut e0 = vec![0.0f32; 16];
        e0[0] = 1.0;
        let x: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1 + 0.5).collect();
        let y = avt.compute_cd_mul_cpu(&e0, &x).unwrap();
        for i in 0..16 {
            assert!((y[i] - x[i]).abs() < 1e-2, "e_0 * x != x at index {i}");
        }
    }

    #[test]
    fn test_cd_mul_cpu_cross_validates_cd_multiply() {
        use crate::construction::cayley_dickson::cd_multiply;
        let avt = TensorAVT::new(16);
        let a: Vec<f32> = (0..16)
            .map(|i| ((i * 7 + 3) % 11) as f32 * 0.2 - 1.0)
            .collect();
        let x: Vec<f32> = (0..16)
            .map(|i| ((i * 13 + 5) % 11) as f32 * 0.15 - 0.8)
            .collect();
        let y_gpu_cpu = avt.compute_cd_mul_cpu(&a, &x).unwrap();

        let a64: Vec<f64> = a.iter().map(|&v| v as f64).collect();
        let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        let y_ref = cd_multiply(&a64, &x64);

        for (i, (got, expected)) in y_gpu_cpu.iter().zip(y_ref.iter()).enumerate() {
            let diff = (*got as f64 - *expected).abs();
            assert!(
                diff < 1e-1,
                "cd_mul mismatch at [{i}]: got {}, expected {}",
                got,
                expected
            );
        }
    }

    #[test]
    fn test_cd_mul_cpu_256d_basis_product() {
        use crate::construction::cayley_dickson::cd_basis_mul_sign;
        let avt = TensorAVT::new(256);
        let mut e1 = vec![0.0f32; 256];
        e1[1] = 1.0;
        let mut e2 = vec![0.0f32; 256];
        e2[2] = 1.0;
        let y = avt.compute_cd_mul_cpu(&e1, &e2).unwrap();

        let expected_idx = 1 ^ 2;
        let expected_sign = cd_basis_mul_sign(256, 1, 2) as f32;
        for (i, value) in y.iter().enumerate().take(256) {
            if i == expected_idx {
                assert!(
                    (*value - expected_sign).abs() < 1e-6,
                    "e_1 * e_2 should be {expected_sign} * e_{expected_idx}, got {}",
                    value
                );
            } else {
                assert!(
                    value.abs() < 1e-6,
                    "e_1 * e_2 should be zero at [{i}], got {}",
                    value
                );
            }
        }
    }

    #[test]
    fn test_cd_mul_cpu_512d_identity() {
        let avt = TensorAVT::new(512);
        let mut e0 = vec![0.0f32; 512];
        e0[0] = 1.0;
        let x: Vec<f32> = (0..512)
            .map(|i| ((i * 37 + 11) % 97) as f32 * 0.01)
            .collect();
        let y = avt.compute_cd_mul_cpu(&e0, &x).unwrap();
        for (i, (got, expected)) in y.iter().zip(x.iter()).enumerate() {
            assert!(
                (*got - *expected).abs() < 1e-2,
                "e_0 * x != x at dim=512 index {i}"
            );
        }
    }

    #[test]
    fn test_cd_mul_cpu_1024d_identity() {
        let avt = TensorAVT::new(1024);
        let mut e0 = vec![0.0f32; 1024];
        e0[0] = 1.0;
        let x: Vec<f32> = (0..1024)
            .map(|i| ((i * 41 + 7) % 101) as f32 * 0.01)
            .collect();
        let y = avt.compute_cd_mul_cpu(&e0, &x).unwrap();
        for (i, (got, expected)) in y.iter().zip(x.iter()).enumerate() {
            assert!(
                (*got - *expected).abs() < 1e-2,
                "e_0 * x != x at dim=1024 index {i}"
            );
        }
    }

    fn reference_cd_multiply(a: &[f32], x: &[f32], dim: usize) -> Vec<f32> {
        use crate::construction::cayley_dickson::cd_basis_mul_sign;
        let mut result = vec![0.0f32; dim];
        for (i, a_i) in a.iter().enumerate().take(dim) {
            for (j, x_j) in x.iter().enumerate().take(dim) {
                let sign = cd_basis_mul_sign(dim, i, j);
                let target_idx = i ^ j;
                result[target_idx] += sign as f32 * *a_i * *x_j;
            }
        }
        result
    }

    #[test]
    fn test_reference_cd_mul_dim16_sincos() {
        let dim = 16;
        let avt = TensorAVT::new(dim);
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.5).sin()).collect();
        let x: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.3).cos()).collect();

        let y_la = avt.compute_cd_mul_cpu(&a, &x).unwrap();
        let y_ref = reference_cd_multiply(&a, &x, dim);

        for (i, (got, expected)) in y_la.iter().zip(y_ref.iter()).enumerate() {
            let diff = (*got - *expected).abs();
            assert!(
                diff < 1e-1,
                "dim=16 sincos mismatch at [{i}]: L_a={got}, ref={expected}, diff={diff}"
            );
        }
    }

    #[test]
    fn test_reference_cd_mul_dim256_sincos() {
        let dim = 256;
        let avt = TensorAVT::new(dim);
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let x: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.7).cos()).collect();

        let y_la = avt.compute_cd_mul_cpu(&a, &x).unwrap();
        let y_ref = reference_cd_multiply(&a, &x, dim);

        for (i, (got, expected)) in y_la.iter().zip(y_ref.iter()).enumerate() {
            let diff = (*got - *expected).abs();
            assert!(
                diff < 1e-1,
                "dim=256 sincos mismatch at [{i}]: L_a={got}, ref={expected}, diff={diff}"
            );
        }
    }

    #[test]
    #[cfg(feature = "gpu")]
    #[ignore = "gpu"]
    fn test_gpu_cd_mul_256d_cross_validates_cpu() {
        use crate::construction::cayley_dickson::cd_multiply;
        if !crate::gpu::is_gpu_available() {
            eprintln!("SKIP: CUDA device not visible to tensor_avt GPU test");
            return;
        }
        let dim = 256;
        let avt = TensorAVT::new(dim);
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let x: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.7).cos()).collect();

        let y_gpu = avt.compute_cd_mul(&a, &x).unwrap();
        let y_ref = reference_cd_multiply(&a, &x, dim);

        let a64: Vec<f64> = a.iter().map(|&v| v as f64).collect();
        let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        let y_cd = cd_multiply(&a64, &x64);

        for i in 0..dim {
            let gpu_val = y_gpu[i];
            let ref_val = y_ref[i];
            let cd_val = y_cd[i] as f32;
            let diff_ref = (gpu_val - ref_val).abs();
            assert!(
                diff_ref < 1e-2,
                "GPU vs ref mismatch at [{i}]: gpu={gpu_val}, ref={ref_val}, diff={diff_ref}"
            );

            let diff_cd = (gpu_val - cd_val).abs();
            assert!(
                diff_cd < 1e-1,
                "GPU vs cd_multiply mismatch at [{i}]: gpu={gpu_val}, cd={cd_val}, diff={diff_cd}"
            );
        }
    }

    #[test]
    #[cfg(feature = "gpu")]
    #[ignore = "gpu"]
    fn test_gpu_cd_mul_16d_identity() {
        if !crate::gpu::is_gpu_available() {
            eprintln!("SKIP: CUDA device not visible to tensor_avt GPU test");
            return;
        }
        let dim = 16;
        let avt = TensorAVT::new(dim);
        let mut e0 = vec![0.0f32; dim];
        e0[0] = 1.0;
        let x: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1 + 0.5).collect();
        let y = avt.compute_cd_mul(&e0, &x).unwrap();
        for i in 0..dim {
            assert!(
                (y[i] - x[i]).abs() < 1e-2,
                "GPU e_0 * x != x at index {i}: got {}, expected {}",
                y[i],
                x[i]
            );
        }
    }

    #[test]
    fn test_cd_mul_batch_cpu_dim16() {
        let dim = 16;
        let batch_size = 4;
        let avt = TensorAVT::new(dim);

        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.3).sin()).collect();
        let mut x_batch = vec![0.0f32; batch_size * dim];
        for b in 0..batch_size {
            for d in 0..dim {
                x_batch[b * dim + d] = ((b * dim + d) as f32 * 0.17).cos();
            }
        }

        let y_batch = avt
            .compute_cd_mul_batch_cpu(&a, &x_batch, batch_size)
            .unwrap();

        for b in 0..batch_size {
            let x_single: Vec<f32> = x_batch[b * dim..(b + 1) * dim].to_vec();
            let y_single = avt.compute_cd_mul_cpu(&a, &x_single).unwrap();

            for d in 0..dim {
                let got = y_batch[b * dim + d];
                let expected = y_single[d];
                let diff = (got - expected).abs();
                assert!(
                    diff < 1e-5,
                    "batch[{b}] dim[{d}]: got {got}, expected {expected}, diff {diff}"
                );
            }
        }
    }

    #[test]
    fn test_cd_mul_batch_cpu_dim256() {
        let dim = 256;
        let batch_size = 16;
        let avt = TensorAVT::new(dim);

        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut x_batch = vec![0.0f32; batch_size * dim];
        for b in 0..batch_size {
            for d in 0..dim {
                x_batch[b * dim + d] = ((b * 3 + d * 7) as f32 * 0.01).cos();
            }
        }

        let y_batch = avt
            .compute_cd_mul_batch_cpu(&a, &x_batch, batch_size)
            .unwrap();

        for b in [0, batch_size - 1] {
            let x_single: Vec<f32> = x_batch[b * dim..(b + 1) * dim].to_vec();
            let y_single = avt.compute_cd_mul_cpu(&a, &x_single).unwrap();

            for d in 0..dim {
                let got = y_batch[b * dim + d];
                let expected = y_single[d];
                let diff = (got - expected).abs();
                assert!(
                    diff < 1e-2,
                    "batch[{b}] dim[{d}]: got {got}, expected {expected}, diff {diff}"
                );
            }
        }
    }

    #[test]
    fn test_mul_session_cpu_scalar_matches_explicit_paths() {
        let dim = 16;
        let batch_size = 4;
        let avt = TensorAVT::new(dim);
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.25).sin()).collect();
        let x_single: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.13).cos()).collect();
        let x_batch: Vec<f32> = (0..(dim * batch_size))
            .map(|i| (i as f32 * 0.09).sin())
            .collect();

        let mut single_session = avt
            .new_mul_session(ComputeBackend::CpuScalar, 1)
            .expect("create CPU scalar single session");
        single_session.load_left(&a).expect("load left");
        single_session
            .load_right(&x_single, 1)
            .expect("load right single");
        single_session.run_single(&avt).expect("run single");
        let got_single = single_session
            .download_output(dim)
            .expect("download single output");
        let expected_single = avt
            .compute_cd_mul_cpu(&a, &x_single)
            .expect("explicit single");
        assert_vec_close(
            &got_single,
            &expected_single,
            1e-6,
            "cpu scalar single session",
        );

        let mut batch_session = avt
            .new_mul_session(ComputeBackend::CpuScalar, batch_size)
            .expect("create CPU scalar batch session");
        batch_session.load_left(&a).expect("load batch left");
        batch_session
            .load_right(&x_batch, batch_size)
            .expect("load batch right");
        batch_session
            .run_batch(&avt, batch_size)
            .expect("run batch");
        let got_batch = batch_session
            .download_output(dim * batch_size)
            .expect("download batch output");
        let expected_batch = avt
            .compute_cd_mul_batch_cpu(&a, &x_batch, batch_size)
            .expect("explicit batch");
        assert_vec_close(
            &got_batch,
            &expected_batch,
            1e-6,
            "cpu scalar batch session",
        );
    }

    #[test]
    fn test_norm_session_cpu_scalar_matches_explicit_path() {
        let dim = 16;
        let batch_size = 3;
        let avt = TensorAVT::new(dim);
        let vectors: Vec<f32> = (0..(dim * batch_size))
            .map(|i| ((i * 5 + 1) as f32 * 0.07).cos())
            .collect();

        let mut session = avt
            .new_norm_session(ComputeBackend::CpuScalar, batch_size)
            .expect("create CPU scalar norm session");
        session
            .load_vectors(&vectors, batch_size)
            .expect("load vectors");
        session.run_norms(&avt, batch_size).expect("run norms");
        let got = session
            .download_norms(batch_size)
            .expect("download norm output");
        let expected = avt
            .compute_norm_sq_batch_cpu(&vectors, batch_size)
            .expect("explicit norms");
        assert_vec_close(&got, &expected, 1e-6, "cpu scalar norm session");
    }

    #[test]
    fn test_vulkan_sessions_are_rejected() {
        let avt = TensorAVT::new(16);
        let mul_err = match avt.new_mul_session(ComputeBackend::Vulkan, 1) {
            Ok(_) => panic!("Vulkan mul session must be unsupported"),
            Err(err) => err,
        };
        assert!(
            mul_err.contains("Vulkan backend is not implemented"),
            "unexpected mul session error: {mul_err}"
        );

        let norm_err = match avt.new_norm_session(ComputeBackend::Vulkan, 1) {
            Ok(_) => panic!("Vulkan norm session must be unsupported"),
            Err(err) => err,
        };
        assert!(
            norm_err.contains("Vulkan backend is not implemented"),
            "unexpected norm session error: {norm_err}"
        );
    }

    #[test]
    fn test_auto_cd_mul_respects_cpu_threshold_override() {
        let dim = 16;
        let avt = TensorAVT::new(dim);
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.17).sin()).collect();
        let x: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.11).cos()).collect();
        let config = TensorAvtAutoConfig {
            backend_order: [
                ComputeBackend::CpuScalar,
                ComputeBackend::CpuSimd,
                ComputeBackend::Cuda,
                ComputeBackend::Vulkan,
            ],
            calibration: TensorAvtCalibrationMode::Disabled,
            threshold_overrides: TensorAvtThresholdOverrides {
                cd_mul_min_problem_size: Some(dim + 1),
                ..TensorAvtThresholdOverrides::default()
            },
        };

        let auto = avt
            .compute_cd_mul_auto_with_config(&a, &x, &config)
            .expect("auto cd_mul");
        let expected = avt.compute_cd_mul_cpu(&a, &x).expect("explicit single");
        assert_eq!(auto.backend, ComputeBackend::CpuScalar);
        assert!(!auto.calibrated_this_call);
        assert_vec_close(&auto.value, &expected, 1e-6, "auto cd_mul override");
    }

    #[test]
    fn test_auto_norm_respects_cpu_threshold_override() {
        let dim = 16;
        let batch_size = 2;
        let avt = TensorAVT::new(dim);
        let vectors: Vec<f32> = (0..(dim * batch_size))
            .map(|i| ((i * 3 + 2) as f32 * 0.19).sin())
            .collect();
        let config = TensorAvtAutoConfig {
            backend_order: [
                ComputeBackend::CpuScalar,
                ComputeBackend::CpuSimd,
                ComputeBackend::Cuda,
                ComputeBackend::Vulkan,
            ],
            calibration: TensorAvtCalibrationMode::Disabled,
            threshold_overrides: TensorAvtThresholdOverrides {
                norm_sq_batch_min_problem_size: Some(dim * batch_size + 1),
                ..TensorAvtThresholdOverrides::default()
            },
        };

        let auto = avt
            .compute_norm_sq_batch_auto_with_config(&vectors, batch_size, &config)
            .expect("auto norms");
        let expected = avt
            .compute_norm_sq_batch_cpu(&vectors, batch_size)
            .expect("explicit norms");
        assert_eq!(auto.backend, ComputeBackend::CpuScalar);
        assert!(!auto.calibrated_this_call);
        assert_vec_close(&auto.value, &expected, 1e-6, "auto norm override");
    }
}

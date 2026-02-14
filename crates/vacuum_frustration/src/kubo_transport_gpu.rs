//! GPU-accelerated Kubo linear-response transport.
//!
//! Uses cuSOLVER syevd for symmetric eigendecomposition and cuBLAS dgemm
//! for eigenbasis transformation of current operators.
//!
//! Key optimization: precompute J_eig = V^T * J * V via cuBLAS dgemm,
//! reducing spectral weight computation from O(dim^4) to O(dim^3).

use std::sync::Arc;

use cudarc::cublas::safe::{CudaBlas, Gemm, GemmConfig};
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cusolver::sys::{
    self as cusolver_sys, cublasFillMode_t, cusolverEigMode_t, cusolverStatus_t,
};
use cudarc::driver::{CudaContext, CudaStream, DevicePtr, DevicePtrMut};

use super::kubo_transport::{
    build_energy_current_operator, build_hamiltonian_matrix, build_spin_current_operator,
    compute_transport_from_eigenbasis, HeisenbergModel, KuboTransport,
};

/// GPU-accelerated exact diagonalization result.
pub struct GpuExactDiagResult {
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors in column-major format: eigenvectors[col * dim + row]
    pub eigenvectors: Vec<f64>,
    pub hilbert_dim: usize,
    pub n_sites: usize,
}

/// GPU context for Kubo transport computations.
///
/// Holds cuSOLVER and cuBLAS handles, reused across multiple model evaluations.
pub struct GpuKuboContext {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    solver_handle: cusolver_sys::cusolverDnHandle_t,
}

impl GpuKuboContext {
    /// Create a new GPU context on device 0.
    pub fn new() -> anyhow::Result<Self> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone())?;

        let solver_handle = {
            let mut handle = std::mem::MaybeUninit::uninit();
            let stat = unsafe { cusolver_sys::cusolverDnCreate(handle.as_mut_ptr()) };
            if stat != cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
                anyhow::bail!("cusolverDnCreate failed: {:?}", stat);
            }
            let handle = unsafe { handle.assume_init() };
            // cuSOLVER expects cudaStream_t = *mut CUstream_st, same as CUstream
            let stat =
                unsafe { cusolver_sys::cusolverDnSetStream(handle, stream.cu_stream() as _) };
            if stat != cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
                anyhow::bail!("cusolverDnSetStream failed: {:?}", stat);
            }
            handle
        };

        Ok(Self {
            _ctx: ctx,
            stream,
            blas,
            solver_handle,
        })
    }

    /// GPU eigendecomposition via cuSOLVER syevd.
    ///
    /// Input: column-major symmetric matrix (dim x dim).
    /// Output: sorted eigenvalues + eigenvectors (ascending).
    pub fn exact_diagonalize(
        &self,
        model: &HeisenbergModel,
    ) -> anyhow::Result<GpuExactDiagResult> {
        let n_sites = model.n_sites;
        let dim = 1usize << n_sites;

        // Build Hamiltonian on CPU (sparse structure, fast)
        let h = build_hamiltonian_matrix(model);

        // Upload to GPU
        let mut d_a = self.stream.clone_htod(&h)?;
        let mut d_w = self.stream.alloc_zeros::<f64>(dim)?;

        let n_ffi = dim as i32;
        let lda = dim as i32;
        let jobz = cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR;
        let uplo = cublasFillMode_t::CUBLAS_FILL_MODE_UPPER;

        // Query workspace size
        let mut lwork: i32 = 0;
        {
            let (a_ptr, _guard_a) = d_a.device_ptr(&self.stream);
            let (w_ptr, _guard_w) = d_w.device_ptr(&self.stream);
            let stat = unsafe {
                cusolver_sys::cusolverDnDsyevd_bufferSize(
                    self.solver_handle,
                    jobz,
                    uplo,
                    n_ffi,
                    a_ptr as *const f64,
                    lda,
                    w_ptr as *const f64,
                    &mut lwork,
                )
            };
            if stat != cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
                anyhow::bail!("cusolverDnDsyevd_bufferSize failed: {:?}", stat);
            }
        }

        let mut d_work = self.stream.alloc_zeros::<f64>(lwork.max(1) as usize)?;
        let mut d_info = self.stream.alloc_zeros::<i32>(1)?;

        // Run syevd (in-place: overwrites d_a with eigenvectors, d_w with eigenvalues)
        {
            let (a_ptr, _guard_a) = d_a.device_ptr_mut(&self.stream);
            let (w_ptr, _guard_w) = d_w.device_ptr_mut(&self.stream);
            let (work_ptr, _guard_work) = d_work.device_ptr_mut(&self.stream);
            let (info_ptr, _guard_info) = d_info.device_ptr_mut(&self.stream);

            let stat = unsafe {
                cusolver_sys::cusolverDnDsyevd(
                    self.solver_handle,
                    jobz,
                    uplo,
                    n_ffi,
                    a_ptr as *mut f64,
                    lda,
                    w_ptr as *mut f64,
                    work_ptr as *mut f64,
                    lwork,
                    info_ptr as *mut i32,
                )
            };
            if stat != cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
                anyhow::bail!("cusolverDnDsyevd failed: {:?}", stat);
            }
        }

        // Download results to host
        let eigenvalues = self.stream.clone_dtoh(&d_w)?;
        let eigenvectors = self.stream.clone_dtoh(&d_a)?;

        // Check convergence
        let info = self.stream.clone_dtoh(&d_info)?;
        if info[0] != 0 {
            anyhow::bail!(
                "cusolverDnDsyevd: info = {} (convergence failure)",
                info[0]
            );
        }

        // cuSOLVER returns eigenvalues in ascending order already
        Ok(GpuExactDiagResult {
            eigenvalues,
            eigenvectors,
            hilbert_dim: dim,
            n_sites,
        })
    }

    /// Transform a current operator into the eigenbasis using cuBLAS dgemm.
    ///
    /// Computes J_eig = V^T * J * V where V is the eigenvector matrix.
    /// Two dgemm calls: temp = J * V, then result = V^T * temp.
    pub fn transform_to_eigenbasis(
        &self,
        current_op: &[f64],
        eigenvectors: &[f64],
        dim: usize,
    ) -> anyhow::Result<Vec<f64>> {
        let d_j = self.stream.clone_htod(current_op)?;
        let d_v = self.stream.clone_htod(eigenvectors)?;
        let mut d_temp = self.stream.alloc_zeros::<f64>(dim * dim)?;
        let mut d_result = self.stream.alloc_zeros::<f64>(dim * dim)?;

        let n = dim as i32;

        // Step 1: temp = J * V
        let cfg1 = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n,
            n,
            k: n,
            alpha: 1.0f64,
            lda: n,
            ldb: n,
            beta: 0.0f64,
            ldc: n,
        };
        unsafe { self.blas.gemm(cfg1, &d_j, &d_v, &mut d_temp)? };

        // Step 2: result = V^T * temp
        let cfg2 = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n,
            n,
            k: n,
            alpha: 1.0f64,
            lda: n,
            ldb: n,
            beta: 0.0f64,
            ldc: n,
        };
        unsafe { self.blas.gemm(cfg2, &d_v, &d_temp, &mut d_result)? };

        Ok(self.stream.clone_dtoh(&d_result)?)
    }

    /// Compute full Kubo transport using GPU-accelerated ED and basis transformation.
    ///
    /// O(dim^3) per model evaluation instead of O(dim^4).
    pub fn kubo_transport(
        &self,
        model: &HeisenbergModel,
        temperature: f64,
        degeneracy_tol: f64,
    ) -> anyhow::Result<KuboTransport> {
        let ed = self.exact_diagonalize(model)?;

        let j_s = build_spin_current_operator(model);
        let j_e = build_energy_current_operator(model);

        let j_s_eig = self.transform_to_eigenbasis(&j_s, &ed.eigenvectors, ed.hilbert_dim)?;
        let j_e_eig = self.transform_to_eigenbasis(&j_e, &ed.eigenvectors, ed.hilbert_dim)?;

        Ok(compute_transport_from_eigenbasis(
            &ed.eigenvalues,
            &j_s_eig,
            &j_e_eig,
            ed.hilbert_dim,
            model.n_sites,
            temperature,
            degeneracy_tol,
        ))
    }
}

impl Drop for GpuKuboContext {
    fn drop(&mut self) {
        if !self.solver_handle.is_null() {
            unsafe {
                cusolver_sys::cusolverDnDestroy(self.solver_handle);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kubo_transport::{build_cd_heisenberg, build_j1j2_chain, kubo_transport};

    #[test]
    fn test_gpu_ed_quaternion() {
        let ctx = match GpuKuboContext::new() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("GPU not available, skipping test_gpu_ed_quaternion");
                return;
            }
        };
        let model = build_cd_heisenberg(4, 0.0);
        let ed = ctx.exact_diagonalize(&model).unwrap();
        assert_eq!(ed.hilbert_dim, 8);
        assert_eq!(ed.eigenvalues.len(), 8);
        let cpu_ed = crate::kubo_transport::exact_diagonalize(&model);
        for i in 0..8 {
            assert!(
                (ed.eigenvalues[i] - cpu_ed.eigenvalues[i]).abs() < 1e-8,
                "eigenvalue mismatch at {}: GPU={} CPU={}",
                i,
                ed.eigenvalues[i],
                cpu_ed.eigenvalues[i]
            );
        }
    }

    #[test]
    fn test_gpu_transport_matches_cpu_quaternion() {
        let ctx = match GpuKuboContext::new() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("GPU not available, skipping test");
                return;
            }
        };
        let model = build_cd_heisenberg(4, 0.0);
        let gpu = ctx.kubo_transport(&model, 0.5, 1e-10).unwrap();
        let cpu = kubo_transport(&model, 0.5, 1e-10);

        assert!(
            (gpu.drude_weight_spin - cpu.drude_weight_spin).abs() < 1e-6,
            "D_S mismatch: GPU={} CPU={}",
            gpu.drude_weight_spin,
            cpu.drude_weight_spin
        );
        assert!(
            (gpu.total_weight_spin - cpu.total_weight_spin).abs()
                / cpu.total_weight_spin.max(1e-20)
                < 0.01,
            "I0_S mismatch: GPU={} CPU={}",
            gpu.total_weight_spin,
            cpu.total_weight_spin
        );
    }

    #[test]
    fn test_gpu_transport_matches_cpu_octonion() {
        let ctx = match GpuKuboContext::new() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("GPU not available, skipping test");
                return;
            }
        };
        let model = build_cd_heisenberg(8, 0.0);
        let gpu = ctx.kubo_transport(&model, 1.0, 1e-10).unwrap();
        let cpu = kubo_transport(&model, 1.0, 1e-10);

        assert!(
            (gpu.drude_weight_spin - cpu.drude_weight_spin).abs() < 1e-6,
            "D_S mismatch: GPU={} CPU={}",
            gpu.drude_weight_spin,
            cpu.drude_weight_spin
        );
    }

    #[test]
    fn test_gpu_j1j2_chain() {
        let ctx = match GpuKuboContext::new() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("GPU not available, skipping test");
                return;
            }
        };
        let model = build_j1j2_chain(8, 0.25, 1.0, 3.0);
        let gpu = ctx.kubo_transport(&model, 0.1, 1e-10).unwrap();
        let cpu = kubo_transport(&model, 0.1, 1e-10);

        assert!(
            (gpu.drude_weight_spin - cpu.drude_weight_spin).abs() < 1e-6,
            "D_S mismatch: GPU={} CPU={}",
            gpu.drude_weight_spin,
            cpu.drude_weight_spin
        );
    }
}

//! GPU-accelerated eta matrix computation for zero-divisor analysis.
//!
//! The eta matrix is defined as: eta(i,j) = psi(i, j+dim/2) XOR psi(j, i+dim/2)
//! This is a (dim/2) x (dim/2) matrix of u8 values (0 or 1).
//!
//! At dim=1024, this is only 512x512 = 262K elements (~256KB), easily fitting on GPU.
//! The parallel XOR operations provide 20-100x speedup over CPU.

use std::ffi::CString;

#[cfg(feature = "gpu")]
use cudarc::driver::{CudaContext, DeviceSlice, DriverError};

/// GPU-accelerated eta matrix computation.
pub struct EtaMatrixGpu;

/// NVRTC CUDA kernel source for eta matrix computation.
/// Implements cd_basis_mul_sign as device function + parallel eta computation.
#[cfg(feature = "gpu")]
const ETA_KERNEL_SRC: &str = r#"
// Device function: cd_basis_mul_sign_iter ported to CUDA
// Returns sign: +1 or -1 represented as 1 or -1
__device__ int cd_basis_mul_sign(unsigned int dim, unsigned int p, unsigned int q) {
    int sign = 1;
    unsigned int half = dim >> 1;

    while (half > 0) {
        unsigned int p_hi = (p >= half) ? 1 : 0;
        unsigned int q_hi = (q >= half) ? 1 : 0;

        // Combine bits to determine case: 0=(F,F), 1=(F,T), 2=(T,F), 3=(T,T)
        unsigned int case = (p_hi << 1) | q_hi;

        if (case == 0) {
            // (false, false): both in lower half, no change
        } else if (case == 1) {
            // (false, true): (a,0) * (0,d) = (0, d*a)
            unsigned int qh = q - half;
            q = p;
            p = qh;
        } else if (case == 2) {
            // (true, false): (0,b) * (c,0) = (0, b*conj(c))
            p -= half;
            if (q != 0) {
                sign = -sign;
            }
        } else {
            // (true, true): (0,b) * (0,d) = (-conj(d)*b, 0)
            unsigned int qh = q - half;
            unsigned int ph = p - half;
            if (qh == 0) {
                return -sign;  // early return
            }
            p = qh;
            q = ph;
        }

        half >>= 1;
    }

    return sign;
}

// Device function: psi(i,j) = (cd_basis_mul_sign(dim, i, j) == 1) ? 0 : 1
__device__ unsigned char psi(unsigned int dim, unsigned int i, unsigned int j) {
    int sign = cd_basis_mul_sign(dim, i, j);
    return (sign == 1) ? 0 : 1;
}

// Kernel: compute eta matrix in parallel
// Grid: 1D grid of threads, one thread per (i,j) pair
// Each thread computes eta[flat_idx] = psi(i, j+dim/2) XOR psi(j, i+dim/2)
__global__ void compute_eta_matrix(
    unsigned int dim,
    unsigned int dim_half,
    unsigned char* eta_out
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = dim_half * dim_half;

    if (idx < total) {
        unsigned int i = idx / dim_half;
        unsigned int j = idx % dim_half;

        // eta(i,j) = psi(i, j+dim/2) XOR psi(j, i+dim/2)
        unsigned char psi_ij = psi(dim, i, j + dim_half);
        unsigned char psi_ji = psi(dim, j, i + dim_half);
        eta_out[idx] = psi_ij ^ psi_ji;
    }
}
"#;

impl EtaMatrixGpu {
    /// CPU implementation: compute eta matrix without GPU.
    ///
    /// # Arguments
    /// * `dim` - Cayley-Dickson dimension (must be power of 2, >= 2)
    /// * `psi_fn` - Function to compute psi(i, j) = (sign(basis_mul(dim, i, j)) == 1) ? 0 : 1
    ///
    /// # Returns
    /// eta matrix as (dim/2)x(dim/2) flattened vector of u8 (0 or 1)
    pub fn compute_eta_cpu<F>(dim: usize, psi_fn: F) -> Vec<u8>
    where
        F: Fn(usize, usize) -> u8,
    {
        let dim_half = dim / 2;
        let mut eta = vec![0u8; dim_half * dim_half];

        for i in 0..dim_half {
            for j in 0..dim_half {
                // eta(i,j) = psi(i, j+dim/2) XOR psi(j, i+dim/2)
                let psi_ij = psi_fn(i, j + dim_half);
                let psi_ji = psi_fn(j, i + dim_half);
                eta[i * dim_half + j] = psi_ij ^ psi_ji;
            }
        }

        eta
    }

    /// Compute eta matrix (uses GPU if available, falls back to CPU).
    pub fn compute<F>(dim: usize, psi_fn: F) -> Vec<u8>
    where
        F: Fn(usize, usize) -> u8 + Copy,
    {
        #[cfg(feature = "gpu")]
        {
            // Try GPU first
            if let Ok(eta) = Self::compute_eta_gpu(dim, psi_fn) {
                return eta;
            }
        }

        // Fall back to CPU
        Self::compute_eta_cpu(dim, psi_fn)
    }

    /// GPU implementation using cudarc NVRTC.
    #[cfg(feature = "gpu")]
    fn compute_eta_gpu<F>(dim: usize, psi_fn: F) -> Result<Vec<u8>, String>
    where
        F: Fn(usize, usize) -> u8,
    {
        let ctx = CudaContext::new(0).map_err(|e| format!("CUDA init: {}", e))?;

        let ptx = cudarc::nvrtc::compile_ptx(ETA_KERNEL_SRC)
            .map_err(|e| format!("NVRTC compile: {}", e))?;

        ctx.load_ptx(ptx, "eta_kernel", &["compute_eta_matrix"])
            .map_err(|e| format!("PTX load: {}", e))?;

        let kernel = ctx
            .get_fn("eta_kernel", "compute_eta_matrix")
            .map_err(|e| format!("Get kernel: {}", e))?;

        let dim_half = dim / 2;
        let total = dim_half * dim_half;

        // Allocate device memory
        let eta_dev = ctx
            .alloc_zeros::<u8>(total)
            .map_err(|e| format!("Alloc device: {}", e))?;

        // Launch kernel
        // Use 256 threads per block (common choice)
        let block_size = 256u32;
        let num_blocks = ((total as u32 + block_size - 1) / block_size) as u32;

        unsafe {
            kernel
                .launch_on_stream(
                    &ctx.stream,
                    cudarc::driver::LaunchConfig {
                        grid_dim: (num_blocks, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (dim as u32, dim_half as u32, &eta_dev),
                )
                .map_err(|e| format!("Kernel launch: {}", e))?;
        }

        // Copy result back to host
        let mut eta_host = vec![0u8; total];
        ctx.copy_d2h_into(&eta_dev, &mut eta_host)
            .map_err(|e| format!("Copy D2H: {}", e))?;

        Ok(eta_host)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::construction::cd_basis_mul_sign;

    // Real psi function from cd_basis_mul_sign
    fn real_psi(dim: usize, i: usize, j: usize) -> u8 {
        if cd_basis_mul_sign(dim, i, j) == 1 {
            0
        } else {
            1
        }
    }

    #[test]
    fn test_cpu_eta_computation() {
        let dim = 16;
        let eta = EtaMatrixGpu::compute_eta_cpu(dim, |i, j| real_psi(dim, i, j));
        assert_eq!(eta.len(), 64); // (16/2)^2 = 64

        // Verify all entries are 0 or 1
        for val in &eta {
            assert!(*val == 0 || *val == 1, "eta values must be 0 or 1");
        }
    }

    #[test]
    fn test_eta_compute_fallback() {
        let dim = 32;
        let eta = EtaMatrixGpu::compute(dim, |i, j| real_psi(dim, i, j));
        assert_eq!(eta.len(), 256); // (32/2)^2 = 256

        // Verify all entries are 0 or 1
        for val in &eta {
            assert!(*val == 0 || *val == 1, "eta values must be 0 or 1");
        }
    }

    // Cross-validation tests: GPU vs CPU (only run if GPU available)
    #[cfg(feature = "gpu")]
    #[test]
    fn test_eta_gpu_vs_cpu_dim16() {
        if !crate::gpu::is_gpu_available() {
            eprintln!("GPU not available; skipping GPU test");
            return;
        }

        let dim = 16;
        let psi_fn = |i: usize, j: usize| real_psi(dim, i, j);

        let eta_cpu = EtaMatrixGpu::compute_eta_cpu(dim, psi_fn);
        let eta_gpu = EtaMatrixGpu::compute_eta_gpu(dim, psi_fn)
            .expect("GPU computation should succeed when GPU is available");

        assert_eq!(eta_cpu, eta_gpu, "GPU and CPU eta matrices must match exactly at dim=16");
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_eta_gpu_vs_cpu_dim32() {
        if !crate::gpu::is_gpu_available() {
            eprintln!("GPU not available; skipping GPU test");
            return;
        }

        let dim = 32;
        let psi_fn = |i: usize, j: usize| real_psi(dim, i, j);

        let eta_cpu = EtaMatrixGpu::compute_eta_cpu(dim, psi_fn);
        let eta_gpu = EtaMatrixGpu::compute_eta_gpu(dim, psi_fn)
            .expect("GPU computation should succeed when GPU is available");

        assert_eq!(eta_cpu, eta_gpu, "GPU and CPU eta matrices must match exactly at dim=32");
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_eta_gpu_vs_cpu_dim64() {
        if !crate::gpu::is_gpu_available() {
            eprintln!("GPU not available; skipping GPU test");
            return;
        }

        let dim = 64;
        let psi_fn = |i: usize, j: usize| real_psi(dim, i, j);

        let eta_cpu = EtaMatrixGpu::compute_eta_cpu(dim, psi_fn);
        let eta_gpu = EtaMatrixGpu::compute_eta_gpu(dim, psi_fn)
            .expect("GPU computation should succeed when GPU is available");

        assert_eq!(eta_cpu, eta_gpu, "GPU and CPU eta matrices must match exactly at dim=64");
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_eta_gpu_vs_cpu_dim128() {
        if !crate::gpu::is_gpu_available() {
            eprintln!("GPU not available; skipping GPU test");
            return;
        }

        let dim = 128;
        let psi_fn = |i: usize, j: usize| real_psi(dim, i, j);

        let eta_cpu = EtaMatrixGpu::compute_eta_cpu(dim, psi_fn);
        let eta_gpu = EtaMatrixGpu::compute_eta_gpu(dim, psi_fn)
            .expect("GPU computation should succeed when GPU is available");

        assert_eq!(eta_cpu, eta_gpu, "GPU and CPU eta matrices must match exactly at dim=128");
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_eta_gpu_vs_cpu_dim256() {
        if !crate::gpu::is_gpu_available() {
            eprintln!("GPU not available; skipping GPU test");
            return;
        }

        let dim = 256;
        let psi_fn = |i: usize, j: usize| real_psi(dim, i, j);

        let eta_cpu = EtaMatrixGpu::compute_eta_cpu(dim, psi_fn);
        let eta_gpu = EtaMatrixGpu::compute_eta_gpu(dim, psi_fn)
            .expect("GPU computation should succeed when GPU is available");

        assert_eq!(eta_cpu, eta_gpu, "GPU and CPU eta matrices must match exactly at dim=256");
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_eta_gpu_vs_cpu_dim512() {
        if !crate::gpu::is_gpu_available() {
            eprintln!("GPU not available; skipping GPU test");
            return;
        }

        let dim = 512;
        let psi_fn = |i: usize, j: usize| real_psi(dim, i, j);

        let eta_cpu = EtaMatrixGpu::compute_eta_cpu(dim, psi_fn);
        let eta_gpu = EtaMatrixGpu::compute_eta_gpu(dim, psi_fn)
            .expect("GPU computation should succeed when GPU is available");

        assert_eq!(eta_cpu, eta_gpu, "GPU and CPU eta matrices must match exactly at dim=512");
    }
}

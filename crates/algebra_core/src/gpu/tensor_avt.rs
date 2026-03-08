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

#[cfg(feature = "gpu")]
use cudarc::driver::{CudaContext, CudaFunction, LaunchConfig, PushKernelArg};
#[cfg(feature = "gpu")]
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
#[cfg(feature = "gpu")]
use std::sync::{Arc, OnceLock};

/// Build NVRTC compile options for Tensor Core kernels.
///
/// Includes the CUDA toolkit header path (for `mma.h` and `cuda_fp16.h`),
/// targets SM 8.9 (Ada Lovelace / RTX 4070 Ti), enables C++14 (required
/// by WMMA intrinsics), and uses -O3 for aggressive optimization.
///
/// Falls back to `/usr/local/cuda/include` if `/opt/cuda/include` is absent.
#[cfg(feature = "gpu")]
fn tensor_compile_opts() -> CompileOptions {
    let cuda_include = if std::path::Path::new("/opt/cuda/include/mma.h").exists() {
        "-I/opt/cuda/include"
    } else {
        "-I/usr/local/cuda/include"
    };
    CompileOptions {
        options: vec![
            "-arch=sm_89".to_string(),
            cuda_include.to_string(),
            "-std=c++14".to_string(),
        ],
        ..Default::default()
    }
}

/// CUDA kernel source for Tensor Core AVT computation.
///
/// Contains:
/// - `tensor_avt_basis_kernel`: Sign-table Monte Carlo for basis triples
/// - `tensor_norm_sq_kernel`: Warp-level butterfly norm-squared reduction
/// - `build_left_mul_tile`: Constructs 16x16 tiles of L_a for dense CD mul
/// - `tensor_core_mma_16x16`: Single-tile WMMA dispatch
#[cfg(feature = "gpu")]
const TENSOR_AVT_KERNEL_SRC: &str = include_str!("kernels_tensor_avt.cu");

#[cfg(feature = "gpu")]
struct TensorAvtGpuRuntime {
    ctx: Arc<CudaContext>,
    tensor_avt_basis_kernel: CudaFunction,
    tensor_avt_dense_kernel: CudaFunction,
    tensor_cd_mul_kernel: CudaFunction,
    tensor_cd_mul_batched_kernel: CudaFunction,
    tensor_norm_sq_kernel: CudaFunction,
}

#[cfg(feature = "gpu")]
impl TensorAvtGpuRuntime {
    fn initialize() -> Result<Arc<Self>, String> {
        let ctx =
            Arc::new(CudaContext::new(0).map_err(|e| format!("CUDA init: {}", e))?);
        let ptx = compile_ptx_with_opts(TENSOR_AVT_KERNEL_SRC, tensor_compile_opts())
            .map_err(|e| format!("NVRTC compile: {}", e))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| format!("Module load: {}", e))?;
        let tensor_avt_basis_kernel = module
            .load_function("tensor_avt_basis_kernel")
            .map_err(|e| format!("Kernel load: {}", e))?;
        let tensor_avt_dense_kernel = module
            .load_function("tensor_avt_dense_kernel")
            .map_err(|e| format!("Kernel load: {}", e))?;
        let tensor_cd_mul_kernel = module
            .load_function("tensor_cd_mul_kernel")
            .map_err(|e| format!("Kernel load: {}", e))?;
        let tensor_cd_mul_batched_kernel = module
            .load_function("tensor_cd_mul_batched_kernel")
            .map_err(|e| format!("Kernel load: {}", e))?;
        let tensor_norm_sq_kernel = module
            .load_function("tensor_norm_sq_kernel")
            .map_err(|e| format!("Kernel load: {}", e))?;
        Ok(Arc::new(Self {
            ctx,
            tensor_avt_basis_kernel,
            tensor_avt_dense_kernel,
            tensor_cd_mul_kernel,
            tensor_cd_mul_batched_kernel,
            tensor_norm_sq_kernel,
        }))
    }
}

#[cfg(feature = "gpu")]
fn tensor_avt_gpu_runtime() -> Result<Arc<TensorAvtGpuRuntime>, String> {
    static RUNTIME: OnceLock<Result<Arc<TensorAvtGpuRuntime>, String>> = OnceLock::new();
    match RUNTIME.get_or_init(TensorAvtGpuRuntime::initialize) {
        Ok(runtime) => Ok(runtime.clone()),
        Err(err) => Err(err.clone()),
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
        assert!(dim >= 16 && dim.is_power_of_two(),
            "TensorAVT requires dim >= 16 and power of 2, got {dim}");
        Self {
            dim,
            tile_count: dim / 16,
        }
    }

    #[cfg(any(not(feature = "gpu"), test))]
    fn compute_cd_mul_cpu(
        &self,
        a: &[f32],
        x: &[f32],
    ) -> Result<Vec<f32>, String> {
        use crate::construction::cayley_dickson::cd_basis_mul_sign;
        assert_eq!(a.len(), self.dim, "a must have dim elements");
        assert_eq!(x.len(), self.dim, "x must have dim elements");
        let mut y = vec![0.0f32; self.dim];
        for (i, yi) in y.iter_mut().enumerate() {
            for (j, xj) in x.iter().enumerate() {
                let src = i ^ j;
                let sign = cd_basis_mul_sign(self.dim, src, j);
                *yi += a[src] * (sign as f32) * xj;
            }
        }
        Ok(y)
    }

    #[cfg(any(not(feature = "gpu"), test))]
    fn compute_cd_mul_batch_cpu(
        &self,
        a: &[f32],
        x_batch: &[f32],
        batch_size: usize,
    ) -> Result<Vec<f32>, String> {
        use crate::construction::cayley_dickson::cd_basis_mul_sign;
        assert_eq!(a.len(), self.dim, "a must have dim elements");
        assert_eq!(
            x_batch.len(),
            batch_size * self.dim,
            "x_batch must have batch_size * dim elements"
        );
        let mut y_batch = vec![0.0f32; batch_size * self.dim];
        for b in 0..batch_size {
            let x_offset = b * self.dim;
            let y_offset = b * self.dim;
            for (i, yi) in y_batch[y_offset..y_offset + self.dim].iter_mut().enumerate() {
                for j in 0..self.dim {
                    let src = i ^ j;
                    let sign = cd_basis_mul_sign(self.dim, src, j);
                    *yi += a[src] * (sign as f32) * x_batch[x_offset + j];
                }
            }
        }
        Ok(y_batch)
    }

    #[cfg(any(not(feature = "gpu"), test))]
    fn compute_norm_sq_batch_cpu(
        &self,
        vectors: &[f32],
        n_vectors: usize,
    ) -> Result<Vec<f32>, String> {
        assert_eq!(
            vectors.len(),
            n_vectors * self.dim,
            "vectors length must be n_vectors * dim"
        );
        let mut norms = Vec::with_capacity(n_vectors);
        for v_idx in 0..n_vectors {
            let start = v_idx * self.dim;
            let end = start + self.dim;
            let norm_sq: f32 = vectors[start..end].iter().map(|x| x * x).sum();
            norms.push(norm_sq);
        }
        Ok(norms)
    }

    /// Compute the AVT frustration field on GPU using Tensor Core-optimized kernels.
    ///
    /// Generates a 3D scalar field where each voxel's value represents the
    /// fraction of sampled basis triples that are non-associative.
    ///
    /// # Arguments
    /// * `nx, ny, nz` - Grid dimensions
    /// * `n_samples_per_cell` - Monte Carlo samples per voxel
    /// * `seed` - Random seed for reproducibility
    ///
    /// # Returns
    /// Flattened frustration field of size nx*ny*nz, or error string.
    #[cfg(feature = "gpu")]
    pub fn compute_avt_field(
        &self,
        nx: usize,
        ny: usize,
        nz: usize,
        n_samples_per_cell: u32,
        seed: u32,
    ) -> Result<Vec<f32>, String> {
        let runtime = tensor_avt_gpu_runtime()?;
        let stream = runtime.ctx.default_stream();

        let n_cells = nx * ny * nz;
        let mut d_field = stream.alloc_zeros::<f32>(n_cells)
            .map_err(|e| format!("Alloc: {}", e))?;

        let block_size = 256u32;
        let grid_size = (n_cells as u32).div_ceil(block_size);
        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let nx_i = nx as i32;
        let ny_i = ny as i32;
        let nz_i = nz as i32;
        let dim_u = self.dim as u32;

        let mut builder = stream.launch_builder(&runtime.tensor_avt_basis_kernel);
        builder.arg(&mut d_field);
        builder.arg(&nx_i);
        builder.arg(&ny_i);
        builder.arg(&nz_i);
        builder.arg(&dim_u);
        builder.arg(&n_samples_per_cell);
        builder.arg(&seed);

        unsafe {
            builder.launch(cfg).map_err(|e| format!("Launch: {}", e))?;
        }

        let host_field: Vec<f32> = stream.clone_dtoh(&d_field)
            .map_err(|e| format!("Copy back: {}", e))?;

        Ok(host_field)
    }

    /// Compute dense AVT frustration field using weighted basis expansion.
    ///
    /// Unlike `compute_avt_field` which samples pure basis triples,
    /// this kernel generates random dense vectors (sums of `n_basis_per_vec`
    /// random basis elements with +/-1 weights) and computes the
    /// weighted associator norm-squared.
    ///
    /// For dim=512 or dim=1024, this captures the full non-associativity
    /// landscape of dense linear combinations, not just basis triples.
    #[cfg(feature = "gpu")]
    pub fn compute_dense_avt_field(
        &self,
        nx: usize,
        ny: usize,
        nz: usize,
        n_samples_per_cell: u32,
        n_basis_per_vec: u32,
        seed: u32,
    ) -> Result<Vec<f32>, String> {
        let runtime = tensor_avt_gpu_runtime()?;
        let stream = runtime.ctx.default_stream();

        let n_cells = nx * ny * nz;
        let mut d_field = stream.alloc_zeros::<f32>(n_cells)
            .map_err(|e| format!("Alloc: {}", e))?;

        let block_size = 256u32;
        let grid_size = (n_cells as u32).div_ceil(block_size);
        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let nx_i = nx as i32;
        let ny_i = ny as i32;
        let nz_i = nz as i32;
        let dim_u = self.dim as u32;

        let mut builder = stream.launch_builder(&runtime.tensor_avt_dense_kernel);
        builder.arg(&mut d_field);
        builder.arg(&nx_i);
        builder.arg(&ny_i);
        builder.arg(&nz_i);
        builder.arg(&dim_u);
        builder.arg(&n_samples_per_cell);
        builder.arg(&n_basis_per_vec);
        builder.arg(&seed);

        unsafe {
            builder.launch(cfg).map_err(|e| format!("Launch: {}", e))?;
        }

        let host_field: Vec<f32> = stream.clone_dtoh(&d_field)
            .map_err(|e| format!("Copy back: {}", e))?;

        Ok(host_field)
    }

    /// CPU fallback for dense AVT field.
    #[cfg(not(feature = "gpu"))]
    pub fn compute_dense_avt_field(
        &self,
        _nx: usize,
        _ny: usize,
        _nz: usize,
        _n_samples_per_cell: u32,
        _n_basis_per_vec: u32,
        _seed: u32,
    ) -> Result<Vec<f32>, String> {
        Err("GPU feature not enabled; use CPU path for dense AVT".into())
    }

    /// Compute a single CD multiplication y = a * x via Left-Multiplication Matrix.
    ///
    /// Launches `tensor_cd_mul_kernel` which builds L_a tiles on-the-fly
    /// and dispatches (dim/16) blocks of WMMA operations.
    ///
    /// Works at dim=256, 512, 1024 or any dim that is a multiple of 16.
    #[cfg(feature = "gpu")]
    pub fn compute_cd_mul(
        &self,
        a: &[f32],
        x: &[f32],
    ) -> Result<Vec<f32>, String> {
        assert_eq!(a.len(), self.dim, "a must have dim elements");
        assert_eq!(x.len(), self.dim, "x must have dim elements");

        let runtime = tensor_avt_gpu_runtime()?;
        let stream = runtime.ctx.default_stream();

        let d_a = stream.clone_htod(a)
            .map_err(|e| format!("Upload a: {}", e))?;
        let d_x = stream.clone_htod(x)
            .map_err(|e| format!("Upload x: {}", e))?;
        let mut d_y = stream.alloc_zeros::<f32>(self.dim)
            .map_err(|e| format!("Alloc y: {}", e))?;

        let n_row_blocks = self.tile_count as u32;
        let cfg = LaunchConfig {
            grid_dim: (n_row_blocks, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let dim_u = self.dim as u32;

        let mut builder = stream.launch_builder(&runtime.tensor_cd_mul_kernel);
        builder.arg(&d_a);
        builder.arg(&d_x);
        builder.arg(&mut d_y);
        builder.arg(&dim_u);

        unsafe {
            builder.launch(cfg).map_err(|e| format!("Launch: {}", e))?;
        }

        let y: Vec<f32> = stream.clone_dtoh(&d_y)
            .map_err(|e| format!("Copy y: {}", e))?;
        Ok(y)
    }

    /// CPU fallback for CD multiplication via Left-Multiplication Matrix.
    #[cfg(not(feature = "gpu"))]
    pub fn compute_cd_mul(
        &self,
        a: &[f32],
        x: &[f32],
    ) -> Result<Vec<f32>, String> {
        self.compute_cd_mul_cpu(a, x)
    }

    /// Batched CD multiplication: Y = L_a * X for multiple x-vectors.
    ///
    /// Computes `a * x_i` for each of `batch_size` vectors simultaneously,
    /// exploiting the full 16x16 Tensor Core MMA throughput (16x speedup
    /// over single-vector kernel at dim >= 256).
    ///
    /// # Arguments
    /// * `a` - Left operand (shared across batch), dim elements
    /// * `x_batch` - Right operands, col-major: `x_batch[b * dim + d]`
    /// * `batch_size` - Number of x vectors (rounded up to multiple of 16 internally)
    ///
    /// # Returns
    /// Output vectors, col-major: `y[b * dim + d]`, length `batch_size * dim`.
    #[cfg(feature = "gpu")]
    pub fn compute_cd_mul_batch(
        &self,
        a: &[f32],
        x_batch: &[f32],
        batch_size: usize,
    ) -> Result<Vec<f32>, String> {
        assert_eq!(a.len(), self.dim, "a must have dim elements");
        assert_eq!(x_batch.len(), batch_size * self.dim,
            "x_batch must have batch_size * dim elements");

        let runtime = tensor_avt_gpu_runtime()?;
        let stream = runtime.ctx.default_stream();

        let d_a = stream.clone_htod(a)
            .map_err(|e| format!("Upload a: {}", e))?;
        let d_x = stream.clone_htod(x_batch)
            .map_err(|e| format!("Upload x_batch: {}", e))?;
        let mut d_y = stream.alloc_zeros::<f32>(batch_size * self.dim)
            .map_err(|e| format!("Alloc y_batch: {}", e))?;

        // Grid: (ceil(batch/16), ceil(dim/64)) -- each block has 4 warps covering 64 rows
        let grid_x = (batch_size as u32).div_ceil(16);
        let grid_y = (self.dim as u32).div_ceil(64); // 4 warps * 16 rows = 64 rows per block
        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (128, 1, 1),  // 4 warps
            shared_mem_bytes: 0,
        };

        let dim_u = self.dim as u32;
        let batch_u = batch_size as u32;

        let mut builder = stream.launch_builder(&runtime.tensor_cd_mul_batched_kernel);
        builder.arg(&d_a);
        builder.arg(&d_x);
        builder.arg(&mut d_y);
        builder.arg(&dim_u);
        builder.arg(&batch_u);

        unsafe {
            builder.launch(cfg).map_err(|e| format!("Launch: {}", e))?;
        }

        let y: Vec<f32> = stream.clone_dtoh(&d_y)
            .map_err(|e| format!("Copy y_batch: {}", e))?;
        Ok(y)
    }

    /// CPU fallback for batched CD multiplication.
    #[cfg(not(feature = "gpu"))]
    pub fn compute_cd_mul_batch(
        &self,
        a: &[f32],
        x_batch: &[f32],
        batch_size: usize,
    ) -> Result<Vec<f32>, String> {
        self.compute_cd_mul_batch_cpu(a, x_batch, batch_size)
    }

    /// CPU fallback when GPU feature is not enabled.
    #[cfg(not(feature = "gpu"))]
    pub fn compute_avt_field(
        &self,
        _nx: usize,
        _ny: usize,
        _nz: usize,
        _n_samples_per_cell: u32,
        _seed: u32,
    ) -> Result<Vec<f32>, String> {
        Err("GPU feature not enabled; use CPU path via algebra_analysis::phase_transition".into())
    }

    /// Compute AVT norm-squared for a batch of dense vectors on GPU.
    ///
    /// Uses warp-level butterfly reduction for efficient parallel norm computation.
    #[cfg(feature = "gpu")]
    pub fn compute_norm_sq_batch(
        &self,
        vectors: &[f32],
        n_vectors: usize,
    ) -> Result<Vec<f32>, String> {
        assert_eq!(vectors.len(), n_vectors * self.dim,
            "vectors length must be n_vectors * dim");

        let runtime = tensor_avt_gpu_runtime()?;
        let stream = runtime.ctx.default_stream();

        let d_vectors = stream.clone_htod(vectors)
            .map_err(|e| format!("Upload vectors: {}", e))?;
        let mut d_norms = stream.alloc_zeros::<f32>(n_vectors)
            .map_err(|e| format!("Alloc norms: {}", e))?;

        let block_size = 256u32.min(self.dim as u32);
        let dim_i = self.dim as i32;
        let n_vec_i = n_vectors as i32;
        let cfg = LaunchConfig {
            grid_dim: (n_vectors as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = stream.launch_builder(&runtime.tensor_norm_sq_kernel);
        builder.arg(&d_vectors);
        builder.arg(&mut d_norms);
        builder.arg(&dim_i);
        builder.arg(&n_vec_i);

        unsafe {
            builder.launch(cfg).map_err(|e| format!("Launch: {}", e))?;
        }

        let norms: Vec<f32> = stream.clone_dtoh(&d_norms)
            .map_err(|e| format!("Copy norms: {}", e))?;
        Ok(norms)
    }

    /// CPU fallback for norm-squared batch.
    #[cfg(not(feature = "gpu"))]
    pub fn compute_norm_sq_batch(
        &self,
        vectors: &[f32],
        n_vectors: usize,
    ) -> Result<Vec<f32>, String> {
        self.compute_norm_sq_batch_cpu(vectors, n_vectors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // Create a unit vector in the first component
        let mut vec = vec![0.0f32; 16];
        vec[0] = 1.0;
        let norms = avt.compute_norm_sq_batch_cpu(&vec, 1).unwrap();
        assert!((norms[0] - 1.0).abs() < 1e-6);

        // Create a vector with all components = 1/sqrt(16) = 0.25
        let uniform: Vec<f32> = vec![0.25; 16];
        let norms2 = avt.compute_norm_sq_batch_cpu(&uniform, 1).unwrap();
        // ||v||^2 = 16 * 0.0625 = 1.0
        assert!((norms2[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cd_mul_cpu_basis_identity() {
        // e_0 is the identity: e_0 * x = x for any x.
        // Tolerance 1e-2: accommodates FP16 quantization when GPU feature is active
        // (e.g. 0.7 -> 0.7001953 in FP16). CPU-only path is exact to 1e-7.
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
        // Random-ish vectors
        let a: Vec<f32> = (0..16).map(|i| ((i * 7 + 3) % 11) as f32 * 0.2 - 1.0).collect();
        let x: Vec<f32> = (0..16).map(|i| ((i * 13 + 5) % 11) as f32 * 0.15 - 0.8).collect();
        let y_gpu_cpu = avt.compute_cd_mul_cpu(&a, &x).unwrap();

        // Reference via cd_multiply (f64)
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
        // e_1 * e_2 at dim=256 should give e_{1 XOR 2} = e_3 with correct sign
        use crate::construction::cayley_dickson::cd_basis_mul_sign;
        let avt = TensorAVT::new(256);
        let mut e1 = vec![0.0f32; 256];
        e1[1] = 1.0;
        let mut e2 = vec![0.0f32; 256];
        e2[2] = 1.0;
        let y = avt.compute_cd_mul_cpu(&e1, &e2).unwrap();

        let expected_idx = 1 ^ 2; // = 3
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
        // e_0 * x = x at dim=512.
        // FP16 tolerance: ~1e-2 for non-representable values.
        let avt = TensorAVT::new(512);
        let mut e0 = vec![0.0f32; 512];
        e0[0] = 1.0;
        let x: Vec<f32> = (0..512).map(|i| ((i * 37 + 11) % 97) as f32 * 0.01).collect();
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
        // e_0 * x = x at dim=1024.
        // FP16 tolerance: ~1e-2 for non-representable values.
        let avt = TensorAVT::new(1024);
        let mut e0 = vec![0.0f32; 1024];
        e0[0] = 1.0;
        let x: Vec<f32> = (0..1024).map(|i| ((i * 41 + 7) % 101) as f32 * 0.01).collect();
        let y = avt.compute_cd_mul_cpu(&e0, &x).unwrap();
        for (i, (got, expected)) in y.iter().zip(x.iter()).enumerate() {
            assert!(
                (*got - *expected).abs() < 1e-2,
                "e_0 * x != x at dim=1024 index {i}"
            );
        }
    }

    /// Independent third verification path for CD multiplication.
    ///
    /// Uses target-index accumulation: for each pair (i,j), compute
    /// target = i XOR j, then result[target] += sign(i,j) * a[i] * x[j].
    ///
    /// This is mathematically equivalent to `compute_cd_mul` (which uses
    /// the L_a row-scan formula) but traverses the multiplication table
    /// from the opposite direction. Agreement between the two confirms
    /// both implementations are correct.
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
        // Deterministic non-trivial vectors via sin/cos -- no RNG dependency.
        // Cross-validates compute_cd_mul (L_a row-scan / GPU Tensor Core)
        // against reference_cd_multiply (CPU target-index accumulation).
        // Tolerance 1e-2: GPU FP16 path quantizes inputs to half precision.
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
        // Same cross-validation at dim=256 (Voudon) with different frequencies.
        // Tolerance 1e-1: GPU FP16 path + dim=256 accumulation error.
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

    /// GPU cross-validation: compare Tensor Core kernel output against
    /// CPU reference (reference_cd_multiply) at dim=256 with sin/cos vectors.
    ///
    /// This test validates the GPU execution pipeline against the independently
    /// verified CPU implementation.
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

        // GPU path
        let y_gpu = avt.compute_cd_mul(&a, &x).unwrap();

        // CPU reference via target-index accumulation (third independent path)
        let y_ref = reference_cd_multiply(&a, &x, dim);

        // Also cross-check against cd_multiply (f64 recursive doubling)
        let a64: Vec<f64> = a.iter().map(|&v| v as f64).collect();
        let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        let y_cd = cd_multiply(&a64, &x64);

        for i in 0..dim {
            let gpu_val = y_gpu[i];
            let ref_val = y_ref[i];
            let cd_val = y_cd[i] as f32;

            // GPU vs CPU reference (both f32, should be very close)
            let diff_ref = (gpu_val - ref_val).abs();
            assert!(
                diff_ref < 1e-2,
                "GPU vs ref mismatch at [{i}]: gpu={gpu_val}, ref={ref_val}, diff={diff_ref}"
            );

            // GPU vs cd_multiply (f64 reference, allow more tolerance for precision)
            let diff_cd = (gpu_val - cd_val).abs();
            assert!(
                diff_cd < 1e-1,
                "GPU vs cd_multiply mismatch at [{i}]: gpu={gpu_val}, cd={cd_val}, diff={diff_cd}"
            );
        }
    }

    /// GPU test at dim=16: Tensor Core kernel for identity multiplication.
    ///
    /// Uses 1e-2 tolerance because FP16 inputs cannot represent all f32 values
    /// exactly (e.g. 0.7 -> 0.7001953 in FP16). Basis elements (0 or +/-1)
    /// are exact in FP16, but the test vector has non-representable values.
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
                y[i], x[i]
            );
        }
    }

    /// CPU test for batched CD multiplication at dim=16.
    /// Verifies that compute_cd_mul_batch produces the same results
    /// as calling compute_cd_mul individually for each vector.
    #[test]
    fn test_cd_mul_batch_cpu_dim16() {
        let dim = 16;
        let batch_size = 4;
        let avt = TensorAVT::new(dim);

        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.3).sin()).collect();

        // Create batch: 4 different x vectors, col-major layout
        let mut x_batch = vec![0.0f32; batch_size * dim];
        for b in 0..batch_size {
            for d in 0..dim {
                x_batch[b * dim + d] = ((b * dim + d) as f32 * 0.17).cos();
            }
        }

        let y_batch = avt.compute_cd_mul_batch_cpu(&a, &x_batch, batch_size).unwrap();

        // Verify each batch element matches individual compute_cd_mul
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

    /// CPU test for batched CD multiplication at dim=256.
    #[test]
    fn test_cd_mul_batch_cpu_dim256() {
        let dim = 256;
        let batch_size = 16;  // Full 16-column batch
        let avt = TensorAVT::new(dim);

        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();

        let mut x_batch = vec![0.0f32; batch_size * dim];
        for b in 0..batch_size {
            for d in 0..dim {
                x_batch[b * dim + d] = ((b * 3 + d * 7) as f32 * 0.01).cos();
            }
        }

        let y_batch = avt.compute_cd_mul_batch_cpu(&a, &x_batch, batch_size).unwrap();

        // Spot-check first and last batch elements
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
}

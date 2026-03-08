#[cfg(feature = "gpu")]
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
#[cfg(feature = "gpu")]
use cudarc::nvrtc::{CompileOptions, Ptx, compile_ptx_with_opts};
#[cfg(feature = "gpu")]
use sha2::{Digest, Sha256};
#[cfg(feature = "gpu")]
use std::fs;
#[cfg(feature = "gpu")]
use std::path::PathBuf;
#[cfg(feature = "gpu")]
use std::sync::{Arc, OnceLock};

use super::TensorAVT;

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

#[cfg(feature = "gpu")]
const TENSOR_AVT_KERNEL_SRC: &str = include_str!("kernels_tensor_avt.cu");

#[cfg(feature = "gpu")]
fn tensor_ptx_cache_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("GOROROBA_TENSOR_AVT_PTX_CACHE_DIR") {
        return PathBuf::from(dir);
    }
    std::env::temp_dir().join("gororoba_tensor_avt_ptx")
}

#[cfg(feature = "gpu")]
fn tensor_ptx_cache_path(opts: &CompileOptions) -> PathBuf {
    let mut hasher = Sha256::new();
    hasher.update(TENSOR_AVT_KERNEL_SRC.as_bytes());
    hasher.update(format!("{opts:?}").as_bytes());
    let digest = format!("{:x}", hasher.finalize());
    tensor_ptx_cache_dir().join(format!("tensor_avt_{digest}.ptx"))
}

#[cfg(feature = "gpu")]
fn compile_tensor_avt_ptx() -> Result<Ptx, String> {
    let opts = tensor_compile_opts();
    let cache_path = tensor_ptx_cache_path(&opts);
    if cache_path.is_file() {
        return Ok(Ptx::from_file(cache_path));
    }

    let ptx = compile_ptx_with_opts(TENSOR_AVT_KERNEL_SRC, opts)
        .map_err(|e| format!("NVRTC compile: {}", e))?;
    let ptx_src = ptx.to_src();
    if let Some(parent) = cache_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let _ = fs::write(&cache_path, &ptx_src);
    Ok(Ptx::from_src(ptx_src))
}

#[cfg(feature = "gpu")]
struct TensorAvtGpuRuntime {
    ctx: Arc<CudaContext>,
    default_stream: Arc<CudaStream>,
    sm_count: u32,
    tensor_avt_basis_kernel: CudaFunction,
    tensor_avt_dense_kernel: CudaFunction,
    tensor_cd_mul_kernel: CudaFunction,
    tensor_cd_mul_batched_kernel: CudaFunction,
    tensor_norm_sq_kernel: CudaFunction,
}

#[cfg(feature = "gpu")]
impl TensorAvtGpuRuntime {
    fn initialize() -> Result<Arc<Self>, String> {
        let ctx = CudaContext::new(0).map_err(|e| format!("CUDA init: {}", e))?;
        let default_stream = ctx.default_stream();
        let sm_count = ctx
            .attribute(
                cudarc::driver::sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )
            .map_err(|e| format!("SM count query: {}", e))? as u32;
        let ptx = compile_tensor_avt_ptx()?;
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
            default_stream,
            sm_count,
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

pub(crate) fn tensor_avt_cuda_available() -> bool {
    #[cfg(feature = "gpu")]
    {
        crate::gpu::is_gpu_available()
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}

#[cfg(feature = "gpu")]
fn tensor_avt_batch_launch_shape(dim: usize, batch_size: usize, sm_count: u32) -> (u32, u32, u32) {
    let grid_x = (batch_size as u32).div_ceil(16);
    for warps_per_block in [4u32, 2u32, 1u32] {
        let tile_rows_per_block = 16 * warps_per_block as usize;
        let grid_y = (dim as u32).div_ceil(tile_rows_per_block as u32);
        if grid_x * grid_y >= sm_count || warps_per_block == 1 {
            return (warps_per_block, grid_x, grid_y);
        }
    }
    unreachable!("expected a launch shape at one warp per block")
}

#[cfg(feature = "gpu")]
pub struct TensorAvtMulGpuWorkspace {
    stream: Arc<CudaStream>,
    d_a: CudaSlice<f32>,
    d_x: CudaSlice<f32>,
    d_y: CudaSlice<f32>,
    h_y: Vec<f32>,
    dim: usize,
    max_batch_size: usize,
}

#[cfg(feature = "gpu")]
impl TensorAvtMulGpuWorkspace {
    pub fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub fn upload_a(&mut self, a: &[f32]) -> Result<(), String> {
        if a.len() != self.dim {
            return Err(format!(
                "a input length {} must equal dim {}",
                a.len(),
                self.dim
            ));
        }
        self.stream
            .memcpy_htod(a, &mut self.d_a)
            .map_err(|e| format!("Upload a: {}", e))
    }

    pub fn upload_x(&mut self, x: &[f32], batch_size: usize, dim: usize) -> Result<(), String> {
        if dim != self.dim {
            return Err(format!(
                "workspace dim {} does not match requested dim {}",
                self.dim, dim
            ));
        }
        let expected = batch_size * dim;
        if x.len() != expected {
            return Err(format!(
                "x input length {} must equal batch_size * dim {}",
                x.len(),
                expected
            ));
        }
        if batch_size > self.max_batch_size {
            return Err(format!(
                "batch size {} exceeds workspace capacity {}",
                batch_size, self.max_batch_size
            ));
        }
        let mut active = self.d_x.slice_mut(0..expected);
        self.stream
            .memcpy_htod(x, &mut active)
            .map_err(|e| format!("Upload x: {}", e))
    }

    pub fn download_y(&mut self, len: usize) -> Result<Vec<f32>, String> {
        if len > self.d_y.len() {
            return Err(format!(
                "output length {} exceeds workspace capacity {}",
                len,
                self.d_y.len()
            ));
        }
        if self.h_y.len() < len {
            self.h_y.resize(len, 0.0);
        }
        let active = self.d_y.slice(0..len);
        self.stream
            .memcpy_dtoh(&active, &mut self.h_y[..len])
            .map_err(|e| format!("Copy y: {}", e))?;
        Ok(self.h_y[..len].to_vec())
    }
}

#[cfg(feature = "gpu")]
pub struct TensorAvtNormGpuWorkspace {
    stream: Arc<CudaStream>,
    d_vectors: CudaSlice<f32>,
    d_norms: CudaSlice<f32>,
    h_norms: Vec<f32>,
    dim: usize,
    max_vectors: usize,
}

#[cfg(feature = "gpu")]
impl TensorAvtNormGpuWorkspace {
    pub fn max_vectors(&self) -> usize {
        self.max_vectors
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub fn upload_vectors(
        &mut self,
        vectors: &[f32],
        n_vectors: usize,
        dim: usize,
    ) -> Result<(), String> {
        if dim != self.dim {
            return Err(format!(
                "workspace dim {} does not match requested dim {}",
                self.dim, dim
            ));
        }
        let expected = n_vectors * dim;
        if vectors.len() != expected {
            return Err(format!(
                "vectors length {} must equal n_vectors * dim {}",
                vectors.len(),
                expected
            ));
        }
        if n_vectors > self.max_vectors {
            return Err(format!(
                "n_vectors {} exceeds workspace capacity {}",
                n_vectors, self.max_vectors
            ));
        }
        let mut active = self.d_vectors.slice_mut(0..expected);
        self.stream
            .memcpy_htod(vectors, &mut active)
            .map_err(|e| format!("Upload vectors: {}", e))
    }

    pub fn download_norms(&mut self, n_vectors: usize) -> Result<Vec<f32>, String> {
        if n_vectors > self.max_vectors {
            return Err(format!(
                "n_vectors {} exceeds workspace capacity {}",
                n_vectors, self.max_vectors
            ));
        }
        if self.h_norms.len() < n_vectors {
            self.h_norms.resize(n_vectors, 0.0);
        }
        let active = self.d_norms.slice(0..n_vectors);
        self.stream
            .memcpy_dtoh(&active, &mut self.h_norms[..n_vectors])
            .map_err(|e| format!("Copy norms: {}", e))?;
        Ok(self.h_norms[..n_vectors].to_vec())
    }
}

#[cfg(feature = "gpu")]
impl TensorAVT {
    fn ensure_gpu_slice<T>(
        &self,
        slice: &CudaSlice<T>,
        expected_len: usize,
        label: &str,
        runtime: &TensorAvtGpuRuntime,
    ) -> Result<(), String> {
        if slice.len() < expected_len {
            return Err(format!(
                "{label} device buffer too small: expected at least {expected_len}, got {}",
                slice.len()
            ));
        }
        if slice.context() != &runtime.ctx {
            return Err(format!(
                "{label} device buffer belongs to CUDA device {}, expected {}",
                slice.context().ordinal(),
                runtime.ctx.ordinal()
            ));
        }
        Ok(())
    }

    fn launch_cd_mul_device(
        &self,
        stream: &Arc<CudaStream>,
        runtime: &TensorAvtGpuRuntime,
        d_a: &CudaSlice<f32>,
        d_x: &CudaSlice<f32>,
        d_y: &mut CudaSlice<f32>,
    ) -> Result<(), String> {
        self.ensure_gpu_slice(d_a, self.dim, "a", runtime)?;
        self.ensure_gpu_slice(d_x, self.dim, "x", runtime)?;
        self.ensure_gpu_slice(d_y, self.dim, "y", runtime)?;

        let cfg = LaunchConfig {
            grid_dim: (self.tile_count as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let dim_u = self.dim as u32;
        let mut builder = stream.launch_builder(&runtime.tensor_cd_mul_kernel);
        builder.arg(d_a);
        builder.arg(d_x);
        builder.arg(d_y);
        builder.arg(&dim_u);
        unsafe {
            builder.launch(cfg).map_err(|e| format!("Launch: {}", e))?;
        }
        Ok(())
    }

    fn launch_cd_mul_batch_device(
        &self,
        stream: &Arc<CudaStream>,
        runtime: &TensorAvtGpuRuntime,
        d_a: &CudaSlice<f32>,
        d_x: &CudaSlice<f32>,
        d_y: &mut CudaSlice<f32>,
        batch_size: usize,
    ) -> Result<(), String> {
        let expected = batch_size * self.dim;
        self.ensure_gpu_slice(d_a, self.dim, "a", runtime)?;
        self.ensure_gpu_slice(d_x, expected, "x_batch", runtime)?;
        self.ensure_gpu_slice(d_y, expected, "y_batch", runtime)?;

        let (warps_per_block, grid_x, grid_y) =
            tensor_avt_batch_launch_shape(self.dim, batch_size, runtime.sm_count);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (warps_per_block * 32, 1, 1),
            shared_mem_bytes: 0,
        };

        let dim_u = self.dim as u32;
        let batch_u = batch_size as u32;
        let mut builder = stream.launch_builder(&runtime.tensor_cd_mul_batched_kernel);
        builder.arg(d_a);
        builder.arg(d_x);
        builder.arg(d_y);
        builder.arg(&dim_u);
        builder.arg(&batch_u);
        unsafe {
            builder.launch(cfg).map_err(|e| format!("Launch: {}", e))?;
        }
        Ok(())
    }

    fn launch_norm_sq_batch_device(
        &self,
        stream: &Arc<CudaStream>,
        runtime: &TensorAvtGpuRuntime,
        d_vectors: &CudaSlice<f32>,
        d_norms: &mut CudaSlice<f32>,
        n_vectors: usize,
    ) -> Result<(), String> {
        let expected = n_vectors * self.dim;
        self.ensure_gpu_slice(d_vectors, expected, "vectors", runtime)?;
        self.ensure_gpu_slice(d_norms, n_vectors, "norms", runtime)?;

        let block_size = 256u32.min(self.dim as u32);
        let dim_i = self.dim as i32;
        let n_vec_i = n_vectors as i32;
        let cfg = LaunchConfig {
            grid_dim: (n_vectors as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = stream.launch_builder(&runtime.tensor_norm_sq_kernel);
        builder.arg(d_vectors);
        builder.arg(d_norms);
        builder.arg(&dim_i);
        builder.arg(&n_vec_i);
        unsafe {
            builder.launch(cfg).map_err(|e| format!("Launch: {}", e))?;
        }
        Ok(())
    }

    pub fn new_gpu_mul_workspace(
        &self,
        max_batch_size: usize,
    ) -> Result<TensorAvtMulGpuWorkspace, String> {
        if max_batch_size == 0 {
            return Err("max_batch_size must be > 0".into());
        }
        let runtime = tensor_avt_gpu_runtime()?;
        let stream = runtime.default_stream.clone();
        let d_a = stream
            .alloc_zeros::<f32>(self.dim)
            .map_err(|e| format!("Alloc a: {}", e))?;
        let d_x = stream
            .alloc_zeros::<f32>(self.dim * max_batch_size)
            .map_err(|e| format!("Alloc x: {}", e))?;
        let d_y = stream
            .alloc_zeros::<f32>(self.dim * max_batch_size)
            .map_err(|e| format!("Alloc y: {}", e))?;
        Ok(TensorAvtMulGpuWorkspace {
            stream,
            d_a,
            d_x,
            d_y,
            h_y: vec![0.0; self.dim * max_batch_size],
            dim: self.dim,
            max_batch_size,
        })
    }

    pub fn new_gpu_norm_workspace(
        &self,
        max_vectors: usize,
    ) -> Result<TensorAvtNormGpuWorkspace, String> {
        if max_vectors == 0 {
            return Err("max_vectors must be > 0".into());
        }
        let runtime = tensor_avt_gpu_runtime()?;
        let stream = runtime.default_stream.clone();
        let d_vectors = stream
            .alloc_zeros::<f32>(self.dim * max_vectors)
            .map_err(|e| format!("Alloc vectors: {}", e))?;
        let d_norms = stream
            .alloc_zeros::<f32>(max_vectors)
            .map_err(|e| format!("Alloc norms: {}", e))?;
        Ok(TensorAvtNormGpuWorkspace {
            stream,
            d_vectors,
            d_norms,
            h_norms: vec![0.0; max_vectors],
            dim: self.dim,
            max_vectors,
        })
    }

    pub fn compute_cd_mul_device(
        &self,
        a: &CudaSlice<f32>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
    ) -> Result<(), String> {
        let runtime = tensor_avt_gpu_runtime()?;
        let stream = y.stream().clone();
        self.launch_cd_mul_device(&stream, &runtime, a, x, y)
    }

    pub fn compute_cd_mul_batch_device(
        &self,
        a: &CudaSlice<f32>,
        x_batch: &CudaSlice<f32>,
        y_batch: &mut CudaSlice<f32>,
        batch_size: usize,
    ) -> Result<(), String> {
        let runtime = tensor_avt_gpu_runtime()?;
        let stream = y_batch.stream().clone();
        self.launch_cd_mul_batch_device(&stream, &runtime, a, x_batch, y_batch, batch_size)
    }

    pub fn compute_norm_sq_batch_device(
        &self,
        vectors: &CudaSlice<f32>,
        norms: &mut CudaSlice<f32>,
        n_vectors: usize,
    ) -> Result<(), String> {
        let runtime = tensor_avt_gpu_runtime()?;
        let stream = norms.stream().clone();
        self.launch_norm_sq_batch_device(&stream, &runtime, vectors, norms, n_vectors)
    }

    pub fn compute_cd_mul_with_workspace(
        &self,
        a: &[f32],
        x: &[f32],
        workspace: &mut TensorAvtMulGpuWorkspace,
    ) -> Result<Vec<f32>, String> {
        assert_eq!(a.len(), self.dim, "a must have dim elements");
        assert_eq!(x.len(), self.dim, "x must have dim elements");
        workspace.upload_a(a)?;
        workspace.upload_x(x, 1, self.dim)?;
        let runtime = tensor_avt_gpu_runtime()?;
        let stream = workspace.stream.clone();
        self.launch_cd_mul_device(
            &stream,
            &runtime,
            &workspace.d_a,
            &workspace.d_x,
            &mut workspace.d_y,
        )?;
        workspace.download_y(self.dim)
    }

    pub fn compute_cd_mul_batch_with_workspace(
        &self,
        a: &[f32],
        x_batch: &[f32],
        batch_size: usize,
        workspace: &mut TensorAvtMulGpuWorkspace,
    ) -> Result<Vec<f32>, String> {
        assert_eq!(a.len(), self.dim, "a must have dim elements");
        assert_eq!(
            x_batch.len(),
            batch_size * self.dim,
            "x_batch must have batch_size * dim elements"
        );
        workspace.upload_a(a)?;
        workspace.upload_x(x_batch, batch_size, self.dim)?;
        let runtime = tensor_avt_gpu_runtime()?;
        let stream = workspace.stream.clone();
        self.launch_cd_mul_batch_device(
            &stream,
            &runtime,
            &workspace.d_a,
            &workspace.d_x,
            &mut workspace.d_y,
            batch_size,
        )?;
        workspace.download_y(batch_size * self.dim)
    }

    pub fn compute_norm_sq_batch_with_workspace(
        &self,
        vectors: &[f32],
        n_vectors: usize,
        workspace: &mut TensorAvtNormGpuWorkspace,
    ) -> Result<Vec<f32>, String> {
        assert_eq!(
            vectors.len(),
            n_vectors * self.dim,
            "vectors length must be n_vectors * dim"
        );
        workspace.upload_vectors(vectors, n_vectors, self.dim)?;
        let runtime = tensor_avt_gpu_runtime()?;
        let stream = workspace.stream.clone();
        self.launch_norm_sq_batch_device(
            &stream,
            &runtime,
            &workspace.d_vectors,
            &mut workspace.d_norms,
            n_vectors,
        )?;
        workspace.download_norms(n_vectors)
    }

    pub fn launch_cd_mul_with_workspace(
        &self,
        workspace: &mut TensorAvtMulGpuWorkspace,
    ) -> Result<(), String> {
        let runtime = tensor_avt_gpu_runtime()?;
        let stream = workspace.stream.clone();
        self.launch_cd_mul_device(
            &stream,
            &runtime,
            &workspace.d_a,
            &workspace.d_x,
            &mut workspace.d_y,
        )
    }

    pub fn launch_cd_mul_batch_with_workspace(
        &self,
        batch_size: usize,
        workspace: &mut TensorAvtMulGpuWorkspace,
    ) -> Result<(), String> {
        if batch_size == 0 || batch_size > workspace.max_batch_size {
            return Err(format!(
                "batch_size must be in 1..={}, got {}",
                workspace.max_batch_size, batch_size
            ));
        }
        let runtime = tensor_avt_gpu_runtime()?;
        let stream = workspace.stream.clone();
        self.launch_cd_mul_batch_device(
            &stream,
            &runtime,
            &workspace.d_a,
            &workspace.d_x,
            &mut workspace.d_y,
            batch_size,
        )
    }

    pub fn launch_norm_sq_batch_with_workspace(
        &self,
        n_vectors: usize,
        workspace: &mut TensorAvtNormGpuWorkspace,
    ) -> Result<(), String> {
        if n_vectors == 0 || n_vectors > workspace.max_vectors {
            return Err(format!(
                "n_vectors must be in 1..={}, got {}",
                workspace.max_vectors, n_vectors
            ));
        }
        let runtime = tensor_avt_gpu_runtime()?;
        let stream = workspace.stream.clone();
        self.launch_norm_sq_batch_device(
            &stream,
            &runtime,
            &workspace.d_vectors,
            &mut workspace.d_norms,
            n_vectors,
        )
    }

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
        let mut d_field = stream
            .alloc_zeros::<f32>(n_cells)
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

        stream
            .clone_dtoh(&d_field)
            .map_err(|e| format!("Copy back: {}", e))
    }

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
        let mut d_field = stream
            .alloc_zeros::<f32>(n_cells)
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

        stream
            .clone_dtoh(&d_field)
            .map_err(|e| format!("Copy back: {}", e))
    }

    pub fn compute_cd_mul(&self, a: &[f32], x: &[f32]) -> Result<Vec<f32>, String> {
        assert_eq!(a.len(), self.dim, "a must have dim elements");
        assert_eq!(x.len(), self.dim, "x must have dim elements");

        let runtime = tensor_avt_gpu_runtime()?;
        let stream = runtime.ctx.default_stream();
        let d_a = stream
            .clone_htod(a)
            .map_err(|e| format!("Upload a: {}", e))?;
        let d_x = stream
            .clone_htod(x)
            .map_err(|e| format!("Upload x: {}", e))?;
        let mut d_y = stream
            .alloc_zeros::<f32>(self.dim)
            .map_err(|e| format!("Alloc y: {}", e))?;
        self.launch_cd_mul_device(&stream, &runtime, &d_a, &d_x, &mut d_y)?;
        stream
            .clone_dtoh(&d_y)
            .map_err(|e| format!("Copy y: {}", e))
    }

    pub fn compute_cd_mul_batch(
        &self,
        a: &[f32],
        x_batch: &[f32],
        batch_size: usize,
    ) -> Result<Vec<f32>, String> {
        assert_eq!(a.len(), self.dim, "a must have dim elements");
        assert_eq!(
            x_batch.len(),
            batch_size * self.dim,
            "x_batch must have batch_size * dim elements"
        );

        let runtime = tensor_avt_gpu_runtime()?;
        let stream = runtime.ctx.default_stream();
        let d_a = stream
            .clone_htod(a)
            .map_err(|e| format!("Upload a: {}", e))?;
        let d_x = stream
            .clone_htod(x_batch)
            .map_err(|e| format!("Upload x_batch: {}", e))?;
        let mut d_y = stream
            .alloc_zeros::<f32>(batch_size * self.dim)
            .map_err(|e| format!("Alloc y_batch: {}", e))?;
        self.launch_cd_mul_batch_device(&stream, &runtime, &d_a, &d_x, &mut d_y, batch_size)?;
        stream
            .clone_dtoh(&d_y)
            .map_err(|e| format!("Copy y_batch: {}", e))
    }

    pub fn compute_norm_sq_batch(
        &self,
        vectors: &[f32],
        n_vectors: usize,
    ) -> Result<Vec<f32>, String> {
        assert_eq!(
            vectors.len(),
            n_vectors * self.dim,
            "vectors length must be n_vectors * dim"
        );

        let runtime = tensor_avt_gpu_runtime()?;
        let stream = runtime.ctx.default_stream();
        let d_vectors = stream
            .clone_htod(vectors)
            .map_err(|e| format!("Upload vectors: {}", e))?;
        let mut d_norms = stream
            .alloc_zeros::<f32>(n_vectors)
            .map_err(|e| format!("Alloc norms: {}", e))?;
        self.launch_norm_sq_batch_device(&stream, &runtime, &d_vectors, &mut d_norms, n_vectors)?;
        stream
            .clone_dtoh(&d_norms)
            .map_err(|e| format!("Copy norms: {}", e))
    }
}

#[cfg(not(feature = "gpu"))]
impl TensorAVT {
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

    pub fn compute_cd_mul(&self, a: &[f32], x: &[f32]) -> Result<Vec<f32>, String> {
        self.compute_cd_mul_cpu(a, x)
    }

    pub fn compute_cd_mul_batch(
        &self,
        a: &[f32],
        x_batch: &[f32],
        batch_size: usize,
    ) -> Result<Vec<f32>, String> {
        self.compute_cd_mul_batch_cpu(a, x_batch, batch_size)
    }

    pub fn compute_norm_sq_batch(
        &self,
        vectors: &[f32],
        n_vectors: usize,
    ) -> Result<Vec<f32>, String> {
        self.compute_norm_sq_batch_cpu(vectors, n_vectors)
    }
}

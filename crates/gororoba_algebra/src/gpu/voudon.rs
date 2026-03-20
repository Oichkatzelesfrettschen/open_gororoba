//! GPU-accelerated Voudon (256D) frustration density computation.

#[cfg(feature = "gpu")]
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
#[cfg(feature = "gpu")]
use cudarc::nvrtc::compile_ptx;
#[cfg(feature = "gpu")]
use std::sync::Arc;

/// CUDA kernel for Voudon (256D) frustration field generation.
#[cfg(feature = "gpu")]
const VOUDON_KERNEL_SRC: &str = r#"
// Device function: cd_basis_mul_sign (optimized for 256D)
__device__ int cd_basis_mul_sign(unsigned int p, unsigned int q) {
    int sign = 1;
    unsigned int half = 128; // Hardcoded for 256D Voudon

    while (half > 0) {
        unsigned int p_hi = (p >= half) ? 1 : 0;
        unsigned int q_hi = (q >= half) ? 1 : 0;
        unsigned int branch = (p_hi << 1) | q_hi;

        if (branch == 1) {
            unsigned int qh = q - half;
            q = p; p = qh;
        } else if (branch == 2) {
            p -= half;
            if (q != 0) sign = -sign;
        } else if (branch == 3) {
            unsigned int qh = q - half;
            unsigned int ph = p - half;
            if (qh == 0) return -sign;
            p = qh; q = ph;
        }
        half >>= 1;
    }
    return sign;
}

// Simple spatial hash for field generation
__device__ float spatial_noise(int x, int y, int z, unsigned int seed) {
    unsigned int h = seed ^ (x * 73856093u) ^ (y * 19349663u) ^ (z * 83492791u);
    h = (h >> 13) ^ h;
    h *= 0x5bd1e995u;
    h = (h >> 15) ^ h;
    return (float)(h & 0xFFFFu) / 65535.0f;
}

extern "C" __global__ void voudon_frustration_kernel(
    float* __restrict__ frustration_field,
    int nx, int ny, int nz,
    unsigned int seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nx * ny * nz;
    if (idx >= N) return;

    int x = idx % nx;
    int y = (idx / nx) % ny;
    int z = idx / (nx * ny);

    // Generate local 256D Voudon sample via spatial hash
    // We sample a subset of basis components to estimate frustration
    float local_frustration = 0.0f;
    int samples = 32; 
    
    for (int s = 0; s < samples; s++) {
        unsigned int i = (unsigned int)(spatial_noise(x, y, z, seed ^ s) * 255.0f);
        unsigned int j = (unsigned int)(spatial_noise(x, y, z, seed ^ (s + 100)) * 255.0f);
        unsigned int k = (unsigned int)(spatial_noise(x, y, z, seed ^ (s + 200)) * 255.0f);
        
        // Count alternativity violations: [i, i, j] != 0
        // In CD algebra, this is mapped to non-associative triples
        int s1 = cd_basis_mul_sign(i, i); // usually 1 or -1
        int ij_idx = i ^ j;
        int s2 = cd_basis_mul_sign(i, j);
        int i_ij_sign = cd_basis_mul_sign(i, ij_idx) * s2;
        
        if (s1 != i_ij_sign) {
            local_frustration += 1.0f;
        }
    }

    frustration_field[idx] = local_frustration / (float)samples;
}
"#;

pub struct Cd256FrustrationKernel;

impl Cd256FrustrationKernel {
    #[cfg(feature = "gpu")]
    pub fn compute_field(nx: usize, ny: usize, nz: usize, seed: u32) -> Result<Vec<f32>, String> {
        let ctx = Arc::new(CudaContext::new(0).map_err(|e| format!("CUDA init: {}", e))?);
        let stream = ctx.default_stream();

        let ptx = compile_ptx(VOUDON_KERNEL_SRC).map_err(|e| format!("NVRTC compile: {}", e))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| format!("Module load: {}", e))?;
        let kernel = module
            .load_function("voudon_frustration_kernel")
            .map_err(|e| format!("Kernel load: {}", e))?;

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
        let mut builder = stream.launch_builder(&kernel);
        builder.arg(&mut d_field);
        builder.arg(&nx_i);
        builder.arg(&ny_i);
        builder.arg(&nz_i);
        builder.arg(&seed);

        unsafe {
            builder.launch(cfg).map_err(|e| format!("Launch: {}", e))?;
        }

        let host_field: Vec<f32> = stream
            .clone_dtoh(&d_field)
            .map_err(|e| format!("Copy back: {}", e))?;

        Ok(host_field)
    }

    #[cfg(not(feature = "gpu"))]
    pub fn compute_field(
        _nx: usize,
        _ny: usize,
        _nz: usize,
        _seed: u32,
    ) -> Result<Vec<f32>, String> {
        Err("GPU feature not enabled".to_string())
    }
}

//! GPU-accelerated Voudon (256D) frustration density computation.

#[cfg(feature = "gpu")]
use cudarc::driver::PushKernelArg;
#[cfg(feature = "gpu")]
use gororoba_gpu_cuda::{Buffer, CompileOptions, LaunchConfig, ModuleRegistry};

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
    pub const DIM: usize = 256;
    pub const SAMPLES_PER_CELL: usize = 32;

    pub fn compute_field_cpu(
        nx: usize,
        ny: usize,
        nz: usize,
        seed: u32,
    ) -> Result<Vec<f32>, String> {
        let n_cells = nx
            .checked_mul(ny)
            .and_then(|xy| xy.checked_mul(nz))
            .ok_or_else(|| format!("Voudon field shape {nx}x{ny}x{nz} overflows usize"))?;
        let mut field = Vec::with_capacity(n_cells);

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let mut local_frustration = 0u32;
                    for sample in 0..Self::SAMPLES_PER_CELL {
                        let i = spatial_index_256(x, y, z, seed ^ sample as u32);
                        let j = spatial_index_256(x, y, z, seed ^ (sample as u32 + 100));

                        let s1 = cd_basis_mul_sign_256(i, i);
                        let ij_idx = i ^ j;
                        let s2 = cd_basis_mul_sign_256(i, j);
                        let i_ij_sign = cd_basis_mul_sign_256(i, ij_idx) * s2;

                        if s1 != i_ij_sign {
                            local_frustration += 1;
                        }
                    }
                    field.push(local_frustration as f32 / Self::SAMPLES_PER_CELL as f32);
                }
            }
        }

        Ok(field)
    }

    #[cfg(feature = "gpu")]
    pub fn compute_field(nx: usize, ny: usize, nz: usize, seed: u32) -> Result<Vec<f32>, String> {
        let ctx_wrapper = gororoba_gpu_cuda::Context::with_default_device()
            .map_err(|e| format!("CUDA init: {}", e))?;
        let stream = ctx_wrapper.default_stream();

        let opts = CompileOptions::empty();
        let ptx = CompileOptions::compile_ptx(VOUDON_KERNEL_SRC, &opts)
            .map_err(|e| format!("NVRTC compile: {}", e))?;
        let registry = ModuleRegistry::load(ctx_wrapper.raw(), ptx, &["voudon_frustration_kernel"])
            .map_err(|e| format!("Module load: {}", e))?;
        let kernel = registry
            .get("voudon_frustration_kernel")
            .map_err(|e| format!("Kernel load: {}", e))?;

        let n_cells = nx
            .checked_mul(ny)
            .and_then(|xy| xy.checked_mul(nz))
            .ok_or_else(|| format!("Voudon field shape {nx}x{ny}x{nz} overflows usize"))?;
        let n_cells_u32 = u32::try_from(n_cells)
            .map_err(|_| format!("Voudon field cell count {n_cells} exceeds u32 dispatch"))?;
        let mut d_field =
            Buffer::<f32>::alloc_zeros(&stream, n_cells).map_err(|e| format!("Alloc: {}", e))?;

        let cfg = LaunchConfig::launch_1d(n_cells_u32);

        let nx_i = nx as i32;
        let ny_i = ny as i32;
        let nz_i = nz as i32;
        let mut builder = stream.launch_builder(&kernel);
        builder.arg(d_field.raw_mut());
        builder.arg(&nx_i);
        builder.arg(&ny_i);
        builder.arg(&nz_i);
        builder.arg(&seed);

        unsafe {
            builder.launch(cfg).map_err(|e| format!("Launch: {}", e))?;
        }

        let host_field: Vec<f32> = d_field
            .dtoh_vec()
            .map_err(|e| format!("Copy back: {}", e))?;

        Ok(host_field)
    }

    #[cfg(not(feature = "gpu"))]
    pub fn compute_field(nx: usize, ny: usize, nz: usize, seed: u32) -> Result<Vec<f32>, String> {
        Self::compute_field_cpu(nx, ny, nz, seed)
    }
}

pub fn cd_basis_mul_sign_256(mut p: u32, mut q: u32) -> i32 {
    let mut sign = 1;
    let mut half = 128u32;

    while half > 0 {
        let p_hi = u32::from(p >= half);
        let q_hi = u32::from(q >= half);
        let branch = (p_hi << 1) | q_hi;

        if branch == 1 {
            let qh = q - half;
            q = p;
            p = qh;
        } else if branch == 2 {
            p -= half;
            if q != 0 {
                sign = -sign;
            }
        } else if branch == 3 {
            let qh = q - half;
            let ph = p - half;
            if qh == 0 {
                return -sign;
            }
            p = qh;
            q = ph;
        }

        half >>= 1;
    }

    sign
}

pub fn spatial_index_256(x: usize, y: usize, z: usize, seed: u32) -> u32 {
    let h = spatial_hash(x as u32, y as u32, z as u32, seed);
    ((h & 0xffff) * 255) / 65_535
}

fn spatial_hash(x: u32, y: u32, z: u32, seed: u32) -> u32 {
    let mut hash = seed ^ x.wrapping_mul(73_856_093) ^ y.wrapping_mul(19_349_663);
    hash ^= z.wrapping_mul(83_492_791);
    hash = (hash >> 13) ^ hash;
    hash = hash.wrapping_mul(0x5bd1_e995);
    (hash >> 15) ^ hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_voudon_field_has_expected_shape_and_range() {
        let field = Cd256FrustrationKernel::compute_field_cpu(3, 2, 2, 42).unwrap();
        assert_eq!(field.len(), 12);
        assert!(field.iter().all(|value| (0.0..=1.0).contains(value)));
    }

    #[test]
    fn spatial_index_matches_voudon_range() {
        for seed in [0u32, 1, 42, 1234] {
            let value = spatial_index_256(3, 5, 7, seed);
            assert!(value < Cd256FrustrationKernel::DIM as u32);
        }
    }
}

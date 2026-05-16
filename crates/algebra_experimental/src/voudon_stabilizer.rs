//! GPU-accelerated search for stable zero-divisor cycles in 256D Voudon algebra.

#[cfg(feature = "gpu")]
use cudarc::driver::{LaunchConfig, PushKernelArg};
#[cfg(feature = "gpu")]
use cudarc::nvrtc::compile_ptx;

/// CUDA kernel to find stable (associative) triples in 256D.
#[cfg(feature = "gpu")]
const STABILIZER_KERNEL_SRC: &str = r#"
__device__ int cd_basis_mul_sign(unsigned int p, unsigned int q) {
    int sign = 1;
    unsigned int half = 128; 
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

extern "C" __global__ void find_stable_cycles_kernel(
    unsigned int* __restrict__ stable_triples, // [3 * max_triples]
    unsigned int* __restrict__ count,
    unsigned int max_triples
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 256) return;

    for (unsigned int j = 0; j < 256; j++) {
        if (i == j) continue;
        
        // Triple (i, i, j) is stable if [i, i, j] = 0 (alternativity holds)
        int s1 = cd_basis_mul_sign(i, i);
        int ij_idx = i ^ j;
        int s2 = cd_basis_mul_sign(i, j);
        int i_ij_sign = cd_basis_mul_sign(i, ij_idx) * s2;
        
        if (s1 == i_ij_sign) {
            unsigned int c = atomicAdd(count, 1);
            if (c < max_triples) {
                stable_triples[c * 3 + 0] = i;
                stable_triples[c * 3 + 1] = i;
                stable_triples[c * 3 + 2] = j;
            }
        }
    }
}
"#;

pub struct Cd256StabilizerKernel;

impl Cd256StabilizerKernel {
    #[cfg(feature = "gpu")]
    pub fn find_stable_cycles(max_triples: usize) -> Result<Vec<(usize, usize, usize)>, String> {
        // Delegate context acquisition to gpu_cuda::Context so the
        // get_count + ordinal-range checks live in one place across
        // the workspace.
        let ctx_wrapper = gororoba_gpu_cuda::Context::with_default_device()
            .map_err(|e| format!("CUDA init: {}", e))?;
        let ctx = ctx_wrapper.raw().clone();
        let stream = ctx.default_stream();

        let ptx =
            compile_ptx(STABILIZER_KERNEL_SRC).map_err(|e| format!("NVRTC compile: {}", e))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| format!("Module load: {}", e))?;
        let kernel = module
            .load_function("find_stable_cycles_kernel")
            .map_err(|e| format!("Kernel load: {}", e))?;

        let mut d_triples = stream
            .alloc_zeros::<u32>(max_triples * 3)
            .map_err(|e| format!("Alloc: {}", e))?;
        let mut d_count = stream
            .alloc_zeros::<u32>(1)
            .map_err(|e| format!("Alloc: {}", e))?;

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = stream.launch_builder(&kernel);
        builder.arg(&mut d_triples);
        builder.arg(&mut d_count);
        let max_triples_u32 = max_triples as u32;
        builder.arg(&max_triples_u32);

        unsafe {
            builder.launch(cfg).map_err(|e| format!("Launch: {}", e))?;
        }

        let count_vec = stream
            .clone_dtoh(&d_count)
            .map_err(|e| format!("Copy count: {}", e))?;
        let count = (count_vec[0] as usize).min(max_triples);

        let triples_vec = stream
            .clone_dtoh(&d_triples)
            .map_err(|e| format!("Copy triples: {}", e))?;
        let mut result = Vec::new();
        for i in 0..count {
            result.push((
                triples_vec[i * 3] as usize,
                triples_vec[i * 3 + 1] as usize,
                triples_vec[i * 3 + 2] as usize,
            ));
        }

        Ok(result)
    }

    #[cfg(not(feature = "gpu"))]
    pub fn find_stable_cycles(_max_triples: usize) -> Result<Vec<(usize, usize, usize)>, String> {
        Err("GPU feature not enabled".to_string())
    }
}

// FP8 e4m3 D3Q19 LBM kernel.
// Storage: __nv_fp8_e4m3 (1 byte/value, AoS, stride 20 per cell).
// Compute: FP32 promoted immediately after load (storage-compute split).
// Requires: CUDA 11.8+, SM 8.9 (Ada Lovelace / RTX 4xxx).
// Bandwidth: 4x reduction vs FP32 (1 byte vs 4 bytes per distribution scalar).
// Risk: FP8 e4m3 has 3-bit mantissa (~2 decimal digits). LBM distributions
//   are O(0.01..0.33), well within e4m3 range (max ~448). However, the
//   limited precision causes ~1% rounding error per step, limiting to
//   short-time dynamics and qualitative flow features only.
// YSU trick: uchar4 vectorized loads (4 fp8 values = 4 bytes in one 32-bit
//   transaction). Stride 20 bytes per cell ensures 4-byte alignment for all idx
//   (20 is divisible by 4); 5 uchar4 loads cover all 20 slots, indices 0-18
//   are active, index 19 is unused padding.
// VRAM at 128^3: 20 * 2,097,152 * 1 * 2 (ping+pong) = ~80 MB.

#include <cuda_fp8.h>

// D3Q19 lattice velocities (suffixed _F8 to avoid ODR conflicts with other kernels)
__constant__ int D3Q19_CX_F8[19] = {
    0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0
};
__constant__ int D3Q19_CY_F8[19] = {
    0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1
};
__constant__ int D3Q19_CZ_F8[19] = {
    0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1
};

__constant__ float D3Q19_WF_F8[19] = {
    1.0f/3.0f,
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
};

__device__ __forceinline__ bool finite_f32_fp8(float x) {
    return (x == x) && (x <= 3.402823466e38f) && (x >= -3.402823466e38f);
}

// Equilibrium: Horner FMA form in FP32.
__device__ void compute_equilibrium_fp8(float* f_eq, float rho, const float* u) {
    float u_sq = u[0]*u[0] + u[1]*u[1] + u[2]*u[2];
    float base = fmaf(-1.5f, u_sq, 1.0f);
    #pragma unroll
    for (int i = 0; i < 19; i++) {
        float eu = (float)D3Q19_CX_F8[i]*u[0]
                 + (float)D3Q19_CY_F8[i]*u[1]
                 + (float)D3Q19_CZ_F8[i]*u[2];
        float w_rho = D3Q19_WF_F8[i] * rho;
        f_eq[i] = w_rho * fmaf(fmaf(eu, 4.5f, 3.0f), eu, base);
    }
}

// Helper: convert __nv_fp8_e4m3 byte to float via half intermediate (CUDA 11.8+, SM 8.9+).
// Using direct __half2float(__nv_cvt_fp8_to_halfraw(...)) is the official path.
__device__ __forceinline__ float fp8_e4m3_to_float(__nv_fp8_storage_t v) {
    return __half2float(__nv_cvt_fp8_to_halfraw(v, __NV_E4M3));
}

// Helper: float to __nv_fp8_e4m3 byte (saturate finite, clamp NaN/Inf to 0).
__device__ __forceinline__ __nv_fp8_storage_t float_to_fp8_e4m3(float v) {
    if (!finite_f32_fp8(v)) v = 0.0f;
    return __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}

// Fused collision + streaming, FP8 storage, FP32 compute.
// Per-cell stride: 20 bytes. 5 uchar4 loads cover all 20 bytes; index 19 is padding.
// __launch_bounds__(128, 4): same reasoning as kernels_fp16.cu -- target 4 blocks/SM
// to stay within ~128 regs/thread and avoid register spill at 128 threads/block.
extern "C" __launch_bounds__(128, 4) __global__ void lbm_step_fused_fp8_kernel(
    const __nv_fp8_storage_t* f_in,   // n_cells * 20 fp8 bytes (stride 20, index 19 unused)
    __nv_fp8_storage_t* f_out,
    float* rho_out,
    float* u_out,
    const float* force,
    const float* tau,
    int nx, int ny, int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_cells = nx * ny * nz;
    if (idx >= n_cells) return;

    int x = idx % nx;
    int y = (idx / nx) % ny;
    int z = idx / (nx * ny);

    // Stride 20: f_base is at idx*20 bytes, always 4-byte aligned (20 % 4 == 0).
    const __nv_fp8_storage_t* f_base = f_in + (long long)idx * 20;
    float f_local[19];

    // 5 uchar4 loads = 20 bytes; indices 0-18 are active, index 19 is padding (ignored).
    #pragma unroll
    for (int j = 0; j < 5; j++) {
        uchar4 q = __ldg((const uchar4*)(f_base + j * 4));
        int base_i = j * 4;
        if (base_i + 0 < 19) f_local[base_i + 0] = fp8_e4m3_to_float(q.x);
        if (base_i + 1 < 19) f_local[base_i + 1] = fp8_e4m3_to_float(q.y);
        if (base_i + 2 < 19) f_local[base_i + 2] = fp8_e4m3_to_float(q.z);
        if (base_i + 3 < 19) f_local[base_i + 3] = fp8_e4m3_to_float(q.w);
    }

    // Macroscopic
    float rho_local = 0.0f;
    float mx = 0.0f, my = 0.0f, mz = 0.0f;
    #pragma unroll
    for (int i = 0; i < 19; i++) {
        rho_local += f_local[i];
        mx += (float)D3Q19_CX_F8[i] * f_local[i];
        my += (float)D3Q19_CY_F8[i] * f_local[i];
        mz += (float)D3Q19_CZ_F8[i] * f_local[i];
    }

    float ux = 0.0f, uy = 0.0f, uz = 0.0f;
    if (finite_f32_fp8(rho_local) && rho_local > 1.0e-20f) {
        float inv_rho = 1.0f / rho_local;
        ux = mx * inv_rho;
        uy = my * inv_rho;
        uz = mz * inv_rho;
    } else {
        rho_local = 1.0f;
    }

    rho_out[idx] = rho_local;
    u_out[idx * 3 + 0] = ux;
    u_out[idx * 3 + 1] = uy;
    u_out[idx * 3 + 2] = uz;

    // Collision + Guo forcing (FP32)
    float f_eq[19];
    float u_vec[3] = {ux, uy, uz};
    compute_equilibrium_fp8(f_eq, rho_local, u_vec);

    float tau_local = tau[idx];
    float inv_tau = 1.0f / tau_local;
    float prefactor = 1.0f - 0.5f * inv_tau;
    float fx = force[idx * 3 + 0];
    float fy = force[idx * 3 + 1];
    float fz = force[idx * 3 + 2];

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        float fi = f_local[i] - (f_local[i] - f_eq[i]) * inv_tau;
        float eix = (float)D3Q19_CX_F8[i];
        float eiy = (float)D3Q19_CY_F8[i];
        float eiz = (float)D3Q19_CZ_F8[i];
        float s_i = ((eix - ux)*fx + (eiy - uy)*fy + (eiz - uz)*fz) * 3.0f
                  + (eix*ux + eiy*uy + eiz*uz) * (eix*fx + eiy*fy + eiz*fz) * 9.0f;
        fi += prefactor * D3Q19_WF_F8[i] * s_i;

        // Streaming (stride 20)
        int x_next = (x + D3Q19_CX_F8[i] + nx) % nx;
        int y_next = (y + D3Q19_CY_F8[i] + ny) % ny;
        int z_next = (z + D3Q19_CZ_F8[i] + nz) % nz;
        long long idx_next = (long long)x_next + nx * ((long long)y_next + ny * z_next);
        f_out[idx_next * 20 + i] = float_to_fp8_e4m3(fi);
    }
}

extern "C" __global__ void initialize_uniform_fp8_kernel(
    __nv_fp8_storage_t* f,
    float* rho_out,
    float* u_out,
    float* tau,
    float rho_init,
    float ux_init,
    float uy_init,
    float uz_init,
    float tau_val,
    int nx, int ny, int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_cells = nx * ny * nz;
    if (idx >= n_cells) return;

    rho_out[idx] = rho_init;
    u_out[idx * 3 + 0] = ux_init;
    u_out[idx * 3 + 1] = uy_init;
    u_out[idx * 3 + 2] = uz_init;
    tau[idx] = tau_val;

    float u_local[3] = {ux_init, uy_init, uz_init};
    float f_eq[19];
    compute_equilibrium_fp8(f_eq, rho_init, u_local);

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        f[idx * 20 + i] = float_to_fp8_e4m3(f_eq[i]);
    }
    // Padding slot: write zero so reads of slot 19 are well-defined.
    f[idx * 20 + 19] = 0;
}

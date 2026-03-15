// FP16 (half-precision) D3Q19 LBM kernel.
// Storage: __half (2 bytes/value, f-ping-pong AoS).
// Compute: FP32 promoted immediately after load.
// YSU tricks:
//   - half2 vectorized loads (2 halves = 4 bytes in one 32-bit transaction)
//   - __ldg() cache-friendly read for f_in (read-only)
//   - Horner FMA equilibrium (fmaf, same form as FP64 kernel)
//   - #pragma unroll on distribution loops
// Bandwidth model vs BF16: identical (both 2 bytes/value, AoS, same kernel shape).
// Key difference from BF16: FP16 has smaller mantissa range (10-bit vs 7-bit)
// but narrower dynamic range than BF16 (5-bit vs 8-bit exponent).
// For LBM: BF16 range is safer for rho spikes; FP16 is safer for small mantissa features.

#include <cuda_fp16.h>

// D3Q19 lattice velocities
__constant__ int D3Q19_CX_F16[19] = {
    0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0
};
__constant__ int D3Q19_CY_F16[19] = {
    0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1
};
__constant__ int D3Q19_CZ_F16[19] = {
    0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1
};

// D3Q19 weights (FP32)
__constant__ float D3Q19_WF_F16[19] = {
    1.0f/3.0f,
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
};

__device__ __forceinline__ bool finite_f32_fp16(float x) {
    return (x == x) && (x <= 3.402823466e38f) && (x >= -3.402823466e38f);
}

// Equilibrium in FP32 with Horner FMA form (fmaf reduces 2 ops to 1 fused).
// Algebraic: f_eq_i = w_i * rho * (1 + 3*(c.u) + 4.5*(c.u)^2 - 1.5*u^2)
// Horner:    f_eq_i = w_i * rho * ((4.5*eu + 3)*eu + base)  where base = 1 - 1.5*usq
__device__ void compute_equilibrium_fp16(float* f_eq, float rho, const float* u) {
    float u_sq = u[0]*u[0] + u[1]*u[1] + u[2]*u[2];
    float base = fmaf(-1.5f, u_sq, 1.0f);

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        float eu = (float)D3Q19_CX_F16[i]*u[0]
                 + (float)D3Q19_CY_F16[i]*u[1]
                 + (float)D3Q19_CZ_F16[i]*u[2];
        float w_rho = D3Q19_WF_F16[i] * rho;
        f_eq[i] = w_rho * fmaf(fmaf(eu, 4.5f, 3.0f), eu, base);
    }
}

// Fused Collision + Streaming kernel using FP16 storage, FP32 compute.
// YSU half2 trick: loads 18 distributions as 9 half2 values (9 coalesced 32-bit
// loads instead of 18 individual 16-bit loads). The 19th is a scalar load.
extern "C" __global__ void lbm_step_fused_fp16_kernel(
    const __half* f_in,   // Input distributions (read-only, n_cells * 19)
    __half* f_out,        // Output distributions (write, n_cells * 19)
    float* rho_out,       // Density field (FP32)
    float* u_out,         // Velocity field (FP32, 3 components)
    const float* force,   // Body force (FP32, 3 components)
    const float* tau,     // Relaxation time (FP32, scalar per cell)
    int nx, int ny, int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_cells = nx * ny * nz;
    if (idx >= n_cells) return;

    int x = idx % nx;
    int y = (idx / nx) % ny;
    int z = idx / (nx * ny);

    // --- Load 19 distributions via half2 vectorized reads ---
    // Load pairs 0-1, 2-3, 4-5, 6-7, 8-9, 10-11, 12-13, 14-15, 16-17 as half2,
    // then load 18 as a scalar half.
    const __half* f_base = f_in + (long long)idx * 19;
    float f_local[19];

    #pragma unroll
    for (int j = 0; j < 9; j++) {
        half2 h2 = __ldg((const half2*)(f_base + j * 2));
        float2 f2 = __half22float2(h2);
        float a = finite_f32_fp16(f2.x) ? f2.x : 0.0f;
        float b = finite_f32_fp16(f2.y) ? f2.y : 0.0f;
        f_local[j * 2]     = a;
        f_local[j * 2 + 1] = b;
    }
    // 19th distribution (index 18) -- scalar
    {
        float v = __half2float(__ldg(f_base + 18));
        f_local[18] = finite_f32_fp16(v) ? v : 0.0f;
    }

    // --- Macroscopic variables ---
    float rho_local = 0.0f;
    float mx = 0.0f, my = 0.0f, mz = 0.0f;
    #pragma unroll
    for (int i = 0; i < 19; i++) {
        rho_local += f_local[i];
        mx += (float)D3Q19_CX_F16[i] * f_local[i];
        my += (float)D3Q19_CY_F16[i] * f_local[i];
        mz += (float)D3Q19_CZ_F16[i] * f_local[i];
    }

    float ux = 0.0f, uy = 0.0f, uz = 0.0f;
    if (finite_f32_fp16(rho_local) && rho_local > 1.0e-20f) {
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

    // --- BGK collision + Guo forcing (FP32) ---
    float f_eq[19];
    float u_vec[3] = {ux, uy, uz};
    compute_equilibrium_fp16(f_eq, rho_local, u_vec);

    float tau_local = tau[idx];
    float inv_tau = 1.0f / tau_local;
    float prefactor = 1.0f - 0.5f * inv_tau;
    float fx = force[idx * 3 + 0];
    float fy = force[idx * 3 + 1];
    float fz = force[idx * 3 + 2];

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        float fi = f_local[i] - (f_local[i] - f_eq[i]) * inv_tau;
        float eix = (float)D3Q19_CX_F16[i];
        float eiy = (float)D3Q19_CY_F16[i];
        float eiz = (float)D3Q19_CZ_F16[i];
        float s_i = ((eix - ux)*fx + (eiy - uy)*fy + (eiz - uz)*fz) * 3.0f
                  + (eix*ux + eiy*uy + eiz*uz) * (eix*fx + eiy*fy + eiz*fz) * 9.0f;
        fi += prefactor * D3Q19_WF_F16[i] * s_i;

        // --- Streaming: scatter to neighbor ---
        int x_next = (x + D3Q19_CX_F16[i] + nx) % nx;
        int y_next = (y + D3Q19_CY_F16[i] + ny) % ny;
        int z_next = (z + D3Q19_CZ_F16[i] + nz) % nz;
        long long idx_next = (long long)x_next + nx * ((long long)y_next + ny * z_next);
        f_out[idx_next * 19 + i] = __float2half(fi);
    }
}

extern "C" __global__ void initialize_uniform_fp16_kernel(
    __half* f,
    float* rho_out,
    float* u_out,
    float* tau,      // tau stored as FP32, consistent with step kernel
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
    compute_equilibrium_fp16(f_eq, rho_init, u_local);

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        f[idx * 19 + i] = __float2half(f_eq[i]);
    }
}

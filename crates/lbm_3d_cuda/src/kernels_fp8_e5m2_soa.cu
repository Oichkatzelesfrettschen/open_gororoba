// FP8 e5m2 i-major SoA D3Q19 LBM kernel -- pull-scheme streaming.
//
// WHY e5m2 SoA vs e4m3 SoA:
//   e5m2: 5-bit exponent (range +-57344), 2-bit mantissa (~25% relative error).
//   e4m3: 4-bit exponent (range +-448),   3-bit mantissa (~12.5% relative error).
//   Use e5m2 when flow velocities or density contrasts exceed e4m3 range (|f| > 448).
//   Use e4m3 for near-equilibrium flows where mantissa precision matters more.
//   THROUGHPUT: AoS benchmark measured e5m2 == e4m3 (3130 vs 3132 MLUPS at 128^3):
//   identical load path (5x uchar4) and identical FP32 collision pipeline.
//   Expected: SoA versions also identical (same bytes/dist, same coalescing).
//   Comparison is primarily for physics, not performance.
//
// Requires SM 8.9+ (Ada Lovelace). Same 1 byte/dist as e4m3 SoA.
// VRAM at 128^3: 19 * 2,097,152 * 1 * 2 (ping+pong) = ~76 MB.

#include <cuda_fp8.h>

__constant__ int CX_F8E5S[19] = {
    0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0
};
__constant__ int CY_F8E5S[19] = {
    0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1
};
__constant__ int CZ_F8E5S[19] = {
    0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1
};

__constant__ float W_F8E5S[19] = {
    1.0f/3.0f,
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
};

__device__ __forceinline__ bool finite_f8e5s(float x) {
    return (x == x) && (x <= 3.402823466e38f) && (x >= -3.402823466e38f);
}

__device__ __forceinline__ float fp8_e5m2_load(__nv_fp8_storage_t v) {
    return __half2float(__nv_cvt_fp8_to_halfraw(v, __NV_E5M2));
}

__device__ __forceinline__ __nv_fp8_storage_t fp8_e5m2_store(float v) {
    if (!finite_f8e5s(v)) v = 0.0f;
    return __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E5M2);
}

// FP8 e5m2 i-major SoA fused collision + pull-streaming kernel.
// SM 8.9 required.
extern "C" __launch_bounds__(128, 4) __global__ void lbm_step_fp8_e5m2_soa_kernel(
    const __nv_fp8_storage_t* __restrict__ f_in,
    __nv_fp8_storage_t* __restrict__ f_out,
    float* __restrict__ rho_out,
    float* __restrict__ u_out,
    const float* __restrict__ tau,
    const float* __restrict__ force,
    int nx, int ny, int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_cells = nx * ny * nz;
    if (idx >= n_cells) return;

    int x = idx % nx;
    int y = (idx / nx) % ny;
    int z = idx / (nx * ny);

    float f_local[19];
    float rho_local = 0.0f;
    float mx = 0.0f, my = 0.0f, mz = 0.0f;

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        int xb = (x - CX_F8E5S[i] + nx) % nx;
        int yb = (y - CY_F8E5S[i] + ny) % ny;
        int zb = (z - CZ_F8E5S[i] + nz) % nz;
        int idx_back = xb + nx * (yb + ny * zb);
        float fi = fp8_e5m2_load(__ldg(&f_in[(long long)i * n_cells + idx_back]));
        if (!finite_f8e5s(fi)) fi = 0.0f;
        f_local[i] = fi;
        rho_local += fi;
        mx += (float)CX_F8E5S[i] * fi;
        my += (float)CY_F8E5S[i] * fi;
        mz += (float)CZ_F8E5S[i] * fi;
    }

    float ux = 0.0f, uy = 0.0f, uz = 0.0f;
    if (finite_f8e5s(rho_local) && rho_local > 1.0e-20f) {
        float inv_rho = 1.0f / rho_local;
        ux = mx * inv_rho;
        uy = my * inv_rho;
        uz = mz * inv_rho;
    } else {
        rho_local = 1.0f;
    }

    rho_out[idx] = rho_local;
    u_out[idx]               = ux;
    u_out[n_cells + idx]     = uy;
    u_out[2 * n_cells + idx] = uz;

    float tau_local = __ldg(&tau[idx]);
    float inv_tau   = 1.0f / tau_local;
    float u_sq      = ux * ux + uy * uy + uz * uz;
    float base      = fmaf(-1.5f, u_sq, 1.0f);

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        float eu   = fmaf((float)CX_F8E5S[i], ux, fmaf((float)CY_F8E5S[i], uy, (float)CZ_F8E5S[i] * uz));
        float w_rho = W_F8E5S[i] * rho_local;
        float f_eq  = w_rho * fmaf(fmaf(eu, 4.5f, 3.0f), eu, base);
        f_local[i] -= (f_local[i] - f_eq) * inv_tau;
    }

    float fx = __ldg(&force[idx]);
    float fy = __ldg(&force[n_cells + idx]);
    float fz = __ldg(&force[2 * n_cells + idx]);
    float force_mag_sq = fx * fx + fy * fy + fz * fz;

    if (force_mag_sq >= 1e-40f) {
        float prefactor = 1.0f - 0.5f * inv_tau;
        #pragma unroll
        for (int i = 0; i < 19; i++) {
            float eix = (float)CX_F8E5S[i], eiy = (float)CY_F8E5S[i], eiz = (float)CZ_F8E5S[i];
            float em_u_dot_f = (eix - ux) * fx + (eiy - uy) * fy + (eiz - uz) * fz;
            float ei_dot_u   = eix * ux + eiy * uy + eiz * uz;
            float ei_dot_f   = eix * fx + eiy * fy + eiz * fz;
            f_local[i] += prefactor * W_F8E5S[i] * (em_u_dot_f * 3.0f + ei_dot_u * ei_dot_f * 9.0f);
        }
    }

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        f_out[(long long)i * n_cells + idx] = fp8_e5m2_store(f_local[i]);
    }
}

extern "C" __global__ void initialize_uniform_fp8_e5m2_soa_kernel(
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
    u_out[idx]               = ux_init;
    u_out[n_cells + idx]     = uy_init;
    u_out[2 * n_cells + idx] = uz_init;
    tau[idx] = tau_val;

    float u_sq = ux_init*ux_init + uy_init*uy_init + uz_init*uz_init;
    float base = fmaf(-1.5f, u_sq, 1.0f);

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        float eu   = (float)CX_F8E5S[i]*ux_init + (float)CY_F8E5S[i]*uy_init + (float)CZ_F8E5S[i]*uz_init;
        float f_eq = W_F8E5S[i] * rho_init * fmaf(fmaf(eu, 4.5f, 3.0f), eu, base);
        f[(long long)i * n_cells + idx] = fp8_e5m2_store(f_eq);
    }
}

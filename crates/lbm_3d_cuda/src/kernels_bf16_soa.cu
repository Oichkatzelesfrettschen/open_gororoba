// BF16 i-major SoA D3Q19 LBM kernel -- pull-scheme streaming.
//
// BF16 vs FP16 SoA:
//   BF16 (bfloat16): 1 sign + 8 exponent + 7 mantissa bits.
//   FP16 (half):     1 sign + 5 exponent + 10 mantissa bits.
//   BF16 has the same range as FP32 (8-bit exponent), but only 7 bits mantissa
//   (~0.78% relative error vs FP16 0.098%). BF16 is less stable for high-Re
//   flows but never overflows for reasonable densities (unlike FP16 saturation
//   at f_i > 65504). Useful for flows with large dynamic range.
//
//   Memory layout: identical to kernels_fp16_soa.cu (2 bytes/dist, i-SoA).
//   Expected MLUPS: same as FP16 SoA at same grid (identical bandwidth profile).
//
// Minimum architecture: SM 8.0+ (Ampere). Ada SM 8.9 has native BF16 ops.
//   __nv_bfloat16 available in CUDA 11.0+.
//
// VRAM at 128^3: 19 * 2,097,152 * 2 * 2 (ping+pong) = ~152 MB.

#include <cuda_bf16.h>

__constant__ int CX_B16S[19] = {
    0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0
};
__constant__ int CY_B16S[19] = {
    0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1
};
__constant__ int CZ_B16S[19] = {
    0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1
};

__constant__ float W_B16S[19] = {
    1.0f/3.0f,
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
};

__device__ __forceinline__ bool finite_b16s(float x) {
    return (x == x) && (x <= 3.402823466e38f) && (x >= -3.402823466e38f);
}

// BF16 i-major SoA fused collision + pull-streaming kernel.
extern "C" __launch_bounds__(128, 4) __global__ void lbm_step_bf16_soa_kernel(
    const __nv_bfloat16* __restrict__ f_in,
    __nv_bfloat16* __restrict__ f_out,
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
        int xb = (x - CX_B16S[i] + nx) % nx;
        int yb = (y - CY_B16S[i] + ny) % ny;
        int zb = (z - CZ_B16S[i] + nz) % nz;
        int idx_back = xb + nx * (yb + ny * zb);
        float fi = __bfloat162float(__ldg(&f_in[(long long)i * n_cells + idx_back]));
        if (!finite_b16s(fi)) fi = 0.0f;
        f_local[i] = fi;
        rho_local += fi;
        mx += (float)CX_B16S[i] * fi;
        my += (float)CY_B16S[i] * fi;
        mz += (float)CZ_B16S[i] * fi;
    }

    float ux = 0.0f, uy = 0.0f, uz = 0.0f;
    if (finite_b16s(rho_local) && rho_local > 1.0e-20f) {
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
        float eu   = fmaf((float)CX_B16S[i], ux, fmaf((float)CY_B16S[i], uy, (float)CZ_B16S[i] * uz));
        float w_rho = W_B16S[i] * rho_local;
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
            float eix = (float)CX_B16S[i], eiy = (float)CY_B16S[i], eiz = (float)CZ_B16S[i];
            float em_u_dot_f = (eix - ux) * fx + (eiy - uy) * fy + (eiz - uz) * fz;
            float ei_dot_u   = eix * ux + eiy * uy + eiz * uz;
            float ei_dot_f   = eix * fx + eiy * fy + eiz * fz;
            f_local[i] += prefactor * W_B16S[i] * (em_u_dot_f * 3.0f + ei_dot_u * ei_dot_f * 9.0f);
        }
    }

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        f_out[(long long)i * n_cells + idx] = __float2bfloat16_rn(f_local[i]);
    }
}

extern "C" __global__ void initialize_uniform_bf16_soa_kernel(
    __nv_bfloat16* f,
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
        float eu   = (float)CX_B16S[i]*ux_init + (float)CY_B16S[i]*uy_init + (float)CZ_B16S[i]*uz_init;
        float f_eq = W_B16S[i] * rho_init * fmaf(fmaf(eu, 4.5f, 3.0f), eu, base);
        f[(long long)i * n_cells + idx] = __float2bfloat16_rn(f_eq);
    }
}

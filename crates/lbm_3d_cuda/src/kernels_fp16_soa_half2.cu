// FP16 i-major SoA D3Q19 LBM -- half2 ILP variant (2 cells per thread).
//
// WHY this variant vs kernels_fp16_soa.cu:
//   Ada SM 8.9 has dual-issue FP32 pipelines and native FP16 FMAD units.
//   Processing 2 cells per thread maximizes instruction-level parallelism:
//   - Two independent 19-element f_local arrays in registers
//   - BGK collision chains for cell A and cell B are interleaved -- the
//     compiler can dual-issue them on Ada's 2x FP32 pipeline
//   - Half the thread count = 50% reduction in index arithmetic overhead
//   - __half2 packs 2 FP16 values in one 32-bit register; __hadd2/__hmul2
//     compute 2 FP16 FMAs per instruction (2x throughput for velocity moments)
//
// DESIGN:
//   Thread k handles cells idx0=2k and idx1=2k+1.
//   Each direction i: load f_in[i*n_cells + back0] and f_in[i*n_cells + back1]
//   separately (different backward neighbors), promote to FP32, compute BGK.
//   Use __half2 for velocity moment accumulation phase:
//     __half2 eu2 = __hmul2(e2, u2);  // compute eu for both cells at once
//   Final BGK collision and store in FP32 to avoid __half2 rounding in collision.
//
// Grid: ceil(n_cells/2) / 128 blocks. Block: 128 threads.
// VRAM: identical to FP16 SoA (19 * n_cells * 2 * 2 = ~152 MB at 128^3).

#include <cuda_fp16.h>

__constant__ int CX_H2[19] = {
    0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0
};
__constant__ int CY_H2[19] = {
    0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1
};
__constant__ int CZ_H2[19] = {
    0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1
};

__constant__ float W_H2[19] = {
    1.0f/3.0f,
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
};

__device__ __forceinline__ bool finite_h2(float x) {
    return (x == x) && (x <= 3.402823466e38f) && (x >= -3.402823466e38f);
}

// FP16 SoA half2 ILP variant.
// Thread k: cell 0 = 2k, cell 1 = 2k+1.
// Uses __half2 for velocity moment accumulation; FP32 for collision and store.
extern "C" __launch_bounds__(128, 4) __global__ void lbm_step_fp16_soa_half2_kernel(
    const __half* __restrict__ f_in,   // [19 * n_cells] FP16, i-major SoA (ping)
    __half* __restrict__ f_out,        // [19 * n_cells] FP16, i-major SoA (pong)
    float* __restrict__ rho_out,
    float* __restrict__ u_out,         // [3 * n_cells] SoA
    const float* __restrict__ tau,
    const float* __restrict__ force,   // [3 * n_cells] SoA
    int nx, int ny, int nz
) {
    int n_cells = nx * ny * nz;
    int half_cells = (n_cells + 1) / 2;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= half_cells) return;

    int idx0 = 2 * k;
    int idx1 = 2 * k + 1;
    bool valid1 = (idx1 < n_cells);

    int x0 = idx0 % nx, y0 = (idx0 / nx) % ny, z0 = idx0 / (nx * ny);
    int x1 = 0, y1 = 0, z1 = 0;
    if (valid1) { x1 = idx1 % nx; y1 = (idx1 / nx) % ny; z1 = idx1 / (nx * ny); }

    // --- Load phase: 19 directions, 2 cells ---
    // FP32 buffers for BGK collision (precision required for stability)
    float fa[19], fb[19];
    // __half2 accumulator for velocity moments (2x throughput on Ada FP16 units)
    __half2 rho2  = __float2half2_rn(0.0f);
    __half2 mx2   = __float2half2_rn(0.0f);
    __half2 my2   = __float2half2_rn(0.0f);
    __half2 mz2   = __float2half2_rn(0.0f);

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        // Backward neighbors for both cells
        int xb0 = (x0 - CX_H2[i] + nx) % nx;
        int yb0 = (y0 - CY_H2[i] + ny) % ny;
        int zb0 = (z0 - CZ_H2[i] + nz) % nz;
        int back0 = xb0 + nx * (yb0 + ny * zb0);
        __half h0 = __ldg(&f_in[(long long)i * n_cells + back0]);
        float f0  = __half2float(h0);
        if (!finite_h2(f0)) f0 = 0.0f;
        fa[i] = f0;

        __half h1 = valid1 ? __ldg(&f_in[(long long)i * n_cells +
                      (((x1 - CX_H2[i] + nx) % nx) + nx * (((y1 - CY_H2[i] + ny) % ny) + ny * ((z1 - CZ_H2[i] + nz) % nz)))])
                           : __float2half(0.0f);
        float f1  = __half2float(h1);
        if (!finite_h2(f1)) f1 = 0.0f;
        fb[i] = f1;

        // Accumulate moments using __half2 (2 FP16 FMAs per instruction)
        __half2 fi2   = __halves2half2(h0, h1);
        rho2 = __hadd2(rho2, fi2);
        __half2 cx2   = __float2half2_rn((float)CX_H2[i]);
        __half2 cy2   = __float2half2_rn((float)CY_H2[i]);
        __half2 cz2   = __float2half2_rn((float)CZ_H2[i]);
        mx2 = __hadd2(mx2, __hmul2(cx2, fi2));
        my2 = __hadd2(my2, __hmul2(cy2, fi2));
        mz2 = __hadd2(mz2, __hmul2(cz2, fi2));
    }

    // Extract rho/u for cell 0 and cell 1 from __half2 accumulators
    float rho0 = __low2float(rho2);
    float rho1 = __high2float(rho2);
    float mx0  = __low2float(mx2),  my0 = __low2float(my2),  mz0 = __low2float(mz2);
    float mx1  = __high2float(mx2), my1 = __high2float(my2), mz1 = __high2float(mz2);

    // Cell 0: FP32 macroscopic + BGK
    float ux0 = 0.0f, uy0 = 0.0f, uz0 = 0.0f;
    if (finite_h2(rho0) && rho0 > 1.0e-20f) {
        float ir0 = 1.0f / rho0; ux0 = mx0 * ir0; uy0 = my0 * ir0; uz0 = mz0 * ir0;
    } else { rho0 = 1.0f; }
    rho_out[idx0] = rho0;
    u_out[idx0] = ux0; u_out[n_cells + idx0] = uy0; u_out[2 * n_cells + idx0] = uz0;

    float tau0 = __ldg(&tau[idx0]);
    float inv_tau0 = 1.0f / tau0;
    float usq0 = ux0*ux0 + uy0*uy0 + uz0*uz0;
    float base0 = fmaf(-1.5f, usq0, 1.0f);
    #pragma unroll
    for (int i = 0; i < 19; i++) {
        float eu = fmaf((float)CX_H2[i], ux0, fmaf((float)CY_H2[i], uy0, (float)CZ_H2[i] * uz0));
        float f_eq = W_H2[i] * rho0 * fmaf(fmaf(eu, 4.5f, 3.0f), eu, base0);
        fa[i] -= (fa[i] - f_eq) * inv_tau0;
    }

    // Cell 1: FP32 macroscopic + BGK
    float ux1 = 0.0f, uy1 = 0.0f, uz1 = 0.0f;
    if (valid1) {
        if (finite_h2(rho1) && rho1 > 1.0e-20f) {
            float ir1 = 1.0f / rho1; ux1 = mx1 * ir1; uy1 = my1 * ir1; uz1 = mz1 * ir1;
        } else { rho1 = 1.0f; }
        rho_out[idx1] = rho1;
        u_out[idx1] = ux1; u_out[n_cells + idx1] = uy1; u_out[2 * n_cells + idx1] = uz1;

        float tau1 = __ldg(&tau[idx1]);
        float inv_tau1 = 1.0f / tau1;
        float usq1 = ux1*ux1 + uy1*uy1 + uz1*uz1;
        float base1 = fmaf(-1.5f, usq1, 1.0f);
        #pragma unroll
        for (int i = 0; i < 19; i++) {
            float eu = fmaf((float)CX_H2[i], ux1, fmaf((float)CY_H2[i], uy1, (float)CZ_H2[i] * uz1));
            float f_eq = W_H2[i] * rho1 * fmaf(fmaf(eu, 4.5f, 3.0f), eu, base1);
            fb[i] -= (fb[i] - f_eq) * inv_tau1;
        }
    }

    // Store: coalesced writes for both cells per direction
    #pragma unroll
    for (int i = 0; i < 19; i++) {
        f_out[(long long)i * n_cells + idx0] = __float2half(fa[i]);
        if (valid1) f_out[(long long)i * n_cells + idx1] = __float2half(fb[i]);
    }
}

extern "C" __global__ void initialize_uniform_fp16_soa_half2_kernel(
    __half* f,
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
        float eu   = (float)CX_H2[i]*ux_init + (float)CY_H2[i]*uy_init + (float)CZ_H2[i]*uz_init;
        float f_eq = W_H2[i] * rho_init * fmaf(fmaf(eu, 4.5f, 3.0f), eu, base);
        f[(long long)i * n_cells + idx] = __float2half(f_eq);
    }
}

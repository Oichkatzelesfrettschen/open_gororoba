// FP16 i-major SoA D3Q19 LBM kernel -- pull-scheme streaming.
//
// WHY i-major SoA over AoS:
//   AoS push (kernels_fp16.cu): scatter-writes f_out[idx_next*20 + i] where
//   idx_next varies per direction. For diagonal directions, adjacent warp threads
//   scatter to non-adjacent cells -> non-coalesced writes, L2 thrash.
//   i-major SoA pull: writes f_out[i*n_cells + idx] -- all warp threads write to
//   consecutive addresses (stride-1 per lane i). Perfect coalescing.
//   Reads f_in[i*n_cells + idx_back] where idx_back = backward neighbor.
//   For x-aligned directions idx_back = idx-1: near-coalesced (31/32 threads).
//   For y/z-aligned directions: stride-nx / stride-nx*ny reads -- partially coalesced.
//   Net effect: writes are always coalesced; reads improve vs AoS scatter.
//   Expected: 1.5-2.5x MLUPS improvement over AoS push at 128^3.
//
// Layout:
//   f[i * n_cells + idx] -- __half elements, no padding (19*n_cells*2 bytes per buf).
//   Macroscopic (rho, u, tau, force): always FP32 SoA (same as kernels_soa.cu).
//
// Bandwidth model: D3Q19_SCALARS_NON_PADDED = 46 scalars (no padding, stride 19).
//   VRAM at 128^3: 19 * 2,097,152 * 2 * 2 (ping+pong) = ~152 MB.

#include <cuda_fp16.h>

// D3Q19 lattice velocities (suffixed _FS to avoid ODR conflicts).
__constant__ int CX_FS[19] = {
    0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0
};
__constant__ int CY_FS[19] = {
    0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1
};
__constant__ int CZ_FS[19] = {
    0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1
};

__constant__ float W_FS[19] = {
    1.0f/3.0f,
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
};

__device__ __forceinline__ bool finite_fs(float x) {
    return (x == x) && (x <= 3.402823466e38f) && (x >= -3.402823466e38f);
}

// FP16 i-major SoA fused collision + pull-streaming kernel.
// Block: 128 threads. Grid: ceil(n_cells / 128).
// __launch_bounds__(128, 4): same occupancy target as AoS FP16 kernel.
extern "C" __launch_bounds__(128, 4) __global__ void lbm_step_fp16_soa_kernel(
    const __half* __restrict__ f_in,   // [19 * n_cells] FP16, i-major SoA (ping)
    __half* __restrict__ f_out,        // [19 * n_cells] FP16, i-major SoA (pong)
    float* __restrict__ rho_out,       // [n_cells] FP32
    float* __restrict__ u_out,         // [3 * n_cells] FP32, SoA
    const float* __restrict__ tau,     // [n_cells] FP32
    const float* __restrict__ force,   // [3 * n_cells] FP32, SoA
    int nx, int ny, int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_cells = nx * ny * nz;
    if (idx >= n_cells) return;

    int x = idx % nx;
    int y = (idx / nx) % ny;
    int z = idx / (nx * ny);

    // Pull reads: for each direction i, read from backward neighbor
    // idx_back = (x - CX[i] + nx) % nx  +  nx * ((y - CY[i] + ny) % ny  +
    //              ny * ((z - CZ[i] + nz) % nz))
    float f_local[19];
    float rho_local = 0.0f;
    float mx = 0.0f, my = 0.0f, mz = 0.0f;

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        int xb = (x - CX_FS[i] + nx) % nx;
        int yb = (y - CY_FS[i] + ny) % ny;
        int zb = (z - CZ_FS[i] + nz) % nz;
        int idx_back = xb + nx * (yb + ny * zb);
        float fi = __half2float(__ldg(&f_in[(long long)i * n_cells + idx_back]));
        if (!finite_fs(fi)) fi = 0.0f;
        f_local[i] = fi;
        rho_local += fi;
        mx += (float)CX_FS[i] * fi;
        my += (float)CY_FS[i] * fi;
        mz += (float)CZ_FS[i] * fi;
    }

    float ux = 0.0f, uy = 0.0f, uz = 0.0f;
    if (finite_fs(rho_local) && rho_local > 1.0e-20f) {
        float inv_rho = 1.0f / rho_local;
        ux = mx * inv_rho;
        uy = my * inv_rho;
        uz = mz * inv_rho;
    } else {
        rho_local = 1.0f;
    }

    rho_out[idx] = rho_local;
    u_out[idx]                = ux;
    u_out[n_cells + idx]      = uy;
    u_out[2 * n_cells + idx]  = uz;

    // BGK collision (FP32)
    float tau_local = __ldg(&tau[idx]);
    float inv_tau   = 1.0f / tau_local;
    float u_sq      = ux * ux + uy * uy + uz * uz;
    float base      = fmaf(-1.5f, u_sq, 1.0f);

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        float eu    = fmaf((float)CX_FS[i], ux, fmaf((float)CY_FS[i], uy, (float)CZ_FS[i] * uz));
        float w_rho = W_FS[i] * rho_local;
        float f_eq  = w_rho * fmaf(fmaf(eu, 4.5f, 3.0f), eu, base);
        f_local[i] -= (f_local[i] - f_eq) * inv_tau;
    }

    // Guo forcing (FP32, SoA force layout)
    float fx = __ldg(&force[idx]);
    float fy = __ldg(&force[n_cells + idx]);
    float fz = __ldg(&force[2 * n_cells + idx]);
    float force_mag_sq = fx * fx + fy * fy + fz * fz;

    if (force_mag_sq >= 1e-40f) {
        float prefactor = 1.0f - 0.5f * inv_tau;
        #pragma unroll
        for (int i = 0; i < 19; i++) {
            float eix = (float)CX_FS[i], eiy = (float)CY_FS[i], eiz = (float)CZ_FS[i];
            float em_u_dot_f = (eix - ux) * fx + (eiy - uy) * fy + (eiz - uz) * fz;
            float ei_dot_u   = eix * ux + eiy * uy + eiz * uz;
            float ei_dot_f   = eix * fx + eiy * fy + eiz * fz;
            f_local[i] += prefactor * W_FS[i] * (em_u_dot_f * 3.0f + ei_dot_u * ei_dot_f * 9.0f);
        }
    }

    // Coalesced write: all warp threads write f_out[i*n_cells + idx] for same i.
    // Since warp threads have consecutive idx, this is a stride-1 write per lane.
    #pragma unroll
    for (int i = 0; i < 19; i++) {
        f_out[(long long)i * n_cells + idx] = __float2half(f_local[i]);
    }
}

extern "C" __global__ void initialize_uniform_fp16_soa_kernel(
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
        float eu   = (float)CX_FS[i]*ux_init + (float)CY_FS[i]*uy_init + (float)CZ_FS[i]*uz_init;
        float f_eq = W_FS[i] * rho_init * fmaf(fmaf(eu, 4.5f, 3.0f), eu, base);
        // i-major SoA: f[i * n_cells + idx]
        f[(long long)i * n_cells + idx] = __float2half(f_eq);
    }
}

// FP32 i-major SoA D3Q19 LBM -- Ada cache-streaming store variant.
//
// WHY this variant exists:
//   The baseline FP32 SoA pull kernel (kernels_soa.cu standard variant) uses
//   default load/store cache policy: reads and writes both target the unified
//   L1+L2 hierarchy. This means pong writes evict hot ping data from L1/L2
//   before the next timestep reuses it -- even though pong data won't be
//   read until the FOLLOWING timestep.
//
//   Ada SM 8.9 offers two cache-policy controls relevant here:
//
//   1. __ldg() on ping reads: marks loads as read-only (const-qualified path
//      through L1 texture cache on pre-Ada; on Ada unified L1, effectively the
//      same as .ca but signals compiler to use non-coherent path).
//
//   2. __stcs() (store streaming): PTX st.global.cs -- writes bypass L1 data
//      cache and are inserted into L2 with EVICT-FIRST policy. Next L2 access
//      to the same line pushes it to DRAM before the streaming store evicts it.
//      NET EFFECT: pong writes do NOT displace ping reads from L2.
//
//   Combined: ping reads stay resident in L2 (L1 used for coefficients via
//   __ldg on tau/force/W/CX). Pong writes stream through L2 without evicting
//   ping data. At 128^3, the FP32 SoA ping buffer is 152 MB -- too large for
//   the 48 MB L2, so full residency is impossible. But the EVICT-FIRST policy
//   means pong writes don't PREMATURELY evict the most recently loaded ping
//   sectors, which helps especially for the sequential access pattern of
//   x-aligned directions.
//
//   Expected improvement: 5-15% over baseline FP32 SoA pull at 128^3+.
//   Effect is larger for FP8/INT8 SoA (76 MB buffer fits in 48 MB L2).
//
// This kernel is otherwise IDENTICAL to kernels_soa.cu pull variant.

__constant__ int CX_CS[19] = {
    0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0
};
__constant__ int CY_CS[19] = {
    0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1
};
__constant__ int CZ_CS[19] = {
    0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1
};

__constant__ float W_CS[19] = {
    1.0f/3.0f,
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
};

__device__ __forceinline__ bool finite_cs(float x) {
    return (x == x) && (x <= 3.402823466e38f) && (x >= -3.402823466e38f);
}

// FP32 i-major SoA pull-scheme with streaming-store writes.
// Ping reads:  __ldg() -- read-only cache path.
// Pong writes: __stcs() -- streaming store, L2 evict-first (does not pollute
//              L2 with cold pong data that won't be read until next timestep).
extern "C" __launch_bounds__(128, 4) __global__ void lbm_step_fp32_soa_cs_kernel(
    const float* __restrict__ f_in,    // [19 * n_cells] FP32, i-major SoA (ping)
    float* __restrict__ f_out,         // [19 * n_cells] FP32, i-major SoA (pong)
    float* __restrict__ rho_out,
    float* __restrict__ u_out,         // [3 * n_cells] SoA
    const float* __restrict__ tau,
    const float* __restrict__ force,   // [3 * n_cells] SoA
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

    // Pull reads via __ldg (read-only cache path; L1 not polluted by stores).
    #pragma unroll
    for (int i = 0; i < 19; i++) {
        int xb = (x - CX_CS[i] + nx) % nx;
        int yb = (y - CY_CS[i] + ny) % ny;
        int zb = (z - CZ_CS[i] + nz) % nz;
        int idx_back = xb + nx * (yb + ny * zb);
        float fi = __ldg(&f_in[(long long)i * n_cells + idx_back]);
        if (!finite_cs(fi)) fi = 0.0f;
        f_local[i] = fi;
        rho_local += fi;
        mx += (float)CX_CS[i] * fi;
        my += (float)CY_CS[i] * fi;
        mz += (float)CZ_CS[i] * fi;
    }

    float ux = 0.0f, uy = 0.0f, uz = 0.0f;
    if (finite_cs(rho_local) && rho_local > 1.0e-20f) {
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
        float eu   = fmaf((float)CX_CS[i], ux, fmaf((float)CY_CS[i], uy, (float)CZ_CS[i] * uz));
        float w_rho = W_CS[i] * rho_local;
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
            float eix = (float)CX_CS[i], eiy = (float)CY_CS[i], eiz = (float)CZ_CS[i];
            float em_u_dot_f = (eix - ux) * fx + (eiy - uy) * fy + (eiz - uz) * fz;
            float ei_dot_u   = eix * ux + eiy * uy + eiz * uz;
            float ei_dot_f   = eix * fx + eiy * fy + eiz * fz;
            f_local[i] += prefactor * W_CS[i] * (em_u_dot_f * 3.0f + ei_dot_u * ei_dot_f * 9.0f);
        }
    }

    // Coalesced streaming-store writes: L2 evict-first, does not pollute L1.
    // PTX: st.global.cs.f32 [addr], val
    #pragma unroll
    for (int i = 0; i < 19; i++) {
        __stcs(&f_out[(long long)i * n_cells + idx], f_local[i]);
    }
}

extern "C" __global__ void initialize_uniform_fp32_soa_cs_kernel(
    float* f,
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
        float eu   = (float)CX_CS[i]*ux_init + (float)CY_CS[i]*uy_init + (float)CZ_CS[i]*uz_init;
        float f_eq = W_CS[i] * rho_init * fmaf(fmaf(eu, 4.5f, 3.0f), eu, base);
        __stcs(&f[(long long)i * n_cells + idx], f_eq);
    }
}

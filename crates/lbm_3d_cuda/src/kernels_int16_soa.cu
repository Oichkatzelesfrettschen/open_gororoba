// INT16 i-major SoA D3Q19 LBM kernel -- pull-scheme streaming.
//
// WHY INT16 SoA over INT16 AoS:
//   Same coalescing argument as INT8 SoA vs AoS (see kernels_int8_soa.cu).
//   SoA pull writes f_out[i*n_cells + idx] -- coalesced for all 19 directions.
//   AoS push scatters to f_out[idx_next*20+i] -- non-coalesced for diagonals.
//
// WHY INT16 SoA vs FP16 SoA:
//   Same bytes/dist (2 bytes), same VRAM. Differences:
//   - INT16 uses integer arithmetic for momentm accumulation (no half2float overhead).
//   - FP16 has hardware FP16 FMAs; INT16 uses FP32 for collision.
//   - INT16 DIST_SCALE=16384 allows rho up to ~1.9999 with finer quantization
//     (LSB=6.1e-5) vs FP16 (mantissa=10 bits, relative error ~0.1%).
//   - Both use 2 bytes/dist; expected MLUPS should be similar.
//
// DIST_SCALE = 16384: range [-32768, 32767] -> f_i in [-2.000, 1.9999].
// Buffer: 19 * n_cells * 2 bytes per buffer.
// VRAM at 128^3: 19 * 2,097,152 * 2 * 2 (ping+pong) = ~152 MB (no padding, 5% less than AoS).

#define DIST_SCALE_I16S 16384.0f
#define INV_DIST_SCALE_I16S (1.0f / 16384.0f)

__constant__ int CX_I16S[19] = {
    0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0
};
__constant__ int CY_I16S[19] = {
    0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1
};
__constant__ int CZ_I16S[19] = {
    0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1
};

__constant__ float W_I16S[19] = {
    1.0f/3.0f,
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
};

__device__ __forceinline__ bool finite_i16s(float x) {
    return (x == x) && (x <= 3.402823466e38f) && (x >= -3.402823466e38f);
}

__device__ __forceinline__ short float_to_i16s(float v) {
    if (!finite_i16s(v)) return (short)0;
    float s = v * DIST_SCALE_I16S;
    if (s >  32767.0f) s =  32767.0f;
    if (s < -32768.0f) s = -32768.0f;
    return (short)(int)s;
}

// INT16 i-major SoA pull-scheme kernel.
extern "C" __launch_bounds__(128, 4) __global__ void lbm_step_int16_soa_kernel(
    const short* __restrict__ f_in,    // [19 * n_cells] INT16, i-major SoA
    short* __restrict__ f_out,         // [19 * n_cells] INT16, i-major SoA
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
    int mx_i32 = 0, my_i32 = 0, mz_i32 = 0;

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        int xb = (x - CX_I16S[i] + nx) % nx;
        int yb = (y - CY_I16S[i] + ny) % ny;
        int zb = (z - CZ_I16S[i] + nz) % nz;
        int idx_back = xb + nx * (yb + ny * zb);
        // Read once into register; derive both FP32 and integer quantities.
        short raw = __ldg(&f_in[(long long)i * n_cells + idx_back]);
        float fi = (float)raw * INV_DIST_SCALE_I16S;
        if (!finite_i16s(fi)) { fi = 0.0f; raw = 0; }
        f_local[i] = fi;
        rho_local  += fi;
        mx_i32     += CX_I16S[i] * (int)raw;
        my_i32     += CY_I16S[i] * (int)raw;
        mz_i32     += CZ_I16S[i] * (int)raw;
    }

    // Integer momentum (in DIST_SCALE units) -> FP32 velocity.
    float ux = 0.0f, uy = 0.0f, uz = 0.0f;
    if (finite_i16s(rho_local) && rho_local > 1.0e-20f) {
        float inv_rho_scaled = INV_DIST_SCALE_I16S / rho_local;
        ux = (float)mx_i32 * inv_rho_scaled;
        uy = (float)my_i32 * inv_rho_scaled;
        uz = (float)mz_i32 * inv_rho_scaled;
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
        float eu   = fmaf((float)CX_I16S[i], ux, fmaf((float)CY_I16S[i], uy, (float)CZ_I16S[i] * uz));
        float w_rho = W_I16S[i] * rho_local;
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
            float eix = (float)CX_I16S[i], eiy = (float)CY_I16S[i], eiz = (float)CZ_I16S[i];
            float em_u_dot_f = (eix - ux) * fx + (eiy - uy) * fy + (eiz - uz) * fz;
            float ei_dot_u   = eix * ux + eiy * uy + eiz * uz;
            float ei_dot_f   = eix * fx + eiy * fy + eiz * fz;
            f_local[i] += prefactor * W_I16S[i] * (em_u_dot_f * 3.0f + ei_dot_u * ei_dot_f * 9.0f);
        }
    }

    // Coalesced write: all warp threads write consecutive f_out[i*n_cells + idx]
    #pragma unroll
    for (int i = 0; i < 19; i++) {
        f_out[(long long)i * n_cells + idx] = float_to_i16s(f_local[i]);
    }
}

extern "C" __global__ void initialize_uniform_int16_soa_kernel(
    short* f,
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
        float eu   = (float)CX_I16S[i]*ux_init + (float)CY_I16S[i]*uy_init + (float)CZ_I16S[i]*uz_init;
        float f_eq = W_I16S[i] * rho_init * fmaf(fmaf(eu, 4.5f, 3.0f), eu, base);
        f[(long long)i * n_cells + idx] = float_to_i16s(f_eq);
    }
}

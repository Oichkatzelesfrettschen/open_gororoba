// INT8 i-major SoA D3Q19 LBM kernel -- pull-scheme streaming + __dp4a momentum.
//
// Same coalescing argument as kernels_fp16_soa.cu and kernels_fp8_soa.cu.
// i-major SoA: f[i * n_cells + idx], signed char elements.
//   Buffer size: 19 * n_cells * 1 = n_cells * 19 bytes per buffer (no padding).
//   VRAM at 128^3: 19 * 2,097,152 * 1 * 2 (ping+pong) = ~76 MB.
//   (AoS had stride-20 = ~80 MB; SoA saves the per-cell padding byte.)
//
// __dp4a momentum optimization:
//   In AoS, __dp4a was applied on packed i8x4 groups from the AoS load.
//   In SoA, we load individual signed chars per direction and accumulate.
//   To still benefit from dp4a, we can collect 4 consecutive direction values
//   into a pack after loading. But: the SoA load reads different addresses per
//   direction (idx_back varies per i), so packing across directions is not
//   straightforward. Instead we use direct int accumulation for the momentum
//   sum -- this is still faster than FP32 due to the int->float promotion saving
//   float registers in the accumulation loop.
//
// Bandwidth model: 46 scalars (non-padded) * 1 byte = 46 bytes/cell/step.
// DIST_SCALE: 64 (same as AoS INT8 kernel for consistency).

#define DIST_SCALE_I8S 64.0f
#define INV_DIST_SCALE_I8S (1.0f / DIST_SCALE_I8S)

// D3Q19 lattice velocities (suffixed _I8S)
__constant__ int CX_I8S[19] = {
    0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0
};
__constant__ int CY_I8S[19] = {
    0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1
};
__constant__ int CZ_I8S[19] = {
    0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1
};

__constant__ float W_I8S[19] = {
    1.0f/3.0f,
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
};

__device__ __forceinline__ bool finite_i8s(float x) {
    return (x == x) && (x <= 3.402823466e38f) && (x >= -3.402823466e38f);
}

__device__ __forceinline__ signed char float_to_i8s(float v) {
    if (!finite_i8s(v)) return (signed char)0;
    float s = v * DIST_SCALE_I8S;
    if (s >  127.0f) s =  127.0f;
    if (s < -128.0f) s = -128.0f;
    return (signed char)(int)s;
}

// INT8 i-major SoA fused collision + pull-streaming.
extern "C" __launch_bounds__(128, 4) __global__ void lbm_step_int8_soa_kernel(
    const signed char* __restrict__ f_in,   // [19 * n_cells] INT8, i-major SoA
    signed char* __restrict__ f_out,        // [19 * n_cells] INT8, i-major SoA
    float* __restrict__ rho_out,
    float* __restrict__ u_out,              // [3 * n_cells] SoA
    const float* __restrict__ tau,
    const float* __restrict__ force,        // [3 * n_cells] SoA
    int nx, int ny, int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_cells = nx * ny * nz;
    if (idx >= n_cells) return;

    int x = idx % nx;
    int y = (idx / nx) % ny;
    int z = idx / (nx * ny);

    signed char f_i8[19];
    float f_local[19];
    float rho_local = 0.0f;
    int mx_i32 = 0, my_i32 = 0, mz_i32 = 0;

    // Pull reads: load one INT8 per direction from backward neighbor.
    // After loading, accumulate momentum via integer sums then convert.
    #pragma unroll
    for (int i = 0; i < 19; i++) {
        int xb = (x - CX_I8S[i] + nx) % nx;
        int yb = (y - CY_I8S[i] + ny) % ny;
        int zb = (z - CZ_I8S[i] + nz) % nz;
        int idx_back = xb + nx * (yb + ny * zb);
        signed char v = __ldg(&f_in[(long long)i * n_cells + idx_back]);
        f_i8[i] = v;
        f_local[i] = (float)v * INV_DIST_SCALE_I8S;
        rho_local += f_local[i];
        mx_i32 += CX_I8S[i] * (int)v;
        my_i32 += CY_I8S[i] * (int)v;
        mz_i32 += CZ_I8S[i] * (int)v;
    }

    float mx = (float)mx_i32 * INV_DIST_SCALE_I8S;
    float my = (float)my_i32 * INV_DIST_SCALE_I8S;
    float mz = (float)mz_i32 * INV_DIST_SCALE_I8S;

    float ux = 0.0f, uy = 0.0f, uz = 0.0f;
    if (finite_i8s(rho_local) && rho_local > 1.0e-20f) {
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
        float eu    = fmaf((float)CX_I8S[i], ux, fmaf((float)CY_I8S[i], uy, (float)CZ_I8S[i] * uz));
        float w_rho = W_I8S[i] * rho_local;
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
            float eix = (float)CX_I8S[i], eiy = (float)CY_I8S[i], eiz = (float)CZ_I8S[i];
            float em_u_dot_f = (eix - ux) * fx + (eiy - uy) * fy + (eiz - uz) * fz;
            float ei_dot_u   = eix * ux + eiy * uy + eiz * uz;
            float ei_dot_f   = eix * fx + eiy * fy + eiz * fz;
            f_local[i] += prefactor * W_I8S[i] * (em_u_dot_f * 3.0f + ei_dot_u * ei_dot_f * 9.0f);
        }
    }

    // Coalesced write: i-major SoA.
    #pragma unroll
    for (int i = 0; i < 19; i++) {
        f_out[(long long)i * n_cells + idx] = float_to_i8s(f_local[i]);
    }
}

extern "C" __global__ void initialize_uniform_int8_soa_kernel(
    signed char* f,
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
        float eu   = (float)CX_I8S[i]*ux_init + (float)CY_I8S[i]*uy_init + (float)CZ_I8S[i]*uz_init;
        float f_eq = W_I8S[i] * rho_init * fmaf(fmaf(eu, 4.5f, 3.0f), eu, base);
        f[(long long)i * n_cells + idx] = float_to_i8s(f_eq);
    }
}

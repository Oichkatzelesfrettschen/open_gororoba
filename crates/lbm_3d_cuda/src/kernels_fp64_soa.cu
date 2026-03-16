// FP64 i-major SoA D3Q19 LBM kernel -- pull-scheme streaming.
//
// WHY FP64 SoA over FP64 AoS:
//   Same coalescing argument as all SoA variants. FP64 AoS writes scatter
//   8-byte values to non-consecutive addresses for diagonal directions,
//   generating 32+ L2 transactions per warp. SoA pull writes f_out[i*N+idx]
//   with consecutive idx across warp lanes = 1 L2 transaction per direction.
//   At 128^3 the improvement is limited by FP64 compute (gaming GPU has ~1/64
//   FP64 throughput vs FP32), but coalescing still matters at smaller grids.
//
// Primary use: numerical validation. FP64 results serve as reference for
//   checking BF16/FP16/FP8 spectral artifacts and physics stability.
//
// Performance note: RTX 4070 Ti has ~1/64 FP64 throughput (~0.3 TFLOPS vs
//   ~20 TFLOPS FP32). Expect ~5-10x slower than FP32 SoA even with coalescing.
//   True compute-bound, not bandwidth-bound, at all grid sizes tested.
//
// Buffer: 19 * n_cells * 8 bytes per buffer.
// VRAM at 128^3: 19 * 2,097,152 * 8 * 2 (ping+pong) = ~609 MB.

__constant__ int CX_F64S[19] = {
    0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0
};
__constant__ int CY_F64S[19] = {
    0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1
};
__constant__ int CZ_F64S[19] = {
    0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1
};

__constant__ double W_F64S[19] = {
    1.0/3.0,
    1.0/18.0, 1.0/18.0, 1.0/18.0,
    1.0/18.0, 1.0/18.0, 1.0/18.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0
};

__device__ __forceinline__ bool finite_f64s(double x) {
    return (x == x) && (x <= 1.7976931348623157e308) && (x >= -1.7976931348623157e308);
}

// FP64 i-major SoA fused collision + pull-streaming.
// One thread per cell; 128 threads/block (FP64 register pressure limits occupancy).
extern "C" __global__ void lbm_step_fp64_soa_kernel(
    const double* __restrict__ f_in,   // [19 * n_cells] FP64, i-major SoA
    double* __restrict__ f_out,        // [19 * n_cells] FP64, i-major SoA
    float* __restrict__ rho_out,       // [n_cells] FP32 (output for post-processing)
    float* __restrict__ u_out,         // [3 * n_cells] FP32 SoA
    const float* __restrict__ tau,     // [n_cells] FP32
    const float* __restrict__ force,   // [3 * n_cells] FP32 SoA
    int nx, int ny, int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_cells = nx * ny * nz;
    if (idx >= n_cells) return;

    int x = idx % nx;
    int y = (idx / nx) % ny;
    int z = idx / (nx * ny);

    double f_local[19];
    double rho_local = 0.0;
    double mx = 0.0, my = 0.0, mz = 0.0;

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        int xb = (x - CX_F64S[i] + nx) % nx;
        int yb = (y - CY_F64S[i] + ny) % ny;
        int zb = (z - CZ_F64S[i] + nz) % nz;
        int idx_back = xb + nx * (yb + ny * zb);
        double fi = __ldg(&f_in[(long long)i * n_cells + idx_back]);
        if (!finite_f64s(fi)) fi = 0.0;
        f_local[i] = fi;
        rho_local += fi;
        mx += (double)CX_F64S[i] * fi;
        my += (double)CY_F64S[i] * fi;
        mz += (double)CZ_F64S[i] * fi;
    }

    double ux = 0.0, uy = 0.0, uz = 0.0;
    if (finite_f64s(rho_local) && rho_local > 1.0e-20) {
        double inv_rho = 1.0 / rho_local;
        ux = mx * inv_rho;
        uy = my * inv_rho;
        uz = mz * inv_rho;
    } else {
        rho_local = 1.0;
    }

    rho_out[idx] = (float)rho_local;
    u_out[idx]               = (float)ux;
    u_out[n_cells + idx]     = (float)uy;
    u_out[2 * n_cells + idx] = (float)uz;

    double tau_local = (double)__ldg(&tau[idx]);
    double inv_tau   = 1.0 / tau_local;
    double u_sq      = ux * ux + uy * uy + uz * uz;
    double base      = fma(-1.5, u_sq, 1.0);

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        double eu   = fma((double)CX_F64S[i], ux, fma((double)CY_F64S[i], uy, (double)CZ_F64S[i] * uz));
        double w_rho = W_F64S[i] * rho_local;
        double f_eq  = w_rho * fma(fma(eu, 4.5, 3.0), eu, base);
        f_local[i] -= (f_local[i] - f_eq) * inv_tau;
    }

    double fx = (double)__ldg(&force[idx]);
    double fy = (double)__ldg(&force[n_cells + idx]);
    double fz = (double)__ldg(&force[2 * n_cells + idx]);
    double force_mag_sq = fx * fx + fy * fy + fz * fz;

    if (force_mag_sq >= 1e-80) {
        double prefactor = 1.0 - 0.5 * inv_tau;
        #pragma unroll
        for (int i = 0; i < 19; i++) {
            double eix = (double)CX_F64S[i], eiy = (double)CY_F64S[i], eiz = (double)CZ_F64S[i];
            double em_u_dot_f = (eix - ux) * fx + (eiy - uy) * fy + (eiz - uz) * fz;
            double ei_dot_u   = eix * ux + eiy * uy + eiz * uz;
            double ei_dot_f   = eix * fx + eiy * fy + eiz * fz;
            f_local[i] += prefactor * W_F64S[i] * (em_u_dot_f * 3.0 + ei_dot_u * ei_dot_f * 9.0);
        }
    }

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        f_out[(long long)i * n_cells + idx] = f_local[i];
    }
}

extern "C" __global__ void initialize_uniform_fp64_soa_kernel(
    double* f,
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

    double d_rho = (double)rho_init;
    double d_ux  = (double)ux_init;
    double d_uy  = (double)uy_init;
    double d_uz  = (double)uz_init;
    double u_sq  = d_ux*d_ux + d_uy*d_uy + d_uz*d_uz;
    double base  = fma(-1.5, u_sq, 1.0);

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        double eu   = (double)CX_F64S[i]*d_ux + (double)CY_F64S[i]*d_uy + (double)CZ_F64S[i]*d_uz;
        double f_eq = W_F64S[i] * d_rho * fma(fma(eu, 4.5, 3.0), eu, base);
        f[(long long)i * n_cells + idx] = f_eq;
    }
}

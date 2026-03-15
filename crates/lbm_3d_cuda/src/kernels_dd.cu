// Double-Double (FP128 emulation) D3Q19 LBM kernel.
// Storage: two double per distribution value (hi + lo), ~106-bit mantissa.
// Layout: i-major SoA -- f_hi[19 * n_cells], f_lo[19 * n_cells].
//   Index: f_hi[i * n_cells + idx].  Within distribution lane i, threads
//   (idx, idx+1, ..., idx+31) access consecutive doubles -- fully coalesced.
//   Scatter writes f_hi_out[i * n_cells + idx_next] are coalesced for
//   interior cells (adjacent threads stream to adjacent x-neighbors).
// Compute: full double-double arithmetic (Knuth 2-sum, Veltkamp/Dekker FMA).
// Element size: 16 bytes/distribution (2 * double).
// WHY: Validates whether FP64 accumulation artifacts in long simulations are
//   significant. On Ada gaming SKUs, FP64 is ~1/64 speed of FP32; DD is
//   another ~16-32x slower than FP64 due to the extra addition chains.
//   Expected MLUPS: 0.01-0.1 (vs ~100 for FP32).
// Primary use: reference accuracy comparison for C-1365 class null-result claims.

// D3Q19 lattice (double suffix _DD to avoid ODR conflicts)
__constant__ int D3Q19_CX_DD[19] = {
    0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0
};
__constant__ int D3Q19_CY_DD[19] = {
    0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1
};
__constant__ int D3Q19_CZ_DD[19] = {
    0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1
};

__constant__ double D3Q19_WD_DD[19] = {
    1.0/3.0,
    1.0/18.0, 1.0/18.0, 1.0/18.0,
    1.0/18.0, 1.0/18.0, 1.0/18.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0
};

// ============================================================================
// Double-Double arithmetic primitives
// ============================================================================

// Knuth 2-sum: error-free addition of two doubles.
// Returns exact sum in (s, e) with |e| <= eps * |s|.
__device__ __forceinline__ void two_sum(double a, double b, double* s, double* e) {
    double x = a + b;
    double bp = x - a;
    double ap = x - bp;
    *s = x;
    *e = (a - ap) + (b - bp);
}

// Veltkamp/Dekker: error-free product of two doubles via FMA.
// fma(a, b, -p) is exact when p = a*b (the FMA computes a*b exactly).
__device__ __forceinline__ void two_prod(double a, double b, double* p, double* e) {
    *p = a * b;
    *e = fma(a, b, -(*p));
}

// DD addition: (a_hi + a_lo) + (b_hi + b_lo) -> (s_hi + s_lo)
// Accurate to ~2^-106 relative error.
__device__ __forceinline__ void dd_add(
    double a_hi, double a_lo,
    double b_hi, double b_lo,
    double* s_hi, double* s_lo
) {
    double sh, sl;
    two_sum(a_hi, b_hi, &sh, &sl);
    sl += a_lo + b_lo;
    two_sum(sh, sl, s_hi, s_lo);
}

// DD subtraction: (a_hi + a_lo) - (b_hi + b_lo)
__device__ __forceinline__ void dd_sub(
    double a_hi, double a_lo,
    double b_hi, double b_lo,
    double* s_hi, double* s_lo
) {
    dd_add(a_hi, a_lo, -b_hi, -b_lo, s_hi, s_lo);
}

// DD multiplication: (a_hi + a_lo) * (b_hi + b_lo) -> (p_hi + p_lo)
__device__ __forceinline__ void dd_mul(
    double a_hi, double a_lo,
    double b_hi, double b_lo,
    double* p_hi, double* p_lo
) {
    double p, e;
    two_prod(a_hi, b_hi, &p, &e);
    // Cross terms: a_hi*b_lo + a_lo*b_hi (lower order; a_lo*b_lo is negligible)
    *p_lo = e + (a_hi * b_lo + a_lo * b_hi);
    *p_hi = p;
}

// DD division: (a_hi + a_lo) / (b_hi + b_lo) via Newton-Raphson.
__device__ __forceinline__ void dd_div(
    double a_hi, double a_lo,
    double b_hi, double b_lo,
    double* q_hi, double* q_lo
) {
    // Initial estimate: q ~= a_hi / b_hi
    double q0 = a_hi / b_hi;
    // Residual: a - q0 * b
    double r_hi, r_lo;
    dd_mul(q0, 0.0, b_hi, b_lo, &r_hi, &r_lo);
    dd_sub(a_hi, a_lo, r_hi, r_lo, &r_hi, &r_lo);
    // Correction: dq = r / b ~= r_hi / b_hi
    double dq = r_hi / b_hi;
    *q_hi = q0 + dq;
    *q_lo = dq - ((*q_hi) - q0);
}

// ============================================================================
// DD LBM equilibrium
// ============================================================================

// Compute equilibrium using full DD arithmetic.
// f_eq_i = w_i * rho * ((4.5*eu + 3)*eu + (1 - 1.5*usq))
// All multiplications and additions are DD-accurate.
__device__ void compute_equilibrium_dd(
    double* f_eq_hi, double* f_eq_lo,
    double rho_hi, double rho_lo,
    double ux_hi, double ux_lo,
    double uy_hi, double uy_lo,
    double uz_hi, double uz_lo
) {
    // u_sq = ux^2 + uy^2 + uz^2  (DD)
    double ux2_hi, ux2_lo, uy2_hi, uy2_lo, uz2_hi, uz2_lo;
    dd_mul(ux_hi, ux_lo, ux_hi, ux_lo, &ux2_hi, &ux2_lo);
    dd_mul(uy_hi, uy_lo, uy_hi, uy_lo, &uy2_hi, &uy2_lo);
    dd_mul(uz_hi, uz_lo, uz_hi, uz_lo, &uz2_hi, &uz2_lo);
    double usq_hi, usq_lo;
    dd_add(ux2_hi, ux2_lo, uy2_hi, uy2_lo, &usq_hi, &usq_lo);
    dd_add(usq_hi, usq_lo, uz2_hi, uz2_lo, &usq_hi, &usq_lo);

    // base = 1 - 1.5 * usq
    double base_hi, base_lo;
    dd_mul(1.5, 0.0, usq_hi, usq_lo, &base_hi, &base_lo);
    dd_sub(1.0, 0.0, base_hi, base_lo, &base_hi, &base_lo);

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        // eu = cx*ux + cy*uy + cz*uz  (DD; cx in {-1,0,1} -> no dd_mul needed for cx)
        double eu_hi = 0.0, eu_lo = 0.0;
        if (D3Q19_CX_DD[i] != 0) {
            double t_hi = (D3Q19_CX_DD[i] == 1) ? ux_hi : -ux_hi;
            double t_lo = (D3Q19_CX_DD[i] == 1) ? ux_lo : -ux_lo;
            dd_add(eu_hi, eu_lo, t_hi, t_lo, &eu_hi, &eu_lo);
        }
        if (D3Q19_CY_DD[i] != 0) {
            double t_hi = (D3Q19_CY_DD[i] == 1) ? uy_hi : -uy_hi;
            double t_lo = (D3Q19_CY_DD[i] == 1) ? uy_lo : -uy_lo;
            dd_add(eu_hi, eu_lo, t_hi, t_lo, &eu_hi, &eu_lo);
        }
        if (D3Q19_CZ_DD[i] != 0) {
            double t_hi = (D3Q19_CZ_DD[i] == 1) ? uz_hi : -uz_hi;
            double t_lo = (D3Q19_CZ_DD[i] == 1) ? uz_lo : -uz_lo;
            dd_add(eu_hi, eu_lo, t_hi, t_lo, &eu_hi, &eu_lo);
        }

        // Horner: (4.5*eu + 3)*eu + base
        double t1_hi, t1_lo;
        dd_mul(4.5, 0.0, eu_hi, eu_lo, &t1_hi, &t1_lo);   // 4.5*eu
        dd_add(t1_hi, t1_lo, 3.0, 0.0, &t1_hi, &t1_lo);   // 4.5*eu + 3
        dd_mul(t1_hi, t1_lo, eu_hi, eu_lo, &t1_hi, &t1_lo); // (4.5*eu+3)*eu
        dd_add(t1_hi, t1_lo, base_hi, base_lo, &t1_hi, &t1_lo); // + base

        // f_eq = w * rho * t1
        double wr_hi, wr_lo;
        dd_mul(D3Q19_WD_DD[i], 0.0, rho_hi, rho_lo, &wr_hi, &wr_lo);
        dd_mul(wr_hi, wr_lo, t1_hi, t1_lo, &f_eq_hi[i], &f_eq_lo[i]);
    }
}

// ============================================================================
// DD LBM step kernel
// ============================================================================

// Buffer layout: f_hi[19 * n_cells] + f_lo[19 * n_cells].
// i-major SoA: distribution lane i occupies f_hi[i*n_cells .. (i+1)*n_cells].
// Coalescing: threads in a warp cover idx..idx+31 for fixed i:
//   f_hi_in[i*n_cells + idx], f_hi_in[i*n_cells + idx+1], ... -- fully coalesced.
// Scatter writes f_hi_out[i*n_cells + idx_next] are coalesced for
// x-direction streaming (idx_next = idx+1 for i=1), and nearly coalesced
// for diagonal directions (idx_next differs by nx or nx*ny).
extern "C" __global__ void lbm_step_fused_dd_kernel(
    const double* f_hi_in,    // n_cells * 19 doubles (hi part of distributions)
    const double* f_lo_in,    // n_cells * 19 doubles (lo part)
    double* f_hi_out,
    double* f_lo_out,
    float* rho_out,           // macroscopic density (FP32 for output)
    float* u_out,             // macroscopic velocity (FP32 for output, 3*n_cells)
    const float* force,
    const float* tau_f32,     // relaxation time (FP32 scalar per cell)
    int nx, int ny, int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_cells = nx * ny * nz;
    if (idx >= n_cells) return;

    int x = idx % nx;
    int y = (idx / nx) % ny;
    int z = idx / (nx * ny);

    // Load 19 distribution pairs -- i-major SoA: lane i starts at i*n_cells.
    // Thread idx reads f_hi_in[i*n_cells + idx]; adjacent threads in a warp
    // read consecutive doubles (coalesced 256-byte transactions).
    double f_hi_local[19], f_lo_local[19];
    #pragma unroll
    for (int i = 0; i < 19; i++) {
        f_hi_local[i] = f_hi_in[(long long)i * n_cells + idx];
        f_lo_local[i] = f_lo_in[(long long)i * n_cells + idx];
    }

    // Density: DD sum over 19 distributions
    double rho_hi = 0.0, rho_lo = 0.0;
    #pragma unroll
    for (int i = 0; i < 19; i++) {
        double sh, sl;
        two_sum(rho_hi, f_hi_local[i], &sh, &sl);
        rho_lo += sl + f_lo_local[i];
        rho_hi = sh;
    }
    two_sum(rho_hi, rho_lo, &rho_hi, &rho_lo);

    // Momentum: cx,cy,cz in {-1,0,1} -> no multiplication needed (just conditional add/sub)
    double mx_hi = 0.0, mx_lo = 0.0;
    double my_hi = 0.0, my_lo = 0.0;
    double mz_hi = 0.0, mz_lo = 0.0;
    #pragma unroll
    for (int i = 0; i < 19; i++) {
        if (D3Q19_CX_DD[i] ==  1) { dd_add(mx_hi, mx_lo, f_hi_local[i], f_lo_local[i], &mx_hi, &mx_lo); }
        if (D3Q19_CX_DD[i] == -1) { dd_sub(mx_hi, mx_lo, f_hi_local[i], f_lo_local[i], &mx_hi, &mx_lo); }
        if (D3Q19_CY_DD[i] ==  1) { dd_add(my_hi, my_lo, f_hi_local[i], f_lo_local[i], &my_hi, &my_lo); }
        if (D3Q19_CY_DD[i] == -1) { dd_sub(my_hi, my_lo, f_hi_local[i], f_lo_local[i], &my_hi, &my_lo); }
        if (D3Q19_CZ_DD[i] ==  1) { dd_add(mz_hi, mz_lo, f_hi_local[i], f_lo_local[i], &mz_hi, &mz_lo); }
        if (D3Q19_CZ_DD[i] == -1) { dd_sub(mz_hi, mz_lo, f_hi_local[i], f_lo_local[i], &mz_hi, &mz_lo); }
    }

    // Velocity = momentum / rho  (DD division)
    double ux_hi = 0.0, ux_lo = 0.0;
    double uy_hi = 0.0, uy_lo = 0.0;
    double uz_hi = 0.0, uz_lo = 0.0;
    if (rho_hi > 1.0e-20) {
        dd_div(mx_hi, mx_lo, rho_hi, rho_lo, &ux_hi, &ux_lo);
        dd_div(my_hi, my_lo, rho_hi, rho_lo, &uy_hi, &uy_lo);
        dd_div(mz_hi, mz_lo, rho_hi, rho_lo, &uz_hi, &uz_lo);
    } else {
        rho_hi = 1.0; rho_lo = 0.0;
    }

    rho_out[idx] = (float)rho_hi;
    u_out[idx * 3 + 0] = (float)ux_hi;
    u_out[idx * 3 + 1] = (float)uy_hi;
    u_out[idx * 3 + 2] = (float)uz_hi;

    // DD equilibrium
    double f_eq_hi[19], f_eq_lo[19];
    compute_equilibrium_dd(f_eq_hi, f_eq_lo,
                           rho_hi, rho_lo,
                           ux_hi, ux_lo, uy_hi, uy_lo, uz_hi, uz_lo);

    // DD BGK collision + Guo forcing
    // tau in DD (promote from FP32)
    double tau_hi = (double)tau_f32[idx];
    double inv_tau_hi = 1.0 / tau_hi;
    double prefactor_hi = 1.0 - 0.5 * inv_tau_hi;

    double fx = (double)force[idx * 3 + 0];
    double fy = (double)force[idx * 3 + 1];
    double fz = (double)force[idx * 3 + 2];

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        // BGK: f_i_new = f_i - (f_i - f_eq_i) / tau  (in DD)
        double diff_hi, diff_lo;
        dd_sub(f_hi_local[i], f_lo_local[i], f_eq_hi[i], f_eq_lo[i], &diff_hi, &diff_lo);
        double corr_hi, corr_lo;
        dd_mul(diff_hi, diff_lo, inv_tau_hi, 0.0, &corr_hi, &corr_lo);
        double fi_hi, fi_lo;
        dd_sub(f_hi_local[i], f_lo_local[i], corr_hi, corr_lo, &fi_hi, &fi_lo);

        // Guo forcing (FP64 -- DD for forcing is overkill; convert to DD)
        double eix = (double)D3Q19_CX_DD[i];
        double eiy = (double)D3Q19_CY_DD[i];
        double eiz = (double)D3Q19_CZ_DD[i];
        double s_i = ((eix - ux_hi)*fx + (eiy - uy_hi)*fy + (eiz - uz_hi)*fz) * 3.0
                   + (eix*ux_hi + eiy*uy_hi + eiz*uz_hi) * (eix*fx + eiy*fy + eiz*fz) * 9.0;
        fi_hi = fi_hi + prefactor_hi * D3Q19_WD_DD[i] * s_i;

        // Streaming -- i-major SoA scatter write.
        // f_hi_out[i*n_cells + idx_next]: coalesced when idx_next = idx +/- 1
        // (x-direction), nearly coalesced for y/z/diagonal directions.
        int x_next = (x + D3Q19_CX_DD[i] + nx) % nx;
        int y_next = (y + D3Q19_CY_DD[i] + ny) % ny;
        int z_next = (z + D3Q19_CZ_DD[i] + nz) % nz;
        long long idx_next = (long long)x_next + nx * ((long long)y_next + ny * z_next);
        f_hi_out[(long long)i * n_cells + idx_next] = fi_hi;
        f_lo_out[(long long)i * n_cells + idx_next] = fi_lo;
    }
}

extern "C" __global__ void initialize_uniform_dd_kernel(
    double* f_hi,
    double* f_lo,
    float* rho_out,
    float* u_out,
    float* tau_f32,
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
    tau_f32[idx] = tau_val;

    double rho = (double)rho_init;
    double ux  = (double)ux_init;
    double uy  = (double)uy_init;
    double uz  = (double)uz_init;
    double u_sq = ux*ux + uy*uy + uz*uz;
    double base = fma(-1.5, u_sq, 1.0);

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        double eu = (double)D3Q19_CX_DD[i]*ux
                  + (double)D3Q19_CY_DD[i]*uy
                  + (double)D3Q19_CZ_DD[i]*uz;
        double w_rho = D3Q19_WD_DD[i] * rho;
        double f_eq = w_rho * fma(fma(eu, 4.5, 3.0), eu, base);
        f_hi[(long long)i * n_cells + idx] = f_eq;
        f_lo[(long long)i * n_cells + idx] = 0.0; // lo part starts at zero (exact representation)
    }
}

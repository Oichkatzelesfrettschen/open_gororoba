// Production SoA D3Q19 LBM kernel for FP32 on Ada Lovelace (SM 8.9)
//
// Memory layout (SoA):
//   f[dir * N + idx] -- 32 threads in a warp read contiguous memory
//   u[comp * N + idx] -- velocity stored SoA (read-only by callers via AoS readback)
//   force[comp * N + idx] -- body force SoA
//   tau[idx] -- per-cell relaxation time (scalar)
//   rho[idx] -- density
//
// Adapted from kernels_dark_halo.cu lbm_step_soa_fused (proven pattern)
// with: Horner FMA equilibrium, unconditional force path, init kernels,
// Smagorinsky tau kernel, batch 4D kernel.
//
// Block: 128 threads (1D). Grid: ceil(N / 128).

// D3Q19 lattice velocities (constant memory -- cached in L1)
__constant__ int CX[19] = {
    0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0
};
__constant__ int CY[19] = {
    0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1
};
__constant__ int CZ[19] = {
    0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1
};

__constant__ float W[19] = {
    1.0f/3.0f,
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
};

__device__ __forceinline__ bool finite_check(float x) {
    return (x == x) && (x <= 3.402823466e38f) && (x >= -3.402823466e38f);
}

// ---------------------------------------------------------------------------
// Fused collision + streaming kernel (SoA layout)
// ---------------------------------------------------------------------------
// Each thread: one lattice cell.
// 1. Read f[dir*N+idx] for all 19 directions (coalesced).
// 2. Compute macroscopic rho, u.
// 3. BGK collision with per-cell tau.
// 4. Guo forcing (unconditional, compiler culls zero-force branch).
// 5. Push-streaming to neighbor with periodic BC.
extern "C" __global__ void lbm_step_soa_fused(
    const float* __restrict__ f_in,   // [19 * N] SoA input
    float* __restrict__ f_out,        // [19 * N] SoA output
    float* __restrict__ rho_out,      // [N] density output
    float* __restrict__ u_out,        // [3 * N] velocity output (SoA)
    const float* __restrict__ tau,    // [N] relaxation time
    const float* __restrict__ force,  // [3 * N] body force (SoA)
    int nx, int ny, int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nx * ny * nz;
    if (idx >= N) return;

    int x = idx % nx;
    int y = (idx / nx) % ny;
    int z = idx / (nx * ny);

    // 1. Macroscopic quantities from SoA reads
    float rho_local = 0.0f;
    float mx = 0.0f, my = 0.0f, mz = 0.0f;
    float f_local[19];

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        float fi = __ldg(&f_in[i * N + idx]);
        if (!finite_check(fi)) fi = 0.0f;
        f_local[i] = fi;
        rho_local += fi;
        mx += CX[i] * fi;
        my += CY[i] * fi;
        mz += CZ[i] * fi;
    }

    float ux = 0.0f, uy = 0.0f, uz = 0.0f;
    if (finite_check(rho_local) && rho_local > 1.0e-20f) {
        float inv_rho = 1.0f / rho_local;
        ux = mx * inv_rho;
        uy = my * inv_rho;
        uz = mz * inv_rho;
    } else {
        rho_local = 1.0f;
    }

    // Write macroscopic (SoA velocity)
    rho_out[idx] = rho_local;
    u_out[idx]         = ux;
    u_out[N + idx]     = uy;
    u_out[2 * N + idx] = uz;

    // 2. BGK collision with Horner FMA equilibrium
    float inv_tau = __ldg(&tau[idx]);  // tau array stores precomputed 1/tau (host fills with inv_tau)
    float u_sq = ux * ux + uy * uy + uz * uz;
    float base = fmaf(-1.5f, u_sq, 1.0f);

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        float eu = fmaf((float)CX[i], ux, fmaf((float)CY[i], uy, (float)CZ[i] * uz));
        float w_rho = W[i] * rho_local;
        float f_eq = w_rho * fmaf(fmaf(eu, 4.5f, 3.0f), eu, base);
        f_local[i] -= (f_local[i] - f_eq) * inv_tau;
    }

    // 3. Guo forcing (SoA force layout)
    float fx = __ldg(&force[idx]);
    float fy = __ldg(&force[N + idx]);
    float fz = __ldg(&force[2 * N + idx]);
    float force_mag_sq = fx * fx + fy * fy + fz * fz;

    if (force_mag_sq >= 1e-40f) {
        float prefactor = 1.0f - 0.5f * inv_tau;

        #pragma unroll
        for (int i = 0; i < 19; i++) {
            float eix = (float)CX[i], eiy = (float)CY[i], eiz = (float)CZ[i];
            float em_u_dot_f = (eix - ux) * fx + (eiy - uy) * fy + (eiz - uz) * fz;
            float ei_dot_u = eix * ux + eiy * uy + eiz * uz;
            float ei_dot_f = eix * fx + eiy * fy + eiz * fz;
            f_local[i] += prefactor * W[i] * (em_u_dot_f * 3.0f + ei_dot_u * ei_dot_f * 9.0f);
        }
    }

    // 4. Streaming (push scheme, periodic BC)
    #pragma unroll
    for (int i = 0; i < 19; i++) {
        int xn = (x + CX[i] + nx) % nx;
        int yn = (y + CY[i] + ny) % ny;
        int zn = (z + CZ[i] + nz) % nz;
        int dst = xn + nx * (yn + ny * zn);
        f_out[i * N + dst] = f_local[i];
    }
}

// ---------------------------------------------------------------------------
// Initialization kernels (SoA)
// ---------------------------------------------------------------------------

// Uniform init: all cells get same rho and zero velocity.
extern "C" __global__ void initialize_uniform_soa_kernel(
    float* __restrict__ f,
    float* __restrict__ rho,
    float* __restrict__ u,
    float rho_init,
    float ux_init,
    float uy_init,
    float uz_init,
    int nx, int ny, int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nx * ny * nz;
    if (idx >= N) return;

    rho[idx] = rho_init;
    u[idx]         = ux_init;
    u[N + idx]     = uy_init;
    u[2 * N + idx] = uz_init;

    float u_sq = ux_init * ux_init + uy_init * uy_init + uz_init * uz_init;
    float base = fmaf(-1.5f, u_sq, 1.0f);

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        float eu = fmaf((float)CX[i], ux_init, fmaf((float)CY[i], uy_init, (float)CZ[i] * uz_init));
        float w_rho = W[i] * rho_init;
        f[i * N + idx] = w_rho * fmaf(fmaf(eu, 4.5f, 3.0f), eu, base);
    }
}

// Per-cell init: each cell gets custom rho (AoS input -> SoA storage).
// rho_in[idx] and u_in[idx*3+comp] are AoS (host format).
extern "C" __global__ void initialize_custom_soa_kernel(
    float* __restrict__ f,
    float* __restrict__ rho,
    float* __restrict__ u,
    const float* __restrict__ rho_in,
    const float* __restrict__ u_in,
    int nx, int ny, int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nx * ny * nz;
    if (idx >= N) return;

    float rho_init = rho_in[idx];
    float ux_init = u_in[idx * 3 + 0];
    float uy_init = u_in[idx * 3 + 1];
    float uz_init = u_in[idx * 3 + 2];

    rho[idx] = rho_init;
    u[idx]         = ux_init;
    u[N + idx]     = uy_init;
    u[2 * N + idx] = uz_init;

    float u_sq = ux_init * ux_init + uy_init * uy_init + uz_init * uz_init;
    float base = fmaf(-1.5f, u_sq, 1.0f);

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        float eu = fmaf((float)CX[i], ux_init, fmaf((float)CY[i], uy_init, (float)CZ[i] * uz_init));
        float w_rho = W[i] * rho_init;
        f[i * N + idx] = w_rho * fmaf(fmaf(eu, 4.5f, 3.0f), eu, base);
    }
}

// ---------------------------------------------------------------------------
// Smagorinsky LES subgrid-scale tau computation (Phase 3)
// ---------------------------------------------------------------------------
// Computes strain rate tensor S_ij from velocity gradients (central differences),
// then sets tau = tau_base + 3 * (C_s * dx)^2 * |S|.
// Velocity layout: SoA u[comp * N + idx].
extern "C" __global__ void compute_smagorinsky_tau_kernel(
    const float* __restrict__ u,      // [3*N] velocity (SoA)
    float* __restrict__ tau,           // [N] output tau
    float tau_base,
    float cs_sq_dx_sq,                // (C_s * dx)^2, precomputed on host
    int nx, int ny, int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nx * ny * nz;
    if (idx >= N) return;

    int x = idx % nx;
    int y = (idx / nx) % ny;
    int z = idx / (nx * ny);

    // Neighbor indices (periodic BC)
    int xp = (x + 1) % nx, xm = (x + nx - 1) % nx;
    int yp = (y + 1) % ny, ym = (y + ny - 1) % ny;
    int zp = (z + 1) % nz, zm = (z + nz - 1) % nz;

    // Helper: linear index for (xi, yi, zi)
    #define IDX3(xi, yi, zi) ((xi) + nx * ((yi) + ny * (zi)))

    // Velocity gradients via central differences (factor 0.5 from central diff)
    float dux_dx = (u[IDX3(xp,y,z)] - u[IDX3(xm,y,z)]) * 0.5f;
    float dux_dy = (u[IDX3(x,yp,z)] - u[IDX3(x,ym,z)]) * 0.5f;
    float dux_dz = (u[IDX3(x,y,zp)] - u[IDX3(x,y,zm)]) * 0.5f;

    float duy_dx = (u[N + IDX3(xp,y,z)] - u[N + IDX3(xm,y,z)]) * 0.5f;
    float duy_dy = (u[N + IDX3(x,yp,z)] - u[N + IDX3(x,ym,z)]) * 0.5f;
    float duy_dz = (u[N + IDX3(x,y,zp)] - u[N + IDX3(x,y,zm)]) * 0.5f;

    float duz_dx = (u[2*N + IDX3(xp,y,z)] - u[2*N + IDX3(xm,y,z)]) * 0.5f;
    float duz_dy = (u[2*N + IDX3(x,yp,z)] - u[2*N + IDX3(x,ym,z)]) * 0.5f;
    float duz_dz = (u[2*N + IDX3(x,y,zp)] - u[2*N + IDX3(x,y,zm)]) * 0.5f;

    #undef IDX3

    // Symmetric strain rate tensor S_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
    float s11 = dux_dx;
    float s22 = duy_dy;
    float s33 = duz_dz;
    float s12 = 0.5f * (dux_dy + duy_dx);
    float s13 = 0.5f * (dux_dz + duz_dx);
    float s23 = 0.5f * (duy_dz + duz_dy);

    // |S| = sqrt(2 * S_ij * S_ij) (Frobenius norm of strain rate)
    float s_mag = sqrtf(2.0f * (s11*s11 + s22*s22 + s33*s33
                                + 2.0f*(s12*s12 + s13*s13 + s23*s23)));

    // nu_turb = (C_s * dx)^2 * |S|, tau = tau_base + 3 * nu_turb
    float tau_new = tau_base + 3.0f * cs_sq_dx_sq * s_mag;

    // Clamp to stability range [0.505, 5.0], then write precomputed 1/tau
    float clamped_tau = fmaxf(0.505f, fminf(5.0f, tau_new));
    tau[idx] = 1.0f / clamped_tau;
}

// ---------------------------------------------------------------------------
// 4D Batch kernel (SoA, Phase 4)
// ---------------------------------------------------------------------------
// Processes multiple galaxies in a single kernel launch.
// Each "world" w has its own 3D periodic domain (no cross-galaxy streaming).
// SoA layout: f[dir * N_total + w * N_3d + local_idx]
extern "C" __global__ void lbm_step_soa_batch_kernel(
    const float* __restrict__ f_in,
    float* __restrict__ f_out,
    float* __restrict__ rho_out,
    float* __restrict__ u_out,
    const float* __restrict__ tau,
    const float* __restrict__ force,
    int nx, int ny, int nz,
    int batch_size
) {
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nx * ny * nz;
    int total = N * batch_size;
    if (linear >= total) return;

    int w = linear / N;           // galaxy index
    int local_idx = linear % N;   // cell index within galaxy
    int offset = w * N;           // per-galaxy offset for scalar fields

    int x = local_idx % nx;
    int y = (local_idx / nx) % ny;
    int z = local_idx / (nx * ny);

    // 1. Macroscopic from SoA (offset by galaxy)
    float rho_local = 0.0f;
    float mx = 0.0f, my = 0.0f, mz = 0.0f;
    float f_local[19];

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        float fi = __ldg(&f_in[i * total + offset + local_idx]);
        if (!finite_check(fi)) fi = 0.0f;
        f_local[i] = fi;
        rho_local += fi;
        mx += CX[i] * fi;
        my += CY[i] * fi;
        mz += CZ[i] * fi;
    }

    float ux = 0.0f, uy = 0.0f, uz = 0.0f;
    if (finite_check(rho_local) && rho_local > 1.0e-20f) {
        float inv_rho = 1.0f / rho_local;
        ux = mx * inv_rho;
        uy = my * inv_rho;
        uz = mz * inv_rho;
    } else {
        rho_local = 1.0f;
    }

    rho_out[offset + local_idx] = rho_local;
    u_out[offset + local_idx]             = ux;
    u_out[total + offset + local_idx]     = uy;
    u_out[2 * total + offset + local_idx] = uz;

    // 2. BGK collision
    float inv_tau = __ldg(&tau[offset + local_idx]);  // stores precomputed 1/tau
    float u_sq = ux * ux + uy * uy + uz * uz;
    float base_val = fmaf(-1.5f, u_sq, 1.0f);

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        float eu = fmaf((float)CX[i], ux, fmaf((float)CY[i], uy, (float)CZ[i] * uz));
        float w_rho = W[i] * rho_local;
        float f_eq = w_rho * fmaf(fmaf(eu, 4.5f, 3.0f), eu, base_val);
        f_local[i] -= (f_local[i] - f_eq) * inv_tau;
    }

    // 3. Guo forcing
    float fx = __ldg(&force[offset + local_idx]);
    float fy = __ldg(&force[total + offset + local_idx]);
    float fz = __ldg(&force[2 * total + offset + local_idx]);
    float force_mag_sq = fx * fx + fy * fy + fz * fz;

    if (force_mag_sq >= 1e-40f) {
        float prefactor = 1.0f - 0.5f * inv_tau;

        #pragma unroll
        for (int i = 0; i < 19; i++) {
            float eix = (float)CX[i], eiy = (float)CY[i], eiz = (float)CZ[i];
            float em_u_dot_f = (eix - ux) * fx + (eiy - uy) * fy + (eiz - uz) * fz;
            float ei_dot_u = eix * ux + eiy * uy + eiz * uz;
            float ei_dot_f = eix * fx + eiy * fy + eiz * fz;
            f_local[i] += prefactor * W[i] * (em_u_dot_f * 3.0f + ei_dot_u * ei_dot_f * 9.0f);
        }
    }

    // 4. Streaming (periodic within w-slice only)
    #pragma unroll
    for (int i = 0; i < 19; i++) {
        int xn = (x + CX[i] + nx) % nx;
        int yn = (y + CY[i] + ny) % ny;
        int zn = (z + CZ[i] + nz) % nz;
        int dst_local = xn + nx * (yn + ny * zn);
        f_out[i * total + offset + dst_local] = f_local[i];
    }
}

// Batch init: per-cell density from AoS input -> SoA storage.
extern "C" __global__ void initialize_custom_soa_batch_kernel(
    float* __restrict__ f,
    float* __restrict__ rho,
    float* __restrict__ u,
    const float* __restrict__ rho_in,  // [batch * N] flat
    const float* __restrict__ u_in,    // [batch * N * 3] AoS
    int nx, int ny, int nz,
    int batch_size
) {
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nx * ny * nz;
    int total = N * batch_size;
    if (linear >= total) return;

    int w = linear / N;
    int local_idx = linear % N;
    int offset = w * N;

    float rho_init = rho_in[offset + local_idx];
    float ux_init = u_in[(offset + local_idx) * 3 + 0];
    float uy_init = u_in[(offset + local_idx) * 3 + 1];
    float uz_init = u_in[(offset + local_idx) * 3 + 2];

    rho[offset + local_idx] = rho_init;
    u[offset + local_idx]             = ux_init;
    u[total + offset + local_idx]     = uy_init;
    u[2 * total + offset + local_idx] = uz_init;

    float u_sq = ux_init * ux_init + uy_init * uy_init + uz_init * uz_init;
    float base_val = fmaf(-1.5f, u_sq, 1.0f);

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        float eu = fmaf((float)CX[i], ux_init, fmaf((float)CY[i], uy_init, (float)CZ[i] * uz_init));
        float w_rho = W[i] * rho_init;
        f[i * total + offset + local_idx] = w_rho * fmaf(fmaf(eu, 4.5f, 3.0f), eu, base_val);
    }
}

// ---------------------------------------------------------------------------
// MRT (Multiple-Relaxation-Time) D3Q19 collision device function
// ---------------------------------------------------------------------------
// d'Humieres (2002) orthogonal basis with 5 distinct relaxation rates:
//   s_nu    = 1/tau  -- physical kinematic viscosity (stress moments 9,11,13-15)
//   s_e     = 1.19   -- energy relaxation (moment 1)
//   s_eps   = 1.4    -- energy-squared relaxation (moment 2)
//   s_q     = 1.2    -- energy-flux relaxation (moments 4,6,8)
//   s_ghost = 1.0    -- instant damping of ghost moments (10,12,16-18)
//
// Ghost moment annihilation (s_ghost=1.0) extends the practical Mach number
// stability limit from ~0.3 (BGK) to ~1.5, enabling lower density floors
// and higher dynamic range for galaxy morphological signal.
//
// Cost: ~722 FMA ops/cell vs ~57 for BGK. Memory-bound at 64^3+, so the
// extra ALU hides within memory latency on Ada Lovelace (SM 8.9).
__device__ __forceinline__ void mrt_collision_d3q19(
    float f[19], float rho, float ux, float uy, float uz, float tau_local
) {
    // Relaxation rates (diagonal S matrix)
    float s_nu    = tau_local;  // tau_local slot contains precomputed inv_tau (host-side 1/tau)
    float s_e     = 1.19f;
    float s_eps   = 1.4f;
    float s_q     = 1.2f;
    float s_ghost = 1.0f;

    float u_sq = ux*ux + uy*uy + uz*uz;

    // ---- Forward transform: m = M * f ----
    // d'Humieres D3Q19 orthogonal basis (exact match of CPU collide_mrt_d3q19)
    //
    // ILP restructure: pre-compute pair sums/diffs to eliminate redundant
    // additions and expose independent computation groups for Ada's dual
    // FP32 pipes. 18 pair values computed first, then 19 moments in 4
    // independent groups that can interleave on the FFMA pipeline.

    // Group A: pair sums and differences (18 independent adds/subs)
    float ax_p = f[1] + f[2];   float ax_m = f[1] - f[2];
    float ay_p = f[3] + f[4];   float ay_m = f[3] - f[4];
    float az_p = f[5] + f[6];   float az_m = f[5] - f[6];
    float d78_p  = f[7]  + f[8];   float d78_m  = f[7]  - f[8];
    float d910_p = f[9]  + f[10];  float d910_m = f[9]  - f[10];
    float d1112_p = f[11] + f[12]; float d1112_m = f[11] - f[12];
    float d1314_p = f[13] + f[14]; float d1314_m = f[13] - f[14];
    float d1516_p = f[15] + f[16]; float d1516_m = f[15] - f[16];
    float d1718_p = f[17] + f[18]; float d1718_m = f[17] - f[18];

    // Group B: aggregate sums (depend on Group A, independent of each other)
    float axis_sum = ax_p + ay_p + az_p;
    float diag_sum = d78_p + d910_p + d1112_p + d1314_p + d1516_p + d1718_p;
    float xy_diag  = d78_p + d910_p + d1112_p + d1314_p;
    float z_diag   = d1516_p + d1718_p;

    // Group C: conserved moments (mass + momentum) -- depend on A,B
    float m0  = f[0] + axis_sum + diag_sum;
    float m3  = ax_m + d78_m + d910_m + d1112_m + d1314_m;
    float m5  = ay_m + d78_m - d910_m + d1516_m + d1718_m;
    float m7  = az_m + d1112_m - d1314_m + d1516_m - d1718_m;

    // Group D: non-conserved moments -- depend on A,B, independent of C
    float m1  = fmaf(-30.0f, f[0], fmaf(-11.0f, axis_sum, 8.0f * diag_sum));
    float m2  = fmaf(12.0f, f[0], fmaf(-4.0f, axis_sum, diag_sum));
    float m4  = fmaf(-4.0f, ax_m, d78_m + d910_m + d1112_m + d1314_m);
    float m6  = fmaf(-4.0f, ay_m, d78_m - d910_m + d1516_m + d1718_m);
    float m8  = fmaf(-4.0f, az_m, d1112_m - d1314_m + d1516_m - d1718_m);
    float m9  = fmaf(2.0f, ax_p, -(ay_p + az_p) + xy_diag - 2.0f * z_diag);
    float m10 = fmaf(-2.0f, ax_p, (ay_p + az_p) + xy_diag - 2.0f * z_diag);
    float m11 = ay_p - az_p + d78_p + d910_p - d1112_p - d1314_p;
    float m12 = -ay_p + az_p + d78_p + d910_p - d1112_p - d1314_p;
    float m13 = d78_p - d910_p;
    float m14 = d1112_p - d1314_p;
    float m15 = d1516_p - d1718_p;
    float m16 = d78_m - d910_m - d1112_m + d1314_m;
    float m17 = -d78_m - d910_m + d1516_m + d1718_m;
    float m18 = d1112_m + d1314_m - d1516_m + d1718_m;

    // ---- Equilibrium moments ----
    float m1_eq  = rho * fmaf(19.0f, u_sq, -11.0f);
    float m2_eq  = rho * fmaf(-5.5f, u_sq, 3.0f);
    float m4_eq  = (-2.0f / 3.0f) * rho * ux;
    float m6_eq  = (-2.0f / 3.0f) * rho * uy;
    float m8_eq  = (-2.0f / 3.0f) * rho * uz;
    float pxx    = fmaf(2.0f, ux*ux, -(uy*uy + uz*uz));
    float m9_eq  = rho * pxx;
    float m10_eq = -0.5f * rho * pxx;
    float pww    = uy*uy - uz*uz;
    float m11_eq = rho * pww;
    float m12_eq = -0.5f * rho * pww;
    float m13_eq = rho * ux * uy;
    float m14_eq = rho * ux * uz;
    float m15_eq = rho * uy * uz;

    // ---- Relax: m* = m - S * (m - m_eq) ----
    // Mass (m0) and momentum (m3, m5, m7) are conserved (S=0, unchanged)
    m1  -= s_e    * (m1  - m1_eq);            // energy
    m2  -= s_eps  * (m2  - m2_eq);            // energy^2
    m4  -= s_q    * (m4  - m4_eq);            // energy flux x
    m6  -= s_q    * (m6  - m6_eq);            // energy flux y
    m8  -= s_q    * (m8  - m8_eq);            // energy flux z
    m9  -= s_nu   * (m9  - m9_eq);            // stress p_xx (physical)
    m10 -= s_ghost* (m10 - m10_eq);           // ghost pi_xx
    m11 -= s_nu   * (m11 - m11_eq);           // stress p_ww (physical)
    m12 -= s_ghost* (m12 - m12_eq);           // ghost pi_ww
    m13 -= s_nu   * (m13 - m13_eq);           // stress p_xy (physical)
    m14 -= s_nu   * (m14 - m14_eq);           // stress p_xz (physical)
    m15 -= s_nu   * (m15 - m15_eq);           // stress p_yz (physical)
    m16 -= s_ghost* m16;                      // ghost m_x (eq=0)
    m17 -= s_ghost* m17;                      // ghost m_y (eq=0)
    m18 -= s_ghost* m18;                      // ghost m_z (eq=0)

    // ---- Inverse transform: f* = M^{-1} * m* ----
    // M^{-1}_{ij} = M_{ji} / ||row_j||^2 (orthogonal non-orthonormal basis)
    // Row norms: [19, 2394, 252, 10, 40, 10, 40, 10, 40, 36, 36, 12, 12, 4, 4, 4, 8, 8, 8]
    //
    // ILP restructure: pre-compute common sub-expressions (base_diag, base_xy,
    // base_xz, base_yz) shared across diagonal direction outputs. Reduces the
    // longest additive chain from 12 terms (sequential) to ~4 (balanced tree).
    // Ada dual FP32 pipes interleave independent chains within each group.
    float r0  = m0  * (1.0f / 19.0f);
    float r1  = m1  * (1.0f / 2394.0f);
    float r2  = m2  * (1.0f / 252.0f);
    float r3  = m3  * (1.0f / 10.0f);
    float r4  = m4  * (1.0f / 40.0f);
    float r5  = m5  * (1.0f / 10.0f);
    float r6  = m6  * (1.0f / 40.0f);
    float r7  = m7  * (1.0f / 10.0f);
    float r8  = m8  * (1.0f / 40.0f);
    float r9  = m9  * (1.0f / 36.0f);
    float r10 = m10 * (1.0f / 36.0f);
    float r11 = m11 * (1.0f / 12.0f);
    float r12 = m12 * (1.0f / 12.0f);
    float r13 = m13 * 0.25f;
    float r14 = m14 * 0.25f;
    float r15 = m15 * 0.25f;
    float r16 = m16 * 0.125f;
    float r17 = m17 * 0.125f;
    float r18 = m18 * 0.125f;

    // Common sub-expressions for the inverse transform.
    // These sub-bases are shared across groups of 4 diagonal outputs,
    // reducing each output from 12+ terms to 6-7 terms (log2 depth ~3).
    float base_diag = r0 + r2;                      // shared by all 12 diag dirs
    float r910       = r9 + r10;                     // stress pair sum
    float r1112      = r11 + r12;                    // stress pair sum
    float s34        = r3 + r4;                      // momentum-x pair
    float s56        = r5 + r6;                      // momentum-y pair
    float s78        = r7 + r8;                      // momentum-z pair
    float base_axis  = fmaf(-11.0f, r1, fmaf(-4.0f, r2, r0)); // shared by f[1]..f[6]
    float base_xy    = base_diag + r910 + r1112;     // shared by f[7]..f[10]
    float base_xz    = base_diag + r910 - r1112;     // shared by f[11]..f[14]
    float base_yz    = fmaf(-2.0f, r910, base_diag); // shared by f[15]..f[18]

    // f[i] = sum_j M[j][i] * r[j]  -- Balanced dependency trees per group.
    //
    // Group 0: f[0] (rest) + f[1..6] (axis directions)
    // Axis dirs share base_axis; each differs only in momentum/stress terms.
    f[0]  = fmaf(-30.0f, r1, fmaf(12.0f, r2, r0));
    f[1]  = fmaf(-4.0f, r4, base_axis + r3 + 2.0f * r9 - 2.0f * r10);
    f[2]  = fmaf( 4.0f, r4, base_axis - r3 + 2.0f * r9 - 2.0f * r10);
    f[3]  = fmaf(-4.0f, r6, base_axis + r5 - r9 + r10 + r11 - r12);
    f[4]  = fmaf( 4.0f, r6, base_axis - r5 - r9 + r10 + r11 - r12);
    f[5]  = fmaf(-4.0f, r8, base_axis + r7 - r9 + r10 - r11 + r12);
    f[6]  = fmaf( 4.0f, r8, base_axis - r7 - r9 + r10 - r11 + r12);

    // Group 1: f[7..10] (xy-plane diagonals)
    // Each: fmaf(8, r1, base_xy +/- s34 +/- s56 +/- r13 +/- r16 +/- r17)
    // Balanced: (s34+s56) and (r13+r16) computed, then combined with r17.
    {
        float p1 = s34 + s56;  float p2 = r13 + r16;
        float n1 = s34 - s56;  float n2 = r13 - r16;
        f[7]  = fmaf(8.0f, r1, base_xy + p1 + p2 - r17);
        f[8]  = fmaf(8.0f, r1, base_xy - p1 + n2 + r17);
        f[9]  = fmaf(8.0f, r1, base_xy + n1 - p2 - r17);
        f[10] = fmaf(8.0f, r1, base_xy - n1 - n2 + r17);
    }

    // Group 2: f[11..14] (xz-plane diagonals)
    {
        float p1 = s34 + s78;  float p2 = r14 + r16;
        float n1 = s34 - s78;  float n2 = r14 - r16;
        f[11] = fmaf(8.0f, r1, base_xz + p1 + n2 + r18);
        f[12] = fmaf(8.0f, r1, base_xz - p1 + p2 - r18);
        f[13] = fmaf(8.0f, r1, base_xz + n1 - n2 + r18);
        f[14] = fmaf(8.0f, r1, base_xz - n1 - p2 - r18);
    }

    // Group 3: f[15..18] (yz-plane diagonals)
    {
        float p1 = s56 + s78;  float p2 = r15 + r17;
        float n1 = s56 - s78;  float n2 = r15 - r17;
        f[15] = fmaf(8.0f, r1, base_yz + p1 + p2 - r18);
        f[16] = fmaf(8.0f, r1, base_yz - p1 + n2 + r18);
        f[17] = fmaf(8.0f, r1, base_yz + n1 - p2 - r18);
        f[18] = fmaf(8.0f, r1, base_yz - n1 - n2 + r18);
    }
}

// ---------------------------------------------------------------------------
// Fused MRT collision + streaming kernel (SoA layout)
// ---------------------------------------------------------------------------
// Identical signature and structure to lbm_step_soa_fused (BGK), but replaces
// the BGK collision with the d'Humieres MRT operator above. Macroscopic
// computation, Guo forcing, and push-streaming are identical.
// Force density contributes its full first moment when the MRT momentum rate is zero.
// The zero-equilibrium operator A gives (Phi+A*Phi)/2 = M^-1*(I-S/2)*M*Phi.
__device__ __forceinline__ void mrt_collision_forced_d3q19(
    float* populations, float density, float ux, float uy, float uz,
    float inv_tau, float fx, float fy, float fz
) {
    if (fx == 0.0f && fy == 0.0f && fz == 0.0f) {
        mrt_collision_d3q19(populations, density, ux, uy, uz, inv_tau);
        return;
    }
    float half_inverse_density = 0.5f / density;
    ux += fx * half_inverse_density;
    uy += fy * half_inverse_density;
    uz += fz * half_inverse_density;
    mrt_collision_d3q19(populations, density, ux, uy, uz, inv_tau);
    float source[19], relaxed_source[19];
    #pragma unroll
    for (int direction = 0; direction < 19; ++direction) {
        float cx = (float)CX[direction], cy = (float)CY[direction], cz = (float)CZ[direction];
        float velocity_dot_force = (cx - ux) * fx + (cy - uy) * fy + (cz - uz) * fz;
        float direction_dot_velocity = cx * ux + cy * uy + cz * uz;
        float direction_dot_force = cx * fx + cy * fy + cz * fz;
        source[direction] = W[direction] * (3.0f * velocity_dot_force
            + 9.0f * direction_dot_velocity * direction_dot_force);
        relaxed_source[direction] = source[direction];
    }
    mrt_collision_d3q19(relaxed_source, 0.0f, 0.0f, 0.0f, 0.0f, inv_tau);
    #pragma unroll
    for (int direction = 0; direction < 19; ++direction) {
        populations[direction] += 0.5f * (source[direction] + relaxed_source[direction]);
    }
}

extern "C" __global__ void lbm_step_soa_mrt_fused(
    const float* __restrict__ f_in,
    float* __restrict__ f_out,
    float* __restrict__ rho_out,
    float* __restrict__ u_out,
    const float* __restrict__ tau,
    const float* __restrict__ force,
    int nx, int ny, int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nx * ny * nz;
    if (idx >= N) return;

    int x = idx % nx;
    int y = (idx / nx) % ny;
    int z = idx / (nx * ny);

    // 1. Macroscopic quantities from SoA reads
    float rho_local = 0.0f;
    float mx = 0.0f, my = 0.0f, mz = 0.0f;
    float f_local[19];

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        float fi = __ldg(&f_in[i * N + idx]);
        if (!finite_check(fi)) fi = 0.0f;
        f_local[i] = fi;
        rho_local += fi;
        mx += CX[i] * fi;
        my += CY[i] * fi;
        mz += CZ[i] * fi;
    }

    float ux = 0.0f, uy = 0.0f, uz = 0.0f;
    if (finite_check(rho_local) && rho_local > 1.0e-20f) {
        float inv_rho = 1.0f / rho_local;
        ux = mx * inv_rho;
        uy = my * inv_rho;
        uz = mz * inv_rho;
    } else {
        rho_local = 1.0f;
    }

    // Write macroscopic (SoA velocity)
    rho_out[idx] = rho_local;
    u_out[idx]         = ux;
    u_out[N + idx]     = uy;
    u_out[2 * N + idx] = uz;

    // 2. MRT collision (replaces BGK)
    float inv_tau = __ldg(&tau[idx]);  // stores precomputed 1/tau
    float fx = force == NULL ? 0.0f : __ldg(&force[idx]);
    float fy = force == NULL ? 0.0f : __ldg(&force[N + idx]);
    float fz = force == NULL ? 0.0f : __ldg(&force[2 * N + idx]);
    mrt_collision_forced_d3q19(f_local, rho_local, ux, uy, uz, inv_tau, fx, fy, fz);

    // 4. Streaming (push scheme, periodic BC)
    #pragma unroll
    for (int i = 0; i < 19; i++) {
        int xn = (x + CX[i] + nx) % nx;
        int yn = (y + CY[i] + ny) % ny;
        int zn = (z + CZ[i] + nz) % nz;
        int dst = xn + nx * (yn + ny * zn);
        f_out[i * N + dst] = f_local[i];
    }
}

// ---------------------------------------------------------------------------
// Device function: process one cell (BGK collision + Guo forcing + streaming)
// ---------------------------------------------------------------------------
// Extracted from lbm_step_soa_fused for use in coarsened kernels.
// __forceinline__ ensures the compiler inlines both calls in the coarsened
// kernel, allowing instruction interleaving between cell A and cell B.
__device__ __forceinline__ void process_cell_bgk(
    const float* __restrict__ f_in,
    float* __restrict__ f_out,
    float* __restrict__ rho_out,
    float* __restrict__ u_out,
    const float* __restrict__ tau,
    const float* __restrict__ force,
    int idx, int N, int nx, int ny, int nz,
    int pull  // 0 = push streaming, 1 = pull (stream-collide)
) {
    int x = idx % nx;
    int y = (idx / nx) % ny;
    int z = idx / (nx * ny);

    float rho_local = 0.0f;
    float mx = 0.0f, my = 0.0f, mz = 0.0f;
    float f_local[19];

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        int src;
        if (pull) {
            // Pull: read from neighbor in opposite direction (scattered read)
            int xn = (x - CX[i] + nx) % nx;
            int yn = (y - CY[i] + ny) % ny;
            int zn = (z - CZ[i] + nz) % nz;
            src = xn + nx * (yn + ny * zn);
        } else {
            src = idx;
        }
        float fi = __ldg(&f_in[i * N + src]);
        if (!finite_check(fi)) fi = 0.0f;
        f_local[i] = fi;
        rho_local += fi;
        mx += CX[i] * fi;
        my += CY[i] * fi;
        mz += CZ[i] * fi;
    }

    float ux = 0.0f, uy = 0.0f, uz = 0.0f;
    if (finite_check(rho_local) && rho_local > 1.0e-20f) {
        float inv_rho = 1.0f / rho_local;
        ux = mx * inv_rho;
        uy = my * inv_rho;
        uz = mz * inv_rho;
    } else {
        rho_local = 1.0f;
    }

    rho_out[idx] = rho_local;
    u_out[idx]         = ux;
    u_out[N + idx]     = uy;
    u_out[2 * N + idx] = uz;

    float inv_tau = __ldg(&tau[idx]);  // tau array stores precomputed 1/tau (host fills with inv_tau)
    float u_sq = ux * ux + uy * uy + uz * uz;
    float base = fmaf(-1.5f, u_sq, 1.0f);

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        float eu = fmaf((float)CX[i], ux, fmaf((float)CY[i], uy, (float)CZ[i] * uz));
        float w_rho = W[i] * rho_local;
        float f_eq = w_rho * fmaf(fmaf(eu, 4.5f, 3.0f), eu, base);
        f_local[i] -= (f_local[i] - f_eq) * inv_tau;
    }

    float fx = __ldg(&force[idx]);
    float fy = __ldg(&force[N + idx]);
    float fz = __ldg(&force[2 * N + idx]);
    float force_mag_sq = fx * fx + fy * fy + fz * fz;

    if (force_mag_sq >= 1e-40f) {
        float prefactor = 1.0f - 0.5f * inv_tau;
        #pragma unroll
        for (int i = 0; i < 19; i++) {
            float eix = (float)CX[i], eiy = (float)CY[i], eiz = (float)CZ[i];
            float em_u_dot_f = (eix - ux) * fx + (eiy - uy) * fy + (eiz - uz) * fz;
            float ei_dot_u = eix * ux + eiy * uy + eiz * uz;
            float ei_dot_f = eix * fx + eiy * fy + eiz * fz;
            f_local[i] += prefactor * W[i] * (em_u_dot_f * 3.0f + ei_dot_u * ei_dot_f * 9.0f);
        }
    }

    // Streaming: push writes to neighbor (scattered), pull writes to self (coalesced)
    #pragma unroll
    for (int i = 0; i < 19; i++) {
        if (pull) {
            f_out[i * N + idx] = f_local[i];
        } else {
            int xn = (x + CX[i] + nx) % nx;
            int yn = (y + CY[i] + ny) % ny;
            int zn = (z + CZ[i] + nz) % nz;
            int dst = xn + nx * (yn + ny * zn);
            f_out[i * N + dst] = f_local[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Device function: process one cell (MRT collision + Guo forcing + streaming)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void process_cell_mrt(
    const float* __restrict__ f_in,
    float* __restrict__ f_out,
    float* __restrict__ rho_out,
    float* __restrict__ u_out,
    const float* __restrict__ tau,
    const float* __restrict__ force,
    int idx, int N, int nx, int ny, int nz,
    int pull  // 0 = push streaming, 1 = pull (stream-collide)
) {
    int x = idx % nx;
    int y = (idx / nx) % ny;
    int z = idx / (nx * ny);

    float rho_local = 0.0f;
    float mx = 0.0f, my = 0.0f, mz = 0.0f;
    float f_local[19];

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        int src;
        if (pull) {
            int xn = (x - CX[i] + nx) % nx;
            int yn = (y - CY[i] + ny) % ny;
            int zn = (z - CZ[i] + nz) % nz;
            src = xn + nx * (yn + ny * zn);
        } else {
            src = idx;
        }
        float fi = __ldg(&f_in[i * N + src]);
        if (!finite_check(fi)) fi = 0.0f;
        f_local[i] = fi;
        rho_local += fi;
        mx += CX[i] * fi;
        my += CY[i] * fi;
        mz += CZ[i] * fi;
    }

    float ux = 0.0f, uy = 0.0f, uz = 0.0f;
    if (finite_check(rho_local) && rho_local > 1.0e-20f) {
        float inv_rho = 1.0f / rho_local;
        ux = mx * inv_rho;
        uy = my * inv_rho;
        uz = mz * inv_rho;
    } else {
        rho_local = 1.0f;
    }

    rho_out[idx] = rho_local;
    u_out[idx]         = ux;
    u_out[N + idx]     = uy;
    u_out[2 * N + idx] = uz;

    float inv_tau = __ldg(&tau[idx]);  // stores precomputed 1/tau
    float fx = force == NULL ? 0.0f : __ldg(&force[idx]);
    float fy = force == NULL ? 0.0f : __ldg(&force[N + idx]);
    float fz = force == NULL ? 0.0f : __ldg(&force[2 * N + idx]);
    mrt_collision_forced_d3q19(f_local, rho_local, ux, uy, uz, inv_tau, fx, fy, fz);

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        if (pull) {
            f_out[i * N + idx] = f_local[i];
        } else {
            int xn = (x + CX[i] + nx) % nx;
            int yn = (y + CY[i] + ny) % ny;
            int zn = (z + CZ[i] + nz) % nz;
            int dst = xn + nx * (yn + ny * zn);
            f_out[i * N + dst] = f_local[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Pull-streaming BGK kernel (Phase 5)
// ---------------------------------------------------------------------------
// Stream-collide order: each thread pulls distributions from opposite
// neighbors (scattered reads), collides locally, and writes to self
// (coalesced writes). On Ada Lovelace L2, coalesced writes typically
// outperform coalesced reads because writes bypass L1 via write-through.
// Full occupancy preserved (no shared memory, no extra registers).
//
// INCOMPATIBLE with float2 coarsening (source cells for contiguous idx
// are non-contiguous for cy/cz != 0). Use non-coarsened path only.
extern "C" __global__ void lbm_step_soa_pull(
    const float* __restrict__ f_in,
    float* __restrict__ f_out,
    float* __restrict__ rho_out,
    float* __restrict__ u_out,
    const float* __restrict__ tau,
    const float* __restrict__ force,
    int nx, int ny, int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nx * ny * nz;
    if (idx >= N) return;
    process_cell_bgk(f_in, f_out, rho_out, u_out, tau, force, idx, N, nx, ny, nz, 1);
}

// ---------------------------------------------------------------------------
// Pull-streaming MRT kernel (Phase 5)
// ---------------------------------------------------------------------------
extern "C" __global__ void lbm_step_soa_mrt_pull(
    const float* __restrict__ f_in,
    float* __restrict__ f_out,
    float* __restrict__ rho_out,
    float* __restrict__ u_out,
    const float* __restrict__ tau,
    const float* __restrict__ force,
    int nx, int ny, int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nx * ny * nz;
    if (idx >= N) return;
    process_cell_mrt(f_in, f_out, rho_out, u_out, tau, force, idx, N, nx, ny, nz, 1);
}

// ---------------------------------------------------------------------------
// Shared-memory tiled pull-scheme BGK kernel (Item 4: YSU-inspired)
// ---------------------------------------------------------------------------
// 8x8x4 tile with 1-cell halo loaded cooperatively into shared memory.
// Pull streaming: each thread reads 19 neighbors from shared memory (fast LDS)
// and writes f_out coalesced (each thread writes its own index).
//
// Tile: 8x8x4 = 256 threads. Padded: 10x10x6 = 600 cells/direction.
// Shared memory: 19 * 600 * 4B = 45600B (fits 48KB default on Ada SM 8.9).
// At 128^3: grid = (16,16,32) = 8192 blocks. Occupancy: 2 blocks/SM (256 thr/blk).
//
// Performance model: replaces 19 scattered global reads with 19 shared-memory
// reads (LDS latency ~20 cycles vs L2 miss ~200 cycles). Writes remain coalesced.
// Estimated 15-25% speedup on the streaming phase.
#define TILE_X 8
#define TILE_Y 8
#define TILE_Z 4
#define PAD_X (TILE_X + 2)
#define PAD_Y (TILE_Y + 2)
#define PAD_Z (TILE_Z + 2)
#define PAD_VOL (PAD_X * PAD_Y * PAD_Z)

extern "C" __global__ void __launch_bounds__(256)
lbm_step_soa_tiled(
    const float* __restrict__ f_in,
    float* __restrict__ f_out,
    float* __restrict__ rho_out,
    float* __restrict__ u_out,
    const float* __restrict__ tau,
    const float* __restrict__ force,
    int nx, int ny, int nz
) {
    __shared__ float sf[19 * PAD_VOL];

    int bx0 = blockIdx.x * TILE_X;
    int by0 = blockIdx.y * TILE_Y;
    int bz0 = blockIdx.z * TILE_Z;
    int N   = nx * ny * nz;
    int tid = threadIdx.x + TILE_X * (threadIdx.y + TILE_Y * threadIdx.z);

    // Phase 1: Cooperative striped load of tile + halo into shared memory.
    // Each thread loads multiple elements in a grid-stride pattern.
    // Global reads are coalesced within each direction's contiguous SoA slice.
    for (int i = tid; i < 19 * PAD_VOL; i += 256) {
        int dir = i / PAD_VOL;
        int rem = i % PAD_VOL;
        int lz  = rem / (PAD_X * PAD_Y);
        int ly  = (rem % (PAD_X * PAD_Y)) / PAD_X;
        int lx  = rem % PAD_X;
        int gx  = (bx0 + lx - 1 + nx) % nx;
        int gy  = (by0 + ly - 1 + ny) % ny;
        int gz  = (bz0 + lz - 1 + nz) % nz;
        sf[i] = __ldg(&f_in[dir * N + gx + nx * (gy + ny * gz)]);
    }
    __syncthreads();

    // Phase 2: Boundary guard -- threads beyond domain exit early.
    int gx = bx0 + (int)threadIdx.x;
    int gy = by0 + (int)threadIdx.y;
    int gz = bz0 + (int)threadIdx.z;
    if (gx >= nx || gy >= ny || gz >= nz) return;

    int idx = gx + nx * (gy + ny * gz);
    // Shared-memory coordinates (offset by +1 for halo).
    int sx = (int)threadIdx.x + 1;
    int sy = (int)threadIdx.y + 1;
    int sz = (int)threadIdx.z + 1;

    // Phase 3: Pull 19 neighbors from shared memory + accumulate moments.
    float f_local[19];
    float rho_local = 0.0f, mx = 0.0f, my = 0.0f, mz = 0.0f;
    #pragma unroll
    for (int i = 0; i < 19; i++) {
        // Pull from opposite direction: read f_in at (x-cx, y-cy, z-cz)
        int lx = sx - CX[i];
        int ly = sy - CY[i];
        int lz = sz - CZ[i];
        float fi = sf[i * PAD_VOL + lz * (PAD_X * PAD_Y) + ly * PAD_X + lx];
        if (!finite_check(fi)) fi = 0.0f;
        f_local[i] = fi;
        rho_local += fi;
        mx += CX[i] * fi;
        my += CY[i] * fi;
        mz += CZ[i] * fi;
    }

    float ux = 0.0f, uy = 0.0f, uz = 0.0f;
    if (finite_check(rho_local) && rho_local > 1.0e-20f) {
        float inv_rho = 1.0f / rho_local;
        ux = mx * inv_rho;
        uy = my * inv_rho;
        uz = mz * inv_rho;
    } else {
        rho_local = 1.0f;
    }

    rho_out[idx] = rho_local;
    u_out[idx]         = ux;
    u_out[N + idx]     = uy;
    u_out[2 * N + idx] = uz;

    // Phase 4: BGK collision
    float inv_tau = __ldg(&tau[idx]);  // stores precomputed 1/tau
    float u_sq      = ux * ux + uy * uy + uz * uz;
    float base      = fmaf(-1.5f, u_sq, 1.0f);

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        float eu = fmaf((float)CX[i], ux, fmaf((float)CY[i], uy, (float)CZ[i] * uz));
        float w_rho = W[i] * rho_local;
        float f_eq  = w_rho * fmaf(fmaf(eu, 4.5f, 3.0f), eu, base);
        f_local[i] -= (f_local[i] - f_eq) * inv_tau;
    }

    // Phase 5: Guo forcing
    float fx = __ldg(&force[idx]);
    float fy = __ldg(&force[N + idx]);
    float fz = __ldg(&force[2 * N + idx]);
    float force_mag_sq = fx * fx + fy * fy + fz * fz;

    if (force_mag_sq >= 1e-40f) {
        float prefactor = 1.0f - 0.5f * inv_tau;
        #pragma unroll
        for (int i = 0; i < 19; i++) {
            float eix = (float)CX[i], eiy = (float)CY[i], eiz = (float)CZ[i];
            float em_u_dot_f = (eix - ux) * fx + (eiy - uy) * fy + (eiz - uz) * fz;
            float ei_dot_u = eix * ux + eiy * uy + eiz * uz;
            float ei_dot_f = eix * fx + eiy * fy + eiz * fz;
            f_local[i] += prefactor * W[i] * (em_u_dot_f * 3.0f + ei_dot_u * ei_dot_f * 9.0f);
        }
    }

    // Phase 6: Coalesced global write (pull scheme: each thread writes own index)
    #pragma unroll
    for (int i = 0; i < 19; i++) {
        f_out[i * N + idx] = f_local[i];
    }
}

// ---------------------------------------------------------------------------
// Shared-memory tiled pull-scheme MRT kernel (Item 4: YSU-inspired)
// ---------------------------------------------------------------------------
// Same tile geometry as tiled BGK above, but uses MRT collision operator.
extern "C" __global__ void __launch_bounds__(256)
lbm_step_soa_mrt_tiled(
    const float* __restrict__ f_in,
    float* __restrict__ f_out,
    float* __restrict__ rho_out,
    float* __restrict__ u_out,
    const float* __restrict__ tau,
    const float* __restrict__ force,
    int nx, int ny, int nz
) {
    __shared__ float sf[19 * PAD_VOL];

    int bx0 = blockIdx.x * TILE_X;
    int by0 = blockIdx.y * TILE_Y;
    int bz0 = blockIdx.z * TILE_Z;
    int N   = nx * ny * nz;
    int tid = threadIdx.x + TILE_X * (threadIdx.y + TILE_Y * threadIdx.z);

    // Phase 1: Cooperative striped load
    for (int i = tid; i < 19 * PAD_VOL; i += 256) {
        int dir = i / PAD_VOL;
        int rem = i % PAD_VOL;
        int lz  = rem / (PAD_X * PAD_Y);
        int ly  = (rem % (PAD_X * PAD_Y)) / PAD_X;
        int lx  = rem % PAD_X;
        int gx  = (bx0 + lx - 1 + nx) % nx;
        int gy  = (by0 + ly - 1 + ny) % ny;
        int gz  = (bz0 + lz - 1 + nz) % nz;
        sf[i] = __ldg(&f_in[dir * N + gx + nx * (gy + ny * gz)]);
    }
    __syncthreads();

    // Phase 2: Boundary guard
    int gx = bx0 + (int)threadIdx.x;
    int gy = by0 + (int)threadIdx.y;
    int gz = bz0 + (int)threadIdx.z;
    if (gx >= nx || gy >= ny || gz >= nz) return;

    int idx = gx + nx * (gy + ny * gz);
    int sx = (int)threadIdx.x + 1;
    int sy = (int)threadIdx.y + 1;
    int sz = (int)threadIdx.z + 1;

    // Phase 3: Pull from shared memory
    float f_local[19];
    float rho_local = 0.0f, mx = 0.0f, my = 0.0f, mz = 0.0f;
    #pragma unroll
    for (int i = 0; i < 19; i++) {
        int lx = sx - CX[i];
        int ly = sy - CY[i];
        int lz = sz - CZ[i];
        float fi = sf[i * PAD_VOL + lz * (PAD_X * PAD_Y) + ly * PAD_X + lx];
        if (!finite_check(fi)) fi = 0.0f;
        f_local[i] = fi;
        rho_local += fi;
        mx += CX[i] * fi;
        my += CY[i] * fi;
        mz += CZ[i] * fi;
    }

    float ux = 0.0f, uy = 0.0f, uz = 0.0f;
    if (finite_check(rho_local) && rho_local > 1.0e-20f) {
        float inv_rho = 1.0f / rho_local;
        ux = mx * inv_rho;
        uy = my * inv_rho;
        uz = mz * inv_rho;
    } else {
        rho_local = 1.0f;
    }

    rho_out[idx] = rho_local;
    u_out[idx]         = ux;
    u_out[N + idx]     = uy;
    u_out[2 * N + idx] = uz;

    // Phase 4: MRT collision
    float inv_tau = __ldg(&tau[idx]);  // stores precomputed 1/tau
    float fx = force == NULL ? 0.0f : __ldg(&force[idx]);
    float fy = force == NULL ? 0.0f : __ldg(&force[N + idx]);
    float fz = force == NULL ? 0.0f : __ldg(&force[2 * N + idx]);
    mrt_collision_forced_d3q19(f_local, rho_local, ux, uy, uz, inv_tau, fx, fy, fz);

    // Phase 6: Coalesced global write
    #pragma unroll
    for (int i = 0; i < 19; i++) {
        f_out[i * N + idx] = f_local[i];
    }
}

#undef TILE_X
#undef TILE_Y
#undef TILE_Z
#undef PAD_X
#undef PAD_Y
#undef PAD_Z
#undef PAD_VOL

// ---------------------------------------------------------------------------
// Thread-coarsened BGK kernel with float2 loads (Phase 8)
// ---------------------------------------------------------------------------
// 1D coarsening: each thread processes 2 contiguous cells (idx, idx+1).
// float2 vectorized loads halve the LSU instruction count: 19 float2 loads
// instead of 38 scalar __ldg. While cell A's collision stalls on a
// dependent FMA, cell B's independent arithmetic fills the pipeline (ILP).
//
// Register budget: ~76 regs/thread. At 128 threads/block, __launch_bounds__
// requests 6 blocks/SM = 768 threads = 37.5% occupancy. ILP gain outweighs
// the occupancy reduction for memory-bound D3Q19.
//
// REQUIREMENT: N = nx*ny*nz MUST be even (true for 64^3, 128^3, 256^3).
// Host dispatch must fall back to non-coarsened kernel for odd N.
extern "C" __global__ void __launch_bounds__(128, 6)
lbm_step_soa_coarsened(
    const float* __restrict__ f_in,
    float* __restrict__ f_out,
    float* __restrict__ rho_out,
    float* __restrict__ u_out,
    const float* __restrict__ tau,
    const float* __restrict__ force,
    int nx, int ny, int nz
) {
    int N = nx * ny * nz;
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (base >= N) return;
    int has_second = (base + 1 < N);

    // Vectorized float2 loads: fetch (cell0, cell1) per direction in one
    // 64-bit transaction. Halves the number of LDG instructions.
    float f0[19], f1[19];
    #pragma unroll
    for (int i = 0; i < 19; i++) {
        float2 fv = __ldg(reinterpret_cast<const float2*>(&f_in[i * N + base]));
        f0[i] = finite_check(fv.x) ? fv.x : 0.0f;
        f1[i] = finite_check(fv.y) ? fv.y : 0.0f;
    }

    // ---- Cell 0 ----
    {
        int idx = base;
        int x = idx % nx;
        int y = (idx / nx) % ny;
        int z = idx / (nx * ny);

        float rho_local = 0.0f, mx = 0.0f, my = 0.0f, mz = 0.0f;
        #pragma unroll
        for (int i = 0; i < 19; i++) {
            rho_local += f0[i];
            mx += CX[i] * f0[i];
            my += CY[i] * f0[i];
            mz += CZ[i] * f0[i];
        }

        float ux = 0.0f, uy = 0.0f, uz = 0.0f;
        if (finite_check(rho_local) && rho_local > 1.0e-20f) {
            float inv_rho = 1.0f / rho_local;
            ux = mx * inv_rho;
            uy = my * inv_rho;
            uz = mz * inv_rho;
        } else {
            rho_local = 1.0f;
        }

        rho_out[idx] = rho_local;
        u_out[idx]         = ux;
        u_out[N + idx]     = uy;
        u_out[2 * N + idx] = uz;

        float inv_tau = __ldg(&tau[idx]);  // stores precomputed 1/tau
        float u_sq = ux * ux + uy * uy + uz * uz;
        float base_eq = fmaf(-1.5f, u_sq, 1.0f);

        #pragma unroll
        for (int i = 0; i < 19; i++) {
            float eu = fmaf((float)CX[i], ux, fmaf((float)CY[i], uy, (float)CZ[i] * uz));
            float w_rho = W[i] * rho_local;
            float f_eq = w_rho * fmaf(fmaf(eu, 4.5f, 3.0f), eu, base_eq);
            f0[i] -= (f0[i] - f_eq) * inv_tau;
        }

        float fx = __ldg(&force[idx]);
        float fy = __ldg(&force[N + idx]);
        float fz = __ldg(&force[2 * N + idx]);
        float force_mag_sq = fx * fx + fy * fy + fz * fz;

        if (force_mag_sq >= 1e-40f) {
            float prefactor = 1.0f - 0.5f * inv_tau;
            #pragma unroll
            for (int i = 0; i < 19; i++) {
                float eix = (float)CX[i], eiy = (float)CY[i], eiz = (float)CZ[i];
                float em_u_dot_f = (eix - ux) * fx + (eiy - uy) * fy + (eiz - uz) * fz;
                float ei_dot_u = eix * ux + eiy * uy + eiz * uz;
                float ei_dot_f = eix * fx + eiy * fy + eiz * fz;
                f0[i] += prefactor * W[i] * (em_u_dot_f * 3.0f + ei_dot_u * ei_dot_f * 9.0f);
            }
        }

        #pragma unroll
        for (int i = 0; i < 19; i++) {
            int xn = (x + CX[i] + nx) % nx;
            int yn = (y + CY[i] + ny) % ny;
            int zn = (z + CZ[i] + nz) % nz;
            int dst = xn + nx * (yn + ny * zn);
            f_out[i * N + dst] = f0[i];
        }
    }

    // ---- Cell 1 (contiguous neighbor, ILP with cell 0) ----
    if (has_second) {
        int idx = base + 1;
        int x = idx % nx;
        int y = (idx / nx) % ny;
        int z = idx / (nx * ny);

        float rho_local = 0.0f, mx = 0.0f, my = 0.0f, mz = 0.0f;
        #pragma unroll
        for (int i = 0; i < 19; i++) {
            rho_local += f1[i];
            mx += CX[i] * f1[i];
            my += CY[i] * f1[i];
            mz += CZ[i] * f1[i];
        }

        float ux = 0.0f, uy = 0.0f, uz = 0.0f;
        if (finite_check(rho_local) && rho_local > 1.0e-20f) {
            float inv_rho = 1.0f / rho_local;
            ux = mx * inv_rho;
            uy = my * inv_rho;
            uz = mz * inv_rho;
        } else {
            rho_local = 1.0f;
        }

        rho_out[idx] = rho_local;
        u_out[idx]         = ux;
        u_out[N + idx]     = uy;
        u_out[2 * N + idx] = uz;

        float inv_tau = __ldg(&tau[idx]);  // stores precomputed 1/tau
        float u_sq = ux * ux + uy * uy + uz * uz;
        float base_eq = fmaf(-1.5f, u_sq, 1.0f);

        #pragma unroll
        for (int i = 0; i < 19; i++) {
            float eu = fmaf((float)CX[i], ux, fmaf((float)CY[i], uy, (float)CZ[i] * uz));
            float w_rho = W[i] * rho_local;
            float f_eq = w_rho * fmaf(fmaf(eu, 4.5f, 3.0f), eu, base_eq);
            f1[i] -= (f1[i] - f_eq) * inv_tau;
        }

        float fx = __ldg(&force[idx]);
        float fy = __ldg(&force[N + idx]);
        float fz = __ldg(&force[2 * N + idx]);
        float force_mag_sq = fx * fx + fy * fy + fz * fz;

        if (force_mag_sq >= 1e-40f) {
            float prefactor = 1.0f - 0.5f * inv_tau;
            #pragma unroll
            for (int i = 0; i < 19; i++) {
                float eix = (float)CX[i], eiy = (float)CY[i], eiz = (float)CZ[i];
                float em_u_dot_f = (eix - ux) * fx + (eiy - uy) * fy + (eiz - uz) * fz;
                float ei_dot_u = eix * ux + eiy * uy + eiz * uz;
                float ei_dot_f = eix * fx + eiy * fy + eiz * fz;
                f1[i] += prefactor * W[i] * (em_u_dot_f * 3.0f + ei_dot_u * ei_dot_f * 9.0f);
            }
        }

        #pragma unroll
        for (int i = 0; i < 19; i++) {
            int xn = (x + CX[i] + nx) % nx;
            int yn = (y + CY[i] + ny) % ny;
            int zn = (z + CZ[i] + nz) % nz;
            int dst = xn + nx * (yn + ny * zn);
            f_out[i * N + dst] = f1[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Thread-coarsened MRT kernel with float2 loads (Phase 8)
// ---------------------------------------------------------------------------
// MRT collision: 722 FMA ops/cell. Thread coarsening provides
// 2 * 722 = 1444 independent FMAs per thread, far exceeding the
// 4.54-cycle FFMA pipeline depth. The MUFU.RCP for 1/tau (41.55 cycles)
// is fully hidden by this ILP window.
//
// float2 loads: 19 float2 transactions replace 38 scalar __ldg, halving
// the LSU instruction count. Requires N = nx*ny*nz to be even.
extern "C" __global__ void __launch_bounds__(128, 6)
lbm_step_soa_mrt_coarsened(
    const float* __restrict__ f_in,
    float* __restrict__ f_out,
    float* __restrict__ rho_out,
    float* __restrict__ u_out,
    const float* __restrict__ tau,
    const float* __restrict__ force,
    int nx, int ny, int nz
) {
    int N = nx * ny * nz;
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (base >= N) return;
    int has_second = (base + 1 < N);

    // Vectorized float2 loads
    float f0[19], f1[19];
    #pragma unroll
    for (int i = 0; i < 19; i++) {
        float2 fv = __ldg(reinterpret_cast<const float2*>(&f_in[i * N + base]));
        f0[i] = finite_check(fv.x) ? fv.x : 0.0f;
        f1[i] = finite_check(fv.y) ? fv.y : 0.0f;
    }

    // ---- Cell 0: MRT collision + forcing + streaming ----
    {
        int idx = base;
        int x = idx % nx;
        int y = (idx / nx) % ny;
        int z = idx / (nx * ny);

        float rho_local = 0.0f, mx = 0.0f, my = 0.0f, mz = 0.0f;
        #pragma unroll
        for (int i = 0; i < 19; i++) {
            rho_local += f0[i];
            mx += CX[i] * f0[i];
            my += CY[i] * f0[i];
            mz += CZ[i] * f0[i];
        }

        float ux = 0.0f, uy = 0.0f, uz = 0.0f;
        if (finite_check(rho_local) && rho_local > 1.0e-20f) {
            float inv_rho = 1.0f / rho_local;
            ux = mx * inv_rho;
            uy = my * inv_rho;
            uz = mz * inv_rho;
        } else {
            rho_local = 1.0f;
        }

        rho_out[idx] = rho_local;
        u_out[idx]         = ux;
        u_out[N + idx]     = uy;
        u_out[2 * N + idx] = uz;

        float inv_tau = __ldg(&tau[idx]);  // stores precomputed 1/tau
        float fx = force == NULL ? 0.0f : __ldg(&force[idx]);
        float fy = force == NULL ? 0.0f : __ldg(&force[N + idx]);
        float fz = force == NULL ? 0.0f : __ldg(&force[2 * N + idx]);
        mrt_collision_forced_d3q19(f0, rho_local, ux, uy, uz, inv_tau, fx, fy, fz);

        #pragma unroll
        for (int i = 0; i < 19; i++) {
            int xn = (x + CX[i] + nx) % nx;
            int yn = (y + CY[i] + ny) % ny;
            int zn = (z + CZ[i] + nz) % nz;
            int dst = xn + nx * (yn + ny * zn);
            f_out[i * N + dst] = f0[i];
        }
    }

    // ---- Cell 1: MRT collision + forcing + streaming (ILP with cell 0) ----
    if (has_second) {
        int idx = base + 1;
        int x = idx % nx;
        int y = (idx / nx) % ny;
        int z = idx / (nx * ny);

        float rho_local = 0.0f, mx = 0.0f, my = 0.0f, mz = 0.0f;
        #pragma unroll
        for (int i = 0; i < 19; i++) {
            rho_local += f1[i];
            mx += CX[i] * f1[i];
            my += CY[i] * f1[i];
            mz += CZ[i] * f1[i];
        }

        float ux = 0.0f, uy = 0.0f, uz = 0.0f;
        if (finite_check(rho_local) && rho_local > 1.0e-20f) {
            float inv_rho = 1.0f / rho_local;
            ux = mx * inv_rho;
            uy = my * inv_rho;
            uz = mz * inv_rho;
        } else {
            rho_local = 1.0f;
        }

        rho_out[idx] = rho_local;
        u_out[idx]         = ux;
        u_out[N + idx]     = uy;
        u_out[2 * N + idx] = uz;

        float inv_tau = __ldg(&tau[idx]);  // stores precomputed 1/tau
        float fx = force == NULL ? 0.0f : __ldg(&force[idx]);
        float fy = force == NULL ? 0.0f : __ldg(&force[N + idx]);
        float fz = force == NULL ? 0.0f : __ldg(&force[2 * N + idx]);
        mrt_collision_forced_d3q19(f1, rho_local, ux, uy, uz, inv_tau, fx, fy, fz);

        #pragma unroll
        for (int i = 0; i < 19; i++) {
            int xn = (x + CX[i] + nx) % nx;
            int yn = (y + CY[i] + ny) % ny;
            int zn = (z + CZ[i] + nz) % nz;
            int dst = xn + nx * (yn + ny * zn);
            f_out[i * N + dst] = f1[i];
        }
    }
}

// ---------------------------------------------------------------------------
// GPU max-speed reduction (Phase 12: Mach number telemetry)
// ---------------------------------------------------------------------------
// Two-pass shared-memory tree reduction over SoA velocity field.
// Pass 1: N cells -> ceil(N/128) per-block maxima
// ---------------------------------------------------------------------------
// A-A (Alternating-Address) streaming: single-buffer, half VRAM
// ---------------------------------------------------------------------------
// Eliminates the ping-pong buffer (d_f_tmp) by using one f array with
// parity-dependent direction remapping. On even steps, read direction i
// and write to opposite direction at the neighbor. On odd steps, read
// from opposite direction and write to direction i. After two steps,
// data is back in canonical order.
//
// VRAM savings: 19*N*4 bytes (76 MB at 128^3, 608 MB at 256^3).
// Performance: ~neutral (one extra LUT lookup per direction vs saved alloc).
// Enables 256^3 on 12 GB GPUs (impossible with ping-pong).

// D3Q19 opposite-direction LUT: opp[i] = index of direction opposite to i.
// dir 0 = rest (self-opposite), dirs 1-18 come in pairs.
__constant__ int OPP[19] = {
    0,  2,  1,  4,  3,  6,  5,  8,  7, 10,
    9, 12, 11, 14, 13, 16, 15, 18, 17
};

// A-A BGK kernel: fused collision + A-A streaming in a single buffer.
//
// parity=0 (even step): read f[i * N + idx], write f[opp[i] * N + dst]
// parity=1 (odd step):  read f[opp[i] * N + src], write f[i * N + idx]
//
// On even steps, the collision reads canonical direction slots and pushes
// post-collision distributions into the opposite slot at the neighbor.
// On odd steps, the collision reads from opposite slots (where the previous
// even step deposited them) via pull from neighbors, and writes back to
// canonical slots at self.
extern "C" __global__ void lbm_step_soa_aa(
    float* __restrict__ f,            // single buffer, in-place
    float* __restrict__ rho_out,
    float* __restrict__ u_out,
    const float* __restrict__ tau,
    const float* __restrict__ force,
    int nx, int ny, int nz,
    int parity                        // 0 = even step, 1 = odd step
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nx * ny * nz;
    if (idx >= N) return;

    int x = idx % nx;
    int y = (idx / nx) % ny;
    int z = idx / (nx * ny);

    // 1. Read distributions (direction depends on parity)
    float rho_local = 0.0f;
    float mx = 0.0f, my = 0.0f, mz = 0.0f;
    float f_local[19];

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        int read_dir, src;
        if (parity == 0) {
            // Even: read canonical direction at self
            read_dir = i;
            src = idx;
        } else {
            // Odd: pull from neighbor's opposite slot
            int xn = (x - CX[i] + nx) % nx;
            int yn = (y - CY[i] + ny) % ny;
            int zn = (z - CZ[i] + nz) % nz;
            src = xn + nx * (yn + ny * zn);
            read_dir = OPP[i];
        }
        float fi = __ldg(&f[read_dir * N + src]);
        if (!finite_check(fi)) fi = 0.0f;
        f_local[i] = fi;
        rho_local += fi;
        mx += CX[i] * fi;
        my += CY[i] * fi;
        mz += CZ[i] * fi;
    }

    // 2. Macroscopic quantities
    float ux = 0.0f, uy = 0.0f, uz = 0.0f;
    if (finite_check(rho_local) && rho_local > 1.0e-20f) {
        float inv_rho = 1.0f / rho_local;
        ux = mx * inv_rho;
        uy = my * inv_rho;
        uz = mz * inv_rho;
    } else {
        rho_local = 1.0f;
    }

    rho_out[idx] = rho_local;
    u_out[idx]         = ux;
    u_out[N + idx]     = uy;
    u_out[2 * N + idx] = uz;

    // 3. BGK collision
    float inv_tau = __ldg(&tau[idx]);  // tau array stores precomputed 1/tau (host fills with inv_tau)
    float u_sq = ux * ux + uy * uy + uz * uz;
    float base = fmaf(-1.5f, u_sq, 1.0f);

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        float eu = fmaf((float)CX[i], ux, fmaf((float)CY[i], uy, (float)CZ[i] * uz));
        float w_rho = W[i] * rho_local;
        float f_eq = w_rho * fmaf(fmaf(eu, 4.5f, 3.0f), eu, base);
        f_local[i] -= (f_local[i] - f_eq) * inv_tau;
    }

    // 4. Guo forcing
    float fx = __ldg(&force[idx]);
    float fy = __ldg(&force[N + idx]);
    float fz = __ldg(&force[2 * N + idx]);
    float force_mag_sq = fx * fx + fy * fy + fz * fz;

    if (force_mag_sq >= 1e-40f) {
        float prefactor = 1.0f - 0.5f * inv_tau;
        #pragma unroll
        for (int i = 0; i < 19; i++) {
            float eix = (float)CX[i], eiy = (float)CY[i], eiz = (float)CZ[i];
            float em_u_dot_f = (eix - ux) * fx + (eiy - uy) * fy + (eiz - uz) * fz;
            float ei_dot_u = eix * ux + eiy * uy + eiz * uz;
            float ei_dot_f = eix * fx + eiy * fy + eiz * fz;
            f_local[i] += prefactor * W[i] * (em_u_dot_f * 3.0f + ei_dot_u * ei_dot_f * 9.0f);
        }
    }

    // 5. Write (direction and destination depend on parity)
    #pragma unroll
    for (int i = 0; i < 19; i++) {
        if (parity == 0) {
            // Even: push to neighbor's opposite slot
            int xn = (x + CX[i] + nx) % nx;
            int yn = (y + CY[i] + ny) % ny;
            int zn = (z + CZ[i] + nz) % nz;
            int dst = xn + nx * (yn + ny * zn);
            f[OPP[i] * N + dst] = f_local[i];
        } else {
            // Odd: write to canonical slot at self
            f[i * N + idx] = f_local[i];
        }
    }
}

// A-A MRT kernel: identical structure but uses MRT collision.
extern "C" __global__ void lbm_step_soa_mrt_aa(
    float* __restrict__ f,
    float* __restrict__ rho_out,
    float* __restrict__ u_out,
    const float* __restrict__ tau,
    const float* __restrict__ force,
    int nx, int ny, int nz,
    int parity
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nx * ny * nz;
    if (idx >= N) return;

    int x = idx % nx;
    int y = (idx / nx) % ny;
    int z = idx / (nx * ny);

    float rho_local = 0.0f;
    float mx = 0.0f, my = 0.0f, mz = 0.0f;
    float f_local[19];

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        int read_dir, src;
        if (parity == 0) {
            read_dir = i;
            src = idx;
        } else {
            int xn = (x - CX[i] + nx) % nx;
            int yn = (y - CY[i] + ny) % ny;
            int zn = (z - CZ[i] + nz) % nz;
            src = xn + nx * (yn + ny * zn);
            read_dir = OPP[i];
        }
        float fi = __ldg(&f[read_dir * N + src]);
        if (!finite_check(fi)) fi = 0.0f;
        f_local[i] = fi;
        rho_local += fi;
        mx += CX[i] * fi;
        my += CY[i] * fi;
        mz += CZ[i] * fi;
    }

    float ux = 0.0f, uy = 0.0f, uz = 0.0f;
    if (finite_check(rho_local) && rho_local > 1.0e-20f) {
        float inv_rho = 1.0f / rho_local;
        ux = mx * inv_rho;
        uy = my * inv_rho;
        uz = mz * inv_rho;
    } else {
        rho_local = 1.0f;
    }

    rho_out[idx] = rho_local;
    u_out[idx]         = ux;
    u_out[N + idx]     = uy;
    u_out[2 * N + idx] = uz;

    float inv_tau = __ldg(&tau[idx]);  // stores precomputed 1/tau
    float fx = force == NULL ? 0.0f : __ldg(&force[idx]);
    float fy = force == NULL ? 0.0f : __ldg(&force[N + idx]);
    float fz = force == NULL ? 0.0f : __ldg(&force[2 * N + idx]);
    mrt_collision_forced_d3q19(f_local, rho_local, ux, uy, uz, inv_tau, fx, fy, fz);

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        if (parity == 0) {
            int xn = (x + CX[i] + nx) % nx;
            int yn = (y + CY[i] + ny) % ny;
            int zn = (z + CZ[i] + nz) % nz;
            int dst = xn + nx * (yn + ny * zn);
            f[OPP[i] * N + dst] = f_local[i];
        } else {
            f[i * N + idx] = f_local[i];
        }
    }
}

// ---------------------------------------------------------------------------
// A-A MRT with ephemeral macroscopic writes (production variant)
// ---------------------------------------------------------------------------
// Identical to lbm_step_soa_mrt_aa but conditionally skips rho/u global
// writes when write_macroscopic == 0. This saves 32 bytes/cell of global
// memory bandwidth per step and eliminates the 4 GB macro buffer allocation
// at 512^3. Only writes macroscopic fields when the host explicitly requests
// them (e.g., for OptiX particle tracing or visualization).
//
// In isothermal LBM, rho and u are derived quantities computed in registers
// for the collision operator and then discarded. They carry no temporal state.
extern "C" __global__ void lbm_step_soa_mrt_aa_ephemeral(
    float* __restrict__ f,
    float* __restrict__ rho_out,      // may be NULL when write_macroscopic == 0
    float* __restrict__ u_out,        // may be NULL when write_macroscopic == 0
    const float* __restrict__ tau,
    const float* __restrict__ force,
    int nx, int ny, int nz,
    int parity,
    int write_macroscopic              // 0 = skip macro writes, 1 = write rho/u
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nx * ny * nz;
    if (idx >= N) return;

    int x = idx % nx;
    int y = (idx / nx) % ny;
    int z = idx / (nx * ny);

    float rho_local = 0.0f;
    float mx = 0.0f, my = 0.0f, mz = 0.0f;
    float f_local[19];

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        int read_dir, src;
        if (parity == 0) {
            read_dir = i;
            src = idx;
        } else {
            int xn = (x - CX[i] + nx) % nx;
            int yn = (y - CY[i] + ny) % ny;
            int zn = (z - CZ[i] + nz) % nz;
            src = xn + nx * (yn + ny * zn);
            read_dir = OPP[i];
        }
        float fi = __ldg(&f[read_dir * N + src]);
        if (!finite_check(fi)) fi = 0.0f;
        f_local[i] = fi;
        rho_local += fi;
        mx += CX[i] * fi;
        my += CY[i] * fi;
        mz += CZ[i] * fi;
    }

    float ux = 0.0f, uy = 0.0f, uz = 0.0f;
    if (finite_check(rho_local) && rho_local > 1.0e-20f) {
        float inv_rho = 1.0f / rho_local;
        ux = mx * inv_rho;
        uy = my * inv_rho;
        uz = mz * inv_rho;
    } else {
        rho_local = 1.0f;
    }

    // Conditionally write macroscopic fields (skip during standard advancement)
    if (write_macroscopic) {
        rho_out[idx] = rho_local;
        u_out[idx]         = ux;
        u_out[N + idx]     = uy;
        u_out[2 * N + idx] = uz;
    }

    float inv_tau = __ldg(&tau[idx]);  // stores precomputed 1/tau
    float fx = force == NULL ? 0.0f : __ldg(&force[idx]);
    float fy = force == NULL ? 0.0f : __ldg(&force[N + idx]);
    float fz = force == NULL ? 0.0f : __ldg(&force[2 * N + idx]);
    mrt_collision_forced_d3q19(f_local, rho_local, ux, uy, uz, inv_tau, fx, fy, fz);

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        if (parity == 0) {
            int xn = (x + CX[i] + nx) % nx;
            int yn = (y + CY[i] + ny) % ny;
            int zn = (z + CZ[i] + nz) % nz;
            int dst = xn + nx * (yn + ny * zn);
            f[OPP[i] * N + dst] = f_local[i];
        } else {
            f[i * N + idx] = f_local[i];
        }
    }
}

// ---------------------------------------------------------------------------
// On-demand FP16 velocity write-back (for OptiX particle tracing)
// ---------------------------------------------------------------------------
// Converts the FP32 velocity field to FP16 (half4 for alignment) on demand.
// Only dispatched when OptiX needs the velocity field for particle tracing.
// At 1024^3, FP32 velocity = 12.9 GB; FP16 = 6.4 GB.
// At 30% sparsity: FP16 velocity = 1.9 GB (transient allocation).
//
// Input:  u_f32[3*N] (SoA FP32 velocity from LBM, may be NULL if ephemeral)
// Output: u_f16[4*N] (half4 packed for alignment)
//
// If u_f32 is NULL (ephemeral mode), this kernel reads the distribution
// function directly and computes velocity on-the-fly before FP16 store.
// FP32-to-FP16 conversion via bitwise manipulation (no cuda_fp16.h dependency).
// Matches IEEE 754 half-precision: 1 sign + 5 exponent + 10 mantissa.
__device__ __forceinline__ unsigned short float_to_fp16_bits(float v) {
    unsigned int bits = __float_as_uint(v);
    unsigned int sign = (bits >> 16) & 0x8000u;
    int exp = ((bits >> 23) & 0xFF) - 127 + 15; // rebias from f32 (127) to f16 (15)
    unsigned int mant = (bits >> 13) & 0x3FFu;   // top 10 bits of f32 mantissa
    if (exp <= 0) return (unsigned short)sign;    // underflow -> zero
    if (exp >= 31) return (unsigned short)(sign | 0x7C00u); // overflow -> inf
    return (unsigned short)(sign | (exp << 10) | mant);
}

extern "C" __global__ void write_velocity_fp16(
    const float* __restrict__ u_f32,    // [3*N] FP32 velocity (SoA), or NULL
    const float* __restrict__ f_dist,   // [19*N] distribution (for on-the-fly compute)
    unsigned short* __restrict__ u_f16,  // [4*N] output FP16 as u16 (half4-aligned)
    int nx, int ny, int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nx * ny * nz;
    if (idx >= N) return;

    float ux, uy, uz;

    if (u_f32 != NULL) {
        ux = u_f32[idx];
        uy = u_f32[N + idx];
        uz = u_f32[2 * N + idx];
    } else {
        // On-the-fly: compute velocity from distribution function
        float rho = 0.0f, mx = 0.0f, my = 0.0f, mz = 0.0f;
        #pragma unroll
        for (int i = 0; i < 19; i++) {
            float fi = f_dist[i * N + idx];
            rho += fi;
            mx += CX[i] * fi;
            my += CY[i] * fi;
            mz += CZ[i] * fi;
        }
        if (rho > 1e-20f) {
            float inv_rho = 1.0f / rho;
            ux = mx * inv_rho;
            uy = my * inv_rho;
            uz = mz * inv_rho;
        } else {
            ux = 0.0f; uy = 0.0f; uz = 0.0f;
        }
    }

    // Write as half4 (u16 x 4, 4th component = 0 for alignment)
    u_f16[idx * 4 + 0] = float_to_fp16_bits(ux);
    u_f16[idx * 4 + 1] = float_to_fp16_bits(uy);
    u_f16[idx * 4 + 2] = float_to_fp16_bits(uz);
    u_f16[idx * 4 + 3] = 0;
}

// ---------------------------------------------------------------------------
// GPU max-speed reduction (Phase 12: Mach number telemetry)
// ---------------------------------------------------------------------------
// Pass 2: per-block maxima -> single max (launch with 1 block)
// Host reads back 1 float (4 bytes) instead of 24 MB sync_to_host().
//
// Uses fmaxf exclusively (branchless FMNMX hardware instruction).
extern "C" __global__ void reduce_max_speed_f32(
    const float* __restrict__ u,   // [3*N] SoA velocity
    float* __restrict__ out_max,   // [gridDim.x] per-block maxima
    int n
) {
    __shared__ float sdata[128];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Grid-stride loop: each thread reduces multiple elements
    float local_max = 0.0f;
    for (int i = gid; i < n; i += gridDim.x * blockDim.x) {
        float ux = u[i];
        float uy = u[n + i];
        float uz = u[2 * n + i];
        float speed = sqrtf(fmaf(ux, ux, fmaf(uy, uy, uz * uz)));
        local_max = fmaxf(local_max, speed);
    }

    sdata[tid] = local_max;
    __syncthreads();

    // Tree reduction with fmaxf (branchless FMNMX)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_max[blockIdx.x] = sdata[0];
    }
}

// ---------------------------------------------------------------------------
// Shared-memory tiled Smagorinsky LES tau computation
// ---------------------------------------------------------------------------
// Copies the proven 8x8x4 tile pattern from lbm_step_soa_tiled.
// Loads ux, uy, uz into shared memory with 1-cell halo, computes all 9
// velocity gradient components from shared memory (~20-cycle LDS vs
// ~200-cycle L2 miss for the 18 scattered global reads in the untiled kernel).
//
// Shared memory budget: 3 components * 10*10*6 * 4B = 7200B = 7.2 KB per block.
// Ada Lovelace has 96 KB per SM -- two blocks/SM use 14.4 KB. Negligible.

#define SMAG_TX 8
#define SMAG_TY 8
#define SMAG_TZ 4
#define SMAG_PX (SMAG_TX + 2)
#define SMAG_PY (SMAG_TY + 2)
#define SMAG_PZ (SMAG_TZ + 2)
#define SMAG_PVOL (SMAG_PX * SMAG_PY * SMAG_PZ)

extern "C" __global__ void __launch_bounds__(256)
compute_smagorinsky_tau_tiled(
    const float* __restrict__ u,      // [3*N] velocity (SoA)
    float* __restrict__ tau_out,      // [N] output tau
    float tau_base,
    float cs_sq_dx_sq,               // (C_s * dx)^2, precomputed on host
    int nx, int ny, int nz
) {
    __shared__ float s_ux[SMAG_PVOL];
    __shared__ float s_uy[SMAG_PVOL];
    __shared__ float s_uz[SMAG_PVOL];

    int N = nx * ny * nz;
    int bx0 = blockIdx.x * SMAG_TX;
    int by0 = blockIdx.y * SMAG_TY;
    int bz0 = blockIdx.z * SMAG_TZ;
    int tid = threadIdx.x + SMAG_TX * (threadIdx.y + SMAG_TY * threadIdx.z);

    // Phase 1: cooperative halo load (same striped pattern as lbm_step_soa_tiled)
    for (int i = tid; i < SMAG_PVOL; i += 256) {
        int lz = i / (SMAG_PX * SMAG_PY);
        int ly = (i % (SMAG_PX * SMAG_PY)) / SMAG_PX;
        int lx = i % SMAG_PX;
        int gx = (bx0 + lx - 1 + nx) % nx;
        int gy = (by0 + ly - 1 + ny) % ny;
        int gz = (bz0 + lz - 1 + nz) % nz;
        int gidx = gx + nx * (gy + ny * gz);
        s_ux[i] = __ldg(&u[gidx]);
        s_uy[i] = __ldg(&u[N + gidx]);
        s_uz[i] = __ldg(&u[2 * N + gidx]);
    }
    __syncthreads();

    // Phase 2: boundary guard
    int gx = bx0 + (int)threadIdx.x;
    int gy = by0 + (int)threadIdx.y;
    int gz = bz0 + (int)threadIdx.z;
    if (gx >= nx || gy >= ny || gz >= nz) return;

    int idx = gx + nx * (gy + ny * gz);

    // Shared-memory coordinates (offset by +1 for halo)
    int sx = (int)threadIdx.x + 1;
    int sy = (int)threadIdx.y + 1;
    int sz = (int)threadIdx.z + 1;

    // Phase 3: compute 3x3 velocity gradient tensor from shared memory
    #define SMAG_IDX(lx, ly, lz) ((lx) + SMAG_PX * ((ly) + SMAG_PY * (lz)))

    float dux_dx = (s_ux[SMAG_IDX(sx+1,sy,sz)] - s_ux[SMAG_IDX(sx-1,sy,sz)]) * 0.5f;
    float dux_dy = (s_ux[SMAG_IDX(sx,sy+1,sz)] - s_ux[SMAG_IDX(sx,sy-1,sz)]) * 0.5f;
    float dux_dz = (s_ux[SMAG_IDX(sx,sy,sz+1)] - s_ux[SMAG_IDX(sx,sy,sz-1)]) * 0.5f;

    float duy_dx = (s_uy[SMAG_IDX(sx+1,sy,sz)] - s_uy[SMAG_IDX(sx-1,sy,sz)]) * 0.5f;
    float duy_dy = (s_uy[SMAG_IDX(sx,sy+1,sz)] - s_uy[SMAG_IDX(sx,sy-1,sz)]) * 0.5f;
    float duy_dz = (s_uy[SMAG_IDX(sx,sy,sz+1)] - s_uy[SMAG_IDX(sx,sy,sz-1)]) * 0.5f;

    float duz_dx = (s_uz[SMAG_IDX(sx+1,sy,sz)] - s_uz[SMAG_IDX(sx-1,sy,sz)]) * 0.5f;
    float duz_dy = (s_uz[SMAG_IDX(sx,sy+1,sz)] - s_uz[SMAG_IDX(sx,sy-1,sz)]) * 0.5f;
    float duz_dz = (s_uz[SMAG_IDX(sx,sy,sz+1)] - s_uz[SMAG_IDX(sx,sy,sz-1)]) * 0.5f;

    #undef SMAG_IDX

    // Symmetric strain rate tensor S_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
    float s11 = dux_dx;
    float s22 = duy_dy;
    float s33 = duz_dz;
    float s12 = 0.5f * (dux_dy + duy_dx);
    float s13 = 0.5f * (dux_dz + duz_dx);
    float s23 = 0.5f * (duy_dz + duz_dy);

    // |S| = sqrt(2 * S_ij * S_ij) (Frobenius norm of strain rate)
    float s_mag = sqrtf(2.0f * (s11*s11 + s22*s22 + s33*s33
                                + 2.0f*(s12*s12 + s13*s13 + s23*s23)));

    // nu_turb = (C_s * dx)^2 * |S|, tau = tau_base + 3 * nu_turb
    float tau_new = tau_base + 3.0f * cs_sq_dx_sq * s_mag;

    // Phase 4: coalesced write to tau_out, clamped to stability range, then 1/tau
    float clamped_tau = fmaxf(0.505f, fminf(5.0f, tau_new));
    tau_out[idx] = 1.0f / clamped_tau;
}

// ---------------------------------------------------------------------------
// Thread-coarsened BGK kernel with float4 loads (Ada Lovelace Optimized)
// ---------------------------------------------------------------------------
// 1D coarsening: each thread processes 4 contiguous cells (idx, idx+1, idx+2, idx+3).
// float4 vectorized loads maximize 128-bit memory bus utilization.
// Requires N = nx*ny*nz to be a multiple of 4.
extern "C" __global__ void __launch_bounds__(128, 4)
lbm_step_soa_coarsened_float4(
    const float* __restrict__ f_in,
    float* __restrict__ f_out,
    float* __restrict__ rho_out,
    float* __restrict__ u_out,
    const float* __restrict__ tau,
    const float* __restrict__ force,
    int nx, int ny, int nz
) {
    int N = nx * ny * nz;
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (base >= N) return;
    
    float f0[19], f1[19], f2[19], f3[19];
    #pragma unroll
    for (int i = 0; i < 19; i++) {
        float4 fv = __ldg(reinterpret_cast<const float4*>(&f_in[i * N + base]));
        f0[i] = finite_check(fv.x) ? fv.x : 0.0f;
        f1[i] = finite_check(fv.y) ? fv.y : 0.0f;
        f2[i] = finite_check(fv.z) ? fv.z : 0.0f;
        f3[i] = finite_check(fv.w) ? fv.w : 0.0f;
    }

    #pragma unroll
    for (int c = 0; c < 4; c++) {
        int idx = base + c;
        if (idx >= N) break;

        int x = idx % nx;
        int y = (idx / nx) % ny;
        int z = idx / (nx * ny);

        float rho_local = 0.0f, mx = 0.0f, my = 0.0f, mz = 0.0f;
        float* f_local = (c == 0) ? f0 : (c == 1) ? f1 : (c == 2) ? f2 : f3;

        #pragma unroll
        for (int i = 0; i < 19; i++) {
            rho_local += f_local[i];
            mx += CX[i] * f_local[i];
            my += CY[i] * f_local[i];
            mz += CZ[i] * f_local[i];
        }

        float ux = 0.0f, uy = 0.0f, uz = 0.0f;
        if (finite_check(rho_local) && rho_local > 1.0e-20f) {
            float inv_rho = 1.0f / rho_local;
            ux = mx * inv_rho;
            uy = my * inv_rho;
            uz = mz * inv_rho;
        } else {
            rho_local = 1.0f;
        }

        rho_out[idx] = rho_local;
        u_out[idx]         = ux;
        u_out[N + idx]     = uy;
        u_out[2 * N + idx] = uz;

        float inv_tau = __ldg(&tau[idx]);  // stores precomputed 1/tau
        float u_sq = ux * ux + uy * uy + uz * uz;
        float base_eq = fmaf(-1.5f, u_sq, 1.0f);

        #pragma unroll
        for (int i = 0; i < 19; i++) {
            float eu = fmaf((float)CX[i], ux, fmaf((float)CY[i], uy, (float)CZ[i] * uz));
            float w_rho = W[i] * rho_local;
            float f_eq = w_rho * fmaf(fmaf(eu, 4.5f, 3.0f), eu, base_eq);
            f_local[i] -= (f_local[i] - f_eq) * inv_tau;
        }

        float fx = __ldg(&force[idx]);
        float fy = __ldg(&force[N + idx]);
        float fz = __ldg(&force[2 * N + idx]);
        float force_mag_sq = fx * fx + fy * fy + fz * fz;

        if (force_mag_sq >= 1e-40f) {
            float prefactor = 1.0f - 0.5f * inv_tau;
            #pragma unroll
            for (int i = 0; i < 19; i++) {
                float eix = (float)CX[i], eiy = (float)CY[i], eiz = (float)CZ[i];
                float em_u_dot_f = (eix - ux) * fx + (eiy - uy) * fy + (eiz - uz) * fz;
                float ei_dot_u = eix * ux + eiy * uy + eiz * uz;
                float ei_dot_f = eix * fx + eiy * fy + eiz * fz;
                f_local[i] += prefactor * W[i] * (em_u_dot_f * 3.0f + ei_dot_u * ei_dot_f * 9.0f);
            }
        }

        #pragma unroll
        for (int i = 0; i < 19; i++) {
            int xn = (x + CX[i] + nx) % nx;
            int yn = (y + CY[i] + ny) % ny;
            int zn = (z + CZ[i] + nz) % nz;
            int dst = xn + nx * (yn + ny * zn);
            f_out[i * N + dst] = f_local[i];
        }
    }
}

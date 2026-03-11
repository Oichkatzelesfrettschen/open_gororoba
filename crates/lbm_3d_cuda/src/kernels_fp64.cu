// FP64 (double-precision) D3Q19 LBM kernels for multi-precision comparison.
// Mechanical translation of kernels.cu: float -> double, f-suffix removed.
// WHY: Reference implementation to determine whether ghost spectral artifacts
// are BF16 quantization noise or genuine physics (Sprint 50B, E-069).

// D3Q19 lattice velocities (shared with FP32/BF16 kernels)
__constant__ int D3Q19_CX_D[19] = {
    0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0
};
__constant__ int D3Q19_CY_D[19] = {
    0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1
};
__constant__ int D3Q19_CZ_D[19] = {
    0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1
};

// D3Q19 weights (double precision -- no f suffix)
__constant__ double D3Q19_WD[19] = {
    1.0/3.0,                          // i=0 (rest)
    1.0/18.0, 1.0/18.0, 1.0/18.0,    // i=1-6 (face neighbors)
    1.0/18.0, 1.0/18.0, 1.0/18.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0,    // i=7-18 (edge neighbors)
    1.0/36.0, 1.0/36.0, 1.0/36.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0
};

// Speed of sound squared (double)
__device__ const double CS_SQ_D = 1.0 / 3.0;

__device__ __forceinline__ bool finite_f64(double x) {
    return (x == x) && (x <= 1.7976931348623157e308) && (x >= -1.7976931348623157e308);
}

// Compute equilibrium distribution (double) -- FMA-optimized Horner form.
// Algebraic identity: f_eq = w*rho * (4.5*eu^2 + 3*eu + 1 - 1.5*usq)
// Horner: (4.5*eu + 3)*eu + base, where base = 1 - 1.5*usq
__device__ void compute_equilibrium_d(
    double* f_eq,
    double rho,
    const double* u
) {
    double u_sq = u[0]*u[0] + u[1]*u[1] + u[2]*u[2];
    double base = fma(-1.5, u_sq, 1.0);  // 1 - 1.5*usq

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        double eu = (double)(D3Q19_CX_D[i])*u[0]
                  + (double)(D3Q19_CY_D[i])*u[1]
                  + (double)(D3Q19_CZ_D[i])*u[2];
        double w_rho = D3Q19_WD[i] * rho;
        // Horner evaluation: (4.5*eu + 3)*eu + base
        f_eq[i] = w_rho * fma(fma(eu, 4.5, 3.0), eu, base);
    }
}

// Fused Collision + Streaming + Guo Forcing (FP64)
extern "C" __global__ void lbm_step_fused_fp64_kernel(
    const double* f_in,
    double* f_out,
    double* rho_out,
    double* u_out,
    const double* force,
    const double* tau,
    int nx, int ny, int nz
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    int idx = x + nx * (y + ny * z);

    // 1. Gather macroscopic (use __ldg for read-only f_in)
    double rho_local = 0.0;
    double mx = 0.0, my = 0.0, mz = 0.0;
    double f_local[19];

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        double val = __ldg(&f_in[idx * 19 + i]);
        if (!finite_f64(val)) {
            val = 0.0;
        }
        f_local[i] = val;
        rho_local += val;
        mx += D3Q19_CX_D[i] * val;
        my += D3Q19_CY_D[i] * val;
        mz += D3Q19_CZ_D[i] * val;
    }

    double ux = 0.0;
    double uy = 0.0;
    double uz = 0.0;
    if (finite_f64(rho_local) && rho_local > 1.0e-20) {
        double inv_rho = 1.0 / rho_local;
        ux = mx * inv_rho;
        uy = my * inv_rho;
        uz = mz * inv_rho;
    } else {
        rho_local = 1.0;
    }

    rho_out[idx] = rho_local;
    u_out[idx * 3 + 0] = ux;
    u_out[idx * 3 + 1] = uy;
    u_out[idx * 3 + 2] = uz;

    // 2. Collision + Forcing
    double f_eq[19];
    double u_vec[3] = {ux, uy, uz};
    compute_equilibrium_d(f_eq, rho_local, u_vec);

    double tau_local = __ldg(&tau[idx]);
    double inv_tau = 1.0 / tau_local;

    double fx = __ldg(&force[idx * 3 + 0]);
    double fy = __ldg(&force[idx * 3 + 1]);
    double fz = __ldg(&force[idx * 3 + 2]);

    // Occupancy culling: check if Guo forcing is needed.
    double force_mag_sq = fx * fx + fy * fy + fz * fz;
    double prefactor = 1.0 - 0.5 * inv_tau;

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        // BGK collision
        double fi = f_local[i] - (f_local[i] - f_eq[i]) * inv_tau;

        // Guo Forcing (skip for cells with negligible force)
        if (force_mag_sq >= 1e-40) {
            double eix = (double)D3Q19_CX_D[i];
            double eiy = (double)D3Q19_CY_D[i];
            double eiz = (double)D3Q19_CZ_D[i];
            double ei_minus_u_dot_f = (eix - ux) * fx + (eiy - uy) * fy + (eiz - uz) * fz;
            double ei_dot_u = eix * ux + eiy * uy + eiz * uz;
            double ei_dot_f = eix * fx + eiy * fy + eiz * fz;
            double s_i = ei_minus_u_dot_f * 3.0 + ei_dot_u * ei_dot_f * 9.0;
            fi += prefactor * D3Q19_WD[i] * s_i;
        }

        // 3. Streaming (Write to neighbor -- always executes)
        int x_next = (x + D3Q19_CX_D[i] + nx) % nx;
        int y_next = (y + D3Q19_CY_D[i] + ny) % ny;
        int z_next = (z + D3Q19_CZ_D[i] + nz) % nz;
        int idx_next = x_next + nx * (y_next + ny * z_next);

        f_out[idx_next * 19 + i] = fi;
    }
}

// Initialize uniform density and velocity (FP64)
extern "C" __global__ void initialize_uniform_fp64_kernel(
    double* f,
    double* rho,
    double* u,
    double rho_init,
    double ux_init,
    double uy_init,
    double uz_init,
    int nx, int ny, int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_cells = nx * ny * nz;
    if (idx >= n_cells) return;

    rho[idx] = rho_init;
    u[idx * 3 + 0] = ux_init;
    u[idx * 3 + 1] = uy_init;
    u[idx * 3 + 2] = uz_init;

    double u_local[3] = {ux_init, uy_init, uz_init};
    double f_eq[19];
    compute_equilibrium_d(f_eq, rho_init, u_local);

    for (int i = 0; i < 19; i++) f[idx * 19 + i] = f_eq[i];
}

// Initialize per-cell density and velocity (FP64)
extern "C" __global__ void initialize_custom_fp64_kernel(
    double* f,
    double* rho,
    double* u,
    const double* rho_in,
    const double* u_in,
    int nx, int ny, int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_cells = nx * ny * nz;
    if (idx >= n_cells) return;

    double rho_init = rho_in[idx];
    double ux_init = u_in[idx * 3 + 0];
    double uy_init = u_in[idx * 3 + 1];
    double uz_init = u_in[idx * 3 + 2];

    rho[idx] = rho_init;
    u[idx * 3 + 0] = ux_init;
    u[idx * 3 + 1] = uy_init;
    u[idx * 3 + 2] = uz_init;

    double u_local[3] = {ux_init, uy_init, uz_init};
    double f_eq[19];
    compute_equilibrium_d(f_eq, rho_init, u_local);
    for (int i = 0; i < 19; i++) f[idx * 19 + i] = f_eq[i];
}

// Reduce sum (double atomicAdd -- native on sm_60+)
extern "C" __global__ void reduce_sum_fp64_kernel(
    const double* input,
    double* output,
    int n_cells
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n_cells; i += blockDim.x * gridDim.x) {
        atomicAdd(output, input[i]);
    }
}

// Zero out a scalar double
extern "C" __global__ void zero_fp64_kernel(double* out) {
    *out = 0.0;
}

// Enstrophy computation (FP64) -- reads from double u[]
extern "C" __global__ void compute_enstrophy_cell_fp64_kernel(
    const double* u,
    double* enstrophy_field,
    int nx, int ny, int nz
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    int idx = x + nx * (y + ny * z);

    int xp = (x + 1) % nx; int xm = (x + nx - 1) % nx;
    int yp = (y + 1) % ny; int ym = (y + ny - 1) % ny;
    int zp = (z + 1) % nz; int zm = (z + nz - 1) % nz;

    auto get_u = [&](int xi, int yi, int zi, int comp) {
        return u[(xi + nx * (yi + ny * zi)) * 3 + comp];
    };

    double duz_dy = (get_u(x, yp, z, 2) - get_u(x, ym, z, 2)) * 0.5;
    double duy_dz = (get_u(x, y, zp, 1) - get_u(x, y, zm, 1)) * 0.5;
    double dux_dz = (get_u(x, y, zp, 0) - get_u(x, y, zm, 0)) * 0.5;
    double duz_dx = (get_u(xp, y, z, 2) - get_u(xm, y, z, 2)) * 0.5;
    double duy_dx = (get_u(xp, y, z, 1) - get_u(xm, y, z, 1)) * 0.5;
    double dux_dy = (get_u(x, yp, z, 0) - get_u(x, ym, z, 0)) * 0.5;

    double wx = duz_dy - duy_dz;
    double wy = dux_dz - duz_dx;
    double wz = duy_dx - dux_dy;

    enstrophy_field[idx] = wx*wx + wy*wy + wz*wz;
}

// Convert real velocity to complex (FP64 -> FP32 complex for cuFFT compat)
// Note: cuFFT operates on float complex; this bridges FP64 u[] to FP32 complex
struct ComplexDeviceF {
    float re;
    float im;
};

extern "C" __global__ void convert_real_fp64_to_complex_f32_kernel(
    const double* u,
    ComplexDeviceF* u_hat,
    int comp,
    int n_cells
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;
    u_hat[idx].re = (float)u[idx * 3 + comp];
    u_hat[idx].im = 0.0f;
}

extern "C" __global__ void convert_complex_f32_to_real_fp64_kernel(
    const ComplexDeviceF* u_hat,
    double* u,
    int comp,
    float scale,
    int n_cells
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;
    u[idx * 3 + comp] = (double)(u_hat[idx].re * scale);
}

// Apply spectral mask (works on FP32 complex, same as FP32 kernel)
extern "C" __global__ void apply_spectral_mask_fp64_kernel(
    ComplexDeviceF* u_hat,
    const float* mask,
    float damping,
    int n_cells
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;
    if (mask[idx] < 0.5f) {
        u_hat[idx].re *= damping;
        u_hat[idx].im *= damping;
    }
}

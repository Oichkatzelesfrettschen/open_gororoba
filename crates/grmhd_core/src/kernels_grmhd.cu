// GRMHD 3D FP64 CUDA kernels for 256^3 finite-volume solver.
//
// Implements the Gammie et al. (2003) GRMHD equations on a logr-theta-phi
// grid with Kerr-Schild metric.  Designed for cudarc NVRTC compilation.
//
// Data layout: SoA (Structure of Arrays) for coalesced memory access.
// Each conserved variable is a separate 3D array [N1 * N2 * N3].
//
// Coordinate convention:
//   dir=0: r-direction
//   dir=1: theta-direction
//   dir=2: phi-direction

// === Constants (set by host at launch via template args) ===
#define NPRIM 8  // rho, u, v1, v2, v3, B1, B2, B3
#define NCONS 8  // D, E, S1, S2, S3, B1, B2, B3

// Primitive variable indices
#define RHO 0
#define UU  1
#define V1  2
#define V2  3
#define V3  4
#define B1  5
#define B2  6
#define B3  7

// Gamma-law EOS parameter (passed as kernel arg)
// P = (gamma - 1) * u

// ============================================================
// Metric precomputation kernel
// ============================================================
// Computes gcov, gcon, sqrt_neg_g, lapse at each grid point.
// Called ONCE at initialization, stored in __constant__ or global memory.

extern "C" __global__ void precompute_metric_kernel(
    double* __restrict__ gcov,    // [5 * N1 * N2] output
    double* __restrict__ gcon,    // [5 * N1 * N2] output
    double* __restrict__ sqrt_g,  // [N1 * N2] output
    double* __restrict__ lapse,   // [N1 * N2] output
    const double a,               // Kerr spin parameter
    const double r_min,
    const double r_max,
    const int N1,
    const int N2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N1 * N2;
    if (idx >= total) return;

    int i = idx / N2;
    int j = idx % N2;

    // Log-radial coordinate
    double xi = (double)i / (double)(N1 - 1);
    double r = r_min * exp(xi * log(r_max / r_min));
    double th = M_PI * ((double)j + 0.5) / (double)N2;

    double sth = sin(th);
    double cth = cos(th);
    double sigma = r * r + a * a * cth * cth;
    double delta = r * r - 2.0 * r + a * a;
    double A_met = (r * r + a * a) * (r * r + a * a) - delta * a * a * sth * sth;

    // Covariant metric: g_tt, g_rr, g_thth, g_phph, g_tph
    int base = 5 * idx;
    gcov[base + 0] = -(1.0 - 2.0 * r / sigma);               // g_tt
    gcov[base + 1] = sigma / delta;                             // g_rr
    gcov[base + 2] = sigma;                                     // g_thth
    gcov[base + 3] = A_met * sth * sth / sigma;                // g_phph
    gcov[base + 4] = -2.0 * a * r * sth * sth / sigma;        // g_tph

    // Contravariant metric
    double det = gcov[base + 0] * gcov[base + 3] - gcov[base + 4] * gcov[base + 4];
    double inv_det = 1.0 / fmax(fabs(det), 1e-40);
    gcon[base + 0] = gcov[base + 3] * inv_det;    // g^tt (with flipped sign from det)
    gcon[base + 1] = delta / sigma;                 // g^rr
    gcon[base + 2] = 1.0 / sigma;                  // g^thth
    gcon[base + 3] = gcov[base + 0] * inv_det;    // g^phph
    gcon[base + 4] = -gcov[base + 4] * inv_det;   // g^tph

    // sqrt(-g) for Kerr-Schild
    sqrt_g[idx] = sigma * fabs(sth);

    // Lapse: alpha = 1/sqrt(-g^tt)
    double g_tt_up = -A_met / (sigma * delta);
    lapse[idx] = 1.0 / sqrt(fmax(-g_tt_up, 1e-40));
}

// ============================================================
// Flux computation kernel (the hot path)
// ============================================================
// Computes F^dir at each cell center using cached metric.
// One thread per cell.

extern "C" __global__ void compute_flux_kernel(
    const double* __restrict__ prims,    // [NPRIM * N_total] SoA
    const double* __restrict__ gcov,     // [5 * N1 * N2] cached
    const double* __restrict__ sqrt_g,   // [N1 * N2] cached
    double* __restrict__ flux,           // [NCONS * N_total] SoA output
    const double gam_m1,                 // gamma - 1 (EOS)
    const int dir,                       // 0=r, 1=theta, 2=phi
    const int N1,
    const int N2,
    const int N3,
    const int N_total                    // N1 * N2 * N3
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= N_total) return;

    // Decode 3D index (k varies fastest for phi-periodic coalescing)
    int tmp = cell;
    int k = tmp % N3; tmp /= N3;
    int j = tmp % N2; tmp /= N2;
    int i = tmp;

    // Metric index (r, theta only -- phi-independent for Kerr)
    int met_idx = i * N2 + j;

    // Load primitives (SoA -> registers)
    double rho = prims[RHO * N_total + cell];
    double u   = prims[UU  * N_total + cell];
    double v1  = prims[V1  * N_total + cell];
    double v2  = prims[V2  * N_total + cell];
    double v3  = prims[V3  * N_total + cell];
    double b1  = prims[B1  * N_total + cell];
    double b2  = prims[B2  * N_total + cell];
    double b3  = prims[B3  * N_total + cell];

    double pressure = gam_m1 * u;

    // Load cached metric
    int gb = 5 * met_idx;
    double g_tt   = gcov[gb + 0];
    double g_rr   = gcov[gb + 1];
    double g_thth = gcov[gb + 2];
    double g_phph = gcov[gb + 3];
    double g_tph  = gcov[gb + 4];
    double sg     = sqrt_g[met_idx];

    // 3-velocity squared and Lorentz factor
    double vsq = g_rr * v1 * v1 + g_thth * v2 * v2 + g_phph * v3 * v3;
    double alpha_sq = -(g_tt + 2.0 * g_tph * v3 + vsq);
    double ut = (alpha_sq > 1e-20) ? rsqrt(alpha_sq) : 1.0;
    double inv_ut = 1.0 / ut;
    double inv_ut_sq = inv_ut * inv_ut;

    // Covariant 4-velocity
    double u_r  = g_rr   * ut * v1;
    double u_th = g_thth * ut * v2;
    double u_ph = g_phph * ut * v3 + g_tph * ut;
    double u_t  = g_tt   * ut + g_tph * ut * v3;

    // Magnetic 4-vector
    double bt = (b1 * u_r + b2 * u_th + b3 * u_ph) * inv_ut;
    double bsq = fmax((b1*b1*g_rr + b2*b2*g_thth + b3*b3*g_phph) * inv_ut_sq
                      + bt*bt*(-inv_ut_sq + vsq), 0.0);

    double w = rho + u + pressure + bsq;
    double ptot = pressure + 0.5 * bsq;

    // u^dir and b^dir (contravariant spatial)
    double v_arr[3] = {v1, v2, v3};
    double b_arr[3] = {b1, b2, b3};
    double u_cov[3] = {u_r, u_th, u_ph};
    double g_diag[3] = {g_rr, g_thth, g_phph};

    double v_dir = v_arr[dir];
    double u_up_dir = ut * v_dir;
    double b_up_dir = (b_arr[dir] + bt * ut * v_arr[dir]) * inv_ut;

    // Mass flux
    flux[0 * N_total + cell] = sg * rho * u_up_dir;

    // Energy flux
    double b_cov_t = g_tt * bt;
    if (dir == 2) b_cov_t += g_tph * b_up_dir;
    double t_dir_t = w * u_up_dir * u_t - b_up_dir * b_cov_t;
    flux[1 * N_total + cell] = sg * (t_dir_t + rho * u_up_dir);

    // Momentum flux: T^dir_j for j=0,1,2
    for (int jj = 0; jj < 3; jj++) {
        double b_up_j = (b_arr[jj] + bt * ut * v_arr[jj]) * inv_ut;
        double b_cov_j = g_diag[jj] * b_up_j;
        double delta_jd = (jj == dir) ? 1.0 : 0.0;
        flux[(2 + jj) * N_total + cell] = sg * (w * u_up_dir * u_cov[jj]
                                                 + ptot * delta_jd
                                                 - b_up_dir * b_cov_j);
    }

    // Induction flux: v^dir * B^j - v^j * B^dir
    double b_dir_val = b_arr[dir];
    for (int jj = 0; jj < 3; jj++) {
        flux[(5 + jj) * N_total + cell] = sg * (v_dir * b_arr[jj]
                                                 - v_arr[jj] * b_dir_val);
    }
}

// ============================================================
// Prim2con kernel
// ============================================================
extern "C" __global__ void prim2con_kernel(
    const double* __restrict__ prims,
    const double* __restrict__ gcov,
    const double* __restrict__ sqrt_g,
    double* __restrict__ cons,
    const double gam_m1,
    const int N1, const int N2, const int N3,
    const int N_total
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= N_total) return;

    int tmp = cell;
    int _k = tmp % N3; tmp /= N3;
    int j = tmp % N2; tmp /= N2;
    int i = tmp;
    int met_idx = i * N2 + j;

    double rho = prims[RHO * N_total + cell];
    double u   = prims[UU  * N_total + cell];
    double v1  = prims[V1  * N_total + cell];
    double v2  = prims[V2  * N_total + cell];
    double v3  = prims[V3  * N_total + cell];
    double b1  = prims[B1  * N_total + cell];
    double b2  = prims[B2  * N_total + cell];
    double b3  = prims[B3  * N_total + cell];
    double pressure = gam_m1 * u;

    int gb = 5 * met_idx;
    double g_tt   = gcov[gb + 0];
    double g_rr   = gcov[gb + 1];
    double g_thth = gcov[gb + 2];
    double g_phph = gcov[gb + 3];
    double g_tph  = gcov[gb + 4];
    double sg     = sqrt_g[met_idx];

    double vsq = g_rr*v1*v1 + g_thth*v2*v2 + g_phph*v3*v3;
    double alpha_sq = -(g_tt + 2.0*g_tph*v3 + vsq);
    double ut = (alpha_sq > 1e-20) ? rsqrt(alpha_sq) : 1.0;
    double inv_ut = 1.0 / ut;

    double u_r  = g_rr   * ut * v1;
    double u_th = g_thth * ut * v2;
    double u_ph = g_phph * ut * v3 + g_tph * ut;
    double u_t  = g_tt   * ut + g_tph * ut * v3;

    double bt = (b1*u_r + b2*u_th + b3*u_ph) * inv_ut;
    double bsq = fmax((b1*b1*g_rr + b2*b2*g_thth + b3*b3*g_phph) / (ut*ut)
                      + bt*bt*(-1.0/(ut*ut) + vsq), 0.0);

    double w = rho + u + pressure + bsq;
    double ptot = pressure + 0.5*bsq;

    // D = sqrt_g * rho * u^t
    cons[0 * N_total + cell] = sg * rho * ut;
    // E = sqrt_g * (T^t_t + rho * u^t)
    double T_t_t = w * ut * u_t + ptot - bt * (g_tt*bt + g_tph*0.0 /* simplify */);
    cons[1 * N_total + cell] = sg * (T_t_t + rho * ut);
    // S_j = sqrt_g * T^t_j
    cons[2 * N_total + cell] = sg * (w * ut * u_r);
    cons[3 * N_total + cell] = sg * (w * ut * u_th);
    cons[4 * N_total + cell] = sg * (w * ut * u_ph);
    // B^j (cell-centered, div-clean via CT)
    cons[5 * N_total + cell] = sg * b1;
    cons[6 * N_total + cell] = sg * b2;
    cons[7 * N_total + cell] = sg * b3;
}

// ============================================================
// Euler update kernel
// ============================================================
extern "C" __global__ void euler_update_kernel(
    const double* __restrict__ U_old,
    const double* __restrict__ rhs,
    double* __restrict__ U_new,
    const double dt,
    const int N_total_vars  // NCONS * N_total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_total_vars) return;
    U_new[idx] = U_old[idx] + dt * rhs[idx];
}

// ============================================================
// Flux difference kernel (RHS accumulation)
// ============================================================
// Computes dU/dt += -(F_{i+1/2} - F_{i-1/2}) / dx for one direction.
// Uses simple upwind differencing on the pre-computed flux arrays.

extern "C" __global__ void flux_divergence_kernel(
    const double* __restrict__ flux,   // [NCONS * N_total]
    double* __restrict__ rhs,          // [NCONS * N_total] accumulated
    const double inv_dx,               // 1.0 / dx for this direction
    const int dir,                     // 0=r, 1=theta, 2=phi
    const int N1, const int N2, const int N3,
    const int N_total
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= N_total) return;

    int tmp = cell;
    int k = tmp % N3; tmp /= N3;
    int j = tmp % N2; tmp /= N2;
    int i = tmp;

    // Strides for each direction
    int stride;
    int idx_m, idx_p;
    bool at_boundary;

    if (dir == 0) {
        stride = N2 * N3;
        at_boundary = (i == 0 || i >= N1 - 1);
        idx_m = cell - stride;
        idx_p = cell + stride;
    } else if (dir == 1) {
        stride = N3;
        at_boundary = (j == 0 || j >= N2 - 1);
        idx_m = cell - stride;
        idx_p = cell + stride;
    } else {
        stride = 1;
        // Periodic in phi
        at_boundary = false;
        idx_m = (k == 0) ? (cell + N3 - 1) : (cell - 1);
        idx_p = (k == N3 - 1) ? (cell - N3 + 1) : (cell + 1);
    }

    if (at_boundary) return;

    for (int var = 0; var < NCONS; var++) {
        int offset = var * N_total;
        double f_p = flux[offset + idx_p];
        double f_m = flux[offset + idx_m];
        // Central difference: dF/dx ~ (F_{i+1} - F_{i-1}) / (2*dx)
        rhs[offset + cell] -= 0.5 * inv_dx * (f_p - f_m);
    }
}

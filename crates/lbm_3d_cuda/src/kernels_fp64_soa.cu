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

// ---------------------------------------------------------------------------
// MRT collision device function (d'Humieres D3Q19, FP64 math)
// Standalone copy for NVRTC compilation unit isolation.
// FP64 variant: all float->double, fmaf->fma for full double precision.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mrt_collision_d3q19_fp64(
    double f[19], double rho, double ux, double uy, double uz, double tau_local
) {
    double s_nu = 1.0 / tau_local, s_e = 1.19, s_eps = 1.4, s_q = 1.2, s_ghost = 1.0;
    double u_sq = ux*ux + uy*uy + uz*uz;
    double ax_p = f[1]+f[2], ax_m = f[1]-f[2], ay_p = f[3]+f[4], ay_m = f[3]-f[4];
    double az_p = f[5]+f[6], az_m = f[5]-f[6];
    double d78_p = f[7]+f[8], d78_m = f[7]-f[8], d910_p = f[9]+f[10], d910_m = f[9]-f[10];
    double d1112_p = f[11]+f[12], d1112_m = f[11]-f[12], d1314_p = f[13]+f[14], d1314_m = f[13]-f[14];
    double d1516_p = f[15]+f[16], d1516_m = f[15]-f[16], d1718_p = f[17]+f[18], d1718_m = f[17]-f[18];
    double axis_sum = ax_p+ay_p+az_p;
    double diag_sum = d78_p+d910_p+d1112_p+d1314_p+d1516_p+d1718_p;
    double xy_diag = d78_p+d910_p+d1112_p+d1314_p, z_diag = d1516_p+d1718_p;
    double m0 = f[0]+axis_sum+diag_sum;
    double m3 = ax_m+d78_m+d910_m+d1112_m+d1314_m;
    double m5 = ay_m+d78_m-d910_m+d1516_m+d1718_m;
    double m7 = az_m+d1112_m-d1314_m+d1516_m-d1718_m;
    double m1 = fma(-30.0,f[0],fma(-11.0,axis_sum,8.0*diag_sum));
    double m2 = fma(12.0,f[0],fma(-4.0,axis_sum,diag_sum));
    double m4 = fma(-4.0,ax_m,d78_m+d910_m+d1112_m+d1314_m);
    double m6 = fma(-4.0,ay_m,d78_m-d910_m+d1516_m+d1718_m);
    double m8 = fma(-4.0,az_m,d1112_m-d1314_m+d1516_m-d1718_m);
    double m9 = fma(2.0,ax_p,-(ay_p+az_p)+xy_diag-2.0*z_diag);
    double m10= fma(-2.0,ax_p,(ay_p+az_p)+xy_diag-2.0*z_diag);
    double m11= ay_p-az_p+d78_p+d910_p-d1112_p-d1314_p;
    double m12= -ay_p+az_p+d78_p+d910_p-d1112_p-d1314_p;
    double m13= d78_p-d910_p, m14= d1112_p-d1314_p, m15= d1516_p-d1718_p;
    double m16= d78_m-d910_m-d1112_m+d1314_m;
    double m17= -d78_m-d910_m+d1516_m+d1718_m;
    double m18= d1112_m+d1314_m-d1516_m+d1718_m;
    double m1_eq = rho*fma(19.0,u_sq,-11.0), m2_eq = rho*fma(-5.5,u_sq,3.0);
    double m4_eq = (-2.0/3.0)*rho*ux, m6_eq = (-2.0/3.0)*rho*uy, m8_eq = (-2.0/3.0)*rho*uz;
    double pxx = fma(2.0,ux*ux,-(uy*uy+uz*uz));
    double m9_eq = rho*pxx, m10_eq = -0.5*rho*pxx;
    double pww = uy*uy-uz*uz;
    double m11_eq = rho*pww, m12_eq = -0.5*rho*pww;
    double m13_eq = rho*ux*uy, m14_eq = rho*ux*uz, m15_eq = rho*uy*uz;
    m1 -= s_e*(m1-m1_eq); m2 -= s_eps*(m2-m2_eq);
    m4 -= s_q*(m4-m4_eq); m6 -= s_q*(m6-m6_eq); m8 -= s_q*(m8-m8_eq);
    m9 -= s_nu*(m9-m9_eq); m10 -= s_ghost*(m10-m10_eq);
    m11 -= s_nu*(m11-m11_eq); m12 -= s_ghost*(m12-m12_eq);
    m13 -= s_nu*(m13-m13_eq); m14 -= s_nu*(m14-m14_eq); m15 -= s_nu*(m15-m15_eq);
    m16 -= s_ghost*m16; m17 -= s_ghost*m17; m18 -= s_ghost*m18;
    double r0=m0*(1.0/19.0),r1=m1*(1.0/2394.0),r2=m2*(1.0/252.0);
    double r3=m3*(1.0/10.0),r4=m4*(1.0/40.0),r5=m5*(1.0/10.0),r6=m6*(1.0/40.0);
    double r7=m7*(1.0/10.0),r8=m8*(1.0/40.0);
    double r9=m9*(1.0/36.0),r10v=m10*(1.0/36.0);
    double r11v=m11*(1.0/12.0),r12v=m12*(1.0/12.0);
    double r13v=m13*0.25,r14v=m14*0.25,r15v=m15*0.25;
    double r16v=m16*0.125,r17v=m17*0.125,r18v=m18*0.125;
    double base_diag=r0+r2, r910v=r9+r10v, r1112v=r11v+r12v;
    double s34=r3+r4, s56=r5+r6, s78=r7+r8;
    double base_axis=fma(-11.0,r1,fma(-4.0,r2,r0));
    double base_xy=base_diag+r910v+r1112v, base_xz=base_diag+r910v-r1112v;
    double base_yz=fma(-2.0,r910v,base_diag);
    f[0]=fma(-30.0,r1,fma(12.0,r2,r0));
    f[1]=fma(-4.0,r4,base_axis+r3+2.0*r9-2.0*r10v);
    f[2]=fma(4.0,r4,base_axis-r3+2.0*r9-2.0*r10v);
    f[3]=fma(-4.0,r6,base_axis+r5-r9+r10v+r11v-r12v);
    f[4]=fma(4.0,r6,base_axis-r5-r9+r10v+r11v-r12v);
    f[5]=fma(-4.0,r8,base_axis+r7-r9+r10v-r11v+r12v);
    f[6]=fma(4.0,r8,base_axis-r7-r9+r10v-r11v+r12v);
    {double p1=s34+s56,p2=r13v+r16v,n1=s34-s56,n2=r13v-r16v;
     f[7]=fma(8.0,r1,base_xy+p1+p2-r17v); f[8]=fma(8.0,r1,base_xy-p1+n2+r17v);
     f[9]=fma(8.0,r1,base_xy+n1-p2-r17v); f[10]=fma(8.0,r1,base_xy-n1-n2+r17v);}
    {double p1=s34+s78,p2=r14v+r16v,n1=s34-s78,n2=r14v-r16v;
     f[11]=fma(8.0,r1,base_xz+p1+n2+r18v); f[12]=fma(8.0,r1,base_xz-p1+p2-r18v);
     f[13]=fma(8.0,r1,base_xz+n1-n2+r18v); f[14]=fma(8.0,r1,base_xz-n1-p2-r18v);}
    {double p1=s56+s78,p2=r15v+r17v,n1=s56-s78,n2=r15v-r17v;
     f[15]=fma(8.0,r1,base_yz+p1+p2-r18v); f[16]=fma(8.0,r1,base_yz-p1+n2+r18v);
     f[17]=fma(8.0,r1,base_yz+n1-p2-r18v); f[18]=fma(8.0,r1,base_yz-n1-n2+r18v);}
}

// FP64 i-major SoA MRT collision + pull-streaming kernel.
// Identical structure to BGK kernel above, but uses MRT collision.
extern "C" __global__ void lbm_step_fp64_soa_mrt_kernel(
    const double* __restrict__ f_in,
    double* __restrict__ f_out,
    float* __restrict__ rho_out,
    float* __restrict__ u_out,
    const float* __restrict__ tau,
    const float* __restrict__ force,
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

    // MRT collision (replaces BGK)
    double tau_local = (double)__ldg(&tau[idx]);
    mrt_collision_d3q19_fp64(f_local, rho_local, ux, uy, uz, tau_local);

    // Guo forcing (identical to BGK path)
    double fx = (double)__ldg(&force[idx]);
    double fy = (double)__ldg(&force[n_cells + idx]);
    double fz = (double)__ldg(&force[2 * n_cells + idx]);
    double force_mag_sq = fx * fx + fy * fy + fz * fz;

    if (force_mag_sq >= 1e-80) {
        double inv_tau = 1.0 / tau_local;
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

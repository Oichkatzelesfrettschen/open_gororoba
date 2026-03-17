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

// ---------------------------------------------------------------------------
// MRT collision device function (d'Humieres D3Q19, FP32 math)
// Standalone copy for NVRTC compilation unit isolation.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mrt_collision_d3q19_int16(
    float f[19], float rho, float ux, float uy, float uz, float tau_local
) {
    float s_nu = 1.0f / tau_local, s_e = 1.19f, s_eps = 1.4f, s_q = 1.2f, s_ghost = 1.0f;
    float u_sq = ux*ux + uy*uy + uz*uz;
    float ax_p = f[1]+f[2], ax_m = f[1]-f[2], ay_p = f[3]+f[4], ay_m = f[3]-f[4];
    float az_p = f[5]+f[6], az_m = f[5]-f[6];
    float d78_p = f[7]+f[8], d78_m = f[7]-f[8], d910_p = f[9]+f[10], d910_m = f[9]-f[10];
    float d1112_p = f[11]+f[12], d1112_m = f[11]-f[12], d1314_p = f[13]+f[14], d1314_m = f[13]-f[14];
    float d1516_p = f[15]+f[16], d1516_m = f[15]-f[16], d1718_p = f[17]+f[18], d1718_m = f[17]-f[18];
    float axis_sum = ax_p+ay_p+az_p;
    float diag_sum = d78_p+d910_p+d1112_p+d1314_p+d1516_p+d1718_p;
    float xy_diag = d78_p+d910_p+d1112_p+d1314_p, z_diag = d1516_p+d1718_p;
    float m0 = f[0]+axis_sum+diag_sum;
    float m3 = ax_m+d78_m+d910_m+d1112_m+d1314_m;
    float m5 = ay_m+d78_m-d910_m+d1516_m+d1718_m;
    float m7 = az_m+d1112_m-d1314_m+d1516_m-d1718_m;
    float m1 = fmaf(-30.0f,f[0],fmaf(-11.0f,axis_sum,8.0f*diag_sum));
    float m2 = fmaf(12.0f,f[0],fmaf(-4.0f,axis_sum,diag_sum));
    float m4 = fmaf(-4.0f,ax_m,d78_m+d910_m+d1112_m+d1314_m);
    float m6 = fmaf(-4.0f,ay_m,d78_m-d910_m+d1516_m+d1718_m);
    float m8 = fmaf(-4.0f,az_m,d1112_m-d1314_m+d1516_m-d1718_m);
    float m9 = fmaf(2.0f,ax_p,-(ay_p+az_p)+xy_diag-2.0f*z_diag);
    float m10= fmaf(-2.0f,ax_p,(ay_p+az_p)+xy_diag-2.0f*z_diag);
    float m11= ay_p-az_p+d78_p+d910_p-d1112_p-d1314_p;
    float m12= -ay_p+az_p+d78_p+d910_p-d1112_p-d1314_p;
    float m13= d78_p-d910_p, m14= d1112_p-d1314_p, m15= d1516_p-d1718_p;
    float m16= d78_m-d910_m-d1112_m+d1314_m;
    float m17= -d78_m-d910_m+d1516_m+d1718_m;
    float m18= d1112_m+d1314_m-d1516_m+d1718_m;
    float m1_eq = rho*fmaf(19.0f,u_sq,-11.0f), m2_eq = rho*fmaf(-5.5f,u_sq,3.0f);
    float m4_eq = (-2.0f/3.0f)*rho*ux, m6_eq = (-2.0f/3.0f)*rho*uy, m8_eq = (-2.0f/3.0f)*rho*uz;
    float pxx = fmaf(2.0f,ux*ux,-(uy*uy+uz*uz));
    float m9_eq = rho*pxx, m10_eq = -0.5f*rho*pxx;
    float pww = uy*uy-uz*uz;
    float m11_eq = rho*pww, m12_eq = -0.5f*rho*pww;
    float m13_eq = rho*ux*uy, m14_eq = rho*ux*uz, m15_eq = rho*uy*uz;
    m1 -= s_e*(m1-m1_eq); m2 -= s_eps*(m2-m2_eq);
    m4 -= s_q*(m4-m4_eq); m6 -= s_q*(m6-m6_eq); m8 -= s_q*(m8-m8_eq);
    m9 -= s_nu*(m9-m9_eq); m10 -= s_ghost*(m10-m10_eq);
    m11 -= s_nu*(m11-m11_eq); m12 -= s_ghost*(m12-m12_eq);
    m13 -= s_nu*(m13-m13_eq); m14 -= s_nu*(m14-m14_eq); m15 -= s_nu*(m15-m15_eq);
    m16 -= s_ghost*m16; m17 -= s_ghost*m17; m18 -= s_ghost*m18;
    float r0=m0*(1.0f/19.0f),r1=m1*(1.0f/2394.0f),r2=m2*(1.0f/252.0f);
    float r3=m3*(1.0f/10.0f),r4=m4*(1.0f/40.0f),r5=m5*(1.0f/10.0f),r6=m6*(1.0f/40.0f);
    float r7=m7*(1.0f/10.0f),r8=m8*(1.0f/40.0f);
    float r9=m9*(1.0f/36.0f),r10v=m10*(1.0f/36.0f);
    float r11v=m11*(1.0f/12.0f),r12v=m12*(1.0f/12.0f);
    float r13v=m13*0.25f,r14v=m14*0.25f,r15v=m15*0.25f;
    float r16v=m16*0.125f,r17v=m17*0.125f,r18v=m18*0.125f;
    float base_diag=r0+r2, r910v=r9+r10v, r1112v=r11v+r12v;
    float s34=r3+r4, s56=r5+r6, s78=r7+r8;
    float base_axis=fmaf(-11.0f,r1,fmaf(-4.0f,r2,r0));
    float base_xy=base_diag+r910v+r1112v, base_xz=base_diag+r910v-r1112v;
    float base_yz=fmaf(-2.0f,r910v,base_diag);
    f[0]=fmaf(-30.0f,r1,fmaf(12.0f,r2,r0));
    f[1]=fmaf(-4.0f,r4,base_axis+r3+2.0f*r9-2.0f*r10v);
    f[2]=fmaf(4.0f,r4,base_axis-r3+2.0f*r9-2.0f*r10v);
    f[3]=fmaf(-4.0f,r6,base_axis+r5-r9+r10v+r11v-r12v);
    f[4]=fmaf(4.0f,r6,base_axis-r5-r9+r10v+r11v-r12v);
    f[5]=fmaf(-4.0f,r8,base_axis+r7-r9+r10v-r11v+r12v);
    f[6]=fmaf(4.0f,r8,base_axis-r7-r9+r10v-r11v+r12v);
    {float p1=s34+s56,p2=r13v+r16v,n1=s34-s56,n2=r13v-r16v;
     f[7]=fmaf(8.0f,r1,base_xy+p1+p2-r17v); f[8]=fmaf(8.0f,r1,base_xy-p1+n2+r17v);
     f[9]=fmaf(8.0f,r1,base_xy+n1-p2-r17v); f[10]=fmaf(8.0f,r1,base_xy-n1-n2+r17v);}
    {float p1=s34+s78,p2=r14v+r16v,n1=s34-s78,n2=r14v-r16v;
     f[11]=fmaf(8.0f,r1,base_xz+p1+n2+r18v); f[12]=fmaf(8.0f,r1,base_xz-p1+p2-r18v);
     f[13]=fmaf(8.0f,r1,base_xz+n1-n2+r18v); f[14]=fmaf(8.0f,r1,base_xz-n1-p2-r18v);}
    {float p1=s56+s78,p2=r15v+r17v,n1=s56-s78,n2=r15v-r17v;
     f[15]=fmaf(8.0f,r1,base_yz+p1+p2-r18v); f[16]=fmaf(8.0f,r1,base_yz-p1+n2+r18v);
     f[17]=fmaf(8.0f,r1,base_yz+n1-p2-r18v); f[18]=fmaf(8.0f,r1,base_yz-n1-n2+r18v);}
}

// INT16 i-major SoA MRT collision + pull-streaming kernel.
// Identical structure to BGK kernel above, but uses MRT collision.
extern "C" __launch_bounds__(128, 4) __global__ void lbm_step_int16_soa_mrt_kernel(
    const short* __restrict__ f_in,
    short* __restrict__ f_out,
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

    float f_local[19];
    float rho_local = 0.0f;
    int mx_i32 = 0, my_i32 = 0, mz_i32 = 0;

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        int xb = (x - CX_I16S[i] + nx) % nx;
        int yb = (y - CY_I16S[i] + ny) % ny;
        int zb = (z - CZ_I16S[i] + nz) % nz;
        int idx_back = xb + nx * (yb + ny * zb);
        short raw = __ldg(&f_in[(long long)i * n_cells + idx_back]);
        float fi = (float)raw * INV_DIST_SCALE_I16S;
        if (!finite_i16s(fi)) { fi = 0.0f; raw = 0; }
        f_local[i] = fi;
        rho_local  += fi;
        mx_i32     += CX_I16S[i] * (int)raw;
        my_i32     += CY_I16S[i] * (int)raw;
        mz_i32     += CZ_I16S[i] * (int)raw;
    }

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

    // MRT collision (replaces BGK)
    float tau_local = __ldg(&tau[idx]);
    mrt_collision_d3q19_int16(f_local, rho_local, ux, uy, uz, tau_local);

    // Guo forcing (identical to BGK path)
    float fx = __ldg(&force[idx]);
    float fy = __ldg(&force[n_cells + idx]);
    float fz = __ldg(&force[2 * n_cells + idx]);
    float force_mag_sq = fx * fx + fy * fy + fz * fz;

    if (force_mag_sq >= 1e-40f) {
        float inv_tau = 1.0f / tau_local;
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

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        f_out[(long long)i * n_cells + idx] = float_to_i16s(f_local[i]);
    }
}

// D3Q19 opposite-direction LUT for A-A streaming (INT16 compilation unit).
__constant__ int OPP_I16S[19] = {
    0,  2,  1,  4,  3,  6,  5,  8,  7, 10,
    9, 12, 11, 14, 13, 16, 15, 18, 17
};

// INT16 i-major SoA A-A MRT kernel: single-buffer, parity-toggled streaming.
// Even step: read f[i*N+idx], collide, write f[OPP[i]*N+dst].
// Odd  step: read f[OPP[i]*N+src], collide, write f[i*N+idx].
extern "C" __launch_bounds__(128, 4) __global__ void lbm_step_int16_soa_mrt_aa_kernel(
    short* __restrict__ f,
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

    float f_local[19];
    float rho_local = 0.0f;
    int mx_i32 = 0, my_i32 = 0, mz_i32 = 0;

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        int read_dir, src;
        if (parity == 0) {
            read_dir = i;
            src = idx;
        } else {
            int xn = (x - CX_I16S[i] + nx) % nx;
            int yn = (y - CY_I16S[i] + ny) % ny;
            int zn = (z - CZ_I16S[i] + nz) % nz;
            src = xn + nx * (yn + ny * zn);
            read_dir = OPP_I16S[i];
        }
        short raw = __ldg(&f[(long long)read_dir * N + src]);
        float fi = (float)raw * INV_DIST_SCALE_I16S;
        if (!finite_i16s(fi)) { fi = 0.0f; raw = 0; }
        f_local[i] = fi;
        rho_local  += fi;
        mx_i32     += CX_I16S[i] * (int)raw;
        my_i32     += CY_I16S[i] * (int)raw;
        mz_i32     += CZ_I16S[i] * (int)raw;
    }

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
    u_out[idx]           = ux;
    u_out[N + idx]       = uy;
    u_out[2 * N + idx]   = uz;

    float tau_local = __ldg(&tau[idx]);
    mrt_collision_d3q19_int16(f_local, rho_local, ux, uy, uz, tau_local);

    float fx = __ldg(&force[idx]);
    float fy = __ldg(&force[N + idx]);
    float fz = __ldg(&force[2 * N + idx]);
    float force_mag_sq = fx * fx + fy * fy + fz * fz;

    if (force_mag_sq >= 1e-40f) {
        float inv_tau = 1.0f / tau_local;
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

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        if (parity == 0) {
            int xn = (x + CX_I16S[i] + nx) % nx;
            int yn = (y + CY_I16S[i] + ny) % ny;
            int zn = (z + CZ_I16S[i] + nz) % nz;
            int dst = xn + nx * (yn + ny * zn);
            f[(long long)OPP_I16S[i] * N + dst] = float_to_i16s(f_local[i]);
        } else {
            f[(long long)i * N + idx] = float_to_i16s(f_local[i]);
        }
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

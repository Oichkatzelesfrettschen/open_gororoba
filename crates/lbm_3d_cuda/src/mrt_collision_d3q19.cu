// D3Q19 MRT collision device function -- SINGLE SOURCE OF TRUTH.
//
// This file is included by Rust via include_str! and prepended to
// each precision-tier kernel source before NVRTC compilation.
// Do NOT add `extern "C"` kernels here -- only device functions.
//
// d'Humieres (2002) orthogonal basis with 5 distinct relaxation rates:
//   s_nu    = 1/tau  -- physical kinematic viscosity (stress moments 9,11,13-15)
//   s_e     = 1.19   -- energy relaxation (moment 1)
//   s_eps   = 1.4    -- energy-squared relaxation (moment 2)
//   s_q     = 1.2    -- energy-flux relaxation (moments 4,6,8)
//   s_ghost = 1.0    -- instant damping of ghost moments (10,12,16-18)

// FP32 MRT collision (used by all precision tiers except FP64).
// All precision tiers promote to FP32 for collision math; the storage
// format only affects load/store.
__device__ __forceinline__ void mrt_collision_d3q19(
    float f[19], float rho, float ux, float uy, float uz, float tau_local
) {
    float s_nu = 1.0f / tau_local, s_e = 1.19f, s_eps = 1.4f, s_q = 1.2f, s_ghost = 1.0f;
    float u_sq = ux*ux + uy*uy + uz*uz;

    // Forward transform: pair sums/diffs (ILP-optimized for Ada dual FP32 pipes)
    float ax_p = f[1]+f[2], ax_m = f[1]-f[2], ay_p = f[3]+f[4], ay_m = f[3]-f[4];
    float az_p = f[5]+f[6], az_m = f[5]-f[6];
    float d78_p = f[7]+f[8], d78_m = f[7]-f[8], d910_p = f[9]+f[10], d910_m = f[9]-f[10];
    float d1112_p = f[11]+f[12], d1112_m = f[11]-f[12], d1314_p = f[13]+f[14], d1314_m = f[13]-f[14];
    float d1516_p = f[15]+f[16], d1516_m = f[15]-f[16], d1718_p = f[17]+f[18], d1718_m = f[17]-f[18];

    float axis_sum = ax_p+ay_p+az_p;
    float diag_sum = d78_p+d910_p+d1112_p+d1314_p+d1516_p+d1718_p;
    float xy_diag = d78_p+d910_p+d1112_p+d1314_p, z_diag = d1516_p+d1718_p;

    // Conserved moments
    float m0 = f[0]+axis_sum+diag_sum;
    float m3 = ax_m+d78_m+d910_m+d1112_m+d1314_m;
    float m5 = ay_m+d78_m-d910_m+d1516_m+d1718_m;
    float m7 = az_m+d1112_m-d1314_m+d1516_m-d1718_m;

    // Non-conserved moments
    float m1  = fmaf(-30.0f, f[0], fmaf(-11.0f, axis_sum, 8.0f * diag_sum));
    float m2  = fmaf(12.0f, f[0], fmaf(-4.0f, axis_sum, diag_sum));
    float m4  = fmaf(-4.0f, ax_m, d78_m+d910_m+d1112_m+d1314_m);
    float m6  = fmaf(-4.0f, ay_m, d78_m-d910_m+d1516_m+d1718_m);
    float m8  = fmaf(-4.0f, az_m, d1112_m-d1314_m+d1516_m-d1718_m);
    float m9  = fmaf(2.0f, ax_p, -(ay_p+az_p) + xy_diag - 2.0f*z_diag);
    float m10 = fmaf(-2.0f, ax_p, (ay_p+az_p) + xy_diag - 2.0f*z_diag);
    float m11 = ay_p-az_p+d78_p+d910_p-d1112_p-d1314_p;
    float m12 = -ay_p+az_p+d78_p+d910_p-d1112_p-d1314_p;
    float m13 = d78_p-d910_p, m14 = d1112_p-d1314_p, m15 = d1516_p-d1718_p;
    float m16 = d78_m-d910_m-d1112_m+d1314_m;
    float m17 = -d78_m-d910_m+d1516_m+d1718_m;
    float m18 = d1112_m+d1314_m-d1516_m+d1718_m;

    // Equilibrium moments
    float m1_eq  = rho * fmaf(19.0f, u_sq, -11.0f);
    float m2_eq  = rho * fmaf(-5.5f, u_sq, 3.0f);
    float m4_eq  = (-2.0f/3.0f) * rho * ux;
    float m6_eq  = (-2.0f/3.0f) * rho * uy;
    float m8_eq  = (-2.0f/3.0f) * rho * uz;
    float pxx    = fmaf(2.0f, ux*ux, -(uy*uy + uz*uz));
    float m9_eq  = rho * pxx;
    float m10_eq = -0.5f * rho * pxx;
    float pww    = uy*uy - uz*uz;
    float m11_eq = rho * pww;
    float m12_eq = -0.5f * rho * pww;
    float m13_eq = rho * ux * uy;
    float m14_eq = rho * ux * uz;
    float m15_eq = rho * uy * uz;

    // Relax: m* = m - S * (m - m_eq)
    m1  -= s_e    * (m1  - m1_eq);
    m2  -= s_eps  * (m2  - m2_eq);
    m4  -= s_q    * (m4  - m4_eq);
    m6  -= s_q    * (m6  - m6_eq);
    m8  -= s_q    * (m8  - m8_eq);
    m9  -= s_nu   * (m9  - m9_eq);
    m10 -= s_ghost* (m10 - m10_eq);
    m11 -= s_nu   * (m11 - m11_eq);
    m12 -= s_ghost* (m12 - m12_eq);
    m13 -= s_nu   * (m13 - m13_eq);
    m14 -= s_nu   * (m14 - m14_eq);
    m15 -= s_nu   * (m15 - m15_eq);
    m16 -= s_ghost* m16;
    m17 -= s_ghost* m17;
    m18 -= s_ghost* m18;

    // Inverse transform: f* = M^{-1} * m*
    float r0  = m0  * (1.0f/19.0f);
    float r1  = m1  * (1.0f/2394.0f);
    float r2  = m2  * (1.0f/252.0f);
    float r3  = m3  * (1.0f/10.0f);
    float r4  = m4  * (1.0f/40.0f);
    float r5  = m5  * (1.0f/10.0f);
    float r6  = m6  * (1.0f/40.0f);
    float r7  = m7  * (1.0f/10.0f);
    float r8  = m8  * (1.0f/40.0f);
    float r9  = m9  * (1.0f/36.0f);
    float r10 = m10 * (1.0f/36.0f);
    float r11 = m11 * (1.0f/12.0f);
    float r12 = m12 * (1.0f/12.0f);
    float r13 = m13 * 0.25f;
    float r14 = m14 * 0.25f;
    float r15 = m15 * 0.25f;
    float r16 = m16 * 0.125f;
    float r17 = m17 * 0.125f;
    float r18 = m18 * 0.125f;

    // Balanced sub-expressions
    float base_diag = r0 + r2;
    float r910  = r9 + r10;
    float r1112 = r11 + r12;
    float s34 = r3+r4, s56 = r5+r6, s78 = r7+r8;
    float base_axis = fmaf(-11.0f, r1, fmaf(-4.0f, r2, r0));
    float base_xy = base_diag + r910 + r1112;
    float base_xz = base_diag + r910 - r1112;
    float base_yz = fmaf(-2.0f, r910, base_diag);

    f[0]  = fmaf(-30.0f, r1, fmaf(12.0f, r2, r0));
    f[1]  = fmaf(-4.0f, r4, base_axis + r3 + 2.0f*r9 - 2.0f*r10);
    f[2]  = fmaf( 4.0f, r4, base_axis - r3 + 2.0f*r9 - 2.0f*r10);
    f[3]  = fmaf(-4.0f, r6, base_axis + r5 - r9 + r10 + r11 - r12);
    f[4]  = fmaf( 4.0f, r6, base_axis - r5 - r9 + r10 + r11 - r12);
    f[5]  = fmaf(-4.0f, r8, base_axis + r7 - r9 + r10 - r11 + r12);
    f[6]  = fmaf( 4.0f, r8, base_axis - r7 - r9 + r10 - r11 + r12);
    {
        float p1 = s34+s56, p2 = r13+r16, n1 = s34-s56, n2 = r13-r16;
        f[7]  = fmaf(8.0f, r1, base_xy + p1 + p2 - r17);
        f[8]  = fmaf(8.0f, r1, base_xy - p1 + n2 + r17);
        f[9]  = fmaf(8.0f, r1, base_xy + n1 - p2 - r17);
        f[10] = fmaf(8.0f, r1, base_xy - n1 - n2 + r17);
    }
    {
        float p1 = s34+s78, p2 = r14+r16, n1 = s34-s78, n2 = r14-r16;
        f[11] = fmaf(8.0f, r1, base_xz + p1 + n2 + r18);
        f[12] = fmaf(8.0f, r1, base_xz - p1 + p2 - r18);
        f[13] = fmaf(8.0f, r1, base_xz + n1 - n2 + r18);
        f[14] = fmaf(8.0f, r1, base_xz - n1 - p2 - r18);
    }
    {
        float p1 = s56+s78, p2 = r15+r17, n1 = s56-s78, n2 = r15-r17;
        f[15] = fmaf(8.0f, r1, base_yz + p1 + p2 - r18);
        f[16] = fmaf(8.0f, r1, base_yz - p1 + n2 + r18);
        f[17] = fmaf(8.0f, r1, base_yz + n1 - p2 - r18);
        f[18] = fmaf(8.0f, r1, base_yz - n1 - n2 + r18);
    }
}

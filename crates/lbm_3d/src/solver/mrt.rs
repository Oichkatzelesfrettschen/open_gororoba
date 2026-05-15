//! D3Q19 Multiple-Relaxation-Time (MRT) Collision Operator.
//!
//! Implements the d'Humieres (2002) orthogonal basis transformation:
//! forward f -> moment space, diagonal relaxation toward equilibrium, and
//! inverse moment -> f. Includes both scalar and AVX2 4-wide variants.
//!
//! These are private to the lbm_3d crate; callers go through the higher-
//! level LbmSolver3D collision dispatcher in solver.rs.

use wide::f64x4;

/// D3Q19 Multiple-Relaxation-Time (MRT) Collision Operator.
///
/// Implements the d'Humieres (2002) orthogonal basis transformation.
/// Forward transform f -> moment space (m = M * f) for D3Q19 MRT.
///
/// Exposed for testing row norms and moment structure. Uses the same
/// d'Humieres orthogonal basis as the collision operator.
#[cfg(test)]
pub(super) fn mrt_forward_transform(f: &[f64; 19]) -> [f64; 19] {
    [
        f[0] + f[1]
            + f[2]
            + f[3]
            + f[4]
            + f[5]
            + f[6]
            + f[7]
            + f[8]
            + f[9]
            + f[10]
            + f[11]
            + f[12]
            + f[13]
            + f[14]
            + f[15]
            + f[16]
            + f[17]
            + f[18],
        -30.0 * f[0] - 11.0 * (f[1] + f[2] + f[3] + f[4] + f[5] + f[6])
            + 8.0
                * (f[7]
                    + f[8]
                    + f[9]
                    + f[10]
                    + f[11]
                    + f[12]
                    + f[13]
                    + f[14]
                    + f[15]
                    + f[16]
                    + f[17]
                    + f[18]),
        12.0 * f[0] - 4.0 * (f[1] + f[2] + f[3] + f[4] + f[5] + f[6])
            + (f[7]
                + f[8]
                + f[9]
                + f[10]
                + f[11]
                + f[12]
                + f[13]
                + f[14]
                + f[15]
                + f[16]
                + f[17]
                + f[18]),
        f[1] - f[2] + f[7] - f[8] + f[9] - f[10] + f[11] - f[12] + f[13] - f[14],
        -4.0 * (f[1] - f[2]) + f[7] - f[8] + f[9] - f[10] + f[11] - f[12] + f[13] - f[14],
        f[3] - f[4] + f[7] - f[8] - f[9] + f[10] + f[15] - f[16] + f[17] - f[18],
        -4.0 * (f[3] - f[4]) + f[7] - f[8] - f[9] + f[10] + f[15] - f[16] + f[17] - f[18],
        f[5] - f[6] + f[11] - f[12] - f[13] + f[14] + f[15] - f[16] - f[17] + f[18],
        -4.0 * (f[5] - f[6]) + f[11] - f[12] - f[13] + f[14] + f[15] - f[16] - f[17] + f[18],
        2.0 * (f[1] + f[2]) - (f[3] + f[4] + f[5] + f[6])
            + f[7]
            + f[8]
            + f[9]
            + f[10]
            + f[11]
            + f[12]
            + f[13]
            + f[14]
            - 2.0 * (f[15] + f[16] + f[17] + f[18]),
        -2.0 * (f[1] + f[2])
            + (f[3] + f[4] + f[5] + f[6])
            + f[7]
            + f[8]
            + f[9]
            + f[10]
            + f[11]
            + f[12]
            + f[13]
            + f[14]
            - 2.0 * (f[15] + f[16] + f[17] + f[18]),
        (f[3] + f[4]) - (f[5] + f[6]) + f[7] + f[8] + f[9] + f[10]
            - (f[11] + f[12] + f[13] + f[14]),
        -(f[3] + f[4]) + (f[5] + f[6]) + f[7] + f[8] + f[9] + f[10]
            - (f[11] + f[12] + f[13] + f[14]),
        f[7] + f[8] - f[9] - f[10],
        f[11] + f[12] - f[13] - f[14],
        f[15] + f[16] - f[17] - f[18],
        f[7] - f[8] - f[9] + f[10] - f[11] + f[12] + f[13] - f[14],
        -f[7] + f[8] - f[9] + f[10] + f[15] - f[16] + f[17] - f[18],
        f[11] - f[12] + f[13] - f[14] - f[15] + f[16] + f[17] - f[18],
    ]
}

/// Ghost moments relax instantly (s=1.0) to annihilate the spurious
/// oscillations that cause BGK divergence at steep density cusps.
///
/// The operator transforms f -> moment space (M*f), relaxes each moment
/// independently via diagonal S matrix, then transforms back (M^{-1}*m*).
/// Physical viscosity s_nu = 1/tau is preserved; ghost moments use s=1.0.
///
/// Cost: ~722 FMA operations per cell vs ~57 for BGK (~12x more FLOPs),
/// but the GPU memory-bound regime absorbs this within the latency window.
#[inline(always)]
pub(super) fn collide_mrt_d3q19(
    f: &[f64; 19],
    rho: f64,
    ux: f64,
    uy: f64,
    uz: f64,
    tau: f64,
) -> [f64; 19] {
    // Relaxation rates (diagonal of S matrix)
    let s_nu = 1.0 / tau; // Physical kinematic viscosity
    let s_e = 1.19; // Energy relaxation
    let s_eps = 1.4; // Energy squared
    let s_q = 1.2; // Energy flux
    let s_ghost = 1.0; // Instant damping for ghost moments

    let u_sq = ux * ux + uy * uy + uz * uz;

    // Forward transform: f -> moment space (m = M * f)
    // Using the d'Humieres D3Q19 orthogonal basis
    let m0 = f[0]
        + f[1]
        + f[2]
        + f[3]
        + f[4]
        + f[5]
        + f[6]
        + f[7]
        + f[8]
        + f[9]
        + f[10]
        + f[11]
        + f[12]
        + f[13]
        + f[14]
        + f[15]
        + f[16]
        + f[17]
        + f[18];
    let m1 = -30.0 * f[0] - 11.0 * (f[1] + f[2] + f[3] + f[4] + f[5] + f[6])
        + 8.0
            * (f[7]
                + f[8]
                + f[9]
                + f[10]
                + f[11]
                + f[12]
                + f[13]
                + f[14]
                + f[15]
                + f[16]
                + f[17]
                + f[18]);
    let m2 = 12.0 * f[0] - 4.0 * (f[1] + f[2] + f[3] + f[4] + f[5] + f[6])
        + (f[7]
            + f[8]
            + f[9]
            + f[10]
            + f[11]
            + f[12]
            + f[13]
            + f[14]
            + f[15]
            + f[16]
            + f[17]
            + f[18]);
    let m3 = f[1] - f[2] + f[7] - f[8] + f[9] - f[10] + f[11] - f[12] + f[13] - f[14];
    let m4 = -4.0 * (f[1] - f[2]) + f[7] - f[8] + f[9] - f[10] + f[11] - f[12] + f[13] - f[14];
    let m5 = f[3] - f[4] + f[7] - f[8] - f[9] + f[10] + f[15] - f[16] + f[17] - f[18];
    let m6 = -4.0 * (f[3] - f[4]) + f[7] - f[8] - f[9] + f[10] + f[15] - f[16] + f[17] - f[18];
    let m7 = f[5] - f[6] + f[11] - f[12] - f[13] + f[14] + f[15] - f[16] - f[17] + f[18];
    let m8 = -4.0 * (f[5] - f[6]) + f[11] - f[12] - f[13] + f[14] + f[15] - f[16] - f[17] + f[18];
    let m9 = 2.0 * (f[1] + f[2]) - (f[3] + f[4] + f[5] + f[6])
        + f[7]
        + f[8]
        + f[9]
        + f[10]
        + f[11]
        + f[12]
        + f[13]
        + f[14]
        - 2.0 * (f[15] + f[16] + f[17] + f[18]);
    let m10 = -2.0 * (f[1] + f[2])
        + (f[3] + f[4] + f[5] + f[6])
        + f[7]
        + f[8]
        + f[9]
        + f[10]
        + f[11]
        + f[12]
        + f[13]
        + f[14]
        - 2.0 * (f[15] + f[16] + f[17] + f[18]);
    let m11 = (f[3] + f[4]) - (f[5] + f[6]) + f[7] + f[8] + f[9] + f[10]
        - (f[11] + f[12] + f[13] + f[14]);
    let m12 = -(f[3] + f[4]) + (f[5] + f[6]) + f[7] + f[8] + f[9] + f[10]
        - (f[11] + f[12] + f[13] + f[14]);
    let m13 = f[7] + f[8] - f[9] - f[10];
    let m14 = f[11] + f[12] - f[13] - f[14];
    let m15 = f[15] + f[16] - f[17] - f[18];
    let m16 = f[7] - f[8] - f[9] + f[10] - f[11] + f[12] + f[13] - f[14];
    let m17 = -f[7] + f[8] - f[9] + f[10] + f[15] - f[16] + f[17] - f[18];
    let m18 = f[11] - f[12] + f[13] - f[14] - f[15] + f[16] + f[17] - f[18];

    // Equilibrium moments
    // m0_eq = rho (conserved, not needed for relaxation)
    let m1_eq = rho * (-11.0 + 19.0 * u_sq);
    let m2_eq = rho * (3.0 - 5.5 * u_sq);
    // m3_eq = rho*ux (conserved)
    let m4_eq = -2.0 / 3.0 * rho * ux;
    // m5_eq = rho*uy (conserved)
    let m6_eq = -2.0 / 3.0 * rho * uy;
    // m7_eq = rho*uz (conserved)
    let m8_eq = -2.0 / 3.0 * rho * uz;
    let m9_eq = rho * (2.0 * ux * ux - uy * uy - uz * uz);
    let m10_eq = -0.5 * rho * (2.0 * ux * ux - uy * uy - uz * uz);
    let m11_eq = rho * (uy * uy - uz * uz);
    let m12_eq = -0.5 * rho * (uy * uy - uz * uz);
    let m13_eq = rho * ux * uy;
    let m14_eq = rho * ux * uz;
    let m15_eq = rho * uy * uz;
    let m16_eq = 0.0;
    let m17_eq = 0.0;
    let m18_eq = 0.0;

    // Relax moments: m* = m - S * (m - m_eq)
    // Mass (m0) and momentum (m3, m5, m7) are conserved (s=0).
    let ms0 = m0; // conserved
    let ms1 = m1 - s_e * (m1 - m1_eq); // energy
    let ms2 = m2 - s_eps * (m2 - m2_eq); // energy^2
    let ms3 = m3; // conserved
    let ms4 = m4 - s_q * (m4 - m4_eq); // energy flux
    let ms5 = m5; // conserved
    let ms6 = m6 - s_q * (m6 - m6_eq); // energy flux
    let ms7 = m7; // conserved
    let ms8 = m8 - s_q * (m8 - m8_eq); // energy flux
    let ms9 = m9 - s_nu * (m9 - m9_eq); // stress (physical)
    let ms10 = m10 - s_ghost * (m10 - m10_eq); // ghost
    let ms11 = m11 - s_nu * (m11 - m11_eq); // stress (physical)
    let ms12 = m12 - s_ghost * (m12 - m12_eq); // ghost
    let ms13 = m13 - s_nu * (m13 - m13_eq); // stress (physical)
    let ms14 = m14 - s_nu * (m14 - m14_eq); // stress (physical)
    let ms15 = m15 - s_nu * (m15 - m15_eq); // stress (physical)
    let ms16 = m16 - s_ghost * (m16 - m16_eq); // ghost
    let ms17 = m17 - s_ghost * (m17 - m17_eq); // ghost
    let ms18 = m18 - s_ghost * (m18 - m18_eq); // ghost

    // Inverse transform: f* = M^{-1} * m*
    // M^{-1}_{ij} = M_{ji} / ||row_j||^2 (orthogonal, non-orthonormal basis).
    // Row squared-norms: [19, 2394, 252, 10, 40, 10, 40, 10, 40, 36, 36, 12, 12, 4, 4, 4, 8, 8, 8]
    let mut fo = [0.0; 19];

    // Reciprocal norms (1 / ||row_j||^2)
    let rn0 = 1.0 / 19.0;
    let rn1 = 1.0 / 2394.0;
    let rn2 = 1.0 / 252.0;
    let rn3 = 1.0 / 10.0;
    let rn4 = 1.0 / 40.0;
    let rn5 = 1.0 / 10.0;
    let rn6 = 1.0 / 40.0;
    let rn7 = 1.0 / 10.0;
    let rn8 = 1.0 / 40.0;
    let rn9 = 1.0 / 36.0;
    let rn10 = 1.0 / 36.0;
    let rn11 = 1.0 / 12.0;
    let rn12 = 1.0 / 12.0;
    let rn13 = 1.0 / 4.0;
    let rn14 = 1.0 / 4.0;
    let rn15 = 1.0 / 4.0;
    let rn16 = 1.0 / 8.0;
    let rn17 = 1.0 / 8.0;
    let rn18 = 1.0 / 8.0;

    // Scale relaxed moments by reciprocal norms: s[j] = ms_j / ||row_j||^2
    let s0 = ms0 * rn0;
    let s1 = ms1 * rn1;
    let s2 = ms2 * rn2;
    let s3 = ms3 * rn3;
    let s4 = ms4 * rn4;
    let s5 = ms5 * rn5;
    let s6 = ms6 * rn6;
    let s7 = ms7 * rn7;
    let s8 = ms8 * rn8;
    let s9 = ms9 * rn9;
    let s10 = ms10 * rn10;
    let s11 = ms11 * rn11;
    let s12 = ms12 * rn12;
    let s13 = ms13 * rn13;
    let s14 = ms14 * rn14;
    let s15 = ms15 * rn15;
    let s16 = ms16 * rn16;
    let s17 = ms17 * rn17;
    let s18 = ms18 * rn18;

    // f[i] = sum_j M[j][i] * s[j]   (M[j][i] = coefficient of f[i] in row j of M)

    // Rest (i=0): M columns [1, -30, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fo[0] = s0 - 30.0 * s1 + 12.0 * s2;

    // Face +x (i=1): [1, -11, -4, 1, -4, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0]
    fo[1] = s0 - 11.0 * s1 - 4.0 * s2 + s3 - 4.0 * s4 + 2.0 * s9 - 2.0 * s10;
    // Face -x (i=2)
    fo[2] = s0 - 11.0 * s1 - 4.0 * s2 - s3 + 4.0 * s4 + 2.0 * s9 - 2.0 * s10;
    // Face +y (i=3): [1, -11, -4, 0, 0, 1, -4, 0, 0, -1, 1, 1, -1, 0, 0, 0, 0, 0, 0]
    fo[3] = s0 - 11.0 * s1 - 4.0 * s2 + s5 - 4.0 * s6 - s9 + s10 + s11 - s12;
    // Face -y (i=4)
    fo[4] = s0 - 11.0 * s1 - 4.0 * s2 - s5 + 4.0 * s6 - s9 + s10 + s11 - s12;
    // Face +z (i=5): [1, -11, -4, 0, 0, 0, 0, 1, -4, -1, 1, -1, 1, 0, 0, 0, 0, 0, 0]
    fo[5] = s0 - 11.0 * s1 - 4.0 * s2 + s7 - 4.0 * s8 - s9 + s10 - s11 + s12;
    // Face -z (i=6)
    fo[6] = s0 - 11.0 * s1 - 4.0 * s2 - s7 + 4.0 * s8 - s9 + s10 - s11 + s12;

    // Edge +x+y (i=7): [1, 8, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, -1, 0]
    fo[7] = s0 + 8.0 * s1 + s2 + s3 + s4 + s5 + s6 + s9 + s10 + s11 + s12 + s13 + s16 - s17;
    // Edge -x-y (i=8)
    fo[8] = s0 + 8.0 * s1 + s2 - s3 - s4 - s5 - s6 + s9 + s10 + s11 + s12 + s13 - s16 + s17;
    // Edge +x-y (i=9)
    fo[9] = s0 + 8.0 * s1 + s2 + s3 + s4 - s5 - s6 + s9 + s10 + s11 + s12 - s13 - s16 - s17;
    // Edge -x+y (i=10)
    fo[10] = s0 + 8.0 * s1 + s2 - s3 - s4 + s5 + s6 + s9 + s10 + s11 + s12 - s13 + s16 + s17;
    // Edge +x+z (i=11): [1, 8, 1, 1, 1, 0, 0, 1, 1, 1, 1, -1, -1, 0, 1, 0, -1, 0, 1]
    fo[11] = s0 + 8.0 * s1 + s2 + s3 + s4 + s7 + s8 + s9 + s10 - s11 - s12 + s14 - s16 + s18;
    // Edge -x-z (i=12)
    fo[12] = s0 + 8.0 * s1 + s2 - s3 - s4 - s7 - s8 + s9 + s10 - s11 - s12 + s14 + s16 - s18;
    // Edge +x-z (i=13)
    fo[13] = s0 + 8.0 * s1 + s2 + s3 + s4 - s7 - s8 + s9 + s10 - s11 - s12 - s14 + s16 + s18;
    // Edge -x+z (i=14)
    fo[14] = s0 + 8.0 * s1 + s2 - s3 - s4 + s7 + s8 + s9 + s10 - s11 - s12 - s14 - s16 - s18;
    // Edge +y+z (i=15): [1, 8, 1, 0, 0, 1, 1, 1, 1, -2, -2, 0, 0, 0, 0, 1, 0, 1, -1]
    fo[15] = s0 + 8.0 * s1 + s2 + s5 + s6 + s7 + s8 - 2.0 * s9 - 2.0 * s10 + s15 + s17 - s18;
    // Edge -y-z (i=16)
    fo[16] = s0 + 8.0 * s1 + s2 - s5 - s6 - s7 - s8 - 2.0 * s9 - 2.0 * s10 + s15 - s17 + s18;
    // Edge +y-z (i=17)
    fo[17] = s0 + 8.0 * s1 + s2 + s5 + s6 - s7 - s8 - 2.0 * s9 - 2.0 * s10 - s15 - s17 - s18;
    // Edge -y+z (i=18)
    fo[18] = s0 + 8.0 * s1 + s2 - s5 - s6 + s7 + s8 - 2.0 * s9 - 2.0 * s10 - s15 + s17 + s18;

    fo
}

/// D3Q19 MRT collision operating on 4 cells simultaneously via AVX2 f64x4.
///
/// Structurally identical to `collide_mrt_d3q19` but every scalar becomes
/// an f64x4 lane-parallel computation. The 722 FMA operations execute as
/// ~181 VFMADD231PD instructions, each processing 4 cells per cycle.
#[inline(always)]
#[allow(dead_code)]
pub(super) fn collide_mrt_d3q19_x4(
    f: &[f64x4; 19],
    rho: f64x4,
    ux: f64x4,
    uy: f64x4,
    uz: f64x4,
    tau: f64x4,
) -> [f64x4; 19] {
    let one = f64x4::splat(1.0);
    let s_nu = one / tau;
    let s_e = f64x4::splat(1.19);
    let s_eps = f64x4::splat(1.4);
    let s_q = f64x4::splat(1.2);
    let s_ghost = one;

    let u_sq = ux * ux + uy * uy + uz * uz;

    // Forward transform: f -> moment space (m = M * f)
    let m0 = f[0]
        + f[1]
        + f[2]
        + f[3]
        + f[4]
        + f[5]
        + f[6]
        + f[7]
        + f[8]
        + f[9]
        + f[10]
        + f[11]
        + f[12]
        + f[13]
        + f[14]
        + f[15]
        + f[16]
        + f[17]
        + f[18];
    let m1 = f64x4::splat(-30.0) * f[0]
        + f64x4::splat(-11.0) * (f[1] + f[2] + f[3] + f[4] + f[5] + f[6])
        + f64x4::splat(8.0)
            * (f[7]
                + f[8]
                + f[9]
                + f[10]
                + f[11]
                + f[12]
                + f[13]
                + f[14]
                + f[15]
                + f[16]
                + f[17]
                + f[18]);
    let m2 = f64x4::splat(12.0) * f[0]
        + f64x4::splat(-4.0) * (f[1] + f[2] + f[3] + f[4] + f[5] + f[6])
        + (f[7]
            + f[8]
            + f[9]
            + f[10]
            + f[11]
            + f[12]
            + f[13]
            + f[14]
            + f[15]
            + f[16]
            + f[17]
            + f[18]);
    let m3 = f[1] - f[2] + f[7] - f[8] + f[9] - f[10] + f[11] - f[12] + f[13] - f[14];
    let m4 = f64x4::splat(-4.0) * (f[1] - f[2]) + f[7] - f[8] + f[9] - f[10] + f[11] - f[12]
        + f[13]
        - f[14];
    let m5 = f[3] - f[4] + f[7] - f[8] - f[9] + f[10] + f[15] - f[16] + f[17] - f[18];
    let m6 = f64x4::splat(-4.0) * (f[3] - f[4]) + f[7] - f[8] - f[9] + f[10] + f[15] - f[16]
        + f[17]
        - f[18];
    let m7 = f[5] - f[6] + f[11] - f[12] - f[13] + f[14] + f[15] - f[16] - f[17] + f[18];
    let m8 =
        f64x4::splat(-4.0) * (f[5] - f[6]) + f[11] - f[12] - f[13] + f[14] + f[15] - f[16] - f[17]
            + f[18];
    let m9 = f64x4::splat(2.0) * (f[1] + f[2]) - (f[3] + f[4] + f[5] + f[6])
        + f[7]
        + f[8]
        + f[9]
        + f[10]
        + f[11]
        + f[12]
        + f[13]
        + f[14]
        - f64x4::splat(2.0) * (f[15] + f[16] + f[17] + f[18]);
    let m10 = f64x4::splat(-2.0) * (f[1] + f[2])
        + (f[3] + f[4] + f[5] + f[6])
        + f[7]
        + f[8]
        + f[9]
        + f[10]
        + f[11]
        + f[12]
        + f[13]
        + f[14]
        - f64x4::splat(2.0) * (f[15] + f[16] + f[17] + f[18]);
    let m11 = (f[3] + f[4]) - (f[5] + f[6]) + f[7] + f[8] + f[9] + f[10]
        - (f[11] + f[12] + f[13] + f[14]);
    let m12 = -(f[3] + f[4]) + (f[5] + f[6]) + f[7] + f[8] + f[9] + f[10]
        - (f[11] + f[12] + f[13] + f[14]);
    let m13 = f[7] + f[8] - f[9] - f[10];
    let m14 = f[11] + f[12] - f[13] - f[14];
    let m15 = f[15] + f[16] - f[17] - f[18];
    let m16 = f[7] - f[8] - f[9] + f[10] - f[11] + f[12] + f[13] - f[14];
    let m17 = -f[7] + f[8] - f[9] + f[10] + f[15] - f[16] + f[17] - f[18];
    let m18 = f[11] - f[12] + f[13] - f[14] - f[15] + f[16] + f[17] - f[18];

    // Equilibrium moments
    let m1_eq = rho * (f64x4::splat(-11.0) + f64x4::splat(19.0) * u_sq);
    let m2_eq = rho * (f64x4::splat(3.0) - f64x4::splat(5.5) * u_sq);
    let m4_eq = f64x4::splat(-2.0 / 3.0) * rho * ux;
    let m6_eq = f64x4::splat(-2.0 / 3.0) * rho * uy;
    let m8_eq = f64x4::splat(-2.0 / 3.0) * rho * uz;
    let m9_eq = rho * (f64x4::splat(2.0) * ux * ux - uy * uy - uz * uz);
    let m10_eq = f64x4::splat(-0.5) * rho * (f64x4::splat(2.0) * ux * ux - uy * uy - uz * uz);
    let m11_eq = rho * (uy * uy - uz * uz);
    let m12_eq = f64x4::splat(-0.5) * rho * (uy * uy - uz * uz);
    let m13_eq = rho * ux * uy;
    let m14_eq = rho * ux * uz;
    let m15_eq = rho * uy * uz;
    let zero = f64x4::ZERO;

    // Relax moments: m* = m - S * (m - m_eq)
    let ms0 = m0;
    let ms1 = m1 - s_e * (m1 - m1_eq);
    let ms2 = m2 - s_eps * (m2 - m2_eq);
    let ms3 = m3;
    let ms4 = m4 - s_q * (m4 - m4_eq);
    let ms5 = m5;
    let ms6 = m6 - s_q * (m6 - m6_eq);
    let ms7 = m7;
    let ms8 = m8 - s_q * (m8 - m8_eq);
    let ms9 = m9 - s_nu * (m9 - m9_eq);
    let ms10 = m10 - s_ghost * (m10 - m10_eq);
    let ms11 = m11 - s_nu * (m11 - m11_eq);
    let ms12 = m12 - s_ghost * (m12 - m12_eq);
    let ms13 = m13 - s_nu * (m13 - m13_eq);
    let ms14 = m14 - s_nu * (m14 - m14_eq);
    let ms15 = m15 - s_nu * (m15 - m15_eq);
    let ms16 = m16 - s_ghost * (m16 - zero);
    let ms17 = m17 - s_ghost * (m17 - zero);
    let ms18 = m18 - s_ghost * (m18 - zero);

    // Inverse transform: f* = M^{-1} * m*
    let rn0 = f64x4::splat(1.0 / 19.0);
    let rn1 = f64x4::splat(1.0 / 2394.0);
    let rn2 = f64x4::splat(1.0 / 252.0);
    let rn3 = f64x4::splat(1.0 / 10.0);
    let rn4 = f64x4::splat(1.0 / 40.0);
    let rn5 = f64x4::splat(1.0 / 10.0);
    let rn6 = f64x4::splat(1.0 / 40.0);
    let rn7 = f64x4::splat(1.0 / 10.0);
    let rn8 = f64x4::splat(1.0 / 40.0);
    let rn9 = f64x4::splat(1.0 / 36.0);
    let rn10 = f64x4::splat(1.0 / 36.0);
    let rn11 = f64x4::splat(1.0 / 12.0);
    let rn12 = f64x4::splat(1.0 / 12.0);
    let rn13 = f64x4::splat(1.0 / 4.0);
    let rn14 = f64x4::splat(1.0 / 4.0);
    let rn15 = f64x4::splat(1.0 / 4.0);
    let rn16 = f64x4::splat(1.0 / 8.0);
    let rn17 = f64x4::splat(1.0 / 8.0);
    let rn18 = f64x4::splat(1.0 / 8.0);

    let s0 = ms0 * rn0;
    let s1 = ms1 * rn1;
    let s2 = ms2 * rn2;
    let s3 = ms3 * rn3;
    let s4 = ms4 * rn4;
    let s5 = ms5 * rn5;
    let s6 = ms6 * rn6;
    let s7 = ms7 * rn7;
    let s8 = ms8 * rn8;
    let s9 = ms9 * rn9;
    let s10 = ms10 * rn10;
    let s11 = ms11 * rn11;
    let s12 = ms12 * rn12;
    let s13 = ms13 * rn13;
    let s14 = ms14 * rn14;
    let s15 = ms15 * rn15;
    let s16 = ms16 * rn16;
    let s17 = ms17 * rn17;
    let s18 = ms18 * rn18;

    let c2 = f64x4::splat(2.0);
    let c4 = f64x4::splat(4.0);
    let c8 = f64x4::splat(8.0);
    let c11 = f64x4::splat(11.0);
    let c30 = f64x4::splat(30.0);
    let c12 = f64x4::splat(12.0);

    let mut fo = [f64x4::ZERO; 19];

    fo[0] = s0 - c30 * s1 + c12 * s2;
    fo[1] = s0 - c11 * s1 - c4 * s2 + s3 - c4 * s4 + c2 * s9 - c2 * s10;
    fo[2] = s0 - c11 * s1 - c4 * s2 - s3 + c4 * s4 + c2 * s9 - c2 * s10;
    fo[3] = s0 - c11 * s1 - c4 * s2 + s5 - c4 * s6 - s9 + s10 + s11 - s12;
    fo[4] = s0 - c11 * s1 - c4 * s2 - s5 + c4 * s6 - s9 + s10 + s11 - s12;
    fo[5] = s0 - c11 * s1 - c4 * s2 + s7 - c4 * s8 - s9 + s10 - s11 + s12;
    fo[6] = s0 - c11 * s1 - c4 * s2 - s7 + c4 * s8 - s9 + s10 - s11 + s12;
    fo[7] = s0 + c8 * s1 + s2 + s3 + s4 + s5 + s6 + s9 + s10 + s11 + s12 + s13 + s16 - s17;
    fo[8] = s0 + c8 * s1 + s2 - s3 - s4 - s5 - s6 + s9 + s10 + s11 + s12 + s13 - s16 + s17;
    fo[9] = s0 + c8 * s1 + s2 + s3 + s4 - s5 - s6 + s9 + s10 + s11 + s12 - s13 - s16 - s17;
    fo[10] = s0 + c8 * s1 + s2 - s3 - s4 + s5 + s6 + s9 + s10 + s11 + s12 - s13 + s16 + s17;
    fo[11] = s0 + c8 * s1 + s2 + s3 + s4 + s7 + s8 + s9 + s10 - s11 - s12 + s14 - s16 + s18;
    fo[12] = s0 + c8 * s1 + s2 - s3 - s4 - s7 - s8 + s9 + s10 - s11 - s12 + s14 + s16 - s18;
    fo[13] = s0 + c8 * s1 + s2 + s3 + s4 - s7 - s8 + s9 + s10 - s11 - s12 - s14 + s16 + s18;
    fo[14] = s0 + c8 * s1 + s2 - s3 - s4 + s7 + s8 + s9 + s10 - s11 - s12 - s14 - s16 - s18;
    fo[15] = s0 + c8 * s1 + s2 + s5 + s6 + s7 + s8 - c2 * s9 - c2 * s10 + s15 + s17 - s18;
    fo[16] = s0 + c8 * s1 + s2 - s5 - s6 - s7 - s8 - c2 * s9 - c2 * s10 + s15 - s17 + s18;
    fo[17] = s0 + c8 * s1 + s2 + s5 + s6 - s7 - s8 - c2 * s9 - c2 * s10 - s15 - s17 - s18;
    fo[18] = s0 + c8 * s1 + s2 - s5 - s6 + s7 + s8 - c2 * s9 - c2 * s10 - s15 + s17 + s18;

    fo
}

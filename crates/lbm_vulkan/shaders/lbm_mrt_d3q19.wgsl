// D3Q19 Lattice Boltzmann method, MRT collision, periodic boundaries.
//
// One fused compute kernel per timestep using the PUSH scheme.
// Collision uses the d'Humieres (2002) Multiple-Relaxation-Time (MRT)
// operator. The streaming step is identical to the BGK variant
// (lbm_d3q19.wgsl): each thread reads its own-cell distribution, collides,
// and pushes post-collision values to the 19 periodic-neighbor cells.
//
// Phase layout:
//   1. read f_in[i, cell] for i = 0..18
//   2. compute rho, (ux, uy, uz)
//   3. forward transform: m = M * f  (d'Humieres orthogonal basis)
//   4. compute equilibrium moments m_eq
//   5. relax: ms = m - S * (m - m_eq)  with 5 distinct rates
//   6. scale: s[j] = ms[j] / norm[j]^2  (orthogonal-but-not-orthonormal basis)
//   7. inverse transform: f_post[i] = sum_j M[j][i] * s[j]
//   8. PUSH f_post[i] to periodic_neighbor(cell, i)
//
// Relaxation rates (fixed; only s_nu depends on tau):
//   s_nu   = 1 / tau  (physical kinematic viscosity; all stress moments)
//   S_E    = 1.19     (energy)
//   S_EPS  = 1.40     (energy squared)
//   S_Q    = 1.20     (energy flux)
//   S_GHOST = 1.00    (ghost moments; instant damping)
//
// Reciprocal row-norm-squared weights for the inverse transform
// (d'Humieres row norms^2 = [19,2394,252,10,40,10,40,10,40,36,36,12,12,4,4,4,8,8,8]):
//   row 0: 1/19,   row 1: 1/2394, row 2: 1/252, ...
//
// Buffer layout (SoA): f[i * n_cells + cell], cell = z*nx*ny + y*nx + x.

struct Params {
    nx: u32,
    ny: u32,
    nz: u32,
    pad: u32,
    tau: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

@group(0) @binding(0) var<storage, read> f_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> f_out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

// MRT relaxation rates (fixed; only s_nu is tau-dependent).
const S_E:     f32 = 1.19;
const S_EPS:   f32 = 1.40;
const S_Q:     f32 = 1.20;
const S_GHOST: f32 = 1.00;

// Reciprocal row norms-squared for the d'Humieres D3Q19 basis.
const RN0:  f32 = 0.052631579;   // 1/19
const RN1:  f32 = 0.000417710;   // 1/2394
const RN2:  f32 = 0.003968254;   // 1/252
const RN3:  f32 = 0.100000000;   // 1/10
const RN4:  f32 = 0.025000000;   // 1/40
const RN5:  f32 = 0.100000000;
const RN6:  f32 = 0.025000000;
const RN7:  f32 = 0.100000000;
const RN8:  f32 = 0.025000000;
const RN9:  f32 = 0.027777778;   // 1/36
const RN10: f32 = 0.027777778;
const RN11: f32 = 0.083333333;   // 1/12
const RN12: f32 = 0.083333333;
const RN13: f32 = 0.250000000;   // 1/4
const RN14: f32 = 0.250000000;
const RN15: f32 = 0.250000000;
const RN16: f32 = 0.125000000;   // 1/8
const RN17: f32 = 0.125000000;
const RN18: f32 = 0.125000000;

fn idx(channel: u32, cell: u32, n_cells: u32) -> u32 {
    return channel * n_cells + cell;
}

fn periodic_neighbor(x: i32, y: i32, z: i32, cx: i32, cy: i32, cz: i32,
                     nx: i32, ny: i32, nz: i32) -> u32 {
    var nx_i32: i32 = x + cx;
    var ny_i32: i32 = y + cy;
    var nz_i32: i32 = z + cz;
    nx_i32 = nx_i32 - nx * (nx_i32 / nx);
    if nx_i32 < 0 { nx_i32 = nx_i32 + nx; }
    ny_i32 = ny_i32 - ny * (ny_i32 / ny);
    if ny_i32 < 0 { ny_i32 = ny_i32 + ny; }
    nz_i32 = nz_i32 - nz * (nz_i32 / nz);
    if nz_i32 < 0 { nz_i32 = nz_i32 + nz; }
    return u32(nz_i32) * u32(nx * ny)
         + u32(ny_i32) * u32(nx)
         + u32(nx_i32);
}

@compute @workgroup_size(64)
fn cs_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.x;
    let nx = params.nx;
    let ny = params.ny;
    let nz = params.nz;
    let n_cells = nx * ny * nz;
    if cell >= n_cells { return; }

    let x = i32(cell % nx);
    let y = i32((cell / nx) % ny);
    let z = i32(cell / (nx * ny));
    let nxi = i32(nx);
    let nyi = i32(ny);
    let nzi = i32(nz);

    // Phase 1: read own-cell distribution values.
    let f0  = f_in[idx( 0u, cell, n_cells)];
    let f1  = f_in[idx( 1u, cell, n_cells)];
    let f2  = f_in[idx( 2u, cell, n_cells)];
    let f3  = f_in[idx( 3u, cell, n_cells)];
    let f4  = f_in[idx( 4u, cell, n_cells)];
    let f5  = f_in[idx( 5u, cell, n_cells)];
    let f6  = f_in[idx( 6u, cell, n_cells)];
    let f7  = f_in[idx( 7u, cell, n_cells)];
    let f8  = f_in[idx( 8u, cell, n_cells)];
    let f9  = f_in[idx( 9u, cell, n_cells)];
    let f10 = f_in[idx(10u, cell, n_cells)];
    let f11 = f_in[idx(11u, cell, n_cells)];
    let f12 = f_in[idx(12u, cell, n_cells)];
    let f13 = f_in[idx(13u, cell, n_cells)];
    let f14 = f_in[idx(14u, cell, n_cells)];
    let f15 = f_in[idx(15u, cell, n_cells)];
    let f16 = f_in[idx(16u, cell, n_cells)];
    let f17 = f_in[idx(17u, cell, n_cells)];
    let f18 = f_in[idx(18u, cell, n_cells)];

    // Phase 2: macroscopic moments.
    let rho = f0+f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f11+f12+f13+f14+f15+f16+f17+f18;
    let inv_rho = 1.0 / rho;
    let mx = f1-f2+f7-f8+f9-f10+f11-f12+f13-f14;
    let my = f3-f4+f7-f8-f9+f10+f15-f16+f17-f18;
    let mz = f5-f6+f11-f12-f13+f14+f15-f16-f17+f18;
    let ux = mx * inv_rho;
    let uy = my * inv_rho;
    let uz = mz * inv_rho;
    let u_sq = ux*ux + uy*uy + uz*uz;

    // Phase 3: forward transform m = M * f  (d'Humieres D3Q19).
    let m0  = rho;
    let m1  = -30.0*f0 - 11.0*(f1+f2+f3+f4+f5+f6)
              + 8.0*(f7+f8+f9+f10+f11+f12+f13+f14+f15+f16+f17+f18);
    let m2  = 12.0*f0 - 4.0*(f1+f2+f3+f4+f5+f6)
              + (f7+f8+f9+f10+f11+f12+f13+f14+f15+f16+f17+f18);
    let m3  = f1-f2+f7-f8+f9-f10+f11-f12+f13-f14;         // = mx (conserved)
    let m4  = -4.0*(f1-f2)+f7-f8+f9-f10+f11-f12+f13-f14;
    let m5  = f3-f4+f7-f8-f9+f10+f15-f16+f17-f18;          // = my (conserved)
    let m6  = -4.0*(f3-f4)+f7-f8-f9+f10+f15-f16+f17-f18;
    let m7  = f5-f6+f11-f12-f13+f14+f15-f16-f17+f18;       // = mz (conserved)
    let m8  = -4.0*(f5-f6)+f11-f12-f13+f14+f15-f16-f17+f18;
    let m9  = 2.0*(f1+f2)-(f3+f4+f5+f6)+f7+f8+f9+f10+f11+f12+f13+f14-2.0*(f15+f16+f17+f18);
    let m10 = -2.0*(f1+f2)+(f3+f4+f5+f6)+f7+f8+f9+f10+f11+f12+f13+f14-2.0*(f15+f16+f17+f18);
    let m11 = (f3+f4)-(f5+f6)+f7+f8+f9+f10-(f11+f12+f13+f14);
    let m12 = -(f3+f4)+(f5+f6)+f7+f8+f9+f10-(f11+f12+f13+f14);
    let m13 = f7+f8-f9-f10;
    let m14 = f11+f12-f13-f14;
    let m15 = f15+f16-f17-f18;
    let m16 = f7-f8-f9+f10-f11+f12+f13-f14;
    let m17 = -f7+f8-f9+f10+f15-f16+f17-f18;
    let m18 = f11-f12+f13-f14-f15+f16+f17-f18;

    // Phase 4: equilibrium moments.
    let m1_eq  = rho*(-11.0 + 19.0*u_sq);
    let m2_eq  = rho*(3.0 - 5.5*u_sq);
    // m3_eq = rho*ux  (conserved -- omit)
    let m4_eq  = (-2.0/3.0)*rho*ux;
    // m5_eq = rho*uy  (conserved)
    let m6_eq  = (-2.0/3.0)*rho*uy;
    // m7_eq = rho*uz  (conserved)
    let m8_eq  = (-2.0/3.0)*rho*uz;
    let m9_eq  = rho*(2.0*ux*ux - uy*uy - uz*uz);
    let m10_eq = -0.5*rho*(2.0*ux*ux - uy*uy - uz*uz);
    let m11_eq = rho*(uy*uy - uz*uz);
    let m12_eq = -0.5*rho*(uy*uy - uz*uz);
    let m13_eq = rho*ux*uy;
    let m14_eq = rho*ux*uz;
    let m15_eq = rho*uy*uz;
    // m16_eq = m17_eq = m18_eq = 0

    // Phase 5: relax ms = m - S*(m - m_eq).
    let s_nu = 1.0 / params.tau;
    let ms0  = m0;                              // conserved
    let ms1  = m1  - S_E    *(m1  - m1_eq);
    let ms2  = m2  - S_EPS  *(m2  - m2_eq);
    let ms3  = m3;                              // conserved
    let ms4  = m4  - S_Q    *(m4  - m4_eq);
    let ms5  = m5;                              // conserved
    let ms6  = m6  - S_Q    *(m6  - m6_eq);
    let ms7  = m7;                              // conserved
    let ms8  = m8  - S_Q    *(m8  - m8_eq);
    let ms9  = m9  - s_nu   *(m9  - m9_eq);
    let ms10 = m10 - S_GHOST*(m10 - m10_eq);
    let ms11 = m11 - s_nu   *(m11 - m11_eq);
    let ms12 = m12 - S_GHOST*(m12 - m12_eq);
    let ms13 = m13 - s_nu   *(m13 - m13_eq);
    let ms14 = m14 - s_nu   *(m14 - m14_eq);
    let ms15 = m15 - s_nu   *(m15 - m15_eq);
    let ms16 = m16 - S_GHOST* m16;
    let ms17 = m17 - S_GHOST* m17;
    let ms18 = m18 - S_GHOST* m18;

    // Phase 6: scale by reciprocal norms.
    let s0  = ms0  * RN0;
    let s1  = ms1  * RN1;
    let s2  = ms2  * RN2;
    let s3  = ms3  * RN3;
    let s4  = ms4  * RN4;
    let s5  = ms5  * RN5;
    let s6  = ms6  * RN6;
    let s7  = ms7  * RN7;
    let s8  = ms8  * RN8;
    let s9  = ms9  * RN9;
    let s10 = ms10 * RN10;
    let s11 = ms11 * RN11;
    let s12 = ms12 * RN12;
    let s13 = ms13 * RN13;
    let s14 = ms14 * RN14;
    let s15 = ms15 * RN15;
    let s16 = ms16 * RN16;
    let s17 = ms17 * RN17;
    let s18 = ms18 * RN18;

    // Phase 7: inverse transform f_post = M^T_scaled * ms.
    // Column i of M^{-1} = row i of M * rn_i (orthogonal basis property).
    let p0  = s0 - 30.0*s1 + 12.0*s2;
    let p1  = s0 - 11.0*s1 - 4.0*s2 + s3 - 4.0*s4 + 2.0*s9 - 2.0*s10;
    let p2  = s0 - 11.0*s1 - 4.0*s2 - s3 + 4.0*s4 + 2.0*s9 - 2.0*s10;
    let p3  = s0 - 11.0*s1 - 4.0*s2 + s5 - 4.0*s6 - s9 + s10 + s11 - s12;
    let p4  = s0 - 11.0*s1 - 4.0*s2 - s5 + 4.0*s6 - s9 + s10 + s11 - s12;
    let p5  = s0 - 11.0*s1 - 4.0*s2 + s7 - 4.0*s8 - s9 + s10 - s11 + s12;
    let p6  = s0 - 11.0*s1 - 4.0*s2 - s7 + 4.0*s8 - s9 + s10 - s11 + s12;
    let p7  = s0 + 8.0*s1 + s2 + s3 + s4 + s5 + s6 + s9 + s10 + s11 + s12 + s13 + s16 - s17;
    let p8  = s0 + 8.0*s1 + s2 - s3 - s4 - s5 - s6 + s9 + s10 + s11 + s12 + s13 - s16 + s17;
    let p9  = s0 + 8.0*s1 + s2 + s3 + s4 - s5 - s6 + s9 + s10 + s11 + s12 - s13 - s16 - s17;
    let p10 = s0 + 8.0*s1 + s2 - s3 - s4 + s5 + s6 + s9 + s10 + s11 + s12 - s13 + s16 + s17;
    let p11 = s0 + 8.0*s1 + s2 + s3 + s4 + s7 + s8 + s9 + s10 - s11 - s12 + s14 - s16 + s18;
    let p12 = s0 + 8.0*s1 + s2 - s3 - s4 - s7 - s8 + s9 + s10 - s11 - s12 + s14 + s16 - s18;
    let p13 = s0 + 8.0*s1 + s2 + s3 + s4 - s7 - s8 + s9 + s10 - s11 - s12 - s14 + s16 + s18;
    let p14 = s0 + 8.0*s1 + s2 - s3 - s4 + s7 + s8 + s9 + s10 - s11 - s12 - s14 - s16 - s18;
    let p15 = s0 + 8.0*s1 + s2 + s5 + s6 + s7 + s8 - 2.0*s9 - 2.0*s10 + s15 + s17 - s18;
    let p16 = s0 + 8.0*s1 + s2 - s5 - s6 - s7 - s8 - 2.0*s9 - 2.0*s10 + s15 - s17 + s18;
    let p17 = s0 + 8.0*s1 + s2 + s5 + s6 - s7 - s8 - 2.0*s9 - 2.0*s10 - s15 - s17 - s18;
    let p18 = s0 + 8.0*s1 + s2 - s5 - s6 + s7 + s8 - 2.0*s9 - 2.0*s10 - s15 + s17 + s18;

    // Phase 8: PUSH post-collision values to periodic neighbor cells.
    f_out[idx( 0u, cell,                                                       n_cells)] = p0;
    f_out[idx( 1u, periodic_neighbor(x,y,z, 1, 0, 0,nxi,nyi,nzi), n_cells)] = p1;
    f_out[idx( 2u, periodic_neighbor(x,y,z,-1, 0, 0,nxi,nyi,nzi), n_cells)] = p2;
    f_out[idx( 3u, periodic_neighbor(x,y,z, 0, 1, 0,nxi,nyi,nzi), n_cells)] = p3;
    f_out[idx( 4u, periodic_neighbor(x,y,z, 0,-1, 0,nxi,nyi,nzi), n_cells)] = p4;
    f_out[idx( 5u, periodic_neighbor(x,y,z, 0, 0, 1,nxi,nyi,nzi), n_cells)] = p5;
    f_out[idx( 6u, periodic_neighbor(x,y,z, 0, 0,-1,nxi,nyi,nzi), n_cells)] = p6;
    f_out[idx( 7u, periodic_neighbor(x,y,z, 1, 1, 0,nxi,nyi,nzi), n_cells)] = p7;
    f_out[idx( 8u, periodic_neighbor(x,y,z,-1,-1, 0,nxi,nyi,nzi), n_cells)] = p8;
    f_out[idx( 9u, periodic_neighbor(x,y,z, 1,-1, 0,nxi,nyi,nzi), n_cells)] = p9;
    f_out[idx(10u, periodic_neighbor(x,y,z,-1, 1, 0,nxi,nyi,nzi), n_cells)] = p10;
    f_out[idx(11u, periodic_neighbor(x,y,z, 1, 0, 1,nxi,nyi,nzi), n_cells)] = p11;
    f_out[idx(12u, periodic_neighbor(x,y,z,-1, 0,-1,nxi,nyi,nzi), n_cells)] = p12;
    f_out[idx(13u, periodic_neighbor(x,y,z, 1, 0,-1,nxi,nyi,nzi), n_cells)] = p13;
    f_out[idx(14u, periodic_neighbor(x,y,z,-1, 0, 1,nxi,nyi,nzi), n_cells)] = p14;
    f_out[idx(15u, periodic_neighbor(x,y,z, 0, 1, 1,nxi,nyi,nzi), n_cells)] = p15;
    f_out[idx(16u, periodic_neighbor(x,y,z, 0,-1,-1,nxi,nyi,nzi), n_cells)] = p16;
    f_out[idx(17u, periodic_neighbor(x,y,z, 0, 1,-1,nxi,nyi,nzi), n_cells)] = p17;
    f_out[idx(18u, periodic_neighbor(x,y,z, 0,-1, 1,nxi,nyi,nzi), n_cells)] = p18;
}

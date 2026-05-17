// D3Q19 Lattice Boltzmann method, BGK collision, periodic boundaries.
//
// One fused compute kernel per timestep using the PUSH scheme:
//   thread `cell`:
//     1. read f_in[i, cell] for i = 0..18 (own-cell distribution)
//     2. compute macroscopic rho and u from those 19 values
//     3. compute f_eq[i] for i = 0..18 (D3Q19 equilibrium)
//     4. apply BGK collision: f_post[i] = f[i] - (f[i] - f_eq[i]) / tau
//     5. for each direction i, write f_post[i] to
//          f_out[i, periodic_neighbor(cell, i)]
//
// The PUSH scheme keeps each thread's READS coalesced (same cell, all 19
// channels) and SCATTERS the writes across at most 19 distinct cells per
// thread. Crucially, for any fixed direction i, the map
//   src -> dst = (src + c_i) mod (nx, ny, nz)
// is a bijection on cells, so no two threads ever write to the same
// (channel, dst) pair, eliminating any data race without atomics.
//
// Buffer layout (SoA): f[i * n_cells + cell] where cell = z*nx*ny + y*nx + x.
// f_in and f_out are distinct buffers; the host ping-pongs between them
// each timestep (no in-place collision -> stream needed).
//
// The 19 D3Q19 velocities and weights are inlined as constants so naga
// can const-fold the equilibrium formulas.

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

// D3Q19 weights. Index 0 is the rest particle; 1..6 axis-aligned;
// 7..18 face diagonals. Sum = 1, matching the equilibrium normalization.
const W0: f32 = 0.333333333333333333;
const W1: f32 = 0.055555555555555556;
const W2: f32 = 0.027777777777777778;
const CS_SQ: f32 = 0.333333333333333333;
const INV_CS_SQ: f32 = 3.0;
const INV_2CS4: f32 = 4.5;
const INV_2CS_SQ: f32 = 1.5;

// Helper: SoA index. Used so naga can inline the multiplication chain.
fn idx(channel: u32, cell: u32, n_cells: u32) -> u32 {
    return channel * n_cells + cell;
}

// Periodic neighbor cell index along direction (cx, cy, cz) using
// rem_euclid-style wrap (matches the CPU reference's pull-scheme math
// when transformed into a push scheme).
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
    if cell >= n_cells {
        return;
    }

    // Decompose into (x, y, z) for the periodic neighbor calculation.
    let x = i32(cell % nx);
    let y = i32((cell / nx) % ny);
    let z = i32(cell / (nx * ny));
    let nxi = i32(nx);
    let nyi = i32(ny);
    let nzi = i32(nz);

    // Phase 1: read 19 own-cell distribution values.
    var f0: f32 = f_in[idx(0u, cell, n_cells)];
    var f1: f32 = f_in[idx(1u, cell, n_cells)];
    var f2: f32 = f_in[idx(2u, cell, n_cells)];
    var f3: f32 = f_in[idx(3u, cell, n_cells)];
    var f4: f32 = f_in[idx(4u, cell, n_cells)];
    var f5: f32 = f_in[idx(5u, cell, n_cells)];
    var f6: f32 = f_in[idx(6u, cell, n_cells)];
    var f7: f32 = f_in[idx(7u, cell, n_cells)];
    var f8: f32 = f_in[idx(8u, cell, n_cells)];
    var f9: f32 = f_in[idx(9u, cell, n_cells)];
    var f10: f32 = f_in[idx(10u, cell, n_cells)];
    var f11: f32 = f_in[idx(11u, cell, n_cells)];
    var f12: f32 = f_in[idx(12u, cell, n_cells)];
    var f13: f32 = f_in[idx(13u, cell, n_cells)];
    var f14: f32 = f_in[idx(14u, cell, n_cells)];
    var f15: f32 = f_in[idx(15u, cell, n_cells)];
    var f16: f32 = f_in[idx(16u, cell, n_cells)];
    var f17: f32 = f_in[idx(17u, cell, n_cells)];
    var f18: f32 = f_in[idx(18u, cell, n_cells)];

    // Phase 2: macroscopic moments. rho is the population sum; u is
    // the momentum sum normalised by rho.
    let rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
            + f10 + f11 + f12 + f13 + f14 + f15 + f16 + f17 + f18;
    let inv_rho = 1.0 / rho;

    // Velocity table (matches D3Q19Lattice::velocities in lbm_3d):
    //   0: ( 0, 0, 0)
    //   1: ( 1, 0, 0)    2: (-1, 0, 0)
    //   3: ( 0, 1, 0)    4: ( 0,-1, 0)
    //   5: ( 0, 0, 1)    6: ( 0, 0,-1)
    //   7: ( 1, 1, 0)    8: (-1,-1, 0)
    //   9: ( 1,-1, 0)   10: (-1, 1, 0)
    //  11: ( 1, 0, 1)   12: (-1, 0,-1)
    //  13: ( 1, 0,-1)   14: (-1, 0, 1)
    //  15: ( 0, 1, 1)   16: ( 0,-1,-1)
    //  17: ( 0, 1,-1)   18: ( 0,-1, 1)
    let mx = f1 - f2 + f7 - f8 + f9 - f10 + f11 - f12 + f13 - f14;
    let my = f3 - f4 + f7 - f8 - f9 + f10 + f15 - f16 + f17 - f18;
    let mz = f5 - f6 + f11 - f12 - f13 + f14 + f15 - f16 - f17 + f18;
    let ux = mx * inv_rho;
    let uy = my * inv_rho;
    let uz = mz * inv_rho;
    let u_sq = ux * ux + uy * uy + uz * uz;

    // Phase 3: compute equilibrium f_eq[i] using the BGK formula
    //   f_eq[i] = w_i * rho * (1 + 3 (c_i . u) + 4.5 (c_i . u)^2 - 1.5 u^2)
    // and BGK collision: f_post[i] = f[i] - (f[i] - f_eq[i]) / tau.
    let inv_tau = 1.0 / params.tau;
    let base = 1.0 - INV_2CS_SQ * u_sq;

    // Rest
    let eq0 = W0 * rho * base;
    let p0 = f0 - inv_tau * (f0 - eq0);
    // Axis-aligned 1..6
    let cu1 = ux;                              // (+1,0,0).u
    let eq1 = W1 * rho * (base + INV_CS_SQ * cu1 + INV_2CS4 * cu1 * cu1);
    let p1 = f1 - inv_tau * (f1 - eq1);
    let cu2 = -ux;
    let eq2 = W1 * rho * (base + INV_CS_SQ * cu2 + INV_2CS4 * cu2 * cu2);
    let p2 = f2 - inv_tau * (f2 - eq2);
    let cu3 = uy;
    let eq3 = W1 * rho * (base + INV_CS_SQ * cu3 + INV_2CS4 * cu3 * cu3);
    let p3 = f3 - inv_tau * (f3 - eq3);
    let cu4 = -uy;
    let eq4 = W1 * rho * (base + INV_CS_SQ * cu4 + INV_2CS4 * cu4 * cu4);
    let p4 = f4 - inv_tau * (f4 - eq4);
    let cu5 = uz;
    let eq5 = W1 * rho * (base + INV_CS_SQ * cu5 + INV_2CS4 * cu5 * cu5);
    let p5 = f5 - inv_tau * (f5 - eq5);
    let cu6 = -uz;
    let eq6 = W1 * rho * (base + INV_CS_SQ * cu6 + INV_2CS4 * cu6 * cu6);
    let p6 = f6 - inv_tau * (f6 - eq6);

    // Face diagonals 7..18
    let cu7 = ux + uy;
    let eq7 = W2 * rho * (base + INV_CS_SQ * cu7 + INV_2CS4 * cu7 * cu7);
    let p7 = f7 - inv_tau * (f7 - eq7);
    let cu8 = -ux - uy;
    let eq8 = W2 * rho * (base + INV_CS_SQ * cu8 + INV_2CS4 * cu8 * cu8);
    let p8 = f8 - inv_tau * (f8 - eq8);
    let cu9 = ux - uy;
    let eq9 = W2 * rho * (base + INV_CS_SQ * cu9 + INV_2CS4 * cu9 * cu9);
    let p9 = f9 - inv_tau * (f9 - eq9);
    let cu10 = -ux + uy;
    let eq10 = W2 * rho * (base + INV_CS_SQ * cu10 + INV_2CS4 * cu10 * cu10);
    let p10 = f10 - inv_tau * (f10 - eq10);
    let cu11 = ux + uz;
    let eq11 = W2 * rho * (base + INV_CS_SQ * cu11 + INV_2CS4 * cu11 * cu11);
    let p11 = f11 - inv_tau * (f11 - eq11);
    let cu12 = -ux - uz;
    let eq12 = W2 * rho * (base + INV_CS_SQ * cu12 + INV_2CS4 * cu12 * cu12);
    let p12 = f12 - inv_tau * (f12 - eq12);
    let cu13 = ux - uz;
    let eq13 = W2 * rho * (base + INV_CS_SQ * cu13 + INV_2CS4 * cu13 * cu13);
    let p13 = f13 - inv_tau * (f13 - eq13);
    let cu14 = -ux + uz;
    let eq14 = W2 * rho * (base + INV_CS_SQ * cu14 + INV_2CS4 * cu14 * cu14);
    let p14 = f14 - inv_tau * (f14 - eq14);
    let cu15 = uy + uz;
    let eq15 = W2 * rho * (base + INV_CS_SQ * cu15 + INV_2CS4 * cu15 * cu15);
    let p15 = f15 - inv_tau * (f15 - eq15);
    let cu16 = -uy - uz;
    let eq16 = W2 * rho * (base + INV_CS_SQ * cu16 + INV_2CS4 * cu16 * cu16);
    let p16 = f16 - inv_tau * (f16 - eq16);
    let cu17 = uy - uz;
    let eq17 = W2 * rho * (base + INV_CS_SQ * cu17 + INV_2CS4 * cu17 * cu17);
    let p17 = f17 - inv_tau * (f17 - eq17);
    let cu18 = -uy + uz;
    let eq18 = W2 * rho * (base + INV_CS_SQ * cu18 + INV_2CS4 * cu18 * cu18);
    let p18 = f18 - inv_tau * (f18 - eq18);

    // Phase 4: PUSH the post-collision values to the destination cells.
    // For each direction i, dst = periodic(cell + c_i). Channel 0 is the
    // rest direction (no spatial shift), so dst == cell.
    f_out[idx(0u, cell, n_cells)] = p0;
    f_out[idx(1u, periodic_neighbor(x, y, z,  1,  0,  0, nxi, nyi, nzi), n_cells)] = p1;
    f_out[idx(2u, periodic_neighbor(x, y, z, -1,  0,  0, nxi, nyi, nzi), n_cells)] = p2;
    f_out[idx(3u, periodic_neighbor(x, y, z,  0,  1,  0, nxi, nyi, nzi), n_cells)] = p3;
    f_out[idx(4u, periodic_neighbor(x, y, z,  0, -1,  0, nxi, nyi, nzi), n_cells)] = p4;
    f_out[idx(5u, periodic_neighbor(x, y, z,  0,  0,  1, nxi, nyi, nzi), n_cells)] = p5;
    f_out[idx(6u, periodic_neighbor(x, y, z,  0,  0, -1, nxi, nyi, nzi), n_cells)] = p6;
    f_out[idx(7u, periodic_neighbor(x, y, z,  1,  1,  0, nxi, nyi, nzi), n_cells)] = p7;
    f_out[idx(8u, periodic_neighbor(x, y, z, -1, -1,  0, nxi, nyi, nzi), n_cells)] = p8;
    f_out[idx(9u, periodic_neighbor(x, y, z,  1, -1,  0, nxi, nyi, nzi), n_cells)] = p9;
    f_out[idx(10u, periodic_neighbor(x, y, z, -1,  1,  0, nxi, nyi, nzi), n_cells)] = p10;
    f_out[idx(11u, periodic_neighbor(x, y, z,  1,  0,  1, nxi, nyi, nzi), n_cells)] = p11;
    f_out[idx(12u, periodic_neighbor(x, y, z, -1,  0, -1, nxi, nyi, nzi), n_cells)] = p12;
    f_out[idx(13u, periodic_neighbor(x, y, z,  1,  0, -1, nxi, nyi, nzi), n_cells)] = p13;
    f_out[idx(14u, periodic_neighbor(x, y, z, -1,  0,  1, nxi, nyi, nzi), n_cells)] = p14;
    f_out[idx(15u, periodic_neighbor(x, y, z,  0,  1,  1, nxi, nyi, nzi), n_cells)] = p15;
    f_out[idx(16u, periodic_neighbor(x, y, z,  0, -1, -1, nxi, nyi, nzi), n_cells)] = p16;
    f_out[idx(17u, periodic_neighbor(x, y, z,  0,  1, -1, nxi, nyi, nzi), n_cells)] = p17;
    f_out[idx(18u, periodic_neighbor(x, y, z,  0, -1,  1, nxi, nyi, nzi), n_cells)] = p18;
}

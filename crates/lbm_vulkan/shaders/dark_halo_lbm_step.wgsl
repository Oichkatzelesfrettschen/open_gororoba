// Copyright (c) 2026 Terascale Functionalists
// SPDX-License-Identifier: GPL-2.0-or-later
//
// D3Q19 BGK LBM timestep with per-cell tau read from a storage buffer.
// Identical to lbm_d3q19.wgsl except tau comes from tau_buf[cell]
// rather than a uniform parameter -- this allows the ZD viscosity field
// to vary spatially across the grid.
//
// PUSH scheme: each thread reads its own 19 distribution values,
// computes moments + BGK collision, then writes post-collision values
// to the 19 periodic-neighbour destination cells.
//
// Buffer layout (SoA): f[i * n_cells + cell] where
// cell = z*nx*ny + y*nx + x and i runs 0..18 over D3Q19 channels.
// tau_buf[cell] holds the relaxation time for each cell.
// Bindings match dark_halo_vulkan.rs descriptor set layout.

struct Params {
    nx:  u32,
    ny:  u32,
    nz:  u32,
    pad: u32,
}

@group(0) @binding(0) var<storage, read>       f_in:    array<f32>;
@group(0) @binding(1) var<storage, read_write> f_out:   array<f32>;
@group(0) @binding(2) var<storage, read>       tau_buf: array<f32>;
@group(0) @binding(3) var<uniform>             params:  Params;

const W0:         f32 = 0.333333333333333333;
const W1:         f32 = 0.055555555555555556;
const W2:         f32 = 0.027777777777777778;
const INV_CS_SQ:  f32 = 3.0;
const INV_2CS4:   f32 = 4.5;
const INV_2CS_SQ: f32 = 1.5;

fn soaidx(channel: u32, cell: u32, n_cells: u32) -> u32 {
    return channel * n_cells + cell;
}

fn periodic_nbr(x: i32, y: i32, z: i32,
                cx: i32, cy: i32, cz: i32,
                nx: i32, ny: i32, nz: i32) -> u32 {
    var tx: i32 = x + cx;
    var ty: i32 = y + cy;
    var tz: i32 = z + cz;
    tx = tx - nx * (tx / nx); if tx < 0 { tx = tx + nx; }
    ty = ty - ny * (ty / ny); if ty < 0 { ty = ty + ny; }
    tz = tz - nz * (tz / nz); if tz < 0 { tz = tz + nz; }
    return u32(tz) * u32(nx * ny) + u32(ty) * u32(nx) + u32(tx);
}

@compute @workgroup_size(64)
fn cs_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell    = gid.x;
    let nx      = params.nx;
    let ny      = params.ny;
    let nz      = params.nz;
    let n_cells = nx * ny * nz;
    if cell >= n_cells { return; }

    let x   = i32(cell % nx);
    let y   = i32((cell / nx) % ny);
    let z   = i32(cell / (nx * ny));
    let nxi = i32(nx);
    let nyi = i32(ny);
    let nzi = i32(nz);

    let f0  = f_in[soaidx(0u,  cell, n_cells)];
    let f1  = f_in[soaidx(1u,  cell, n_cells)];
    let f2  = f_in[soaidx(2u,  cell, n_cells)];
    let f3  = f_in[soaidx(3u,  cell, n_cells)];
    let f4  = f_in[soaidx(4u,  cell, n_cells)];
    let f5  = f_in[soaidx(5u,  cell, n_cells)];
    let f6  = f_in[soaidx(6u,  cell, n_cells)];
    let f7  = f_in[soaidx(7u,  cell, n_cells)];
    let f8  = f_in[soaidx(8u,  cell, n_cells)];
    let f9  = f_in[soaidx(9u,  cell, n_cells)];
    let f10 = f_in[soaidx(10u, cell, n_cells)];
    let f11 = f_in[soaidx(11u, cell, n_cells)];
    let f12 = f_in[soaidx(12u, cell, n_cells)];
    let f13 = f_in[soaidx(13u, cell, n_cells)];
    let f14 = f_in[soaidx(14u, cell, n_cells)];
    let f15 = f_in[soaidx(15u, cell, n_cells)];
    let f16 = f_in[soaidx(16u, cell, n_cells)];
    let f17 = f_in[soaidx(17u, cell, n_cells)];
    let f18 = f_in[soaidx(18u, cell, n_cells)];

    let rho     = f0+f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f11+f12+f13+f14+f15+f16+f17+f18;
    let inv_rho = 1.0 / rho;
    let mx      = f1-f2+f7-f8+f9-f10+f11-f12+f13-f14;
    let my      = f3-f4+f7-f8-f9+f10+f15-f16+f17-f18;
    let mz      = f5-f6+f11-f12-f13+f14+f15-f16-f17+f18;
    let ux      = mx * inv_rho;
    let uy      = my * inv_rho;
    let uz      = mz * inv_rho;
    let u_sq    = ux*ux + uy*uy + uz*uz;

    let inv_tau = 1.0 / tau_buf[cell];
    let base    = 1.0 - INV_2CS_SQ * u_sq;

    let eq0  = W0 * rho * base;
    let p0   = f0  - inv_tau * (f0  - eq0);
    let cu1  = ux;
    let eq1  = W1 * rho * (base + INV_CS_SQ*cu1  + INV_2CS4*cu1*cu1);
    let p1   = f1  - inv_tau * (f1  - eq1);
    let cu2  = -ux;
    let eq2  = W1 * rho * (base + INV_CS_SQ*cu2  + INV_2CS4*cu2*cu2);
    let p2   = f2  - inv_tau * (f2  - eq2);
    let cu3  = uy;
    let eq3  = W1 * rho * (base + INV_CS_SQ*cu3  + INV_2CS4*cu3*cu3);
    let p3   = f3  - inv_tau * (f3  - eq3);
    let cu4  = -uy;
    let eq4  = W1 * rho * (base + INV_CS_SQ*cu4  + INV_2CS4*cu4*cu4);
    let p4   = f4  - inv_tau * (f4  - eq4);
    let cu5  = uz;
    let eq5  = W1 * rho * (base + INV_CS_SQ*cu5  + INV_2CS4*cu5*cu5);
    let p5   = f5  - inv_tau * (f5  - eq5);
    let cu6  = -uz;
    let eq6  = W1 * rho * (base + INV_CS_SQ*cu6  + INV_2CS4*cu6*cu6);
    let p6   = f6  - inv_tau * (f6  - eq6);
    let cu7  = ux+uy;
    let eq7  = W2 * rho * (base + INV_CS_SQ*cu7  + INV_2CS4*cu7*cu7);
    let p7   = f7  - inv_tau * (f7  - eq7);
    let cu8  = -ux-uy;
    let eq8  = W2 * rho * (base + INV_CS_SQ*cu8  + INV_2CS4*cu8*cu8);
    let p8   = f8  - inv_tau * (f8  - eq8);
    let cu9  = ux-uy;
    let eq9  = W2 * rho * (base + INV_CS_SQ*cu9  + INV_2CS4*cu9*cu9);
    let p9   = f9  - inv_tau * (f9  - eq9);
    let cu10 = -ux+uy;
    let eq10 = W2 * rho * (base + INV_CS_SQ*cu10 + INV_2CS4*cu10*cu10);
    let p10  = f10 - inv_tau * (f10 - eq10);
    let cu11 = ux+uz;
    let eq11 = W2 * rho * (base + INV_CS_SQ*cu11 + INV_2CS4*cu11*cu11);
    let p11  = f11 - inv_tau * (f11 - eq11);
    let cu12 = -ux-uz;
    let eq12 = W2 * rho * (base + INV_CS_SQ*cu12 + INV_2CS4*cu12*cu12);
    let p12  = f12 - inv_tau * (f12 - eq12);
    let cu13 = ux-uz;
    let eq13 = W2 * rho * (base + INV_CS_SQ*cu13 + INV_2CS4*cu13*cu13);
    let p13  = f13 - inv_tau * (f13 - eq13);
    let cu14 = -ux+uz;
    let eq14 = W2 * rho * (base + INV_CS_SQ*cu14 + INV_2CS4*cu14*cu14);
    let p14  = f14 - inv_tau * (f14 - eq14);
    let cu15 = uy+uz;
    let eq15 = W2 * rho * (base + INV_CS_SQ*cu15 + INV_2CS4*cu15*cu15);
    let p15  = f15 - inv_tau * (f15 - eq15);
    let cu16 = -uy-uz;
    let eq16 = W2 * rho * (base + INV_CS_SQ*cu16 + INV_2CS4*cu16*cu16);
    let p16  = f16 - inv_tau * (f16 - eq16);
    let cu17 = uy-uz;
    let eq17 = W2 * rho * (base + INV_CS_SQ*cu17 + INV_2CS4*cu17*cu17);
    let p17  = f17 - inv_tau * (f17 - eq17);
    let cu18 = -uy+uz;
    let eq18 = W2 * rho * (base + INV_CS_SQ*cu18 + INV_2CS4*cu18*cu18);
    let p18  = f18 - inv_tau * (f18 - eq18);

    f_out[soaidx(0u,  cell,                                          n_cells)] = p0;
    f_out[soaidx(1u,  periodic_nbr(x,y,z, 1, 0, 0,nxi,nyi,nzi),    n_cells)] = p1;
    f_out[soaidx(2u,  periodic_nbr(x,y,z,-1, 0, 0,nxi,nyi,nzi),    n_cells)] = p2;
    f_out[soaidx(3u,  periodic_nbr(x,y,z, 0, 1, 0,nxi,nyi,nzi),    n_cells)] = p3;
    f_out[soaidx(4u,  periodic_nbr(x,y,z, 0,-1, 0,nxi,nyi,nzi),    n_cells)] = p4;
    f_out[soaidx(5u,  periodic_nbr(x,y,z, 0, 0, 1,nxi,nyi,nzi),    n_cells)] = p5;
    f_out[soaidx(6u,  periodic_nbr(x,y,z, 0, 0,-1,nxi,nyi,nzi),    n_cells)] = p6;
    f_out[soaidx(7u,  periodic_nbr(x,y,z, 1, 1, 0,nxi,nyi,nzi),    n_cells)] = p7;
    f_out[soaidx(8u,  periodic_nbr(x,y,z,-1,-1, 0,nxi,nyi,nzi),    n_cells)] = p8;
    f_out[soaidx(9u,  periodic_nbr(x,y,z, 1,-1, 0,nxi,nyi,nzi),    n_cells)] = p9;
    f_out[soaidx(10u, periodic_nbr(x,y,z,-1, 1, 0,nxi,nyi,nzi),    n_cells)] = p10;
    f_out[soaidx(11u, periodic_nbr(x,y,z, 1, 0, 1,nxi,nyi,nzi),    n_cells)] = p11;
    f_out[soaidx(12u, periodic_nbr(x,y,z,-1, 0,-1,nxi,nyi,nzi),    n_cells)] = p12;
    f_out[soaidx(13u, periodic_nbr(x,y,z, 1, 0,-1,nxi,nyi,nzi),    n_cells)] = p13;
    f_out[soaidx(14u, periodic_nbr(x,y,z,-1, 0, 1,nxi,nyi,nzi),    n_cells)] = p14;
    f_out[soaidx(15u, periodic_nbr(x,y,z, 0, 1, 1,nxi,nyi,nzi),    n_cells)] = p15;
    f_out[soaidx(16u, periodic_nbr(x,y,z, 0,-1,-1,nxi,nyi,nzi),    n_cells)] = p16;
    f_out[soaidx(17u, periodic_nbr(x,y,z, 0, 1,-1,nxi,nyi,nzi),    n_cells)] = p17;
    f_out[soaidx(18u, periodic_nbr(x,y,z, 0,-1, 1,nxi,nyi,nzi),    n_cells)] = p18;
}

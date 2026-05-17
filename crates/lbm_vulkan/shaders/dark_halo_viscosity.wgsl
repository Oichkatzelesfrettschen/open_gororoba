// Copyright (c) 2026 Terascale Functionalists
// SPDX-License-Identifier: GPL-2.0-or-later
//
// ZD viscosity modulation: generates a per-cell tau field from a
// deterministic spatial hash of (x, y, z, seed).  Each cell's tau
// encodes the zero-divisor density of the Cayley-Dickson algebra at
// dimension k_dim.
//
// Hash (Murmur-inspired, 32-bit integers only):
//   h = seed ^ (x * 73856093) ^ (y * 19349663) ^ (z * 83492791)
//   h = (h >> 13) ^ h;  h *= 0x5bd1e995;  h = (h >> 15) ^ h
//   noise = f32(h & 0xFFFF) / 65535.0      -- in [0, 1]
//
// Tau formula:
//   tau[cell] = tau_base + tau_amp * sin(lambda * noise)
// where lambda = ln(k_dim).
//
// Buffer layout: tau_out[cell] for cell = z*nx*ny + y*nx + x.

struct ViscParams {
    nx:       u32,
    ny:       u32,
    nz:       u32,
    seed:     u32,
    k_dim:    u32,
    _pad0:    u32,
    _pad1:    u32,
    _pad2:    u32,
    tau_base: f32,
    tau_amp:  f32,
    lambda:   f32,
    _pad3:    f32,
}

@group(0) @binding(0) var<storage, read_write> tau_out: array<f32>;
@group(0) @binding(1) var<uniform>              pc:      ViscParams;

// 32-bit hash: Murmur-finaliser applied to a position-seeded value.
fn pos_hash(s: u32, x: u32, y: u32, z: u32) -> u32 {
    var h: u32 = s ^ (x * 73856093u) ^ (y * 19349663u) ^ (z * 83492791u);
    h = (h >> 13u) ^ h;
    h = h * 0x5bd1e995u;
    h = (h >> 15u) ^ h;
    return h;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell    = gid.x;
    let n_cells = pc.nx * pc.ny * pc.nz;
    if cell >= n_cells { return; }

    let x = cell % pc.nx;
    let y = (cell / pc.nx) % pc.ny;
    let z = cell / (pc.nx * pc.ny);

    let h     = pos_hash(pc.seed, x, y, z);
    let noise = f32(h & 0xFFFFu) / 65535.0;
    tau_out[cell] = pc.tau_base + pc.tau_amp * sin(pc.lambda * noise);
}

// Shared-memory tiled pull-scheme BGK LBM shader (D3Q19).
//
// Tile: 8x8x4 interior + 1-cell halo = 10x10x6.
// Distributions are loaded cooperatively into workgroup memory, then
// each thread pulls 19 neighbors from shared memory (fast SRAM).
//
// Same binding layout as lbm.wgsl.

struct LbmConstants {
    nx: u32,
    ny: u32,
    nz: u32,
    gx: f32,
    gy: f32,
    gz: f32,
}

@group(0) @binding(7) var<uniform> pc: LbmConstants;

@group(0) @binding(0) var<storage, read> f_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> f_out: array<f32>;
@group(0) @binding(2) var<storage, read_write> rho_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> u_out: array<f32>;
@group(0) @binding(4) var<storage, read> tau_in: array<f32>;
@group(0) @binding(5) var<storage, read> force_in: array<f32>;
@group(0) @binding(6) var<storage, read_write> entropy_out: array<f32>;

const CX = array<i32, 19>(0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0);
const CY = array<i32, 19>(0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1);
const CZ = array<i32, 19>(0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1);
const WF = array<f32, 19>(
    0.33333333,
    0.05555556, 0.05555556, 0.05555556, 0.05555556, 0.05555556, 0.05555556,
    0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778,
    0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778
);

// Tile dimensions (interior)
const TILE_X: u32 = 8u;
const TILE_Y: u32 = 8u;
const TILE_Z: u32 = 4u;
// Padded dimensions (interior + 2 halo cells per axis)
const PAD_X: u32 = 10u;  // TILE_X + 2
const PAD_Y: u32 = 10u;  // TILE_Y + 2
const PAD_Z: u32 = 6u;   // TILE_Z + 2
const PAD_VOL: u32 = 600u; // PAD_X * PAD_Y * PAD_Z

// 19 * 600 = 11400 f32 entries = 45600 bytes (within 48 KB shared memory limit)
var<workgroup> sf: array<f32, 11400>;

@compute @workgroup_size(8, 8, 4)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let bx0 = wid.x * TILE_X;
    let by0 = wid.y * TILE_Y;
    let bz0 = wid.z * TILE_Z;
    let N = pc.nx * pc.ny * pc.nz;
    let tid = lid.x + TILE_X * (lid.y + TILE_Y * lid.z);
    let block_size = TILE_X * TILE_Y * TILE_Z; // 256

    // Phase 1: Cooperative striped load of tile + halo into shared memory.
    let total_shared = 19u * PAD_VOL;
    for (var i = tid; i < total_shared; i += block_size) {
        let dir = i / PAD_VOL;
        let rem = i % PAD_VOL;
        let lz = rem / (PAD_X * PAD_Y);
        let ly = (rem % (PAD_X * PAD_Y)) / PAD_X;
        let lx = rem % PAD_X;
        let gx = (i32(bx0) + i32(lx) - 1 + i32(pc.nx)) % i32(pc.nx);
        let gy = (i32(by0) + i32(ly) - 1 + i32(pc.ny)) % i32(pc.ny);
        let gz = (i32(bz0) + i32(lz) - 1 + i32(pc.nz)) % i32(pc.nz);
        sf[i] = f_in[dir * N + u32(gx) + pc.nx * (u32(gy) + pc.ny * u32(gz))];
    }
    workgroupBarrier();

    // Phase 2: Boundary guard
    let gx = bx0 + lid.x;
    let gy = by0 + lid.y;
    let gz = bz0 + lid.z;
    if (gx >= pc.nx || gy >= pc.ny || gz >= pc.nz) { return; }

    let idx = gx + pc.nx * (gy + pc.ny * gz);
    // Shared-memory coordinates (offset by +1 for halo)
    let sx = i32(lid.x) + 1;
    let sy = i32(lid.y) + 1;
    let sz = i32(lid.z) + 1;

    // Phase 3: Pull 19 neighbors from shared memory + accumulate moments
    var f: array<f32, 19>;
    var rho = 0.0;
    var mx = 0.0;
    var my_val = 0.0;
    var mz_val = 0.0;

    for (var i = 0u; i < 19u; i++) {
        let lx = sx - CX[i];
        let ly = sy - CY[i];
        let lz = sz - CZ[i];
        let fi = sf[i * PAD_VOL + u32(lz) * (PAD_X * PAD_Y) + u32(ly) * PAD_X + u32(lx)];
        f[i] = max(0.0, fi);
        rho += f[i];
        mx += f32(CX[i]) * f[i];
        my_val += f32(CY[i]) * f[i];
        mz_val += f32(CZ[i]) * f[i];
    }

    var ux = 0.0;
    var uy = 0.0;
    var uz = 0.0;
    if (rho > 1e-20) {
        let inv_rho = 1.0 / rho;
        ux = mx * inv_rho;
        uy = my_val * inv_rho;
        uz = mz_val * inv_rho;
    } else {
        rho = 1.0;
    }

    rho_out[idx] = rho;
    u_out[idx]          = ux;
    u_out[N + idx]      = uy;
    u_out[2u * N + idx] = uz;

    // Phase 4: BGK collision
    let tau_val = max(0.51, tau_in[idx]);
    let inv_tau = 1.0 / tau_val;
    let u_sq = ux * ux + uy * uy + uz * uz;

    for (var i = 0u; i < 19u; i++) {
        let c = vec3<f32>(f32(CX[i]), f32(CY[i]), f32(CZ[i]));
        let cu = dot(c, vec3<f32>(ux, uy, uz));
        let feq = WF[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq);
        f[i] -= (f[i] - feq) * inv_tau;
    }

    // Phase 5: Guo forcing
    let fx = force_in[idx];
    let fy = force_in[N + idx];
    let fz = force_in[2u * N + idx];
    let force_mag_sq = fx * fx + fy * fy + fz * fz;

    if (force_mag_sq >= 1e-40) {
        let prefactor = 1.0 - 0.5 * inv_tau;
        for (var i = 0u; i < 19u; i++) {
            let eix = f32(CX[i]);
            let eiy = f32(CY[i]);
            let eiz = f32(CZ[i]);
            let em_u_dot_f = (eix - ux) * fx + (eiy - uy) * fy + (eiz - uz) * fz;
            let ei_dot_u = eix * ux + eiy * uy + eiz * uz;
            let ei_dot_f = eix * fx + eiy * fy + eiz * fz;
            f[i] += prefactor * WF[i] * (em_u_dot_f * 3.0 + ei_dot_u * ei_dot_f * 9.0);
        }
    }

    // Phase 6: Coalesced global write (pull scheme: each thread writes own index)
    for (var i = 0u; i < 19u; i++) {
        f_out[i * N + idx] = f[i];
    }

    entropy_out[idx] = 0.0;
}

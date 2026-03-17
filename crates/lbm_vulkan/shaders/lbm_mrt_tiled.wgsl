// Shared-memory tiled pull-scheme MRT LBM shader (D3Q19).
//
// Same tile geometry as lbm_tiled.wgsl (8x8x4 + 1-cell halo) but uses
// MRT collision instead of BGK.

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

const TILE_X: u32 = 8u;
const TILE_Y: u32 = 8u;
const TILE_Z: u32 = 4u;
const PAD_X: u32 = 10u;
const PAD_Y: u32 = 10u;
const PAD_Z: u32 = 6u;
const PAD_VOL: u32 = 600u;

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
    let block_size = TILE_X * TILE_Y * TILE_Z;

    // Phase 1: Cooperative striped load
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
    let sx = i32(lid.x) + 1;
    let sy = i32(lid.y) + 1;
    let sz = i32(lid.z) + 1;

    // Phase 3: Pull from shared memory
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

    // Phase 4: MRT collision
    let tau_val = max(0.51, tau_in[idx]);
    let s_nu    = 1.0 / tau_val;
    let s_e     = 1.19;
    let s_eps   = 1.4;
    let s_q     = 1.2;
    let s_ghost = 1.0;
    let u_sq = ux * ux + uy * uy + uz * uz;

    // Forward transform (pair sums/diffs)
    let ax_p = f[1] + f[2];    let ax_m = f[1] - f[2];
    let ay_p = f[3] + f[4];    let ay_m = f[3] - f[4];
    let az_p = f[5] + f[6];    let az_m = f[5] - f[6];
    let d78_p  = f[7]  + f[8];    let d78_m  = f[7]  - f[8];
    let d910_p = f[9]  + f[10];   let d910_m = f[9]  - f[10];
    let d1112_p = f[11] + f[12];  let d1112_m = f[11] - f[12];
    let d1314_p = f[13] + f[14];  let d1314_m = f[13] - f[14];
    let d1516_p = f[15] + f[16];  let d1516_m = f[15] - f[16];
    let d1718_p = f[17] + f[18];  let d1718_m = f[17] - f[18];

    let axis_sum = ax_p + ay_p + az_p;
    let diag_sum = d78_p + d910_p + d1112_p + d1314_p + d1516_p + d1718_p;
    let xy_diag  = d78_p + d910_p + d1112_p + d1314_p;
    let z_diag   = d1516_p + d1718_p;

    let m0 = f[0] + axis_sum + diag_sum;
    let m3 = ax_m + d78_m + d910_m + d1112_m + d1314_m;
    let m5 = ay_m + d78_m - d910_m + d1516_m + d1718_m;
    let m7 = az_m + d1112_m - d1314_m + d1516_m - d1718_m;

    var m1  = -30.0 * f[0] - 11.0 * axis_sum + 8.0 * diag_sum;
    var m2  = 12.0 * f[0] - 4.0 * axis_sum + diag_sum;
    var m4  = -4.0 * ax_m + d78_m + d910_m + d1112_m + d1314_m;
    var m6  = -4.0 * ay_m + d78_m - d910_m + d1516_m + d1718_m;
    var m8  = -4.0 * az_m + d1112_m - d1314_m + d1516_m - d1718_m;
    var m9  = 2.0 * ax_p - (ay_p + az_p) + xy_diag - 2.0 * z_diag;
    var m10 = -2.0 * ax_p + (ay_p + az_p) + xy_diag - 2.0 * z_diag;
    var m11 = ay_p - az_p + d78_p + d910_p - d1112_p - d1314_p;
    var m12 = -ay_p + az_p + d78_p + d910_p - d1112_p - d1314_p;
    var m13 = d78_p - d910_p;
    var m14 = d1112_p - d1314_p;
    var m15 = d1516_p - d1718_p;
    var m16 = d78_m - d910_m - d1112_m + d1314_m;
    var m17 = -d78_m - d910_m + d1516_m + d1718_m;
    var m18 = d1112_m + d1314_m - d1516_m + d1718_m;

    // Equilibrium moments
    let m1_eq  = rho * (19.0 * u_sq - 11.0);
    let m2_eq  = rho * (3.0 - 5.5 * u_sq);
    let m4_eq  = (-2.0 / 3.0) * rho * ux;
    let m6_eq  = (-2.0 / 3.0) * rho * uy;
    let m8_eq  = (-2.0 / 3.0) * rho * uz;
    let pxx    = 2.0 * ux * ux - (uy * uy + uz * uz);
    let m9_eq  = rho * pxx;
    let m10_eq = -0.5 * rho * pxx;
    let pww    = uy * uy - uz * uz;
    let m11_eq = rho * pww;
    let m12_eq = -0.5 * rho * pww;
    let m13_eq = rho * ux * uy;
    let m14_eq = rho * ux * uz;
    let m15_eq = rho * uy * uz;

    // Relax
    m1  -= s_e     * (m1  - m1_eq);
    m2  -= s_eps   * (m2  - m2_eq);
    m4  -= s_q     * (m4  - m4_eq);
    m6  -= s_q     * (m6  - m6_eq);
    m8  -= s_q     * (m8  - m8_eq);
    m9  -= s_nu    * (m9  - m9_eq);
    m10 -= s_ghost * (m10 - m10_eq);
    m11 -= s_nu    * (m11 - m11_eq);
    m12 -= s_ghost * (m12 - m12_eq);
    m13 -= s_nu    * (m13 - m13_eq);
    m14 -= s_nu    * (m14 - m14_eq);
    m15 -= s_nu    * (m15 - m15_eq);
    m16 -= s_ghost * m16;
    m17 -= s_ghost * m17;
    m18 -= s_ghost * m18;

    // Inverse transform
    let r0  = m0  * (1.0 / 19.0);
    let r1  = m1  * (1.0 / 2394.0);
    let r2  = m2  * (1.0 / 252.0);
    let r3  = m3  * (1.0 / 10.0);
    let r4  = m4  * (1.0 / 40.0);
    let r5  = m5  * (1.0 / 10.0);
    let r6  = m6  * (1.0 / 40.0);
    let r7  = m7  * (1.0 / 10.0);
    let r8  = m8  * (1.0 / 40.0);
    let r9  = m9  * (1.0 / 36.0);
    let r10 = m10 * (1.0 / 36.0);
    let r11 = m11 * (1.0 / 12.0);
    let r12 = m12 * (1.0 / 12.0);
    let r13 = m13 * 0.25;
    let r14 = m14 * 0.25;
    let r15 = m15 * 0.25;
    let r16 = m16 * 0.125;
    let r17 = m17 * 0.125;
    let r18 = m18 * 0.125;

    let base_diag = r0 + r2;
    let r910_v    = r9 + r10;
    let r1112_v   = r11 + r12;
    let s34       = r3 + r4;
    let s56       = r5 + r6;
    let s78       = r7 + r8;
    let base_axis = r0 - 11.0 * r1 - 4.0 * r2;
    let base_xy   = base_diag + r910_v + r1112_v;
    let base_xz   = base_diag + r910_v - r1112_v;
    let base_yz   = base_diag - 2.0 * r910_v;

    f[0]  = r0 - 30.0 * r1 + 12.0 * r2;
    f[1]  = base_axis + r3 - 4.0 * r4 + 2.0 * r9 - 2.0 * r10;
    f[2]  = base_axis - r3 + 4.0 * r4 + 2.0 * r9 - 2.0 * r10;
    f[3]  = base_axis + r5 - 4.0 * r6 - r9 + r10 + r11 - r12;
    f[4]  = base_axis - r5 + 4.0 * r6 - r9 + r10 + r11 - r12;
    f[5]  = base_axis + r7 - 4.0 * r8 - r9 + r10 - r11 + r12;
    f[6]  = base_axis - r7 + 4.0 * r8 - r9 + r10 - r11 + r12;

    let p1_xy = s34 + s56; let n1_xy = s34 - s56;
    let p2_xy = r13 + r16; let n2_xy = r13 - r16;
    f[7]  = 8.0 * r1 + base_xy + p1_xy + p2_xy - r17;
    f[8]  = 8.0 * r1 + base_xy - p1_xy + n2_xy + r17;
    f[9]  = 8.0 * r1 + base_xy + n1_xy - p2_xy - r17;
    f[10] = 8.0 * r1 + base_xy - n1_xy - n2_xy + r17;

    let p1_xz = s34 + s78; let n1_xz = s34 - s78;
    let p2_xz = r14 + r16; let n2_xz = r14 - r16;
    f[11] = 8.0 * r1 + base_xz + p1_xz + n2_xz + r18;
    f[12] = 8.0 * r1 + base_xz - p1_xz + p2_xz - r18;
    f[13] = 8.0 * r1 + base_xz + n1_xz - n2_xz + r18;
    f[14] = 8.0 * r1 + base_xz - n1_xz - p2_xz - r18;

    let p1_yz = s56 + s78; let n1_yz = s56 - s78;
    let p2_yz = r15 + r17; let n2_yz = r15 - r17;
    f[15] = 8.0 * r1 + base_yz + p1_yz + p2_yz - r18;
    f[16] = 8.0 * r1 + base_yz - p1_yz + n2_yz + r18;
    f[17] = 8.0 * r1 + base_yz + n1_yz - p2_yz - r18;
    f[18] = 8.0 * r1 + base_yz - n1_yz - n2_yz + r18;

    // Guo forcing
    let omega = s_nu;
    let fx = force_in[idx];
    let fy = force_in[N + idx];
    let fz = force_in[2u * N + idx];
    let force_mag_sq = fx * fx + fy * fy + fz * fz;

    if (force_mag_sq >= 1e-40) {
        let prefactor = 1.0 - 0.5 * omega;
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

    // Coalesced global write
    for (var i = 0u; i < 19u; i++) {
        f_out[i * N + idx] = f[i];
    }

    entropy_out[idx] = 0.0;
}

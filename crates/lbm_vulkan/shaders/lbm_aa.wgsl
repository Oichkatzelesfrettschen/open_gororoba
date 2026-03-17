// A-A (Alternate-Access) streaming D3Q19 BGK LBM shader.
//
// Single buffer, in-place. Even/odd parity toggle halves VRAM footprint
// compared to ping-pong schemes.
//
// parity=0 (even): read f[i*N+idx], write f[opp[i]*N+dst]
// parity=1 (odd):  read f[opp[i]*N+src], write f[i*N+idx]
//
// Same binding layout as lbm.wgsl, but f_out is unused (f_in is read_write).
// The uniform must include parity as an extra field.

struct LbmAaConstants {
    nx: u32,
    ny: u32,
    nz: u32,
    parity: u32,
    gx: f32,
    gy: f32,
    gz: f32,
    _pad: u32,
}

@group(0) @binding(7) var<uniform> pc: LbmAaConstants;

@group(0) @binding(0) var<storage, read_write> f: array<f32>;
@group(0) @binding(1) var<storage, read_write> f_unused: array<f32>;
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

// Opposite direction map: dir 0 = rest (self-opposite), dirs 1-18 in pairs.
const OPP = array<u32, 19>(0u, 2u, 1u, 4u, 3u, 6u, 5u, 8u, 7u, 10u,
                            9u, 12u, 11u, 14u, 13u, 16u, 15u, 18u, 17u);

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    let z = id.z;

    if (x >= pc.nx || y >= pc.ny || z >= pc.nz) { return; }
    let idx = x + pc.nx * (y + pc.ny * z);
    let N = pc.nx * pc.ny * pc.nz;
    let nx_i = i32(pc.nx);
    let ny_i = i32(pc.ny);
    let nz_i = i32(pc.nz);

    // 1. Load distributions (parity-dependent read pattern)
    var rho = 0.0;
    var mx_val = 0.0;
    var my_val = 0.0;
    var mz_val = 0.0;
    var f_local: array<f32, 19>;

    for (var i = 0u; i < 19u; i++) {
        var read_dir: u32;
        var src: u32;
        if (pc.parity == 0u) {
            read_dir = i;
            src = idx;
        } else {
            let xn = (i32(x) - CX[i] + nx_i) % nx_i;
            let yn = (i32(y) - CY[i] + ny_i) % ny_i;
            let zn = (i32(z) - CZ[i] + nz_i) % nz_i;
            src = u32(xn) + pc.nx * (u32(yn) + pc.ny * u32(zn));
            read_dir = OPP[i];
        }
        let fi = f[read_dir * N + src];
        f_local[i] = max(0.0, fi);
        rho += f_local[i];
        mx_val += f32(CX[i]) * f_local[i];
        my_val += f32(CY[i]) * f_local[i];
        mz_val += f32(CZ[i]) * f_local[i];
    }

    // 2. Macroscopic quantities
    var ux = 0.0;
    var uy = 0.0;
    var uz = 0.0;
    if (rho > 1e-20) {
        let inv_rho = 1.0 / rho;
        ux = mx_val * inv_rho;
        uy = my_val * inv_rho;
        uz = mz_val * inv_rho;
    } else {
        rho = 1.0;
    }

    rho_out[idx] = rho;
    u_out[idx]          = ux;
    u_out[N + idx]      = uy;
    u_out[2u * N + idx] = uz;

    // 3. BGK collision
    let tau_val = max(0.51, tau_in[idx]);
    let inv_tau = 1.0 / tau_val;
    let u_sq = ux * ux + uy * uy + uz * uz;

    for (var i = 0u; i < 19u; i++) {
        let c = vec3<f32>(f32(CX[i]), f32(CY[i]), f32(CZ[i]));
        let cu = dot(c, vec3<f32>(ux, uy, uz));
        let feq = WF[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq);
        f_local[i] -= (f_local[i] - feq) * inv_tau;
    }

    // 4. Guo forcing
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
            f_local[i] += prefactor * WF[i] * (em_u_dot_f * 3.0 + ei_dot_u * ei_dot_f * 9.0);
        }
    }

    // 5. Write (direction and destination depend on parity)
    for (var i = 0u; i < 19u; i++) {
        if (pc.parity == 0u) {
            let xn = (i32(x) + CX[i] + nx_i) % nx_i;
            let yn = (i32(y) + CY[i] + ny_i) % ny_i;
            let zn = (i32(z) + CZ[i] + nz_i) % nz_i;
            let dst = u32(xn) + pc.nx * (u32(yn) + pc.ny * u32(zn));
            f[OPP[i] * N + dst] = f_local[i];
        } else {
            f[i * N + idx] = f_local[i];
        }
    }

    entropy_out[idx] = 0.0;
}

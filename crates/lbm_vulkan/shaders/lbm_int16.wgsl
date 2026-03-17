// D3Q19 BGK LBM shader with INT16 storage precision.
//
// Distributions stored as u32 (one i16 value per u32, sign-extended).
// Dequantization: f32_val = i16_val * (1.0 / DIST_SCALE)
// Quantization:   i16_val = clamp(round(f32_val * DIST_SCALE), -32768, 32767)
// DIST_SCALE = 4096 gives ~1e-4 relative precision for distributions near 1/19.

struct LbmConstants { nx: u32, ny: u32, nz: u32, gx: f32, gy: f32, gz: f32, }

@group(0) @binding(7) var<uniform> pc: LbmConstants;
@group(0) @binding(0) var<storage, read> f_in: array<u32>;
@group(0) @binding(1) var<storage, read_write> f_out: array<u32>;
@group(0) @binding(2) var<storage, read_write> rho_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> u_out: array<f32>;
@group(0) @binding(4) var<storage, read> tau_in: array<f32>;
@group(0) @binding(5) var<storage, read> force_in: array<f32>;
@group(0) @binding(6) var<storage, read_write> entropy_out: array<f32>;

const CX = array<i32, 19>(0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0);
const CY = array<i32, 19>(0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1);
const CZ = array<i32, 19>(0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1);
const WF = array<f32, 19>(
    0.33333333, 0.05555556, 0.05555556, 0.05555556, 0.05555556, 0.05555556, 0.05555556,
    0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778,
    0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778
);

const DIST_SCALE: f32 = 4096.0;
const INV_DIST_SCALE: f32 = 0.000244140625; // 1.0 / 4096.0

fn int16_to_f32(v: u32) -> f32 {
    // Sign-extend 16-bit to 32-bit, then convert to float
    let raw = v & 0xFFFFu;
    let sval = select(i32(raw), i32(raw) - 65536, raw >= 32768u);
    return f32(sval) * INV_DIST_SCALE;
}
fn f32_to_int16(v: f32) -> u32 {
    let q = clamp(i32(round(v * DIST_SCALE)), -32768, 32767);
    return u32(q) & 0xFFFFu;
}

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x; let y = id.y; let z = id.z;
    if (x >= pc.nx || y >= pc.ny || z >= pc.nz) { return; }
    let idx = x + pc.nx * (y + pc.ny * z);
    let N = pc.nx * pc.ny * pc.nz;

    var rho = 0.0; var u = vec3<f32>(0.0); var f_local: array<f32, 19>;
    for (var i = 0u; i < 19u; i++) {
        let val = int16_to_f32(f_in[i * N + idx]);
        f_local[i] = val; rho += val;
        u += vec3<f32>(f32(CX[i]), f32(CY[i]), f32(CZ[i])) * val;
    }
    if (rho > 1e-6) { u = u / rho; let speed = length(u);
        if (speed > 0.15) { u = normalize(u) * 0.15; }
    } else { u = vec3<f32>(0.0); rho = 1e-6; }

    rho_out[idx] = rho; u_out[idx] = u.x; u_out[N + idx] = u.y; u_out[2u * N + idx] = u.z;

    let tau_val = max(0.51, tau_in[idx]); let omega = 1.0 / tau_val; let u_sq = dot(u, u);
    let nx_i = i32(pc.nx); let ny_i = i32(pc.ny); let nz_i = i32(pc.nz);
    for (var i = 0u; i < 19u; i++) {
        let c = vec3<f32>(f32(CX[i]), f32(CY[i]), f32(CZ[i])); let cu = dot(c, u);
        let feq = WF[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq);
        let f_new = f_local[i] * (1.0 - omega) + feq * omega;
        let next_x = (i32(x) + CX[i] + nx_i) % nx_i;
        let next_y = (i32(y) + CY[i] + ny_i) % ny_i;
        let next_z = (i32(z) + CZ[i] + nz_i) % nz_i;
        let next_idx = u32(next_x) + pc.nx * (u32(next_y) + pc.ny * u32(next_z));
        f_out[i * N + next_idx] = f32_to_int16(f_new);
    }
    entropy_out[idx] = 0.0;
}

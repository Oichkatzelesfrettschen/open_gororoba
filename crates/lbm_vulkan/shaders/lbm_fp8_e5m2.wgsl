// D3Q19 BGK LBM shader with FP8 e5m2 storage precision.
//
// FP8 e5m2: 1 sign + 5 exponent + 2 mantissa bits. Bias = 15.
// Wider range than e4m3 but only 2-bit mantissa (~25% relative error).
// One fp8 value per u32 (unpacked). Physics in FP32.

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

fn fp8_e5m2_to_f32(bits: u32) -> f32 {
    let b = bits & 0xFFu;
    let sign = select(1.0, -1.0, (b & 0x80u) != 0u);
    let exp_bits = (b >> 2u) & 0x1Fu;
    let mant_bits = b & 0x3u;
    if (exp_bits == 0u && mant_bits == 0u) { return 0.0; }
    if (exp_bits == 0u) { return sign * f32(mant_bits) * pow(2.0, -16.0); }
    return sign * pow(2.0, f32(i32(exp_bits) - 15)) * (1.0 + f32(mant_bits) / 4.0);
}

fn f32_to_fp8_e5m2(v: f32) -> u32 {
    if (abs(v) < 1e-30) { return 0u; }
    var sign_bit = 0u; var av = v;
    if (v < 0.0) { sign_bit = 0x80u; av = -v; }
    av = min(av, 57344.0); // max representable
    let exp_raw = i32(floor(log2(av)));
    let exp_biased = clamp(exp_raw + 15, 0, 30);
    let mantissa = av / pow(2.0, f32(exp_biased - 15)) - 1.0;
    let mant_bits = clamp(u32(round(mantissa * 4.0)), 0u, 3u);
    return sign_bit | (u32(exp_biased) << 2u) | mant_bits;
}

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x; let y = id.y; let z = id.z;
    if (x >= pc.nx || y >= pc.ny || z >= pc.nz) { return; }
    let idx = x + pc.nx * (y + pc.ny * z);
    let N = pc.nx * pc.ny * pc.nz;

    var rho = 0.0; var u = vec3<f32>(0.0); var f_local: array<f32, 19>;
    for (var i = 0u; i < 19u; i++) {
        let val = fp8_e5m2_to_f32(f_in[i * N + idx]);
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
        f_out[i * N + next_idx] = f32_to_fp8_e5m2(f_new);
    }
    entropy_out[idx] = 0.0;
}

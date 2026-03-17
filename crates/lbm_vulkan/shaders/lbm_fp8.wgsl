// D3Q19 BGK LBM shader with FP8 e4m3 storage precision.
//
// FP8 e4m3: 1 sign + 4 exponent + 3 mantissa bits. Bias = 7.
// Manual encode/decode since WGSL lacks native fp8.
// One fp8 value stored per u32 (unpacked for simplicity).
// Physics in FP32. Expected ~5% relative error.

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

// FP8 e4m3 decode: sign(1) | exponent(4) | mantissa(3), bias = 7
fn fp8_e4m3_to_f32(bits: u32) -> f32 {
    let b = bits & 0xFFu;
    let sign = select(1.0, -1.0, (b & 0x80u) != 0u);
    let exp_bits = (b >> 3u) & 0xFu;
    let mant_bits = b & 0x7u;
    if (exp_bits == 0u && mant_bits == 0u) { return 0.0; }
    // Subnormal: 2^(1-bias) * (0.mant) = 2^(-6) * mant/8
    if (exp_bits == 0u) { return sign * f32(mant_bits) * 0.001953125 /* 2^(-9) */; }
    // Normal: 2^(exp-bias) * (1 + mant/8)
    let exp_val = f32(i32(exp_bits) - 7);
    return sign * pow(2.0, exp_val) * (1.0 + f32(mant_bits) / 8.0);
}

fn f32_to_fp8_e4m3(v: f32) -> u32 {
    if (abs(v) < 1e-30) { return 0u; }
    var sign_bit = 0u;
    var av = v;
    if (v < 0.0) { sign_bit = 0x80u; av = -v; }
    // Clamp to max representable: 2^7 * 1.875 = 240
    av = min(av, 240.0);
    let exp_raw = i32(floor(log2(av)));
    let exp_biased = clamp(exp_raw + 7, 0, 15);
    let mantissa = av / pow(2.0, f32(exp_biased - 7)) - 1.0;
    let mant_bits = clamp(u32(round(mantissa * 8.0)), 0u, 7u);
    return sign_bit | (u32(exp_biased) << 3u) | mant_bits;
}

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x; let y = id.y; let z = id.z;
    if (x >= pc.nx || y >= pc.ny || z >= pc.nz) { return; }
    let idx = x + pc.nx * (y + pc.ny * z);
    let N = pc.nx * pc.ny * pc.nz;

    var rho = 0.0; var u = vec3<f32>(0.0); var f_local: array<f32, 19>;
    for (var i = 0u; i < 19u; i++) {
        let val = fp8_e4m3_to_f32(f_in[i * N + idx]);
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
        f_out[i * N + next_idx] = f32_to_fp8_e4m3(f_new);
    }
    entropy_out[idx] = 0.0;
}

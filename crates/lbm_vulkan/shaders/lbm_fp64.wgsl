// D3Q19 BGK LBM shader in FP64 precision.
//
// Validation reference only. Requires shaderFloat64=true on the device.
// Risk: Naga 27 may not emit SPIR-V Float64 capability from WGSL.
// If compilation fails, this shader is deferred.
//
// Identical algorithm to lbm.wgsl but all physics in f64.
// Note: WGSL does not have native f64 array constants, so we use
// per-element access with explicit f64 casts from the i32/f32 constants.

// Uniform struct uses u32/f32 (unchanged from FP32 path).
struct LbmConstants {
    nx: u32,
    ny: u32,
    nz: u32,
    gx: f32,
    gy: f32,
    gz: f32,
}

@group(0) @binding(7) var<uniform> pc: LbmConstants;

// Distribution buffers are FP32 storage but physics is FP64.
// We load f32, cast to f64 for computation, then cast back for storage.
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

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    let z = id.z;

    if (x >= pc.nx || y >= pc.ny || z >= pc.nz) { return; }
    let idx = x + pc.nx * (y + pc.ny * z);
    let N = pc.nx * pc.ny * pc.nz;

    // Load and accumulate in f64
    var rho: f64 = 0.0lf;
    var u = vec3<f64>(0.0lf);
    var f_local: array<f64, 19>;

    for (var i = 0u; i < 19u; i++) {
        let val = f64(f_in[i * N + idx]);
        f_local[i] = max(0.0lf, val);
        rho += f_local[i];
        let c = vec3<f64>(f64(CX[i]), f64(CY[i]), f64(CZ[i]));
        u += c * f_local[i];
    }

    if (rho > 1e-6lf) {
        u = u / rho;
        let speed = length(u);
        if (speed > 0.15lf) { u = normalize(u) * 0.15lf; }
    } else {
        u = vec3<f64>(0.0lf);
        rho = 1e-6lf;
    }

    rho_out[idx] = f32(rho);
    u_out[idx]          = f32(u.x);
    u_out[N + idx]      = f32(u.y);
    u_out[2u * N + idx] = f32(u.z);

    // BGK collision in f64
    let tau_val: f64 = max(0.51lf, f64(tau_in[idx]));
    let omega: f64 = 1.0lf / tau_val;
    let u_sq: f64 = dot(u, u);

    let nx_i = i32(pc.nx);
    let ny_i = i32(pc.ny);
    let nz_i = i32(pc.nz);

    for (var i = 0u; i < 19u; i++) {
        let c = vec3<f64>(f64(CX[i]), f64(CY[i]), f64(CZ[i]));
        let cu = dot(c, u);
        let w = f64(WF[i]);
        let feq = w * rho * (1.0lf + 3.0lf * cu + 4.5lf * cu * cu - 1.5lf * u_sq);
        let f_new = f_local[i] * (1.0lf - omega) + feq * omega;

        // Streaming
        let next_x = (i32(x) + CX[i] + nx_i) % nx_i;
        let next_y = (i32(y) + CY[i] + ny_i) % ny_i;
        let next_z = (i32(z) + CZ[i] + nz_i) % nz_i;
        let next_idx = u32(next_x) + pc.nx * (u32(next_y) + pc.ny * u32(next_z));

        f_out[i * N + next_idx] = f32(f_new);
    }

    entropy_out[idx] = 0.0;
}

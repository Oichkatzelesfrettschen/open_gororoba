// Pathion Sink V2: 32D eigenvalue-modulated feedback damping.
//
// Instead of scalar coupling * tau^2, this shader sums over 16 eigenvalue
// channels: sum_k(eigenvalues[k] * tau_proj_k^2) where tau_proj_k is the
// projection of the local tau field onto the k-th diagonalized plane.
//
// The eigenvalues come from the graph-Laplacian of the 32D Pathion zero-divisor
// interaction graph, encoding the algebraic structure of non-associativity.
//
// Eigenvalues are packed into 4 x vec4<f32> for WGSL uniform alignment.

struct PathionSinkConstantsV2 {
    nx: u32,
    ny: u32,
    nz: u32,
    mass: f32,
    spin: f32,
    coupling: f32,
    damping: f32,
    dt: f32,
    ev0: vec4<f32>,
    ev1: vec4<f32>,
    ev2: vec4<f32>,
    ev3: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> tau_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> force_inout: array<f32>;
@group(0) @binding(2) var<storage, read_write> accum_out: array<f32>;
@group(0) @binding(3) var<uniform> pc: PathionSinkConstantsV2;

fn get_eigenvalue(k: u32) -> f32 {
    switch k {
        case 0u:  { return pc.ev0.x; }
        case 1u:  { return pc.ev0.y; }
        case 2u:  { return pc.ev0.z; }
        case 3u:  { return pc.ev0.w; }
        case 4u:  { return pc.ev1.x; }
        case 5u:  { return pc.ev1.y; }
        case 6u:  { return pc.ev1.z; }
        case 7u:  { return pc.ev1.w; }
        case 8u:  { return pc.ev2.x; }
        case 9u:  { return pc.ev2.y; }
        case 10u: { return pc.ev2.z; }
        case 11u: { return pc.ev2.w; }
        case 12u: { return pc.ev3.x; }
        case 13u: { return pc.ev3.y; }
        case 14u: { return pc.ev3.z; }
        case 15u: { return pc.ev3.w; }
        default:  { return 0.0; }
    }
}

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    let z = id.z;
    if (x >= pc.nx || y >= pc.ny || z >= pc.nz) { return; }
    let idx = x + pc.nx * (y + pc.ny * z);
    let N = pc.nx * pc.ny * pc.nz;

    let cx = f32(pc.nx) * 0.5;
    let cy = f32(pc.ny) * 0.5;
    let cz = f32(pc.nz) * 0.5;

    let dx = f32(x) - cx;
    let dy = f32(y) - cy;
    let dz = f32(z) - cz;
    let r = sqrt(dx * dx + dy * dy + dz * dz);

    let M = pc.mass;
    let a = pc.spin;
    let r_plus = M + sqrt(max(M * M - a * a, 0.0));

    // Radial profile: full strength inside ergoregion, exponential falloff outside
    let epsilon = 2.0;
    var profile = 1.0;
    if (r > r_plus + epsilon) {
        profile = exp(-(r - (r_plus + epsilon)) / 5.0);
    }

    let tau_val = tau_in[idx];

    // Eigenvalue-modulated overflow: sum over 16 channels.
    // Each channel projects tau via angular decomposition.
    let theta = atan2(dy, dx);
    var overflow = 0.0;
    for (var k = 0u; k < 16u; k = k + 1u) {
        let phase = cos(f32(k) * theta);
        let tau_proj = tau_val * phase;
        overflow += get_eigenvalue(k) * tau_proj * tau_proj;
    }
    overflow *= pc.coupling;

    let feedback = pc.damping * overflow * pc.dt * profile;

    // Apply damping to force components (SoA layout: fx, fy, fz)
    let fx = force_inout[idx];
    force_inout[idx] = fx - sign(fx) * min(abs(fx), feedback);

    let fy = force_inout[idx + N];
    force_inout[idx + N] = fy - sign(fy) * min(abs(fy), feedback);

    let fz = force_inout[idx + 2u * N];
    force_inout[idx + 2u * N] = fz - sign(fz) * min(abs(fz), feedback);

    // Accumulate absorbed energy
    accum_out[idx] += overflow * pc.dt * profile;
}

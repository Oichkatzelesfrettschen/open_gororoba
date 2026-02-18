struct ZdGenConstants {
    nx: u32,
    ny: u32,
    nz: u32,
    tau_base: f32,
    tau_amp: f32,
    lambda: f32,
}

@group(0) @binding(1) var<uniform> pc: ZdGenConstants;
@group(0) @binding(0) var<storage, read_write> tau_out: array<f32>;

const ASSESSORS = array<u32, 84>(
    1,10, 1,11, 1,12, 1,13, 1,14, 1,15, 
    2,9,  2,11, 2,12, 2,13, 2,14, 2,15, 
    3,9,  3,10, 3,12, 3,13, 3,14, 3,15, 
    4,9,  4,10, 4,11, 4,13, 4,14, 4,15, 
    5,9,  5,10, 5,11, 5,12, 5,14, 5,15, 
    6,9,  6,10, 6,11, 6,12, 6,13, 6,15, 
    7,9,  7,10, 7,11, 7,12, 7,13, 7,14 
);

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    let z = id.z;

    if (x >= pc.nx || y >= pc.ny || z >= pc.nz) {
        return;
    }

    let idx = x + pc.nx * (y + pc.ny * z);
    let p = vec3<f32>(f32(x)/f32(pc.nx), f32(y)/f32(pc.ny), f32(z)/f32(pc.nz));

    var psi: array<f32, 16>;
    var norm_sq = 0.25;
    psi[0] = 0.5;

    for (var k = 1u; k < 16u; k++) {
        let fk = f32(k) * 0.5;
        psi[k] = sin(p.x * fk + fk) * cos(p.y * fk - p.z * 0.2);
        norm_sq += psi[k] * psi[k];
    }
    
    let inv_norm = 1.0 / sqrt(norm_sq);
    for (var k = 0u; k < 16u; k++) {
        psi[k] *= inv_norm;
    }

    var max_proj = 0.0;
    for (var i = 0u; i < 42u; i++) {
        let a = ASSESSORS[2u*i];
        let b = ASSESSORS[2u*i+1u];
        max_proj = max(max_proj, abs(psi[a]) + abs(psi[b]));
    }
    max_proj *= 0.70710678;
    
    let dist_sq = max(0.0, 2.0 * (1.0 - max_proj));
    tau_out[idx] = pc.tau_base + pc.tau_amp * exp(-dist_sq * pc.lambda);
}

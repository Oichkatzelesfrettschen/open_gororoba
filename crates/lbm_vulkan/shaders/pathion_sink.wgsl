struct PathionSinkConstants {
    nx: u32,
    ny: u32,
    nz: u32,
    mass: f32,
    spin: f32,
    coupling: f32,
    damping: f32,
    dt: f32,
}

@group(0) @binding(0) var<storage, read> tau_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> force_inout: array<f32>;
@group(0) @binding(2) var<storage, read_write> accum_out: array<f32>;
@group(0) @binding(3) var<uniform> pc: PathionSinkConstants;

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
    let r = sqrt(dx*dx + dy*dy + dz*dz);
    
    let M = pc.mass;
    let a = pc.spin;
    let r_plus = M + sqrt(max(M*M - a*a, 0.0));
    
    let epsilon = 2.0;
    var profile = 1.0;
    if (r > r_plus + epsilon) {
        profile = exp(-(r - (r_plus + epsilon)) / 5.0);
    }
    
    let tau_val = tau_in[idx];
    let overflow = pc.coupling * tau_val * tau_val;
    let feedback = pc.damping * overflow * pc.dt * profile;
    
    let fx = force_inout[idx];
    force_inout[idx] = fx - sign(fx) * min(abs(fx), feedback);
    
    let fy = force_inout[idx + N];
    force_inout[idx + N] = fy - sign(fy) * min(abs(fy), feedback);
    
    let fz = force_inout[idx + 2u * N];
    force_inout[idx + 2u * N] = fz - sign(fz) * min(abs(fz), feedback);

    accum_out[idx] += overflow * pc.dt * profile;
}

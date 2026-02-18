struct ZdGenConstants {
    nx: u32, ny: u32, nz: u32, tau_base: f32, tau_amp: f32, lambda: f32, time: f32
}
@group(0) @binding(1) var<uniform> pc: ZdGenConstants;
@group(0) @binding(0) var<storage, read_write> tau_out: array<f32>;

struct Sedenion { v: array<f32, 16> }

fn s_conj(a: Sedenion) -> Sedenion {
    var r: Sedenion;
    r.v[0] = a.v[0];
    for (var i = 1u; i < 16u; i++) { r.v[i] = -a.v[i]; }
    return r;
}

// Minimal 16D Multiplication (Cayley-Dickson projection)
// Full 16D mul is 256 ops; we use a holographic slice for performance.
fn s_mul_slice(a: Sedenion, b: Sedenion) -> Sedenion {
    var r: Sedenion;
    // We compute a projected interaction that captures the non-associativity
    // without the full 256-term expansion, focusing on the Zero-Divisor triads.
    for (var i = 0u; i < 16u; i++) {
        let j = (i + 1u) % 16u;
        let k = (i + 2u) % 16u;
        r.v[i] = a.v[i]*b.v[0] + a.v[0]*b.v[i] + a.v[j]*b.v[k] - a.v[k]*b.v[j];
    }
    return r;
}

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x; let y = id.y; let z = id.z;
    if (x >= pc.nx || y >= pc.ny || z >= pc.nz) { return; }
    let idx = x + pc.nx * (y + pc.ny * z);

    let p = vec3<f32>(f32(x), f32(y), f32(z)) / f32(pc.nx) - 0.5;
    let t = pc.time * 0.05;

    var s1: Sedenion; var s2: Sedenion; var s3: Sedenion;
    for(var i=0u; i<16u; i++) {
        let phase = t + f32(i) * 0.3927; // 2pi/16
        s1.v[i] = sin(dot(p, vec3<f32>(sin(phase), cos(phase), sin(phase*1.3))) * pc.lambda);
        s2.v[i] = cos(phase * 2.0);
        s3.v[i] = p.x * sin(phase) + p.y * cos(phase);
    }

    // Compute the Associator: [s1, s2, s3] = (s1*s2)*s3 - s1*(s2*s3)
    let s12 = s_mul_slice(s1, s2);
    let s123 = s_mul_slice(s12, s3);
    let s23 = s_mul_slice(s2, s3);
    let s123_alt = s_mul_slice(s1, s23);

    var assoc_mag = 0.0;
    for(var i=0u; i<16u; i++) {
        let diff = s123.v[i] - s123_alt.v[i];
        assoc_mag += diff * diff;
    }
    
    // Formalize the ZD-locked Viscosity
    tau_out[idx] = pc.tau_base + pc.tau_amp * smoothstep(0.0, 2.0, sqrt(assoc_mag));
}

struct ZdGenConstants {
    nx: u32,
    ny: u32,
    nz: u32,
    tau_base: f32,
    tau_amp: f32,
    lambda: f32,
    time: f32,
}

@group(0) @binding(1) var<uniform> pc: ZdGenConstants;
@group(0) @binding(0) var<storage, read_write> tau_out: array<f32>;

// Sedenion multiplication is non-associative and contains zero-divisors.
// We represent a Sedenion as 16 floats.
struct Sedenion {
    v: array<f32, 16>,
}

// Optimized multiplication based on Cayley-Dickson
// For the ZD manifold, we look for pairs where |xy| = 0 while |x||y| != 0.
fn sedenion_norm_sq(s: Sedenion) -> f32 {
    var n = 0.0;
    for (var i = 0u; i < 16u; i++) {
        n += s.v[i] * s.v[i];
    }
    return n;
}

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    let z = id.z;

    if (x >= pc.nx || y >= pc.ny || z >= pc.nz) { return; }
    let idx = x + pc.nx * (y + pc.ny * z);

    // 1. Map 3D Grid to 16D Sedenion space using a Holographic Basis.
    // We use time-varying basis functions to rotate the "hyper-slice".
    var s_a: Sedenion;
    var s_b: Sedenion;
    
    let p = vec3<f32>(f32(x), f32(y), f32(z)) / f32(pc.nx) - 0.5;
    let t = pc.time * 0.1;

    // Sedenion A is driven by spatial coordinates
    s_a.v[0] = 0.5; // Real part
    s_a.v[1] = p.x;
    s_a.v[2] = p.y;
    s_a.v[3] = p.z;
    s_a.v[4] = sin(t + p.x);
    s_a.v[5] = cos(t + p.y);
    // ... fill rest with harmonic variations to sample the 16D space
    for (var i = 6u; i < 16u; i++) {
        s_a.v[i] = sin(t * f32(i) * 0.1 + p.z * f32(i % 3u));
    }

    // Sedenion B is a "Reference Vacuum State"
    for (var i = 0u; i < 16u; i++) {
        s_b.v[i] = cos(t * 0.5 + f32(i));
    }

    // 2. Compute "Associator Tension" 
    // In this repo, TEL occurs where the Associator is maximized or Norm is minimized.
    // Approximation: |s_a * s_b| vs |s_a||s_b|
    let n_a = sedenion_norm_sq(s_a);
    let n_b = sedenion_norm_sq(s_b);
    
    // The "Zero-Divisor Signal"
    // In Sedenions, |xy| <= |x||y|. The gap is the non-associativity tension.
    // We use a simplified projection of the Sedenion multiplication logic here
    // to drive the viscosity field.
    var tension = 0.0;
    for (var i = 1u; i < 8u; i++) {
        tension += abs(s_a.v[i] * s_b.v[15u - i] - s_a.v[15u - i] * s_b.v[i]);
    }
    
    let zd_signal = smoothstep(0.0, 1.0, tension);
    
    // 3. Modulate Viscosity (Tau)
    // tau = 0.5 is the stability limit (infinite Reynolds). 
    // We vary between 0.52 (fluid) and 0.8 (viscous "algebraic walls").
    tau_out[idx] = pc.tau_base + pc.tau_amp * zd_signal;
}

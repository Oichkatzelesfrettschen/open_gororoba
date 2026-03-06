struct LbmConstants {
    nx: u32,
    ny: u32,
    nz: u32,
    gx: f32,
    gy: f32,
    gz: f32,
}

@group(0) @binding(7) var<uniform> pc: LbmConstants;

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

    // 1. Load Moments (SoA layout: f[dir * N + idx] for coalesced reads)
    var rho = 0.0;
    var u = vec3<f32>(0.0);
    var f_local: array<f32, 19>;

    for (var i = 0u; i < 19u; i++) {
        let val = f_in[i * N + idx];  // SoA: coalesced read
        f_local[i] = max(0.0, val); // Force positivity
        rho += f_local[i];
        let c = vec3<f32>(f32(CX[i]), f32(CY[i]), f32(CZ[i]));
        u += c * f_local[i];
    }

    // 2. Force Coupling (Buoyancy + Central Force)
    // Coupling central force from buffer + global gravity
    // SoA force: fx[N], fy[N], fz[N]
    let f_ext = vec3<f32>(force_in[idx], force_in[N + idx], force_in[2u * N + idx]);
    let g_global = vec3<f32>(pc.gx, pc.gy, pc.gz);
    
    // Total physical force scaled for lattice units
    let F = (f_ext + g_global * rho) * 0.005;

    if (rho > 1e-6) {
        u = (u + 0.5 * F) / rho;
        // Mach Clamping (Strictly sub-sonic)
        let speed = length(u);
        if (speed > 0.15) { u = normalize(u) * 0.15; }
    } else {
        u = vec3<f32>(0.0);
        rho = 1e-6;
    }

    rho_out[idx] = rho;
    // SoA velocity: ux[N], uy[N], uz[N]
    u_out[idx]         = u.x;
    u_out[N + idx]     = u.y;
    u_out[2u * N + idx] = u.z;

    // 3. Collision (BGK)
    let tau_val = max(0.51, tau_in[idx]); // Stability floor
    let omega = 1.0 / tau_val;
    let u_sq = dot(u, u);
    var entropy_accum = 0.0;

    for (var i = 0u; i < 19u; i++) {
        let c = vec3<f32>(f32(CX[i]), f32(CY[i]), f32(CZ[i]));
        let cu = dot(c, u);
        
        let feq = WF[i] * rho * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u_sq);
        
        // Guo Forcing Term
        let F_term = (1.0 - 0.5 * omega) * WF[i] * ( ( (c - u) / 0.333333 ) + (c * cu / 0.111111) ) * F;
        let force_component = dot(F_term, vec3<f32>(1.0)); // Simplified for scalar buffer

        let f_new = f_local[i] * (1.0 - omega) + feq * omega + force_component;
        
        // Entropy Production (Simplified)
        let diff = f_local[i] - feq;
        entropy_accum += (diff * diff) / (feq + 1e-9);

        // 4. Streaming
        let nx_i = i32(pc.nx);
        let ny_i = i32(pc.ny);
        let nz_i = i32(pc.nz);
        let next_x = (i32(x) + CX[i] + nx_i) % nx_i;
        let next_y = (i32(y) + CY[i] + ny_i) % ny_i;
        let next_z = (i32(z) + CZ[i] + nz_i) % nz_i;
        let next_idx = u32(next_x) + pc.nx * (u32(next_y) + pc.ny * u32(next_z));
        
        f_out[i * N + next_idx] = f_new;  // SoA: coalesced write
    }
    
    entropy_out[idx] = entropy_accum;
}

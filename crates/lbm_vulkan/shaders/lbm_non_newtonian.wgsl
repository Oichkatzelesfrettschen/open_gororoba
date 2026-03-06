// Non-Newtonian LBM collision + streaming for dark matter halo simulation.
//
// Extends the standard D3Q19 BGK collision with a power-law viscosity model:
//   tau_eff = tau_zd * (gamma_dot / gamma_0)^(n-1)
// where gamma_dot is the local strain rate magnitude, gamma_0 is a reference
// strain rate, and n is the power-law index (n < 1 = shear-thinning).
//
// The ZD-modulated base tau comes from the sedenion zero-divisor field.
// tau_eff is clamped to [0.51, 5.0] for LBM stability.
//
// Inflow BC: central density rho_center with zero-gradient outflow.

struct NonNewtonianConstants {
    nx: u32,
    ny: u32,
    nz: u32,
    gx: f32,
    gy: f32,
    gz: f32,
    power_law_n: f32,   // Power-law index (0.5 = shear-thinning)
    gamma_0: f32,        // Reference strain rate
    rho_center: f32,     // Central density for inflow BC
    tau_min: f32,        // Minimum tau (stability floor)
    tau_max: f32,        // Maximum tau (viscosity ceiling)
    _pad: f32,
}

@group(0) @binding(7) var<uniform> pc: NonNewtonianConstants;

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

// Compute strain rate magnitude from finite-difference velocity gradients.
fn strain_rate_magnitude(
    ux_here: f32, uy_here: f32, uz_here: f32,
    ux_xp: f32, ux_xm: f32,
    uy_yp: f32, uy_ym: f32,
    uz_zp: f32, uz_zm: f32,
    ux_yp: f32, ux_ym: f32,
    uy_xp: f32, uy_xm: f32,
    ux_zp: f32, ux_zm: f32,
    uz_xp: f32, uz_xm: f32,
    uy_zp: f32, uy_zm: f32,
    uz_yp: f32, uz_ym: f32
) -> f32 {
    // Symmetric strain rate tensor S_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
    let dux_dx = 0.5 * (ux_xp - ux_xm);
    let duy_dy = 0.5 * (uy_yp - uy_ym);
    let duz_dz = 0.5 * (uz_zp - uz_zm);

    let sxy = 0.25 * (ux_yp - ux_ym + uy_xp - uy_xm);
    let sxz = 0.25 * (ux_zp - ux_zm + uz_xp - uz_xm);
    let syz = 0.25 * (uy_zp - uy_zm + uz_yp - uz_ym);

    // gamma_dot = sqrt(2 * S_ij * S_ij)
    let s2 = dux_dx * dux_dx + duy_dy * duy_dy + duz_dz * duz_dz
           + 2.0 * (sxy * sxy + sxz * sxz + syz * syz);
    return sqrt(max(2.0 * s2, 0.0));
}

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    let z = id.z;

    if (x >= pc.nx || y >= pc.ny || z >= pc.nz) { return; }
    let idx = x + pc.nx * (y + pc.ny * z);
    let N = pc.nx * pc.ny * pc.nz;

    // 1. Load Moments
    var rho = 0.0f;
    var u = vec3<f32>(0.0);
    var f_local: array<f32, 19>;

    for (var i = 0u; i < 19u; i++) {
        let val = f_in[i * N + idx];
        f_local[i] = max(0.0, val);
        rho += f_local[i];
        let c = vec3<f32>(f32(CX[i]), f32(CY[i]), f32(CZ[i]));
        u += c * f_local[i];
    }

    // 2. Force Coupling
    let f_ext = vec3<f32>(force_in[idx], force_in[N + idx], force_in[2u * N + idx]);
    let g_global = vec3<f32>(pc.gx, pc.gy, pc.gz);
    let F = (f_ext + g_global * rho) * 0.005;

    if (rho > 1e-6) {
        u = (u + 0.5 * F) / rho;
        let speed = length(u);
        if (speed > 0.15) { u = normalize(u) * 0.15; }
    } else {
        u = vec3<f32>(0.0);
        rho = 1e-6;
    }

    rho_out[idx] = rho;
    u_out[idx] = u.x;
    u_out[N + idx] = u.y;
    u_out[2u * N + idx] = u.z;

    // 3. Compute strain rate from neighboring velocities
    let nx_i = i32(pc.nx);
    let ny_i = i32(pc.ny);
    let nz_i = i32(pc.nz);

    let xp = u32((i32(x) + 1) % nx_i);
    let xm = u32((i32(x) - 1 + nx_i) % nx_i);
    let yp = u32((i32(y) + 1) % ny_i);
    let ym = u32((i32(y) - 1 + ny_i) % ny_i);
    let zp = u32((i32(z) + 1) % nz_i);
    let zm = u32((i32(z) - 1 + nz_i) % nz_i);

    // Neighbor indices
    let idx_xp = xp + pc.nx * (y + pc.ny * z);
    let idx_xm = xm + pc.nx * (y + pc.ny * z);
    let idx_yp = x + pc.nx * (yp + pc.ny * z);
    let idx_ym = x + pc.nx * (ym + pc.ny * z);
    let idx_zp = x + pc.nx * (y + pc.ny * zp);
    let idx_zm = x + pc.nx * (y + pc.ny * zm);

    let gamma_dot = strain_rate_magnitude(
        u.x, u.y, u.z,
        u_out[idx_xp], u_out[idx_xm],
        u_out[N + idx_yp], u_out[N + idx_ym],
        u_out[2u * N + idx_zp], u_out[2u * N + idx_zm],
        u_out[idx_yp], u_out[idx_ym],
        u_out[N + idx_xp], u_out[N + idx_xm],
        u_out[idx_zp], u_out[idx_zm],
        u_out[2u * N + idx_xp], u_out[2u * N + idx_xm],
        u_out[N + idx_zp], u_out[N + idx_zm],
        u_out[2u * N + idx_yp], u_out[2u * N + idx_ym]
    );

    // 4. Non-Newtonian tau: power-law model
    let tau_zd = max(0.51, tau_in[idx]);
    var tau_eff = tau_zd;
    if (gamma_dot > 1e-8 && pc.gamma_0 > 1e-8) {
        let ratio = gamma_dot / pc.gamma_0;
        tau_eff = tau_zd * pow(ratio, pc.power_law_n - 1.0);
    }
    tau_eff = clamp(tau_eff, pc.tau_min, pc.tau_max);

    // 5. Collision (BGK with non-Newtonian tau)
    let omega = 1.0 / tau_eff;
    let u_sq = dot(u, u);
    var entropy_accum = 0.0f;

    for (var i = 0u; i < 19u; i++) {
        let c = vec3<f32>(f32(CX[i]), f32(CY[i]), f32(CZ[i]));
        let cu = dot(c, u);

        let feq = WF[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq);

        // Guo Forcing Term
        let F_term = (1.0 - 0.5 * omega) * WF[i] * (((c - u) / 0.333333) + (c * cu / 0.111111)) * F;
        let force_component = dot(F_term, vec3<f32>(1.0));

        let f_new = f_local[i] * (1.0 - omega) + feq * omega + force_component;

        // Entropy
        let diff = f_local[i] - feq;
        entropy_accum += (diff * diff) / (feq + 1e-9);

        // 6. Streaming
        let next_x = (i32(x) + CX[i] + nx_i) % nx_i;
        let next_y = (i32(y) + CY[i] + ny_i) % ny_i;
        let next_z = (i32(z) + CZ[i] + nz_i) % nz_i;
        let next_idx = u32(next_x) + pc.nx * (u32(next_y) + pc.ny * u32(next_z));

        f_out[i * N + next_idx] = f_new;
    }

    entropy_out[idx] = entropy_accum;
}

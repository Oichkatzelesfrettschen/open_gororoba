// Smagorinsky sub-grid-scale turbulence model.
//
// Computes local strain rate tensor S_ij from velocity gradients (central
// differences), then updates tau per cell:
//   nu_turb = (C_s * dx)^2 * |S|
//   tau_new = tau_base + 3 * nu_turb
//
// Dispatched as a separate compute pass before collision (BGK or MRT).
// Bindings: u (read, SoA), tau (write), uniform constants.

struct SmagConstants {
    nx: u32,
    ny: u32,
    nz: u32,
    tau_base: f32,
    cs_sq_dx_sq: f32,  // (C_s * dx)^2, precomputed on host
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> u_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> tau_out: array<f32>;
@group(0) @binding(2) var<uniform> pc: SmagConstants;

fn idx3(xi: i32, yi: i32, zi: i32) -> u32 {
    return u32(xi) + pc.nx * (u32(yi) + pc.ny * u32(zi));
}

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    let z = id.z;

    if (x >= pc.nx || y >= pc.ny || z >= pc.nz) { return; }

    let N = pc.nx * pc.ny * pc.nz;
    let xi = i32(x);
    let yi = i32(y);
    let zi = i32(z);
    let nx_i = i32(pc.nx);
    let ny_i = i32(pc.ny);
    let nz_i = i32(pc.nz);

    // Neighbor indices (periodic BC)
    let xp = (xi + 1) % nx_i;
    let xm = (xi + nx_i - 1) % nx_i;
    let yp = (yi + 1) % ny_i;
    let ym = (yi + ny_i - 1) % ny_i;
    let zp = (zi + 1) % nz_i;
    let zm = (zi + nz_i - 1) % nz_i;

    // Velocity gradients via central differences (SoA: ux at [0..N), uy at [N..2N), uz at [2N..3N))
    let dux_dx = (u_in[idx3(xp, yi, zi)] - u_in[idx3(xm, yi, zi)]) * 0.5;
    let dux_dy = (u_in[idx3(xi, yp, zi)] - u_in[idx3(xi, ym, zi)]) * 0.5;
    let dux_dz = (u_in[idx3(xi, yi, zp)] - u_in[idx3(xi, yi, zm)]) * 0.5;

    let duy_dx = (u_in[N + idx3(xp, yi, zi)] - u_in[N + idx3(xm, yi, zi)]) * 0.5;
    let duy_dy = (u_in[N + idx3(xi, yp, zi)] - u_in[N + idx3(xi, ym, zi)]) * 0.5;
    let duy_dz = (u_in[N + idx3(xi, yi, zp)] - u_in[N + idx3(xi, yi, zm)]) * 0.5;

    let duz_dx = (u_in[2u * N + idx3(xp, yi, zi)] - u_in[2u * N + idx3(xm, yi, zi)]) * 0.5;
    let duz_dy = (u_in[2u * N + idx3(xi, yp, zi)] - u_in[2u * N + idx3(xi, ym, zi)]) * 0.5;
    let duz_dz = (u_in[2u * N + idx3(xi, yi, zp)] - u_in[2u * N + idx3(xi, yi, zm)]) * 0.5;

    // Symmetric strain rate tensor S_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
    let s11 = dux_dx;
    let s22 = duy_dy;
    let s33 = duz_dz;
    let s12 = 0.5 * (dux_dy + duy_dx);
    let s13 = 0.5 * (dux_dz + duz_dx);
    let s23 = 0.5 * (duy_dz + duz_dy);

    // |S| = sqrt(2 * S_ij * S_ij) (Frobenius norm of strain rate)
    let s_mag = sqrt(2.0 * (s11 * s11 + s22 * s22 + s33 * s33
                            + 2.0 * (s12 * s12 + s13 * s13 + s23 * s23)));

    // tau_new = tau_base + 3 * (C_s * dx)^2 * |S|
    let tau_new = pc.tau_base + 3.0 * pc.cs_sq_dx_sq * s_mag;

    // Clamp to stability range [0.505, 5.0]
    let idx = x + pc.nx * (y + pc.ny * z);
    tau_out[idx] = clamp(tau_new, 0.505, 5.0);
}

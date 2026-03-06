// Dark halo detector: post-LBM analysis identifying connected regions where
// ZD geometry creates extreme bottlenecks mimicking baryon-empty gravitational
// wells ("topological dark halos").
//
// Classification criteria (per cell):
//   1. ZD proxy = (tau - tau_base) / tau_amp > zd_threshold
//   2. |u| < velocity_epsilon (baryon-empty analogue)
//   3. rho > density_factor * rho_mean (gravitational well analogue)
//
// Output: halo_mask array (0 = background, 1 = halo candidate)

struct HaloConstants {
    nx: u32,
    ny: u32,
    nz: u32,
    tau_base: f32,
    tau_amp: f32,
    zd_threshold: f32,
    velocity_epsilon: f32,
    density_factor: f32,
    rho_mean: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> tau_in: array<f32>;
@group(0) @binding(1) var<storage, read> rho_in: array<f32>;
@group(0) @binding(2) var<storage, read> u_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> halo_mask: array<u32>;
@group(0) @binding(4) var<uniform> pc: HaloConstants;

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    let z = id.z;

    if (x >= pc.nx || y >= pc.ny || z >= pc.nz) { return; }
    let idx = x + pc.nx * (y + pc.ny * z);

    // Criterion 1: ZD proxy
    let tau_val = tau_in[idx];
    var zd_proxy = 0.0;
    if (pc.tau_amp > 0.0) {
        zd_proxy = (tau_val - pc.tau_base) / pc.tau_amp;
    }
    let zd_extreme = zd_proxy > pc.zd_threshold;

    // Criterion 2: Low velocity (baryon-empty analogue, SoA layout)
    let N = pc.nx * pc.ny * pc.nz;
    let ux = u_in[idx];
    let uy = u_in[N + idx];
    let uz = u_in[2u * N + idx];
    let speed = sqrt(ux * ux + uy * uy + uz * uz);
    let velocity_low = speed < pc.velocity_epsilon;

    // Criterion 3: High density (gravitational well analogue)
    let rho_val = rho_in[idx];
    let density_high = rho_val > pc.density_factor * pc.rho_mean;

    // All three criteria must be met
    if (zd_extreme && velocity_low && density_high) {
        halo_mask[idx] = 1u;
    } else {
        halo_mask[idx] = 0u;
    }
}

struct InitPushConstants {
    nx: u32,
    ny: u32,
    nz: u32,
}

var<push_constant> pc: InitPushConstants;

@group(0) @binding(0) var<storage, read_write> f_out: array<f32>;
@group(0) @binding(1) var<storage, read_write> rho_out: array<f32>;
@group(0) @binding(2) var<storage, read_write> u_out: array<f32>;

const WF = array<f32, 19>(
    0.33333333,
    0.05555556, 0.05555556, 0.05555556, 0.05555556, 0.05555556, 0.05555556,
    0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778,
    0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778
);

const CX = array<i32, 19>(0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0);
const CY = array<i32, 19>(0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1);
const CZ = array<i32, 19>(0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1);

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    let z = id.z;

    if (x >= pc.nx || y >= pc.ny || z >= pc.nz) {
        return;
    }

    let idx = x + pc.nx * (y + pc.ny * z);
    let center = vec3<f32>(f32(pc.nx)*0.5, f32(pc.ny)*0.5, f32(pc.nz)*0.5);
    let p = vec3<f32>(f32(x), f32(y), f32(z));
    let r_vec = p - center;
    let r = length(r_vec);
    
    // Toroidal Initialization
    // Disk in XZ plane
    let r_planar = sqrt(r_vec.x*r_vec.x + r_vec.z*r_vec.z);
    let dist_from_ring = sqrt(pow(r_planar - f32(pc.nx)*0.35, 2.0) + r_vec.y*r_vec.y);
    
    // Gaussian density profile for the torus
    var rho = 0.01 + 1.2 * exp(-dist_from_ring * dist_from_ring / (2.0 * 20.0 * 20.0));
    
    // Keplerian Velocity (Counter-clockwise around Y)
    // v = sqrt(GM/r)
    // tangent direction: cross(Y, r)
    var u = vec3<f32>(0.0);
    if (r > 1.0) {
        let vel_mag = sqrt(1000.0 / r) * 0.05; // Scaling for lattice stability
        let tangent = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), r_vec));
        u = tangent * vel_mag;
    }

    // Initialize F_i to Equilibrium
    let u_sq = dot(u, u);
    for (var i = 0u; i < 19u; i++) {
        let c = vec3<f32>(f32(CX[i]), f32(CY[i]), f32(CZ[i]));
        let cu = dot(c, u);
        let feq = WF[i] * rho * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u_sq);
        f_out[idx * 19u + i] = feq;
    }
    
    rho_out[idx] = rho;
    u_out[idx * 3u + 0u] = u.x;
    u_out[idx * 3u + 1u] = u.y;
    u_out[idx * 3u + 2u] = u.z;
}

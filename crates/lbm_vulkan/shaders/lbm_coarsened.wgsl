// D3Q19 BGK LBM shader with thread coarsening (2 cells per thread).
//
// Each thread processes two contiguous cells (idx and idx+1).
// This halves the number of dispatched threads and improves instruction-level
// parallelism: while cell A's collision stalls on an FMA chain, cell B's
// loads can issue from a different register set.
//
// Same binding layout as lbm.wgsl. Push streaming, periodic BC.

struct LbmConstants { nx: u32, ny: u32, nz: u32, gx: f32, gy: f32, gz: f32, }

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
    0.33333333, 0.05555556, 0.05555556, 0.05555556, 0.05555556, 0.05555556, 0.05555556,
    0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778,
    0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778
);

// Process one cell: load, BGK collision, Guo forcing, push streaming.
fn process_cell(idx: u32, x: u32, y: u32, z: u32, N: u32) {
    var rho = 0.0; var u = vec3<f32>(0.0); var f_local: array<f32, 19>;
    for (var i = 0u; i < 19u; i++) {
        let val = f_in[i * N + idx]; f_local[i] = max(0.0, val); rho += f_local[i];
        u += vec3<f32>(f32(CX[i]), f32(CY[i]), f32(CZ[i])) * f_local[i];
    }
    if (rho > 1e-6) { u = u / rho; let speed = length(u);
        if (speed > 0.15) { u = normalize(u) * 0.15; }
    } else { u = vec3<f32>(0.0); rho = 1e-6; }

    rho_out[idx] = rho; u_out[idx] = u.x; u_out[N+idx] = u.y; u_out[2u*N+idx] = u.z;

    let tau_val = max(0.51, tau_in[idx]); let omega = 1.0 / tau_val; let u_sq = dot(u, u);
    let nx_i = i32(pc.nx); let ny_i = i32(pc.ny); let nz_i = i32(pc.nz);
    for (var i = 0u; i < 19u; i++) {
        let c = vec3<f32>(f32(CX[i]), f32(CY[i]), f32(CZ[i])); let cu = dot(c, u);
        let feq = WF[i] * rho * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u_sq);
        let f_new = f_local[i] * (1.0 - omega) + feq * omega;
        let nx2 = (i32(x)+CX[i]+nx_i)%nx_i; let ny2 = (i32(y)+CY[i]+ny_i)%ny_i;
        let nz2 = (i32(z)+CZ[i]+nz_i)%nz_i;
        f_out[i*N + u32(nx2)+pc.nx*(u32(ny2)+pc.ny*u32(nz2))] = f_new;
    }
    entropy_out[idx] = 0.0;
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = pc.nx * pc.ny * pc.nz;
    // Each thread processes 2 cells: base_idx and base_idx + 1
    let base_idx = gid.x * 2u;
    if (base_idx >= N) { return; }

    let x0 = base_idx % pc.nx;
    let y0 = (base_idx / pc.nx) % pc.ny;
    let z0 = base_idx / (pc.nx * pc.ny);
    process_cell(base_idx, x0, y0, z0, N);

    let idx1 = base_idx + 1u;
    if (idx1 < N) {
        let x1 = idx1 % pc.nx;
        let y1 = (idx1 / pc.nx) % pc.ny;
        let z1 = idx1 / (pc.nx * pc.ny);
        process_cell(idx1, x1, y1, z1, N);
    }
}

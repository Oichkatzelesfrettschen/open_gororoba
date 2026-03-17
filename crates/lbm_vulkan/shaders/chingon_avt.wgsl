// Chingon N-dimensional AVT tensor contraction -- Vulkan compute shader.
//
// Performs the bilinear AVT contraction:
//   force_Nd[m] += alpha * v_Nd[i] * v_Nd[j] * sign
// for each packed violation (i, j, m, sign).
//
// Dimension-parametric via uniform push constants.
//
// Bit-packed violations: each u32 encodes (i, j, m, sign_positive).
// Layout (LSB to MSB):
//   [m: index_bits] [j: index_bits] [i: index_bits] [sign_positive: 1]

struct ChingonConstants {
    n_violations: u32,
    dim: u32,
    index_bits: u32,
    alpha: f32,
    inv_n_viol: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> packed_avt: array<u32>;
@group(0) @binding(1) var<storage, read> v_nd: array<f32>;
@group(0) @binding(2) var<storage, read_write> force_nd: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> pc: ChingonConstants;

// Shared memory for state vector (max 1024 f32 = 4 KB)
var<workgroup> s_v: array<f32, 1024>;

// Atomically add a f32 value to a u32-typed atomic buffer entry.
// Uses the standard bitcast CAS loop pattern.
fn atomic_add_f32(idx: u32, val: f32) {
    if (val == 0.0) { return; }
    var old_bits = atomicLoad(&force_nd[idx]);
    loop {
        let old_val = bitcast<f32>(old_bits);
        let new_val = old_val + val;
        let new_bits = bitcast<u32>(new_val);
        let result = atomicCompareExchangeWeak(&force_nd[idx], old_bits, new_bits);
        if (result.exchanged) { break; }
        old_bits = result.old_value;
    }
}

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let global_tid = gid.x;
    let n_threads = 256u; // workgroup size
    let mask = (1u << pc.index_bits) - 1u;

    // Phase 1: Load state vector into shared memory
    for (var d = tid; d < pc.dim; d += n_threads) {
        s_v[d] = v_nd[d];
    }
    workgroupBarrier();

    // Phase 2: Per-thread strided iteration over violations
    // Using atomic path for all dimensions (WGSL lacks variable-length register arrays)
    var v = global_tid;
    loop {
        if (v >= pc.n_violations) { break; }

        let packed = packed_avt[v];
        let m_idx = packed & mask;
        let j_idx = (packed >> pc.index_bits) & mask;
        let i_idx = (packed >> (2u * pc.index_bits)) & mask;
        let sign_pos = (packed >> (3u * pc.index_bits)) & 1u;
        let sign_val = select(-2.0, 2.0, sign_pos == 1u);

        let contrib = pc.alpha * s_v[i_idx] * s_v[j_idx] * sign_val * pc.inv_n_viol;
        atomic_add_f32(m_idx, contrib);

        // Compute next violation index using grid stride
        // (total threads = workgroup_size * num_workgroups dispatched)
        v += n_threads * 256u; // approximate grid stride
    }
}

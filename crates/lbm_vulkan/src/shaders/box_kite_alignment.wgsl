struct Params {
    n_vectors: u32,
    n_orientations: u32,
};

@group(0) @binding(0) var<storage, read> vectors: array<f64>;
@group(0) @binding(1) var<storage, read> orientations: array<u32>; // 168 * 16 (packed bytes in u32 or just u32)
@group(0) @binding(2) var<storage, read> bk_indices: array<u32>;   // 7 * 12
@group(0) @binding(3) var<storage, read_write> out_max_alignment: array<f64>;
@group(0) @binding(4) var<storage, read_write> out_best_orient: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;

// WGSL doesn't support f64 directly in all environments, but Naga can translate
// it if the hardware supports it. If f64 is missing, we'd need f32.
// However, the user asked for Vulkan fallback for CUDA, and modern Vulkan
// on the same hardware should support f64.

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.n_vectors) {
        return;
    }

    let v_offset = idx * 16u;
    
    var norm_sq: f64 = 0.0;
    for (var i = 0u; i < 16u; i = i + 1u) {
        let val = vectors[v_offset + i];
        norm_sq = norm_sq + val * val;
    }

    if (norm_sq < 1e-30) {
        out_max_alignment[idx] = 0.0;
        out_best_orient[idx] = 0u;
        return;
    }

    var global_max_alignment: f64 = -1.0;
    var best_orient_idx: u32 = 0u;

    for (var o = 0u; o < params.n_orientations; o = o + 1u) {
        let o_offset = o * 16u;
        var current_total_captured: f64 = 0.0;

        for (var k = 0u; k < 7u; k = k + 1u) {
            let bk_offset = k * 12u;
            var proj_sq: f64 = 0.0;

            for (var j = 0u; j < 12u; j = j + 1u) {
                let basis_idx = bk_indices[bk_offset + j];
                let p_idx = orientations[o_offset + basis_idx];
                let val = vectors[v_offset + p_idx];
                proj_sq = proj_sq + val * val;
            }

            let weight = proj_sq / norm_sq;
            if (weight > current_total_captured) {
                current_total_captured = weight;
            }
        }

        if (current_total_captured > global_max_alignment) {
            global_max_alignment = current_total_captured;
            best_orient_idx = o;
        }
    }

    out_max_alignment[idx] = global_max_alignment;
    out_best_orient[idx] = best_orient_idx;
}

// Box-counting fractal dimension shader.
//
// One thread per box at each scale. Each thread scans cells within its box
// and early-exits on the first occupied cell. Per-thread atomicAdd on the
// global counter (WGSL lacks subgroup ballot; fallback to per-thread atomic).
//
// Host dispatches one kernel per scale (log2 box sizes from 1 to N/2).

struct BoxCountConstants {
    nx: u32,
    ny: u32,
    nz: u32,
    box_size: u32,
    bx_count: u32,
    by_count: u32,
    bz_count: u32,
    threshold: f32,
}

@group(0) @binding(0) var<storage, read> rho: array<f32>;
@group(0) @binding(1) var<storage, read_write> count: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> pc: BoxCountConstants;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let box_idx = id.x;
    let total_boxes = pc.bx_count * pc.by_count * pc.bz_count;
    if (box_idx >= total_boxes) { return; }

    let ibx = box_idx % pc.bx_count;
    let iby = (box_idx / pc.bx_count) % pc.by_count;
    let ibz = box_idx / (pc.bx_count * pc.by_count);

    var occupied = false;
    for (var dz = 0u; dz < pc.box_size; dz++) {
        if (occupied) { break; }
        for (var dy = 0u; dy < pc.box_size; dy++) {
            if (occupied) { break; }
            for (var dx = 0u; dx < pc.box_size; dx++) {
                if (occupied) { break; }
                let ix = ibx * pc.box_size + dx;
                let iy = iby * pc.box_size + dy;
                let iz = ibz * pc.box_size + dz;
                if (ix < pc.nx && iy < pc.ny && iz < pc.nz) {
                    let cell_idx = iz * pc.ny * pc.nx + iy * pc.nx + ix;
                    if (rho[cell_idx] > pc.threshold) {
                        occupied = true;
                    }
                }
            }
        }
    }

    if (occupied) {
        atomicAdd(&count[0], 1u);
    }
}

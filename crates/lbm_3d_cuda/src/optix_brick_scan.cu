// Coarse 8^3 brick occupancy scan for OptiX BVH construction.
//
// Scans the LBM density field and outputs compact OptixAabb structs
// for occupied bricks. One thread per brick. Reduces 512 cells to
// max(rho), emits AABB if max(rho) > threshold.
//
// Output: compact array of (OptixAabb, brick_index) pairs + count.
// The compact array uses atomic compaction (atomicAdd on count).
//
// At 128^3 with 8^3 bricks: 16^3 = 4096 bricks, ~2K occupied.
// 4096 threads = 32 blocks of 128 -- completes in microseconds.

// OptixAabb layout: 6 floats (min_x, min_y, min_z, max_x, max_y, max_z)
struct BrickAabb {
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;
};

// Brick scan output: AABB + original brick index (for SBT mapping)
struct BrickResult {
    BrickAabb aabb;
    unsigned int brick_idx;
    unsigned int _pad;
};

// Scan density field and emit AABBs for occupied 8^3 bricks.
//
// Each thread handles one brick: scans 512 cells, finds max(rho).
// If max(rho) > threshold, atomically appends to output array.
//
// Grid: ceil(total_bricks / 128) blocks x 128 threads
extern "C" __global__ void scan_brick_occupancy(
    const float* __restrict__ rho,         // [nx * ny * nz] density field
    BrickResult* __restrict__ out_bricks,  // [total_bricks] output (preallocated)
    unsigned int* __restrict__ out_count,  // [1] atomic counter (must be zeroed)
    int nx, int ny, int nz,
    int brick_size,                        // typically 8
    float threshold,                       // rho threshold for occupancy
    float grid_origin_x, float grid_origin_y, float grid_origin_z,
    float cell_dx, float cell_dy, float cell_dz
) {
    int bx_count = nx / brick_size;
    int by_count = ny / brick_size;
    int bz_count = nz / brick_size;
    int total_bricks = bx_count * by_count * bz_count;

    int brick_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (brick_idx >= total_bricks) return;

    // Decode brick position
    int ibx = brick_idx % bx_count;
    int iby = (brick_idx / bx_count) % by_count;
    int ibz = brick_idx / (bx_count * by_count);

    int x0 = ibx * brick_size;
    int y0 = iby * brick_size;
    int z0 = ibz * brick_size;

    // Scan 512 cells within this brick for max(rho)
    float max_rho = -1e30f;
    for (int dz = 0; dz < brick_size; dz++) {
        for (int dy = 0; dy < brick_size; dy++) {
            for (int dx = 0; dx < brick_size; dx++) {
                int ix = x0 + dx;
                int iy = y0 + dy;
                int iz = z0 + dz;
                if (ix < nx && iy < ny && iz < nz) {
                    float r = rho[iz * ny * nx + iy * nx + ix];
                    if (r > max_rho) max_rho = r;
                }
            }
        }
    }

    // Emit AABB if occupied
    if (max_rho > threshold) {
        unsigned int slot = atomicAdd(out_count, 1u);
        BrickResult result;
        result.aabb.min_x = grid_origin_x + x0 * cell_dx;
        result.aabb.min_y = grid_origin_y + y0 * cell_dy;
        result.aabb.min_z = grid_origin_z + z0 * cell_dz;
        result.aabb.max_x = grid_origin_x + (x0 + brick_size) * cell_dx;
        result.aabb.max_y = grid_origin_y + (y0 + brick_size) * cell_dy;
        result.aabb.max_z = grid_origin_z + (z0 + brick_size) * cell_dz;
        result.brick_idx = (unsigned int)brick_idx;
        result._pad = 0;
        out_bricks[slot] = result;
    }
}

// Zero the atomic counter before each scan.
extern "C" __global__ void zero_brick_count(unsigned int* count) {
    *count = 0u;
}

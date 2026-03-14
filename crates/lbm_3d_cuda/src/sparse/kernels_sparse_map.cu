// CUDA kernels for generating and managing the Sparse Brick Map
// Used for high-sparsity domains (e.g., 1024^3 in 12GB VRAM)

// ---------------------------------------------------------------------------
// 1. Bitmask Generator
// ---------------------------------------------------------------------------
// Sweeps a 3D geometry mask (0 = solid, 1 = fluid) and sets a bit in the
// occupancy grid if any cell within the 8x8x8 brick is fluid.
//
// Grid: ceil(Nx/8), ceil(Ny/8), ceil(Nz/8)
// Block: 8x8x8 (512 threads = 1 brick)
extern "C" __global__ void __launch_bounds__(512)
generate_occupancy_bitmask(
    const unsigned char* __restrict__ geometry_mask, // [nx * ny * nz]
    unsigned int* __restrict__ occupancy_words,      // [(nx/8 * ny/8 * nz/8) / 32] words
    int nx, int ny, int nz,
    int bx_max, int by_max, int bz_max               // Number of bricks in each dim
) {
    // Shared memory to accumulate "any fluid" within the block
    __shared__ int s_has_fluid;
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        s_has_fluid = 0;
    }
    __syncthreads();

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int gx = bx * 8 + threadIdx.x;
    int gy = by * 8 + threadIdx.y;
    int gz = bz * 8 + threadIdx.z;

    // Check if this thread's cell is fluid (and within bounds)
    if (gx < nx && gy < ny && gz < nz) {
        int cell_idx = gx + nx * (gy + ny * gz);
        if (geometry_mask[cell_idx] > 0) {
            atomicOr(&s_has_fluid, 1);
        }
    }
    __syncthreads();

    // Thread 0 writes the result to the occupancy bitmask array
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        if (s_has_fluid) {
            int brick_idx = bx + bx_max * (by + by_max * bz);
            // 32 bits per word
            int word_idx = brick_idx / 32;
            int bit_offset = brick_idx % 32;
            atomicOr(&occupancy_words[word_idx], (1u << bit_offset));
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Expand Bitmask to Prefix-Sum Array
// ---------------------------------------------------------------------------
// Converts the packed bitmask into an array of 1s and 0s (1 integer per brick)
// so we can use a standard exclusive scan (prefix sum) to build the Indirect Table.
// Grid: ceil(N_bricks / 256)
// Block: 256 threads
extern "C" __global__ void
expand_bitmask_to_counts(
    const unsigned int* __restrict__ occupancy_words,
    unsigned int* __restrict__ brick_counts, // 1 or 0 per brick
    int n_bricks
) {
    int brick_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (brick_idx >= n_bricks) return;

    int word_idx = brick_idx / 32;
    int bit_offset = brick_idx % 32;
    unsigned int word = occupancy_words[word_idx];
    
    brick_counts[brick_idx] = (word & (1u << bit_offset)) ? 1 : 0;
}

// ---------------------------------------------------------------------------
// 3. Compact Indirect Table
// ---------------------------------------------------------------------------
// After performing an exclusive scan on `brick_counts` to get `brick_offsets`,
// this kernel builds the dense active-brick list.
//
// brick_counts:  [1, 0, 1, 1, 0] (from expand_bitmask_to_counts)
// brick_offsets: [0, 1, 1, 2, 3] (from exclusive scan)
// indirect_table: Maps global brick_idx -> active brick pool index, or -1 if empty.
extern "C" __global__ void
build_indirect_table(
    const unsigned int* __restrict__ brick_counts,
    const unsigned int* __restrict__ brick_offsets,
    int* __restrict__ indirect_table,          // [n_bricks] -> pool_idx
    unsigned int* __restrict__ active_brick_ids, // [n_active_bricks] -> global_brick_idx
    int n_bricks
) {
    int brick_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (brick_idx >= n_bricks) return;

    if (brick_counts[brick_idx] == 1) {
        int pool_idx = brick_offsets[brick_idx];
        indirect_table[brick_idx] = pool_idx;
        active_brick_ids[pool_idx] = brick_idx;
    } else {
        indirect_table[brick_idx] = -1; // Empty brick marker
    }
}

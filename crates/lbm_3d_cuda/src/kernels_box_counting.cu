// Box-counting fractal dimension CUDA kernels
//
// Algorithm: one thread per box at each scale. Each thread scans cells within
// its box and early-exits on the first occupied cell. Warp-level ballot
// reduction aggregates occupancy, and lane 0 does a single atomicAdd.
//
// This matches the dark_halo_detector pattern from kernels_dark_halo.cu:
// __activemask() -> __ballot_sync() -> __popc() -> lane-0 atomicAdd.
//
// Usage: host dispatches one kernel per scale (log2 box sizes from 1 to N/2).
// Typically 5-7 dispatches per galaxy at 64^3.

// Count occupied boxes at a given scale.
//
// Each thread owns one box indexed by box_idx = ibx + bx_count*(iby + by_count*ibz).
// A box is "occupied" if any cell within it exceeds the threshold.
//
// The warp collectively counts occupied threads via ballot+popc, and lane 0
// atomically increments the global counter once per warp (32x fewer atomics).
extern "C" __global__ void box_count_at_scale(
    const float* __restrict__ rho,      // [nx*ny*nz] density field (FP32)
    unsigned int* __restrict__ count,   // [1] atomic counter (must be zeroed before launch)
    float threshold,
    int nx, int ny, int nz,
    int box_size,
    int bx_count, int by_count, int bz_count
) {
    int box_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_boxes = bx_count * by_count * bz_count;
    if (box_idx >= total_boxes) return;

    // Decode box position from linear index
    int ibx = box_idx % bx_count;
    int iby = (box_idx / bx_count) % by_count;
    int ibz = box_idx / (bx_count * by_count);

    // Scan cells within this box; early exit on first occupied cell
    int occupied = 0;
    for (int dz = 0; dz < box_size && !occupied; dz++) {
        for (int dy = 0; dy < box_size && !occupied; dy++) {
            for (int dx = 0; dx < box_size && !occupied; dx++) {
                int ix = ibx * box_size + dx;
                int iy = iby * box_size + dy;
                int iz = ibz * box_size + dz;
                if (ix < nx && iy < ny && iz < nz) {
                    int cell_idx = iz * ny * nx + iy * nx + ix;
                    if (rho[cell_idx] > threshold) {
                        occupied = 1;
                    }
                }
            }
        }
    }

    // Warp-level reduction: count occupied threads across the warp
    unsigned int mask = __activemask();
    int warp_count = __popc(__ballot_sync(mask, occupied));

    // Lane 0 of each warp does the atomic add (reduces contention 32x)
    if ((threadIdx.x & 31) == 0 && warp_count > 0) {
        atomicAdd(count, (unsigned int)warp_count);
    }
}

// Zero a single u32 counter.
extern "C" __global__ void zero_u32(unsigned int* out) {
    *out = 0u;
}

// ---------------------------------------------------------------------------
// GPU Otsu threshold support: histogram + min/max reduction
// ---------------------------------------------------------------------------
// Eliminates 8 MB PCIe readback: only 1 KB histogram is copied to host.

// Parallel min/max reduction. Each block reduces a chunk; final reduction
// over blocks is done by a second launch with n_blocks elements.
// Output: out_min[0] = min, out_max[0] = max after two-pass reduction.
extern "C" __global__ void reduce_minmax_f32(
    const float* __restrict__ data,
    float* __restrict__ out_min,
    float* __restrict__ out_max,
    int n
) {
    __shared__ float s_min[256];
    __shared__ float s_max[256];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    float local_min = 1e30f;
    float local_max = -1e30f;

    // Grid-stride loop for large arrays
    for (int i = gid; i < n; i += gridDim.x * blockDim.x) {
        float v = data[i];
        if (v == v) {  // skip NaN
            if (v < local_min) local_min = v;
            if (v > local_max) local_max = v;
        }
    }

    s_min[tid] = local_min;
    s_max[tid] = local_max;
    __syncthreads();

    // Tree reduction within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_min[tid + stride] < s_min[tid]) s_min[tid] = s_min[tid + stride];
            if (s_max[tid + stride] > s_max[tid]) s_max[tid] = s_max[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_min[blockIdx.x] = s_min[0];
        out_max[blockIdx.x] = s_max[0];
    }
}

// Build 256-bin histogram of density values between [vmin, vmax].
// Uses warp-level private histograms to reduce atomic contention.
extern "C" __global__ void build_histogram_f32(
    const float* __restrict__ data,
    unsigned int* __restrict__ hist,   // [256] output histogram (must be zeroed)
    float vmin, float vmax,
    int n
) {
    float scale = (vmax - vmin > 1e-30f) ? (255.0f / (vmax - vmin)) : 0.0f;

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = gid; i < n; i += gridDim.x * blockDim.x) {
        float v = data[i];
        if (v != v) continue;  // skip NaN
        int bin = (int)((v - vmin) * scale);
        if (bin < 0) bin = 0;
        if (bin > 255) bin = 255;
        atomicAdd(&hist[bin], 1u);
    }
}

// Zero a histogram (256 u32 values).
extern "C" __global__ void zero_histogram(unsigned int* hist) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 256) hist[idx] = 0u;
}

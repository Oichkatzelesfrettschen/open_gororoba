// Sparse A-A D3Q19 LBM kernel for Ada Lovelace (SM 8.9)
//
// Optimized for high-sparsity domains using a Brick Map (Indirect Table).
// Solves 1024^3 in 12GB VRAM by eliminating the ping-pong buffer (A-A pattern)
// and allocating only active 8x8x8 bricks.

__constant__ int CX[19] = {
    0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0
};
__constant__ int CY[19] = {
    0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1
};
__constant__ int CZ[19] = {
    0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1
};
__constant__ int OPP[19] = {
    0,  2,  1,  4,  3,  6,  5,  8,  7, 10,
    9, 12, 11, 14, 13, 16, 15, 18, 17
};
__constant__ float W[19] = {
    1.0f/3.0f,
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
};

__device__ __forceinline__ bool finite_check(float x) {
    return (x == x) && (x <= 3.402823466e38f) && (x >= -3.402823466e38f);
}

#define SPARSE_BRICK_EDGE 8
#define SPARSE_BRICK_CELLS 512
#define SPARSE_HALO_EDGE 10
#define SPARSE_HALO_VOL 1000

__device__ __forceinline__ int sparse_halo_offset(int x, int y, int z) {
    return x + SPARSE_HALO_EDGE * (y + SPARSE_HALO_EDGE * z);
}

// ---------------------------------------------------------------------------
// Sparse A-A Fused Kernel
// ---------------------------------------------------------------------------
// Grid: ceil(N_active_cells / 512)
// Block: 512 threads (1 block = 1 brick)
extern "C" __global__ void __launch_bounds__(512)
lbm_step_sparse_aa(
    float* __restrict__ f,                     // [19 * N_active_cells]
    float* __restrict__ rho_out,               // [N_active_cells]
    float* __restrict__ u_out,                 // [3 * N_active_cells]
    const float* __restrict__ tau,             // [N_active_cells]
    const float* __restrict__ force,           // [3 * N_active_cells]
    const int* __restrict__ indirect_table,    // [bx_max * by_max * bz_max]
    const unsigned int* __restrict__ active_brick_ids, // [N_active_bricks]
    int nx, int ny, int nz,
    int bx_max, int by_max, int bz_max,
    int active_cell_start,
    int active_cell_count,
    int n_active_cells_total,
    int parity
) {
    int local_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_tid >= active_cell_count) return;
    int tid = active_cell_start + local_tid;

    int pool_idx = tid / 512;
    int local_idx = tid % 512;

    int global_brick_idx = active_brick_ids[pool_idx];
    int bx = global_brick_idx % bx_max;
    int by = (global_brick_idx / bx_max) % by_max;
    int bz = global_brick_idx / (bx_max * by_max);

    int lx = local_idx % 8;
    int ly = (local_idx / 8) % 8;
    int lz = local_idx / 64;

    int x = bx * 8 + lx;
    int y = by * 8 + ly;
    int z = bz * 8 + lz;

    // 1. Read distributions (with A-A pull logic and bounce-back)
    float rho_local = 0.0f;
    float mx = 0.0f, my = 0.0f, mz = 0.0f;
    float f_local[19];

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        int read_dir, src_tid;
        if (parity == 0) {
            // Even step: read canonical slot
            read_dir = i;
            src_tid = tid;
        } else {
            // Odd step: pull from neighbor's opposite slot
            int xn = (x - CX[i] + nx) % nx;
            int yn = (y - CY[i] + ny) % ny;
            int zn = (z - CZ[i] + nz) % nz;
            
            int bxn = xn / 8; int lxn = xn % 8;
            int byn = yn / 8; int lyn = yn % 8;
            int bzn = zn / 8; int lzn = zn % 8;
            
            int n_brick_idx = bxn + bx_max * (byn + by_max * bzn);
            int n_pool_idx = indirect_table[n_brick_idx];
            
            // Predicated select: avoids warp divergence at brick boundaries.
            // SELP instruction instead of divergent branch.
            // Pattern source: YSU-engine sass_re/instant_ngp/volume_render.cu
            int is_fluid = (n_pool_idx != -1);
            int dst_local = lxn + 8 * (lyn + 8 * lzn);
            read_dir = is_fluid ? OPP[i] : i;
            src_tid  = is_fluid ? (n_pool_idx * 512 + dst_local) : tid;
        }
        
        float fi = __ldg(&f[read_dir * n_active_cells_total + src_tid]);
        if (!finite_check(fi)) fi = 0.0f;
        f_local[i] = fi;
        rho_local += fi;
        mx += CX[i] * fi;
        my += CY[i] * fi;
        mz += CZ[i] * fi;
    }

    // 2. Macroscopic quantities
    float ux = 0.0f, uy = 0.0f, uz = 0.0f;
    if (finite_check(rho_local) && rho_local > 1.0e-20f) {
        float inv_rho = 1.0f / rho_local;
        ux = mx * inv_rho;
        uy = my * inv_rho;
        uz = mz * inv_rho;
    } else {
        rho_local = 1.0f;
    }

    rho_out[tid] = rho_local;
    u_out[tid]                        = ux;
    u_out[n_active_cells_total + tid]       = uy;
    u_out[2 * n_active_cells_total + tid]   = uz;

    // 3. BGK collision
    float tau_local = __ldg(&tau[tid]);
    float inv_tau = 1.0f / tau_local;
    float u_sq = ux * ux + uy * uy + uz * uz;
    float base = fmaf(-1.5f, u_sq, 1.0f);

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        float eu = fmaf((float)CX[i], ux, fmaf((float)CY[i], uy, (float)CZ[i] * uz));
        float w_rho = W[i] * rho_local;
        float f_eq = w_rho * fmaf(fmaf(eu, 4.5f, 3.0f), eu, base);
        f_local[i] -= (f_local[i] - f_eq) * inv_tau;
    }

    // 4. Guo forcing
    float fx = __ldg(&force[tid]);
    float fy = __ldg(&force[n_active_cells_total + tid]);
    float fz = __ldg(&force[2 * n_active_cells_total + tid]);
    float force_mag_sq = fx * fx + fy * fy + fz * fz;

    if (force_mag_sq >= 1e-40f) {
        float prefactor = 1.0f - 0.5f * inv_tau;
        #pragma unroll
        for (int i = 0; i < 19; i++) {
            float eix = (float)CX[i], eiy = (float)CY[i], eiz = (float)CZ[i];
            float em_u_dot_f = (eix - ux) * fx + (eiy - uy) * fy + (eiz - uz) * fz;
            float ei_dot_u = eix * ux + eiy * uy + eiz * uz;
            float ei_dot_f = eix * fx + eiy * fy + eiz * fz;
            f_local[i] += prefactor * W[i] * (em_u_dot_f * 3.0f + ei_dot_u * ei_dot_f * 9.0f);
        }
    }

    // 5. Write (with A-A push logic and bounce-back)
    #pragma unroll
    for (int i = 0; i < 19; i++) {
        if (parity == 0) {
            // Even step: push to neighbor's opposite slot
            int xn = (x + CX[i] + nx) % nx;
            int yn = (y + CY[i] + ny) % ny;
            int zn = (z + CZ[i] + nz) % nz;
            
            int bxn = xn / 8; int lxn = xn % 8;
            int byn = yn / 8; int lyn = yn % 8;
            int bzn = zn / 8; int lzn = zn % 8;
            
            int n_brick_idx = bxn + bx_max * (byn + by_max * bzn);
            int n_pool_idx = indirect_table[n_brick_idx];
            
            // Predicated push: same SELP pattern as pull phase.
            int is_fluid = (n_pool_idx != -1);
            int dst_local = lxn + 8 * (lyn + 8 * lzn);
            int write_tid = is_fluid ? (n_pool_idx * 512 + dst_local) : tid;
            f[OPP[i] * n_active_cells_total + write_tid] = f_local[i];
        } else {
            // Odd step: write to canonical slot at self
            f[i * n_active_cells_total + tid] = f_local[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Sparse A-A Tiled Kernel
// ---------------------------------------------------------------------------
// One sparse brick per block, with one distribution direction staged at a time
// through a shared-memory 10^3 halo tile. The 8^3 brick core is loaded through
// vectorized float4 groups where the brick-local x-lanes are contiguous.
extern "C" __global__ void __launch_bounds__(512)
lbm_step_sparse_aa_tiled(
    float* __restrict__ f,                     // [19 * N_active_cells]
    float* __restrict__ rho_out,               // [N_active_cells]
    float* __restrict__ u_out,                 // [3 * N_active_cells]
    const float* __restrict__ tau,             // [N_active_cells]
    const float* __restrict__ force,           // [3 * N_active_cells]
    const int* __restrict__ indirect_table,    // [bx_max * by_max * bz_max]
    const unsigned int* __restrict__ active_brick_ids, // [N_active_bricks]
    int nx, int ny, int nz,
    int bx_max, int by_max, int bz_max,
    int active_cell_start,
    int active_cell_count,
    int n_active_cells_total,
    int parity
) {
    __shared__ float s_dir[SPARSE_HALO_VOL];

    int local_idx = threadIdx.x;
    int active_brick_start = active_cell_start / SPARSE_BRICK_CELLS;
    int active_brick_count = (active_cell_count + SPARSE_BRICK_CELLS - 1) / SPARSE_BRICK_CELLS;
    int pool_idx = active_brick_start + blockIdx.x;
    if (blockIdx.x >= active_brick_count) return;

    int tid = pool_idx * SPARSE_BRICK_CELLS + local_idx;
    if (tid >= active_cell_start + active_cell_count || tid >= n_active_cells_total) return;

    int global_brick_idx = active_brick_ids[pool_idx];
    int bx = global_brick_idx % bx_max;
    int by = (global_brick_idx / bx_max) % by_max;
    int bz = global_brick_idx / (bx_max * by_max);

    int lx = local_idx % SPARSE_BRICK_EDGE;
    int ly = (local_idx / SPARSE_BRICK_EDGE) % SPARSE_BRICK_EDGE;
    int lz = local_idx / (SPARSE_BRICK_EDGE * SPARSE_BRICK_EDGE);

    int x = bx * SPARSE_BRICK_EDGE + lx;
    int y = by * SPARSE_BRICK_EDGE + ly;
    int z = bz * SPARSE_BRICK_EDGE + lz;
    int sx = lx + 1;
    int sy = ly + 1;
    int sz = lz + 1;

    float rho_local = 0.0f;
    float mx = 0.0f, my = 0.0f, mz = 0.0f;
    float f_local[19];

    if (parity == 0) {
        #pragma unroll
        for (int i = 0; i < 19; i++) {
            float fi = __ldg(&f[i * n_active_cells_total + tid]);
            if (!finite_check(fi)) fi = 0.0f;
            f_local[i] = fi;
            rho_local += fi;
            mx += CX[i] * fi;
            my += CY[i] * fi;
            mz += CZ[i] * fi;
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 19; i++) {
            int stage_dir = OPP[i];

            if ((local_idx & 3) == 0) {
                int base_tid = tid;
                float4 fv = *reinterpret_cast<const float4*>(
                    &f[stage_dir * n_active_cells_total + base_tid]
                );
                int sbase = sparse_halo_offset(sx, sy, sz);
                s_dir[sbase] = finite_check(fv.x) ? fv.x : 0.0f;
                s_dir[sbase + 1] = finite_check(fv.y) ? fv.y : 0.0f;
                s_dir[sbase + 2] = finite_check(fv.z) ? fv.z : 0.0f;
                s_dir[sbase + 3] = finite_check(fv.w) ? fv.w : 0.0f;
            }

            for (int halo_idx = local_idx; halo_idx < SPARSE_HALO_VOL; halo_idx += SPARSE_BRICK_CELLS) {
                int hx = halo_idx % SPARSE_HALO_EDGE;
                int hy = (halo_idx / SPARSE_HALO_EDGE) % SPARSE_HALO_EDGE;
                int hz = halo_idx / (SPARSE_HALO_EDGE * SPARSE_HALO_EDGE);
                bool in_core = (hx > 0 && hx < 9) && (hy > 0 && hy < 9) && (hz > 0 && hz < 9);
                if (in_core) {
                    continue;
                }

                int gx = (bx * SPARSE_BRICK_EDGE + hx - 1 + nx) % nx;
                int gy = (by * SPARSE_BRICK_EDGE + hy - 1 + ny) % ny;
                int gz = (bz * SPARSE_BRICK_EDGE + hz - 1 + nz) % nz;

                int bxn = gx / SPARSE_BRICK_EDGE;
                int byn = gy / SPARSE_BRICK_EDGE;
                int bzn = gz / SPARSE_BRICK_EDGE;
                int lxn = gx % SPARSE_BRICK_EDGE;
                int lyn = gy % SPARSE_BRICK_EDGE;
                int lzn = gz % SPARSE_BRICK_EDGE;

                int n_brick_idx = bxn + bx_max * (byn + by_max * bzn);
                int n_pool_idx = indirect_table[n_brick_idx];
                float staged = 0.0f;
                if (n_pool_idx != -1) {
                    int src_tid = n_pool_idx * SPARSE_BRICK_CELLS
                        + lxn
                        + SPARSE_BRICK_EDGE * (lyn + SPARSE_BRICK_EDGE * lzn);
                    staged = __ldg(&f[stage_dir * n_active_cells_total + src_tid]);
                }
                s_dir[halo_idx] = finite_check(staged) ? staged : 0.0f;
            }
            __syncthreads();

            int xn = (x - CX[i] + nx) % nx;
            int yn = (y - CY[i] + ny) % ny;
            int zn = (z - CZ[i] + nz) % nz;

            int bxn = xn / SPARSE_BRICK_EDGE;
            int byn = yn / SPARSE_BRICK_EDGE;
            int bzn = zn / SPARSE_BRICK_EDGE;
            int lxn = xn % SPARSE_BRICK_EDGE;
            int lyn = yn % SPARSE_BRICK_EDGE;
            int lzn = zn % SPARSE_BRICK_EDGE;

            int n_brick_idx = bxn + bx_max * (byn + by_max * bzn);
            int n_pool_idx = indirect_table[n_brick_idx];
            float fi = 0.0f;
            if (n_pool_idx != -1) {
                fi = s_dir[sparse_halo_offset(sx - CX[i], sy - CY[i], sz - CZ[i])];
            } else {
                fi = __ldg(&f[i * n_active_cells_total + tid]);
            }
            if (!finite_check(fi)) fi = 0.0f;
            f_local[i] = fi;
            rho_local += fi;
            mx += CX[i] * fi;
            my += CY[i] * fi;
            mz += CZ[i] * fi;
            __syncthreads();
        }
    }

    float ux = 0.0f, uy = 0.0f, uz = 0.0f;
    if (finite_check(rho_local) && rho_local > 1.0e-20f) {
        float inv_rho = 1.0f / rho_local;
        ux = mx * inv_rho;
        uy = my * inv_rho;
        uz = mz * inv_rho;
    } else {
        rho_local = 1.0f;
    }

    rho_out[tid] = rho_local;
    u_out[tid] = ux;
    u_out[n_active_cells_total + tid] = uy;
    u_out[2 * n_active_cells_total + tid] = uz;

    float tau_local = __ldg(&tau[tid]);
    float inv_tau = 1.0f / tau_local;
    float u_sq = ux * ux + uy * uy + uz * uz;
    float base = fmaf(-1.5f, u_sq, 1.0f);

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        float eu = fmaf((float)CX[i], ux, fmaf((float)CY[i], uy, (float)CZ[i] * uz));
        float w_rho = W[i] * rho_local;
        float f_eq = w_rho * fmaf(fmaf(eu, 4.5f, 3.0f), eu, base);
        f_local[i] -= (f_local[i] - f_eq) * inv_tau;
    }

    float fx = __ldg(&force[tid]);
    float fy = __ldg(&force[n_active_cells_total + tid]);
    float fz = __ldg(&force[2 * n_active_cells_total + tid]);
    float force_mag_sq = fx * fx + fy * fy + fz * fz;
    if (force_mag_sq >= 1e-40f) {
        float prefactor = 1.0f - 0.5f * inv_tau;
        #pragma unroll
        for (int i = 0; i < 19; i++) {
            float eix = (float)CX[i], eiy = (float)CY[i], eiz = (float)CZ[i];
            float em_u_dot_f = (eix - ux) * fx + (eiy - uy) * fy + (eiz - uz) * fz;
            float ei_dot_u = eix * ux + eiy * uy + eiz * uz;
            float ei_dot_f = eix * fx + eiy * fy + eiz * fz;
            f_local[i] += prefactor * W[i] * (em_u_dot_f * 3.0f + ei_dot_u * ei_dot_f * 9.0f);
        }
    }

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        if (parity == 0) {
            int xn = (x + CX[i] + nx) % nx;
            int yn = (y + CY[i] + ny) % ny;
            int zn = (z + CZ[i] + nz) % nz;

            int bxn = xn / SPARSE_BRICK_EDGE; int lxn = xn % SPARSE_BRICK_EDGE;
            int byn = yn / SPARSE_BRICK_EDGE; int lyn = yn % SPARSE_BRICK_EDGE;
            int bzn = zn / SPARSE_BRICK_EDGE; int lzn = zn % SPARSE_BRICK_EDGE;

            int n_brick_idx = bxn + bx_max * (byn + by_max * bzn);
            int n_pool_idx = indirect_table[n_brick_idx];

            if (n_pool_idx != -1) {
                int dst_tid = n_pool_idx * SPARSE_BRICK_CELLS + lxn + SPARSE_BRICK_EDGE * (lyn + SPARSE_BRICK_EDGE * lzn);
                f[OPP[i] * n_active_cells_total + dst_tid] = f_local[i];
            } else {
                f[OPP[i] * n_active_cells_total + tid] = f_local[i];
            }
        } else if ((local_idx & 3) == 0) {
            unsigned int lane = (unsigned int)(threadIdx.x & 31);
            unsigned int leader = lane & ~3u;
            float lane_value = f_local[i];
            float4 packed = {
                lane_value,
                __shfl_sync(0xffffffffu, lane_value, leader + 1),
                __shfl_sync(0xffffffffu, lane_value, leader + 2),
                __shfl_sync(0xffffffffu, lane_value, leader + 3),
            };
            *reinterpret_cast<float4*>(&f[i * n_active_cells_total + tid]) = packed;
        }
    }
}

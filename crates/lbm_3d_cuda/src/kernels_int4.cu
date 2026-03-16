// INT4 nibble-packed D3Q19 LBM kernel -- bandwidth ceiling benchmark.
//
// IMPORTANT: INT4 is NOT physically viable for standard D3Q19 LBM.
//   With 4-bit signed storage (-8..+7) and DIST_SCALE_I4 = 14:
//   - Face weight: 1/18 * 14 = 0.78 -> quantized to 1. OK.
//   - Edge weight: 1/36 * 14 = 0.39 -> quantized to 0. ZERO! Physics corrupted.
//   The edge-velocity populations (directions 7-18) collapse to zero, breaking
//   mass conservation and producing unphysical results.
//   USE CASE: bandwidth ceiling test only. INT4 measures the maximum MLUPS
//   achievable when distributions are 2x denser than INT8 in memory. This
//   establishes the upper bound for any future sub-byte precision tier.
//   Physics output is intentionally ignored; only MLUPS and BW% are reported.
//
// Storage: 2 nibbles per byte, i-major SoA layout.
//   f[i * n_cells + (idx/2)], nibble position = idx % 2.
//   This is equivalent to packing f[i*n_cells+0] and f[i*n_cells+1] into one byte.
//   Buffer size: 19 * ceil(n_cells/2) * 1 bytes per buffer.
//   VRAM at 128^3: 19 * 1,048,576 * 1 * 2 (ping+pong) = ~38 MB.
//
// AoS alternative (stride-10 bytes per cell, 20 nibbles including 1 pad):
//   stride 10 lacks 4-byte alignment for vectorized loads.
//   stride 12 (pad to 3 int32s) is 4-byte aligned but wastes 4 nibbles per cell.
//   We use i-major SoA to avoid the stride problem entirely:
//   lane i has ceil(n_cells/2) bytes, each contiguous and 4-byte alignable
//   when n_cells is a multiple of 8 (always true for power-of-2 grids).
//
// Load: for each direction i, load 1 byte, extract nibble for idx from byte
//   at position i*ceil_n_cells + idx/2. Nibble hi or lo based on idx&1.
// Store: read-modify-write a byte. Non-atomic RMW causes a race condition for
//   adjacent cells! To avoid this, we use i-major SoA and pack (idx, idx+1)
//   into the same byte -- but threads idx and idx+1 both write to the same byte.
//
// RACE CONDITION MITIGATION: Use stride 1 per full byte (one thread per byte).
//   That is: each thread handles TWO cells (idx and idx+1 in the same byte).
//   This halves the thread count and avoids atomic RMW.
//   Thread k handles cells 2k and 2k+1.
//   Grid: ceil(n_cells / 2) threads, 128 threads/block.
//
// DIST_SCALE_I4 = 14: 1/3*14=4.67->5 (rest weight OK), 1/18*14=0.78->1 (face
//   weight barely OK), 1/36*14=0.39->0 (edge weight ZERO, physics broken).

#define DIST_SCALE_I4 14.0f
#define INV_DIST_SCALE_I4 (1.0f / DIST_SCALE_I4)

// D3Q19 lattice velocities (suffixed _I4)
__constant__ int CX_I4[19] = {
    0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0
};
__constant__ int CY_I4[19] = {
    0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1
};
__constant__ int CZ_I4[19] = {
    0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1
};

__constant__ float W_I4[19] = {
    1.0f/3.0f,
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
};

__device__ __forceinline__ bool finite_i4(float x) {
    return (x == x) && (x <= 3.402823466e38f) && (x >= -3.402823466e38f);
}

// Extract 4-bit signed nibble from a packed byte.
// pos=0 -> lower nibble, pos=1 -> upper nibble.
__device__ __forceinline__ int nibble_extract(unsigned char byte, int pos) {
    int raw = (pos == 0) ? (int)(byte & 0xF) : (int)((byte >> 4) & 0xF);
    // Sign-extend from 4 bits: 0-7 -> 0..7, 8-15 -> -8..-1.
    return (raw >= 8) ? (raw - 16) : raw;
}

// Pack two 4-bit signed nibbles into one byte.
// lo_val and hi_val are clamped to [-8, 7].
__device__ __forceinline__ unsigned char nibble_pack(int lo_val, int hi_val) {
    // Clamp to [-8, 7] for each
    if (lo_val >  7) lo_val =  7;
    if (lo_val < -8) lo_val = -8;
    if (hi_val >  7) hi_val =  7;
    if (hi_val < -8) hi_val = -8;
    unsigned char lo_bits = (unsigned char)(lo_val & 0xF);
    unsigned char hi_bits = (unsigned char)(hi_val & 0xF);
    return lo_bits | (hi_bits << 4);
}

__device__ __forceinline__ int float_to_i4_raw(float v) {
    if (!finite_i4(v)) return 0;
    int s = (int)(v * DIST_SCALE_I4 + 0.5f);
    if (s >  7) s =  7;
    if (s < -8) s = -8;
    return s;
}

// INT4 i-major SoA fused collision + pull-streaming kernel.
// Each thread handles TWO cells: idx_lo = 2*k, idx_hi = 2*k+1.
// This avoids RMW races when storing nibbles into shared bytes.
// n_cells MUST be even (guaranteed for power-of-2 grid sizes).
extern "C" __global__ void lbm_step_fused_int4_kernel(
    const unsigned char* __restrict__ f_in,   // [19 * (n_cells/2)] packed nibble pairs
    unsigned char* __restrict__ f_out,        // [19 * (n_cells/2)] packed nibble pairs
    float* __restrict__ rho_out,
    float* __restrict__ u_out,                // [3 * n_cells] SoA FP32
    const float* __restrict__ tau,
    const float* __restrict__ force,          // [3 * n_cells] SoA FP32
    int nx, int ny, int nz
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int n_cells = nx * ny * nz;
    int half_cells = n_cells / 2;   // number of bytes per direction
    if (k >= half_cells) return;

    // Two cells handled per thread
    int idx_lo = 2 * k;
    int idx_hi = 2 * k + 1;

    // Cell coordinates for both cells
    int x_lo = idx_lo % nx, y_lo = (idx_lo / nx) % ny, z_lo = idx_lo / (nx * ny);
    int x_hi = idx_hi % nx, y_hi = (idx_hi / nx) % ny, z_hi = idx_hi / (nx * ny);

    float f_lo[19], f_hi[19];
    float rho_lo = 0.0f, rho_hi = 0.0f;
    float mx_lo = 0.0f, my_lo = 0.0f, mz_lo = 0.0f;
    float mx_hi = 0.0f, my_hi = 0.0f, mz_hi = 0.0f;

    // Pull reads: for each direction i, read two backward-neighbor nibbles.
    #pragma unroll
    for (int i = 0; i < 19; i++) {
        // Backward neighbor for idx_lo
        int xbl = (x_lo - CX_I4[i] + nx) % nx;
        int ybl = (y_lo - CY_I4[i] + ny) % ny;
        int zbl = (z_lo - CZ_I4[i] + nz) % nz;
        int back_lo = xbl + nx * (ybl + ny * zbl);

        // Backward neighbor for idx_hi
        int xbh = (x_hi - CX_I4[i] + nx) % nx;
        int ybh = (y_hi - CY_I4[i] + ny) % ny;
        int zbh = (z_hi - CZ_I4[i] + nz) % nz;
        int back_hi = xbh + nx * (ybh + ny * zbh);

        // Load bytes containing each backward neighbor's nibble
        int byte_lo = back_lo / 2;
        int pos_lo  = back_lo & 1;
        int byte_hi = back_hi / 2;
        int pos_hi  = back_hi & 1;

        unsigned char blo = __ldg(&f_in[(long long)i * half_cells + byte_lo]);
        unsigned char bhi = __ldg(&f_in[(long long)i * half_cells + byte_hi]);

        float flo = (float)nibble_extract(blo, pos_lo) * INV_DIST_SCALE_I4;
        float fhi = (float)nibble_extract(bhi, pos_hi) * INV_DIST_SCALE_I4;

        f_lo[i] = flo;  f_hi[i] = fhi;
        rho_lo += flo;  rho_hi += fhi;
        mx_lo += (float)CX_I4[i] * flo;  mx_hi += (float)CX_I4[i] * fhi;
        my_lo += (float)CY_I4[i] * flo;  my_hi += (float)CY_I4[i] * fhi;
        mz_lo += (float)CZ_I4[i] * flo;  mz_hi += (float)CZ_I4[i] * fhi;
    }

    // Macroscopic for both cells
    float ux_lo = 0.0f, uy_lo = 0.0f, uz_lo = 0.0f;
    float ux_hi = 0.0f, uy_hi = 0.0f, uz_hi = 0.0f;
    if (finite_i4(rho_lo) && rho_lo > 1.0e-20f) {
        float ir = 1.0f / rho_lo;
        ux_lo = mx_lo * ir; uy_lo = my_lo * ir; uz_lo = mz_lo * ir;
    } else { rho_lo = 1.0f; }
    if (finite_i4(rho_hi) && rho_hi > 1.0e-20f) {
        float ir = 1.0f / rho_hi;
        ux_hi = mx_hi * ir; uy_hi = my_hi * ir; uz_hi = mz_hi * ir;
    } else { rho_hi = 1.0f; }

    rho_out[idx_lo] = rho_lo;
    rho_out[idx_hi] = rho_hi;
    u_out[idx_lo]               = ux_lo; u_out[n_cells + idx_lo]     = uy_lo; u_out[2*n_cells + idx_lo] = uz_lo;
    u_out[idx_hi]               = ux_hi; u_out[n_cells + idx_hi]     = uy_hi; u_out[2*n_cells + idx_hi] = uz_hi;

    // BGK collision for both cells (FP32)
    float tau_lo = __ldg(&tau[idx_lo]);
    float tau_hi = __ldg(&tau[idx_hi]);
    float inv_tau_lo = 1.0f / tau_lo;
    float inv_tau_hi = 1.0f / tau_hi;

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        float eu_lo = fmaf((float)CX_I4[i], ux_lo, fmaf((float)CY_I4[i], uy_lo, (float)CZ_I4[i] * uz_lo));
        float u_sq_lo = ux_lo*ux_lo + uy_lo*uy_lo + uz_lo*uz_lo;
        float f_eq_lo = W_I4[i] * rho_lo * fmaf(fmaf(eu_lo, 4.5f, 3.0f), eu_lo, fmaf(-1.5f, u_sq_lo, 1.0f));
        f_lo[i] -= (f_lo[i] - f_eq_lo) * inv_tau_lo;

        float eu_hi = fmaf((float)CX_I4[i], ux_hi, fmaf((float)CY_I4[i], uy_hi, (float)CZ_I4[i] * uz_hi));
        float u_sq_hi = ux_hi*ux_hi + uy_hi*uy_hi + uz_hi*uz_hi;
        float f_eq_hi = W_I4[i] * rho_hi * fmaf(fmaf(eu_hi, 4.5f, 3.0f), eu_hi, fmaf(-1.5f, u_sq_hi, 1.0f));
        f_hi[i] -= (f_hi[i] - f_eq_hi) * inv_tau_hi;
    }

    // Pack both cells' post-collision distributions into nibble-pairs and write.
    // Coalesced: thread k writes byte k for each direction i.
    #pragma unroll
    for (int i = 0; i < 19; i++) {
        int lo_raw = float_to_i4_raw(f_lo[i]);
        int hi_raw = float_to_i4_raw(f_hi[i]);
        f_out[(long long)i * half_cells + k] = nibble_pack(lo_raw, hi_raw);
    }
}

extern "C" __global__ void initialize_uniform_int4_kernel(
    unsigned char* f,
    float* rho_out,
    float* u_out,
    float* tau,
    float rho_init,
    float ux_init,
    float uy_init,
    float uz_init,
    float tau_val,
    int nx, int ny, int nz
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int n_cells = nx * ny * nz;
    int half_cells = n_cells / 2;
    if (k >= half_cells) return;

    int idx_lo = 2 * k;
    int idx_hi = 2 * k + 1;

    rho_out[idx_lo] = rho_init; rho_out[idx_hi] = rho_init;
    u_out[idx_lo] = ux_init; u_out[n_cells + idx_lo] = uy_init; u_out[2*n_cells + idx_lo] = uz_init;
    u_out[idx_hi] = ux_init; u_out[n_cells + idx_hi] = uy_init; u_out[2*n_cells + idx_hi] = uz_init;
    tau[idx_lo] = tau_val; tau[idx_hi] = tau_val;

    float u_sq = ux_init*ux_init + uy_init*uy_init + uz_init*uz_init;
    float base = fmaf(-1.5f, u_sq, 1.0f);

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        float eu = (float)CX_I4[i]*ux_init + (float)CY_I4[i]*uy_init + (float)CZ_I4[i]*uz_init;
        float f_eq = W_I4[i] * rho_init * fmaf(fmaf(eu, 4.5f, 3.0f), eu, base);
        int lo_raw = float_to_i4_raw(f_eq);
        int hi_raw = float_to_i4_raw(f_eq);
        f[(long long)i * half_cells + k] = nibble_pack(lo_raw, hi_raw);
    }
}

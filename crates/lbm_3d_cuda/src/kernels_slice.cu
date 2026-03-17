// On-the-fly 2D slice extraction from INT8 SoA distribution buffer.
//
// Reconstructs macroscopic rho and |u| for a single 2D slice without
// materializing the full 3D macroscopic field. This enables live viewing
// of 512^3 INT8 A-A simulations within the 12 GB VRAM budget.
//
// VRAM cost: 2 * slice_w * slice_h * sizeof(float) = ~2 MB at 512^2.
// Compare: full macroscopic at 512^3 = ~4 GB.
//
// The kernel reads 19 INT8 distributions per cell, dequantizes to f32,
// computes density (sum) and velocity magnitude (dot product), and writes
// to a compact 2D output buffer.

// D3Q19 lattice velocities (same as other kernels)
__constant__ int SLICE_CX[19] = {
    0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0
};
__constant__ int SLICE_CY[19] = {
    0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1
};
__constant__ int SLICE_CZ[19] = {
    0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1
};

// INT8 dequantization scale (DIST_SCALE = 64)
#define SLICE_INV_SCALE (1.0f / 64.0f)

// Extract a 2D slice of rho and velocity magnitude from INT8 SoA distributions.
//
// slice_axis: 0 = X-plane (output YZ), 1 = Y-plane (output XZ), 2 = Z-plane (output XY)
// slice_idx: index along the sliced axis
//
// Output: rho_out[sw * sh], vel_mag_out[sw * sh] where sw/sh are the slice dims.
extern "C" __global__ void read_slice_int8_soa(
    const signed char* __restrict__ f,  // [19 * N] INT8 SoA distribution buffer
    float* __restrict__ rho_out,        // [sw * sh] output density
    float* __restrict__ vel_mag_out,    // [sw * sh] output velocity magnitude
    int nx, int ny, int nz,
    int slice_axis,                     // 0=X, 1=Y, 2=Z
    int slice_idx                       // index along sliced axis
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nx * ny * nz;

    // Determine slice dimensions
    int sw, sh;
    if (slice_axis == 0) { sw = ny; sh = nz; }
    else if (slice_axis == 1) { sw = nx; sh = nz; }
    else { sw = nx; sh = ny; }

    int slice_size = sw * sh;
    if (tid >= slice_size) return;

    // Map 2D slice coord to 3D grid index
    int s_col = tid % sw;  // column in slice
    int s_row = tid / sw;  // row in slice
    int ix, iy, iz;
    if (slice_axis == 0) {
        ix = slice_idx; iy = s_col; iz = s_row;
    } else if (slice_axis == 1) {
        ix = s_col; iy = slice_idx; iz = s_row;
    } else {
        ix = s_col; iy = s_row; iz = slice_idx;
    }

    // Clamp to grid bounds
    if (ix >= nx) ix = nx - 1;
    if (iy >= ny) iy = ny - 1;
    if (iz >= nz) iz = nz - 1;

    int idx = iz * ny * nx + iy * nx + ix;

    // Read 19 INT8 distributions and compute macroscopic quantities
    float rho = 0.0f;
    float mx = 0.0f, my = 0.0f, mz = 0.0f;

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        signed char v = f[(long long)i * N + idx];
        float fv = (float)v * SLICE_INV_SCALE;
        rho += fv;
        mx += (float)SLICE_CX[i] * fv;
        my += (float)SLICE_CY[i] * fv;
        mz += (float)SLICE_CZ[i] * fv;
    }

    float ux = 0.0f, uy = 0.0f, uz = 0.0f;
    if (rho > 1e-20f) {
        float inv_rho = 1.0f / rho;
        ux = mx * inv_rho;
        uy = my * inv_rho;
        uz = mz * inv_rho;
    }

    rho_out[tid] = rho;
    vel_mag_out[tid] = sqrtf(ux * ux + uy * uy + uz * uz);
}

// ---------------------------------------------------------------------------
// INT8 velocity extraction for OptiX: reconstruct FP16 velocity from INT8
// distributions for ALL cells in active bricks. Output as half4 (u16x4).
// ---------------------------------------------------------------------------
// This enables OptiX particle tracing at 512^3 INT8 within the VRAM budget:
// only occupied-brick cells get velocity computed and written.
//
// Launch with: one thread per cell in the active brick list.
// The caller provides a flat list of cell indices (from the brick map).
extern "C" __global__ void extract_velocity_int8_to_fp16(
    const signed char* __restrict__ f,  // [19 * N] INT8 SoA distributions
    unsigned short* __restrict__ u_fp16, // [4 * n_active] output FP16 velocity (half4)
    const int* __restrict__ active_cells, // [n_active] indices of active cells
    int n_active,
    int N                               // total cells (nx * ny * nz)
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_active) return;

    int idx = active_cells[tid];
    if (idx < 0 || idx >= N) return;

    // Reconstruct velocity from 19 INT8 distributions
    float rho = 0.0f;
    float mx = 0.0f, my = 0.0f, mz = 0.0f;
    #pragma unroll
    for (int i = 0; i < 19; i++) {
        float fv = (float)f[(long long)i * N + idx] * SLICE_INV_SCALE;
        rho += fv;
        mx += (float)SLICE_CX[i] * fv;
        my += (float)SLICE_CY[i] * fv;
        mz += (float)SLICE_CZ[i] * fv;
    }

    float ux = 0.0f, uy = 0.0f, uz = 0.0f;
    if (rho > 1e-20f) {
        float inv_rho = 1.0f / rho;
        ux = mx * inv_rho;
        uy = my * inv_rho;
        uz = mz * inv_rho;
    }

    // FP32-to-FP16 inline device function (NVRTC-compatible, no statement expressions)
    auto f2h = [](float v) -> unsigned short {
        unsigned int b = __float_as_uint(v);
        unsigned int s = (b >> 16) & 0x8000u;
        int e = (int)((b >> 23) & 0xFF) - 127 + 15;
        unsigned int m = (b >> 13) & 0x3FFu;
        if (e <= 0) return (unsigned short)s;
        if (e >= 31) return (unsigned short)(s | 0x7C00u);
        return (unsigned short)(s | (e << 10) | m);
    };

    u_fp16[tid * 4 + 0] = f2h(ux);
    u_fp16[tid * 4 + 1] = f2h(uy);
    u_fp16[tid * 4 + 2] = f2h(uz);
    u_fp16[tid * 4 + 3] = 0;
}

// Same kernel for FP32 SoA distributions (for the standard solver path).
extern "C" __global__ void read_slice_fp32_soa(
    const float* __restrict__ f,        // [19 * N] FP32 SoA distribution buffer
    float* __restrict__ rho_out,
    float* __restrict__ vel_mag_out,
    int nx, int ny, int nz,
    int slice_axis,
    int slice_idx
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nx * ny * nz;

    int sw, sh;
    if (slice_axis == 0) { sw = ny; sh = nz; }
    else if (slice_axis == 1) { sw = nx; sh = nz; }
    else { sw = nx; sh = ny; }

    int slice_size = sw * sh;
    if (tid >= slice_size) return;

    int s_col = tid % sw;
    int s_row = tid / sw;
    int ix, iy, iz;
    if (slice_axis == 0) { ix = slice_idx; iy = s_col; iz = s_row; }
    else if (slice_axis == 1) { ix = s_col; iy = slice_idx; iz = s_row; }
    else { ix = s_col; iy = s_row; iz = slice_idx; }

    if (ix >= nx) ix = nx - 1;
    if (iy >= ny) iy = ny - 1;
    if (iz >= nz) iz = nz - 1;

    int idx = iz * ny * nx + iy * nx + ix;

    float rho = 0.0f;
    float mx = 0.0f, my = 0.0f, mz = 0.0f;

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        float fv = f[(long long)i * N + idx];
        rho += fv;
        mx += (float)SLICE_CX[i] * fv;
        my += (float)SLICE_CY[i] * fv;
        mz += (float)SLICE_CZ[i] * fv;
    }

    float ux = 0.0f, uy = 0.0f, uz = 0.0f;
    if (rho > 1e-20f) {
        float inv_rho = 1.0f / rho;
        ux = mx * inv_rho;
        uy = my * inv_rho;
        uz = mz * inv_rho;
    }

    rho_out[tid] = rho;
    vel_mag_out[tid] = sqrtf(ux * ux + uy * uy + uz * uz);
}

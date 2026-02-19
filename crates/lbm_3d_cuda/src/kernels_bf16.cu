#include <cuda_bf16.h>

// D3Q19 lattice velocities (constant memory)
__constant__ int D3Q19_CX[19] = {
    0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0
};
__constant__ int D3Q19_CY[19] = {
    0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1
};
__constant__ int D3Q19_CZ[19] = {
    0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1
};

// D3Q19 weights (float)
__constant__ float D3Q19_WF[19] = {
    1.0f/3.0f,                            // i=0 (rest)
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,      // i=1-6 (face neighbors)
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,      // i=7-18 (edge neighbors)
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
};

__device__ const float CS_SQ_F = 1.0f / 3.0f;

__device__ __forceinline__ bool finite_f32(float x) {
    return (x == x) && (x <= 3.402823466e38f) && (x >= -3.402823466e38f);
}

// Compute equilibrium distribution (math in float)
__device__ void compute_equilibrium_bf16(
    float* f_eq,
    float rho,
    const float* u
) {
    float u_sq = u[0]*u[0] + u[1]*u[1] + u[2]*u[2];

    for (int i = 0; i < 19; i++) {
        float c_dot_u = D3Q19_CX[i]*u[0] + D3Q19_CY[i]*u[1] + D3Q19_CZ[i]*u[2];
        float c_dot_u_sq = c_dot_u * c_dot_u;

        f_eq[i] = D3Q19_WF[i] * rho * (
            1.0f +
            c_dot_u / CS_SQ_F +
            c_dot_u_sq / (2.0f * CS_SQ_F * CS_SQ_F) -
            u_sq / (2.0f * CS_SQ_F)
        );
    }
}

// Kernel: Fused LBM Step using BFloat16 for Storage, FP32 for Compute
extern "C" __global__ void lbm_step_fused_bf16_kernel(
    const __nv_bfloat16* f_in,      // Input distributions
    __nv_bfloat16* f_out,           // Output distributions
    __nv_bfloat16* rho_out,         // Density output
    __nv_bfloat16* u_out,           // Velocity output
    const __nv_bfloat16* force,     // Force field
    const __nv_bfloat16* tau,       // Relaxation time
    int nx, int ny, int nz
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    int idx = x + nx * (y + ny * z);

    // 1. Load macroscopic (convert BF16 -> FP32)
    float rho_local = 0.0f;
    float mx = 0.0f, my = 0.0f, mz = 0.0f;
    float f_local[19];

    for (int i = 0; i < 19; i++) {
        float val = __bfloat162float(f_in[idx * 19 + i]);
        if (!finite_f32(val)) {
            val = 0.0f;
        }
        f_local[i] = val;
        rho_local += val;
        mx += D3Q19_CX[i] * val;
        my += D3Q19_CY[i] * val;
        mz += D3Q19_CZ[i] * val;
    }

    float ux = 0.0f;
    float uy = 0.0f;
    float uz = 0.0f;
    if (finite_f32(rho_local) && rho_local > 1.0e-20f) {
        float inv_rho = 1.0f / rho_local;
        ux = mx * inv_rho;
        uy = my * inv_rho;
        uz = mz * inv_rho;
    } else {
        rho_local = 1.0f;
    }

    rho_out[idx] = __float2bfloat16(rho_local);
    u_out[idx * 3 + 0] = __float2bfloat16(ux);
    u_out[idx * 3 + 1] = __float2bfloat16(uy);
    u_out[idx * 3 + 2] = __float2bfloat16(uz);

    // 2. Collision (math in FP32)
    float f_eq[19];
    float u_vec[3] = {ux, uy, uz};
    compute_equilibrium_bf16(f_eq, rho_local, u_vec);

    float tau_local = __bfloat162float(tau[idx]);
    float inv_tau = 1.0f / tau_local;
    float prefactor = 1.0f - 0.5f * inv_tau;

    float fx = __bfloat162float(force[idx * 3 + 0]);
    float fy = __bfloat162float(force[idx * 3 + 1]);
    float fz = __bfloat162float(force[idx * 3 + 2]);

    for (int i = 0; i < 19; i++) {
        // BGK + Forcing
        float fi = f_local[i] - (f_local[i] - f_eq[i]) * inv_tau;
        float eix = (float)D3Q19_CX[i]; float eiy = (float)D3Q19_CY[i]; float eiz = (float)D3Q19_CZ[i];
        float s_i = ( (eix - ux) * fx + (eiy - uy) * fy + (eiz - uz) * fz ) * 3.0f 
                  + ( eix * ux + eiy * uy + eiz * uz ) * ( eix * fx + eiy * fy + eiz * fz ) * 9.0f;
        
        fi += prefactor * D3Q19_WF[i] * s_i;

        // 3. Streaming (convert FP32 -> BF16 and store)
        int x_next = (x + D3Q19_CX[i] + nx) % nx;
        int y_next = (y + D3Q19_CY[i] + ny) % ny;
        int z_next = (z + D3Q19_CZ[i] + nz) % nz;
        int idx_next = x_next + nx * (y_next + ny * z_next);
        
        f_out[idx_next * 19 + i] = __float2bfloat16(fi);
    }
}

// 4D Kernel: Treats 4th dim (w) as independent 3D worlds.
// Allows massive batch simulation of 32^4 or 64^4 grids in one launch.
// Indexing: idx = x + nx*(y + ny*(z + nz*w))
// Streaming: periodic in x, y, z; isolated in w.
extern "C" __global__ void lbm_step_fused_bf16_4d_batch_kernel(
    const __nv_bfloat16* f_in,      // Input distributions
    __nv_bfloat16* f_out,           // Output distributions
    __nv_bfloat16* rho_out,         // Density output
    __nv_bfloat16* u_out,           // Velocity output
    const __nv_bfloat16* force,     // Force field
    const __nv_bfloat16* tau,       // Relaxation time
    int nx, int ny, int nz, int nw
) {
    // 4D grid mapping strategy:
    // We map CUDA 3D grid (gx, gy, gz) to 4D problem space.
    // Standard mapping: x=thread.x, y=thread.y, zw=thread.z.
    // zw = z + nz*w.
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int zw_global = blockIdx.z * blockDim.z + threadIdx.z;
    
    int z = zw_global % nz;
    int w = zw_global / nz;

    if (x >= nx || y >= ny || w >= nw) return;

    // Linear index in 4D array
    int vol_3d = nx * ny * nz;
    int idx_3d = x + nx * (y + ny * z);
    long long idx_4d = (long long)w * vol_3d + idx_3d;

    // 1. Load macroscopic (convert BF16 -> FP32)
    float rho_local = 0.0f;
    float mx = 0.0f, my = 0.0f, mz = 0.0f;
    float f_local[19];

    for (int i = 0; i < 19; i++) {
        float val = __bfloat162float(f_in[idx_4d * 19 + i]);
        if (!finite_f32(val)) val = 0.0f;
        f_local[i] = val;
        rho_local += val;
        mx += D3Q19_CX[i] * val;
        my += D3Q19_CY[i] * val;
        mz += D3Q19_CZ[i] * val;
    }

    float ux = 0.0f, uy = 0.0f, uz = 0.0f;
    if (finite_f32(rho_local) && rho_local > 1.0e-20f) {
        float inv_rho = 1.0f / rho_local;
        ux = mx * inv_rho;
        uy = my * inv_rho;
        uz = mz * inv_rho;
    } else {
        rho_local = 1.0f;
    }

    rho_out[idx_4d] = __float2bfloat16(rho_local);
    u_out[idx_4d * 3 + 0] = __float2bfloat16(ux);
    u_out[idx_4d * 3 + 1] = __float2bfloat16(uy);
    u_out[idx_4d * 3 + 2] = __float2bfloat16(uz);

    // 2. Collision
    float f_eq[19];
    float u_vec[3] = {ux, uy, uz};
    compute_equilibrium_bf16(f_eq, rho_local, u_vec);

    float tau_local = __bfloat162float(tau[idx_4d]);
    float inv_tau = 1.0f / tau_local;
    float prefactor = 1.0f - 0.5f * inv_tau;

    float fx = __bfloat162float(force[idx_4d * 3 + 0]);
    float fy = __bfloat162float(force[idx_4d * 3 + 1]);
    float fz = __bfloat162float(force[idx_4d * 3 + 2]);

    for (int i = 0; i < 19; i++) {
        float fi = f_local[i] - (f_local[i] - f_eq[i]) * inv_tau;
        float eix = (float)D3Q19_CX[i]; float eiy = (float)D3Q19_CY[i]; float eiz = (float)D3Q19_CZ[i];
        float s_i = ( (eix - ux) * fx + (eiy - uy) * fy + (eiz - uz) * fz ) * 3.0f 
                  + ( eix * ux + eiy * uy + eiz * uz ) * ( eix * fx + eiy * fy + eiz * fz ) * 9.0f;
        fi += prefactor * D3Q19_WF[i] * s_i;

        // 3. Streaming (3D periodic within w-slice)
        int x_next = (x + D3Q19_CX[i] + nx) % nx;
        int y_next = (y + D3Q19_CY[i] + ny) % ny;
        int z_next = (z + D3Q19_CZ[i] + nz) % nz;
        
        long long idx_next_4d = (long long)w * vol_3d + (x_next + nx * (y_next + ny * z_next));
        
        f_out[idx_next_4d * 19 + i] = __float2bfloat16(fi);
    }
}

extern "C" __global__ void initialize_uniform_bf16_kernel(
    __nv_bfloat16* f,
    __nv_bfloat16* rho,
    __nv_bfloat16* u,
    float rho_init,
    float ux_init,
    float uy_init,
    float uz_init,
    int nx,
    int ny,
    int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_cells = nx * ny * nz;
    if (idx >= n_cells) return;

    rho[idx] = __float2bfloat16(rho_init);
    u[idx * 3 + 0] = __float2bfloat16(ux_init);
    u[idx * 3 + 1] = __float2bfloat16(uy_init);
    u[idx * 3 + 2] = __float2bfloat16(uz_init);

    float u_local[3] = {ux_init, uy_init, uz_init};
    float f_eq[19];
    compute_equilibrium_bf16(f_eq, rho_init, u_local);
    for (int i = 0; i < 19; i++) {
        f[idx * 19 + i] = __float2bfloat16(f_eq[i]);
    }
}

extern "C" __global__ void initialize_custom_bf16_kernel(
    __nv_bfloat16* f,
    __nv_bfloat16* rho,
    __nv_bfloat16* u,
    const __nv_bfloat16* rho_in,
    const __nv_bfloat16* u_in,
    int nx,
    int ny,
    int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_cells = nx * ny * nz;
    if (idx >= n_cells) return;

    float rho_init = __bfloat162float(rho_in[idx]);
    float ux_init = __bfloat162float(u_in[idx * 3 + 0]);
    float uy_init = __bfloat162float(u_in[idx * 3 + 1]);
    float uz_init = __bfloat162float(u_in[idx * 3 + 2]);

    rho[idx] = __float2bfloat16(rho_init);
    u[idx * 3 + 0] = __float2bfloat16(ux_init);
    u[idx * 3 + 1] = __float2bfloat16(uy_init);
    u[idx * 3 + 2] = __float2bfloat16(uz_init);

    float u_local[3] = {ux_init, uy_init, uz_init};
    float f_eq[19];
    compute_equilibrium_bf16(f_eq, rho_init, u_local);
    for (int i = 0; i < 19; i++) {
        f[idx * 19 + i] = __float2bfloat16(f_eq[i]);
    }
}

// Support Kernels
struct ComplexDeviceF { float re; float im; };

extern "C" __global__ void convert_real_bf16_to_complex_f32_kernel(const __nv_bfloat16* u, ComplexDeviceF* u_hat, int comp, int n_cells) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_cells) { u_hat[idx].re = __bfloat162float(u[idx * 3 + comp]); u_hat[idx].im = 0.0f; }
}

extern "C" __global__ void convert_complex_f32_to_real_bf16_kernel(const ComplexDeviceF* u_hat, __nv_bfloat16* u, int comp, float scale, int n_cells) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_cells) u[idx * 3 + comp] = __float2bfloat16(u_hat[idx].re * scale);
}

extern "C" __global__ void zero_f32_kernel(float* out) { *out = 0.0f; }

extern "C" __global__ void reduce_sum_bf16_to_f32_kernel(const __nv_bfloat16* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) sum += __bfloat162float(in[i]);
    atomicAdd(out, sum);
}

extern "C" __global__ void reduce_sum_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) sum += in[i];
    atomicAdd(out, sum);
}

extern "C" __global__ void apply_spectral_mask_kernel(ComplexDeviceF* u_hat, const float* mask, float damping, int n_cells) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_cells && mask[idx] < 0.5f) { u_hat[idx].re *= damping; u_hat[idx].im *= damping; }
}

extern "C" __global__ void compute_enstrophy_cell_kernel(const __nv_bfloat16* u, float* enstrophy_field, int nx, int ny, int nz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;
    int idx = x + nx * (y + ny * z);
    // Nearest neighbor indices with periodic boundary conditions
    int xp = (x + 1) % nx; int xm = (x + nx - 1) % nx;
    int yp = (y + 1) % ny; int ym = (y + ny - 1) % ny;
    int zp = (z + 1) % nz; int zm = (z + nz - 1) % nz;
    // Helper to get velocity component at a specific grid point
    auto get_u = [&](int xi, int yi, int zi, int comp) { return __bfloat162float(u[(xi + nx * (yi + ny * zi)) * 3 + comp]); };
    // Calculate velocity gradients using central differences
    float duz_dy = (get_u(x, yp, z, 2) - get_u(x, ym, z, 2)) * 0.5f;
    float duy_dz = (get_u(x, y, zp, 1) - get_u(x, y, zm, 1)) * 0.5f;
    float dux_dz = (get_u(x, y, zp, 0) - get_u(x, y, zm, 0)) * 0.5f;
    float duz_dx = (get_u(xp, y, z, 2) - get_u(xm, y, z, 2)) * 0.5f;
    float duy_dx = (get_u(xp, y, z, 1) - get_u(xm, y, z, 1)) * 0.5f;
    float dux_dy = (get_u(x, yp, z, 0) - get_u(x, ym, z, 0)) * 0.5f;
    // Calculate vorticity components
    float wx = duz_dy - duy_dz; float wy = dux_dz - duz_dx; float wz = duy_dx - dux_dy;
    // Enstrophy for this cell
    enstrophy_field[idx] = wx*wx + wy*wy + wz*wz;
}

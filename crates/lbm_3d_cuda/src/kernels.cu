// Kernels switched to float (f32) for massive throughput on Ada Lovelace
// Target: 60+ steps/s at 128^3

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

// Speed of sound squared (float)
__device__ const float CS_SQ_F = 1.0f / 3.0f;

__device__ __forceinline__ bool finite_f32(float x) {
    return (x == x) && (x <= 3.402823466e38f) && (x >= -3.402823466e38f);
}

// Compute equilibrium distribution (float) -- FMA-optimized Horner form.
// Algebraic identity: f_eq = w*rho * (4.5*eu^2 + 3*eu + 1 - 1.5*usq)
// Horner: (4.5*eu + 3)*eu + base, where base = 1 - 1.5*usq
// Two FMA ops via fmaf() instead of separate mul+add+div.
__device__ void compute_equilibrium_f(
    float* f_eq,
    float rho,
    const float* u
) {
    float u_sq = u[0]*u[0] + u[1]*u[1] + u[2]*u[2];
    float base = fmaf(-1.5f, u_sq, 1.0f);  // 1 - 1.5*usq

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        float eu = (float)(D3Q19_CX[i])*u[0]
                 + (float)(D3Q19_CY[i])*u[1]
                 + (float)(D3Q19_CZ[i])*u[2];
        float w_rho = D3Q19_WF[i] * rho;
        // Horner evaluation: (4.5*eu + 3)*eu + base
        f_eq[i] = w_rho * fmaf(fmaf(eu, 4.5f, 3.0f), eu, base);
    }
}

// Kernel 1: Compute macroscopic quantities (float)
extern "C" __global__ void compute_macroscopic_kernel(
    const float* f,    
    float* rho,        
    float* u,          
    int nx, int ny, int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_cells = nx * ny * nz;
    if (idx >= n_cells) return;

    // Batch-issue all 19 loads before accumulation (software pipelining).
    // Pattern source: YSU-engine sass_re/instant_ngp/hashgrid_encode.cu
    // The AoS layout (19 contiguous floats per cell) is not 16-byte aligned
    // (19*4=76 bytes), so we use scalar __ldg rather than float4 to avoid
    // alignment penalties. The batch-issue pattern still hides L2 latency.
    const float* base = &f[idx * 19];
    float f_local[19];
    #pragma unroll
    for (int i = 0; i < 19; i++) f_local[i] = __ldg(base + i);

    float rho_local = 0.0f;
    #pragma unroll
    for (int i = 0; i < 19; i++) rho_local += f_local[i];

    float ux = 0.0f, uy = 0.0f, uz = 0.0f;
    #pragma unroll
    for (int i = 0; i < 19; i++) {
        ux += D3Q19_CX[i] * f_local[i];
        uy += D3Q19_CY[i] * f_local[i];
        uz += D3Q19_CZ[i] * f_local[i];
    }

    if (!finite_f32(rho_local) || rho_local <= 1.0e-20f) {
        rho[idx] = 1.0f;
        u[idx * 3 + 0] = 0.0f;
        u[idx * 3 + 1] = 0.0f;
        u[idx * 3 + 2] = 0.0f;
        return;
    }

    float inv_rho = 1.0f / rho_local;
    rho[idx] = rho_local;
    u[idx * 3 + 0] = ux * inv_rho;
    u[idx * 3 + 1] = uy * inv_rho;
    u[idx * 3 + 2] = uz * inv_rho;
}

// Kernel 2: BGK collision (float)
extern "C" __global__ void bgk_collision_kernel(
    float* f,              
    const float* rho,      
    const float* u,        
    const float* tau,      
    int nx, int ny, int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_cells = nx * ny * nz;
    if (idx >= n_cells) return;

    float rho_local = rho[idx];
    float u_local[3] = {u[idx*3 + 0], u[idx*3 + 1], u[idx*3 + 2]};
    float tau_local = tau[idx];

    float f_eq[19];
    compute_equilibrium_f(f_eq, rho_local, u_local);

    float inv_tau = 1.0f / tau_local;
    #pragma unroll
    for (int i = 0; i < 19; i++) {
        int f_idx = idx * 19 + i;
        f[f_idx] -= (f[f_idx] - f_eq[i]) * inv_tau;
    }
}

// Kernel 3: Streaming (float)
extern "C" __global__ void streaming_kernel(
    const float* f_in,   
    float* f_out,        
    int nx, int ny, int nz
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    int idx = x + nx * (y + ny * z);
    for (int i = 0; i < 19; i++) {
        int x_next = (x + D3Q19_CX[i] + nx) % nx;
        int y_next = (y + D3Q19_CY[i] + ny) % ny;
        int z_next = (z + D3Q19_CZ[i] + nz) % nz;
        int idx_next = x_next + nx * (y_next + ny * z_next);
        f_out[idx_next * 19 + i] = f_in[idx * 19 + i];
    }
}

// Kernel 4: Guo forcing (float)
extern "C" __global__ void guo_forcing_kernel(
    float* f,              
    const float* u,        
    const float* force,    
    const float* tau,      
    int nx, int ny, int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_cells = nx * ny * nz;
    if (idx >= n_cells) return;

    float ux = u[idx * 3 + 0]; float uy = u[idx * 3 + 1]; float uz = u[idx * 3 + 2];
    float fx = force[idx * 3 + 0]; float fy = force[idx * 3 + 1]; float fz = force[idx * 3 + 2];

    // Occupancy culling: skip cells with negligible force (sparse DM regions).
    if (fx * fx + fy * fy + fz * fz < 1e-40f) return;

    float tau_local = tau[idx];
    float prefactor = 1.0f - 1.0f / (2.0f * tau_local);

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        float eix = (float)D3Q19_CX[i]; float eiy = (float)D3Q19_CY[i]; float eiz = (float)D3Q19_CZ[i];
        float ei_minus_u_dot_f = (eix - ux) * fx + (eiy - uy) * fy + (eiz - uz) * fz;
        float ei_dot_u = eix * ux + eiy * uy + eiz * uz;
        float ei_dot_f = eix * fx + eiy * fy + eiz * fz;
        float s_i = ei_minus_u_dot_f * 3.0f + ei_dot_u * ei_dot_f * 9.0f;
        f[idx * 19 + i] += prefactor * D3Q19_WF[i] * s_i;
    }
}

// Kernel 5: Initialize uniform density and velocity (float)
extern "C" __global__ void initialize_uniform_kernel(
    float* f,           
    float* rho,         
    float* u,           
    float rho_init,     
    float ux_init,      
    float uy_init,      
    float uz_init,      
    int nx, int ny, int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_cells = nx * ny * nz;
    if (idx >= n_cells) return;

    rho[idx] = rho_init;
    u[idx * 3 + 0] = ux_init;
    u[idx * 3 + 1] = uy_init;
    u[idx * 3 + 2] = uz_init;

    float u_local[3] = {ux_init, uy_init, uz_init};
    float f_eq[19];
    compute_equilibrium_f(f_eq, rho_init, u_local);

    for (int i = 0; i < 19; i++) f[idx * 19 + i] = f_eq[i];
}

// Kernel 5b: Initialize per-cell density and velocity (float)
extern "C" __global__ void initialize_custom_kernel(
    float* f,
    float* rho,
    float* u,
    const float* rho_in,
    const float* u_in,
    int nx,
    int ny,
    int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_cells = nx * ny * nz;
    if (idx >= n_cells) return;

    float rho_init = rho_in[idx];
    float ux_init = u_in[idx * 3 + 0];
    float uy_init = u_in[idx * 3 + 1];
    float uz_init = u_in[idx * 3 + 2];

    rho[idx] = rho_init;
    u[idx * 3 + 0] = ux_init;
    u[idx * 3 + 1] = uy_init;
    u[idx * 3 + 2] = uz_init;

    float u_local[3] = {ux_init, uy_init, uz_init};
    float f_eq[19];
    compute_equilibrium_f(f_eq, rho_init, u_local);
    for (int i = 0; i < 19; i++) f[idx * 19 + i] = f_eq[i];
}

// Complex number structure (float)
struct ComplexDeviceF {
    float re;
    float im;
};

// Kernel 6: Apply spectral mask (float)
extern "C" __global__ void apply_spectral_mask_kernel(
    ComplexDeviceF* u_hat,   
    const float* mask,     
    float damping,         
    int n_cells
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;
    if (mask[idx] < 0.5f) {
        u_hat[idx].re *= damping;
        u_hat[idx].im *= damping;
    }
}

// Kernel 7: Compute enstrophy contribution per cell (float)
extern "C" __global__ void compute_enstrophy_cell_kernel(
    const float* u,        
    float* enstrophy_field, 
    int nx, int ny, int nz
) {
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
    auto get_u = [&](int xi, int yi, int zi, int comp) {
        return u[(xi + nx * (yi + ny * zi)) * 3 + comp];
    };

    // Calculate velocity gradients using central differences
    float duz_dy = (get_u(x, yp, z, 2) - get_u(x, ym, z, 2)) * 0.5f;
    float duy_dz = (get_u(x, y, zp, 1) - get_u(x, y, zm, 1)) * 0.5f;
    float dux_dz = (get_u(x, y, zp, 0) - get_u(x, y, zm, 0)) * 0.5f;
    float duz_dx = (get_u(xp, y, z, 2) - get_u(xm, y, z, 2)) * 0.5f;
    float duy_dx = (get_u(xp, y, z, 1) - get_u(xm, y, z, 1)) * 0.5f;
    float dux_dy = (get_u(x, yp, z, 0) - get_u(x, ym, z, 0)) * 0.5f;

    // Calculate vorticity components
    float wx = duz_dy - duy_dz;
    float wy = dux_dz - duz_dx;
    float wz = duy_dx - dux_dy;
    
    // Enstrophy for this cell
    enstrophy_field[idx] = wx*wx + wy*wy + wz*wz;
}

// Kernel 8: Convert real velocity component to ComplexDevice field (float)
extern "C" __global__ void convert_real_to_complex_kernel(
    const float* u,        
    ComplexDeviceF* u_hat,   
    int comp,               
    int n_cells
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;
    u_hat[idx].re = u[idx * 3 + comp];
    u_hat[idx].im = 0.0f;
}

// Kernel 9: Convert ComplexDevice field back to real velocity component (float)
extern "C" __global__ void convert_complex_to_real_kernel(
    const ComplexDeviceF* u_hat, 
    float* u,                  
    int comp,                   
    float scale,
    int n_cells
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;
    u[idx * 3 + comp] = u_hat[idx].re * scale;
}

// Kernel 10: Reduce sum (float atomicAdd)
extern "C" __global__ void reduce_sum_kernel(
    const float* input,    
    float* output,         
    int n_cells
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n_cells; i += blockDim.x * gridDim.x) {
        atomicAdd(output, input[i]);
    }
}

// Kernel 11: Zero out a scalar float
extern "C" __global__ void zero_kernel(float* out) {
    *out = 0.0f;
}

// Kernel 12: Fused Collision + Streaming + Guo Forcing (Ultra-Throughput)
extern "C" __global__ void lbm_step_fused_kernel(
    const float* f_in,      // Input distributions
    float* f_out,           // Output distributions (after streaming)
    float* rho_out,         // Density output
    float* u_out,           // Velocity output
    const float* force,     // Force field
    const float* tau,       // Relaxation time
    int nx, int ny, int nz
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    int idx = x + nx * (y + ny * z);

    // 1. Gather macroscopic (use __ldg for read-only f_in)
    float rho_local = 0.0f;
    float mx = 0.0f, my = 0.0f, mz = 0.0f;
    float f_local[19];

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        float val = __ldg(&f_in[idx * 19 + i]);
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

    rho_out[idx] = rho_local;
    u_out[idx * 3 + 0] = ux;
    u_out[idx * 3 + 1] = uy;
    u_out[idx * 3 + 2] = uz;

    // 2. Collision + Forcing
    float f_eq[19];
    float u_vec[3] = {ux, uy, uz};
    compute_equilibrium_f(f_eq, rho_local, u_vec);

    float tau_local = __ldg(&tau[idx]);
    float inv_tau = 1.0f / tau_local;

    float fx = __ldg(&force[idx * 3 + 0]);
    float fy = __ldg(&force[idx * 3 + 1]);
    float fz = __ldg(&force[idx * 3 + 2]);

    // Occupancy culling: check if Guo forcing is needed.
    float force_mag_sq = fx * fx + fy * fy + fz * fz;
    float prefactor = 1.0f - 0.5f * inv_tau;

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        // BGK collision
        float fi = f_local[i] - (f_local[i] - f_eq[i]) * inv_tau;

        // Guo Forcing (skip for cells with negligible force)
        if (force_mag_sq >= 1e-40f) {
            float eix = (float)D3Q19_CX[i];
            float eiy = (float)D3Q19_CY[i];
            float eiz = (float)D3Q19_CZ[i];
            float ei_minus_u_dot_f = (eix - ux) * fx + (eiy - uy) * fy + (eiz - uz) * fz;
            float ei_dot_u = eix * ux + eiy * uy + eiz * uz;
            float ei_dot_f = eix * fx + eiy * fy + eiz * fz;
            float s_i = ei_minus_u_dot_f * 3.0f + ei_dot_u * ei_dot_f * 9.0f;
            fi += prefactor * D3Q19_WF[i] * s_i;
        }

        // 3. Streaming (Write to neighbor -- always executes)
        int x_next = (x + D3Q19_CX[i] + nx) % nx;
        int y_next = (y + D3Q19_CY[i] + ny) % ny;
        int z_next = (z + D3Q19_CZ[i] + nz) % nz;
        int idx_next = x_next + nx * (y_next + ny * z_next);

        f_out[idx_next * 19 + i] = fi;
    }
}
// Voudon-LBM Bridge: Viscosity modulation from 256D frustration field

extern "C" __global__ void update_tau_from_voudon_frustration_kernel(
    float* __restrict__ tau,
    const float* __restrict__ frustration,
    float tau_base,
    float alpha_voudon,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float f = frustration[idx];
    
    // Viscosity modulation formula: tau = tau_base * (1 + alpha * frustration)
    // High frustration (non-associativity) creates "stiff" temporal zones
    float tau_new = tau_base * (1.0f + alpha_voudon * f);
    
    // Clamp to physical bounds [0.505, 5.0]
    tau[idx] = fmaxf(0.505f, fminf(5.0f, tau_new));
}

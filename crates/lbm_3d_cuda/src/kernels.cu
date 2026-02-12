// CUDA kernels for D3Q19 Lattice Boltzmann Method
// Compiled at runtime via cudarc NVRTC
// Target: NVIDIA RTX 4070 Ti (Compute 8.9)

// D3Q19 lattice velocities (constant memory for fast access)
__constant__ int D3Q19_CX[19] = {
    0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0
};
__constant__ int D3Q19_CY[19] = {
    0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1
};
__constant__ int D3Q19_CZ[19] = {
    0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1
};

// D3Q19 weights
__constant__ double D3Q19_W[19] = {
    1.0/3.0,                            // i=0 (rest)
    1.0/18.0, 1.0/18.0, 1.0/18.0,      // i=1-6 (face neighbors)
    1.0/18.0, 1.0/18.0, 1.0/18.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0,      // i=7-18 (edge neighbors)
    1.0/36.0, 1.0/36.0, 1.0/36.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0
};

// Speed of sound squared (lattice units)
__device__ const double CS_SQ = 1.0 / 3.0;

// Compute equilibrium distribution for D3Q19
__device__ void compute_equilibrium(
    double* f_eq,
    double rho,
    const double* u // u[3]: velocity components
) {
    double u_sq = u[0]*u[0] + u[1]*u[1] + u[2]*u[2];

    for (int i = 0; i < 19; i++) {
        double c_dot_u = D3Q19_CX[i]*u[0] + D3Q19_CY[i]*u[1] + D3Q19_CZ[i]*u[2];
        double c_dot_u_sq = c_dot_u * c_dot_u;

        f_eq[i] = D3Q19_W[i] * rho * (
            1.0 +
            c_dot_u / CS_SQ +
            c_dot_u_sq / (2.0 * CS_SQ * CS_SQ) -
            u_sq / (2.0 * CS_SQ)
        );
    }
}

// Kernel 1: Compute macroscopic quantities (rho, u) from distributions f
extern "C" __global__ void compute_macroscopic_kernel(
    const double* f,    // Input: distributions (19 x n_cells)
    double* rho,        // Output: density (n_cells)
    double* u,          // Output: velocity (3 x n_cells)
    int nx,
    int ny,
    int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_cells = nx * ny * nz;

    if (idx >= n_cells) return;

    // Sum distributions to get density
    double rho_local = 0.0;
    for (int i = 0; i < 19; i++) {
        rho_local += f[idx * 19 + i];
    }

    // Compute momentum
    double ux = 0.0, uy = 0.0, uz = 0.0;
    for (int i = 0; i < 19; i++) {
        double fi = f[idx * 19 + i];
        ux += D3Q19_CX[i] * fi;
        uy += D3Q19_CY[i] * fi;
        uz += D3Q19_CZ[i] * fi;
    }

    // Velocity = momentum / density
    double inv_rho = 1.0 / rho_local;

    rho[idx] = rho_local;
    u[idx * 3 + 0] = ux * inv_rho;
    u[idx * 3 + 1] = uy * inv_rho;
    u[idx * 3 + 2] = uz * inv_rho;
}

// Kernel 2: BGK collision with spatially-varying relaxation time
extern "C" __global__ void bgk_collision_kernel(
    double* f,              // In/Out: distributions (19 x n_cells)
    const double* rho,      // Input: density (n_cells)
    const double* u,        // Input: velocity (3 x n_cells)
    const double* tau,      // Input: relaxation time per cell (n_cells) - SPATIALLY VARYING!
    int nx,
    int ny,
    int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_cells = nx * ny * nz;

    if (idx >= n_cells) return;

    // Get local macroscopic quantities
    double rho_local = rho[idx];
    double u_local[3] = {u[idx*3 + 0], u[idx*3 + 1], u[idx*3 + 2]};

    // Get local relaxation time (per-cell!)
    double tau_local = tau[idx];

    // Compute equilibrium distribution
    double f_eq[19];
    compute_equilibrium(f_eq, rho_local, u_local);

    // BGK collision: f_new = f - (f - f_eq) / tau
    double inv_tau = 1.0 / tau_local;
    for (int i = 0; i < 19; i++) {
        int f_idx = idx * 19 + i;
        f[f_idx] -= (f[f_idx] - f_eq[i]) * inv_tau;
    }
}

// Kernel 3: Streaming (propagate distributions to neighbor cells)
extern "C" __global__ void streaming_kernel(
    const double* f_in,   // Input: distributions before streaming
    double* f_out,        // Output: distributions after streaming
    int nx,
    int ny,
    int nz
) {
    // 3D thread indexing
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;

    int idx = x + nx * (y + ny * z);

    // Stream each velocity component to neighbor cell
    for (int i = 0; i < 19; i++) {
        // Destination cell (periodic boundaries)
        int x_next = (x + D3Q19_CX[i] + nx) % nx;
        int y_next = (y + D3Q19_CY[i] + ny) % ny;
        int z_next = (z + D3Q19_CZ[i] + nz) % nz;

        int idx_next = x_next + nx * (y_next + ny * z_next);

        // Copy distribution to neighbor
        f_out[idx_next * 19 + i] = f_in[idx * 19 + i];
    }
}

// Kernel 4: Initialize uniform density and velocity
extern "C" __global__ void initialize_uniform_kernel(
    double* f,           // Output: distributions (19 x n_cells)
    double* rho,         // Output: density (n_cells)
    double* u,           // Output: velocity (3 x n_cells)
    double rho_init,     // Initial density
    double ux_init,      // Initial velocity x
    double uy_init,      // Initial velocity y
    double uz_init,      // Initial velocity z
    int nx,
    int ny,
    int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_cells = nx * ny * nz;

    if (idx >= n_cells) return;

    // Set macroscopic quantities
    rho[idx] = rho_init;
    u[idx * 3 + 0] = ux_init;
    u[idx * 3 + 1] = uy_init;
    u[idx * 3 + 2] = uz_init;

    // Compute equilibrium distribution
    double u_local[3] = {ux_init, uy_init, uz_init};
    double f_eq[19];
    compute_equilibrium(f_eq, rho_init, u_local);

    // Initialize distributions to equilibrium
    for (int i = 0; i < 19; i++) {
        f[idx * 19 + i] = f_eq[i];
    }
}

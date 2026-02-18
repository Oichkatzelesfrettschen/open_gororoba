#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

// D3Q19 Constants
const int CX[19] = int[](0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0);
const int CY[19] = int[](0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1);
const int CZ[19] = int[](0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1);
const float WF[19] = float[](
    1.0/3.0,
    1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
);

layout(std430, set = 0, binding = 0) readonly buffer InputDist {
    float f_in[];
};

layout(std430, set = 0, binding = 1) writeonly buffer OutputDist {
    float f_out[];
};

layout(std430, set = 0, binding = 2) buffer MacroRho {
    float rho_out[];
};

layout(std430, set = 0, binding = 3) buffer MacroU {
    float u_out[]; // Stride 4 (x, y, z, padding) or packed vec3? Std430 vec3 is 16 bytes.
};

layout(push_constant) uniform Constants {
    uint nx;
    uint ny;
    uint nz;
    float tau;
} pc;

void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    uint z = gl_GlobalInvocationID.z;

    if (x >= pc.nx || y >= pc.ny || z >= pc.nz) return;

    uint idx = x + pc.nx * (y + pc.ny * z);
    uint n_cells = pc.nx * pc.ny * pc.nz;

    // 1. Compute Macroscopic Moments
    float rho = 0.0;
    vec3 u = vec3(0.0);

    // Load f_in from "Structure of Arrays" layout for coalescing? 
    // Or Array of Structures? The CUDA code used SoA implicitly by index calc?
    // Let's use Array of Structures (AoS) per cell for simplicity first: f[idx * 19 + i]
    // CUDA: f[idx * 19 + i]
    
    float f_local[19];

    for (int i = 0; i < 19; i++) {
        f_local[i] = f_in[idx * 19 + i];
        rho += f_local[i];
        u += vec3(CX[i], CY[i], CZ[i]) * f_local[i];
    }

    if (rho > 1e-9) {
        u /= rho;
    } else {
        u = vec3(0.0);
        rho = 1.0; // Reset vacuum
    }

    // Write macroscopic
    rho_out[idx] = rho;
    // std430 vec3 alignment is 16 bytes (vec4)
    // We'll write manually to a float array to be safe and dense
    u_out[idx * 3 + 0] = u.x;
    u_out[idx * 3 + 1] = u.y;
    u_out[idx * 3 + 2] = u.z;

    // 2. Collision (BGK)
    float u_sq = dot(u, u);
    float omega = 1.0 / pc.tau;
    float cs_sq = 1.0 / 3.0;
    float cs_sq_2 = 2.0 * cs_sq * cs_sq;
    float cs_sq_inv = 1.0 / cs_sq;

    for (int i = 0; i < 19; i++) {
        vec3 c = vec3(CX[i], CY[i], CZ[i]);
        float cu = dot(c, u);
        
        float feq = WF[i] * rho * (1.0 + cu * cs_sq_inv + (cu * cu) / cs_sq_2 - u_sq / (2.0 * cs_sq));
        
        f_local[i] = f_local[i] * (1.0 - omega) + feq * omega;
    }

    // 3. Streaming
    // Write f_local[i] to neighbor's input buffer (f_out)
    // f_out[neighbor_idx * 19 + i] = f_local[i]
    
    for (int i = 0; i < 19; i++) {
        // Periodic boundaries
        int next_x = (int(x) + CX[i] + int(pc.nx)) % int(pc.nx);
        int next_y = (int(y) + CY[i] + int(pc.ny)) % int(pc.ny);
        int next_z = (int(z) + CZ[i] + int(pc.nz)) % int(pc.nz);

        uint next_idx = uint(next_x) + pc.nx * (uint(next_y) + pc.ny * uint(next_z));
        
        f_out[next_idx * 19 + i] = f_local[i];
    }
}

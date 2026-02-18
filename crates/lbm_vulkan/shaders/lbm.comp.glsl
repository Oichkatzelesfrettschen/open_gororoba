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

layout(std430, set = 0, binding = 0) readonly buffer InputDist { float f_in[]; };
layout(std430, set = 0, binding = 1) writeonly buffer OutputDist { float f_out[]; };
layout(std430, set = 0, binding = 2) buffer MacroRho { float rho_out[]; };
layout(std430, set = 0, binding = 3) buffer MacroU { float u_out[]; };

// New Bindings for C-756
layout(std430, set = 0, binding = 4) readonly buffer TauField { float tau_in[]; };
layout(std430, set = 0, binding = 5) readonly buffer ForceField { float force_in[]; }; // Stride 3 (fx, fy, fz)
layout(std430, set = 0, binding = 6) writeonly buffer EntropyOut { float entropy_gen_out[]; };

layout(push_constant) uniform Constants {
    uint nx;
    uint ny;
    uint nz;
    float global_tau_scale; // To modulate the field if needed
} pc;

void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    uint z = gl_GlobalInvocationID.z;

    if (x >= pc.nx || y >= pc.ny || z >= pc.nz) return;

    uint idx = x + pc.nx * (y + pc.ny * z);
    
    // Load Spatially Varying Parameters
    float tau_local = tau_in[idx];
    float fx = force_in[idx * 3 + 0];
    float fy = force_in[idx * 3 + 1];
    float fz = force_in[idx * 3 + 2];
    vec3 F = vec3(fx, fy, fz);

    // 1. Compute Macroscopic Moments
    float rho = 0.0;
    vec3 u = vec3(0.0);
    float f_local[19];

    for (int i = 0; i < 19; i++) {
        f_local[i] = f_in[idx * 19 + i];
        rho += f_local[i];
        u += vec3(CX[i], CY[i], CZ[i]) * f_local[i];
    }

    // Apply half-force correction to velocity (Guo forcing prerequisite)
    // u = (momentum + F/2) / rho
    if (rho > 1e-9) {
        u = (u + 0.5 * F) / rho;
    } else {
        u = vec3(0.0);
        rho = 1.0; 
    }

    // Write macroscopic
    rho_out[idx] = rho;
    u_out[idx * 3 + 0] = u.x;
    u_out[idx * 3 + 1] = u.y;
    u_out[idx * 3 + 2] = u.z;

    // 2. Collision (BGK + Guo Forcing)
    float u_sq = dot(u, u);
    float omega = 1.0 / tau_local;
    float cs_sq = 1.0 / 3.0;
    float cs_sq_inv = 3.0;
    
    // Terms for forcing
    // w_i * (1 - 1/2tau) * [ (c_i - u).F / cs^2 + (c_i.u)(c_i.F) / cs^4 ]
    float force_prefactor = (1.0 - 0.5 * omega); 

    float entropy_accum = 0.0;

    for (int i = 0; i < 19; i++) {
        vec3 c = vec3(CX[i], CY[i], CZ[i]);
        float cu = dot(c, u);
        
        // Equilibrium
        float feq = WF[i] * rho * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u_sq);
        
        // Forcing Source Term (Guo)
        float cF = dot(c, F);
        float uF = dot(u, F);
        // S_i = w_i * [ (c-u).F/cs^2 + (c.u)(c.F)/cs^4 ]
        //     = w_i * [ (cF - uF)*3.0 + (cu * cF)*9.0 ]
        float Si = WF[i] * ((cF - uF) * 3.0 + (cu * cF) * 9.0);

        // Collision step
        float f_new = f_local[i] * (1.0 - omega) + feq * omega + force_prefactor * Si;
        
        // Entropy Production Estimate (Non-equilibrium intensity)
        // Delta S ~ sum (f_neq^2 / f_eq)
        float f_neq = f_local[i] - feq;
        entropy_accum += (f_neq * f_neq) / (feq + 1e-15);

        // 3. Streaming (Write to neighbor)
        int next_x = (int(x) + CX[i] + int(pc.nx)) % int(pc.nx);
        int next_y = (int(y) + CY[i] + int(pc.ny)) % int(pc.ny);
        int next_z = (int(z) + CZ[i] + int(pc.nz)) % int(pc.nz);
        uint next_idx = uint(next_x) + pc.nx * (uint(next_y) + pc.ny * uint(next_z));
        
        f_out[next_idx * 19 + i] = f_new;
    }
    
    entropy_gen_out[idx] = entropy_accum;
}

#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

const int CX[19] = int[](0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0);
const int CY[19] = int[](0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1);
const int CZ[19] = int[](0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1);
const float WF[19] = float[](1.0/3.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0);

layout(std430, set = 0, binding = 0) readonly buffer InputDist { float f_in[]; };
layout(std430, set = 0, binding = 1) writeonly buffer OutputDist { float f_out[]; };
layout(std430, set = 0, binding = 2) buffer MacroRho { float rho_out[]; };
layout(std430, set = 0, binding = 3) buffer MacroU { float u_out[]; };
layout(std430, set = 0, binding = 4) readonly buffer TauField { float tau_in[]; };
layout(std430, set = 0, binding = 5) readonly buffer ForceField { float force_in[]; };
layout(std430, set = 0, binding = 6) writeonly buffer EntropyOut { float entropy_gen_out[]; };

layout(push_constant) uniform Constants { uint nx; uint ny; uint nz; float global_tau_scale; } pc;

void main() {
    uint x = gl_GlobalInvocationID.x; uint y = gl_GlobalInvocationID.y; uint z = gl_GlobalInvocationID.z;
    if (x >= pc.nx || y >= pc.ny || z >= pc.nz) return;
    uint idx = x + pc.nx * (y + pc.ny * z);
    
    float tau_local = max(0.55, tau_in[idx]); // Stability floor
    vec3 F = vec3(force_in[idx * 3 + 0], force_in[idx * 3 + 1], force_in[idx * 3 + 2]) * 0.01; // Reduced magnitude

    float rho = 0.0; vec3 u = vec3(0.0); float f_local[19];
    for (int i = 0; i < 19; i++) {
        f_local[i] = max(0.0, f_in[idx * 19 + i]); // Positivity
        rho += f_local[i];
        u += vec3(CX[i], CY[i], CZ[i]) * f_local[i];
    }

    if (rho > 1e-6) {
        u = (u + 0.5 * F) / rho;
    } else {
        u = vec3(0.0); rho = 1e-6; 
    }
    
    // Clamp velocity to Mach 0.3 for stability
    float speed = length(u);
    if (speed > 0.1) { u = normalize(u) * 0.1; }

    rho_out[idx] = rho;
    u_out[idx * 3 + 0] = u.x; u_out[idx * 3 + 1] = u.y; u_out[idx * 3 + 2] = u.z;

    float u_sq = dot(u, u);
    float omega = 1.0 / tau_local;
    float force_prefactor = (1.0 - 0.5 * omega); 
    float entropy_accum = 0.0;

    for (int i = 0; i < 19; i++) {
        vec3 c = vec3(CX[i], CY[i], CZ[i]);
        float cu = dot(c, u);
        float feq = WF[i] * rho * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u_sq);
        
        float cF = dot(c, F);
        float uF = dot(u, F);
        float Si = WF[i] * ((cF - uF) * 3.0 + (cu * cF) * 9.0);

        float f_new = f_local[i] * (1.0 - omega) + feq * omega + force_prefactor * Si;
        entropy_accum += (f_local[i] - feq) * (f_local[i] - feq) / (feq + 1e-9);

        int nx = int(pc.nx); int ny = int(pc.ny); int nz = int(pc.nz);
        int next_x = (int(x) + CX[i] + nx) % nx;
        int next_y = (int(y) + CY[i] + ny) % ny;
        int next_z = (int(z) + CZ[i] + nz) % nz;
        uint next_idx = uint(next_x) + pc.nx * (uint(next_y) + pc.ny * uint(next_z));
        f_out[next_idx * 19 + i] = f_new;
    }
    entropy_gen_out[idx] = entropy_accum;
}

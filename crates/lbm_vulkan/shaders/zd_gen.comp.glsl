#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
layout(std430, set = 0, binding = 0) writeonly buffer TauField { float tau_out[]; };

layout(push_constant) uniform Constants {
    uint nx; uint ny; uint nz;
    float tau_base; float tau_amp; float lambda;
} pc;

const uint assessors[84] = uint[](1,10, 1,11, 1,12, 1,13, 1,14, 1,15, 2,9, 2,11, 2,12, 2,13, 2,14, 2,15, 3,9, 3,10, 3,12, 3,13, 3,14, 3,15, 4,9, 4,10, 4,11, 4,13, 4,14, 4,15, 5,9, 5,10, 5,11, 5,12, 5,14, 5,15, 6,9, 6,10, 6,11, 6,12, 6,13, 6,15, 7,9, 7,10, 7,11, 7,12, 7,13, 7,14);

void main() {
    uint x = gl_GlobalInvocationID.x; uint y = gl_GlobalInvocationID.y; uint z = gl_GlobalInvocationID.z;
    if (x >= pc.nx || y >= pc.ny || z >= pc.nz) return;
    uint idx = x + pc.nx * (y + pc.ny * z);

    vec3 p = vec3(float(x)/float(pc.nx), float(y)/float(pc.ny), float(z)/float(pc.nz));
    float psi[16];
    float norm_sq = 0.0;
    
    // Smooth, low-frequency folding
    psi[0] = 0.5; norm_sq += 0.25;
    for (int k = 1; k < 16; k++) {
        float fk = float(k) * 0.5;
        psi[k] = sin(p.x * fk + fk) * cos(p.y * fk - p.z * 0.2);
        norm_sq += psi[k]*psi[k];
    }
    float inv_norm = inversesqrt(norm_sq);
    for (int k = 0; k < 16; k++) { psi[k] *= inv_norm; }

    float max_proj = 0.0;
    for (int i = 0; i < 42; i++) {
        uint a = assessors[2*i]; uint b = assessors[2*i+1];
        max_proj = max(max_proj, abs(psi[a]) + abs(psi[b]));
    }
    max_proj *= 0.70710678;
    
    float dist_sq = 2.0 * (1.0 - max_proj);
    // High viscosity near ZDs
    tau_out[idx] = pc.tau_base + pc.tau_amp * exp(-dist_sq * pc.lambda);
}

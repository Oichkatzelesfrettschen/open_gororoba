#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

layout(std430, set = 0, binding = 0) writeonly buffer TauField { float tau_out[]; };

layout(push_constant) uniform Constants {
    uint nx;
    uint ny;
    uint nz;
    float tau_base;
    float tau_amp;
    float lambda; // Decay rate
} pc;

// 42 Primitive Assessor Pairs (a, b)
// The 84 ZDs are e_a + e_b and e_a - e_b
const uint assessors[84] = uint[](
    1,10, 1,11, 1,12, 1,13, 1,14, 1,15, 
    2,9,  2,11, 2,12, 2,13, 2,14, 2,15, 
    3,9,  3,10, 3,12, 3,13, 3,14, 3,15, 
    4,9,  4,10, 4,11, 4,13, 4,14, 4,15, 
    5,9,  5,10, 5,11, 5,12, 5,14, 5,15, 
    6,9,  6,10, 6,11, 6,12, 6,13, 6,15, 
    7,9,  7,10, 7,11, 7,12, 7,13, 7,14 
);

// Deterministic 16D Vector Generation
// "Spectral Folding" of 3D space into 16D Algebra
void generate_psi(vec3 p, out float psi[16]) {
    float norm_sq = 0.0;
    
    // Base frequency
    float f = 3.14159 * 2.0; 
    
    // e0: Real component (vacuum energy)
    psi[0] = 0.1; 
    norm_sq += psi[0]*psi[0];

    // e1..e3: Aligned with space
    psi[1] = p.x;
    psi[2] = p.y;
    psi[3] = p.z;
    norm_sq += p.x*p.x + p.y*p.y + p.z*p.z;

    // e4..e15: Higher modes (folded)
    for (int k = 4; k < 16; k++) {
        float fk = float(k);
        // Complex interference pattern
        float val = sin(p.x * fk * f) * cos(p.y * fk * f + p.z);
        psi[k] = val * 0.5; // Dampen higher modes
        norm_sq += psi[k]*psi[k];
    }

    // Normalize to unit sphere S^15
    float inv_norm = inversesqrt(norm_sq);
    for (int k = 0; k < 16; k++) {
        psi[k] *= inv_norm;
    }
}

void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    uint z = gl_GlobalInvocationID.z;

    if (x >= pc.nx || y >= pc.ny || z >= pc.nz) return;

    uint idx = x + pc.nx * (y + pc.ny * z);

    // Normalized coordinates [0, 1]
    vec3 p = vec3(float(x)/float(pc.nx), float(y)/float(pc.ny), float(z)/float(pc.nz));

    // 1. Generate local Sedenion Field State
    float psi[16];
    generate_psi(p, psi);

    // 2. Find distance to nearest Zero Divisor
    // Maximize projection P = |psi_a| + |psi_b| ? No, max(psi_a + psi_b, psi_a - psi_b)
    // = |psi_a| + sign(psi_a*psi_b)*psi_b? No.
    // Maximize P = |psi_a + s*psi_b| is wrong.
    // ZDs are (e_a + e_b)/sqrt(2) and (e_a - e_b)/sqrt(2).
    // Projection = dot(psi, zd).
    // P_plus  = (psi[a] + psi[b]) * 0.7071
    // P_minus = (psi[a] - psi[b]) * 0.7071
    // We want max(P_plus, P_minus) -> max(|psi[a] + psi[b]|, |psi[a] - psi[b]|) 
    // Actually, simple algebra: max(|psi[a] + psi[b]|, |psi[a] - psi[b]|) = |psi[a]| + |psi[b]|.
    // Wait. |x+y| vs |x-y|. 
    // if signs match, |x+y| = |x|+|y|. If differ, |x-y| = |x|+|y|.
    // So yes, max projection onto the *pair* (a,b) regardless of sign is just |psi[a]| + |psi[b]|.
    // BUT we must normalize the ZD vector by 1/sqrt(2).
    
    float max_proj = 0.0;
    
    // Iterate 42 assessors (stored as 84 ints in flat array)
    for (int i = 0; i < 42; i++) {
        uint a = assessors[2*i];
        uint b = assessors[2*i+1];
        
        float val_a = psi[a];
        float val_b = psi[b];
        
        // Project onto subspace spanned by e_a, e_b
        // The ZDs are the diagonals.
        // Max projection of (va, vb) onto (1,1) or (1,-1) is |va| + |vb|.
        float p_local = abs(val_a) + abs(val_b);
        if (p_local > max_proj) {
            max_proj = p_local;
        }
    }
    
    // The ZDs are unit vectors, so we multiply by 1/sqrt(2)
    max_proj *= 0.70710678; // 1/sqrt(2)
    
    // Distance squared d^2 = ||psi - zd||^2 = ||psi||^2 + ||zd||^2 - 2 dot
    // = 1 + 1 - 2 * max_proj = 2 * (1 - max_proj)
    // Distance d = sqrt(2 * (1 - max_proj))
    
    // If max_proj = 1.0 (perfect alignment), d=0.
    
    float dist_sq = 2.0 * (1.0 - max_proj);
    if (dist_sq < 0.0) dist_sq = 0.0; // Numeric safety

    // 3. Map to Viscosity
    // High viscosity near ZDs (dist ~ 0)
    float tau = pc.tau_base + pc.tau_amp * exp(-dist_sq * pc.lambda);

    tau_out[idx] = tau;
}

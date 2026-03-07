// Chingon N-dimensional AVT tensor contraction -- CUDA kernel (FP32)
//
// Performs the bilinear AVT contraction:
//   force_Nd[m] += alpha * v_Nd[i] * v_Nd[j] * sign
// for each packed violation (i, j, m, sign).
//
// Dimension-parametric: works for 64D, 128D, 256D, 512D, 1024D.
//
// The kernel is designed for the RTX 4070 Ti (AD104, SM 8.9):
//   - FP32: 128 ALUs/SM = 7680 total, 40 TFLOPS
//   - FP64: 2 ALUs/SM = 120 total, 1/64th FP32 rate
//   -> ALL math is FP32. Orbital mechanics stays on CPU in f64.
//
// Bit-packed violations: each uint32 encodes (i, j, m, sign_positive).
// Layout (LSB to MSB):
//   [m: index_bits] [j: index_bits] [i: index_bits] [sign_positive: 1]
//
// At 64D:  index_bits=6,  total=19 bits per violation
// At 128D: index_bits=7,  total=22 bits per violation
// At 256D: index_bits=8,  total=25 bits per violation
// At 1024D: index_bits=10, total=31 bits per violation

// Maximum dimension we support (1024D = DekaVoudon)
#define MAX_DIM 1024

// Shared memory for the N-dimensional state vector.
// At 256D: 256 * 4 bytes = 1 KB (fits easily in SM shared memory).
// At 1024D: 1024 * 4 bytes = 4 KB (still fine, SM has 48-100 KB).
extern __shared__ float s_v_Nd[];

// ------------------------------------------------------------------
// Kernel 1: AVT tensor contraction with register accumulation
//
// Each BLOCK processes the ENTIRE violation list against ONE state vector.
//
// Strategy:
//   1. Load v_Nd into shared memory (coalesced, one load per thread)
//   2. Each thread processes a stride of violations: idx, idx+blockDim, ...
//   3. Accumulate per-thread partial sums in local array
//   4. AtomicAdd to global force output
// ------------------------------------------------------------------
extern "C" __global__ void chingon_avt_contraction(
    const unsigned int* __restrict__ packed_avt,   // [n_violations] bit-packed
    const float* __restrict__ v_Nd,                // [dim] state vector
    float* __restrict__ force_Nd,                  // [dim] output force (zeroed by caller)
    unsigned int n_violations,
    unsigned int dim,
    unsigned int index_bits,
    float alpha,
    float inv_n_viol                               // 1.0 / n_violations
) {
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int global_tid = bid * blockDim.x + tid;
    unsigned int n_threads = blockDim.x * gridDim.x;
    unsigned int mask = (1u << index_bits) - 1u;

    // Phase 1: Load state vector into shared memory
    for (unsigned int d = tid; d < dim; d += blockDim.x) {
        s_v_Nd[d] = v_Nd[d];
    }
    __syncthreads();

    // Phase 2: Per-thread accumulation over strided violations
    // Register path: float local_force[dim] uses `dim` registers per thread.
    // SM 8.9 has 65536 regs/SM. At 128D: 128 regs/thread -> 512 threads/SM (2 blocks).
    // At 256D: 256+ regs/thread -> 1 block/SM, likely register spilling to L1.
    // Threshold at 128D: use register path for 64D/128D, atomic path for 256D+.
    if (dim <= 128) {
        // Register-based accumulation (optimal for 64D/128D)
        float local_force[128];
        for (unsigned int d = 0; d < dim; d++) {
            local_force[d] = 0.0f;
        }

        for (unsigned int v = global_tid; v < n_violations; v += n_threads) {
            unsigned int packed = packed_avt[v];
            unsigned int m_idx = packed & mask;
            unsigned int j_idx = (packed >> index_bits) & mask;
            unsigned int i_idx = (packed >> (2u * index_bits)) & mask;
            unsigned int sign_pos = (packed >> (3u * index_bits)) & 1u;
            float sign_val = sign_pos ? 2.0f : -2.0f;

            float contrib = alpha * s_v_Nd[i_idx] * s_v_Nd[j_idx] * sign_val;
            local_force[m_idx] += contrib;
        }

        // Normalize and write to global (atomicAdd for cross-thread safety)
        for (unsigned int d = 0; d < dim; d++) {
            float val = local_force[d] * inv_n_viol;
            if (val != 0.0f) {
                atomicAdd(&force_Nd[d], val);
            }
        }
    } else {
        // Atomic path for dim >= 256 (256D, 512D, 1024D)
        for (unsigned int v = global_tid; v < n_violations; v += n_threads) {
            unsigned int packed = packed_avt[v];
            unsigned int m_idx = packed & mask;
            unsigned int j_idx = (packed >> index_bits) & mask;
            unsigned int i_idx = (packed >> (2u * index_bits)) & mask;
            unsigned int sign_pos = (packed >> (3u * index_bits)) & 1u;
            float sign_val = sign_pos ? 2.0f : -2.0f;

            float contrib = alpha * s_v_Nd[i_idx] * s_v_Nd[j_idx] * sign_val * inv_n_viol;
            atomicAdd(&force_Nd[m_idx], contrib);
        }
    }
}

// ------------------------------------------------------------------
// Kernel 2: Build N-D state vector from orbital parameters
//
// Dimension-parametric: accepts block layout as kernel arguments.
// One thread per axis (launch with blockDim.x >= dim).
//
// Block layout:
//   Block 1 [b1_start .. b1_start + block_size): h_earth via Earth triad
//   Block 2 [b2_start .. b2_start + block_size): v_rel via Lunar triad
//   Block 3 [b3_start .. b3_start + b3_size):    cross-coupling via Solar triad
//
// For 64D:  blocks = 21/21/21, n_phases = 7
// For 128D: blocks = 42/42/43, n_phases_b12 = 14, n_phases_b3 = 15
// For 256D: blocks = 85/85/85, n_phases = 29
// ------------------------------------------------------------------
extern "C" __global__ void chingon_build_state_3body(
    float* __restrict__ v_Nd,         // [dim] output state vector
    // Block layout parameters
    unsigned int dim,
    unsigned int block_size,          // block1 and block2 size
    unsigned int block3_size,         // block3 size (may differ for prime dims)
    unsigned int b1_start,            // first axis of block 1
    unsigned int b2_start,            // first axis of block 2
    unsigned int b3_start,            // first axis of block 3
    unsigned int n_phases,            // trig phases for blocks 1 & 2
    unsigned int n_phases_b3,         // trig phases for block 3
    // Earth triad (row-major 3x3)
    float e_v_earth_x, float e_v_earth_y, float e_v_earth_z,
    float e_h_earth_x, float e_h_earth_y, float e_h_earth_z,
    float e_n_earth_x, float e_n_earth_y, float e_n_earth_z,
    // Lunar triad
    float e_v_lunar_x, float e_v_lunar_y, float e_v_lunar_z,
    float e_h_lunar_x, float e_h_lunar_y, float e_h_lunar_z,
    float e_n_lunar_x, float e_n_lunar_y, float e_n_lunar_z,
    // Solar triad
    float e_v_solar_x, float e_v_solar_y, float e_v_solar_z,
    float e_h_solar_x, float e_h_solar_y, float e_h_solar_z,
    float e_n_solar_x, float e_n_solar_y, float e_n_solar_z,
    // h_earth projected into Earth triad [3]
    float h_triad_earth_0, float h_triad_earth_1, float h_triad_earth_2,
    // v_rel projected into Lunar triad [3]
    float vrel_triad_lunar_0, float vrel_triad_lunar_1, float vrel_triad_lunar_2,
    // h_earth projected into Solar triad [3]
    float h_triad_solar_0, float h_triad_solar_1, float h_triad_solar_2,
    // v_hat projected into Solar triad [3]
    float vhat_triad_solar_0, float vhat_triad_solar_1, float vhat_triad_solar_2,
    // Scalars
    float h_earth_norm,
    float v_rel_norm,
    float cross_sign
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= dim) return;

    const float TAU = 6.28318530717958647692f;
    float h_triad_earth[3] = { h_triad_earth_0, h_triad_earth_1, h_triad_earth_2 };
    float vrel_triad_lunar[3] = { vrel_triad_lunar_0, vrel_triad_lunar_1, vrel_triad_lunar_2 };
    float h_triad_solar[3] = { h_triad_solar_0, h_triad_solar_1, h_triad_solar_2 };
    float vhat_triad_solar[3] = { vhat_triad_solar_0, vhat_triad_solar_1, vhat_triad_solar_2 };

    float val = 0.0f;

    if (tid == 0) {
        // Axis 0: real component, always 0
        val = 0.0f;
    } else if (tid >= b1_start && tid < b1_start + block_size) {
        // Block 1: angular momentum via Earth triad
        unsigned int axis = tid - b1_start;
        unsigned int comp = axis % 3;
        unsigned int phase_idx = axis / 3;
        float phase = TAU * (float)phase_idx / (float)n_phases;
        float cos_p = cosf(phase);
        val = h_triad_earth[comp] * cos_p;
    } else if (tid >= b2_start && tid < b2_start + block_size) {
        // Block 2: velocity via Lunar triad
        unsigned int axis = tid - b2_start;
        unsigned int comp = axis % 3;
        unsigned int phase_idx = axis / 3;
        float phase = TAU * (float)phase_idx / (float)n_phases;
        float sin_p = sinf(phase);
        float cos_p = cosf(phase);
        val = vrel_triad_lunar[comp] * (sin_p + cos_p);
    } else if (tid >= b3_start && tid < b3_start + block3_size) {
        // Block 3: cross-coupling via Solar triad
        unsigned int axis = tid - b3_start;
        unsigned int comp = axis % 3;
        unsigned int phase_idx = axis / 3;
        float phase = TAU * (float)phase_idx / (float)n_phases_b3;
        float sin_p = sinf(phase);
        float cos_p = cosf(phase);
        float h_inv = (h_earth_norm > 1.0e-30f) ? (1.0f / h_earth_norm) : 0.0f;
        float h_c = h_triad_solar[comp] * h_inv;
        float v_c = vhat_triad_solar[comp];
        val = cross_sign * v_rel_norm * (h_c * sin_p + v_c * cos_p);
    }

    v_Nd[tid] = val;
}

// ------------------------------------------------------------------
// Kernel 3: Project N-D force back to 3D via Solar triad
//
// Dimension-parametric: accepts block3_start and block3_size as args.
// Single-thread since O(block3_size) work.
// ------------------------------------------------------------------
extern "C" __global__ void chingon_project_3body(
    const float* __restrict__ force_Nd,  // [dim] force in N-D
    float* __restrict__ force_3d,        // [3] output force in ECI
    // Block layout
    unsigned int dim,
    unsigned int b3_start,
    unsigned int block3_size,
    unsigned int n_phases_b3,
    // Solar triad (for cross-coupling block projection)
    float e_v_solar_x, float e_v_solar_y, float e_v_solar_z,
    float e_h_solar_x, float e_h_solar_y, float e_h_solar_z,
    float e_n_solar_x, float e_n_solar_y, float e_n_solar_z,
    // h_earth projected into Solar triad [3]
    float h_triad_solar_0, float h_triad_solar_1, float h_triad_solar_2,
    // v_hat projected into Solar triad [3]
    float vhat_triad_solar_0, float vhat_triad_solar_1, float vhat_triad_solar_2,
    float h_earth_norm,
    float cross_sign
) {
    const float TAU = 6.28318530717958647692f;
    float h_triad_solar[3] = { h_triad_solar_0, h_triad_solar_1, h_triad_solar_2 };
    float vhat_triad_solar[3] = { vhat_triad_solar_0, vhat_triad_solar_1, vhat_triad_solar_2 };

    float res_triad[3] = { 0.0f, 0.0f, 0.0f };
    float h_inv = (h_earth_norm > 1.0e-30f) ? (1.0f / h_earth_norm) : 0.0f;

    for (unsigned int axis = 0; axis < block3_size; axis++) {
        unsigned int comp = axis % 3;
        unsigned int phase_idx = axis / 3;
        float phase = TAU * (float)phase_idx / (float)n_phases_b3;
        float sin_p = sinf(phase);
        float cos_p = cosf(phase);
        float h_c = h_triad_solar[comp] * h_inv;
        float v_c = vhat_triad_solar[comp];
        float proj = cross_sign * (h_c * sin_p + v_c * cos_p);
        res_triad[comp] += force_Nd[b3_start + axis] * proj;
    }

    // Rotate from Solar triad back to ECI, then divide by dim
    float inv_dim = 1.0f / (float)dim;
    force_3d[0] = (e_v_solar_x * res_triad[0] + e_h_solar_x * res_triad[1] + e_n_solar_x * res_triad[2]) * inv_dim;
    force_3d[1] = (e_v_solar_y * res_triad[0] + e_h_solar_y * res_triad[1] + e_n_solar_y * res_triad[2]) * inv_dim;
    force_3d[2] = (e_v_solar_z * res_triad[0] + e_h_solar_z * res_triad[1] + e_n_solar_z * res_triad[2]) * inv_dim;
}

// ------------------------------------------------------------------
// Kernel 4: Zero a float buffer (utility)
// ------------------------------------------------------------------
extern "C" __global__ void chingon_zero_buffer(
    float* __restrict__ buf,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        buf[idx] = 0.0f;
    }
}

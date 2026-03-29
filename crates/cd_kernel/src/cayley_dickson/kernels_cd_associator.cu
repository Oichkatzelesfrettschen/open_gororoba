// CUDA kernel for batch CD associator norm computation.
//
// Computes ||(a*b)*c - a*(b*c)|| for sliding window triplets (a,b,c)
// from a batch of embedded vectors. One thread per triplet.
//
// The CD multiplication uses the Cayley-Dickson doubling formula:
//   (a,b)(c,d) = (ac - conj(d)*b, d*a + b*conj(c))
// implemented recursively with iterative stack simulation for dim >= 16.
//
// Data layout: batch of N vectors, each of dimension D (f32, SoA or AoS).
// Output: N-2 associator norms (one per sliding-window triplet).

#define MAX_DIM 4096
#define MAX_STACK 128  // log2(4096) * ~10 entries per level

// Base case: complex multiplication (dim=2)
__device__ void cd_mul_2(const float* a, const float* b, float* out) {
    out[0] = a[0] * b[0] - a[1] * b[1];
    out[1] = a[0] * b[1] + a[1] * b[0];
}

// Base case: quaternion multiplication (dim=4)
__device__ void cd_mul_4(const float* a, const float* b, float* out) {
    out[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3];
    out[1] = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2];
    out[2] = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1];
    out[3] = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0];
}

// General CD multiply for arbitrary power-of-2 dimension.
// Uses per-thread local memory for workspace (safe up to ~256D on modern GPUs).
// For dim > 256, uses global workspace buffer.
//
// (a_l, a_r) * (c_l, c_r) = (a_l*c_l - conj(c_r)*a_r, c_r*a_l + a_r*conj(c_l))
__device__ void cd_multiply_device(
    const float* a,
    const float* b,
    float* out,
    int dim,
    float* workspace   // at least 6 * dim floats
) {
    if (dim == 1) {
        out[0] = a[0] * b[0];
        return;
    }
    if (dim == 2) {
        cd_mul_2(a, b, out);
        return;
    }
    if (dim == 4) {
        cd_mul_4(a, b, out);
        return;
    }

    int half = dim / 2;

    // Workspace layout: [conj_cr | conj_cl | t1 | t2 | t3 | t4 | recurse_ws...]
    float* conj_cr  = workspace;
    float* conj_cl  = workspace + half;
    float* t1       = workspace + 2 * half;
    float* t2       = workspace + 3 * half;
    float* t3       = workspace + 4 * half;
    float* t4       = workspace + 5 * half;
    float* rec_ws   = workspace + 6 * half;

    // Compute conjugates
    conj_cr[0] = b[half];
    conj_cl[0] = b[0];
    for (int i = 1; i < half; i++) {
        conj_cr[i] = -b[half + i];
        conj_cl[i] = -b[i];
    }

    // 4 sub-multiplies
    cd_multiply_device(&a[0],    &b[0],    t1, half, rec_ws);  // a_l * c_l
    cd_multiply_device(conj_cr,  &a[half], t2, half, rec_ws);  // conj(c_r) * a_r
    cd_multiply_device(&b[half], &a[0],    t3, half, rec_ws);  // c_r * a_l
    cd_multiply_device(&a[half], conj_cl,  t4, half, rec_ws);  // a_r * conj(c_l)

    // Combine
    for (int i = 0; i < half; i++) {
        out[i]        = t1[i] - t2[i];
        out[half + i] = t3[i] + t4[i];
    }
}

// Compute L2 norm of difference: ||a - b||
__device__ float diff_norm(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sqrtf(sum);
}

// Workspace size for cd_multiply_device at given dim:
// Each level needs 6*half, total across all levels = 6 * (dim/2 + dim/4 + ... + 2) < 6*dim
__device__ int cd_ws_size(int dim) {
    return 6 * dim;  // conservative upper bound
}

// =============================================================
// Main kernel: batch sliding-window associator norms
// =============================================================
// Input: vectors[N * D] in AoS layout (vector i at offset i * D)
// Output: norms[N - 2] (one per triplet)
//
// Thread i computes ||(v[i]*v[i+1])*v[i+2] - v[i]*(v[i+1]*v[i+2])||

extern "C" __global__ void batch_cd_associator_kernel(
    const float* __restrict__ vectors,   // [N * D]
    float* __restrict__ norms,           // [N - 2]
    float* __restrict__ workspace,       // [n_threads * ws_per_thread]
    const int N,                         // number of vectors
    const int D,                         // dimension (power of 2)
    const int ws_per_thread              // workspace floats per thread
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_triplets = N - 2;
    if (tid >= n_triplets) return;

    // Pointers to the 3 vectors forming this triplet
    const float* a = vectors + tid * D;
    const float* b = vectors + (tid + 1) * D;
    const float* c = vectors + (tid + 2) * D;

    // Per-thread workspace in global memory
    float* my_ws = workspace + tid * ws_per_thread;

    // Divide workspace: [buf_ab | buf_bc | buf_abc | buf_a_bc | mul_ws]
    float* buf_ab  = my_ws;
    float* buf_bc  = my_ws + D;
    float* buf_abc = my_ws + 2 * D;
    float* buf_a_bc= my_ws + 3 * D;
    float* mul_ws  = my_ws + 4 * D;

    // Step 1: buf_ab = a * b
    cd_multiply_device(a, b, buf_ab, D, mul_ws);
    // Step 2: buf_bc = b * c
    cd_multiply_device(b, c, buf_bc, D, mul_ws);
    // Step 3: buf_abc = (a*b) * c
    cd_multiply_device(buf_ab, c, buf_abc, D, mul_ws);
    // Step 4: buf_a_bc = a * (b*c)
    cd_multiply_device(a, buf_bc, buf_a_bc, D, mul_ws);

    // Step 5: norm = ||buf_abc - buf_a_bc||
    norms[tid] = diff_norm(buf_abc, buf_a_bc, D);
}

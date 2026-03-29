// TurboQuant CUDA kernels: quantize, dequant+dot, sign-sketch inner product.
//
// Design rules (from steinmarder SASS measurements on Ada sm_89):
//   - Maximize FFMA (4.54 cyc, 44.6 ops/clk)
//   - AVOID MUFU.RCP (41.53 cyc) and MUFU.EX2 (17.55 cyc)
//   - Minimize LDG L2 miss (92-123 cyc), target L1 residency (128KB on Ada)
//   - POPC = 7-8 cyc, fast enough for inline sign operations
//   - FTZ = zero overhead on Ada
//
// Layout: SoA (Structure-of-Arrays)
//   indices[coord_idx * n_vectors + vec_idx]  -- coalesced per-coordinate access
//   signs packed as u32 words (32 signs per word)
//
// Storage-compute split (from steinmarder INT8 SoA LBM, 2.85x speedup):
//   Store as u8 indices, promote to f32 centroid values at point of use.

extern "C" {

// ---------- Kernel 1: Boundary-search quantization ----------
// Quantize n_values f32 values using sorted boundaries.
// Each thread handles one value.
//
// boundaries: sorted boundary array, length (2^bits - 1)
// values:     input f32 values, length n_values
// indices:    output u8 indices, length n_values
// n_boundaries: number of boundaries (2^bits - 1)
// n_values:   total values to quantize
__global__ void turboquant_quantize_boundary(
    const float* __restrict__ boundaries,
    const float* __restrict__ values,
    unsigned char* __restrict__ indices,
    int n_boundaries,
    int n_values
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_values) return;

    float v = values[idx];
    unsigned char count = 0;

    // Boundary-count: for 3-bit (7 boundaries), this is 7 comparisons.
    // The compiler will unroll for small n_boundaries.
    for (int b = 0; b < n_boundaries; b++) {
        if (v > boundaries[b]) count++;
    }

    indices[idx] = count;
}

// ---------- Kernel 2: Fused dequant + dot product ----------
// For each query, compute dot product with all compressed key vectors.
// Storage-compute split: read u8 index, look up f32 centroid, accumulate.
//
// This is the hot kernel for attention score computation.
// SoA layout: key_indices[coord * n_keys + key_idx]
//
// queries:      (n_queries, d) f32, row-major
// key_indices:  (d, n_keys) u8, SoA layout (coord-major)
// centroids:    (n_levels,) f32 codebook
// key_norms:    (n_keys,) f32, per-key normalization factors
// scores:       (n_queries, n_keys) f32 output
// d:            head dimension
// n_queries:    number of query vectors
// n_keys:       number of key vectors
// n_levels:     codebook size (2^bits)
__global__ void turboquant_dequant_dot(
    const float* __restrict__ queries,
    const unsigned char* __restrict__ key_indices,
    const float* __restrict__ centroids,
    const float* __restrict__ key_norms,
    float* __restrict__ scores,
    int d,
    int n_queries,
    int n_keys,
    int n_levels
) {
    // Each thread computes one (query, key) attention score.
    int qi = blockIdx.y;  // query index
    int ki = blockIdx.x * blockDim.x + threadIdx.x;  // key index
    if (qi >= n_queries || ki >= n_keys) return;

    const float* q = queries + qi * d;
    float dot = 0.0f;

    // Storage-compute split: read u8 index, promote to f32 centroid
    for (int c = 0; c < d; c++) {
        unsigned char idx = key_indices[c * n_keys + ki];  // SoA read (coalesced)
        float k_val = centroids[idx];                       // codebook lookup (L1 cached)
        dot += q[c] * k_val;                                // FFMA
    }

    // Apply per-key norm
    scores[qi * n_keys + ki] = dot * key_norms[ki];
}

// ---------- Kernel 3: Sign-sketch inner product (QJL correction) ----------
// Compute <S @ query, packed_signs> for the QJL residual correction.
// Signs are bit-packed: 32 signs per u32 word.
//
// s_matrix:     (m, d) f32, QJL projection matrix (row-major)
// query:        (d,) f32, single query vector
// packed_signs: (n_keys, n_words) u32, bit-packed signs
// residual_norms: (n_keys,) f32
// correction:   (n_keys,) f32 output, the QJL correction term
// d:            head dimension
// m:            QJL projection dimension
// n_keys:       number of key vectors
// n_words:      words per sign vector (ceil(m / 32))
// scale:        sqrt(pi/2) / m
__global__ void turboquant_sign_dot(
    const float* __restrict__ s_matrix,
    const float* __restrict__ query,
    const unsigned int* __restrict__ packed_signs,
    const float* __restrict__ residual_norms,
    float* __restrict__ correction,
    int d,
    int m,
    int n_keys,
    int n_words,
    float scale
) {
    int ki = blockIdx.x * blockDim.x + threadIdx.x;
    if (ki >= n_keys) return;

    // Project query: proj[j] = sum_i S[j*d + i] * query[i]
    // Then compute <proj, signs> where signs are bit-packed
    float result = 0.0f;
    const unsigned int* signs = packed_signs + ki * n_words;

    for (int j = 0; j < m; j++) {
        // Compute projection for dimension j
        float proj = 0.0f;
        const float* s_row = s_matrix + j * d;
        for (int i = 0; i < d; i++) {
            proj += s_row[i] * query[i];  // FFMA
        }

        // Extract sign bit: +1 if set, -1 if clear
        int word_idx = j / 32;
        int bit_idx = j % 32;
        float sign = (signs[word_idx] >> bit_idx) & 1 ? 1.0f : -1.0f;

        result += proj * sign;
    }

    // Apply scaling: ||r|| * sqrt(pi/2) / m * <S@q, signs>
    correction[ki] = residual_norms[ki] * scale * result;
}

// ---------- Kernel 4: Fast Walsh-Hadamard Transform ----------
// In-place WHT for d values per vector, processing n_vectors in parallel.
// Each thread handles one vector.  The butterfly is sequential within
// the vector (d = 128 -> 7 levels, 896 ops per vector).
//
// For d=128 on Ada: 7 * 64 FFMA = 448 FFMA per vector = ~100 cycles.
// At 128 threads/block, 128 vectors per block = 12.8K cycles/block.
//
// data:      (n_vectors, d) f32, in-place transform
// d1:        (d,) f32, Rademacher diagonal 1 (+/-1 values)
// d2:        (d,) f32, Rademacher diagonal 2 (+/-1 values)
// d:         dimension (must be power of 2)
// n_vectors: number of vectors to transform
// inv_sqrt_d: 1.0 / sqrt(d), precomputed normalization
__global__ void turboquant_fast_jl_rotate(
    float* __restrict__ data,
    const float* __restrict__ d1,
    const float* __restrict__ d2,
    int d,
    int n_vectors,
    float inv_sqrt_d
) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= n_vectors) return;

    float* vec = data + vid * d;

    // Step 1: multiply by D2
    for (int i = 0; i < d; i++) {
        vec[i] *= d2[i];
    }

    // Step 2: in-place WHT butterfly
    for (int h = 1; h < d; h *= 2) {
        for (int i = 0; i < d; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = vec[j];
                float b = vec[j + h];
                vec[j] = a + b;
                vec[j + h] = a - b;
            }
        }
    }

    // Step 3: normalize and multiply by D1
    for (int i = 0; i < d; i++) {
        vec[i] *= inv_sqrt_d * d1[i];
    }
}

// ---------- Kernel 5: Q16.16 fixed-point dequant+dot ----------
// Exact order-independent attention score computation via integer arithmetic.
// Based on steinmarder's Q16.16 LBM kernel: 1945 MLUPS, zero mass drift.
//
// Q16.16 format: 32-bit integer, bits[31:16]=integer, bits[15:0]=fraction.
// Resolution: 1/65536 = 1.53e-5.  Accumulation via IADD3 is exact.
//
// centroids_q16: (n_levels,) int32, Q16.16 centroids
// queries_q16:   (n_queries, d) int32, Q16.16 query vectors
// key_indices:   (d, n_keys) u8, SoA layout
// key_norms_q16: (n_keys,) int32, Q16.16 norms
// scores:        (n_queries, n_keys) float, output (converted from Q16.16)
__global__ void turboquant_dequant_dot_q16(
    const int* __restrict__ centroids_q16,
    const int* __restrict__ queries_q16,
    const unsigned char* __restrict__ key_indices,
    const int* __restrict__ key_norms_q16,
    float* __restrict__ scores,
    int d,
    int n_queries,
    int n_keys
) {
    int qi = blockIdx.y;
    int ki = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= n_queries || ki >= n_keys) return;

    const int* q = queries_q16 + qi * d;

    // Integer accumulation: exact, order-independent.
    // Uses IADD3 on Ada (3-input integer add, 2.51 cyc latency).
    long long acc = 0;  // i64 accumulator

    for (int c = 0; c < d; c++) {
        unsigned char idx = key_indices[c * n_keys + ki];  // SoA coalesced read
        int k_val = centroids_q16[idx];                     // Q16.16 centroid
        int q_val = q[c];                                    // Q16.16 query
        acc += (long long)q_val * (long long)k_val;          // Exact integer multiply
    }

    // Finalize: shift right by 16 to get Q16.16 result, apply norm, convert to float
    int dot_q16 = (int)(acc >> 16);
    long long scaled = (long long)dot_q16 * (long long)key_norms_q16[ki];
    float result = (float)(scaled >> 16) / 65536.0f;  // Q16.16 -> float
    scores[qi * n_keys + ki] = result;
}

}  // extern "C"

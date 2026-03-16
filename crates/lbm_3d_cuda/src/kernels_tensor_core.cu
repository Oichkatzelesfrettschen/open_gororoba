// Tensor Core throughput proxy benchmark kernels (WMMA API).
// These are NOT LBM steps -- they measure raw Tensor Core GFLOPS for different
// precision tiers available on Ada Lovelace (SM 8.9).
//
// Purpose: quantify the throughput gap between custom scalar LBM kernels and
//   the maximum achievable with Tensor Core paths (cuBLAS/WMMA).
//   LBM kernels are bandwidth-bound; Tensor Cores are compute-bound.
//   This benchmark answers: "how much headroom do TC paths have?"
//
// Available WMMA shapes on Ada SM 8.9:
//   TF32:  M=16 N=16 K=8   -- ~165 TFLOPS (TF32 accumulate to FP32)
//   FP16:  M=16 N=16 K=16  -- ~330 TFLOPS (FP16 accumulate to FP32)
//   BF16:  M=16 N=16 K=16  -- ~330 TFLOPS (same as FP16 on Ada)
//   INT8:  M=16 N=16 K=16  -- ~330 TOPS  (accumulate to INT32)
//   INT4:  M=8  N=8  K=32  -- ~660 TOPS  (experimental, accumulate to INT32)
//
// Launch geometry: EXACTLY 1 warp (32 threads) per block, enforced via
//   __launch_bounds__(32, 1).  With 1 warp/block there is no warpId
//   ambiguity: every block executes one WMMA tile and stores to its own
//   output tile at C + blockIdx.x * M * N.
// GFLOPS = 2 * M * N * K * n_iters * n_warps / elapsed_s.
// (Factor of 2 for multiply + add.)

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
using namespace nvcuda;

// ============================================================================
// TF32 Tensor Core proxy (SM 8.0+, dominant path on Ada)
// ============================================================================
// M=16, N=16, K=8  (TF32 operands, FP32 accumulator)
// A: M x K = 16x8, row_major -> ldm = K = 8.
// B: K x N = 8x16, col_major -> ldm = K = 8 (column-major leading dim = #rows = K).

extern "C" __launch_bounds__(32, 1)
__global__ void tensor_core_tf32_proxy(
    const float* A,    // M*K floats (interpreted as tf32 fragments)
    const float* B,    // K*N floats
    float* C,          // M*N float accumulator (input+output)
    int n_iters        // number of WMMA mma_sync calls per warp
) {
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag;

    wmma::load_matrix_sync(a_frag, A, 8);   // A row_major: ldm = K = 8
    wmma::load_matrix_sync(b_frag, B, 8);   // B col_major: ldm = K = 8
    wmma::fill_fragment(c_frag, 0.0f);

    for (int iter = 0; iter < n_iters; iter++) {
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + blockIdx.x * 16 * 16, c_frag, 16, wmma::mem_row_major);
}

// ============================================================================
// FP16 Tensor Core proxy (SM 7.0+)
// ============================================================================
// M=16, N=16, K=16  (half operands, FP32 accumulator)
// A: M x K = 16x16, row_major -> ldm = K = 16.
// B: K x N = 16x16, col_major -> ldm = K = 16.

extern "C" __launch_bounds__(32, 1)
__global__ void tensor_core_fp16_proxy(
    const __half* A,   // M*K halves
    const __half* B,   // K*N halves
    float* C,          // M*N float accumulator
    int n_iters
) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::load_matrix_sync(a_frag, A, 16);  // A row_major: ldm = K = 16
    wmma::load_matrix_sync(b_frag, B, 16);  // B col_major: ldm = K = 16
    wmma::fill_fragment(c_frag, 0.0f);

    for (int iter = 0; iter < n_iters; iter++) {
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + blockIdx.x * 16 * 16, c_frag, 16, wmma::mem_row_major);
}

// ============================================================================
// INT8 Tensor Core proxy (SM 7.5+)
// ============================================================================
// M=16, N=16, K=16  (int8 operands, int32 accumulator)
// A: M x K = 16x16, row_major -> ldm = K = 16.
// B: K x N = 16x16, col_major -> ldm = K = 16.

extern "C" __launch_bounds__(32, 1)
__global__ void tensor_core_int8_proxy(
    const signed char* A,   // M*K int8 values
    const signed char* B,   // K*N int8 values
    int* C,                 // M*N int32 accumulator
    int n_iters
) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag;

    wmma::load_matrix_sync(a_frag, A, 16);  // A row_major: ldm = K = 16
    wmma::load_matrix_sync(b_frag, B, 16);  // B col_major: ldm = K = 16
    wmma::fill_fragment(c_frag, 0);

    for (int iter = 0; iter < n_iters; iter++) {
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + blockIdx.x * 16 * 16, c_frag, 16, wmma::mem_row_major);
}

// ============================================================================
// INT4 Tensor Core proxy (SM 7.5+, experimental)
// ============================================================================
// M=8, N=8, K=32  (4-bit signed operands, int32 accumulator)
// A: M x K = 8x32, row_major -> ldm = K = 32.
// B: K x N = 32x8, col_major -> ldm = K = 32 (column-major leading dim = #rows = K).
// Each element is 4 bits; two elements packed per byte.
// This tier is only accessible via wmma experimental::precision::s4.

extern "C" __launch_bounds__(32, 1)
__global__ void tensor_core_int4_proxy(
    const int* A,    // M*(K/8) packed int4 values (8 int4 per int32)
    const int* B,    // (K/8)*N packed int4 values
    int* C,          // M*N int32 accumulator
    int n_iters
) {
    wmma::fragment<wmma::matrix_a, 8, 8, 32, wmma::experimental::precision::s4, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 8, 8, 32, wmma::experimental::precision::s4, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 8, 8, 32, int> c_frag;

    wmma::load_matrix_sync(a_frag, A, 32);  // A row_major: ldm = K = 32
    wmma::load_matrix_sync(b_frag, B, 32);  // B col_major: ldm = K = 32
    wmma::fill_fragment(c_frag, 0);

    for (int iter = 0; iter < n_iters; iter++) {
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + blockIdx.x * 8 * 8, c_frag, 8, wmma::mem_row_major);
}

// ============================================================================
// BF16 Tensor Core proxy (SM 8.0+, same throughput as FP16 on Ada)
// ============================================================================
// M=16, N=16, K=16  (bfloat16 operands, FP32 accumulator)
// BF16 has 8-bit exponent (same as FP32) and 7-bit mantissa vs FP16's 10-bit.
// On Ada SM 8.9, BF16 TC throughput is identical to FP16 TC (~330 TFLOPS).
// Both use the same 16x16x16 WMMA shape and the same hardware datapath.
// Including BF16 here validates this parity and completes the WMMA suite:
//   TF32, FP16, BF16, INT8, INT4.
// A: M x K = 16x16, row_major -> ldm = K = 16.
// B: K x N = 16x16, col_major -> ldm = K = 16.

extern "C" __launch_bounds__(32, 1)
__global__ void tensor_core_bf16_proxy(
    const __nv_bfloat16* A,  // M*K bfloat16 values
    const __nv_bfloat16* B,  // K*N bfloat16 values
    float* C,                // M*N float accumulator (input+output)
    int n_iters
) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::load_matrix_sync(a_frag, A, 16);  // A row_major: ldm = K = 16
    wmma::load_matrix_sync(b_frag, B, 16);  // B col_major: ldm = K = 16
    wmma::fill_fragment(c_frag, 0.0f);

    for (int iter = 0; iter < n_iters; iter++) {
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + blockIdx.x * 16 * 16, c_frag, 16, wmma::mem_row_major);
}

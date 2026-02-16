#include <cuda_bf16.h>

extern "C" __global__ void test_bf16(__nv_bfloat16* out) {
    *out = __float2bfloat16(1.0f);
}

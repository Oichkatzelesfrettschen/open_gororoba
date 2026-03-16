// INT16 AoS stride-20 D3Q19 LBM kernel.
//
// WHY INT16 over INT8:
//   INT8 DIST_SCALE=64 limits rho to ~1.98. INT16 DIST_SCALE=16384 allows
//   rho up to ~1.9999 with far finer quantization (one LSB = 6.1e-5 vs 1.6e-2).
//   Useful for moderate-Reynolds flows where INT8 saturation causes instability
//   but FP32/BF16 is bandwidth-overkill. 2 bytes/dist = same as FP16 AoS.
//
// AoS stride-20: same non-coalescing penalty as FP16 AoS for diagonal directions.
//   Use kernels_int16_soa.cu for coalesced pull-scheme variant.
//
// Vectorized loads: 5x int2 (8 bytes each) = 40 bytes = 20 shorts (indices 0-18
//   + 1 pad short). AoS array: f[idx * 20 + dir], short elements.
//
// DIST_SCALE = 16384: range [-32768, 32767] -> f_i in [-2.0000, 1.9999].
//   Rest eq (w=1/3, rho=1): 1/3 * 16384 = 5461.3 -> stored 5461, error 4.6e-5.
//   Face eq (w=1/18): 910.2 -> 910, error 2.2e-4. Edge (w=1/36): 455.1 -> 455.
// VRAM at 128^3: 20 * 2,097,152 * 2 * 2 (ping+pong) = ~160 MB (same as FP16 AoS).

#define DIST_SCALE_I16 16384.0f
#define INV_DIST_SCALE_I16 (1.0f / 16384.0f)

__constant__ int CX_I16[19] = {
    0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0
};
__constant__ int CY_I16[19] = {
    0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1
};
__constant__ int CZ_I16[19] = {
    0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1
};

__constant__ float W_I16[19] = {
    1.0f/3.0f,
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
};

__device__ __forceinline__ bool finite_i16(float x) {
    return (x == x) && (x <= 3.402823466e38f) && (x >= -3.402823466e38f);
}

__device__ __forceinline__ short float_to_i16(float v) {
    if (!finite_i16(v)) return (short)0;
    float s = v * DIST_SCALE_I16;
    if (s >  32767.0f) s =  32767.0f;
    if (s < -32768.0f) s = -32768.0f;
    return (short)(int)s;
}

// INT16 AoS push-scheme kernel.
// Loads f[idx*20+dir] via 5x int2 vectorized reads (4 shorts per load).
// Computes BGK in FP32, pushes to f_out[idx_next*20+dir].
// NOTE: force before tau to match BenchKernelRunner::step_n() calling convention.
extern "C" __launch_bounds__(128, 4) __global__ void lbm_step_int16_kernel(
    const short* __restrict__ f_in,
    short* __restrict__ f_out,
    float* __restrict__ rho_out,
    float* __restrict__ u_out,         // [3 * n_cells] SoA
    const float* __restrict__ force,   // [3 * n_cells] SoA  (arg 4)
    const float* __restrict__ tau,     //                    (arg 5)
    int nx, int ny, int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_cells = nx * ny * nz;
    if (idx >= n_cells) return;

    int x = idx % nx;
    int y = (idx / nx) % ny;
    int z = idx / (nx * ny);

    // 5x int2 vectorized loads: each int2 = 4 bytes = 2 shorts.
    // Covers indices [0..3], [4..7], [8..11], [12..15], [16..19].
    float f_local[19];
    {
        const int2* base = reinterpret_cast<const int2*>(f_in + (long long)idx * 20);
        #pragma unroll
        for (int k = 0; k < 5; k++) {
            int2 v = __ldg(base + k);
            short s0 = (short)(v.x & 0xFFFF);
            short s1 = (short)(v.x >> 16);
            short s2 = (short)(v.y & 0xFFFF);
            short s3 = (short)(v.y >> 16);
            int base_dir = k * 4;
            if (base_dir     < 19) f_local[base_dir    ] = (float)s0 * INV_DIST_SCALE_I16;
            if (base_dir + 1 < 19) f_local[base_dir + 1] = (float)s1 * INV_DIST_SCALE_I16;
            if (base_dir + 2 < 19) f_local[base_dir + 2] = (float)s2 * INV_DIST_SCALE_I16;
            if (base_dir + 3 < 19) f_local[base_dir + 3] = (float)s3 * INV_DIST_SCALE_I16;
        }
    }

    float rho_local = 0.0f;
    float mx = 0.0f, my = 0.0f, mz = 0.0f;
    #pragma unroll
    for (int i = 0; i < 19; i++) {
        float fi = f_local[i];
        if (!finite_i16(fi)) fi = 0.0f;
        f_local[i] = fi;
        rho_local += fi;
        mx += (float)CX_I16[i] * fi;
        my += (float)CY_I16[i] * fi;
        mz += (float)CZ_I16[i] * fi;
    }

    float ux = 0.0f, uy = 0.0f, uz = 0.0f;
    if (finite_i16(rho_local) && rho_local > 1.0e-20f) {
        float inv_rho = 1.0f / rho_local;
        ux = mx * inv_rho;
        uy = my * inv_rho;
        uz = mz * inv_rho;
    } else {
        rho_local = 1.0f;
    }

    rho_out[idx] = rho_local;
    u_out[idx]               = ux;
    u_out[n_cells + idx]     = uy;
    u_out[2 * n_cells + idx] = uz;

    float tau_local = __ldg(&tau[idx]);
    float inv_tau   = 1.0f / tau_local;
    float u_sq      = ux * ux + uy * uy + uz * uz;
    float base      = fmaf(-1.5f, u_sq, 1.0f);

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        float eu   = fmaf((float)CX_I16[i], ux, fmaf((float)CY_I16[i], uy, (float)CZ_I16[i] * uz));
        float w_rho = W_I16[i] * rho_local;
        float f_eq  = w_rho * fmaf(fmaf(eu, 4.5f, 3.0f), eu, base);
        f_local[i] -= (f_local[i] - f_eq) * inv_tau;
    }

    float fx = __ldg(&force[idx]);
    float fy = __ldg(&force[n_cells + idx]);
    float fz = __ldg(&force[2 * n_cells + idx]);
    float force_mag_sq = fx * fx + fy * fy + fz * fz;

    if (force_mag_sq >= 1e-40f) {
        float prefactor = 1.0f - 0.5f * inv_tau;
        #pragma unroll
        for (int i = 0; i < 19; i++) {
            float eix = (float)CX_I16[i], eiy = (float)CY_I16[i], eiz = (float)CZ_I16[i];
            float em_u_dot_f = (eix - ux) * fx + (eiy - uy) * fy + (eiz - uz) * fz;
            float ei_dot_u   = eix * ux + eiy * uy + eiz * uz;
            float ei_dot_f   = eix * fx + eiy * fy + eiz * fz;
            f_local[i] += prefactor * W_I16[i] * (em_u_dot_f * 3.0f + ei_dot_u * ei_dot_f * 9.0f);
        }
    }

    // AoS push: scatter-write to neighbor cell
    #pragma unroll
    for (int i = 0; i < 19; i++) {
        int xn = (x + CX_I16[i] + nx) % nx;
        int yn = (y + CY_I16[i] + ny) % ny;
        int zn = (z + CZ_I16[i] + nz) % nz;
        int dst = xn + nx * (yn + ny * zn);
        f_out[(long long)dst * 20 + i] = float_to_i16(f_local[i]);
    }
}

extern "C" __global__ void initialize_uniform_int16_kernel(
    short* f,
    float* rho_out,
    float* u_out,
    float* tau,
    float rho_init,
    float ux_init,
    float uy_init,
    float uz_init,
    float tau_val,
    int nx, int ny, int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_cells = nx * ny * nz;
    if (idx >= n_cells) return;

    rho_out[idx] = rho_init;
    u_out[idx]               = ux_init;
    u_out[n_cells + idx]     = uy_init;
    u_out[2 * n_cells + idx] = uz_init;
    tau[idx] = tau_val;

    float u_sq = ux_init*ux_init + uy_init*uy_init + uz_init*uz_init;
    float base = fmaf(-1.5f, u_sq, 1.0f);

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        float eu   = (float)CX_I16[i]*ux_init + (float)CY_I16[i]*uy_init + (float)CZ_I16[i]*uz_init;
        float f_eq = W_I16[i] * rho_init * fmaf(fmaf(eu, 4.5f, 3.0f), eu, base);
        f[(long long)idx * 20 + i] = float_to_i16(f_eq);
    }
    // Zero padding slot 19
    f[(long long)idx * 20 + 19] = 0;
}

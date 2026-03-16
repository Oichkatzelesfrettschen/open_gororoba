// INT8 fixed-point D3Q19 LBM kernel.
// Storage: int8_t / signed char (1 byte/value, AoS, stride 20 per cell).
// Compute: FP32 promoted immediately after dequantize.
// Scale factor: DIST_SCALE = 64.0  (f_int8 = clamp(f_float * 64, -128, 127))
//   -- distributions are typically [0, 1/3]; w_i * rho * max = 1/3 -> int8 = 85.
//   -- leaving room for collision transients up to f_i = 1.98 before clamping.
// YSU dp4a trick: __dp4a() computes dot4(a_i8x4, b_i8x4) + acc in one instruction.
//   Applied to the momentum sum (sum_i cx[i]*f[i]) for groups of 4 distributions.
//   D3Q19 cx has only {-1, 0, 1} -> products fit in int8 exactly -> __dp4a valid.
// Bandwidth: 4x reduction vs FP32 (1 byte vs 4 bytes per distribution scalar).
// Alignment: stride 20 bytes per cell ensures 4-byte alignment for all idx
//   (20 % 4 == 0); 5 int32 loads cover all 20 bytes (indices 0-18 active,
//   index 19 is unused padding).
// VRAM at 128^3: 20 * 2,097,152 * 1 * 2 (ping+pong) = ~80 MB.

#define DIST_SCALE 64.0f
#define INV_DIST_SCALE (1.0f / DIST_SCALE)

// D3Q19 lattice velocities (suffixed _I8)
__constant__ int D3Q19_CX_I8[19] = {
    0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0
};
__constant__ int D3Q19_CY_I8[19] = {
    0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1
};
__constant__ int D3Q19_CZ_I8[19] = {
    0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1
};

__constant__ float D3Q19_WF_I8[19] = {
    1.0f/3.0f,
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
};

// Precomputed D3Q19 velocities as int8 for __dp4a
// Groups of 4 for __dp4a:  [0..3], [4..7], [8..11], [12..15]; remainder [16..18]
__constant__ signed char DP4A_CX[20] = {
    0, 1,-1, 0,  0, 0, 0, 1,  -1, 1,-1, 1,  -1, 1,-1, 0,  0, 0, 0, 0
};
__constant__ signed char DP4A_CY[20] = {
    0, 0, 0, 1, -1, 0, 0, 1,  -1,-1, 1, 0,   0, 0, 0, 1, -1, 1,-1, 0
};
__constant__ signed char DP4A_CZ[20] = {
    0, 0, 0, 0,  0, 1,-1, 0,   0, 0, 0, 1,  -1,-1, 1, 1, -1,-1, 1, 0
};

__device__ __forceinline__ bool finite_f32_i8(float x) {
    return (x == x) && (x <= 3.402823466e38f) && (x >= -3.402823466e38f);
}

// Pack 4 signed chars into an int32 for __dp4a.
__device__ __forceinline__ int pack_i8x4(signed char a, signed char b,
                                          signed char c, signed char d) {
    int r;
    unsigned char* p = (unsigned char*)&r;
    p[0] = (unsigned char)a;
    p[1] = (unsigned char)b;
    p[2] = (unsigned char)c;
    p[3] = (unsigned char)d;
    return r;
}

// Equilibrium in FP32 with Horner FMA form.
__device__ void compute_equilibrium_i8(float* f_eq, float rho, const float* u) {
    float u_sq = u[0]*u[0] + u[1]*u[1] + u[2]*u[2];
    float base = fmaf(-1.5f, u_sq, 1.0f);
    #pragma unroll
    for (int i = 0; i < 19; i++) {
        float eu = (float)D3Q19_CX_I8[i]*u[0]
                 + (float)D3Q19_CY_I8[i]*u[1]
                 + (float)D3Q19_CZ_I8[i]*u[2];
        float w_rho = D3Q19_WF_I8[i] * rho;
        f_eq[i] = w_rho * fmaf(fmaf(eu, 4.5f, 3.0f), eu, base);
    }
}

// Saturate-clamp float to INT8 range after scaling.
// NaN and Inf map to 0 to avoid undefined cast behavior.
__device__ __forceinline__ signed char float_to_i8(float v) {
    if (!finite_f32_i8(v)) return (signed char)0;
    float scaled = v * DIST_SCALE;
    if (scaled > 127.0f)  scaled = 127.0f;
    if (scaled < -128.0f) scaled = -128.0f;
    return (signed char)(int)scaled;
}

// Fused collision + streaming with INT8 storage, FP32 compute.
// dp4a applied to momentum accumulation (cx/cy/cz dot product with f values).
// Per-cell stride: 20 bytes (index 19 is padding, never accessed after init).
// __launch_bounds__(128, 4): target 4 blocks/SM; dp4a groups add to register count.
// This hint keeps the compiler from allocating more than ~128 regs/thread, preserving
// adequate occupancy (4 active blocks per SM = 512 threads = 50% warp capacity).
extern "C" __launch_bounds__(128, 4) __global__ void lbm_step_fused_int8_kernel(
    const signed char* f_in,   // n_cells * 20 int8 values (stride 20, index 19 unused)
    signed char* f_out,
    float* rho_out,
    float* u_out,
    const float* force,
    const float* tau,
    int nx, int ny, int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_cells = nx * ny * nz;
    if (idx >= n_cells) return;

    int x = idx % nx;
    int y = (idx / nx) % ny;
    int z = idx / (nx * ny);

    // Stride 20: f_base is at idx*20 bytes, always 4-byte aligned (20 % 4 == 0).
    const signed char* f_base = f_in + (long long)idx * 20;

    // 5 int32 packed loads = 20 bytes; indices 0-18 are active, index 19 is padding.
    signed char f_i8[20];
    #pragma unroll
    for (int j = 0; j < 5; j++) {
        int packed = *((const int*)(f_base + j * 4));
        f_i8[j*4 + 0] = (signed char)(packed & 0xFF);
        f_i8[j*4 + 1] = (signed char)((packed >> 8) & 0xFF);
        f_i8[j*4 + 2] = (signed char)((packed >> 16) & 0xFF);
        f_i8[j*4 + 3] = (signed char)((packed >> 24) & 0xFF);
    }

    float f_local[19];
    #pragma unroll
    for (int i = 0; i < 19; i++) {
        f_local[i] = (float)f_i8[i] * INV_DIST_SCALE;
    }

    // Macroscopic density: simple sum
    float rho_local = 0.0f;
    #pragma unroll
    for (int i = 0; i < 19; i++) rho_local += f_local[i];

    // Momentum via __dp4a: accumulates sum(cx[i]*f_i8[i]) in integer domain.
    // 5 groups of 4 (indices 0-19); index 19 is padding (dp4a includes it,
    // but DP4A_CX/CY/CZ[19] = 0, so the contribution is zero by construction).
    int mx_i32 = 0, my_i32 = 0, mz_i32 = 0;
    #pragma unroll
    for (int j = 0; j < 5; j++) {
        int f_pack = pack_i8x4(f_i8[j*4], f_i8[j*4+1], f_i8[j*4+2], f_i8[j*4+3]);
        int cx_pack = pack_i8x4(DP4A_CX[j*4], DP4A_CX[j*4+1], DP4A_CX[j*4+2], DP4A_CX[j*4+3]);
        int cy_pack = pack_i8x4(DP4A_CY[j*4], DP4A_CY[j*4+1], DP4A_CY[j*4+2], DP4A_CY[j*4+3]);
        int cz_pack = pack_i8x4(DP4A_CZ[j*4], DP4A_CZ[j*4+1], DP4A_CZ[j*4+2], DP4A_CZ[j*4+3]);
        mx_i32 = __dp4a(f_pack, cx_pack, mx_i32);
        my_i32 = __dp4a(f_pack, cy_pack, my_i32);
        mz_i32 = __dp4a(f_pack, cz_pack, mz_i32);
    }

    float mx = (float)mx_i32 * INV_DIST_SCALE;
    float my = (float)my_i32 * INV_DIST_SCALE;
    float mz = (float)mz_i32 * INV_DIST_SCALE;

    float ux = 0.0f, uy = 0.0f, uz = 0.0f;
    if (finite_f32_i8(rho_local) && rho_local > 1.0e-20f) {
        float inv_rho = 1.0f / rho_local;
        ux = mx * inv_rho;
        uy = my * inv_rho;
        uz = mz * inv_rho;
    } else {
        rho_local = 1.0f;
    }

    rho_out[idx] = rho_local;
    u_out[idx * 3 + 0] = ux;
    u_out[idx * 3 + 1] = uy;
    u_out[idx * 3 + 2] = uz;

    // Collision + Guo forcing (FP32)
    float f_eq[19];
    float u_vec[3] = {ux, uy, uz};
    compute_equilibrium_i8(f_eq, rho_local, u_vec);

    float tau_local = tau[idx];
    float inv_tau = 1.0f / tau_local;
    float prefactor = 1.0f - 0.5f * inv_tau;
    float fx = force[idx * 3 + 0];
    float fy = force[idx * 3 + 1];
    float fz = force[idx * 3 + 2];

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        float fi = f_local[i] - (f_local[i] - f_eq[i]) * inv_tau;
        float eix = (float)D3Q19_CX_I8[i];
        float eiy = (float)D3Q19_CY_I8[i];
        float eiz = (float)D3Q19_CZ_I8[i];
        float s_i = ((eix - ux)*fx + (eiy - uy)*fy + (eiz - uz)*fz) * 3.0f
                  + (eix*ux + eiy*uy + eiz*uz) * (eix*fx + eiy*fy + eiz*fz) * 9.0f;
        fi += prefactor * D3Q19_WF_I8[i] * s_i;

        // Streaming (stride 20)
        int x_next = (x + D3Q19_CX_I8[i] + nx) % nx;
        int y_next = (y + D3Q19_CY_I8[i] + ny) % ny;
        int z_next = (z + D3Q19_CZ_I8[i] + nz) % nz;
        long long idx_next = (long long)x_next + nx * ((long long)y_next + ny * z_next);
        f_out[idx_next * 20 + i] = float_to_i8(fi);
    }
}

extern "C" __global__ void initialize_uniform_int8_kernel(
    signed char* f,
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
    u_out[idx * 3 + 0] = ux_init;
    u_out[idx * 3 + 1] = uy_init;
    u_out[idx * 3 + 2] = uz_init;
    tau[idx] = tau_val;

    float u_local[3] = {ux_init, uy_init, uz_init};
    float f_eq[19];
    compute_equilibrium_i8(f_eq, rho_init, u_local);

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        f[idx * 20 + i] = float_to_i8(f_eq[i]);
    }
    // Padding slot: write zero so reads of slot 19 are well-defined.
    f[idx * 20 + 19] = 0;
}

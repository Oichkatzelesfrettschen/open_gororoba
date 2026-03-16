// FP4 E2M1 nibble-packed D3Q19 bandwidth ceiling benchmark.
//
// IMPORTANT: FP4 E2M1 is NOT physically viable for D3Q19 LBM.
//   Format: 1 sign + 2 exponent + 1 mantissa bit (NV FP4 E2M1 spec, Blackwell).
//   POSITIVE VALUES: {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
//   Distribution accuracy analysis:
//     Rest weight   (1/3  ~ 0.333): FP4 rounds to 0.5 (50% error).
//     Face weight   (1/18 ~ 0.056): FP4 rounds to 0.5 (9x overestimate!).
//     Edge weight   (1/36 ~ 0.028): FP4 rounds to 0.0 or 0.5 (catastrophic).
//   Physics is broken worse than INT4 (which at least has a continuous scale).
//   Ada SM 8.9 has NO hardware FP4 intrinsics (Blackwell SM 10.0 only).
//   We emulate FP4 E2M1 via nibble pack/unpack with CPU-side scalar conversion.
//
// USE CASE: bandwidth ceiling + FP4 decode overhead measurement.
//   Answers: "If Blackwell FP4 hardware existed for Ada, what BW ceiling applies?"
//   Establishes comparison point to INT4 (same nibble layout, different encoding).
//   Expected MLUPS slightly lower than INT4 due to E2M1 decode overhead vs
//   integer unpack (extra shift/mask/float ops vs simple scale).
//
// Storage: 2 nibbles per byte, i-major SoA. Identical to INT4 nibble layout.
//   Buffer size: 19 * ceil(n_cells/2) bytes per buffer.
//   VRAM at 128^3: 19 * 1,048,576 * 1 * 2 (ping+pong) = ~38 MB.
//
// Race mitigation: same as INT4 -- thread k handles cells 2k and 2k+1
//   into the same byte, eliminating concurrent RMW to adjacent threads.

#define DIST_SCALE_FP4 1.0f   // FP4 stores actual float representation

// D3Q19 lattice velocities (suffixed _FP4)
__constant__ int CX_FP4[19] = {
    0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0
};
__constant__ int CY_FP4[19] = {
    0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1
};
__constant__ int CZ_FP4[19] = {
    0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1
};

__constant__ float W_FP4[19] = {
    1.0f/3.0f,
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
};

// FP4 E2M1 decode table (4-bit signed index -> float value).
// Positive: 0=0.0, 1=0.5, 2=1.0, 3=1.5, 4=2.0, 5=3.0, 6=4.0, 7=6.0
// Negative: 8=-0.0, 9=-0.5, 10=-1.0, 11=-1.5, 12=-2.0, 13=-3.0, 14=-4.0, 15=-6.0
__constant__ float FP4_DECODE[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
};

// FP4 E2M1 encode: quantize float to nearest FP4 representable value (nibble 0-15).
__device__ __forceinline__ unsigned char float_to_fp4(float v) {
    // Map to absolute value + sign
    unsigned char sign_bit = (v < 0.0f) ? 8u : 0u;
    float av = fabsf(v);
    // Clamp and quantize to {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
    unsigned char mag;
    if      (av < 0.25f) mag = 0u;         // -> 0.0
    else if (av < 0.75f) mag = 1u;         // -> 0.5
    else if (av < 1.25f) mag = 2u;         // -> 1.0
    else if (av < 1.75f) mag = 3u;         // -> 1.5
    else if (av < 2.50f) mag = 4u;         // -> 2.0
    else if (av < 3.50f) mag = 5u;         // -> 3.0
    else if (av < 5.00f) mag = 6u;         // -> 4.0
    else                 mag = 7u;         // -> 6.0 (clamp)
    return sign_bit | mag;
}

// FP4 E2M1 bandwidth ceiling kernel.
// Thread k handles cells 2k and 2k+1.
// Grid: ceil(n_cells/2) / 128 blocks.
extern "C" __launch_bounds__(128, 4) __global__ void lbm_step_fp4_kernel(
    const unsigned char* __restrict__ f_in,   // [19 * ceil(n_cells/2)], nibble-packed
    unsigned char* __restrict__ f_out,        // [19 * ceil(n_cells/2)], nibble-packed
    float* __restrict__ rho_out,              // [n_cells]
    float* __restrict__ u_out,               // [3 * n_cells]
    const float* __restrict__ tau,
    const float* __restrict__ force,
    int nx, int ny, int nz
) {
    int half_cells = (nx * ny * nz + 1) / 2;
    int k = blockIdx.x * blockDim.x + threadIdx.x;   // byte index
    if (k >= half_cells) return;

    int n_cells = nx * ny * nz;
    int idx0 = 2 * k;       // first cell
    int idx1 = 2 * k + 1;   // second cell (may be == n_cells for odd grids)
    bool valid1 = (idx1 < n_cells);

    int x0 = idx0 % nx, y0 = (idx0 / nx) % ny, z0 = idx0 / (nx * ny);
    int x1 = 0, y1 = 0, z1 = 0;
    if (valid1) { x1 = idx1 % nx; y1 = (idx1 / nx) % ny; z1 = idx1 / (nx * ny); }

    float f0[19], f1[19];
    float rho0 = 0.0f, rho1 = 0.0f;
    float mx0 = 0.0f, my0 = 0.0f, mz0 = 0.0f;
    float mx1 = 0.0f, my1 = 0.0f, mz1 = 0.0f;

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        // Backward neighbor for cell 0
        int xb0 = (x0 - CX_FP4[i] + nx) % nx;
        int yb0 = (y0 - CY_FP4[i] + ny) % ny;
        int zb0 = (z0 - CZ_FP4[i] + nz) % nz;
        int back0 = xb0 + nx * (yb0 + ny * zb0);

        unsigned char byte0 = __ldg(&f_in[(long long)i * half_cells + back0 / 2]);
        float fi0 = FP4_DECODE[(back0 & 1) ? (byte0 >> 4) & 0xF : byte0 & 0xF];
        f0[i] = fi0;
        rho0 += fi0;
        mx0 += (float)CX_FP4[i] * fi0;
        my0 += (float)CY_FP4[i] * fi0;
        mz0 += (float)CZ_FP4[i] * fi0;

        if (valid1) {
            int xb1 = (x1 - CX_FP4[i] + nx) % nx;
            int yb1 = (y1 - CY_FP4[i] + ny) % ny;
            int zb1 = (z1 - CZ_FP4[i] + nz) % nz;
            int back1 = xb1 + nx * (yb1 + ny * zb1);
            unsigned char byte1 = __ldg(&f_in[(long long)i * half_cells + back1 / 2]);
            float fi1 = FP4_DECODE[(back1 & 1) ? (byte1 >> 4) & 0xF : byte1 & 0xF];
            f1[i] = fi1;
            rho1 += fi1;
            mx1 += (float)CX_FP4[i] * fi1;
            my1 += (float)CY_FP4[i] * fi1;
            mz1 += (float)CZ_FP4[i] * fi1;
        }
    }

    // Macroscopic quantities and BGK collision for cell 0
    float ux0 = 0.0f, uy0 = 0.0f, uz0 = 0.0f;
    if (rho0 > 1.0e-20f) { float ir = 1.0f / rho0; ux0 = mx0 * ir; uy0 = my0 * ir; uz0 = mz0 * ir; }
    else rho0 = 1.0f;
    rho_out[idx0] = rho0;
    u_out[idx0] = ux0; u_out[n_cells + idx0] = uy0; u_out[2 * n_cells + idx0] = uz0;

    float tau0 = __ldg(&tau[idx0]);
    float inv_tau0 = 1.0f / tau0;
    float usq0 = ux0*ux0 + uy0*uy0 + uz0*uz0;
    float base0 = fmaf(-1.5f, usq0, 1.0f);

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        float eu = fmaf((float)CX_FP4[i], ux0, fmaf((float)CY_FP4[i], uy0, (float)CZ_FP4[i] * uz0));
        float f_eq = W_FP4[i] * rho0 * fmaf(fmaf(eu, 4.5f, 3.0f), eu, base0);
        f0[i] -= (f0[i] - f_eq) * inv_tau0;
    }

    // BGK collision for cell 1
    float ux1 = 0.0f, uy1 = 0.0f, uz1 = 0.0f;
    if (valid1) {
        if (rho1 > 1.0e-20f) { float ir = 1.0f / rho1; ux1 = mx1 * ir; uy1 = my1 * ir; uz1 = mz1 * ir; }
        else rho1 = 1.0f;
        rho_out[idx1] = rho1;
        u_out[idx1] = ux1; u_out[n_cells + idx1] = uy1; u_out[2 * n_cells + idx1] = uz1;

        float tau1 = __ldg(&tau[idx1]);
        float inv_tau1 = 1.0f / tau1;
        float usq1 = ux1*ux1 + uy1*uy1 + uz1*uz1;
        float base1 = fmaf(-1.5f, usq1, 1.0f);

        #pragma unroll
        for (int i = 0; i < 19; i++) {
            float eu = fmaf((float)CX_FP4[i], ux1, fmaf((float)CY_FP4[i], uy1, (float)CZ_FP4[i] * uz1));
            float f_eq = W_FP4[i] * rho1 * fmaf(fmaf(eu, 4.5f, 3.0f), eu, base1);
            f1[i] -= (f1[i] - f_eq) * inv_tau1;
        }
    }

    // Pack and write: thread k owns byte k in each direction's nibble buffer.
    // lo nibble = cell 2k (idx0), hi nibble = cell 2k+1 (idx1).
    #pragma unroll
    for (int i = 0; i < 19; i++) {
        unsigned char lo = float_to_fp4(f0[i]);
        unsigned char hi = valid1 ? float_to_fp4(f1[i]) : 0u;
        f_out[(long long)i * half_cells + k] = lo | (hi << 4);
    }
}

extern "C" __global__ void initialize_uniform_fp4_kernel(
    unsigned char* f,
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
    int half_cells = (nx * ny * nz + 1) / 2;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= half_cells) return;

    int n_cells = nx * ny * nz;
    int idx0 = 2 * k;
    int idx1 = 2 * k + 1;
    bool valid1 = (idx1 < n_cells);

    rho_out[idx0] = rho_init;
    u_out[idx0] = ux_init; u_out[n_cells + idx0] = uy_init; u_out[2 * n_cells + idx0] = uz_init;
    tau[idx0] = tau_val;
    if (valid1) {
        rho_out[idx1] = rho_init;
        u_out[idx1] = ux_init; u_out[n_cells + idx1] = uy_init; u_out[2 * n_cells + idx1] = uz_init;
        tau[idx1] = tau_val;
    }

    float u_sq = ux_init*ux_init + uy_init*uy_init + uz_init*uz_init;
    float base = fmaf(-1.5f, u_sq, 1.0f);

    #pragma unroll
    for (int i = 0; i < 19; i++) {
        float eu   = (float)CX_FP4[i]*ux_init + (float)CY_FP4[i]*uy_init + (float)CZ_FP4[i]*uz_init;
        float f_eq = W_FP4[i] * rho_init * fmaf(fmaf(eu, 4.5f, 3.0f), eu, base);
        unsigned char lo = float_to_fp4(f_eq);
        unsigned char hi = valid1 ? lo : 0u;  // both cells same equilibrium at init
        f[(long long)i * half_cells + k] = lo | (hi << 4);
    }
}

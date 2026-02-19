// CUDA Gradient-Index (GRIN) Ray Marcher for Algebraic Lensing
// Optimized for SM89 (Ada Lovelace)
// Uses Manual Trilinear Interpolation for maximum control and portability.

#include <cuda_runtime.h>

__device__ __forceinline__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 operator*(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 normalize(float3 v) {
    float inv_n = rsqrtf(dot(v, v) + 1e-30f);
    return v * inv_n;
}

// Manual trilinear interpolation from flat density buffer
__device__ __forceinline__ float sample_density(
    const float* density_field,
    int3 grid_size,
    float3 p
) {
    // Periodic boundary wrap
    float x = fmodf(p.x, (float)grid_size.x); if (x < 0) x += grid_size.x;
    float y = fmodf(p.y, (float)grid_size.y); if (y < 0) y += grid_size.y;
    float z = fmodf(p.z, (float)grid_size.z); if (z < 0) z += grid_size.z;

    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int z0 = (int)floorf(z);

    int x1 = (x0 + 1) % grid_size.x;
    int y1 = (y0 + 1) % grid_size.y;
    int z1 = (z0 + 1) % grid_size.z;

    float u = x - floorf(x);
    float v = y - floorf(y);
    float w = z - floorf(z);

    auto get = [&](int ix, int iy, int iz) {
        return density_field[iz * (grid_size.x * grid_size.y) + iy * grid_size.x + ix];
    };

    float c000 = get(x0, y0, z0);
    float c100 = get(x1, y0, z0);
    float c010 = get(x0, y1, z0);
    float c110 = get(x1, y1, z0);
    float c001 = get(x0, y0, z1);
    float c101 = get(x1, y0, z1);
    float c011 = get(x0, y1, z1);
    float c111 = get(x1, y1, z1);

    float i00 = c000 * (1.0f - u) + c100 * u;
    float i10 = c010 * (1.0f - u) + c110 * u;
    float i01 = c001 * (1.0f - u) + c101 * u;
    float i11 = c011 * (1.0f - u) + c111 * u;

    float i0 = i00 * (1.0f - v) + i10 * v;
    float i1 = i01 * (1.0f - v) + i11 * v;

    return i0 * (1.0f - w) + i1 * w;
}

__device__ __forceinline__ float sample_n(
    const float* density_field,
    int3 grid_size,
    float3 p,
    float alpha
) {
    float density = sample_density(density_field, grid_size, p);
    return 1.0f + alpha * density;
}

__device__ __forceinline__ float3 get_gradient_n(
    const float* density_field,
    int3 grid_size,
    float3 p,
    float alpha,
    float eps,
    float& n_out
) {
    float3 grad;
    grad.x = (sample_n(density_field, grid_size, make_float3(p.x + eps, p.y, p.z), alpha) - 
              sample_n(density_field, grid_size, make_float3(p.x - eps, p.y, p.z), alpha)) / (2.0f * eps);
    grad.y = (sample_n(density_field, grid_size, make_float3(p.x, p.y + eps, p.z), alpha) - 
              sample_n(density_field, grid_size, make_float3(p.x, p.y - eps, p.z), alpha)) / (2.0f * eps);
    grad.z = (sample_n(density_field, grid_size, make_float3(p.x, p.y, p.z + eps), alpha) - 
              sample_n(density_field, grid_size, make_float3(p.x, p.y, p.z - eps), alpha)) / (2.0f * eps);

    n_out = sample_n(density_field, grid_size, p, alpha);
    return grad;
}

__device__ __forceinline__ void get_derivatives(
    const float* density_field,
    int3 grid_size,
    float3 p,
    float3 d,
    float alpha,
    float eps,
    float3& dp_ds,
    float3& dd_ds
) {
    float n;
    float3 grad_n = get_gradient_n(density_field, grid_size, p, alpha, eps, n);
    float d_dot_grad = dot(d, grad_n);
    
    dp_ds = d;
    dd_ds = (grad_n - d * d_dot_grad) * (1.0f / n);
}

extern "C" __global__ void trace_rays_kernel(
    const float* density_field,
    int nx, int ny, int nz,
    const float3* initial_pos,
    const float3* initial_dir,
    float3* final_pos,
    float3* final_dir,
    int n_rays,
    float alpha,
    float dt,
    int max_steps,
    float eps_grad
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rays) return;

    float3 p = initial_pos[idx];
    float3 d = initial_dir[idx];
    int3 grid_size = make_int3(nx, ny, nz);

    for (int i = 0; i < max_steps; i++) {
        float3 k1_p, k1_d;
        get_derivatives(density_field, grid_size, p, d, alpha, eps_grad, k1_p, k1_d);

        float3 p2 = p + k1_p * (dt * 0.5f);
        float3 d2 = normalize(d + k1_d * (dt * 0.5f));
        float3 k2_p, k2_d;
        get_derivatives(density_field, grid_size, p2, d2, alpha, eps_grad, k2_p, k2_d);

        float3 p3 = p + k2_p * (dt * 0.5f);
        float3 d3 = normalize(d + k2_d * (dt * 0.5f));
        float3 k3_p, k3_d;
        get_derivatives(density_field, grid_size, p3, d3, alpha, eps_grad, k3_p, k3_d);

        float3 p4 = p + k3_p * dt;
        float3 d4 = normalize(d + k3_d * dt);
        float3 k4_p, k4_d;
        get_derivatives(density_field, grid_size, p4, d4, alpha, eps_grad, k4_p, k4_d);

        p = p + (k1_p + k2_p * 2.0f + k3_p * 2.0f + k4_p) * (dt / 6.0f);
        d = normalize(d + (k1_d + k2_d * 2.0f + k3_d * 2.0f + k4_d) * (dt / 6.0f));

        // Boundary exit (optional for periodic, but good for termination)
        if (p.x < -100.0f || p.x > 200.0f || p.y < -100.0f || p.y > 200.0f || p.z < -100.0f || p.z > 200.0f) {
            break;
        }
    }

    final_pos[idx] = p;
    final_dir[idx] = d;
}

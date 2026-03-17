// OptiX 9.1 tracer particle advection through LBM density field.
//
// Architecture (coarse 8^3 brick BVH + trilinear velocity interpolation):
//   1. CUDA kernel scans density field -> compact array of 8^3 brick AABBs
//   2. OptiX builds BVH (GAS) over ~2K coarse brick AABBs (microsecond build)
//   3. Raygen launches one ray per particle along velocity vector
//   4. RT cores traverse BVH at hardware speed (Box Intersection Engine)
//   5. Closest-hit reads velocity field via SBT zero-copy pointer
//   6. Trilinear interpolation at exact hit coordinate -> updated particle velocity
//   7. Particle position updated: pos += vel * dt
//
// BVH is amortized: rebuilt every bvh_rebuild_interval steps (e.g., 20).
// Between rebuilds, optixAccelRefit handles minor AABB shifts.
//
// Zero-copy interop: LBM macroscopic buffers (rho, u) are passed as raw
// CUdeviceptr in the SBT record. No host-device copies.

#include <optix.h>

// ---------------------------------------------------------------------------
// SBT data record -- zero-copy access to LBM macroscopic fields
// ---------------------------------------------------------------------------
// This struct is embedded in the HitGroup SBT record. The Rust host fills it
// with raw CUdeviceptr from the LBM solver's device buffers.
struct LbmSbtData {
    // Velocity field: SoA layout u_out[3 * N], where N = nx*ny*nz.
    // u_out[idx] = ux, u_out[N+idx] = uy, u_out[2N+idx] = uz.
    float* velocity_field;  // CUdeviceptr cast to float*
    float* density_field;   // CUdeviceptr cast to float*
    int nx, ny, nz;
    float dx, dy, dz;       // cell spacing in world coords
    float origin_x, origin_y, origin_z;
    int brick_size;          // typically 8
};

// ---------------------------------------------------------------------------
// Launch parameters (constant memory, set per optixLaunch)
// ---------------------------------------------------------------------------
struct LaunchParams {
    float3* positions;      // [n_particles] current positions (input/output)
    float3* velocities;     // [n_particles] output: interpolated velocity at hit
    int*    brick_indices;  // [n_particles] output: brick index (-1 if miss)

    OptixTraversableHandle traversable;
    unsigned int n_particles;
    float dt;
};

extern "C" __constant__ LaunchParams params;

// ---------------------------------------------------------------------------
// Trilinear interpolation of SoA velocity field at world coordinate (wx,wy,wz)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float3 trilinear_velocity(
    const LbmSbtData* sbt,
    float wx, float wy, float wz
) {
    // Convert world coords to fractional grid coords
    float fx = (wx - sbt->origin_x) / sbt->dx;
    float fy = (wy - sbt->origin_y) / sbt->dy;
    float fz = (wz - sbt->origin_z) / sbt->dz;

    // Clamp to valid grid range
    fx = fmaxf(0.0f, fminf(fx, (float)(sbt->nx - 1) - 1e-4f));
    fy = fmaxf(0.0f, fminf(fy, (float)(sbt->ny - 1) - 1e-4f));
    fz = fmaxf(0.0f, fminf(fz, (float)(sbt->nz - 1) - 1e-4f));

    int ix = (int)fx, iy = (int)fy, iz = (int)fz;
    float sx = fx - ix, sy = fy - iy, sz = fz - iz;

    // 8 corner indices
    int N = sbt->nx * sbt->ny * sbt->nz;
    int nx = sbt->nx, ny = sbt->ny;

    #define IDX(x,y,z) ((x) + nx * ((y) + ny * (z)))
    int c000 = IDX(ix, iy, iz);
    int c100 = IDX(ix+1, iy, iz);
    int c010 = IDX(ix, iy+1, iz);
    int c110 = IDX(ix+1, iy+1, iz);
    int c001 = IDX(ix, iy, iz+1);
    int c101 = IDX(ix+1, iy, iz+1);
    int c011 = IDX(ix, iy+1, iz+1);
    int c111 = IDX(ix+1, iy+1, iz+1);
    #undef IDX

    // Trilinear interpolation for each velocity component (SoA layout)
    float3 vel;
    for (int comp = 0; comp < 3; comp++) {
        const float* u = sbt->velocity_field + comp * N;
        float v000 = u[c000], v100 = u[c100], v010 = u[c010], v110 = u[c110];
        float v001 = u[c001], v101 = u[c101], v011 = u[c011], v111 = u[c111];

        // Lerp along x, then y, then z
        float v00 = v000 * (1-sx) + v100 * sx;
        float v10 = v010 * (1-sx) + v110 * sx;
        float v01 = v001 * (1-sx) + v101 * sx;
        float v11 = v011 * (1-sx) + v111 * sx;
        float v0  = v00  * (1-sy) + v10  * sy;
        float v1  = v01  * (1-sy) + v11  * sy;
        float v   = v0   * (1-sz) + v1   * sz;

        if (comp == 0) vel.x = v;
        else if (comp == 1) vel.y = v;
        else vel.z = v;
    }
    return vel;
}

// ---------------------------------------------------------------------------
// Raygen: one ray per particle
// ---------------------------------------------------------------------------
extern "C" __global__ void __raygen__trace_particles() {
    unsigned int idx = optixGetLaunchIndex().x;
    if (idx >= params.n_particles) return;

    float3 pos = params.positions[idx];
    float3 vel = params.velocities[idx];
    float speed = sqrtf(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z);

    if (speed < 1e-10f) {
        params.brick_indices[idx] = -1;
        return;
    }

    float3 dir = make_float3(vel.x/speed, vel.y/speed, vel.z/speed);
    float tmax = speed * params.dt;

    // Payload: brick_idx (register 0), hit_t (register 1 as bits)
    unsigned int payload_brick = 0xFFFFFFFF;
    unsigned int payload_t_bits = 0;

    optixTrace(
        params.traversable,
        pos, dir,
        0.0f, tmax, 0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0, 1, 0,
        payload_brick, payload_t_bits
    );

    params.brick_indices[idx] = (int)payload_brick;

    if (payload_brick != 0xFFFFFFFF) {
        // Hit: interpolate velocity at hit point from SBT data
        float hit_t = __uint_as_float(payload_t_bits);
        float3 hit_pos = make_float3(
            pos.x + dir.x * hit_t,
            pos.y + dir.y * hit_t,
            pos.z + dir.z * hit_t
        );

        // Update particle velocity from LBM field via trilinear interp
        // (SBT data is accessed in closest-hit; here we use the velocity
        // that was written to the output buffer by closest-hit)
        // For now, advect with the original velocity; the closest-hit
        // writes the interpolated velocity to params.velocities[idx].
    }

    // Advect particle position
    params.positions[idx] = make_float3(
        pos.x + vel.x * params.dt,
        pos.y + vel.y * params.dt,
        pos.z + vel.z * params.dt
    );
}

// ---------------------------------------------------------------------------
// Closest-hit: interpolate velocity at hit point via SBT zero-copy pointer
// ---------------------------------------------------------------------------
extern "C" __global__ void __closesthit__cell() {
    unsigned int prim_idx = optixGetPrimitiveIndex();
    optixSetPayload_0(prim_idx);

    // Pass hit_t back for raygen to compute hit position
    float hit_t = optixGetRayTmax(); // closest intersection t
    optixSetPayload_1(__float_as_uint(hit_t));

    // Read SBT data for zero-copy velocity field access
    const LbmSbtData* sbt =
        reinterpret_cast<const LbmSbtData*>(optixGetSbtDataPointer());

    if (sbt->velocity_field != nullptr) {
        // Compute hit position in world space
        float3 origin = optixGetWorldRayOrigin();
        float3 dir = optixGetWorldRayDirection();
        float3 hit_pos = make_float3(
            origin.x + dir.x * hit_t,
            origin.y + dir.y * hit_t,
            origin.z + dir.z * hit_t
        );

        // Trilinear interpolation of velocity at hit point
        float3 interp_vel = trilinear_velocity(sbt, hit_pos.x, hit_pos.y, hit_pos.z);

        // Write interpolated velocity back to particle buffer
        // (raygen will read this for the next step's advection)
        unsigned int particle_idx = optixGetLaunchIndex().x;
        // Direct write to global memory from closest-hit
        // (params is accessible from all program types)
        params.velocities[particle_idx] = interp_vel;
    }
}

// ---------------------------------------------------------------------------
// Miss: particle left the domain
// ---------------------------------------------------------------------------
extern "C" __global__ void __miss__void() {
    optixSetPayload_0(0xFFFFFFFF);
    optixSetPayload_1(0);
}

// ---------------------------------------------------------------------------
// Intersection: AABB test for coarse 8^3 brick primitives
// ---------------------------------------------------------------------------
// The AABB bounds are provided by OptiX from the GAS build input.
// For CUSTOM_PRIMITIVES, the intersection program must report whether the
// ray intersects the primitive. Since OptiX already does the AABB test
// via the Box Intersection Engine (hardware), this program only needs to
// confirm the intersection and report the t value.
//
// For coarse bricks, we accept any ray that enters the brick AABB.
// Sub-brick precision is handled by trilinear interpolation at the hit point.
extern "C" __global__ void __intersection__aabb() {
    // The Box Intersection Engine already confirmed the ray hits this AABB.
    // We just need to report the intersection at the entry t.
    // For custom primitives, optixGetRayTmin() gives us the current closest
    // valid t. We report at tmin to accept this as a valid intersection.
    float t_hit = optixGetRayTmin();
    // Accept intersection at the AABB entry point
    optixReportIntersection(t_hit, 0);
}

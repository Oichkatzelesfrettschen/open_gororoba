//! OptiX 9.1 tracer particle advection through LBM density field.
//!
//! Uses RT cores (Ada SM 8.9) for O(log N) particle-cell intersection via
//! BVH traversal of CUSTOM_PRIMITIVES (coarse 8^3 brick AABBs).
//!
//! # Architecture
//!
//! 1. CUDA kernel scans density field -> compact array of 8^3 brick AABBs
//! 2. OptiX builds BVH (GAS) over ~2K coarse brick AABBs (microsecond build)
//! 3. Raygen launches one ray per tracer particle along velocity vector
//! 4. RT cores traverse BVH at hardware speed (Box Intersection Engine)
//! 5. Closest-hit reads velocity field via SBT zero-copy pointer
//! 6. Trilinear interpolation at hit coordinate -> updated particle velocity
//!
//! BVH is amortized: rebuilt every `bvh_rebuild_interval` steps.
//! Between rebuilds, topology changes slowly vs acoustic speed.
//!
//! Zero-copy interop: LBM macroscopic buffers (rho, u) are passed as raw
//! CUdeviceptr in the SBT record. No host-device copies.
//!
//! # Feature gate
//!
//! Requires `optix` feature flag. OptiX headers must be at `/usr/include/optix/`.

use gororoba_sparse_grid::{BrickGrid3d, BrickShape3d, LogicalGrid3d, OccupancyBitsetStats};

/// OptiX tracer configuration.
#[derive(Debug, Clone)]
pub struct OptiXTracerConfig {
    /// Grid dimensions (nx, ny, nz).
    pub grid_dim: (u32, u32, u32),
    /// World-space origin of the grid.
    pub grid_origin: [f32; 3],
    /// Size of one cell in world coordinates.
    pub cell_size: [f32; 3],
    /// Density threshold for occupied bricks.
    pub rho_threshold: f32,
    /// Advection timestep.
    pub dt: f32,
    /// Brick size for coarse BVH (typically 8).
    pub brick_size: u32,
    /// Steps between BVH rebuilds (0 = rebuild every step).
    pub bvh_rebuild_interval: u32,
    /// Occupancy change threshold to force early rebuild (fraction, e.g., 0.05).
    pub rebuild_occupancy_delta: f32,
}

impl Default for OptiXTracerConfig {
    fn default() -> Self {
        Self {
            grid_dim: (128, 128, 128),
            grid_origin: [0.0, 0.0, 0.0],
            cell_size: [1.0, 1.0, 1.0],
            rho_threshold: 0.5,
            dt: 0.001,
            brick_size: 8,
            bvh_rebuild_interval: 20,
            rebuild_occupancy_delta: 0.05,
        }
    }
}

/// SBT data record for zero-copy access to LBM macroscopic fields.
///
/// This struct is embedded in the HitGroup SBT record and matches
/// the `LbmSbtData` struct in `optix_tracer.cu` exactly.
///
/// The velocity and density pointers are raw CUdeviceptr from the LBM
/// solver's device buffers. No host-device copy occurs.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LbmSbtData {
    /// Velocity field device pointer (SoA: `u_out[3*N]`).
    pub velocity_field_ptr: u64,
    /// Density field device pointer (`rho_out[N]`).
    pub density_field_ptr: u64,
    pub nx: i32,
    pub ny: i32,
    pub nz: i32,
    pub dx: f32,
    pub dy: f32,
    pub dz: f32,
    pub origin_x: f32,
    pub origin_y: f32,
    pub origin_z: f32,
    pub brick_size: i32,
}

impl LbmSbtData {
    /// Create SBT data from config and device pointers.
    pub fn from_config(
        config: &OptiXTracerConfig,
        velocity_device_ptr: u64,
        density_device_ptr: u64,
    ) -> Self {
        Self {
            velocity_field_ptr: velocity_device_ptr,
            density_field_ptr: density_device_ptr,
            nx: config.grid_dim.0 as i32,
            ny: config.grid_dim.1 as i32,
            nz: config.grid_dim.2 as i32,
            dx: config.cell_size[0],
            dy: config.cell_size[1],
            dz: config.cell_size[2],
            origin_x: config.grid_origin[0],
            origin_y: config.grid_origin[1],
            origin_z: config.grid_origin[2],
            brick_size: config.brick_size as i32,
        }
    }
}

/// AABB for one occupied fluid brick.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BrickAabb {
    pub min_x: f32,
    pub min_y: f32,
    pub min_z: f32,
    pub max_x: f32,
    pub max_y: f32,
    pub max_z: f32,
}

/// Result from the brick occupancy scan kernel.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BrickResult {
    pub aabb: BrickAabb,
    pub brick_idx: u32,
    pub _pad: u32,
}

/// Build coarse 8^3 brick AABBs from a density field (CPU fallback).
///
/// For the GPU path, use the `scan_brick_occupancy` CUDA kernel from
/// `optix_brick_scan.cu` which runs on the device without readback.
pub fn build_brick_aabbs(rho: &[f32], config: &OptiXTracerConfig) -> Vec<BrickResult> {
    let (nx, ny, nz) = config.grid_dim;
    let bs = config.brick_size;
    let brick_grid = BrickGrid3d::from_logical_grid(
        LogicalGrid3d { nx, ny, nz },
        BrickShape3d {
            core_edge_cells: bs,
            halo_edge_cells: bs + 2,
        },
    );
    let bx_count = brick_grid.bricks_x;
    let by_count = brick_grid.bricks_y;
    let bz_count = brick_grid.bricks_z;

    let mut results = Vec::new();
    for ibz in 0..bz_count {
        for iby in 0..by_count {
            for ibx in 0..bx_count {
                let x0 = ibx * bs;
                let y0 = iby * bs;
                let z0 = ibz * bs;

                // Find max(rho) in this brick
                let mut max_rho = f32::NEG_INFINITY;
                for dz in 0..bs {
                    for dy in 0..bs {
                        for dx in 0..bs {
                            let ix = (x0 + dx) as usize;
                            let iy = (y0 + dy) as usize;
                            let iz = (z0 + dz) as usize;
                            if ix < nx as usize && iy < ny as usize && iz < nz as usize {
                                let idx = iz * ny as usize * nx as usize + iy * nx as usize + ix;
                                if rho[idx] > max_rho {
                                    max_rho = rho[idx];
                                }
                            }
                        }
                    }
                }

                if max_rho > config.rho_threshold {
                    let brick_idx = ibx + bx_count * (iby + by_count * ibz);
                    results.push(BrickResult {
                        aabb: BrickAabb {
                            min_x: config.grid_origin[0] + x0 as f32 * config.cell_size[0],
                            min_y: config.grid_origin[1] + y0 as f32 * config.cell_size[1],
                            min_z: config.grid_origin[2] + z0 as f32 * config.cell_size[2],
                            max_x: config.grid_origin[0] + (x0 + bs) as f32 * config.cell_size[0],
                            max_y: config.grid_origin[1] + (y0 + bs) as f32 * config.cell_size[1],
                            max_z: config.grid_origin[2] + (z0 + bs) as f32 * config.cell_size[2],
                        },
                        brick_idx,
                        _pad: 0,
                    });
                }
            }
        }
    }
    results
}

/// Determine whether the BVH should be rebuilt based on occupancy change.
pub fn should_rebuild_bvh(
    step: u64,
    config: &OptiXTracerConfig,
    current_occupancy: f64,
    last_rebuild_occupancy: f64,
) -> bool {
    if config.bvh_rebuild_interval == 0 {
        return true; // rebuild every step
    }
    if step.is_multiple_of(config.bvh_rebuild_interval as u64) {
        return true; // periodic rebuild
    }
    // Early rebuild if occupancy changed significantly
    let delta = (current_occupancy - last_rebuild_occupancy).abs();
    delta > config.rebuild_occupancy_delta as f64
}

/// Compute the occupancy fraction (cells > threshold / total cells).
pub fn occupancy_fraction(rho: &[f32], threshold: f32) -> f64 {
    let count = rho.iter().filter(|&&r| r > threshold).count();
    count as f64 / rho.len() as f64
}

/// Compute brick-level occupancy (bricks with max(rho) > threshold / total bricks).
pub fn brick_occupancy_fraction(rho: &[f32], config: &OptiXTracerConfig) -> f64 {
    let (nx, ny, nz) = config.grid_dim;
    let bs = config.brick_size;
    let brick_grid = BrickGrid3d::from_logical_grid(
        LogicalGrid3d { nx, ny, nz },
        BrickShape3d {
            core_edge_cells: bs,
            halo_edge_cells: bs + 2,
        },
    );
    let stats = OccupancyBitsetStats {
        total_bricks: brick_grid.total_bricks(),
        active_bricks: build_brick_aabbs(rho, config).len() as u64,
    };
    stats.occupancy_fraction()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_brick_aabbs_uniform() {
        let config = OptiXTracerConfig {
            grid_dim: (16, 16, 16),
            grid_origin: [0.0, 0.0, 0.0],
            cell_size: [1.0, 1.0, 1.0],
            rho_threshold: 0.5,
            brick_size: 8,
            ..Default::default()
        };
        let rho = vec![1.0f32; 16 * 16 * 16];
        let bricks = build_brick_aabbs(&rho, &config);
        // 16/8 = 2 bricks per axis, 2^3 = 8 total
        assert_eq!(bricks.len(), 8);
        // First brick: (0,0,0) to (8,8,8)
        assert!((bricks[0].aabb.min_x - 0.0).abs() < 1e-6);
        assert!((bricks[0].aabb.max_x - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_build_brick_aabbs_sparse() {
        let config = OptiXTracerConfig {
            grid_dim: (16, 16, 16),
            grid_origin: [0.0, 0.0, 0.0],
            cell_size: [1.0, 1.0, 1.0],
            rho_threshold: 0.5,
            brick_size: 8,
            ..Default::default()
        };
        // Only cell (0,0,0) above threshold
        let mut rho = vec![0.1f32; 16 * 16 * 16];
        rho[0] = 1.0;
        let bricks = build_brick_aabbs(&rho, &config);
        assert_eq!(bricks.len(), 1); // only brick (0,0,0) occupied
    }

    #[test]
    fn test_should_rebuild_bvh() {
        let config = OptiXTracerConfig {
            bvh_rebuild_interval: 20,
            rebuild_occupancy_delta: 0.05,
            ..Default::default()
        };
        assert!(should_rebuild_bvh(0, &config, 0.5, 0.5));
        assert!(!should_rebuild_bvh(1, &config, 0.5, 0.5));
        assert!(should_rebuild_bvh(20, &config, 0.5, 0.5));
        assert!(should_rebuild_bvh(5, &config, 0.6, 0.5)); // delta > 0.05
        assert!(!should_rebuild_bvh(5, &config, 0.52, 0.5)); // delta < 0.05
    }

    #[test]
    fn test_sbt_data_layout() {
        let config = OptiXTracerConfig::default();
        let sbt = LbmSbtData::from_config(&config, 0xDEADBEEF, 0xCAFEBABE);
        assert_eq!(sbt.velocity_field_ptr, 0xDEADBEEF);
        assert_eq!(sbt.density_field_ptr, 0xCAFEBABE);
        assert_eq!(sbt.nx, 128);
        assert_eq!(sbt.brick_size, 8);
        // Verify struct size matches C layout expectations
        assert_eq!(
            std::mem::size_of::<LbmSbtData>(),
            2 * 8 + 3 * 4 + 3 * 4 + 3 * 4 + 4, // 2 u64 + 3 i32 + 3 f32 + 3 f32 + 1 i32
            "LbmSbtData size mismatch"
        );
    }

    #[test]
    fn test_occupancy_fraction() {
        let rho = vec![0.1, 0.6, 0.3, 0.8];
        assert!((occupancy_fraction(&rho, 0.5) - 0.5).abs() < 1e-10);
    }
}

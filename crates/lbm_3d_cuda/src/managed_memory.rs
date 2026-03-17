//! CUDA Unified Memory management for out-of-core LBM at 1024^3+.
//!
//! Uses `cuMemAllocManaged` + `cuMemPrefetchAsync` to transparently page
//! sparse brick data between GPU VRAM and system RAM via PCIe 4.0.
//!
//! # Architecture
//!
//! At 1024^3 with 30% sparsity + ephemeral macros + null force:
//! - INT8 distributions: 6.1 GB (permanent, GPU-resident)
//! - tau field: 1.3 GB (permanent, GPU-resident)
//! - FP16 velocity: 2.6 GB (transient, allocated on-demand for OptiX)
//!
//! When occupancy exceeds the 12 GB VRAM budget, Unified Memory pages
//! inactive bricks to system RAM automatically. Explicit prefetch hints
//! via `cuMemPrefetchAsync` bring the active Z-slab bricks to GPU before
//! each LBM step, keeping the collision operator GPU-resident.
//!
//! # Expected Performance
//!
//! PCIe 4.0 x16: ~25 GB/s bidirectional. With 30% sparsity:
//! - Active slab migration per step: ~200 MB (one Z-layer of bricks)
//! - Migration time: ~8 ms (vs ~35 ms LBM step at 512^3)
//! - Overhead: ~23% of step time (acceptable for 1024^3 out-of-core)
//!
//! # Feature Gate
//!
//! Uses cudarc's `CudaContext::alloc_unified()` API (no feature gate needed).

use std::sync::Arc;

/// Configuration for managed memory allocation.
#[derive(Debug, Clone)]
pub struct ManagedMemoryConfig {
    /// Grid dimensions (nx, ny, nz).
    pub grid_dim: (usize, usize, usize),
    /// Brick size for prefetch granularity.
    pub brick_size: usize,
    /// Number of Z-slabs to prefetch ahead of the current step.
    pub prefetch_ahead: usize,
    /// Estimated sparsity fraction (0.0 = fully dense, 1.0 = fully empty).
    pub sparsity: f64,
}

impl Default for ManagedMemoryConfig {
    fn default() -> Self {
        Self {
            grid_dim: (1024, 1024, 1024),
            brick_size: 8,
            prefetch_ahead: 2,
            sparsity: 0.70, // 30% occupied
        }
    }
}

impl ManagedMemoryConfig {
    /// Estimate the GPU-resident VRAM requirement for this configuration.
    ///
    /// Accounts for INT8 distributions (19 bytes/cell) + tau (4 bytes/cell),
    /// multiplied by occupancy fraction. Does not include transient buffers
    /// (FP16 velocity, OptiX BVH).
    pub fn estimate_gpu_resident_mb(&self) -> usize {
        let (nx, ny, nz) = self.grid_dim;
        let n_cells = nx * ny * nz;
        let occupancy = 1.0 - self.sparsity;
        let active_cells = (n_cells as f64 * occupancy) as usize;
        // INT8 dist (19 bytes/cell, single buffer A-A) + tau (4 bytes/cell)
        let bytes = active_cells * (19 + 4);
        bytes / (1024 * 1024)
    }

    /// Estimate the Z-slab prefetch size in MB.
    pub fn slab_prefetch_mb(&self) -> usize {
        let (nx, ny, _nz) = self.grid_dim;
        let slab_cells = nx * ny * self.brick_size; // one Z-slab of bricks
        let occupancy = 1.0 - self.sparsity;
        let active_slab = (slab_cells as f64 * occupancy) as usize;
        active_slab * 19 / (1024 * 1024) // INT8 dist bytes per slab
    }

    /// Estimate PCIe migration overhead per LBM step in milliseconds.
    ///
    /// Assumes PCIe 4.0 x16 at 25 GB/s effective throughput.
    pub fn estimate_migration_ms(&self) -> f64 {
        let slab_mb = self.slab_prefetch_mb() * self.prefetch_ahead;
        slab_mb as f64 / 25_000.0 * 1000.0 // MB / (MB/s) -> ms
    }
}

/// Allocate a unified memory buffer for LBM distributions.
///
/// The returned `UnifiedSlice` is accessible from both host and device.
/// The driver automatically pages data between VRAM and system RAM.
///
/// Use `prefetch_to_device()` to hint which regions should be GPU-resident
/// before launching a kernel.
pub fn alloc_unified_distributions(
    ctx: &Arc<cudarc::driver::CudaContext>,
    n_cells: usize,
    elem_bytes: usize,
) -> anyhow::Result<cudarc::driver::UnifiedSlice<u8>> {
    let total_bytes = n_cells * 19 * elem_bytes;
    // Safety: alloc_unified allocates managed memory accessible from host and device.
    // The caller must ensure the buffer is used correctly (no concurrent host/device access
    // without proper synchronization).
    let unified = unsafe { ctx.alloc_unified::<u8>(total_bytes, true) }?;
    Ok(unified)
}

/// Prefetch the entire unified distribution buffer to the GPU.
///
/// This is a non-blocking hint to the CUDA driver to migrate the data
/// from system RAM to VRAM. For slab-level granularity, use the raw
/// `cuMemPrefetchAsync` driver API on a subrange pointer.
pub fn prefetch_to_device(
    unified: &cudarc::driver::UnifiedSlice<u8>,
) -> anyhow::Result<()> {
    unified.prefetch()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = ManagedMemoryConfig::default();
        assert_eq!(cfg.grid_dim, (1024, 1024, 1024));
        assert_eq!(cfg.brick_size, 8);
    }

    #[test]
    fn test_estimate_gpu_resident() {
        let cfg = ManagedMemoryConfig {
            grid_dim: (1024, 1024, 1024),
            sparsity: 0.70,
            ..Default::default()
        };
        let mb = cfg.estimate_gpu_resident_mb();
        // 1.07B cells * 0.30 occupancy * 23 bytes/cell = ~7.4 GB
        assert!(mb > 7000 && mb < 8000, "Expected ~7400 MB, got {mb}");
    }

    #[test]
    fn test_slab_prefetch_size() {
        let cfg = ManagedMemoryConfig {
            grid_dim: (512, 512, 512),
            sparsity: 0.70,
            brick_size: 8,
            ..Default::default()
        };
        let mb = cfg.slab_prefetch_mb();
        // 512 * 512 * 8 * 0.30 * 19 = ~12 MB
        assert!(mb > 5 && mb < 20, "Expected ~12 MB, got {mb}");
    }

    #[test]
    fn test_migration_overhead() {
        let cfg = ManagedMemoryConfig::default();
        let ms = cfg.estimate_migration_ms();
        // Should be in the range of a few ms for one Z-slab at PCIe 4.0
        assert!(ms < 100.0, "Migration overhead too high: {ms} ms");
    }
}

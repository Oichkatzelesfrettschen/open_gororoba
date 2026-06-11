//! LaunchConfig helpers tuned per CUDA architecture.
//!
//! Consolidates 20+ hard-coded `LaunchConfig { grid_dim, block_dim, ...
//! }` sites. The workspace's block-dim conventions:
//!   - 256 threads/block: Ampere+ default (covers most LBM, TurboQuant,
//!     algebra paths).
//!   - 128 threads/block: FP64 paths on Ada that hit register pressure
//!     (per project memory CUDA D3Q19 note).

use cudarc::driver::LaunchConfig as CudarcLaunchConfig;

use crate::probe::DeviceProbe;

/// Builder for `cudarc::driver::LaunchConfig`.
pub struct LaunchConfig;

impl LaunchConfig {
    /// 1D launch: `n` elements, default 256 threads per block.
    /// Uses `(n + 255) / 256` blocks.
    pub fn launch_1d(n: u32) -> CudarcLaunchConfig {
        Self::launch_1d_with_block(n, 256)
    }

    /// 1D launch with custom block size. Use 128 for FP64-heavy
    /// kernels per the project's "FP64 D3Q19 needs 128 threads/block"
    /// note.
    pub fn launch_1d_with_block(n: u32, block_dim: u32) -> CudarcLaunchConfig {
        let block_dim = block_dim.max(1);
        let grid = n.div_ceil(block_dim);
        CudarcLaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// 1D launch for callers that already selected the block count.
    pub fn launch_blocks_1d(grid_dim: u32, block_dim: u32) -> CudarcLaunchConfig {
        CudarcLaunchConfig {
            grid_dim: (grid_dim.max(1), 1, 1),
            block_dim: (block_dim.max(1), 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// 2D launch: `nx` x `ny`, default 16x16 threads per block.
    pub fn launch_2d(nx: u32, ny: u32) -> CudarcLaunchConfig {
        let block_x = 16;
        let block_y = 16;
        CudarcLaunchConfig {
            grid_dim: (nx.div_ceil(block_x), ny.div_ceil(block_y), 1),
            block_dim: (block_x, block_y, 1),
            shared_mem_bytes: 0,
        }
    }

    /// 2D grid launch for callers that already selected the block shape.
    pub fn launch_blocks_2d(
        grid_x: u32,
        grid_y: u32,
        block_x: u32,
        block_y: u32,
    ) -> CudarcLaunchConfig {
        CudarcLaunchConfig {
            grid_dim: (grid_x.max(1), grid_y.max(1), 1),
            block_dim: (block_x.max(1), block_y.max(1), 1),
            shared_mem_bytes: 0,
        }
    }

    /// 3D grid launch for callers that already selected the block shape.
    pub fn launch_blocks_3d(
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        block_x: u32,
        block_y: u32,
        block_z: u32,
    ) -> CudarcLaunchConfig {
        CudarcLaunchConfig {
            grid_dim: (grid_x.max(1), grid_y.max(1), grid_z.max(1)),
            block_dim: (block_x.max(1), block_y.max(1), block_z.max(1)),
            shared_mem_bytes: 0,
        }
    }

    /// 3D launch: `nx` x `ny` x `nz`, default 8x8x8 threads per block.
    pub fn launch_3d(nx: u32, ny: u32, nz: u32) -> CudarcLaunchConfig {
        let block_x = 8;
        let block_y = 8;
        let block_z = 8;
        CudarcLaunchConfig {
            grid_dim: (
                nx.div_ceil(block_x),
                ny.div_ceil(block_y),
                nz.div_ceil(block_z),
            ),
            block_dim: (block_x, block_y, block_z),
            shared_mem_bytes: 0,
        }
    }

    /// Architecture-tuned 1D launch. Falls back to 128 threads/block on
    /// pre-Ampere devices and on FP64-heavy paths on Ada (which hit
    /// register pressure at 256 threads/block).
    ///
    /// `is_fp64_heavy` is a caller hint; default to false unless the
    /// kernel uses `double` throughout.
    pub fn launch_1d_tuned(n: u32, probe: &DeviceProbe, is_fp64_heavy: bool) -> CudarcLaunchConfig {
        let block = if probe.major < 8 || (is_fp64_heavy && probe.is_ada()) {
            128
        } else {
            256
        };
        Self::launch_1d_with_block(n, block)
    }
}

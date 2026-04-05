//! Sparse LBM Manager and Kernel Interface.
//! Integrates the sparse brick map with cudarc for 1024^3 in 12GB VRAM.

use anyhow::{Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, DeviceSlice, LaunchConfig,
    PushKernelArg, UnifiedSlice, result, sys,
};
use gororoba_sparse_grid::{
    ActiveBrickWindow, BrickGrid3d, BrickShape3d, LogicalGrid3d, OccupancyBitsetStats,
};
use std::{env, sync::Arc};

const KERNEL_SPARSE_MAP_SRC: &str = include_str!("kernels_sparse_map.cu");

/// Manages the sparse occupancy map and indirect table.
pub struct SparseBrickMap {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub bx_max: usize,
    pub by_max: usize,
    pub bz_max: usize,
    pub n_bricks: usize,
    pub n_active_bricks: usize,

    // Device memory handles
    pub d_occupancy_words: CudaSlice<u32>,
    pub d_brick_counts: CudaSlice<u32>,
    pub d_brick_offsets: CudaSlice<u32>,
    pub d_indirect_table: CudaSlice<i32>,
    pub d_active_brick_ids: CudaSlice<u32>,

    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
}

impl SparseBrickMap {
    /// Builds the sparse metadata structures from a given 3D geometry mask.
    pub fn new_from_geometry(
        ctx: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        nx: usize,
        ny: usize,
        nz: usize,
        d_geometry_mask: &CudaSlice<u8>,
    ) -> Result<Self> {
        // Compile kernels
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some(crate::preferred_cuda_arch()),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(KERNEL_SPARSE_MAP_SRC, opts)
            .context("Failed to compile kernels_sparse_map.cu")?;
        let module = ctx
            .load_module(ptx)
            .context("Failed to load sparse map module")?;

        let generate_occupancy_kernel = module.load_function("generate_occupancy_bitmask")?;
        let compact_bitmask_kernel = module.load_function("compact_bitmask_atomic")?;

        let brick_grid = BrickGrid3d::from_logical_grid(
            LogicalGrid3d {
                nx: nx as u32,
                ny: ny as u32,
                nz: nz as u32,
            },
            BrickShape3d {
                core_edge_cells: 8,
                halo_edge_cells: 10,
            },
        );
        let bx_max = brick_grid.bricks_x as usize;
        let by_max = brick_grid.bricks_y as usize;
        let bz_max = brick_grid.bricks_z as usize;
        let n_bricks = brick_grid.total_bricks() as usize;

        let num_words = n_bricks.div_ceil(32);

        let mut d_occupancy_words = stream.alloc_zeros::<u32>(num_words)?;
        let mut d_brick_counts = stream.alloc_zeros::<u32>(n_bricks)?;
        let d_brick_offsets = stream.alloc_zeros::<u32>(n_bricks)?;
        let mut d_indirect_table = stream.alloc_zeros::<i32>(n_bricks)?;
        let mut d_active_brick_ids = stream.alloc_zeros::<u32>(n_bricks.max(1))?;
        let mut d_active_brick_count = stream.alloc_zeros::<u32>(1)?;

        // 1. Generate Bitmask
        let block = (8, 8, 8);
        let grid = (
            (nx as u32).div_ceil(block.0),
            (ny as u32).div_ceil(block.1),
            (nz as u32).div_ceil(block.2),
        );
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: block,
            shared_mem_bytes: 0,
        };
        let nx_i = nx as i32;
        let ny_i = ny as i32;
        let nz_i = nz as i32;
        let bx_max_i = bx_max as i32;
        let by_max_i = by_max as i32;
        let bz_max_i = bz_max as i32;
        let mut b1 = stream.launch_builder(&generate_occupancy_kernel);
        b1.arg(d_geometry_mask)
            .arg(&mut d_occupancy_words)
            .arg(&nx_i)
            .arg(&ny_i)
            .arg(&nz_i)
            .arg(&bx_max_i)
            .arg(&by_max_i)
            .arg(&bz_max_i);
        unsafe { b1.launch(cfg) }?;

        // 2. Compact active bricks on the GPU.
        let block2 = 256;
        let grid2 = (n_bricks as u32).div_ceil(block2);
        let n_bricks_i = n_bricks as i32;
        let mut b2 = stream.launch_builder(&compact_bitmask_kernel);
        b2.arg(&d_occupancy_words)
            .arg(&mut d_brick_counts)
            .arg(&mut d_indirect_table)
            .arg(&mut d_active_brick_ids)
            .arg(&mut d_active_brick_count)
            .arg(&n_bricks_i);
        unsafe {
            b2.launch(LaunchConfig {
                grid_dim: (grid2, 1, 1),
                block_dim: (block2, 1, 1),
                shared_mem_bytes: 0,
            })
        }?;

        // 3. Read back only the active-brick count; compaction itself stayed on GPU.
        let h_active_counts = stream.clone_dtoh(&d_active_brick_count)?;
        let n_active_bricks = h_active_counts.first().copied().unwrap_or_default() as usize;

        Ok(Self {
            nx,
            ny,
            nz,
            bx_max,
            by_max,
            bz_max,
            n_bricks,
            n_active_bricks,
            d_occupancy_words,
            d_brick_counts,
            d_brick_offsets,
            d_indirect_table,
            d_active_brick_ids,
            ctx,
            stream,
        })
    }

    /// Sparse occupancy stats for the current brick map.
    #[must_use]
    pub fn occupancy_stats(&self) -> OccupancyBitsetStats {
        OccupancyBitsetStats {
            total_bricks: self.n_bricks as u64,
            active_bricks: self.n_active_bricks as u64,
        }
    }
}

const KERNEL_SPARSE_LBM_SRC: &str = include_str!("kernels_sparse_lbm.cu");

/// Sparse-kernel execution variant.
///
/// Both variants use the same sparse brick map and A-A storage scheme. The
/// tiled variant stages one distribution direction at a time through a `10^3`
/// halo tile in shared memory, while the direct variant reads neighbors
/// straight from global memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SparseKernelVariant {
    /// Direct global-memory sparse A-A kernel.
    #[default]
    DirectGlobal,
    /// Shared-memory halo staging for sparse brick pulls, with brick-local
    /// vectorized fast paths where the inner core is contiguous.
    SharedHaloTiled,
}

/// Sparse LBM Solver using the Brick Map and A-A streaming pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SparseMemoryMode {
    /// Keep sparse state in device-local VRAM. This remains the default fast path.
    #[default]
    DeviceLocal,
    /// Allocate sparse state in CUDA managed/unified memory and prefetch toward
    /// the active device before stepping. This is the CUDA Unified Memory
    /// overflow fallback.
    ManagedUnifiedPrefetch,
    /// Allocate sparse state in CUDA managed/unified memory and step the sparse
    /// domain in active-brick tiles, prefetching only the current tile to the
    /// device.
    ///
    /// This is the lower-headroom fallback for runs where the full sparse
    /// working set should remain unified-memory backed, but the hot loop should
    /// stay closer to the planner's per-tile footprint instead of migrating the
    /// whole window at once. In NVIDIA documentation this is unified-memory
    /// oversubscription plus explicit prefetch/tiling.
    ManagedUnifiedTilePrefetch,
}

#[derive(Debug)]
enum SparseFieldBuffer {
    Device(CudaSlice<f32>),
    Unified(UnifiedSlice<f32>),
}

impl SparseFieldBuffer {
    fn alloc_zeroed(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        len: usize,
        mode: SparseMemoryMode,
    ) -> Result<Self> {
        let len = len.max(1);
        match mode {
            SparseMemoryMode::DeviceLocal => Ok(Self::Device(stream.alloc_zeros::<f32>(len)?)),
            SparseMemoryMode::ManagedUnifiedPrefetch
            | SparseMemoryMode::ManagedUnifiedTilePrefetch => {
                let mut buf = unsafe {
                    ctx.alloc_unified::<f32>(len, true)
                        .context("alloc unified sparse buffer")?
                };
                buf.as_mut_slice()
                    .context("map unified sparse buffer on host")?
                    .fill(0.0);
                buf.prefetch()
                    .context("prefetch unified sparse buffer to device")?;
                Ok(Self::Unified(buf))
            }
        }
    }
}

pub struct SparseLbmSolver {
    pub map: SparseBrickMap,
    d_f: SparseFieldBuffer,
    d_rho: SparseFieldBuffer,
    d_u: SparseFieldBuffer,
    d_tau: SparseFieldBuffer,
    d_force: SparseFieldBuffer,
    memory_mode: SparseMemoryMode,
    kernel_variant: SparseKernelVariant,
    prefetch_stream: Option<Arc<CudaStream>>,

    pub step: usize,

    lbm_step_kernel: CudaFunction,
}

const CELLS_PER_BRICK: usize = 512;
const D3Q19_DIRS: usize = 19;
const F32_BYTES: usize = std::mem::size_of::<f32>();

struct SparseTilePrefetchInputs<'a> {
    d_f: &'a UnifiedSlice<f32>,
    d_rho: &'a UnifiedSlice<f32>,
    d_u: &'a UnifiedSlice<f32>,
    d_tau: &'a UnifiedSlice<f32>,
    d_force: &'a UnifiedSlice<f32>,
    stream: &'a Arc<CudaStream>,
    active_cell_start: usize,
    active_cell_count: usize,
    total_active_cells: usize,
}

impl SparseLbmSolver {
    pub fn new(map: SparseBrickMap) -> Result<Self> {
        Self::new_with_mode(map, SparseMemoryMode::DeviceLocal)
    }

    /// Create a sparse solver with an explicit memory mode.
    ///
    /// `DeviceLocal` keeps the sparse state fully in VRAM and remains the
    /// primary fast path. `ManagedUnifiedPrefetch` allocates the sparse state
    /// in CUDA managed/unified memory and prefetched it toward the active GPU
    /// as an overflow-safe fallback when VRAM headroom is uncertain. In NVIDIA
    /// terminology this is unified-memory oversubscription, not ReBAR-based
    /// "extra VRAM".
    pub fn new_with_mode(map: SparseBrickMap, memory_mode: SparseMemoryMode) -> Result<Self> {
        let n_active_cells = map.n_active_bricks * 512;
        let ctx = map.ctx.clone();
        let stream = map.stream.clone();

        // Allocate only for active cells
        let d_f = SparseFieldBuffer::alloc_zeroed(&ctx, &stream, 19 * n_active_cells, memory_mode)?;
        let d_rho = SparseFieldBuffer::alloc_zeroed(&ctx, &stream, n_active_cells, memory_mode)?;
        let d_u = SparseFieldBuffer::alloc_zeroed(&ctx, &stream, 3 * n_active_cells, memory_mode)?;
        let d_tau = SparseFieldBuffer::alloc_zeroed(&ctx, &stream, n_active_cells, memory_mode)?;
        let d_force =
            SparseFieldBuffer::alloc_zeroed(&ctx, &stream, 3 * n_active_cells, memory_mode)?;

        // Compile kernel
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some(crate::preferred_cuda_arch()),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(KERNEL_SPARSE_LBM_SRC, opts)
            .context("Failed to compile kernels_sparse_lbm.cu")?;
        let module = map
            .ctx
            .load_module(ptx)
            .context("Failed to load sparse lbm module")?;

        let kernel_variant = preferred_sparse_kernel_variant();
        let lbm_step_kernel = module.load_function(match kernel_variant {
            SparseKernelVariant::DirectGlobal => "lbm_step_sparse_aa",
            SparseKernelVariant::SharedHaloTiled => "lbm_step_sparse_aa_tiled",
        })?;
        let prefetch_stream = match memory_mode {
            SparseMemoryMode::ManagedUnifiedTilePrefetch => Some(
                map.ctx
                    .new_stream()
                    .context("create sparse prefetch stream")?,
            ),
            _ => None,
        };

        Ok(Self {
            map,
            d_f,
            d_rho,
            d_u,
            d_tau,
            d_force,
            memory_mode,
            kernel_variant,
            prefetch_stream,
            step: 0,
            lbm_step_kernel,
        })
    }

    /// Return the active sparse-memory mode.
    pub fn memory_mode(&self) -> SparseMemoryMode {
        self.memory_mode
    }

    /// Return the active sparse-kernel execution variant.
    pub fn kernel_variant(&self) -> SparseKernelVariant {
        self.kernel_variant
    }

    pub fn evolve(&mut self, steps: usize) -> Result<()> {
        if self.map.n_active_bricks == 0 {
            self.step += steps;
            return Ok(());
        }

        let n_active_cells = self.map.n_active_bricks * CELLS_PER_BRICK;
        let nx_i = self.map.nx as i32;
        let ny_i = self.map.ny as i32;
        let nz_i = self.map.nz as i32;
        let bx_max_i = self.map.bx_max as i32;
        let by_max_i = self.map.by_max as i32;
        let bz_max_i = self.map.bz_max as i32;
        let n_active_cells_total_i = n_active_cells as i32;

        for _ in 0..steps {
            let parity = (self.step % 2) as i32;
            match self.memory_mode {
                SparseMemoryMode::DeviceLocal => {
                    let active_cell_start_i = 0_i32;
                    let active_cell_count_i = n_active_cells as i32;
                    let cfg = launch_cfg_for_cells(n_active_cells);
                    let mut b = self.map.stream.launch_builder(&self.lbm_step_kernel);
                    let SparseFieldBuffer::Device(d_f) = &mut self.d_f else {
                        unreachable!("device-local sparse solver requires device buffers")
                    };
                    let SparseFieldBuffer::Device(d_rho) = &mut self.d_rho else {
                        unreachable!("device-local sparse solver requires device buffers")
                    };
                    let SparseFieldBuffer::Device(d_u) = &mut self.d_u else {
                        unreachable!("device-local sparse solver requires device buffers")
                    };
                    let SparseFieldBuffer::Device(d_tau) = &self.d_tau else {
                        unreachable!("device-local sparse solver requires device buffers")
                    };
                    let SparseFieldBuffer::Device(d_force) = &self.d_force else {
                        unreachable!("device-local sparse solver requires device buffers")
                    };
                    b.arg(d_f)
                        .arg(d_rho)
                        .arg(d_u)
                        .arg(d_tau)
                        .arg(d_force)
                        .arg(&self.map.d_indirect_table)
                        .arg(&self.map.d_active_brick_ids)
                        .arg(&nx_i)
                        .arg(&ny_i)
                        .arg(&nz_i)
                        .arg(&bx_max_i)
                        .arg(&by_max_i)
                        .arg(&bz_max_i)
                        .arg(&active_cell_start_i)
                        .arg(&active_cell_count_i)
                        .arg(&n_active_cells_total_i)
                        .arg(&parity);
                    unsafe { b.launch(cfg) }?;
                }
                SparseMemoryMode::ManagedUnifiedPrefetch => {
                    let active_cell_start_i = 0_i32;
                    let active_cell_count_i = n_active_cells as i32;
                    let cfg = launch_cfg_for_cells(n_active_cells);
                    let mut b = self.map.stream.launch_builder(&self.lbm_step_kernel);
                    let SparseFieldBuffer::Unified(d_f) = &mut self.d_f else {
                        unreachable!("managed sparse solver requires unified buffers")
                    };
                    let SparseFieldBuffer::Unified(d_rho) = &mut self.d_rho else {
                        unreachable!("managed sparse solver requires unified buffers")
                    };
                    let SparseFieldBuffer::Unified(d_u) = &mut self.d_u else {
                        unreachable!("managed sparse solver requires unified buffers")
                    };
                    let SparseFieldBuffer::Unified(d_tau) = &self.d_tau else {
                        unreachable!("managed sparse solver requires unified buffers")
                    };
                    let SparseFieldBuffer::Unified(d_force) = &self.d_force else {
                        unreachable!("managed sparse solver requires unified buffers")
                    };
                    b.arg(d_f)
                        .arg(d_rho)
                        .arg(d_u)
                        .arg(d_tau)
                        .arg(d_force)
                        .arg(&self.map.d_indirect_table)
                        .arg(&self.map.d_active_brick_ids)
                        .arg(&nx_i)
                        .arg(&ny_i)
                        .arg(&nz_i)
                        .arg(&bx_max_i)
                        .arg(&by_max_i)
                        .arg(&bz_max_i)
                        .arg(&active_cell_start_i)
                        .arg(&active_cell_count_i)
                        .arg(&n_active_cells_total_i)
                        .arg(&parity);
                    unsafe { b.launch(cfg) }?;
                }
                SparseMemoryMode::ManagedUnifiedTilePrefetch => {
                    let tile_bricks = recommended_tile_bricks(self.map.n_active_bricks);
                    let SparseFieldBuffer::Unified(d_f) = &mut self.d_f else {
                        unreachable!("managed sparse solver requires unified buffers")
                    };
                    let SparseFieldBuffer::Unified(d_rho) = &mut self.d_rho else {
                        unreachable!("managed sparse solver requires unified buffers")
                    };
                    let SparseFieldBuffer::Unified(d_u) = &mut self.d_u else {
                        unreachable!("managed sparse solver requires unified buffers")
                    };
                    let SparseFieldBuffer::Unified(d_tau) = &self.d_tau else {
                        unreachable!("managed sparse solver requires unified buffers")
                    };
                    let SparseFieldBuffer::Unified(d_force) = &self.d_force else {
                        unreachable!("managed sparse solver requires unified buffers")
                    };
                    let prefetch_stream = self
                        .prefetch_stream
                        .as_ref()
                        .expect("managed tiled mode must own a prefetch stream");
                    let tile_windows =
                        build_tile_windows(self.map.n_active_bricks, tile_bricks, CELLS_PER_BRICK);
                    if let Some((first_tile, remaining_tiles)) = tile_windows.split_first() {
                        let mut ready_event = prefetch_sparse_tile(SparseTilePrefetchInputs {
                            d_f,
                            d_rho,
                            d_u,
                            d_tau,
                            d_force,
                            stream: prefetch_stream,
                            active_cell_start: first_tile.active_cell_start as usize,
                            active_cell_count: first_tile.active_cell_count as usize,
                            total_active_cells: n_active_cells,
                        })?;

                        for (tile_idx, tile) in tile_windows.iter().enumerate() {
                            self.map
                                .stream
                                .wait(&ready_event)
                                .context("wait for sparse tile prefetch")?;

                            if let Some(next_tile) = remaining_tiles.get(tile_idx) {
                                ready_event = prefetch_sparse_tile(SparseTilePrefetchInputs {
                                    d_f,
                                    d_rho,
                                    d_u,
                                    d_tau,
                                    d_force,
                                    stream: prefetch_stream,
                                    active_cell_start: next_tile.active_cell_start as usize,
                                    active_cell_count: next_tile.active_cell_count as usize,
                                    total_active_cells: n_active_cells,
                                })?;
                            }

                            let active_cell_start_i = tile.active_cell_start as i32;
                            let active_cell_count_i = tile.active_cell_count as i32;
                            let cfg = launch_cfg_for_cells(tile.active_cell_count as usize);
                            let mut b = self.map.stream.launch_builder(&self.lbm_step_kernel);
                            b.arg(&mut *d_f)
                                .arg(&mut *d_rho)
                                .arg(&mut *d_u)
                                .arg(d_tau)
                                .arg(d_force)
                                .arg(&self.map.d_indirect_table)
                                .arg(&self.map.d_active_brick_ids)
                                .arg(&nx_i)
                                .arg(&ny_i)
                                .arg(&nz_i)
                                .arg(&bx_max_i)
                                .arg(&by_max_i)
                                .arg(&bz_max_i)
                                .arg(&active_cell_start_i)
                                .arg(&active_cell_count_i)
                                .arg(&n_active_cells_total_i)
                                .arg(&parity);
                            unsafe { b.launch(cfg) }?;
                        }
                    }
                }
            }
            self.step += 1;
        }

        // Wait for operations to complete before returning
        // In cudarc 0.19, `CudaStream` might not have `sync()`. We'll synchronize via device context.
        self.map.ctx.synchronize()?;
        Ok(())
    }
}

fn launch_cfg_for_cells(active_cell_count: usize) -> LaunchConfig {
    let block = CELLS_PER_BRICK as u32;
    let grid = (active_cell_count as u32).div_ceil(block);
    LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn preferred_sparse_kernel_variant() -> SparseKernelVariant {
    if let Ok(value) = env::var("GOROROBA_SPARSE_KERNEL") {
        match value.as_str() {
            "tiled" | "shared" | "shared-halo" | "shared-halo-tiled" => {
                return SparseKernelVariant::SharedHaloTiled;
            }
            "direct" | "global" | "direct-global" => return SparseKernelVariant::DirectGlobal,
            _ => {}
        }
    }
    if crate::probe_cuda_device_props()
        .map(|props| props.sparse_tile_preferred)
        .unwrap_or(false)
    {
        SparseKernelVariant::SharedHaloTiled
    } else {
        SparseKernelVariant::DirectGlobal
    }
}

fn recommended_tile_bricks(total_active_bricks: usize) -> usize {
    if let Ok(value) = env::var("GOROROBA_SPARSE_TILE_BRICKS")
        && let Ok(parsed) = value.parse::<usize>()
    {
        return parsed.max(1).min(total_active_bricks.max(1));
    }

    let bytes_per_brick = sparse_bytes_per_brick();
    let requested_bytes = bytes_per_brick.saturating_mul(total_active_bricks);
    if let Some(props) = crate::probe_cuda_device_props() {
        let policy = crate::plan_managed_memory_policy(props, requested_bytes);
        let tile_bytes = env::var("GOROROBA_SPARSE_TILE_BYTES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(policy.recommended_tile_bytes);
        return (tile_bytes / bytes_per_brick)
            .max(1)
            .min(total_active_bricks.max(1));
    }

    total_active_bricks.max(1)
}

fn sparse_bytes_per_brick() -> usize {
    let f = D3Q19_DIRS * CELLS_PER_BRICK * F32_BYTES;
    let rho = CELLS_PER_BRICK * F32_BYTES;
    let u = 3 * CELLS_PER_BRICK * F32_BYTES;
    let tau = CELLS_PER_BRICK * F32_BYTES;
    let force = 3 * CELLS_PER_BRICK * F32_BYTES;
    f + rho + u + tau + force
}

fn build_tile_windows(
    total_active_bricks: usize,
    tile_bricks: usize,
    cells_per_brick: usize,
) -> Vec<ActiveBrickWindow> {
    let mut windows = Vec::new();
    let mut tile_start_brick = 0usize;
    while tile_start_brick < total_active_bricks {
        let tile_brick_count = (total_active_bricks - tile_start_brick).min(tile_bricks);
        windows.push(ActiveBrickWindow {
            active_brick_start: tile_start_brick as u64,
            active_brick_count: tile_brick_count as u64,
            active_cell_start: (tile_start_brick * cells_per_brick) as u64,
            active_cell_count: (tile_brick_count * cells_per_brick) as u64,
        });
        tile_start_brick += tile_brick_count;
    }
    windows
}

fn prefetch_sparse_tile(inputs: SparseTilePrefetchInputs<'_>) -> Result<cudarc::driver::CudaEvent> {
    for dir in 0..D3Q19_DIRS {
        let start = dir * inputs.total_active_cells + inputs.active_cell_start;
        let view = inputs.d_f.slice(start..start + inputs.active_cell_count);
        prefetch_unified_view_to_device(&view, inputs.stream)?;
    }
    prefetch_unified_view_to_device(
        &inputs
            .d_rho
            .slice(inputs.active_cell_start..inputs.active_cell_start + inputs.active_cell_count),
        inputs.stream,
    )?;
    prefetch_unified_view_to_device(
        &inputs
            .d_u
            .slice(inputs.active_cell_start..inputs.active_cell_start + inputs.active_cell_count),
        inputs.stream,
    )?;
    prefetch_unified_view_to_device(
        &inputs.d_u.slice(
            inputs.total_active_cells + inputs.active_cell_start
                ..inputs.total_active_cells + inputs.active_cell_start + inputs.active_cell_count,
        ),
        inputs.stream,
    )?;
    prefetch_unified_view_to_device(
        &inputs.d_u.slice(
            2 * inputs.total_active_cells + inputs.active_cell_start
                ..2 * inputs.total_active_cells
                    + inputs.active_cell_start
                    + inputs.active_cell_count,
        ),
        inputs.stream,
    )?;
    prefetch_unified_view_to_device(
        &inputs
            .d_tau
            .slice(inputs.active_cell_start..inputs.active_cell_start + inputs.active_cell_count),
        inputs.stream,
    )?;
    prefetch_unified_view_to_device(
        &inputs
            .d_force
            .slice(inputs.active_cell_start..inputs.active_cell_start + inputs.active_cell_count),
        inputs.stream,
    )?;
    prefetch_unified_view_to_device(
        &inputs.d_force.slice(
            inputs.total_active_cells + inputs.active_cell_start
                ..inputs.total_active_cells + inputs.active_cell_start + inputs.active_cell_count,
        ),
        inputs.stream,
    )?;
    prefetch_unified_view_to_device(
        &inputs.d_force.slice(
            2 * inputs.total_active_cells + inputs.active_cell_start
                ..2 * inputs.total_active_cells
                    + inputs.active_cell_start
                    + inputs.active_cell_count,
        ),
        inputs.stream,
    )?;
    inputs
        .stream
        .record_event(None)
        .context("record sparse tile prefetch event")
}

fn prefetch_unified_view_to_device<T, V>(view: &V, stream: &Arc<CudaStream>) -> Result<()>
where
    V: DeviceSlice<T> + DevicePtr<T>,
{
    let location = sys::CUmemLocation {
        type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
        __bindgen_anon_1: sys::CUmemLocation_st__bindgen_ty_1 {
            id: stream.context().ordinal() as i32,
        },
    };
    let (ptr, _sync) = view.device_ptr(stream.as_ref());
    unsafe {
        result::mem_prefetch_async(
            ptr,
            view.len() * std::mem::size_of::<T>(),
            location,
            stream.cu_stream(),
        )
    }
    .context("prefetch managed sparse tile range to device")?;
    Ok(())
}

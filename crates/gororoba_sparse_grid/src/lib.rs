//! Backend-neutral sparse-grid geometry and active-window metadata.
//!
//! This crate intentionally stops at geometry, occupancy statistics, metadata
//! footprint estimates, and active-window bookkeeping. It does not encode
//! density thresholds, solver byte formulas, CUDA launch policy, or OptiX BVH
//! ownership.

/// Logical domain dimensions in cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogicalGrid3d {
    pub nx: u32,
    pub ny: u32,
    pub nz: u32,
}

impl LogicalGrid3d {
    /// Total number of logical cells in the domain.
    #[must_use]
    pub fn cell_count(self) -> u64 {
        self.nx as u64 * self.ny as u64 * self.nz as u64
    }
}

/// Brick core and halo geometry in cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BrickShape3d {
    pub core_edge_cells: u32,
    pub halo_edge_cells: u32,
}

impl BrickShape3d {
    /// Number of core cells in one brick.
    #[must_use]
    pub fn core_cell_count(self) -> u64 {
        self.core_edge_cells as u64 * self.core_edge_cells as u64 * self.core_edge_cells as u64
    }

    /// Number of halo cells in one halo-expanded brick tile.
    #[must_use]
    pub fn halo_cell_count(self) -> u64 {
        self.halo_edge_cells as u64 * self.halo_edge_cells as u64 * self.halo_edge_cells as u64
    }
}

/// Brick-grid dimensions implied by a logical domain and brick shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BrickGrid3d {
    pub logical_grid: LogicalGrid3d,
    pub brick_shape: BrickShape3d,
    pub bricks_x: u32,
    pub bricks_y: u32,
    pub bricks_z: u32,
}

impl BrickGrid3d {
    /// Build the brick-grid dimensions for a logical domain.
    #[must_use]
    pub fn from_logical_grid(logical_grid: LogicalGrid3d, brick_shape: BrickShape3d) -> Self {
        let core = brick_shape.core_edge_cells.max(1);
        Self {
            logical_grid,
            brick_shape,
            bricks_x: logical_grid.nx.div_ceil(core),
            bricks_y: logical_grid.ny.div_ceil(core),
            bricks_z: logical_grid.nz.div_ceil(core),
        }
    }

    /// Total number of logical bricks in the domain.
    #[must_use]
    pub fn total_bricks(self) -> u64 {
        self.bricks_x as u64 * self.bricks_y as u64 * self.bricks_z as u64
    }
}

/// Occupancy statistics over a sparse brick set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OccupancyBitsetStats {
    pub total_bricks: u64,
    pub active_bricks: u64,
}

impl OccupancyBitsetStats {
    /// Fraction of bricks that are active.
    #[must_use]
    pub fn occupancy_fraction(self) -> f64 {
        if self.total_bricks == 0 {
            0.0
        } else {
            self.active_bricks as f64 / self.total_bricks as f64
        }
    }

    /// Number of bytes needed for a one-bit-per-brick occupancy bitset.
    #[must_use]
    pub fn bitset_bytes(self) -> u64 {
        self.total_bricks.div_ceil(8)
    }
}

/// Shape of an indirect table that maps logical bricks to active storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndirectBrickTableShape {
    pub entry_count: u64,
    pub bytes_per_entry: u32,
}

impl IndirectBrickTableShape {
    /// Encoded byte size of the full indirect table.
    #[must_use]
    pub fn byte_len(self) -> u64 {
        self.entry_count * self.bytes_per_entry as u64
    }
}

/// One active-brick window for tiled execution or tiled metadata traversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActiveBrickWindow {
    pub active_brick_start: u64,
    pub active_brick_count: u64,
    pub active_cell_start: u64,
    pub active_cell_count: u64,
}

/// Metadata-only sparse footprint estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseMetadataFootprint {
    pub occupancy_bitset_bytes: u64,
    pub indirect_table_bytes: u64,
    pub active_brick_id_bytes: u64,
}

impl SparseMetadataFootprint {
    /// Total bytes across all metadata surfaces.
    #[must_use]
    pub fn total_bytes(self) -> u64 {
        self.occupancy_bitset_bytes + self.indirect_table_bytes + self.active_brick_id_bytes
    }
}

/// Metadata-first tile-planning summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseTilePlan {
    pub window_count: u32,
    pub peak_active_bricks_per_window: u64,
    pub recommended_tile_bytes: u64,
    pub metadata_hotset_fits_gpu_l2: Option<bool>,
}

/// Estimate metadata-only sparse footprint from occupancy and indirect-table
/// sizing choices.
#[must_use]
pub fn estimate_metadata_footprint(
    occupancy: OccupancyBitsetStats,
    indirect_table: IndirectBrickTableShape,
    active_brick_id_bytes_per_entry: u32,
) -> SparseMetadataFootprint {
    SparseMetadataFootprint {
        occupancy_bitset_bytes: occupancy.bitset_bytes(),
        indirect_table_bytes: indirect_table.byte_len(),
        active_brick_id_bytes: occupancy.active_bricks * active_brick_id_bytes_per_entry as u64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brick_grid_rounds_up_partial_edge_domains() {
        let grid = BrickGrid3d::from_logical_grid(
            LogicalGrid3d {
                nx: 1024,
                ny: 1023,
                nz: 1000,
            },
            BrickShape3d {
                core_edge_cells: 8,
                halo_edge_cells: 10,
            },
        );
        assert_eq!(grid.bricks_x, 128);
        assert_eq!(grid.bricks_y, 128);
        assert_eq!(grid.bricks_z, 125);
    }

    #[test]
    fn occupancy_stats_compute_fraction_and_bitset_size() {
        let stats = OccupancyBitsetStats {
            total_bricks: 100,
            active_bricks: 25,
        };
        assert_eq!(stats.occupancy_fraction(), 0.25);
        assert_eq!(stats.bitset_bytes(), 13);
    }

    #[test]
    fn metadata_footprint_sums_expected_bytes() {
        let footprint = estimate_metadata_footprint(
            OccupancyBitsetStats {
                total_bricks: 1024,
                active_bricks: 64,
            },
            IndirectBrickTableShape {
                entry_count: 1024,
                bytes_per_entry: 4,
            },
            4,
        );
        assert_eq!(footprint.occupancy_bitset_bytes, 128);
        assert_eq!(footprint.indirect_table_bytes, 4096);
        assert_eq!(footprint.active_brick_id_bytes, 256);
        assert_eq!(footprint.total_bytes(), 4480);
    }
}

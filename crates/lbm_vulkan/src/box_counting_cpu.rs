//! CPU reference for box-counting fractal dimension.
//!
//! WHY: The Vulkan, CUDA, and (forthcoming) cubecl GPU box-counting paths
//! need a deterministic bit-identical oracle. This module is that oracle.
//! It depends only on `std` -- no GPU dependencies, no rayon (single-threaded
//! to keep the count order deterministic across runs and machines).
//!
//! WHAT: For a 3D scalar density field `rho` of shape (nx, ny, nz) and a
//! per-axis box width `box_size`, count how many boxes contain at least
//! one cell with `rho[ix, iy, iz] > threshold`. Boxes at the right/top
//! boundary that fall partially outside the grid are still counted as long
//! as at least one in-bounds cell is occupied.
//!
//! HOW: Iterates boxes in linear `(iz, iy, ix)` order; for each box, scans
//! its cells with an early-exit on the first occupied cell. Returns the
//! integer count.
//!
//! This mirrors the WGSL kernel at `shaders/box_counting.wgsl` algorithm-
//! exactly so that CPU and GPU counts are bit-identical (the only
//! non-determinism in the WGSL version is the order of `atomicAdd` writes,
//! which is irrelevant to the final scalar count).

/// Count occupied boxes at a single box size.
///
/// # Parameters
/// - `rho`: density grid of length `nx * ny * nz`, indexed as
///   `rho[iz * ny * nx + iy * nx + ix]` (z-major as in the WGSL kernel).
/// - `threshold`: a cell is "occupied" when `rho[i] > threshold`.
/// - `nx`, `ny`, `nz`: grid dimensions in cells.
/// - `box_size`: side length of each box, in cells. Must be >= 1.
///
/// # Returns
/// Number of boxes with at least one occupied cell. The returned `u32`
/// matches the WGSL kernel's `atomic<u32>` counter exactly.
///
/// # Panics
/// Panics if `box_size == 0` or `rho.len() != nx * ny * nz`.
pub fn count_occupied_boxes(
    rho: &[f32],
    threshold: f32,
    nx: usize,
    ny: usize,
    nz: usize,
    box_size: usize,
) -> u32 {
    assert!(box_size >= 1, "box_size must be at least 1");
    assert_eq!(
        rho.len(),
        nx * ny * nz,
        "rho length {} != nx*ny*nz = {}",
        rho.len(),
        nx * ny * nz
    );

    // Number of boxes along each axis (rounded up so boundary boxes are
    // counted even when their last row of cells sticks outside the grid).
    let bx_count = nx.div_ceil(box_size);
    let by_count = ny.div_ceil(box_size);
    let bz_count = nz.div_ceil(box_size);

    let mut occupied_total: u32 = 0;
    for ibz in 0..bz_count {
        for iby in 0..by_count {
            for ibx in 0..bx_count {
                let mut occupied = false;
                'box_scan: for dz in 0..box_size {
                    for dy in 0..box_size {
                        for dx in 0..box_size {
                            let ix = ibx * box_size + dx;
                            let iy = iby * box_size + dy;
                            let iz = ibz * box_size + dz;
                            if ix < nx && iy < ny && iz < nz {
                                let cell_idx = iz * ny * nx + iy * nx + ix;
                                if rho[cell_idx] > threshold {
                                    occupied = true;
                                    break 'box_scan;
                                }
                            }
                        }
                    }
                }
                if occupied {
                    occupied_total += 1;
                }
            }
        }
    }
    occupied_total
}

/// Generate the standard sequence of box sizes for a grid of size `n` along
/// the smallest axis: powers of two from 1 up to the largest `s` for which
/// at least two boxes fit (`s <= n / 2`).
///
/// This matches the per-scale dispatch loop used by `lbm_3d_cuda::box_counting_gpu`
/// and `lbm_vulkan::box_counting_vulkan` so a parity test can re-use the
/// same scale set across all three backends.
pub fn default_box_sizes(n_min: usize) -> Vec<usize> {
    let mut sizes = Vec::new();
    let mut s = 1usize;
    while s * 2 <= n_min {
        sizes.push(s);
        s *= 2;
    }
    sizes
}

/// Compute the full (box_size, count) sweep on CPU.
///
/// This is the canonical oracle for the parity tests. The returned pairs
/// are in the same order as `default_box_sizes(min(nx, ny, nz))`.
pub fn fractal_dimension_counts_cpu(
    rho: &[f32],
    threshold: f32,
    nx: usize,
    ny: usize,
    nz: usize,
) -> Vec<(u32, u32)> {
    let n_min = nx.min(ny).min(nz);
    let sizes = default_box_sizes(n_min);
    sizes
        .into_iter()
        .map(|s| (s as u32, count_occupied_boxes(rho, threshold, nx, ny, nz, s)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fully-filled 8x8x8 cube has count = (N/s)^3 at each scale s.
    #[test]
    fn cube_fills_at_every_scale() {
        let rho = vec![1.0f32; 8 * 8 * 8];
        assert_eq!(count_occupied_boxes(&rho, 0.5, 8, 8, 8, 1), 8 * 8 * 8);
        assert_eq!(count_occupied_boxes(&rho, 0.5, 8, 8, 8, 2), 4 * 4 * 4);
        assert_eq!(count_occupied_boxes(&rho, 0.5, 8, 8, 8, 4), 2 * 2 * 2);
        assert_eq!(count_occupied_boxes(&rho, 0.5, 8, 8, 8, 8), 1);
    }

    /// An empty grid has zero occupied boxes at every scale.
    #[test]
    fn empty_grid_counts_zero() {
        let rho = vec![0.0f32; 8 * 8 * 8];
        for s in [1, 2, 4, 8].iter().copied() {
            assert_eq!(count_occupied_boxes(&rho, 0.5, 8, 8, 8, s), 0);
        }
    }

    /// A single occupied cell occupies exactly one box at every scale that
    /// can contain it.
    #[test]
    fn single_cell_one_box_at_every_scale() {
        let mut rho = vec![0.0f32; 8 * 8 * 8];
        // Cell at (1, 2, 3) — z=3, y=2, x=1.
        let cell_idx = 3 * 8 * 8 + 2 * 8 + 1;
        rho[cell_idx] = 1.0;
        for s in [1, 2, 4, 8].iter().copied() {
            assert_eq!(
                count_occupied_boxes(&rho, 0.5, 8, 8, 8, s),
                1,
                "single occupied cell must give count=1 at scale s={}",
                s
            );
        }
    }

    /// Default box sizes for a 64-cell axis: 1, 2, 4, 8, 16, 32.
    #[test]
    fn default_sizes_geometric() {
        assert_eq!(default_box_sizes(64), vec![1, 2, 4, 8, 16, 32]);
        assert_eq!(default_box_sizes(8), vec![1, 2, 4]);
        assert_eq!(default_box_sizes(1), Vec::<usize>::new());
    }

    /// Boundary box rounding-up: a 9^3 grid with box_size=4 has
    /// 3x3x3 = 27 boxes (each axis: ceil(9/4) = 3). When the grid is fully
    /// filled, all 27 are occupied.
    #[test]
    fn boundary_boxes_count_when_partially_outside() {
        let rho = vec![1.0f32; 9 * 9 * 9];
        assert_eq!(count_occupied_boxes(&rho, 0.5, 9, 9, 9, 4), 27);
    }

    /// fractal_dimension_counts_cpu emits the correct number of scales.
    #[test]
    fn full_sweep_emits_expected_scales() {
        let rho = vec![1.0f32; 16 * 16 * 16];
        let pairs = fractal_dimension_counts_cpu(&rho, 0.5, 16, 16, 16);
        assert_eq!(
            pairs.iter().map(|(s, _)| *s).collect::<Vec<_>>(),
            vec![1, 2, 4, 8]
        );
    }
}

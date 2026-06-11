//! Backend-neutral sparse D3Q19 brick-map helpers.
//!
//! The CUDA sparse path stores one 8x8x8 brick contiguously and addresses
//! active cells through `active_brick_ids` plus an indirect logical-brick
//! table. This module keeps that ABI available to Vulkan and cubecl so their
//! kernels can be parity-tested against the same direct active-brick model.

use gororoba_sparse_grid::{BrickGrid3d, BrickShape3d, LogicalGrid3d, OccupancyBitsetStats};

pub const SPARSE_BRICK_EDGE: usize = 8;
pub const SPARSE_BRICK_CELLS: usize = SPARSE_BRICK_EDGE * SPARSE_BRICK_EDGE * SPARSE_BRICK_EDGE;
pub const D3Q19_CHANNELS: usize = 19;

const CX: [i32; D3Q19_CHANNELS] = [0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0];
const CY: [i32; D3Q19_CHANNELS] = [0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1];
const CZ: [i32; D3Q19_CHANNELS] = [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1];
const OPP: [usize; D3Q19_CHANNELS] = [
    0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17,
];
const W: [f32; D3Q19_CHANNELS] = [
    1.0 / 3.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
];

#[derive(Debug, thiserror::Error)]
pub enum SparseLbmError {
    #[error("grid dimensions must all be positive (got nx={nx}, ny={ny}, nz={nz})")]
    EmptyGrid { nx: usize, ny: usize, nz: usize },
    #[error("geometry mask length {got} does not match nx*ny*nz = {expected}")]
    GeometryLengthMismatch { got: usize, expected: usize },
    #[error("f slice length {got} does not match 19*active_bricks*512 = {expected}")]
    DistributionLengthMismatch { got: usize, expected: usize },
    #[error("tau must satisfy tau > 0.5 for BGK stability (got {0})")]
    UnstableTau(f32),
    #[error("sparse grid has no active bricks")]
    NoActiveBricks,
}

#[derive(Clone, Debug)]
pub struct SparseLbmPlan {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub bricks_x: usize,
    pub bricks_y: usize,
    pub bricks_z: usize,
    pub active_brick_ids: Vec<u32>,
    pub indirect_table: Vec<i32>,
}

impl SparseLbmPlan {
    pub fn from_geometry_mask(
        nx: usize,
        ny: usize,
        nz: usize,
        geometry_mask: &[u8],
    ) -> Result<Self, SparseLbmError> {
        if nx == 0 || ny == 0 || nz == 0 {
            return Err(SparseLbmError::EmptyGrid { nx, ny, nz });
        }
        let expected = nx * ny * nz;
        if geometry_mask.len() != expected {
            return Err(SparseLbmError::GeometryLengthMismatch {
                got: geometry_mask.len(),
                expected,
            });
        }
        let brick_grid = BrickGrid3d::from_logical_grid(
            LogicalGrid3d {
                nx: nx as u32,
                ny: ny as u32,
                nz: nz as u32,
            },
            BrickShape3d {
                core_edge_cells: SPARSE_BRICK_EDGE as u32,
                halo_edge_cells: (SPARSE_BRICK_EDGE + 2) as u32,
            },
        );
        let bricks_x = brick_grid.bricks_x as usize;
        let bricks_y = brick_grid.bricks_y as usize;
        let bricks_z = brick_grid.bricks_z as usize;
        let n_bricks = bricks_x * bricks_y * bricks_z;
        let mut active_flags = vec![false; n_bricks];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let cell = z * nx * ny + y * nx + x;
                    if geometry_mask[cell] != 0 {
                        let bx = x / SPARSE_BRICK_EDGE;
                        let by = y / SPARSE_BRICK_EDGE;
                        let bz = z / SPARSE_BRICK_EDGE;
                        active_flags[brick_id(bx, by, bz, bricks_x, bricks_y)] = true;
                    }
                }
            }
        }
        let mut indirect_table = vec![-1_i32; n_bricks];
        let mut active_brick_ids = Vec::new();
        for (brick, active) in active_flags.into_iter().enumerate() {
            if active {
                indirect_table[brick] = active_brick_ids.len() as i32;
                active_brick_ids.push(brick as u32);
            }
        }
        Ok(Self {
            nx,
            ny,
            nz,
            bricks_x,
            bricks_y,
            bricks_z,
            active_brick_ids,
            indirect_table,
        })
    }

    #[must_use]
    pub fn n_active_bricks(&self) -> usize {
        self.active_brick_ids.len()
    }

    #[must_use]
    pub fn n_active_cells(&self) -> usize {
        self.n_active_bricks() * SPARSE_BRICK_CELLS
    }

    #[must_use]
    pub fn occupancy_stats(&self) -> OccupancyBitsetStats {
        OccupancyBitsetStats {
            total_bricks: self.indirect_table.len() as u64,
            active_bricks: self.active_brick_ids.len() as u64,
        }
    }

    pub fn validate_f_len(&self, f: &[f32]) -> Result<(), SparseLbmError> {
        let expected = self.n_active_cells() * D3Q19_CHANNELS;
        if f.len() != expected {
            return Err(SparseLbmError::DistributionLengthMismatch {
                got: f.len(),
                expected,
            });
        }
        Ok(())
    }

    #[must_use]
    pub fn equilibrium_at_rest(&self) -> Vec<f32> {
        let n = self.n_active_cells();
        let mut f = vec![0.0_f32; D3Q19_CHANNELS * n];
        for i in 0..D3Q19_CHANNELS {
            for cell in 0..n {
                f[i * n + cell] = W[i];
            }
        }
        f
    }
}

pub fn evolve_sparse_d3q19_cpu(
    plan: &SparseLbmPlan,
    tau: f32,
    f_init: &[f32],
    num_steps: usize,
) -> Result<Vec<f32>, SparseLbmError> {
    validate_sparse_evolution(plan, tau, f_init)?;
    let mut f = f_init.to_vec();
    let n = plan.n_active_cells();
    let inv_tau = 1.0_f32 / tau;
    for step in 0..num_steps {
        let parity = step & 1;
        for tid in 0..n {
            let Some(coords) = SparseCellCoords::from_tid(plan, tid) else {
                continue;
            };
            let (mut local, state) = read_sparse_cell_state(plan, &f, n, parity, coords);
            collide_sparse_cell_state(&mut local, state, inv_tau);
            write_sparse_cell_state(plan, &mut f, n, parity, coords, &local);
        }
    }
    Ok(f)
}

fn validate_sparse_evolution(
    plan: &SparseLbmPlan,
    tau: f32,
    f_init: &[f32],
) -> Result<(), SparseLbmError> {
    if tau.is_nan() || tau <= 0.5 {
        return Err(SparseLbmError::UnstableTau(tau));
    }
    if plan.n_active_bricks() == 0 {
        return Err(SparseLbmError::NoActiveBricks);
    }
    plan.validate_f_len(f_init)
}

#[derive(Clone, Copy)]
struct SparseCellCoords {
    tid: usize,
    x: usize,
    y: usize,
    z: usize,
}

impl SparseCellCoords {
    fn from_tid(plan: &SparseLbmPlan, tid: usize) -> Option<Self> {
        let (x, y, z) = logical_cell_for_tid(plan, tid)?;
        Some(Self { tid, x, y, z })
    }
}

#[derive(Clone, Copy)]
struct SparseCellState {
    rho: f32,
    ux: f32,
    uy: f32,
    uz: f32,
}

fn read_sparse_cell_state(
    plan: &SparseLbmPlan,
    f: &[f32],
    n: usize,
    parity: usize,
    coords: SparseCellCoords,
) -> ([f32; D3Q19_CHANNELS], SparseCellState) {
    let mut moments = SparseCellMoments::default();
    let mut local = [0.0_f32; D3Q19_CHANNELS];
    for (i, local_fi) in local.iter_mut().enumerate() {
        let (read_dir, src_tid) = read_source(plan, parity, coords, i);
        let fi = finite_or_zero(f[read_dir * n + src_tid]);
        *local_fi = fi;
        moments.accumulate(i, fi);
    }
    (local, moments.into_state())
}

#[derive(Default)]
struct SparseCellMoments {
    rho: f32,
    mx: f32,
    my: f32,
    mz: f32,
}

impl SparseCellMoments {
    fn accumulate(&mut self, i: usize, fi: f32) {
        self.rho += fi;
        self.mx += CX[i] as f32 * fi;
        self.my += CY[i] as f32 * fi;
        self.mz += CZ[i] as f32 * fi;
    }

    fn into_state(self) -> SparseCellState {
        if self.rho.is_finite() && self.rho > 1.0e-20 {
            let inv_rho = 1.0 / self.rho;
            SparseCellState {
                rho: self.rho,
                ux: self.mx * inv_rho,
                uy: self.my * inv_rho,
                uz: self.mz * inv_rho,
            }
        } else {
            SparseCellState {
                rho: 1.0,
                ux: 0.0,
                uy: 0.0,
                uz: 0.0,
            }
        }
    }
}

fn read_source(
    plan: &SparseLbmPlan,
    parity: usize,
    coords: SparseCellCoords,
    i: usize,
) -> (usize, usize) {
    if parity == 0 {
        return (i, coords.tid);
    }
    neighbor_tid(plan, coords.x, coords.y, coords.z, -CX[i], -CY[i], -CZ[i])
        .map_or((i, coords.tid), |src_tid| (OPP[i], src_tid))
}

fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
}

fn collide_sparse_cell_state(
    local: &mut [f32; D3Q19_CHANNELS],
    state: SparseCellState,
    inv_tau: f32,
) {
    let u_sq = state.ux * state.ux + state.uy * state.uy + state.uz * state.uz;
    let base = 1.0 - 1.5 * u_sq;
    for i in 0..D3Q19_CHANNELS {
        let eu = CX[i] as f32 * state.ux + CY[i] as f32 * state.uy + CZ[i] as f32 * state.uz;
        let f_eq = W[i] * state.rho * (base + 3.0 * eu + 4.5 * eu * eu);
        local[i] -= (local[i] - f_eq) * inv_tau;
    }
}

fn write_sparse_cell_state(
    plan: &SparseLbmPlan,
    f: &mut [f32],
    n: usize,
    parity: usize,
    coords: SparseCellCoords,
    local: &[f32; D3Q19_CHANNELS],
) {
    for i in 0..D3Q19_CHANNELS {
        if parity == 0 {
            let write_tid = neighbor_tid(plan, coords.x, coords.y, coords.z, CX[i], CY[i], CZ[i])
                .unwrap_or(coords.tid);
            f[OPP[i] * n + write_tid] = local[i];
        } else {
            f[i * n + coords.tid] = local[i];
        }
    }
}

fn logical_cell_for_tid(plan: &SparseLbmPlan, tid: usize) -> Option<(usize, usize, usize)> {
    let pool_idx = tid / SPARSE_BRICK_CELLS;
    let local_idx = tid % SPARSE_BRICK_CELLS;
    let brick = *plan.active_brick_ids.get(pool_idx)? as usize;
    let bx = brick % plan.bricks_x;
    let by = (brick / plan.bricks_x) % plan.bricks_y;
    let bz = brick / (plan.bricks_x * plan.bricks_y);
    let lx = local_idx % SPARSE_BRICK_EDGE;
    let ly = (local_idx / SPARSE_BRICK_EDGE) % SPARSE_BRICK_EDGE;
    let lz = local_idx / (SPARSE_BRICK_EDGE * SPARSE_BRICK_EDGE);
    let x = bx * SPARSE_BRICK_EDGE + lx;
    let y = by * SPARSE_BRICK_EDGE + ly;
    let z = bz * SPARSE_BRICK_EDGE + lz;
    if x < plan.nx && y < plan.ny && z < plan.nz {
        Some((x, y, z))
    } else {
        None
    }
}

fn neighbor_tid(
    plan: &SparseLbmPlan,
    x: usize,
    y: usize,
    z: usize,
    dx: i32,
    dy: i32,
    dz: i32,
) -> Option<usize> {
    let nx = plan.nx as i32;
    let ny = plan.ny as i32;
    let nz = plan.nz as i32;
    let xn = wrap_i32(x as i32 + dx, nx) as usize;
    let yn = wrap_i32(y as i32 + dy, ny) as usize;
    let zn = wrap_i32(z as i32 + dz, nz) as usize;
    let bx = xn / SPARSE_BRICK_EDGE;
    let by = yn / SPARSE_BRICK_EDGE;
    let bz = zn / SPARSE_BRICK_EDGE;
    let brick = brick_id(bx, by, bz, plan.bricks_x, plan.bricks_y);
    let pool = *plan.indirect_table.get(brick)?;
    if pool < 0 {
        return None;
    }
    let lx = xn % SPARSE_BRICK_EDGE;
    let ly = yn % SPARSE_BRICK_EDGE;
    let lz = zn % SPARSE_BRICK_EDGE;
    Some(
        pool as usize * SPARSE_BRICK_CELLS + lx + SPARSE_BRICK_EDGE * (ly + SPARSE_BRICK_EDGE * lz),
    )
}

fn wrap_i32(value: i32, modulus: i32) -> i32 {
    let mut wrapped = value % modulus;
    if wrapped < 0 {
        wrapped += modulus;
    }
    wrapped
}

fn brick_id(bx: usize, by: usize, bz: usize, bricks_x: usize, bricks_y: usize) -> usize {
    bx + bricks_x * (by + bricks_y * bz)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_marks_active_bricks_from_geometry() {
        let mut mask = vec![0u8; 16 * 8 * 8];
        mask[0] = 1;
        mask[15] = 1;
        let plan = SparseLbmPlan::from_geometry_mask(16, 8, 8, &mask).unwrap();
        assert_eq!(plan.active_brick_ids, vec![0, 1]);
        assert_eq!(plan.occupancy_stats().active_bricks, 2);
    }

    #[test]
    fn cpu_sparse_rest_equilibrium_is_stable() {
        let mask = vec![1u8; 8 * 8 * 8];
        let plan = SparseLbmPlan::from_geometry_mask(8, 8, 8, &mask).unwrap();
        let f0 = plan.equilibrium_at_rest();
        let out = evolve_sparse_d3q19_cpu(&plan, 1.0, &f0, 2).unwrap();
        for (got, expected) in out.iter().zip(f0.iter()) {
            assert!((got - expected).abs() < 2.0e-7);
        }
    }
}

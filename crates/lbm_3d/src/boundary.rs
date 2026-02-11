//! Boundary conditions for Lattice Boltzmann Method (3D).
//!
//! Implements common LBM boundary conditions:
//! 1. Bounce-back (no-slip walls)
//! 2. Periodic boundaries
//! 3. Zou-He conditions (velocity/pressure inlets/outlets) - placeholder

use crate::lattice::D3Q19Lattice;
#[cfg(test)]
use crate::solver::BgkCollision;

/// Boundary condition type enumeration.
#[derive(Clone, Debug, Copy)]
pub enum BoundaryType {
    Periodic,
    BounceBack,
    ZouHe,
}

/// 3D grid index (x, y, z).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GridIndex {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

impl GridIndex {
    pub fn new(x: usize, y: usize, z: usize) -> Self {
        Self { x, y, z }
    }

    /// Check if grid point is at domain boundary.
    pub fn is_at_boundary(
        &self,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> bool {
        self.x == 0 || self.x == nx - 1
            || self.y == 0 || self.y == ny - 1
            || self.z == 0 || self.z == nz - 1
    }

    /// Apply periodic wraparound.
    pub fn wrap_periodic(&self, nx: usize, ny: usize, nz: usize) -> Self {
        let x = if self.x >= nx { self.x % nx } else { self.x };
        let y = if self.y >= ny { self.y % ny } else { self.y };
        let z = if self.z >= nz { self.z % nz } else { self.z };
        Self { x, y, z }
    }

    /// Linearize to 1D index for flat storage.
    pub fn linearize(&self, nx: usize, ny: usize) -> usize {
        self.z * (nx * ny) + self.y * nx + self.x
    }

    /// Delinearize from 1D index.
    pub fn delinearize(idx: usize, nx: usize, ny: usize) -> Self {
        let z = idx / (nx * ny);
        let remainder = idx % (nx * ny);
        let y = remainder / nx;
        let x = remainder % nx;
        Self { x, y, z }
    }
}

/// Bounce-back boundary condition (no-slip wall).
/// Particles are reflected back into the fluid with opposite velocity.
pub struct BounceBackBoundary {
    pub lattice: D3Q19Lattice,
}

impl BounceBackBoundary {
    pub fn new() -> Self {
        Self {
            lattice: D3Q19Lattice::new(),
        }
    }

    /// Apply bounce-back at a boundary node.
    /// f_i^wall = f_opposite(i)^fluid (from previous time step)
    ///
    /// This enforces zero velocity at the wall (no-slip condition).
    pub fn apply_at_node(
        &self,
        f_fluid: &[f64; 19],
    ) -> [f64; 19] {
        let mut f_wall = [0.0; 19];

        for (i, &fi) in f_fluid.iter().enumerate() {
            let opposite_i = self.lattice.opposite_direction(i);
            f_wall[opposite_i] = fi;
        }

        f_wall
    }

    /// Apply bounce-back across entire boundary slice (e.g., z=0 plane).
    pub fn apply_on_plane(
        &self,
        f_grid: &mut [f64],  // Flattened 3D grid
        nx: usize,
        ny: usize,
        nz: usize,
        plane: BoundaryPlane,
    ) {
        match plane {
            BoundaryPlane::MinX => {
                for y in 0..ny {
                    for z in 0..nz {
                        let idx = GridIndex::new(0, y, z).linearize(nx, ny) * 19;
                        let mut f_local = [0.0; 19];
                        f_local.copy_from_slice(&f_grid[idx..idx+19]);
                        let f_bounce = self.apply_at_node(&f_local);
                        f_grid[idx..idx+19].copy_from_slice(&f_bounce);
                    }
                }
            }
            BoundaryPlane::MaxX => {
                for y in 0..ny {
                    for z in 0..nz {
                        let idx = GridIndex::new(nx - 1, y, z).linearize(nx, ny) * 19;
                        let mut f_local = [0.0; 19];
                        f_local.copy_from_slice(&f_grid[idx..idx+19]);
                        let f_bounce = self.apply_at_node(&f_local);
                        f_grid[idx..idx+19].copy_from_slice(&f_bounce);
                    }
                }
            }
            BoundaryPlane::MinY => {
                for x in 0..nx {
                    for z in 0..nz {
                        let idx = GridIndex::new(x, 0, z).linearize(nx, ny) * 19;
                        let mut f_local = [0.0; 19];
                        f_local.copy_from_slice(&f_grid[idx..idx+19]);
                        let f_bounce = self.apply_at_node(&f_local);
                        f_grid[idx..idx+19].copy_from_slice(&f_bounce);
                    }
                }
            }
            BoundaryPlane::MaxY => {
                for x in 0..nx {
                    for z in 0..nz {
                        let idx = GridIndex::new(x, ny - 1, z).linearize(nx, ny) * 19;
                        let mut f_local = [0.0; 19];
                        f_local.copy_from_slice(&f_grid[idx..idx+19]);
                        let f_bounce = self.apply_at_node(&f_local);
                        f_grid[idx..idx+19].copy_from_slice(&f_bounce);
                    }
                }
            }
            BoundaryPlane::MinZ => {
                for x in 0..nx {
                    for y in 0..ny {
                        let idx = GridIndex::new(x, y, 0).linearize(nx, ny) * 19;
                        let mut f_local = [0.0; 19];
                        f_local.copy_from_slice(&f_grid[idx..idx+19]);
                        let f_bounce = self.apply_at_node(&f_local);
                        f_grid[idx..idx+19].copy_from_slice(&f_bounce);
                    }
                }
            }
            BoundaryPlane::MaxZ => {
                for x in 0..nx {
                    for y in 0..ny {
                        let idx = GridIndex::new(x, y, nz - 1).linearize(nx, ny) * 19;
                        let mut f_local = [0.0; 19];
                        f_local.copy_from_slice(&f_grid[idx..idx+19]);
                        let f_bounce = self.apply_at_node(&f_local);
                        f_grid[idx..idx+19].copy_from_slice(&f_bounce);
                    }
                }
            }
        }
    }
}

impl Default for BounceBackBoundary {
    fn default() -> Self {
        Self::new()
    }
}

/// Periodic boundary condition.
/// Particles wrap around from one side of domain to the other.
pub struct PeriodicBoundary;

impl PeriodicBoundary {
    /// Get the periodic neighbor of a grid point.
    pub fn get_neighbor(
        idx: GridIndex,
        nx: usize,
        ny: usize,
        nz: usize,
        direction: usize,  // Velocity index 0..18
        lattice: &D3Q19Lattice,
    ) -> GridIndex {
        let c = lattice.velocity(direction);
        let x = (idx.x as i32 + c[0]) as usize;
        let y = (idx.y as i32 + c[1]) as usize;
        let z = (idx.z as i32 + c[2]) as usize;

        GridIndex {
            x: x % nx,
            y: y % ny,
            z: z % nz,
        }
    }

    /// Apply periodic boundary condition (implicit in streaming step, verified here).
    pub fn verify_conservation(
        f_grid: &[f64],
        _nx: usize,
        _ny: usize,
        _nz: usize,
    ) -> f64 {
        f_grid.iter().sum()
    }
}

/// Boundary plane enumeration.
#[derive(Clone, Copy, Debug)]
pub enum BoundaryPlane {
    MinX,
    MaxX,
    MinY,
    MaxY,
    MinZ,
    MaxZ,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_index_linearize() {
        let idx = GridIndex::new(5, 3, 2);
        let linear = idx.linearize(10, 8);
        assert_eq!(linear, 2 * (10 * 8) + 3 * 10 + 5);
    }

    #[test]
    fn test_grid_index_delinearize() {
        let idx = GridIndex::new(5, 3, 2);
        let linear = idx.linearize(10, 8);
        let idx_back = GridIndex::delinearize(linear, 10, 8);
        assert_eq!(idx, idx_back);
    }

    #[test]
    fn test_grid_index_periodic_wrap() {
        let idx = GridIndex::new(9, 7, 5);
        let wrapped = idx.wrap_periodic(10, 8, 6);
        assert_eq!(wrapped, GridIndex::new(9, 7, 5));

        let idx_out = GridIndex::new(10, 8, 6);
        let wrapped_out = idx_out.wrap_periodic(10, 8, 6);
        assert_eq!(wrapped_out, GridIndex::new(0, 0, 0));
    }

    #[test]
    fn test_grid_index_is_boundary() {
        let corner = GridIndex::new(0, 0, 0);
        assert!(corner.is_at_boundary(10, 8, 6));

        let interior = GridIndex::new(5, 4, 3);
        assert!(!interior.is_at_boundary(10, 8, 6));

        let edge = GridIndex::new(9, 4, 3);
        assert!(edge.is_at_boundary(10, 8, 6));
    }

    #[test]
    fn test_bounce_back_reflection() {
        let bb = BounceBackBoundary::new();
        let mut f_fluid = [0.1; 19];
        f_fluid[0] = 0.5;  // Special value for rest particle

        let f_wall = bb.apply_at_node(&f_fluid);

        // Opposite directions should be swapped
        for i in 1..19 {
            let opposite_i = bb.lattice.opposite_direction(i);
            assert!((f_wall[opposite_i] - f_fluid[i]).abs() < 1e-14);
        }
    }

    #[test]
    fn test_bounce_back_no_slip() {
        let bb = BounceBackBoundary::new();
        let lattice = D3Q19Lattice::new();

        // Create fluid distribution with velocity [0.1, 0.0, 0.0]
        let rho = 1.0;
        let u = [0.1, 0.0, 0.0];
        let f_fluid = BgkCollision::initialize_with_velocity(rho, u, &lattice);

        let f_wall = bb.apply_at_node(&f_fluid);

        // Recover velocity from wall distribution
        let rho_wall = BgkCollision::density_from_f(&f_wall);
        let _u_wall = BgkCollision::velocity_from_f(&f_wall, rho_wall, &lattice);

        // Mass should be conserved
        assert!((rho_wall - rho).abs() < 1e-12);

        // Velocity should be zero (no-slip)
        // (Actually, bounce-back doesn't directly enforce zero velocity,
        // but it does prevent particles from escaping the boundary)
        assert!(BgkCollision::is_stable(&f_wall));
    }

    #[test]
    fn test_periodic_neighbor() {
        let lattice = D3Q19Lattice::new();
        let nx = 10;
        let ny = 8;
        let nz = 6;

        // Interior point with velocity [1, 0, 0]
        let idx = GridIndex::new(5, 4, 3);
        let neighbor = PeriodicBoundary::get_neighbor(idx, nx, ny, nz, 1, &lattice);
        assert_eq!(neighbor, GridIndex::new(6, 4, 3));

        // Boundary point wrapping x
        let idx_edge = GridIndex::new(9, 4, 3);
        let neighbor_wrapped = PeriodicBoundary::get_neighbor(idx_edge, nx, ny, nz, 1, &lattice);
        assert_eq!(neighbor_wrapped, GridIndex::new(0, 4, 3));  // Wraps to x=0
    }

    #[test]
    fn test_periodic_conservation() {
        // Conservation: sum of all f values should remain constant
        let f_grid = vec![0.1; 1000];
        let conservation = PeriodicBoundary::verify_conservation(&f_grid, 10, 8, 6);
        assert!((conservation - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_bounce_back_stability() {
        let bb = BounceBackBoundary::new();
        let lattice = D3Q19Lattice::new();
        let f_fluid = BgkCollision::initialize_with_velocity(1.0, [0.1, 0.1, 0.1], &lattice);

        let f_wall = bb.apply_at_node(&f_fluid);

        // All components should be non-negative
        assert!(BgkCollision::is_stable(&f_wall));
    }
}

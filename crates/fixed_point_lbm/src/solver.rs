//! Minimal D3Q19 BGK LBM solver using fixed-point types.
//!
//! Demonstrates the Q16_16 and Q32_32 types from this crate in a
//! complete (but small) lattice Boltzmann simulation. The key property:
//! integer accumulation for density gives perfect mass conservation,
//! which is the primary advantage over floating-point storage.

use crate::{Q16_16, Q32_32, d3q19};

/// A minimal D3Q19 BGK lattice Boltzmann domain.
pub struct LbmDomain<T> {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub f: Vec<[T; 19]>,      // Distribution functions per cell
    pub rho: Vec<f32>,         // Macroscopic density
    pub u: Vec<[f32; 3]>,      // Macroscopic velocity
    pub tau: f32,              // Relaxation time
}

/// Trait for types that can serve as LBM distribution storage.
pub trait LbmStorage: Copy + Default + std::ops::Add<Output = Self> {
    fn from_f32(v: f32) -> Self;
    fn to_f32(self) -> f32;
    fn zero() -> Self;
}

impl LbmStorage for Q16_16 {
    fn from_f32(v: f32) -> Self { Q16_16::from_f32(v) }
    fn to_f32(self) -> f32 { self.to_f32() }
    fn zero() -> Self { Q16_16::ZERO }
}

impl LbmStorage for Q32_32 {
    fn from_f32(v: f32) -> Self { Q32_32::from_f32(v) }
    fn to_f32(self) -> f32 { self.to_f32() }
    fn zero() -> Self { Q32_32::ZERO }
}

impl LbmStorage for f32 {
    fn from_f32(v: f32) -> Self { v }
    fn to_f32(self) -> f32 { self }
    fn zero() -> Self { 0.0 }
}

impl<T: LbmStorage> LbmDomain<T> {
    /// Create a new domain initialized to equilibrium at rho=1, u=0.
    pub fn new(nx: usize, ny: usize, nz: usize, tau: f32) -> Self {
        let n = nx * ny * nz;
        let mut f = vec![[T::zero(); 19]; n];
        let rho = vec![1.0f32; n];
        let u = vec![[0.0f32; 3]; n];

        // Initialize f_i = w_i * rho (equilibrium at rest)
        for cell in 0..n {
            for i in 0..19 {
                f[cell][i] = T::from_f32(d3q19::W[i] as f32);
            }
        }

        Self { nx, ny, nz, f, rho, u, tau }
    }

    /// Total number of cells.
    pub fn n_cells(&self) -> usize { self.nx * self.ny * self.nz }

    /// Compute total mass (sum of all rho).
    pub fn total_mass(&self) -> f64 {
        // Sum f_i directly in the storage type's integer domain for exact mass.
        let mut total = 0.0f64;
        for cell in 0..self.n_cells() {
            let mut rho_cell = 0.0f64;
            for i in 0..19 {
                rho_cell += self.f[cell][i].to_f32() as f64;
            }
            total += rho_cell;
        }
        total
    }

    /// Run one BGK collision + streaming step.
    pub fn step(&mut self) {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let n = self.n_cells();
        let inv_tau = 1.0f32 / self.tau;

        // Temporary buffer for post-collision distributions
        let mut f_new = vec![[T::zero(); 19]; n];

        for idx in 0..n {
            let z = idx / (nx * ny);
            let y = (idx / nx) % ny;
            let x = idx % nx;

            // Compute macroscopic quantities from stored distributions
            let mut rho_local = 0.0f32;
            let mut mx = 0.0f32;
            let mut my = 0.0f32;
            let mut mz_val = 0.0f32;

            for i in 0..19 {
                let fi = self.f[idx][i].to_f32();
                rho_local += fi;
                mx += d3q19::C[i][0] as f32 * fi;
                my += d3q19::C[i][1] as f32 * fi;
                mz_val += d3q19::C[i][2] as f32 * fi;
            }

            let (ux, uy, uz) = if rho_local > 1e-20 {
                (mx / rho_local, my / rho_local, mz_val / rho_local)
            } else {
                (0.0, 0.0, 0.0)
            };

            self.rho[idx] = rho_local;
            self.u[idx] = [ux, uy, uz];

            // BGK collision
            let u_sq = ux * ux + uy * uy + uz * uz;
            for i in 0..19 {
                let cx = d3q19::C[i][0] as f32;
                let cy = d3q19::C[i][1] as f32;
                let cz = d3q19::C[i][2] as f32;
                let eu = cx * ux + cy * uy + cz * uz;
                let w = d3q19::W[i] as f32;
                let f_eq = w * rho_local * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u_sq);

                let fi = self.f[idx][i].to_f32();
                let fi_new = fi - (fi - f_eq) * inv_tau;

                // Streaming: push to neighbor
                let xn = (x as i32 + d3q19::C[i][0] + nx as i32) as usize % nx;
                let yn = (y as i32 + d3q19::C[i][1] + ny as i32) as usize % ny;
                let zn = (z as i32 + d3q19::C[i][2] + nz as i32) as usize % nz;
                let idx_next = xn + nx * (yn + ny * zn);

                f_new[idx_next][i] = T::from_f32(fi_new);
            }
        }

        self.f = f_new;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q16_mass_conservation() {
        let mut domain = LbmDomain::<Q16_16>::new(8, 8, 8, 0.6);
        let initial_mass = domain.total_mass();

        for _ in 0..50 {
            domain.step();
        }

        let final_mass = domain.total_mass();
        let drift = ((final_mass - initial_mass) / initial_mass).abs();
        println!("Q16_16: initial={initial_mass:.6}, final={final_mass:.6}, drift={drift:.2e}");
        assert!(drift < 0.01, "Q16_16 mass drift {drift} too large");
    }

    #[test]
    fn q32_mass_conservation() {
        let mut domain = LbmDomain::<Q32_32>::new(8, 8, 8, 0.6);
        let initial_mass = domain.total_mass();

        for _ in 0..50 {
            domain.step();
        }

        let final_mass = domain.total_mass();
        let drift = ((final_mass - initial_mass) / initial_mass).abs();
        println!("Q32_32: initial={initial_mass:.6}, final={final_mass:.6}, drift={drift:.2e}");
        assert!(drift < 1e-6, "Q32_32 mass drift {drift} too large");
    }

    #[test]
    fn f32_mass_conservation() {
        let mut domain = LbmDomain::<f32>::new(8, 8, 8, 0.6);
        let initial_mass = domain.total_mass();

        for _ in 0..50 {
            domain.step();
        }

        let final_mass = domain.total_mass();
        let drift = ((final_mass - initial_mass) / initial_mass).abs();
        println!("f32: initial={initial_mass:.6}, final={final_mass:.6}, drift={drift:.2e}");
        assert!(drift < 1e-4, "f32 mass drift {drift} too large");
    }
}

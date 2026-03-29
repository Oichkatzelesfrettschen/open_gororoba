//! Primitive variables for ideal GRMHD.
//!
//! The 8 primitive variables are:
//!   P[0] = rho     (rest-mass density)
//!   P[1] = u       (internal energy density)
//!   P[2] = v1      (contravariant 3-velocity, r-direction)
//!   P[3] = v2      (contravariant 3-velocity, theta-direction)
//!   P[4] = v3      (contravariant 3-velocity, phi-direction)
//!   P[5] = B1      (magnetic field, r-component)
//!   P[6] = B2      (magnetic field, theta-component)
//!   P[7] = B3      (magnetic field, phi-component)
//!
//! The 3-velocity v^i is defined as u^i / u^t where u^mu is the 4-velocity.
//! B^i is the magnetic field in the fluid frame (actually the lab-frame
//! spatial components of the dual of the Faraday tensor).

/// Number of primitive variables.
pub const NPRIM: usize = 8;

/// Indices into the primitive array.
pub const RHO: usize = 0;
pub const UU: usize = 1;
pub const V1: usize = 2;
pub const V2: usize = 3;
pub const V3: usize = 4;
pub const B1: usize = 5;
pub const B2: usize = 6;
pub const B3: usize = 7;

/// A single cell's primitive variables.
pub type Prim = [f64; NPRIM];

/// State vector for the full grid.
/// Stored as a flat array: data[idx * NPRIM + var].
pub struct PrimGrid {
    pub data: Vec<f64>,
    pub n_cells: usize,
}

impl PrimGrid {
    pub fn new(n_cells: usize) -> Self {
        Self {
            data: vec![0.0; n_cells * NPRIM],
            n_cells,
        }
    }

    /// Get a reference to the primitive array at cell `idx`.
    #[inline]
    pub fn get(&self, idx: usize) -> &[f64] {
        let start = idx * NPRIM;
        &self.data[start..start + NPRIM]
    }

    /// Get a mutable reference to the primitive array at cell `idx`.
    #[inline]
    pub fn get_mut(&mut self, idx: usize) -> &mut [f64] {
        let start = idx * NPRIM;
        &mut self.data[start..start + NPRIM]
    }

    /// Set the primitive variables at cell `idx`.
    #[inline]
    pub fn set(&mut self, idx: usize, p: &Prim) {
        let start = idx * NPRIM;
        self.data[start..start + NPRIM].copy_from_slice(p);
    }

    /// Apply density and internal energy floors.
    pub fn apply_floors(&mut self, rho_floor: f64, u_floor: f64) {
        for idx in 0..self.n_cells {
            let start = idx * NPRIM;
            if self.data[start + RHO] < rho_floor {
                self.data[start + RHO] = rho_floor;
            }
            if self.data[start + UU] < u_floor {
                self.data[start + UU] = u_floor;
            }
        }
    }

    /// Apply B-field ceiling to prevent div(B)-driven runaway.
    /// Caps each component at +/- b_max. This is a pragmatic stability
    /// measure for the mini-solver, not a substitute for proper CT.
    pub fn apply_b_ceiling(&mut self, b_max: f64) {
        for idx in 0..self.n_cells {
            let start = idx * NPRIM;
            for b in [B1, B2, B3] {
                let val = self.data[start + b];
                if val > b_max { self.data[start + b] = b_max; }
                if val < -b_max { self.data[start + b] = -b_max; }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prim_grid() {
        let mut pg = PrimGrid::new(100);
        let p: Prim = [1.0, 0.1, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0];
        pg.set(50, &p);
        assert_eq!(pg.get(50)[RHO], 1.0);
        assert_eq!(pg.get(50)[B1], 0.01);
    }

    #[test]
    fn test_floors() {
        let mut pg = PrimGrid::new(10);
        pg.set(0, &[1e-20, 1e-20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        pg.apply_floors(1e-6, 1e-8);
        assert!(pg.get(0)[RHO] >= 1e-6);
        assert!(pg.get(0)[UU] >= 1e-8);
    }
}

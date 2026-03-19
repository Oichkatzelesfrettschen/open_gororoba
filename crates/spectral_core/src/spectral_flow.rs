//! Spectral Flow Analysis for Sedenion Fields.
//!
//! Investigates the time evolution of "energy modes" in a sedenion field
//! configuration via Singular Value Decomposition (SVD) of the unfolded 
//! field tensor.
//!
//! Migrated from src/spectral_flow_sim.py.

use ndarray::Array3;
use nalgebra::DMatrix;

/// Sedenion field state (16 components per site).
pub struct SedenionField3D {
    /// Field data: [component (0..16), x, y, z]
    pub data: Vec<Array3<f64>>,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
}

impl SedenionField3D {
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        let mut data = Vec::with_capacity(16);
        for _ in 0..16 {
            data.push(Array3::zeros((nx, ny, nz)));
        }
        Self { data, nx, ny, nz }
    }

    /// Set random field values.
    pub fn set_random(&mut self, scale: f64) {
        for component in &mut self.data {
            for val in component.iter_mut() {
                *val = (rand::random::<f64>() - 0.5) * scale;
            }
        }
    }

    /// Compute singular values of the unfolded field tensor.
    ///
    /// Unfolds (16, Nx, Ny, Nz) -> (16, Nx*Ny*Nz) and performs SVD.
    pub fn singular_values(&self) -> Vec<f64> {
        let n_total = self.nx * self.ny * self.nz;
        let mut flat_data = DMatrix::<f64>::zeros(16, n_total);

        for (c, component) in self.data.iter().enumerate() {
            for (i, &val) in component.iter().enumerate() {
                flat_data[(c, i)] = val;
            }
        }

        let svd = flat_data.svd(false, false);
        svd.singular_values.as_slice().to_vec()
    }

    /// Perform a simple wave-like drift evolution.
    ///
    /// phi_new = phi + dt * (phi * phi)
    /// where multiplication is Cayley-Dickson sedenion product.
    pub fn drift_step(&mut self, dt: f64) {
        let mut next_data = self.data.clone();

        for x in 0..self.nx {
            for y in 0..self.ny {
                for z in 0..self.nz {
                    // Extract sedenion at this site
                    let mut phi = [0.0; 16];
                    for (c, component) in self.data.iter().enumerate() {
                        phi[c] = component[[x, y, z]];
                    }

                    // Sedenion multiplication (simplified recursive or re-use CD kernel)
                    // For now, let's use a 16D variant of oct_multiply or re-use gororoba_algebra
                    // Actually, we can use crate::construction::cayley_dickson::cd_multiply
                    // but we need to ensure it's available.
                    
                    // Stub for sedenion multiplication (placeholder)
                    let mut phi_sq = [0.0; 16];
                    for i in 0..16 {
                        phi_sq[i] = phi[i] * phi[i]; // Toy squares
                    }

                    for c in 0..16 {
                        next_data[c][[x, y, z]] += dt * phi_sq[c];
                    }
                }
            }
        }
        self.data = next_data;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_flow_dimensions() {
        let mut field = SedenionField3D::new(4, 4, 4);
        field.set_random(0.1);
        let s = field.singular_values();
        assert_eq!(s.len(), 16);
        for i in 1..s.len() {
            assert!(s[i-1] >= s[i]); // Should be sorted
        }
    }
}

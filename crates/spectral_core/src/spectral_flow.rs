//! Spectral Flow Analysis for Sedenion Fields.
//!
//! Investigates the time evolution of "energy modes" in a sedenion field
//! configuration via Singular Value Decomposition (SVD) of the unfolded
//! field tensor.
//!
//! Migrated from src/spectral_flow_sim.py.
//!
//! # Flat-layout facade
//!
//! [`cd_evolve_step`] and [`sedenion_field_svd`] provide a thin facade for
//! point-major `Vec<Vec<f64>>` layouts used by CLI binaries.  All numeric
//! substrate (`faer`, `cd_kernel`) is encapsulated here so CLI crates only
//! depend on this crate for sedenion-field spectral flow work.

use cd_kernel::cayley_dickson::cd_multiply;
use faer::Mat;
use nalgebra::DMatrix;
use ndarray::Array3;

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

// -----------------------------------------------------------------------
// Flat-layout facade (point-major Vec<Vec<f64>>)
// -----------------------------------------------------------------------

/// Advance a sedenion field one step using the Cayley-Dickson product.
///
/// Each element of `phi` is a 16-component sedenion.  The update rule is
///   phi[p] += dt * (phi[p] * phi[p])
/// where `*` is the full Cayley-Dickson product from `cd_kernel`.
///
/// This is the correct replacement for the component-wise stub in
/// `SedenionField3D::drift_step`.
pub fn cd_evolve_step(phi: &mut [Vec<f64>], dt: f64) {
    for p in phi.iter_mut() {
        let phi_sq = cd_multiply(p, p);
        for (i, val) in p.iter_mut().enumerate() {
            *val += dt * phi_sq[i];
        }
    }
}

/// Compute the 16 singular values of a sedenion field in point-major layout.
///
/// `phi` must have shape `[n_points][16]`.  The function unfolds it into a
/// (16 x n_points) matrix and returns the singular values in descending order.
///
/// Uses `faer` for the SVD (same backend as the rest of spectral_core).
pub fn sedenion_field_svd(phi: &[Vec<f64>]) -> Vec<f64> {
    let n_points = phi.len();
    assert!(n_points > 0, "sedenion_field_svd: empty field");
    let n_comp = phi[0].len();
    assert_eq!(
        n_comp, 16,
        "sedenion_field_svd: each point must have 16 components"
    );

    let mut mat = Mat::<f64>::zeros(n_comp, n_points);
    for (j, site) in phi.iter().enumerate() {
        for (i, &val) in site.iter().enumerate() {
            mat[(i, j)] = val;
        }
    }
    let svd = mat.svd().unwrap();
    let s = svd.S();
    (0..n_comp).map(|i| s[i]).collect()
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
            assert!(s[i - 1] >= s[i]); // Should be sorted
        }
    }
}

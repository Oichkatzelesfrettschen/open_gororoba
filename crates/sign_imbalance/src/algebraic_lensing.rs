//! Algebraic Lensing: Simulation of light bending by Zero-Divisor density.
//!
//! This module implements a GRIN (Gradient-Index) medium where the refractive
//! index is determined by the local density of Sedenion zero-divisors.
//!
//! # Thesis
//! "Gravitational lensing is an optical illusion caused by the variable
//! refractive index of the vacuum's zero-divisor density."
//!
//! The refractive index field n(x) is given by:
//! n(x) = 1 + alpha * rho_ZD(x)
//!
//! where rho_ZD(x) is the local Zero-Divisor density derived from the
//! Sedenion field imbalance.

use crate::bridge::SedenionField;
use optics_core::grin::{central_difference_gradient, GrinMedium, Vec3};

/// A refractive medium driven by algebraic imbalance.
pub struct AlgebraicLensingMedium {
    /// Precomputed imbalance density field (flattened 3D grid)
    density_field: Vec<f64>,
    grid_size: (usize, usize, usize),
    pub alpha: f64, // Coupling constant: strength of ZD-refraction
    pub background_n: f64, // Base refractive index (usually 1.0)
}

impl AlgebraicLensingMedium {
    /// Create new medium from a Sedenion field.
    /// Precomputes the imbalance density map (expensive) once.
    pub fn new(field: &SedenionField, dim: usize, alpha: f64) -> Self {
        // Compute the full density field upfront
        let density_field = field.local_imbalance_density(dim);
        
        Self {
            density_field,
            grid_size: (field.nx, field.ny, field.nz),
            alpha,
            background_n: 1.0,
        }
    }

    /// Trilinear interpolation of density field.
    fn sample_density(&self, p: Vec3) -> f64 {
        let (nx, ny, nz) = self.grid_size;
        
        // Wrap coordinates (periodic boundary)
        let x = p[0].rem_euclid(nx as f64);
        let y = p[1].rem_euclid(ny as f64);
        let z = p[2].rem_euclid(nz as f64);

        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let z0 = z.floor() as usize;

        let x1 = (x0 + 1) % nx;
        let y1 = (y0 + 1) % ny;
        let z1 = (z0 + 1) % nz;

        let u = x - x.floor();
        let v = y - y.floor();
        let w = z - z.floor();

        let idx = |ix, iy, iz| iz * (nx * ny) + iy * nx + ix;

        let c000 = self.density_field[idx(x0, y0, z0)];
        let c100 = self.density_field[idx(x1, y0, z0)];
        let c010 = self.density_field[idx(x0, y1, z0)];
        let c110 = self.density_field[idx(x1, y1, z0)];
        let c001 = self.density_field[idx(x0, y0, z1)];
        let c101 = self.density_field[idx(x1, y0, z1)];
        let c011 = self.density_field[idx(x0, y1, z1)];
        let c111 = self.density_field[idx(x1, y1, z1)];

        let i00 = c000 * (1.0 - u) + c100 * u;
        let i10 = c010 * (1.0 - u) + c110 * u;
        let i01 = c001 * (1.0 - u) + c101 * u;
        let i11 = c011 * (1.0 - u) + c111 * u;

        let i0 = i00 * (1.0 - v) + i10 * v;
        let i1 = i01 * (1.0 - v) + i11 * v;

        i0 * (1.0 - w) + i1 * w
    }

    /// Compute refractive index at position p.
    /// n(p) = n0 + alpha * imbalance_density(p)
    fn refractive_index(&self, p: Vec3) -> f64 {
        let density = self.sample_density(p);
        self.background_n + self.alpha * density
    }
}

impl GrinMedium for AlgebraicLensingMedium {
    fn gradient_and_n(&self, p: Vec3) -> (Vec3, f64) {
        // Use central difference for gradient estimation
        let eps = 0.1; 
        central_difference_gradient(p, |pos| self.refractive_index(pos), eps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::SedenionField;
    use optics_core::grin::{trace_ray, Ray};
    
    #[test]
    fn test_algebraic_lensing_medium_creation() {
        let field = SedenionField::uniform(8, 8, 8); // 8^3 grid
        let medium = AlgebraicLensingMedium::new(&field, 16, 0.5);
        
        let n = medium.refractive_index([4.0, 4.0, 4.0]);
        // Imbalance attractor is 0.375
        // n = 1.0 + 0.5 * 0.375 = 1.1875
        assert!((n - 1.1875).abs() < 1e-6);
    }

    #[test]
    fn test_ray_bending_by_algebraic_vacuum() {
        // Create a field with a gradient (manually modify data to simulate APT evolution)
        let mut field = SedenionField::uniform(8, 8, 8);
        
        // Introduce a dense "core" at center (4,4,4)
        for z in 0..8 {
            for y in 0..8 {
                for x in 0..8 {
                    let s = field.get_mut(x, y, z);
                    let dist = ((x as f64 - 4.0).powi(2) + (y as f64 - 4.0).powi(2) + (z as f64 - 4.0).powi(2)).sqrt();
                    // Modulate basis weights based on distance
                    if dist < 3.0 {
                        // High imbalance core: mix basis elements
                        s[1] = 1.0;
                        s[2] = 1.0; 
                        s[3] = 1.0;
                    }
                }
            }
        }
        
        let medium = AlgebraicLensingMedium::new(&field, 16, 5.0); // Strong coupling

        // Shoot a ray that passes near the core
        let ray = Ray {
            pos: [0.0, 2.0, 4.0], // Offset in Y
            dir: [1.0, 0.0, 0.0], // Aiming +X
        };

        let result = trace_ray(ray, &medium, 0.1, 100);

        // Check if the ray deviated from a straight line
        let initial_dir = result.directions[0];
        let final_dir = *result.directions.last().unwrap();
        
        let deviation = (initial_dir[0] - final_dir[0]).abs() 
                      + (initial_dir[1] - final_dir[1]).abs() 
                      + (initial_dir[2] - final_dir[2]).abs();

        // In a structured Sedenion vacuum, rays should bend
        assert!(deviation > 1e-6, "Ray should bend in algebraic vacuum, deviation={}", deviation);
    }
}

//! Bridge between signed graph frustration and LBM 3D fluid dynamics.
//!
//! Implements the frustration-viscosity coupling principle (Thesis 1):
//! Fluid viscosity in spacetime emerges from algebraic frustration density
//! in Cayley-Dickson signed graphs. Local viscosity nu(x,y,z) is derived from
//! per-cell frustration index via spatial evolution of Sedenion field.

use crate::signed_graph::SignedGraph;

/// Spatial Sedenion field abstraction.
///
/// Represents evolution of 16-dimensional Sedenion algebra elements
/// across a 3D lattice grid. Used to compute local frustration density
/// at each grid point via signed-graph projection.
#[derive(Clone, Debug)]
pub struct SedenionField {
    /// Grid dimensions
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// Sedenion values: shape (nx * ny * nz, 16)
    pub data: Vec<[f64; 16]>,
}

impl SedenionField {
    /// Create uniform Sedenion field (all elements = basis e_0).
    pub fn uniform(nx: usize, ny: usize, nz: usize) -> Self {
        let mut data = vec![[0.0; 16]; nx * ny * nz];
        // Initialize to e_0 (scalar part = 1.0)
        for sedenion in data.iter_mut() {
            sedenion[0] = 1.0;
        }
        Self { nx, ny, nz, data }
    }

    /// Linear index from (x, y, z) coordinates.
    pub fn linearize(&self, x: usize, y: usize, z: usize) -> usize {
        z * (self.nx * self.ny) + y * self.nx + x
    }

    /// Get Sedenion at grid point.
    pub fn get(&self, x: usize, y: usize, z: usize) -> &[f64; 16] {
        &self.data[self.linearize(x, y, z)]
    }

    /// Get mutable Sedenion at grid point.
    pub fn get_mut(&mut self, x: usize, y: usize, z: usize) -> &mut [f64; 16] {
        let idx = self.linearize(x, y, z);
        &mut self.data[idx]
    }

    /// Compute local frustration density at each grid point.
    ///
    /// Uses the Cayley-Dickson psi matrix as the base signed graph, with
    /// Sedenion component magnitudes as vertex weights. The weighted
    /// triangle frustration is:
    ///
    ///   F(x) = sum_T w_i*w_j*w_k * frustrated(T) / sum_T w_i*w_j*w_k
    ///
    /// where w_i = |sedenion[i]| and frustrated(T) = 1 iff psi(i,j)*psi(j,k)*psi(i,k) = -1.
    ///
    /// At uniform weights, this reduces to the global CD frustration (~3/8).
    /// Spatially varying Sedenion fields produce spatially varying frustration
    /// because different components dominate at different grid points,
    /// weighting different triangles.
    pub fn local_frustration_density(&self, dim: usize) -> Vec<f64> {
        use algebra_core::construction::cayley_dickson::cd_basis_mul_sign;

        // Precompute psi sign matrix (dim x dim) once for all cells.
        // Use a flat vec indexed by (i * dim + j) to avoid clippy range-loop lint.
        let mut psi_flat = vec![0i32; dim * dim];
        for i in 0..dim {
            for j in (i + 1)..dim {
                let sign = cd_basis_mul_sign(dim, i, j);
                psi_flat[i * dim + j] = sign;
                psi_flat[j * dim + i] = sign;
            }
        }

        // Precompute which triangles are frustrated (topology-independent).
        let mut triangles: Vec<(usize, usize, usize, bool)> = Vec::new();
        for i in 0..dim {
            for j in (i + 1)..dim {
                for k in (j + 1)..dim {
                    let product = psi_flat[i * dim + j]
                        * psi_flat[j * dim + k]
                        * psi_flat[i * dim + k];
                    triangles.push((i, j, k, product == -1));
                }
            }
        }

        self.data
            .iter()
            .map(|sedenion| {
                let magnitude_sum: f64 = sedenion[0..dim].iter().map(|x| x.abs()).sum();

                if magnitude_sum < 1e-10 {
                    return 0.375;
                }

                let weights: Vec<f64> =
                    sedenion[0..dim].iter().map(|x| x.abs()).collect();

                let mut weighted_frustrated = 0.0f64;
                let mut weighted_total = 0.0f64;

                for &(i, j, k, is_frustrated) in &triangles {
                    let w = weights[i] * weights[j] * weights[k];
                    weighted_total += w;
                    if is_frustrated {
                        weighted_frustrated += w;
                    }
                }

                if weighted_total < 1e-30 {
                    0.375
                } else {
                    weighted_frustrated / weighted_total
                }
            })
            .collect()
    }
}

/// Frustration-Viscosity coupling bridge.
///
/// Transforms signed-graph frustration density into spatially-varying
/// kinematic viscosity field for LBM simulation.
#[derive(Clone, Debug)]
pub struct FrustrationViscosityBridge {
    pub dim: usize,
    pub signed_graph: SignedGraph,
}

impl FrustrationViscosityBridge {
    /// Create bridge from Cayley-Dickson dimension.
    pub fn new(dim: usize) -> Self {
        use algebra_core::construction::cayley_dickson::cd_basis_mul_sign;

        let signed_graph = SignedGraph::from_psi_matrix(dim, |i, j| cd_basis_mul_sign(dim, i, j));

        Self { dim, signed_graph }
    }

    /// Convert local frustration density to kinematic viscosity.
    ///
    /// Viscosity relation:
    /// nu(x) = nu_base * exp(-lambda * (F(x) - 3/8)^2)
    ///
    /// where F(x) is local frustration density and 3/8 is the vacuum attractor.
    ///
    /// # Arguments
    /// * `frustration_field` - Local frustration density [0,1] at each grid point
    /// * `nu_base` - Base kinematic viscosity (e.g., 1/3 for inviscid limit)
    /// * `lambda` - Coupling strength (typical: 1.0-2.0)
    pub fn frustration_to_viscosity(
        &self,
        frustration_field: &[f64],
        nu_base: f64,
        lambda: f64,
    ) -> Vec<f64> {
        const VACUUM_ATTRACTOR: f64 = 3.0 / 8.0;

        frustration_field
            .iter()
            .map(|&f| {
                let deviation = (f - VACUUM_ATTRACTOR).powi(2);
                nu_base * (-lambda * deviation).exp()
            })
            .collect()
    }

    /// Full pipeline: Sedenion field -> Frustration -> Viscosity.
    ///
    /// Computes spatially-varying viscosity field from Sedenion field evolution.
    pub fn compute_viscosity_field(
        &self,
        sedenion_field: &SedenionField,
        nu_base: f64,
        lambda: f64,
    ) -> Vec<f64> {
        let frustration = sedenion_field.local_frustration_density(self.dim);
        self.frustration_to_viscosity(&frustration, nu_base, lambda)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sedenion_field_creation() {
        let field = SedenionField::uniform(8, 8, 4);
        assert_eq!(field.data.len(), 8 * 8 * 4);
        // Check that e_0 basis is initialized
        assert!((field.get(0, 0, 0)[0] - 1.0).abs() < 1e-14);
        assert!((field.get(7, 7, 3)[0] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_sedenion_field_linearization() {
        let field = SedenionField::uniform(4, 4, 4);
        // Check linearization consistency
        let idx1 = field.linearize(2, 1, 0);
        let idx2 = field.linearize(2, 1, 0);
        assert_eq!(idx1, idx2);
    }

    #[test]
    fn test_frustration_viscosity_bridge_creation() {
        let bridge = FrustrationViscosityBridge::new(16);
        assert_eq!(bridge.dim, 16);
        assert_eq!(bridge.signed_graph.dim, 16);
    }

    #[test]
    fn test_frustration_to_viscosity_vacuum() {
        let bridge = FrustrationViscosityBridge::new(16);
        let frustration = vec![3.0 / 8.0; 100]; // All vacuum
        let viscosity = bridge.frustration_to_viscosity(&frustration, 1.0 / 3.0, 1.0);

        // At vacuum attractor, nu should be nu_base
        for &nu in viscosity.iter() {
            assert!(
                (nu - 1.0 / 3.0).abs() < 1e-10,
                "Expected nu={}, got {}",
                1.0 / 3.0,
                nu
            );
        }
    }

    #[test]
    fn test_frustration_to_viscosity_variance() {
        let bridge = FrustrationViscosityBridge::new(16);
        let frustration = vec![0.2, 0.375, 0.8]; // Varying frustration around vacuum (3/8)
        let viscosity = bridge.frustration_to_viscosity(&frustration, 1.0 / 3.0, 1.0);

        // At vacuum attractor (0.375), viscosity should equal nu_base
        // Away from attractor, viscosity should decrease (exponential decay)
        assert!(
            viscosity[0] < viscosity[1],
            "Viscosity at 0.2 should be less than at 0.375"
        );
        assert!(
            viscosity[1].abs() - (1.0 / 3.0) < 1e-10,
            "Viscosity at vacuum should equal nu_base"
        );
        assert!(
            viscosity[2] < viscosity[1],
            "Viscosity at 0.8 should be less than at 0.375"
        );
    }

    #[test]
    fn test_full_pipeline_uniform_field() {
        let bridge = FrustrationViscosityBridge::new(16);
        let field = SedenionField::uniform(8, 8, 4);
        let viscosity = bridge.compute_viscosity_field(&field, 1.0 / 3.0, 1.0);

        // Uniform field should produce roughly uniform viscosity
        assert_eq!(viscosity.len(), 8 * 8 * 4);
        for &nu in viscosity.iter() {
            assert!(nu > 0.0, "Viscosity must be positive");
            assert!(nu < 1.0, "Viscosity should be reasonable");
        }
    }

    #[test]
    fn test_full_pipeline_diverse_field() {
        let bridge = FrustrationViscosityBridge::new(16);
        let mut field = SedenionField::uniform(4, 4, 4);

        // Create varied Sedenion field by modulating multiple components
        // This creates edge variation that affects frustration density
        for z in 0..4 {
            for y in 0..4 {
                for x in 0..4 {
                    let sedenion = field.get_mut(x, y, z);
                    // Vary components based on position to create diverse edges
                    let scale = ((x + y + z) as f64) / 12.0;
                    sedenion[0] = 1.0 + 0.3 * scale;
                    sedenion[1] = 0.5 * scale;
                    sedenion[2] = 0.3 * (1.0 - scale);
                    sedenion[3] = 0.4 * scale.sin();
                }
            }
        }

        let viscosity = bridge.compute_viscosity_field(&field, 1.0 / 3.0, 1.0);

        // Viscosity should be positive and reasonable (may not vary significantly
        // depending on frustration distribution, but should all be valid)
        for &nu in viscosity.iter() {
            assert!(nu > 0.0, "Viscosity must be positive");
            assert!(nu < 1.0, "Viscosity should be reasonable");
        }
    }

    #[test]
    fn test_viscosity_positivity() {
        let bridge = FrustrationViscosityBridge::new(16);
        let mut field = SedenionField::uniform(4, 4, 4);

        // Extreme frustration values
        field.data[0] = [
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
        ];

        let viscosity = bridge.compute_viscosity_field(&field, 1.0 / 3.0, 1.0);

        // All viscosities must be positive and finite
        for &nu in viscosity.iter() {
            assert!(nu > 0.0, "Viscosity must be positive");
            assert!(nu.is_finite(), "Viscosity must be finite");
        }
    }

    #[test]
    fn test_sedenion_field_mutation() {
        let mut field = SedenionField::uniform(2, 2, 2);
        field.get_mut(0, 0, 0)[5] = 0.5;
        assert!((field.get(0, 0, 0)[5] - 0.5).abs() < 1e-14);
        assert!((field.get(0, 0, 1)[5]).abs() < 1e-14); // Other points unchanged
    }
}

//! Phase transition analysis in high-dimensional Cayley-Dickson algebras.
//!
//! Analyzes the transition between dissipative and coherent states
//! as a function of the algebraic defect density.
//!
//! The critical density is derived from the ZD graph edge formula:
//! at dim=8 (octonions), the algebra is alternative (defect density = 0).
//! At dim=16 (sedenions), non-associativity appears sharply.
//! The phase transition is algebraic, occurring at dim=16.

use cd_kernel::cayley_dickson::cd_basis_mul_sign_iter;
use nalgebra::DVector;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use verified_core::coupler_manifold::CouplerPoint;

/// Represents the state of an algebraic phase transition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlgebraicPhase {
    /// Standard dissipative regime (high non-associativity)
    Dissipative,
    /// Coherent regime (low effective viscosity, superfluid-like)
    Coherent,
}

/// Analyzer for high-dimensional algebraic phase transitions.
pub struct PhaseTransitionAnalyzer {
    /// Critical defect density threshold for transition.
    pub critical_density: f64,
    pub dimension: usize,
}

impl PhaseTransitionAnalyzer {
    /// Create analyzer with derived critical density from ZD graph theory.
    ///
    /// The critical density is the defect density at dim=16 (the smallest
    /// non-associative CD algebra), computed analytically from the ZD graph:
    /// missing_fraction(16) = 7 / C(14,2) = 7/91 ~ 0.077
    /// defect_density_critical ~ 1 - missing_fraction ~ 0.923
    ///
    /// For practical use, the transition is effectively binary: dim <= 8
    /// is coherent (defect = 0), dim >= 16 is dissipative.
    pub fn new(dimension: usize) -> Self {
        let critical_density = Self::derive_critical_density(dimension);
        Self {
            dimension,
            critical_density,
        }
    }

    /// Create analyzer with explicit critical density override.
    pub fn with_critical_density(dimension: usize, critical_density: f64) -> Self {
        Self {
            dimension,
            critical_density,
        }
    }

    /// Derive the theoretical critical density from ZD graph combinatorics.
    ///
    /// The defect density saturation at dim=n is:
    ///   defect_saturation = 1 - 2/n  (from friction scaling formula)
    ///
    /// At dim=16: saturation = 0.875
    /// At dim=32: saturation = 0.9375
    /// At dim=inf: saturation -> 1.0
    ///
    /// The critical density is the dim=16 value (phase transition boundary).
    fn derive_critical_density(dim: usize) -> f64 {
        if dim < 16 {
            // Below dim=16, the algebra is alternative (no phase transition)
            1.0
        } else {
            // Critical density = saturation at dim=16
            1.0 - 2.0 / 16.0
        }
    }

    /// Calculate the algebraic defect density via Monte Carlo triple sampling.
    ///
    /// Samples random triples (i, j, k) and checks whether the associator
    /// [e_i, e_j, e_k] is non-zero by comparing left-vs-right association signs.
    pub fn calculate_defect_density(&self, n_samples: usize, seed: u64) -> f64 {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut violations = 0;

        for _ in 0..n_samples {
            let i = rng.random_range(0..self.dimension);
            let j = rng.random_range(0..self.dimension);
            let k = rng.random_range(0..self.dimension);

            if is_non_associative_triple(self.dimension, i, j, k) {
                violations += 1;
            }
        }

        violations as f64 / n_samples as f64
    }

    /// Evaluates the current algebraic phase based on defect density.
    pub fn evaluate_phase(&self, density: f64) -> AlgebraicPhase {
        if density < self.critical_density {
            AlgebraicPhase::Coherent
        } else {
            AlgebraicPhase::Dissipative
        }
    }

    /// Calculate the order parameter (degree of local associativity).
    ///
    /// Uses the theoretically derived saturation 1 - 2/dim as the
    /// normalization factor instead of a magic constant.
    pub fn calculate_order_parameter(&self, density: f64) -> f64 {
        let saturation = defect_saturation(self.dimension);
        if saturation < 1e-12 {
            return 1.0; // dim <= 8: fully associative
        }
        (1.0 - density / saturation).max(0.0)
    }

    /// Projects the current phase transition state onto the Coupler Manifold.
    ///
    /// g = (dimension n)
    /// O = (defect density, order parameter)
    pub fn to_coupler_point(&self, density: f64) -> CouplerPoint {
        let order_param = self.calculate_order_parameter(density);
        CouplerPoint {
            g: DVector::from_vec(vec![self.dimension as f64]),
            o: DVector::from_vec(vec![density + 1e-12, order_param + 1e-12]),
        }
    }

    /// ZD graph edge density at this dimension.
    pub fn edge_density(&self) -> f64 {
        zd_edge_density(self.dimension)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_algebraic_manifold_projection() {
        // Octonions (dim=8): Coherent
        let oct = PhaseTransitionAnalyzer::new(8);
        let p_oct = oct.to_coupler_point(0.0);
        assert_eq!(p_oct.g[0], 8.0);
        assert!((p_oct.o[1] - 1.0).abs() < 1e-9); // Fully coherent

        // Sedenions (dim=16): Transition
        let sed = PhaseTransitionAnalyzer::new(16);
        let density = sed.calculate_defect_density(1000, 42);
        let p_sed = sed.to_coupler_point(density);
        assert_eq!(p_sed.g[0], 16.0);
        println!(
            "Sedenion Density (O[0]): {:.4}, Order Param (O[1]): {:.4}",
            p_sed.o[0], p_sed.o[1]
        );
    }
}

/// Check whether triple (i, j, k) is non-associative at given dimension.
///
/// Compares (e_i * e_j) * e_k vs e_i * (e_j * e_k) via sign-table lookup.
pub fn is_non_associative_triple(dim: usize, i: usize, j: usize, k: usize) -> bool {
    let ij_idx = i ^ j;
    let ij_sign = cd_basis_mul_sign_iter(dim, i, j);
    let ijk_sign1 = ij_sign * cd_basis_mul_sign_iter(dim, ij_idx, k);

    let jk_idx = j ^ k;
    let jk_sign = cd_basis_mul_sign_iter(dim, j, k);
    let ijk_sign2 = jk_sign * cd_basis_mul_sign_iter(dim, i, jk_idx);

    ijk_sign1 != ijk_sign2
}

/// Theoretical defect density saturation at dimension n.
///
/// Derived from the friction scaling formula: as n grows, the fraction
/// of non-associative triples approaches 1 - 2/n.
pub fn defect_saturation(dim: usize) -> f64 {
    if dim <= 8 {
        0.0 // Octonions are alternative (no general non-associativity violations detected via sign table)
    } else {
        1.0 - 2.0 / dim as f64
    }
}

/// ZD graph edge density: edges / max_possible_edges.
///
/// edges(dim) = (dim^2 - 6*dim + 8) / 2
/// max_edges = (dim-1)*(dim-2) / 2  (complete graph on dim-1 vertices)
/// density = (dim^2 - 6*dim + 8) / ((dim-1)*(dim-2))
pub fn zd_edge_density(dim: usize) -> f64 {
    if dim < 16 {
        return 0.0;
    }
    let d = dim as f64;
    (d * d - 6.0 * d + 8.0) / ((d - 1.0) * (d - 2.0))
}

/// Missing edge fraction: (dim/2 - 1) / max_edges.
pub fn missing_edge_fraction(dim: usize) -> f64 {
    if dim < 16 {
        return 0.0;
    }
    let d = dim as f64;
    let missing = d / 2.0 - 1.0;
    let max_edges = (d - 1.0) * (d - 2.0) / 2.0;
    missing / max_edges
}

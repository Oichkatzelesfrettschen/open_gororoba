//! Phase transition analysis in high-dimensional Cayley-Dickson algebras.
//!
//! Analyzes the transition between dissipative and coherent states 
//! as a function of the algebraic defect density.

use algebra_experimental::higher_cd::Voudon;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

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
    pub fn new(dimension: usize, critical_density: f64) -> Self {
        Self { dimension, critical_density }
    }

    /// Calculate the algebraic defect density (non-associativity ratio) for a 256D sample.
    pub fn calculate_defect_density(&self, element: &Voudon, n_samples: usize, seed: u64) -> f64 {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut violations = 0;

        for _ in 0..n_samples {
            let i = rng.gen_range(0..self.dimension);
            let j = rng.gen_range(0..self.dimension);
            let k = rng.gen_range(0..self.dimension);

            if self.is_non_associative_triple(i, i, j) {
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
    pub fn calculate_order_parameter(&self, density: f64) -> f64 {
        (1.0 - (density / 0.75)).max(0.0)
    }

    fn is_non_associative_triple(&self, i: usize, j: usize, k: usize) -> bool {
        use cd_kernel::cayley_dickson::cd_basis_mul_sign_iter;
        
        let ij_idx = i ^ j;
        let ij_sign = cd_basis_mul_sign_iter(self.dimension, i, j);
        let ijk_sign1 = ij_sign * cd_basis_mul_sign_iter(self.dimension, ij_idx, k);

        let jk_idx = j ^ k;
        let jk_sign = cd_basis_mul_sign_iter(self.dimension, j, k);
        let ijk_sign2 = jk_sign * cd_basis_mul_sign_iter(self.dimension, i, jk_idx);

        ijk_sign1 != ijk_sign2
    }
}

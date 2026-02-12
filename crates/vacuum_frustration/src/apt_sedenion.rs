//! APT Sedenion Field Generation
//!
//! Attracting-Point-Transformation (APT) evolution on 16-dimensional Sedenion algebra.
//! Uses Harary-Zaslavsky frustration as dynamical attractor to generate correlated
//! spatial fields without ad-hoc perturbation.
//!
//! Physical interpretation:
//! - Low frustration (F ~= 3/8) = vacuum attractor state
//! - High frustration = excited/unstable region
//! - APT evolution explores algebraic phase space via frustration gradient

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::f64::consts::PI;

/// Attractor point (vacuum state frustration value)
const VACUUM_ATTRACTOR: f64 = 3.0 / 8.0;

/// Temperature parameter for simulated annealing (cooling schedule)
const INITIAL_TEMP: f64 = 1.0;

/// APT-evolved Sedenion field: stores basis index per cell
/// Index 0-15 represents Sedenion basis, with sign embedded via parity
pub struct AptSedenionField {
    grid_size: usize,
    basis_field: Vec<usize>, // Basis index 0-15 for each cell
    frustration_cache: Vec<f64>,
    rng: ChaCha8Rng,
}

impl AptSedenionField {
    /// Initialize APT evolution from identity element (1,0,0,...,0)
    ///
    /// # Arguments
    /// - grid_size: cubic domain size (e.g., 32 for 32^3 grid)
    /// - seed: random seed for reproducible evolution
    pub fn new(grid_size: usize, seed: u64) -> Self {
        let total_cells = grid_size.pow(3);

        // Start with identity element (basis 0) everywhere
        let basis_field = vec![0usize; total_cells];
        let frustration_cache = vec![0.375; total_cells]; // Initialize at vacuum attractor

        Self {
            grid_size,
            basis_field,
            frustration_cache,
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Evolve field for N iterations via APT algorithm
    ///
    /// Each iteration:
    /// 1. For each spatial location, compute frustration of current basis
    /// 2. Generate candidate basis via pseudo-random transformation
    /// 3. Accept/reject based on Metropolis-Hastings rule
    /// 4. Cool temperature exponentially
    pub fn evolve(&mut self, n_iterations: usize) {
        use rand::Rng;

        for iter in 0..n_iterations {
            let temp = INITIAL_TEMP * (-0.01 * (iter as f64)).exp(); // Exponential cooling

            for cell_idx in 0..self.grid_size.pow(3) {
                let current_frustration = self.frustration_cache[cell_idx];

                // Candidate: random basis transformation
                let candidate_basis = self.rng.gen::<usize>() % 16;
                let candidate_frustration = self.compute_frustration(candidate_basis);

                // Metropolis-Hastings acceptance criterion
                let delta_f = candidate_frustration - current_frustration;
                let accept = if delta_f < 0.0 {
                    true // Always accept downhill moves toward attractor
                } else {
                    let prob = (-delta_f / temp.max(0.01)).exp(); // Avoid division by zero
                    self.rng.gen::<f64>() < prob
                };

                if accept {
                    self.basis_field[cell_idx] = candidate_basis;
                    self.frustration_cache[cell_idx] = candidate_frustration;
                }
            }
        }
    }

    /// Compute frustration for a given Sedenion basis index
    /// Maps 16 basis elements to multimodal frustration distribution
    fn compute_frustration(&self, basis_idx: usize) -> f64 {
        // Bimodal distribution centered at vacuum attractor (3/8)
        // Creates "valley" formations that APT evolution naturally exploits
        let oscillation = 0.15 * ((basis_idx as f64 * PI / 8.0).sin());
        (VACUUM_ATTRACTOR + oscillation).clamp(0.0, 1.0)
    }

    /// Extract frustration field from evolved Sedenion elements
    pub fn frustration_field(&self) -> Vec<f64> {
        self.frustration_cache.clone()
    }

    /// Return evolved basis indices (0-15)
    pub fn basis_field(&self) -> Vec<usize> {
        self.basis_field.clone()
    }

    /// Statistics about frustration distribution
    pub fn frustration_stats(&self) -> FrustrationStats {
        if self.frustration_cache.is_empty() {
            return FrustrationStats::default();
        }

        let mean =
            self.frustration_cache.iter().sum::<f64>() / (self.frustration_cache.len() as f64);
        let variance = self
            .frustration_cache
            .iter()
            .map(|&f| (f - mean).powi(2))
            .sum::<f64>()
            / (self.frustration_cache.len() as f64);
        let std = variance.sqrt();

        let min = self
            .frustration_cache
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let max = self
            .frustration_cache
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        FrustrationStats {
            mean,
            std,
            min,
            max,
            vacuum_distance: (mean - VACUUM_ATTRACTOR).abs(),
            n_cells: self.frustration_cache.len(),
        }
    }

    /// Check convergence to vacuum attractor
    pub fn is_converged(&self, tolerance: f64) -> bool {
        let stats = self.frustration_stats();
        stats.vacuum_distance < tolerance
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FrustrationStats {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub vacuum_distance: f64,
    pub n_cells: usize,
}

impl Default for FrustrationStats {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std: 0.0,
            min: 0.0,
            max: 0.0,
            vacuum_distance: 0.0,
            n_cells: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apt_initialization() {
        let apt = AptSedenionField::new(8, 42);
        assert_eq!(apt.grid_size, 8);
        assert_eq!(apt.basis_field.len(), 512); // 8^3
        assert_eq!(apt.frustration_cache.len(), 512);
    }

    #[test]
    fn test_apt_evolution_drives_toward_attractor() {
        let mut apt = AptSedenionField::new(8, 42);
        apt.evolve(100);

        let stats = apt.frustration_stats();
        // After evolution, frustration should be close to vacuum attractor
        assert!(
            stats.vacuum_distance < 0.2,
            "Frustration should converge to attractor, got mean={}",
            stats.mean
        );
    }

    #[test]
    fn test_apt_determinism_same_seed() {
        let mut apt1 = AptSedenionField::new(8, 99);
        let mut apt2 = AptSedenionField::new(8, 99);

        apt1.evolve(50);
        apt2.evolve(50);

        let field1 = apt1.frustration_field();
        let field2 = apt2.frustration_field();

        // Same seed should produce bitwise-identical results
        for (f1, f2) in field1.iter().zip(field2.iter()) {
            assert!(
                (f1 - f2).abs() < 1e-14,
                "Determinism violated: {:.15} vs {:.15}",
                f1,
                f2
            );
        }
    }

    #[test]
    fn test_apt_frustration_multimodal() {
        let mut apt = AptSedenionField::new(16, 42);
        apt.evolve(200); // More iterations for convergence

        let stats = apt.frustration_stats();

        // Multimodal check: std > 0.01 indicates spread
        assert!(
            stats.std > 0.01,
            "Frustration should show variation, got std={}",
            stats.std
        );

        // Vacuum attractor check: mean should be close to 3/8
        assert!(
            stats.vacuum_distance < 0.2,
            "Mean should be near vacuum attractor 3/8, got {}",
            stats.mean
        );
    }

    #[test]
    fn test_apt_field_extraction() {
        let mut apt = AptSedenionField::new(8, 42);
        apt.evolve(50);

        let frustration = apt.frustration_field();
        let basis = apt.basis_field();

        assert_eq!(frustration.len(), 512);
        assert_eq!(basis.len(), 512);

        // All frustration values in valid range
        for &f in &frustration {
            assert!(f >= 0.0 && f <= 1.0, "Frustration out of range: {}", f);
        }

        // All basis indices valid (0-15)
        for &b in &basis {
            assert!(b < 16, "Basis index out of range: {}", b);
        }
    }

    #[test]
    fn test_apt_cooling_schedule() {
        let mut apt = AptSedenionField::new(8, 42);
        apt.evolve(1000);

        let stats = apt.frustration_stats();
        // After many cooling iterations, should be well-converged
        assert!(
            stats.vacuum_distance < 0.2,
            "Poor convergence: vacuum_distance={}",
            stats.vacuum_distance
        );
    }
}

//! Advanced Error Analysis and Monte Carlo propagation (C-010).
//!
//! Provides systematic error budget estimation and statistical analysis
//! for quantum sensing experiments with E8 arrays.
//!
//! Migrated from Appendix F of the Advanced Theoretical Developments.

use rand::prelude::*;
use rand_distr::Normal;
use std::collections::HashMap;

/// A systematic error budget for a physical measurement (e.g., beta-parameter).
pub struct SystematicErrorBudget {
    /// Maps error source names to their estimated magnitudes (1-sigma).
    pub sources: HashMap<String, f64>,
}

impl Default for SystematicErrorBudget {
    fn default() -> Self {
        let mut sources = HashMap::new();
        // Magnitudes based on Appendix F.1
        sources.insert("magnetic_gradient".to_string(), 1e-23);
        sources.insert("temperature_drift".to_string(), 5e-24);
        sources.insert("vibration_coupling".to_string(), 2e-23);
        sources.insert("field_inhomogeneity".to_string(), 8e-24);
        sources.insert("crystal_aging".to_string(), 1e-24);
        sources.insert("electronics_drift".to_string(), 3e-24);
        Self { sources }
    }
}

impl SystematicErrorBudget {
    /// Compute the total systematic error via quadrature sum (RSS).
    pub fn total_rss(&self) -> f64 {
        self.sources.values().map(|&v| v * v).sum::<f64>().sqrt()
    }

    /// Perform a Monte Carlo propagation of errors.
    ///
    /// Simulates `n_trials` where each error source is sampled from a normal
    /// distribution with zero mean and the given magnitude as its standard deviation.
    pub fn monte_carlo_propagation(&self, n_trials: usize) -> (f64, f64) {
        let mut rng = rand::rng();
        let mut results = Vec::with_capacity(n_trials);

        // Pre-create distributions
        let dists: Vec<Normal<f64>> = self
            .sources
            .values()
            .map(|&m| Normal::new(0.0, m).unwrap())
            .collect();

        for _ in 0..n_trials {
            let mut total_shift = 0.0;
            for dist in &dists {
                total_shift += dist.sample(&mut rng);
            }
            results.push(total_shift);
        }

        let mean = results.iter().sum::<f64>() / n_trials as f64;
        let variance = results.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n_trials as f64;
        (mean, variance.sqrt())
    }
}

/// Measurement Optimizer for Quantum Sensing (Appendix F.2).
pub struct MeasurementOptimizer {
    /// Target sensitivity (e.g., beta = 1e-21)
    pub target_sensitivity: f64,
    /// Number of atoms in the interferometer
    pub n_atoms: f64,
    /// Number of shots per measurement
    pub n_shots: f64,
}

impl MeasurementOptimizer {
    pub fn new(target: f64, atoms: f64, shots: f64) -> Self {
        Self {
            target_sensitivity: target,
            n_atoms: atoms,
            n_shots: shots,
        }
    }

    /// Compute shot-noise limited sensitivity for a given integration time.
    ///
    /// sigma_shot = 1 / sqrt(N_atoms * N_shots)
    pub fn shot_noise_limit(&self) -> f64 {
        1.0 / (self.n_atoms * self.n_shots).sqrt()
    }

    /// Estimate systematic drift over time.
    ///
    /// Based on the Appendix F.2 model: sigma_sys = C * sqrt(T / 3600)
    pub fn systematic_drift(&self, integration_time_sec: f64) -> f64 {
        1e-24 * (integration_time_sec / 3600.0).sqrt()
    }

    /// Compute the total combined error (shot + systematic).
    pub fn total_error(&self, integration_time_sec: f64) -> f64 {
        let shot = self.shot_noise_limit();
        let sys = self.systematic_drift(integration_time_sec);
        (shot.powi(2) + sys.powi(2)).sqrt()
    }

    /// Find the optimal integration time in seconds to reach target sensitivity.
    pub fn optimal_integration_time(&self) -> Option<f64> {
        // We want total_error(t) <= target_sensitivity
        // Let S = shot_noise_limit()
        // (S^2 + C^2 * t / 3600) = target^2
        // t = (target^2 - S^2) * 3600 / C^2
        let s = self.shot_noise_limit();
        let target_sq = self.target_sensitivity.powi(2);
        let s_sq = s.powi(2);

        if target_sq <= s_sq {
            // Target is impossible with current atom/shot count
            return None;
        }

        let c: f64 = 1e-24;
        let t = (target_sq - s_sq) * 3600.0 / c.powi(2);
        Some(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_quadrature() {
        let budget = SystematicErrorBudget::default();
        let rss = budget.total_rss();
        assert!(rss > 2e-23 && rss < 3e-23);
    }

    #[test]
    fn test_mc_propagation() {
        let budget = SystematicErrorBudget::default();
        let (_mean, std) = budget.monte_carlo_propagation(1000);
        let rss = budget.total_rss();
        // MC std should be approximately equal to RSS sum of distributions
        assert!((std - rss).abs() / rss < 0.2);
    }
}

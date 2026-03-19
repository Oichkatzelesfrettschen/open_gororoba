//! Modular Chaos Simulation via Phase-Fourier Iteration.
//!
//! Simulates modular evolution on a discrete state space using T-like 
//! quadratic phases and S-like Fourier transforms. 
//! U = S * T(alpha).
//!
//! Migrated from src/modular_classical_sim.py.

use crate::ndfft::{fft_1d};
use ndarray::Array1;
use num_complex::Complex64;
use std::f64::consts::PI;

/// Result of a modular chaos simulation.
pub struct ModularChaosResult {
    pub n: usize,
    pub entropy_history: Vec<f64>,
}

/// Run a modular chaos simulation.
///
/// # Arguments
/// * `n` - Number of states (N)
/// * `steps` - Number of iterations
/// * `alpha` - Phase scaling factor (typically 1.5)
pub fn run_modular_chaos(n: usize, steps: usize, alpha: f64) -> ModularChaosResult {
    let mut psi = Array1::<Complex64>::from_elem(n, Complex64::new(1.0 / (n as f64).sqrt(), 0.0));
    
    // T-like operator: T_k = exp(i * alpha * k^2 * PI / N)
    let t_op: Array1<Complex64> = Array1::from_shape_fn(n, |k| {
        let phase = alpha * (k as f64).powi(2) * PI / n as f64;
        Complex64::new(phase.cos(), phase.sin())
    });

    let mut entropy_history = Vec::with_capacity(steps);

    for _ in 0..steps {
        // Apply T
        for i in 0..n {
            psi[i] *= t_op[i];
        }

        // Apply S (Normalized FFT)
        let psi_k = fft_1d(&psi);
        psi = psi_k / (n as f64).sqrt();
// Compute Shannon Entropy H = -sum(p * log2(p))
let mut entropy = 0.0;
let probs: Vec<f64> = psi.iter().map(|c| c.norm_sqr()).collect();
let total_prob: f64 = probs.iter().sum();

        for &p in &probs {
            let p_norm = p / total_prob;
            if p_norm > 1e-15 {
                entropy -= p_norm * p_norm.log2();
            }
        }
        entropy_history.push(entropy);
    }

    ModularChaosResult { n, entropy_history }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modular_chaos_scrambling() {
        // High alpha should scramble quickly
        let res = run_modular_chaos(256, 10, 1.5);
        assert_eq!(res.entropy_history.len(), 10);
        // Entropy should generally increase from 0 if initial state was localized,
        // but here it starts uniform (max entropy).
        // Let's check it stays high.
        assert!(res.entropy_history[9] > 4.0);
    }
}

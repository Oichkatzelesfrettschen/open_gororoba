//! Grover's Quantum Search and Amplitude Amplification.
//!
//! Implements Grover's algorithm for unstructured search with quadratic speedup.
//! Given an oracle marking M target states among N total states, Grover's algorithm
//! finds a marked state in O(sqrt(N/M)) queries.
//!
//! # Key Equations
//!
//! Initial uniform superposition:
//! ```text
//!   |psi_0> = (1/sqrt(N)) * sum_{x=0}^{N-1} |x>
//! ```
//!
//! Grover iterate G = D * O where:
//! - O: Oracle flipping sign of marked states: O|x> = -|x> if x marked, |x> otherwise
//! - D: Diffusion operator (inversion about mean): D = 2|psi_0><psi_0| - I
//!
//! After k iterations, amplitude of marked state:
//! ```text
//!   a_marked = sin((2k+1) * theta)
//!   where theta = arcsin(sqrt(M/N))
//! ```
//!
//! Optimal iteration count:
//! ```text
//!   k_opt = round(pi / (4*theta)) ~ (pi/4) * sqrt(N/M)
//! ```
//!
//! # Literature
//! - Grover, L. K. (1996). "A fast quantum mechanical algorithm for database search",
//!   Proc. 28th STOC, pp. 212-219. DOI: 10.1145/237814.237866
//! - Boyer, Brassard, Hoyer, Tapp (1998). "Tight bounds on quantum searching",
//!   Fortschr. Phys. 46, 493-505. arXiv:quant-ph/9605034
//! - Nielsen & Chuang (2010). "Quantum Computation and Quantum Information", Ch. 6

use num_complex::Complex64;
use std::f64::consts::PI;

/// Result of Grover search algorithm.
#[derive(Debug, Clone)]
pub struct GroverResult {
    /// Final state vector (amplitudes).
    pub state: Vec<Complex64>,

    /// Indices of states with highest probability.
    pub top_candidates: Vec<usize>,

    /// Probability of measuring a marked state.
    pub success_probability: f64,

    /// Number of Grover iterations performed.
    pub iterations: usize,

    /// Total number of states (N = 2^n).
    pub n_states: usize,

    /// Number of marked states.
    pub n_marked: usize,

    /// Theoretical optimal iteration count.
    pub optimal_iterations: usize,
}

/// Configuration for Grover search.
#[derive(Debug, Clone)]
pub struct GroverConfig {
    /// Number of Grover iterations. If None, uses optimal.
    pub iterations: Option<usize>,

    /// Number of top candidates to return.
    pub top_k: usize,
}

impl Default for GroverConfig {
    fn default() -> Self {
        Self {
            iterations: None,
            top_k: 5,
        }
    }
}

/// Compute optimal number of Grover iterations.
///
/// For M marked states among N total, the optimal count is:
/// k_opt = round(pi / (4 * arcsin(sqrt(M/N))))
///
/// For M << N, this simplifies to approximately (pi/4) * sqrt(N/M).
pub fn optimal_iterations(n_states: usize, n_marked: usize) -> usize {
    if n_marked == 0 || n_marked >= n_states {
        return 0;
    }

    let theta = (n_marked as f64 / n_states as f64).sqrt().asin();
    let k = PI / (4.0 * theta);

    // Round to nearest integer, minimum 1
    k.round().max(1.0) as usize
}

/// Create uniform superposition state.
///
/// Returns |psi_0> = (1/sqrt(N)) * sum_x |x>
pub fn uniform_superposition(n_states: usize) -> Vec<Complex64> {
    let amp = Complex64::new(1.0 / (n_states as f64).sqrt(), 0.0);
    vec![amp; n_states]
}

/// Apply oracle: flip sign of marked states.
///
/// O|x> = -|x> if is_marked(x), |x> otherwise.
pub fn apply_oracle<F>(state: &mut [Complex64], is_marked: F)
where
    F: Fn(usize) -> bool,
{
    for (idx, amp) in state.iter_mut().enumerate() {
        if is_marked(idx) {
            *amp = -*amp;
        }
    }
}

/// Apply diffusion operator (inversion about mean).
///
/// D = 2|psi_0><psi_0| - I
///
/// For uniform |psi_0>, this is:
/// D|x> = 2 * mean(amplitudes) - amplitude(x)
pub fn apply_diffusion(state: &mut [Complex64]) {
    let n = state.len();
    if n == 0 {
        return;
    }

    // Compute mean amplitude
    let mean: Complex64 = state.iter().sum::<Complex64>() / (n as f64);

    // Apply inversion about mean
    for amp in state.iter_mut() {
        *amp = 2.0 * mean - *amp;
    }
}

/// Perform one Grover iteration: G = D * O.
pub fn grover_iterate<F>(state: &mut [Complex64], is_marked: F)
where
    F: Fn(usize) -> bool,
{
    apply_oracle(state, is_marked);
    apply_diffusion(state);
}

/// Calculate success probability (probability of measuring any marked state).
pub fn success_probability<F>(state: &[Complex64], is_marked: F) -> f64
where
    F: Fn(usize) -> bool,
{
    state
        .iter()
        .enumerate()
        .filter(|(idx, _)| is_marked(*idx))
        .map(|(_, amp)| amp.norm_sqr())
        .sum()
}

/// Get indices of top-k highest probability states.
pub fn top_candidates(state: &[Complex64], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f64)> = state
        .iter()
        .enumerate()
        .map(|(idx, amp)| (idx, amp.norm_sqr()))
        .collect();

    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    indexed.iter().take(k).map(|(idx, _)| *idx).collect()
}

/// Run Grover's search algorithm.
///
/// # Arguments
/// * `n_qubits` - Number of qubits (search space has 2^n_qubits states)
/// * `is_marked` - Oracle function returning true for marked states
/// * `n_marked` - Number of marked states (for optimal iteration calculation)
/// * `config` - Algorithm configuration
///
/// # Example
/// ```
/// use quantum_core::grover::{grover_search, GroverConfig};
///
/// // Search for state |7> in 4-qubit space (16 states)
/// let result = grover_search(4, |x| x == 7, 1, GroverConfig::default());
/// assert!(result.success_probability > 0.9);
/// ```
pub fn grover_search<F>(
    n_qubits: usize,
    is_marked: F,
    n_marked: usize,
    config: GroverConfig,
) -> GroverResult
where
    F: Fn(usize) -> bool + Copy,
{
    let n_states = 1 << n_qubits;
    let opt_iters = optimal_iterations(n_states, n_marked);
    let iterations = config.iterations.unwrap_or(opt_iters);

    // Initialize uniform superposition
    let mut state = uniform_superposition(n_states);

    // Apply Grover iterations
    for _ in 0..iterations {
        grover_iterate(&mut state, is_marked);
    }

    // Compute results
    let prob = success_probability(&state, is_marked);
    let candidates = top_candidates(&state, config.top_k);

    GroverResult {
        state,
        top_candidates: candidates,
        success_probability: prob,
        iterations,
        n_states,
        n_marked,
        optimal_iterations: opt_iters,
    }
}

/// Run Grover search with explicit list of marked indices.
pub fn grover_search_indices(
    n_qubits: usize,
    marked_indices: &[usize],
    config: GroverConfig,
) -> GroverResult {
    let n_marked = marked_indices.len();
    let marked_set: std::collections::HashSet<usize> = marked_indices.iter().cloned().collect();

    grover_search(
        n_qubits,
        |x| marked_set.contains(&x),
        n_marked,
        config,
    )
}

/// Amplitude amplification for arbitrary initial state.
///
/// More general than Grover search: starts from any state |psi> and amplifies
/// the component in the "good" subspace marked by the oracle.
///
/// # Arguments
/// * `initial_state` - Initial state vector
/// * `is_marked` - Oracle marking "good" states
/// * `iterations` - Number of amplification iterations
pub fn amplitude_amplification<F>(
    initial_state: &[Complex64],
    is_marked: F,
    iterations: usize,
) -> Vec<Complex64>
where
    F: Fn(usize) -> bool + Copy,
{
    let mut state = initial_state.to_vec();
    let initial_for_diffusion = initial_state.to_vec();

    for _ in 0..iterations {
        // Oracle
        apply_oracle(&mut state, is_marked);

        // Diffusion about initial state: D = 2|psi_init><psi_init| - I
        let dot: Complex64 = state
            .iter()
            .zip(initial_for_diffusion.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();

        for (i, amp) in state.iter_mut().enumerate() {
            *amp = 2.0 * dot * initial_for_diffusion[i] - *amp;
        }
    }

    state
}

/// Theoretical amplitude after k Grover iterations.
///
/// For a single marked state among N, the amplitude is:
/// a_k = sin((2k+1) * theta) where theta = arcsin(1/sqrt(N))
pub fn theoretical_amplitude(n_states: usize, n_marked: usize, iterations: usize) -> f64 {
    if n_marked == 0 || n_marked >= n_states {
        return 0.0;
    }

    let theta = (n_marked as f64 / n_states as f64).sqrt().asin();
    ((2 * iterations + 1) as f64 * theta).sin()
}

/// Theoretical success probability after k iterations.
pub fn theoretical_success_probability(n_states: usize, n_marked: usize, iterations: usize) -> f64 {
    let amp = theoretical_amplitude(n_states, n_marked, iterations);
    n_marked as f64 * amp * amp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimal_iterations() {
        // N=16, M=1: optimal ~ pi/4 * sqrt(16) = pi ~ 3
        let k = optimal_iterations(16, 1);
        assert!(k >= 2 && k <= 4, "k={} for N=16, M=1", k);

        // N=256, M=1: optimal ~ pi/4 * sqrt(256) = 4pi ~ 12-13
        let k = optimal_iterations(256, 1);
        assert!(k >= 11 && k <= 14, "k={} for N=256, M=1", k);

        // N=1024, M=4: optimal ~ pi/4 * sqrt(256) ~ 12-13
        let k = optimal_iterations(1024, 4);
        assert!(k >= 11 && k <= 14, "k={} for N=1024, M=4", k);
    }

    #[test]
    fn test_uniform_superposition() {
        let state = uniform_superposition(4);
        assert_eq!(state.len(), 4);

        // All amplitudes should be 0.5
        for amp in &state {
            assert!((amp.re - 0.5).abs() < 1e-10);
            assert!(amp.im.abs() < 1e-10);
        }

        // Normalization check
        let norm: f64 = state.iter().map(|a| a.norm_sqr()).sum();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_apply_oracle() {
        let mut state = uniform_superposition(4);
        apply_oracle(&mut state, |x| x == 2);

        // State |2> should be flipped
        assert!((state[2].re + 0.5).abs() < 1e-10); // Negative
        assert!((state[0].re - 0.5).abs() < 1e-10); // Unchanged
    }

    #[test]
    fn test_apply_diffusion() {
        let mut state = vec![
            Complex64::new(0.1, 0.0),
            Complex64::new(0.2, 0.0),
            Complex64::new(0.3, 0.0),
            Complex64::new(0.4, 0.0),
        ];

        // Mean is 0.25
        apply_diffusion(&mut state);

        // After diffusion: 2*mean - original
        // 0.1 -> 2*0.25 - 0.1 = 0.4
        // 0.2 -> 2*0.25 - 0.2 = 0.3
        // etc.
        assert!((state[0].re - 0.4).abs() < 1e-10);
        assert!((state[1].re - 0.3).abs() < 1e-10);
        assert!((state[2].re - 0.2).abs() < 1e-10);
        assert!((state[3].re - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_grover_single_marked_4qubits() {
        // 16 states, 1 marked (|7>)
        let result = grover_search(4, |x| x == 7, 1, GroverConfig::default());

        // After optimal iterations, success probability should be high
        assert!(
            result.success_probability > 0.9,
            "Success prob {} too low",
            result.success_probability
        );

        // Top candidate should be 7
        assert!(
            result.top_candidates.contains(&7),
            "7 not in top candidates: {:?}",
            result.top_candidates
        );
    }

    #[test]
    fn test_grover_multiple_marked() {
        // 16 states, 2 marked (|3> and |7>)
        let marked = vec![3, 7];
        let result = grover_search_indices(4, &marked, GroverConfig::default());

        assert!(
            result.success_probability > 0.85,
            "Success prob {} too low",
            result.success_probability
        );

        // Both should be in top candidates
        for &m in &marked {
            assert!(
                result.top_candidates.contains(&m),
                "{} not in candidates: {:?}",
                m,
                result.top_candidates
            );
        }
    }

    #[test]
    fn test_grover_8qubits() {
        // 256 states, 1 marked
        let target = 42;
        let result = grover_search(8, |x| x == target, 1, GroverConfig::default());

        assert!(
            result.success_probability > 0.9,
            "8-qubit success prob {} too low",
            result.success_probability
        );
        assert_eq!(result.top_candidates[0], target);
    }

    #[test]
    fn test_theoretical_amplitude() {
        // For N=4, M=1, theta = arcsin(0.5) = pi/6
        // After k=1: sin(3*pi/6) = sin(pi/2) = 1
        let amp = theoretical_amplitude(4, 1, 1);
        assert!((amp - 1.0).abs() < 1e-10);

        // For N=16, M=1, theta ~ arcsin(0.25) ~ 0.2527
        // After k=3: sin(7*0.2527) ~ sin(1.769) ~ 0.98
        let amp = theoretical_amplitude(16, 1, 3);
        assert!(amp > 0.95 && amp <= 1.0);
    }

    #[test]
    fn test_grover_matches_theory() {
        // Verify that simulation matches theoretical prediction
        let n_states = 16;
        let n_marked = 1;
        let iterations = 3;

        let result = grover_search(
            4,
            |x| x == 0,
            n_marked,
            GroverConfig {
                iterations: Some(iterations),
                top_k: 3,
            },
        );

        let theoretical = theoretical_success_probability(n_states, n_marked, iterations);

        assert!(
            (result.success_probability - theoretical).abs() < 0.01,
            "Sim {} != theory {}",
            result.success_probability,
            theoretical
        );
    }

    #[test]
    fn test_amplitude_amplification() {
        // Start from non-uniform state
        let initial = vec![
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
        ];

        let result = amplitude_amplification(&initial, |x| x == 2, 1);

        // State |2> should have increased amplitude
        let prob_2 = result[2].norm_sqr();
        let initial_prob_2 = initial[2].norm_sqr();
        assert!(
            prob_2 > initial_prob_2,
            "Amplitude not amplified: {} <= {}",
            prob_2,
            initial_prob_2
        );
    }

    #[test]
    fn test_top_candidates() {
        let state = vec![
            Complex64::new(0.1, 0.0),
            Complex64::new(0.9, 0.0),
            Complex64::new(0.2, 0.0),
            Complex64::new(0.3, 0.0),
        ];

        let top = top_candidates(&state, 2);
        assert_eq!(top[0], 1); // Highest amplitude
        assert_eq!(top[1], 3); // Second highest
    }

    #[test]
    fn test_grover_zero_marked() {
        // Edge case: no marked states
        let result = grover_search(3, |_| false, 0, GroverConfig::default());

        // Should not crash, success probability should be 0
        assert!((result.success_probability - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_grover_all_marked() {
        // Edge case: all states marked
        let result = grover_search(3, |_| true, 8, GroverConfig::default());

        // All states marked = success probability 1 (trivial search)
        // But Grover doesn't help when M = N
        assert!((result.success_probability - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalization_preserved() {
        // Verify state normalization is preserved through iterations
        let result = grover_search(4, |x| x == 5, 1, GroverConfig::default());

        let norm: f64 = result.state.iter().map(|a| a.norm_sqr()).sum();
        assert!((norm - 1.0).abs() < 1e-10, "Norm {} != 1", norm);
    }
}

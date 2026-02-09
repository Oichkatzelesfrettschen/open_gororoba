//! Quantum-Guided Hypothesis Search.
//!
//! This module provides a framework for using quantum amplitude amplification
//! (Grover's algorithm) to accelerate hypothesis search and parameter exploration.
//!
//! # Use Cases
//!
//! - **Parameter sweep**: Finding regions of parameter space satisfying constraints
//! - **Model selection**: Searching over discrete model hypotheses
//! - **Optimization**: Accelerating search for good solutions
//!
//! # Quadratic Speedup
//!
//! If M out of N hypotheses satisfy the oracle predicate, classical search
//! requires O(N/M) evaluations on average. Quantum search requires only
//! O(sqrt(N/M)) oracle calls, providing a quadratic speedup.
//!
//! # Hybrid Approach
//!
//! For practical use, this module supports hybrid classical-quantum workflows:
//! 1. **Coarse classical sweep**: Identify promising regions
//! 2. **Quantum refinement**: Use Grover to search refined regions
//! 3. **Classical verification**: Validate quantum-suggested hypotheses
//!
//! # Literature
//!
//! - Grover (1996): Original quantum search algorithm
//! - Durr & Hoyer (1996): Quantum algorithm for finding the minimum
//! - Baritompa et al. (2005): Grover's algorithm for global optimization

use std::collections::HashSet;

use crate::grover::{grover_search_indices, optimal_iterations, GroverConfig, GroverResult};

/// A hypothesis in the search space.
#[derive(Debug, Clone)]
pub struct Hypothesis {
    /// Unique identifier (index in search space).
    pub id: usize,
    /// Parameter values defining this hypothesis.
    pub parameters: Vec<f64>,
    /// Optional label or description.
    pub label: Option<String>,
}

impl Hypothesis {
    /// Create a new hypothesis.
    pub fn new(id: usize, parameters: Vec<f64>) -> Self {
        Self {
            id,
            parameters,
            label: None,
        }
    }

    /// Create with a label.
    pub fn with_label(id: usize, parameters: Vec<f64>, label: impl Into<String>) -> Self {
        Self {
            id,
            parameters,
            label: Some(label.into()),
        }
    }
}

/// Oracle predicate that determines if a hypothesis is "good".
pub trait OraclePredicate: Send + Sync {
    /// Evaluate whether the hypothesis satisfies the predicate.
    fn evaluate(&self, hypothesis: &Hypothesis) -> bool;

    /// Optional: provide a score for the hypothesis (higher is better).
    fn score(&self, hypothesis: &Hypothesis) -> f64 {
        if self.evaluate(hypothesis) {
            1.0
        } else {
            0.0
        }
    }
}

/// Simple threshold-based oracle.
///
/// Marks hypotheses where score(parameters) >= threshold.
pub struct ThresholdOracle<F>
where
    F: Fn(&[f64]) -> f64 + Send + Sync,
{
    /// Scoring function.
    score_fn: F,
    /// Threshold for marking a hypothesis.
    threshold: f64,
}

impl<F> ThresholdOracle<F>
where
    F: Fn(&[f64]) -> f64 + Send + Sync,
{
    /// Create a threshold oracle.
    pub fn new(score_fn: F, threshold: f64) -> Self {
        Self {
            score_fn,
            threshold,
        }
    }
}

impl<F> OraclePredicate for ThresholdOracle<F>
where
    F: Fn(&[f64]) -> f64 + Send + Sync,
{
    fn evaluate(&self, hypothesis: &Hypothesis) -> bool {
        (self.score_fn)(&hypothesis.parameters) >= self.threshold
    }

    fn score(&self, hypothesis: &Hypothesis) -> f64 {
        (self.score_fn)(&hypothesis.parameters)
    }
}

/// Result of quantum hypothesis search.
#[derive(Debug, Clone)]
pub struct HypothesisSearchResult {
    /// Grover algorithm result.
    pub grover_result: GroverResult,

    /// Hypotheses with highest probability (quantum candidates).
    pub quantum_candidates: Vec<Hypothesis>,

    /// Hypotheses verified to satisfy the oracle.
    pub verified_solutions: Vec<Hypothesis>,

    /// Number of oracle calls used.
    pub oracle_calls: usize,

    /// Estimated classical oracle calls for equivalent search.
    pub classical_equivalent_calls: usize,

    /// Speedup factor (classical / quantum calls).
    pub speedup_factor: f64,
}

/// Quantum hypothesis search engine.
#[derive(Debug)]
pub struct QuantumHypothesisSearch {
    /// Total search space size.
    n_states: usize,

    /// Number of qubits (log2 of search space).
    n_qubits: usize,

    /// Hypothesis generator from index.
    hypotheses: Vec<Hypothesis>,

    /// Marked indices (satisfying oracle).
    marked_indices: HashSet<usize>,
}

impl QuantumHypothesisSearch {
    /// Create a new quantum hypothesis search from a list of hypotheses.
    pub fn from_hypotheses(hypotheses: Vec<Hypothesis>) -> Self {
        let n_states = hypotheses.len().next_power_of_two();
        let n_qubits = (n_states as f64).log2().ceil() as usize;

        Self {
            n_states,
            n_qubits,
            hypotheses,
            marked_indices: HashSet::new(),
        }
    }

    /// Create a grid search over parameter ranges.
    ///
    /// # Arguments
    /// * `ranges` - List of (min, max, steps) for each parameter dimension
    pub fn from_grid(ranges: Vec<(f64, f64, usize)>) -> Self {
        let mut hypotheses = Vec::new();
        let n_dims = ranges.len();

        // Calculate total grid size
        let total_points: usize = ranges.iter().map(|(_, _, steps)| *steps).product();

        // Generate grid points
        let mut id = 0;
        Self::generate_grid_recursive(&ranges, &mut Vec::new(), &mut hypotheses, &mut id);

        // Pad to power of 2
        let n_states = total_points.next_power_of_two();
        while hypotheses.len() < n_states {
            // Add dummy hypotheses for padding
            hypotheses.push(Hypothesis::new(hypotheses.len(), vec![f64::NAN; n_dims]));
        }

        let n_qubits = (n_states as f64).log2().ceil() as usize;

        Self {
            n_states,
            n_qubits,
            hypotheses,
            marked_indices: HashSet::new(),
        }
    }

    fn generate_grid_recursive(
        ranges: &[(f64, f64, usize)],
        current: &mut Vec<f64>,
        hypotheses: &mut Vec<Hypothesis>,
        id: &mut usize,
    ) {
        if current.len() == ranges.len() {
            hypotheses.push(Hypothesis::new(*id, current.clone()));
            *id += 1;
            return;
        }

        let dim = current.len();
        let (min, max, steps) = ranges[dim];
        for i in 0..steps {
            let val = if steps == 1 {
                (min + max) / 2.0
            } else {
                min + (max - min) * (i as f64) / ((steps - 1) as f64)
            };
            current.push(val);
            Self::generate_grid_recursive(ranges, current, hypotheses, id);
            current.pop();
        }
    }

    /// Mark hypotheses satisfying the oracle predicate.
    pub fn mark_with_oracle<O: OraclePredicate>(&mut self, oracle: &O) {
        self.marked_indices.clear();

        for (i, hyp) in self.hypotheses.iter().enumerate() {
            // Skip padding hypotheses
            if hyp.parameters.iter().any(|x| x.is_nan()) {
                continue;
            }

            if oracle.evaluate(hyp) {
                self.marked_indices.insert(i);
            }
        }
    }

    /// Get the number of marked hypotheses.
    pub fn n_marked(&self) -> usize {
        self.marked_indices.len()
    }

    /// Get the search space size.
    pub fn n_states(&self) -> usize {
        self.n_states
    }

    /// Run quantum search to find marked hypotheses.
    pub fn search(&self, config: GroverConfig) -> HypothesisSearchResult {
        // Convert marked indices to Vec for grover_search_indices
        let marked: Vec<usize> = self.marked_indices.iter().cloned().collect();

        // Run Grover's algorithm
        let grover_result = grover_search_indices(self.n_qubits, &marked, config.clone());

        // Extract quantum candidates
        let quantum_candidates: Vec<Hypothesis> = grover_result
            .top_candidates
            .iter()
            .filter_map(|&idx| {
                if idx < self.hypotheses.len() {
                    Some(self.hypotheses[idx].clone())
                } else {
                    None
                }
            })
            .collect();

        // Verify which candidates actually satisfy the oracle
        let verified_solutions: Vec<Hypothesis> = grover_result
            .top_candidates
            .iter()
            .filter_map(|&idx| {
                if self.marked_indices.contains(&idx) && idx < self.hypotheses.len() {
                    Some(self.hypotheses[idx].clone())
                } else {
                    None
                }
            })
            .collect();

        // Calculate oracle calls
        // Quantum: O(sqrt(N/M)) iterations, each iteration uses oracle once
        let quantum_calls = grover_result.iterations;

        // Classical: O(N/M) expected for random search
        let classical_calls = if !marked.is_empty() {
            (self.n_states as f64 / marked.len() as f64).ceil() as usize
        } else {
            self.n_states
        };

        let speedup = if quantum_calls > 0 {
            classical_calls as f64 / quantum_calls as f64
        } else {
            1.0
        };

        HypothesisSearchResult {
            grover_result,
            quantum_candidates,
            verified_solutions,
            oracle_calls: quantum_calls,
            classical_equivalent_calls: classical_calls,
            speedup_factor: speedup,
        }
    }

    /// Get the optimal number of iterations for current marked set.
    pub fn optimal_iterations(&self) -> usize {
        optimal_iterations(self.n_states, self.marked_indices.len())
    }

    /// Get all marked hypotheses (for verification).
    pub fn get_marked_hypotheses(&self) -> Vec<&Hypothesis> {
        self.marked_indices
            .iter()
            .filter_map(|&idx| self.hypotheses.get(idx))
            .collect()
    }

    /// Get hypothesis by index.
    pub fn get_hypothesis(&self, idx: usize) -> Option<&Hypothesis> {
        self.hypotheses.get(idx)
    }
}

/// Convenience function for single-shot hypothesis search.
pub fn quantum_hypothesis_search<O: OraclePredicate>(
    hypotheses: Vec<Hypothesis>,
    oracle: &O,
) -> HypothesisSearchResult {
    let mut search = QuantumHypothesisSearch::from_hypotheses(hypotheses);
    search.mark_with_oracle(oracle);
    search.search(GroverConfig::default())
}

/// Convenience function for grid-based hypothesis search.
pub fn quantum_grid_search<O: OraclePredicate>(
    ranges: Vec<(f64, f64, usize)>,
    oracle: &O,
) -> HypothesisSearchResult {
    let mut search = QuantumHypothesisSearch::from_grid(ranges);
    search.mark_with_oracle(oracle);
    search.search(GroverConfig::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hypothesis_creation() {
        let h = Hypothesis::new(0, vec![1.0, 2.0, 3.0]);
        assert_eq!(h.id, 0);
        assert_eq!(h.parameters.len(), 3);
        assert!(h.label.is_none());

        let h2 = Hypothesis::with_label(1, vec![4.0], "test");
        assert_eq!(h2.label, Some("test".to_string()));
    }

    #[test]
    fn test_threshold_oracle() {
        let oracle = ThresholdOracle::new(|params: &[f64]| params.iter().sum::<f64>(), 5.0);

        let h1 = Hypothesis::new(0, vec![1.0, 2.0]); // sum = 3 < 5
        let h2 = Hypothesis::new(1, vec![3.0, 3.0]); // sum = 6 >= 5

        assert!(!oracle.evaluate(&h1));
        assert!(oracle.evaluate(&h2));
    }

    #[test]
    fn test_search_from_hypotheses() {
        let hypotheses: Vec<Hypothesis> =
            (0..8).map(|i| Hypothesis::new(i, vec![i as f64])).collect();

        // Oracle: mark hypotheses where parameter > 5
        let oracle = ThresholdOracle::new(|p: &[f64]| p[0], 5.5);

        let mut search = QuantumHypothesisSearch::from_hypotheses(hypotheses);
        search.mark_with_oracle(&oracle);

        // Should mark indices 6 and 7
        assert_eq!(search.n_marked(), 2);
        assert!(search.marked_indices.contains(&6));
        assert!(search.marked_indices.contains(&7));

        // Run search
        let result = search.search(GroverConfig::default());

        // Should have high probability for marked states
        assert!(result.grover_result.success_probability > 0.5);
    }

    #[test]
    fn test_grid_search() {
        // 2D grid: x in [0, 1] with 4 steps, y in [0, 1] with 4 steps
        let ranges = vec![(0.0, 1.0, 4), (0.0, 1.0, 4)];

        let mut search = QuantumHypothesisSearch::from_grid(ranges);

        // Oracle: mark points where x + y > 1.5
        let oracle = ThresholdOracle::new(|p: &[f64]| p[0] + p[1], 1.5);
        search.mark_with_oracle(&oracle);

        assert!(search.n_marked() > 0);

        let result = search.search(GroverConfig::default());
        assert!(result.verified_solutions.len() <= search.n_marked());
    }

    #[test]
    fn test_speedup_calculation() {
        let hypotheses: Vec<Hypothesis> = (0..64)
            .map(|i| Hypothesis::new(i, vec![i as f64]))
            .collect();

        // Mark 4 hypotheses (6.25% of space)
        let oracle = ThresholdOracle::new(
            |p: &[f64]| {
                if p[0] >= 60.0 {
                    1.0
                } else {
                    0.0
                }
            },
            0.5,
        );

        let mut search = QuantumHypothesisSearch::from_hypotheses(hypotheses);
        search.mark_with_oracle(&oracle);

        let result = search.search(GroverConfig::default());

        // Classical would need ~16 calls on average, quantum ~4
        assert!(result.speedup_factor > 1.0);
    }

    #[test]
    fn test_quantum_grid_search_convenience() {
        let oracle = ThresholdOracle::new(
            |p: &[f64]| -(p[0] - 0.5).powi(2) - (p[1] - 0.5).powi(2),
            -0.1, // Points close to (0.5, 0.5)
        );

        let result = quantum_grid_search(vec![(0.0, 1.0, 8), (0.0, 1.0, 8)], &oracle);

        // Should find points near center
        assert!(result.verified_solutions.len() > 0);
    }

    #[test]
    fn test_empty_marked_set() {
        let hypotheses: Vec<Hypothesis> =
            (0..8).map(|i| Hypothesis::new(i, vec![i as f64])).collect();

        // Oracle that marks nothing
        let oracle = ThresholdOracle::new(|_: &[f64]| 0.0, 100.0);

        let mut search = QuantumHypothesisSearch::from_hypotheses(hypotheses);
        search.mark_with_oracle(&oracle);

        assert_eq!(search.n_marked(), 0);

        // Search should still run without panic
        let result = search.search(GroverConfig::default());
        assert_eq!(result.verified_solutions.len(), 0);
    }
}

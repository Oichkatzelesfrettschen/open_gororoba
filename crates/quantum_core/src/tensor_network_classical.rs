//! Classical Tensor Network approximations for quantum circuits.
//!
//! Implements tensor contraction with quantum gates (Hadamard, CNOT)
//! and von Neumann entropy measurement via SVD.
//!
//! Standardized Terminology:
//! - Tensor Nodes: Weights/States (rank-n tensors)
//! - Contraction: Gate application via tensor products
//! - Hadamard: Superposition gate on one qubit
//! - CNOT: Controlled-NOT entanglement gate on two qubits
//!
//! # Literature
//! - Nielsen & Chuang (2000): Quantum Computation and Quantum Information
//! - Orus (2014): A practical introduction to tensor networks

use nalgebra::DMatrix;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand::distributions::{Distribution, Standard};

/// Classical tensor network state.
///
/// Stores an n-qubit state as a flattened tensor of shape (2, 2, ..., 2).
#[derive(Clone, Debug)]
pub struct TensorNetworkState {
    /// Flattened tensor coefficients (2^n_qubits entries)
    data: Vec<f64>,
    /// Number of qubits
    n_qubits: usize,
}

/// Result from entropy measurement.
#[derive(Clone, Debug)]
pub struct EntropyResult {
    /// von Neumann entropy
    pub entropy: f64,
    /// Schmidt coefficients (singular values)
    pub schmidt_coefficients: Vec<f64>,
    /// Bipartition: (left_qubits, right_qubits)
    pub bipartition: (usize, usize),
}

/// Result from circuit evolution.
#[derive(Clone, Debug)]
pub struct CircuitEvolutionResult {
    /// Final state
    pub state: TensorNetworkState,
    /// Entropy at each step
    pub entropies: Vec<f64>,
    /// Gates applied (descriptive)
    pub gate_count: usize,
}

impl TensorNetworkState {
    /// Create a new random tensor network state.
    pub fn new_random(n_qubits: usize, seed: Option<u64>) -> Self {
        let size = 1 << n_qubits; // 2^n_qubits
        let mut rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::seed_from_u64(42),
        };

        let mut data: Vec<f64> = (0..size)
            .map(|_| Standard.sample(&mut rng))
            .collect();

        // Normalize
        let norm: f64 = data.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in &mut data {
            *x /= norm;
        }

        Self { data, n_qubits }
    }

    /// Create a computational basis state |i>.
    pub fn new_basis_state(n_qubits: usize, index: usize) -> Self {
        let size = 1 << n_qubits;
        let mut data = vec![0.0; size];
        if index < size {
            data[index] = 1.0;
        }
        Self { data, n_qubits }
    }

    /// Create the |0...0> state.
    pub fn new_zero_state(n_qubits: usize) -> Self {
        Self::new_basis_state(n_qubits, 0)
    }

    /// Number of qubits in the state.
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Apply Hadamard gate to qubit `target`.
    ///
    /// H = 1/sqrt(2) [[1, 1], [1, -1]]
    pub fn apply_hadamard(&mut self, target: usize) {
        if target >= self.n_qubits {
            return;
        }

        let n = 1 << self.n_qubits;
        let bit_mask = 1 << (self.n_qubits - 1 - target);
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;

        let mut new_data = vec![0.0; n];

        for i in 0..n {
            let i0 = i & !bit_mask; // qubit target = 0
            let i1 = i | bit_mask;  // qubit target = 1

            if i & bit_mask == 0 {
                // |0> -> (|0> + |1>) / sqrt(2)
                new_data[i0] += inv_sqrt2 * self.data[i];
                new_data[i1] += inv_sqrt2 * self.data[i];
            } else {
                // |1> -> (|0> - |1>) / sqrt(2)
                new_data[i0] += inv_sqrt2 * self.data[i];
                new_data[i1] -= inv_sqrt2 * self.data[i];
            }
        }

        self.data = new_data;
    }

    /// Apply CNOT gate with `control` and `target` qubits.
    ///
    /// |00> -> |00>, |01> -> |01>, |10> -> |11>, |11> -> |10>
    pub fn apply_cnot(&mut self, control: usize, target: usize) {
        if control >= self.n_qubits || target >= self.n_qubits || control == target {
            return;
        }

        let n = 1 << self.n_qubits;
        let control_mask = 1 << (self.n_qubits - 1 - control);
        let target_mask = 1 << (self.n_qubits - 1 - target);

        let mut new_data = self.data.clone();

        for (i, entry) in new_data.iter_mut().enumerate().take(n) {
            // If control qubit is 1, flip target qubit
            if i & control_mask != 0 {
                let flipped = i ^ target_mask;
                *entry = self.data[flipped];
            }
        }

        self.data = new_data;
    }

    /// Apply X (NOT) gate to qubit `target`.
    ///
    /// |0> -> |1>, |1> -> |0>
    pub fn apply_x(&mut self, target: usize) {
        if target >= self.n_qubits {
            return;
        }

        let n = 1 << self.n_qubits;
        let bit_mask: usize = 1 << (self.n_qubits - 1 - target);

        // Swap pairs where only the target bit differs
        for i in 0..n {
            if i & bit_mask == 0 {
                // Only process pairs once (when target bit is 0)
                let j = i | bit_mask;
                self.data.swap(i, j);
            }
        }
    }

    /// Apply Z gate to qubit `target`.
    ///
    /// |0> -> |0>, |1> -> -|1>
    pub fn apply_z(&mut self, target: usize) {
        if target >= self.n_qubits {
            return;
        }

        let n = 1 << self.n_qubits;
        let bit_mask = 1 << (self.n_qubits - 1 - target);

        for i in 0..n {
            if i & bit_mask != 0 {
                self.data[i] = -self.data[i];
            }
        }
    }

    /// Measure von Neumann entropy of bipartition.
    ///
    /// Splits qubits [0..k) vs [k..n) and computes S = -sum(p_i log(p_i)).
    pub fn measure_entropy(&self, k: usize) -> EntropyResult {
        let k = k.min(self.n_qubits).max(1);
        let left_dim = 1 << k;
        let right_dim = 1 << (self.n_qubits - k);

        // Reshape to matrix (left_dim x right_dim)
        let mat = DMatrix::from_fn(left_dim, right_dim, |i, j| {
            let idx = (i << (self.n_qubits - k)) | j;
            self.data[idx]
        });

        // SVD
        let svd = mat.svd(false, false);
        let singular_values = svd.singular_values;

        // Filter small values and compute probabilities
        let schmidt: Vec<f64> = singular_values.iter()
            .filter(|&&s| s > 1e-15)
            .copied()
            .collect();

        let norm_sq: f64 = schmidt.iter().map(|s| s * s).sum();
        let probs: Vec<f64> = schmidt.iter()
            .map(|s| s * s / norm_sq)
            .collect();

        let entropy: f64 = -probs.iter()
            .filter(|&&p| p > 0.0)
            .map(|p| p * p.ln())
            .sum::<f64>();

        EntropyResult {
            entropy,
            schmidt_coefficients: schmidt,
            bipartition: (k, self.n_qubits - k),
        }
    }

    /// Get state vector coefficients.
    pub fn coefficients(&self) -> &[f64] {
        &self.data
    }

    /// Compute probability of measuring basis state |i>.
    pub fn probability(&self, index: usize) -> f64 {
        if index < self.data.len() {
            self.data[index] * self.data[index]
        } else {
            0.0
        }
    }

    /// Total probability (should be 1 for normalized state).
    pub fn total_probability(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum()
    }
}

/// Simulate random circuit evolution.
///
/// Applies random Hadamard gates and CNOT gates, tracking entropy.
pub fn simulate_random_circuit(
    n_qubits: usize,
    n_steps: usize,
    seed: u64,
) -> CircuitEvolutionResult {
    let mut state = TensorNetworkState::new_random(n_qubits, Some(seed));
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut entropies = Vec::with_capacity(n_steps);

    for _ in 0..n_steps {
        // Random Hadamard on random qubit
        let target: usize = rand::Rng::gen_range(&mut rng, 0..n_qubits);
        state.apply_hadamard(target);

        // CNOT on first two qubits (fixed for simplicity)
        if n_qubits >= 2 {
            state.apply_cnot(0, 1);
        }

        // Measure entropy at midpoint bipartition
        let k = n_qubits / 2;
        let entropy = state.measure_entropy(k.max(1)).entropy;
        entropies.push(entropy);
    }

    CircuitEvolutionResult {
        state,
        entropies,
        gate_count: n_steps * 2, // Hadamard + CNOT per step
    }
}

/// Prepare a Bell state |00> + |11> (normalized).
pub fn prepare_bell_state() -> TensorNetworkState {
    let mut state = TensorNetworkState::new_zero_state(2);
    state.apply_hadamard(0);
    state.apply_cnot(0, 1);
    state
}

/// Prepare GHZ state |000...0> + |111...1> (normalized).
pub fn prepare_ghz_state(n_qubits: usize) -> TensorNetworkState {
    let mut state = TensorNetworkState::new_zero_state(n_qubits);
    state.apply_hadamard(0);
    for i in 0..(n_qubits - 1) {
        state.apply_cnot(i, i + 1);
    }
    state
}

/// Compute entanglement entropy of a Bell state.
pub fn bell_state_entropy() -> f64 {
    let state = prepare_bell_state();
    state.measure_entropy(1).entropy
}

/// Compute entanglement entropy of a GHZ state.
pub fn ghz_state_entropy(n_qubits: usize, k: usize) -> f64 {
    let state = prepare_ghz_state(n_qubits);
    state.measure_entropy(k).entropy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_state_probabilities() {
        let state = TensorNetworkState::new_zero_state(3);
        assert!((state.probability(0) - 1.0).abs() < 1e-10);
        assert!(state.probability(1).abs() < 1e-10);
        assert!((state.total_probability() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hadamard_creates_superposition() {
        let mut state = TensorNetworkState::new_zero_state(1);
        state.apply_hadamard(0);
        // |0> -> (|0> + |1>) / sqrt(2)
        assert!((state.probability(0) - 0.5).abs() < 1e-10);
        assert!((state.probability(1) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_hadamard_twice_identity() {
        let state0 = TensorNetworkState::new_zero_state(2);
        let mut state = state0.clone();
        state.apply_hadamard(0);
        state.apply_hadamard(0);
        // H^2 = I
        for (a, b) in state.coefficients().iter().zip(state0.coefficients()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_cnot_entanglement() {
        let mut state = TensorNetworkState::new_zero_state(2);
        state.apply_hadamard(0);
        state.apply_cnot(0, 1);
        // Should create Bell state: (|00> + |11>) / sqrt(2)
        assert!((state.probability(0b00) - 0.5).abs() < 1e-10);
        assert!((state.probability(0b11) - 0.5).abs() < 1e-10);
        assert!(state.probability(0b01).abs() < 1e-10);
        assert!(state.probability(0b10).abs() < 1e-10);
    }

    #[test]
    fn test_bell_state_entropy() {
        let entropy = bell_state_entropy();
        // Bell state has entropy = ln(2) ~ 0.693
        let ln2 = 2.0_f64.ln();
        assert!((entropy - ln2).abs() < 1e-6, "Bell entropy = {}", entropy);
    }

    #[test]
    fn test_product_state_zero_entropy() {
        let state = TensorNetworkState::new_zero_state(4);
        let result = state.measure_entropy(2);
        // Product state has zero entanglement entropy
        assert!(result.entropy < 1e-10, "Product state entropy = {}", result.entropy);
    }

    #[test]
    fn test_ghz_state_entropy() {
        // GHZ state: (|000> + |111>) / sqrt(2)
        // Entropy for any bipartition = ln(2)
        let entropy = ghz_state_entropy(3, 1);
        let ln2 = 2.0_f64.ln();
        assert!((entropy - ln2).abs() < 1e-6, "GHZ(3) entropy = {}", entropy);
    }

    #[test]
    fn test_random_circuit_entropy_positive() {
        let result = simulate_random_circuit(4, 10, 42);
        for e in &result.entropies {
            assert!(*e >= 0.0, "Entropy should be non-negative");
        }
        assert_eq!(result.gate_count, 20);
    }

    #[test]
    fn test_z_gate() {
        let mut state = TensorNetworkState::new_zero_state(1);
        state.apply_hadamard(0);
        // Now |+> = (|0> + |1>) / sqrt(2)
        let c0 = state.coefficients()[0];
        let c1 = state.coefficients()[1];

        state.apply_z(0);
        // Now |-> = (|0> - |1>) / sqrt(2)
        assert!((state.coefficients()[0] - c0).abs() < 1e-10);
        assert!((state.coefficients()[1] + c1).abs() < 1e-10);
    }

    #[test]
    fn test_normalization_preserved() {
        let mut state = TensorNetworkState::new_random(4, Some(123));
        for _ in 0..10 {
            state.apply_hadamard(0);
            state.apply_cnot(1, 2);
            state.apply_z(3);
        }
        let total = state.total_probability();
        assert!((total - 1.0).abs() < 1e-10, "Total prob = {}", total);
    }
}

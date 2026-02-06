//! Unified tensor network facade.
//!
//! This module provides a single entry point for all tensor network functionality:
//!
//! - **Statevector**: Full statevector simulation for small systems (up to ~25 qubits)
//! - **MPS**: Matrix Product States for 1D systems with area-law entanglement
//! - **PEPS**: Projected Entangled Pair States for 2D systems
//!
//! # Choosing the Right Representation
//!
//! | System | Entanglement | Recommended |
//! |--------|--------------|-------------|
//! | Small (< 20 qubits) | Any | Statevector |
//! | 1D chain | Area law | MPS |
//! | 2D lattice | Area law | PEPS |
//! | Large entanglement | Volume law | Statevector (limited) |
//!
//! # Example
//!
//! ```rust
//! use quantum_core::tensor_networks::{
//!     TensorNetworkState, MatrixProductState, Peps,
//! };
//!
//! // Statevector for small systems
//! let mut sv = TensorNetworkState::new_zero_state(4);
//! sv.apply_hadamard(0);
//! sv.apply_cnot(0, 1);
//!
//! // MPS for 1D chains
//! let mut mps = MatrixProductState::new_zero_state(10);
//! mps.apply_hadamard(0);
//!
//! // PEPS for 2D lattices
//! let mut peps = Peps::new_zero_state(3, 3);
//! peps.apply_hadamard(0, 0);
//! ```

// Re-export all tensor network types
pub use crate::tensor_network_classical::{
    TensorNetworkState, EntropyResult, CircuitEvolutionResult,
    simulate_random_circuit, prepare_bell_state, prepare_ghz_state,
    bell_state_entropy, ghz_state_entropy,
};

pub use crate::mps::{
    MatrixProductState, MpsTensor,
};

pub use crate::peps::{
    Peps, PepsTensor,
};

/// Trait for types that can compute entanglement entropy.
pub trait EntanglementMeasure {
    /// Compute von Neumann entropy of the specified subsystem.
    ///
    /// For bipartite entanglement, this measures the entropy of
    /// the reduced density matrix after tracing out the complement.
    fn entanglement_entropy(&self, subsystem: &[usize]) -> f64;

    /// Check if the state is approximately a product state.
    fn is_product_state(&self, tolerance: f64) -> bool {
        // A product state has zero entanglement for any bipartition
        let n = self.n_sites();
        if n < 2 {
            return true;
        }
        let entropy = self.entanglement_entropy(&[0]);
        entropy.abs() < tolerance
    }

    /// Number of sites (qubits) in the system.
    fn n_sites(&self) -> usize;
}

impl EntanglementMeasure for TensorNetworkState {
    fn entanglement_entropy(&self, subsystem: &[usize]) -> f64 {
        // Use the maximum index in subsystem as the bipartition point
        if subsystem.is_empty() {
            return 0.0;
        }
        let k = *subsystem.iter().max().unwrap_or(&0);
        let entropy_result = self.measure_entropy(k);
        entropy_result.entropy
    }

    fn n_sites(&self) -> usize {
        self.n_qubits()
    }
}

impl EntanglementMeasure for MatrixProductState {
    fn entanglement_entropy(&self, subsystem: &[usize]) -> f64 {
        // For MPS, use the first site in subsystem as cut point
        if subsystem.is_empty() {
            return 0.0;
        }
        let cut_site = *subsystem.iter().max().unwrap_or(&0);
        self.measure_entropy(cut_site)
    }

    fn n_sites(&self) -> usize {
        self.n_qubits
    }
}

impl EntanglementMeasure for Peps {
    fn entanglement_entropy(&self, _subsystem: &[usize]) -> f64 {
        // PEPS entropy computation is more complex - use boundary MPS
        // For now, return estimate based on boundary length
        // TODO: Implement proper boundary contraction entropy
        0.0
    }

    fn n_sites(&self) -> usize {
        self.rows * self.cols
    }
}

/// Estimate memory requirements for different tensor network representations.
pub fn estimate_memory_bytes(n_qubits: usize, representation: &str) -> usize {
    match representation {
        "statevector" => {
            // 2^n complex numbers, each 16 bytes (2 f64)
            (1_usize << n_qubits) * 16
        }
        "mps" => {
            // Roughly O(n * chi^2 * d) where chi is bond dimension
            // Assume chi = 32, d = 2
            let chi = 32;
            let d = 2;
            n_qubits * chi * chi * d * 16
        }
        "peps" => {
            // O(n * chi^4 * d) for 2D with 4 virtual bonds
            let chi = 4; // PEPS typically needs smaller chi
            let d = 2;
            n_qubits * chi * chi * chi * chi * d * 16
        }
        _ => 0,
    }
}

/// Suggest the best tensor network representation for a given system.
pub fn suggest_representation(n_qubits: usize, geometry: &str) -> &'static str {
    match geometry {
        "chain" | "1d" | "linear" => {
            if n_qubits <= 20 {
                "statevector"
            } else {
                "mps"
            }
        }
        "grid" | "2d" | "lattice" => {
            if n_qubits <= 16 {
                "statevector"
            } else {
                "peps"
            }
        }
        "all_to_all" | "complete" => {
            // High entanglement - statevector is often best
            "statevector"
        }
        _ => {
            if n_qubits <= 20 {
                "statevector"
            } else {
                "mps"
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entanglement_measure_statevector() {
        let mut state = TensorNetworkState::new_zero_state(2);
        // Zero state is product state
        assert!(state.is_product_state(1e-10));

        // Create Bell state
        state.apply_hadamard(0);
        state.apply_cnot(0, 1);
        // Bell state is maximally entangled
        let entropy = state.entanglement_entropy(&[0]);
        assert!(entropy > 0.5); // Should be close to ln(2) ~ 0.693
    }

    #[test]
    fn test_entanglement_measure_mps() {
        let mps = MatrixProductState::new_zero_state(4);
        // Product state has zero entropy
        assert!(mps.is_product_state(1e-10));
    }

    #[test]
    fn test_memory_estimation() {
        // 10 qubits statevector: 2^10 * 16 = 16384 bytes
        assert_eq!(estimate_memory_bytes(10, "statevector"), 16384);

        // For large systems (20+ qubits), MPS is much smaller
        // 20 qubits: SV = 2^20 * 16 = 16 MB
        // MPS (chi=32): 20 * 32 * 32 * 2 * 16 = 655,360 bytes
        let mps_mem = estimate_memory_bytes(20, "mps");
        let sv_mem = estimate_memory_bytes(20, "statevector");
        assert!(mps_mem < sv_mem);
    }

    #[test]
    fn test_representation_suggestion() {
        assert_eq!(suggest_representation(8, "chain"), "statevector");
        assert_eq!(suggest_representation(30, "chain"), "mps");
        assert_eq!(suggest_representation(25, "grid"), "peps");
    }
}

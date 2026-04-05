//! Quantum Information Geometry and Exceptional Structures.
//!
//! Models Innovation 3 from the Unified Synthesis: Quantum Geometry.
//! Integrates quantum mechanical observables with exceptional geometries (G2, F4, E8).

use crate::higher_cd::SparseApeironState;

/// Metric tensor representation for Quantum Information Geometry.
/// In the exceptional context, the distance between quantum states is
/// modulated by the underlying Cayley-Dickson structure.
pub struct QuantumInformationMetric {
    pub dimension: usize,
}

impl QuantumInformationMetric {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    /// Fubini-Study distance between two sparse states in an exceptional geometry.
    /// ds^2 = arccos(|<psi|phi>|)
    pub fn fubini_study_distance(&self, psi: &SparseApeironState, phi: &SparseApeironState) -> f64 {
        assert_eq!(psi.dim, self.dimension);
        assert_eq!(phi.dim, self.dimension);

        let dot = psi.dot(phi);
        let norm_psi = psi.norm();
        let norm_phi = phi.norm();

        if norm_psi < 1e-15 || norm_phi < 1e-15 {
            return 0.0;
        }

        let cos_theta = (dot / (norm_psi * norm_phi)).abs().min(1.0);
        cos_theta.acos()
    }

    /// Heuristic Ricci scalar contribution from the exceptional symmetry group.
    /// Higher exceptional groups (E8) provide more "curvature" to the state space.
    pub fn exceptional_curvature(&self) -> f64 {
        match self.dimension {
            14 => 1.0,  // G2
            52 => 2.0,  // F4
            78 => 3.0,  // E6
            133 => 4.0, // E7
            248 => 5.0, // E8
            _ => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_distance() {
        let metric = QuantumInformationMetric::new(16);
        let psi = SparseApeironState::from_pairs(16, vec![(0, 1.0)]);
        let phi = SparseApeironState::from_pairs(16, vec![(1, 1.0)]);

        let dist = metric.fubini_study_distance(&psi, &phi);
        // Orthogonal states should have distance PI/2
        assert!((dist - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    }
}

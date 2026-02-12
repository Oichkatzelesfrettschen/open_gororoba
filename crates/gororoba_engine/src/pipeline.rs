//! End-to-end engine orchestration.

use crate::bit_source::FibonacciBitSource;
use crate::correction_layer::HomotopyCorrectionLayer;
use crate::dynamics_field::VacuumDynamicsLayer;
use crate::parity_filter::AntiDiagonalParityFilter;
use crate::topology_geometry::SlidingTriadTopology;
use crate::traits::{
    BitSourceLayer, CorrectionLayer, DynamicsLayer, ParityLayer, PipelineState, TopologyLayer,
    VerificationLayer, VerificationReport,
};
use crate::verification_layer::ThesisVerifier;

/// Concrete engine with default layer implementations.
#[derive(Debug, Clone)]
pub struct GororobaEngine {
    pub bit_source: FibonacciBitSource,
    pub parity: AntiDiagonalParityFilter,
    pub topology: SlidingTriadTopology,
    pub dynamics: VacuumDynamicsLayer,
    pub correction: HomotopyCorrectionLayer,
    pub verifier: ThesisVerifier,
}

impl Default for GororobaEngine {
    fn default() -> Self {
        Self {
            bit_source: FibonacciBitSource,
            parity: AntiDiagonalParityFilter,
            topology: SlidingTriadTopology::default(),
            dynamics: VacuumDynamicsLayer::default(),
            correction: HomotopyCorrectionLayer::default(),
            verifier: ThesisVerifier::default(),
        }
    }
}

impl GororobaEngine {
    /// Execute one full pipeline pass.
    pub fn run(&self, n_words: usize) -> (PipelineState, VerificationReport) {
        let words = self.bit_source.sample_words(n_words);
        let signs = self.parity.compute_signs(&words);
        let frustration = self.topology.frustration_density(&signs);
        let viscosity = self.dynamics.viscosity_field(&frustration);
        let correction_gain = self.correction.correction_gain(&frustration);

        let state = PipelineState {
            words,
            signs,
            frustration,
            viscosity,
            correction_gain,
        };
        let report = self.verifier.verify(&state);
        (state, report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_run_shapes() {
        let engine = GororobaEngine::default();
        let (state, report) = engine.run(128);
        assert_eq!(state.words.len(), 128);
        assert_eq!(state.signs.len(), 128);
        assert_eq!(state.frustration.len(), 128);
        assert_eq!(state.viscosity.len(), 128);
        assert!(!report.messages.is_empty());
    }
}

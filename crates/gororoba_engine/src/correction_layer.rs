//! Correction layer driven by neural-homotopy surrogate alignment.

use crate::traits::CorrectionLayer;
use neural_homotopy::{train_homotopy_surrogate, HomotopyTrainingConfig};

/// Correction layer computing a gain from algebraic frustration and model traces.
#[derive(Debug, Clone, Copy)]
pub struct HomotopyCorrectionLayer {
    pub epochs: usize,
}

impl Default for HomotopyCorrectionLayer {
    fn default() -> Self {
        Self { epochs: 24 }
    }
}

impl CorrectionLayer for HomotopyCorrectionLayer {
    fn correction_gain(&self, frustration: &[f64]) -> f64 {
        let mean_f = if frustration.is_empty() {
            0.0
        } else {
            frustration.iter().sum::<f64>() / frustration.len() as f64
        };
        let cfg = HomotopyTrainingConfig {
            epochs: self.epochs,
            ..HomotopyTrainingConfig::default()
        };
        let trace = train_homotopy_surrogate(cfg);
        (1.0 + mean_f) * trace.hubble_alignment
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::CorrectionLayer;

    #[test]
    fn test_correction_gain_finite() {
        let corr = HomotopyCorrectionLayer::default();
        let g = corr.correction_gain(&[0.2, 0.4, 0.6]);
        assert!(g.is_finite());
        assert!(g >= 0.0);
    }
}

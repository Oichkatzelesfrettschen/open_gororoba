//! Verification layer for falsifiability checks.

use crate::traits::{PipelineState, VerificationLayer, VerificationReport};
use vacuum_frustration::{omega_eff_from_phi, ScalarFrustrationMap, CASSINI_OMEGA_BD_LOWER_BOUND};

/// Default verification policy for thesis execution.
#[derive(Debug, Clone, Copy)]
pub struct ThesisVerifier {
    pub lambda: f64,
    pub omega_reference: f64,
}

impl Default for ThesisVerifier {
    fn default() -> Self {
        Self {
            lambda: 1.0,
            omega_reference: 50_000.0,
        }
    }
}

impl VerificationLayer for ThesisVerifier {
    fn verify(&self, state: &PipelineState) -> VerificationReport {
        let mut messages = Vec::new();
        let mut pass = true;

        if state.frustration.is_empty() {
            pass = false;
            messages.push("no frustration field produced".to_string());
        } else {
            let mean_f = state.frustration.iter().sum::<f64>() / state.frustration.len() as f64;
            let phi = ScalarFrustrationMap::new(self.lambda).phi(mean_f);
            let omega_eff = omega_eff_from_phi(phi, self.omega_reference);
            if omega_eff < CASSINI_OMEGA_BD_LOWER_BOUND {
                pass = false;
                messages.push(format!(
                    "Cassini refutation triggered: omega_eff={omega_eff:.2} < {}",
                    CASSINI_OMEGA_BD_LOWER_BOUND
                ));
            } else {
                messages.push(format!("Cassini bound satisfied: omega_eff={omega_eff:.2}"));
            }
        }

        if !state.viscosity.iter().all(|v| *v > 0.0 && v.is_finite()) {
            pass = false;
            messages.push("non-positive or non-finite viscosity detected".to_string());
        }

        VerificationReport { pass, messages }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verifier_flags_empty_state() {
        let verifier = ThesisVerifier::default();
        let state = PipelineState::default();
        let r = verifier.verify(&state);
        assert!(!r.pass);
    }
}

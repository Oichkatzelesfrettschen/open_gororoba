//! Symmetric-composition scaffolding for para-Hurwitz and triality-sensitive work.
//!
//! The fully classical Okubo multiplication remains future work. This module
//! currently provides the para-Hurwitz baseline, a first-class Okubo carrier,
//! and explicit triality actions.

use super::{exotic_octonions::ParaOctonion, octonion::Octonion};
use std::fmt;

pub fn para_hurwitz_multiply(lhs: &Octonion, rhs: &Octonion) -> Octonion {
    ParaOctonion::new(*lhs)
        .multiply(&ParaOctonion::new(*rhs))
        .value
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrialityAction {
    Identity,
    CycleForward,
    CycleBackward,
}

impl TrialityAction {
    pub fn label(&self) -> &'static str {
        match self {
            TrialityAction::Identity => "identity",
            TrialityAction::CycleForward => "cycle-forward",
            TrialityAction::CycleBackward => "cycle-backward",
        }
    }

    pub fn permutation(&self) -> [usize; 3] {
        match self {
            TrialityAction::Identity => [0, 1, 2],
            TrialityAction::CycleForward => [1, 2, 0],
            TrialityAction::CycleBackward => [2, 0, 1],
        }
    }

    pub fn apply_triplet<T: Clone>(&self, triple: &[T; 3]) -> [T; 3] {
        match self {
            TrialityAction::Identity => triple.clone(),
            TrialityAction::CycleForward => {
                [triple[1].clone(), triple[2].clone(), triple[0].clone()]
            }
            TrialityAction::CycleBackward => {
                [triple[2].clone(), triple[0].clone(), triple[1].clone()]
            }
        }
    }
}

impl fmt::Display for TrialityAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrialityOrbitSummary {
    pub action: TrialityAction,
    pub permutation: [usize; 3],
    pub input_norms: [f64; 3],
    pub output_norms: [f64; 3],
}

impl TrialityOrbitSummary {
    pub fn summary_row(&self) -> [f64; 6] {
        [
            self.input_norms[0],
            self.input_norms[1],
            self.input_norms[2],
            self.output_norms[0],
            self.output_norms[1],
            self.output_norms[2],
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OkuboElement {
    coeffs: [f64; 8],
}

impl OkuboElement {
    pub fn new(coeffs: [f64; 8]) -> Self {
        Self { coeffs }
    }

    pub fn zero() -> Self {
        Self { coeffs: [0.0; 8] }
    }

    pub fn coeffs(&self) -> &[f64; 8] {
        &self.coeffs
    }

    pub fn norm_squared(&self) -> f64 {
        self.coeffs.iter().map(|value| value * value).sum()
    }

    pub fn as_para_octonion(&self) -> ParaOctonion {
        ParaOctonion::new(Octonion::new(self.coeffs))
    }

    pub fn triality_orbit(
        &self,
        action: TrialityAction,
        peers: &[OkuboElement; 3],
    ) -> [OkuboElement; 3] {
        let _ = self;
        action.apply_triplet(peers)
    }

    pub fn orbit_summary(
        &self,
        action: TrialityAction,
        peers: &[OkuboElement; 3],
    ) -> TrialityOrbitSummary {
        let _ = self;
        let orbit = action.apply_triplet(peers);
        TrialityOrbitSummary {
            action,
            permutation: action.permutation(),
            input_norms: peers.map(|value| value.norm_squared()),
            output_norms: orbit.map(|value| value.norm_squared()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triality_cycle_roundtrip() {
        let triple = [1, 2, 3];
        let cycled = TrialityAction::CycleForward.apply_triplet(&triple);
        assert_eq!(cycled, [2, 3, 1]);
        assert_eq!(TrialityAction::CycleForward.permutation(), [1, 2, 0]);

        let restored = TrialityAction::CycleBackward.apply_triplet(&cycled);
        assert_eq!(restored, triple);
    }

    #[test]
    fn test_para_hurwitz_basis_square_flips_sign() {
        let e1 = Octonion::basis(1);
        let product = para_hurwitz_multiply(&e1, &e1);
        assert!((product.components[0] + 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_okubo_triality_orbit_uses_explicit_action() {
        let a = OkuboElement::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let b = OkuboElement::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let c = OkuboElement::new([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let orbit = a.triality_orbit(TrialityAction::CycleForward, &[a, b, c]);
        assert_eq!(orbit, [b, c, a]);
    }

    #[test]
    fn test_triality_orbit_summary_preserves_norm_inventory() {
        let a = OkuboElement::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let b = OkuboElement::new([0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let c = OkuboElement::new([0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let summary = a.orbit_summary(TrialityAction::CycleForward, &[a, b, c]);
        assert_eq!(summary.action.to_string(), "cycle-forward");
        assert_eq!(summary.permutation, [1, 2, 0]);
        assert_eq!(summary.input_norms, [1.0, 4.0, 9.0]);
        assert_eq!(summary.output_norms, [4.0, 9.0, 1.0]);
        assert_eq!(summary.summary_row(), [1.0, 4.0, 9.0, 4.0, 9.0, 1.0]);
    }
}

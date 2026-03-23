//! Symmetric-composition scaffolding for para-Hurwitz and triality-sensitive work.
//!
//! The fully classical Okubo multiplication remains future work. This module
//! currently provides the para-Hurwitz baseline, a first-class Okubo carrier,
//! and explicit triality actions.

use super::{exotic_octonions::ParaOctonion, octonion::Octonion};

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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triality_cycle_roundtrip() {
        let triple = [1, 2, 3];
        let cycled = TrialityAction::CycleForward.apply_triplet(&triple);
        assert_eq!(cycled, [2, 3, 1]);

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
}

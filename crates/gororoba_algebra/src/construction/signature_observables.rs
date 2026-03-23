//! Signature-aware observable routing for compact and split algebra lanes.
//!
//! The multiplication kernel is signature-parameterized, but not every
//! observable has the same interpretation in every metric regime. This module
//! makes that distinction explicit for the bridge architecture.

use super::{exotic_octonions::HybridSignatureOctonion, split_octonion::SplitOctonion};
use cd_kernel::{
    cayley_dickson::CdSignature, cross_generational_friction, is_zero_divisor_koebisu, koebisu_d2,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObservableSignatureRegime {
    CompactEuclidean,
    SplitIndefinite,
}

pub trait SignatureAwareObservable {
    type Output;

    fn regime(&self) -> ObservableSignatureRegime;
    fn evaluate(&self) -> Self::Output;
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompactKoebisuResult {
    pub d2: f64,
    pub is_zero_divisor: bool,
}

pub struct CompactKoebisuObservable<'a> {
    vector: &'a [f64],
    epsilon: f64,
}

impl<'a> CompactKoebisuObservable<'a> {
    pub fn new(vector: &'a [f64], epsilon: f64) -> Self {
        Self { vector, epsilon }
    }
}

impl SignatureAwareObservable for CompactKoebisuObservable<'_> {
    type Output = CompactKoebisuResult;

    fn regime(&self) -> ObservableSignatureRegime {
        ObservableSignatureRegime::CompactEuclidean
    }

    fn evaluate(&self) -> Self::Output {
        let d2 = koebisu_d2(self.vector);
        CompactKoebisuResult {
            d2,
            is_zero_divisor: is_zero_divisor_koebisu(self.vector, self.epsilon),
        }
    }
}

pub fn koebisu_d2_compact(vector: &[f64], epsilon: f64) -> CompactKoebisuResult {
    CompactKoebisuObservable::new(vector, epsilon).evaluate()
}

pub struct CompactFrictionObservable<'a> {
    mode_a: usize,
    mode_b: usize,
    subalgebra_i: &'a [usize],
    subalgebra_j: &'a [usize],
}

impl<'a> CompactFrictionObservable<'a> {
    pub fn new(
        mode_a: usize,
        mode_b: usize,
        subalgebra_i: &'a [usize],
        subalgebra_j: &'a [usize],
    ) -> Self {
        Self {
            mode_a,
            mode_b,
            subalgebra_i,
            subalgebra_j,
        }
    }
}

impl SignatureAwareObservable for CompactFrictionObservable<'_> {
    type Output = f64;

    fn regime(&self) -> ObservableSignatureRegime {
        ObservableSignatureRegime::CompactEuclidean
    }

    fn evaluate(&self) -> Self::Output {
        cross_generational_friction(
            self.mode_a,
            self.mode_b,
            self.subalgebra_i,
            self.subalgebra_j,
        )
    }
}

pub fn cross_generational_friction_compact(
    mode_a: usize,
    mode_b: usize,
    subalgebra_i: &[usize],
    subalgebra_j: &[usize],
) -> f64 {
    CompactFrictionObservable::new(mode_a, mode_b, subalgebra_i, subalgebra_j).evaluate()
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplitNormDiagnostics {
    pub norm_squared: f64,
    pub is_isotropic: bool,
}

pub struct SplitNormObservable<'a> {
    value: &'a SplitOctonion,
}

impl<'a> SplitNormObservable<'a> {
    pub fn new(value: &'a SplitOctonion) -> Self {
        Self { value }
    }
}

impl SignatureAwareObservable for SplitNormObservable<'_> {
    type Output = SplitNormDiagnostics;

    fn regime(&self) -> ObservableSignatureRegime {
        ObservableSignatureRegime::SplitIndefinite
    }

    fn evaluate(&self) -> Self::Output {
        let norm_squared = self.value.norm_squared();
        SplitNormDiagnostics {
            norm_squared,
            is_isotropic: norm_squared.abs() < 1e-12,
        }
    }
}

pub fn split_octonion_diagnostics(value: &SplitOctonion) -> SplitNormDiagnostics {
    SplitNormObservable::new(value).evaluate()
}

pub fn regime_for_signature(signature: &CdSignature) -> ObservableSignatureRegime {
    if signature.is_standard() {
        ObservableSignatureRegime::CompactEuclidean
    } else {
        ObservableSignatureRegime::SplitIndefinite
    }
}

pub fn regime_for_hybrid_octonion(value: &HybridSignatureOctonion) -> ObservableSignatureRegime {
    regime_for_signature(value.signature())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compact_koebisu_wrapper_detects_standard_witness() {
        let mut witness = [0.0; 16];
        witness[1] = 1.0;
        witness[10] = 1.0;

        let result = koebisu_d2_compact(&witness, 1e-12);
        assert!(result.d2.abs() < 1e-12);
        assert!(result.is_zero_divisor);
    }

    #[test]
    fn test_compact_friction_wrapper_matches_kernel() {
        let left = [1, 2, 4, 7];
        let right = [1, 3, 5, 6];
        let wrapped = cross_generational_friction_compact(1, 8, &left, &right);
        let direct = cross_generational_friction(1, 8, &left, &right);
        assert!((wrapped - direct).abs() < 1e-12);
    }

    #[test]
    fn test_split_norm_diagnostics_detect_isotropy() {
        let isotropic = SplitOctonion::new([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        let diagnostics = split_octonion_diagnostics(&isotropic);
        assert!(diagnostics.is_isotropic);
        assert!(diagnostics.norm_squared.abs() < 1e-12);
    }

    #[test]
    fn test_hybrid_signature_regime_tracks_split_levels() {
        let compact = HybridSignatureOctonion::new(CdSignature::standard(8), [0.0; 8]);
        let split = HybridSignatureOctonion::new(CdSignature::split(8), [0.0; 8]);

        assert_eq!(
            regime_for_hybrid_octonion(&compact),
            ObservableSignatureRegime::CompactEuclidean
        );
        assert_eq!(
            regime_for_hybrid_octonion(&split),
            ObservableSignatureRegime::SplitIndefinite
        );
    }
}

//! Exotic octonion-adjacent constructions promoted from the test lane.
//!
//! These types give the repo reusable surfaces for para-Hurwitz, dual,
//! bioctonion, and hybrid-signature experiments.

use super::octonion::Octonion;
use cd_kernel::cayley_dickson::{CdSignature, cd_multiply_split};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DualOctonion {
    pub real: Octonion,
    pub dual: Octonion,
}

impl DualOctonion {
    pub fn new(real: Octonion, dual: Octonion) -> Self {
        Self { real, dual }
    }

    pub fn multiply(&self, other: &Self) -> Self {
        let ac = self.real.multiply(&other.real);
        let ad = self.real.multiply(&other.dual);
        let bc = self.dual.multiply(&other.real);
        Self {
            real: ac,
            dual: ad.add(&bc),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bioctonion {
    pub real: Octonion,
    pub imag: Octonion,
}

impl Bioctonion {
    pub fn new(real: Octonion, imag: Octonion) -> Self {
        Self { real, imag }
    }

    pub fn multiply(&self, other: &Self) -> Self {
        let ac = self.real.multiply(&other.real);
        let bd = self.imag.multiply(&other.imag);
        let ad = self.real.multiply(&other.imag);
        let bc = self.imag.multiply(&other.real);
        Self {
            real: ac.sub(&bd),
            imag: ad.add(&bc),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParaOctonion {
    pub value: Octonion,
}

impl ParaOctonion {
    pub fn new(value: Octonion) -> Self {
        Self { value }
    }

    pub fn multiply(&self, other: &Self) -> Self {
        Self {
            value: self.value.conjugate().multiply(&other.value.conjugate()),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct HybridSignatureOctonion {
    signature: CdSignature,
    pub coeffs: [f64; 8],
}

impl HybridSignatureOctonion {
    pub fn new(signature: CdSignature, coeffs: [f64; 8]) -> Self {
        assert_eq!(
            signature.dim(),
            8,
            "Hybrid octonions require an 8D signature"
        );
        Self { signature, coeffs }
    }

    pub fn signature(&self) -> &CdSignature {
        &self.signature
    }

    pub fn multiply(&self, other: &Self) -> Self {
        assert_eq!(
            self.signature, other.signature,
            "Hybrid octonions must share a signature"
        );
        let product = cd_multiply_split(&self.coeffs, &other.coeffs, &self.signature);
        let mut coeffs = [0.0; 8];
        coeffs.copy_from_slice(&product);
        Self {
            signature: self.signature.clone(),
            coeffs,
        }
    }
}

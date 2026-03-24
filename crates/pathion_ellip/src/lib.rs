pub mod angular_size;
pub mod carlson;
pub mod diagonalizer;
pub mod pathion_eigenvalues;
pub mod pathion_resonance;
pub mod pathion_shadow;
pub mod quartic;
pub mod shadow_boundary;

pub use pathion_eigenvalues::PathionEigenvalueSpectrum;
pub use pathion_resonance::{PathionResonanceReport, ResonanceBand, ResonanceConfig};

#[cfg(test)]
mod tests;

//! cosmology_core: Gravastar TOV solver, spectral dimensions, fractal cosmology.
//!
//! This crate provides:
//! - Polytropic and anisotropic gravastar models
//! - Spectral dimension flow analysis
//! - Kraichnan k^{-3} enstrophy cascade matching
//!
//! # Literature
//! - Mazur & Mottola (2004): Gravastar proposal
//! - Cattoen, Faber & Visser (2005): Anisotropic pressure
//! - Calcagni (2010): Fractal spacetime
//! - Kraichnan (1967): 2D enstrophy cascade

pub mod gravastar;
pub mod spectral;

pub use gravastar::{
    PolytropicEos, AnisotropicParams, TovState, GravastarSolution,
    GravastarConfig, solve_gravastar, polytropic_stability_sweep,
    anisotropic_stability_test, StabilityResult, AnisotropicStabilityResult,
};

pub use spectral::{
    calcagni_spectral_dimension, cdt_spectral_dimension,
    k_minus_3_spectrum, kolmogorov_spectrum, kraichnan_enstrophy_spectrum,
    parisi_sourlas_effective_dimension, parisi_sourlas_spectrum_exponent,
    analyze_k_minus_3_origin, SpectralAnalysisResult,
};

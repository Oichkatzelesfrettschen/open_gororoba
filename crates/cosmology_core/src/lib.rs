//! cosmology_core: Gravastar TOV solver, spectral dimensions, bounce cosmology.
//!
//! This crate provides:
//! - Polytropic and anisotropic gravastar models
//! - Spectral dimension flow analysis
//! - Kraichnan k^{-3} enstrophy cascade matching
//! - Quantum bounce cosmology with observational fitting
//!
//! # Literature
//! - Mazur & Mottola (2004): Gravastar proposal
//! - Cattoen, Faber & Visser (2005): Anisotropic pressure
//! - Calcagni (2010): Fractal spacetime
//! - Kraichnan (1967): 2D enstrophy cascade
//! - Pinto-Neto & Fabris (2013): Bohmian bounce cosmology

pub mod gravastar;
pub mod spectral;
pub mod bounce;

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

pub use bounce::{
    BounceState, BounceParams, BounceResult, FitResult,
    simulate_bounce, hubble_e_lcdm, hubble_e_bounce,
    luminosity_distance, distance_modulus, cmb_shift_parameter,
    bao_sound_horizon, spectral_index_bounce, chi2_distance_modulus,
    C_KM_S, OMEGA_B_H2, Z_STAR,
};

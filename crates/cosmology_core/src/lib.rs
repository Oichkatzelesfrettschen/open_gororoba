//! cosmology_core: Gravastar TOV solver, spectral dimensions, bounce cosmology.
//!
//! This crate provides:
//! - Polytropic and anisotropic gravastar models
//! - Spectral dimension flow analysis
//! - Kraichnan k^{-3} enstrophy cascade matching
//! - Quantum bounce cosmology with observational fitting
//! - Flat Lambda-CDM (FLRW) cosmology with struct-based interface
//! - Axiodilaton scalar field cosmology
//!
//! # Literature
//! - Mazur & Mottola (2004): Gravastar proposal
//! - Cattoen, Faber & Visser (2005): Anisotropic pressure
//! - Calcagni (2010): Fractal spacetime
//! - Kraichnan (1967): 2D enstrophy cascade
//! - Pinto-Neto & Fabris (2013): Bohmian bounce cosmology
//! - Planck Collaboration VI (2020): A&A 641, A6
//! - Hogg (1999): arXiv:astro-ph/9905116 (distance measures)

use gauss_quad::GaussLegendre;

pub mod axiodilaton;
pub mod bounce;
pub mod dimensional_geometry;
pub mod distances;
pub mod flrw;
pub mod gravastar;
pub mod observational;
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

pub use bounce::{
    BounceState, BounceParams, BounceResult, FitResult,
    simulate_bounce, hubble_e_lcdm, hubble_e_bounce,
    luminosity_distance, distance_modulus, cmb_shift_parameter,
    bao_sound_horizon, spectral_index_bounce, chi2_distance_modulus,
    // Synthetic data and fitting pipeline
    SyntheticSnData, SyntheticBaoData,
    generate_synthetic_sn_data, generate_synthetic_bao_data,
    chi2_sn, chi2_bao, fit_model, run_observational_fit,
    C_KM_S, OMEGA_B_H2, Z_STAR,
};

pub use dimensional_geometry::{
    unit_sphere_surface_area, ball_volume, sample_dimensional_range,
};

pub use observational::{
    RealSnData, RealBaoData, ObsFitResult, ModelComparison,
    chi2_sn_real, chi2_bao_real, bao_data_point_count,
    fit_real_data, compare_models,
    filter_pantheon_data, desi_to_real_bao,
};

pub use distances::{
    comoving_distance, macquart_dm_cosmic, dm_excess_to_redshift,
    dm_to_comoving, angular_diameter_distance, radec_to_cartesian,
};

pub use flrw::{
    FlatLCDM, deceleration_parameter, z_equality,
    distance_duality_deviation, verify_distance_duality,
    PLANCK18_H0, PLANCK18_OMEGA_M, PLANCK18_OMEGA_B,
    PLANCK18_T_CMB, PLANCK18_SOUND_HORIZON,
};

/// Gauss-Legendre quadrature over [a, b].
///
/// Replaces hand-rolled Simpson's rule with gauss-quad's GaussLegendre,
/// which converges exponentially for smooth integrands. Degree 50 gives
/// far better accuracy than 500-point Simpson at 1/10th the evaluations.
pub(crate) fn gl_integrate<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, degree: usize) -> f64 {
    let quad = GaussLegendre::new(degree).expect("valid quadrature degree");
    quad.integrate(a, b, f)
}

//! cosmology_core: Gravastar TOV solver, spectral dimensions, bounce cosmology.
//!
//! This crate provides:
//! - Polytropic and anisotropic gravastar models
//! - Spectral dimension flow analysis
//! - Kraichnan k^{-3} enstrophy cascade matching
//! - Quantum bounce cosmology with observational fitting
//! - Flat Lambda-CDM (FLRW) cosmology with struct-based interface
//! - Axiodilaton scalar field cosmology
//! - Relativistic polytropic equation of state
//! - Neutron star TOV solver with tidal deformability
//!
//! # Literature
//! - Mazur & Mottola (2004): Gravastar proposal
//! - Cattoen, Faber & Visser (2005): Anisotropic pressure
//! - Calcagni (2010): Fractal spacetime
//! - Kraichnan (1967): 2D enstrophy cascade
//! - Pinto-Neto & Fabris (2013): Bohmian bounce cosmology
//! - Planck Collaboration VI (2020): A&A 641, A6
//! - Hogg (1999): arXiv:astro-ph/9905116 (distance measures)
//! - Shapiro & Teukolsky (1983): Black Holes, White Dwarfs, and Neutron Stars
//! - Oppenheimer & Volkoff (1939): Phys. Rev. 55, 374
//! - Hinderer (2008): ApJ 677, 1216 (tidal Love number)
//! - Yagi & Yunes (2013): Science 341, 365 (I-Love-Q universality)
//! - Abbott et al. (2017): PRL 119, 161101 (GW170817)

use gauss_quad::GaussLegendre;

pub mod axiodilaton;
pub mod bounce;
pub mod dimensional_geometry;
pub mod distances;
pub mod eos;
pub mod flrw;
pub mod gravastar;
pub mod observational;
pub mod spectral;
pub mod tov;

pub use gravastar::{
    anisotropic_stability_test, polytropic_stability_sweep, solve_gravastar, AnisotropicParams,
    AnisotropicStabilityResult, GravastarConfig, GravastarSolution, PolytropicEos, StabilityResult,
    TovState,
};

pub use spectral::{
    analyze_k_minus_3_origin, calcagni_spectral_dimension, cdt_spectral_dimension,
    k_minus_3_spectrum, kolmogorov_spectrum, kraichnan_enstrophy_spectrum,
    parisi_sourlas_effective_dimension, parisi_sourlas_spectrum_exponent, SpectralAnalysisResult,
};

pub use bounce::{
    bao_sound_horizon,
    chi2_bao,
    chi2_distance_modulus,
    chi2_sn,
    cmb_shift_parameter,
    distance_modulus,
    fit_model,
    generate_synthetic_bao_data,
    generate_synthetic_sn_data,
    hubble_e_bounce,
    hubble_e_lcdm,
    luminosity_distance,
    run_observational_fit,
    simulate_bounce,
    spectral_index_bounce,
    BounceParams,
    BounceResult,
    BounceState,
    FitResult,
    SyntheticBaoData,
    // Synthetic data and fitting pipeline
    SyntheticSnData,
    C_KM_S,
    OMEGA_B_H2,
    Z_STAR,
};

pub use dimensional_geometry::{ball_volume, sample_dimensional_range, unit_sphere_surface_area};

pub use observational::{
    bao_data_point_count, chi2_bao_real, chi2_sn_real, compare_models, desi_to_real_bao,
    filter_pantheon_data, fit_real_data, ModelComparison, ObsFitResult, RealBaoData, RealSnData,
};

pub use distances::{
    angular_diameter_distance, comoving_distance, dm_excess_to_redshift, dm_to_comoving,
    macquart_dm_cosmic, radec_to_cartesian,
};

pub use eos::{
    gamma_from_index, polytropic_index, Polytrope, GAMMA_NONREL_DEGENERATE, GAMMA_RADIATION,
    GAMMA_STIFF, GAMMA_ULTRAREL_DEGENERATE,
};

pub use tov::{
    combined_tidal_deformability, integrate_neutron_star, mass_radius_relation,
    tidal_deformability, tidal_love_number_k2, tov_maximum_mass, MassRadiusPoint,
    NeutronStarProfile,
};

pub use flrw::{
    age_at_z, apply_redshift_to_wavelength, deceleration_parameter, distance_duality_deviation,
    lookback_time, redshift_flux_dimming, universe_age, verify_distance_duality,
    wavelength_to_redshift, z_equality, FlatLCDM, PLANCK18_H0, PLANCK18_OMEGA_B, PLANCK18_OMEGA_M,
    PLANCK18_SOUND_HORIZON, PLANCK18_T_CMB,
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

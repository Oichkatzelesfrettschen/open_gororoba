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
//! - UDG halo kinematics (CDG-2 / Perseus; fifth observational pillar)
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
//! - Li et al. (2025): arXiv:2506.15644 (CDG-2 ultra-diffuse galaxy)
//!
//! # Canonical docs
//!
//! Use crate rustdoc and the mdBook as the primary web-viewable documentation
//! surfaces for this crate. The generated LaTeX appendices under `docs/latex/`
//! are publication outputs and may lag behind the actual Rust implementation.
//!
//! In particular:
//!
//! - module docs and tests define current code behavior
//! - `docs/book/` explains architecture and navigation
//! - `registry/canonical/control_plane.sqlite3` tracks evidence posture for
//!   physical interpretations
//! - `registry/claims.toml` is a generated compatibility export from that
//!   control plane

use gauss_quad::GaussLegendre;

pub mod axiodilaton;
pub mod bounce;
pub mod bypass_models;
pub mod cdg2_mapping;
pub mod cmb_axis;
pub mod deka_voudon_cmb;
pub mod dimensional_geometry;
pub mod distances;
pub mod eos;
#[cfg(feature = "euclid-catalog")]
pub mod euclid_morphology;
pub mod flrw;
pub mod galaxy_pipeline;
pub mod gravastar;
pub mod gravastar_potential;
pub mod halo;
pub mod halo_profile;
pub mod harmonic_halos;
pub mod harmonic_stacking;
pub mod homotopy_bridge;
pub mod manga_sim;
pub mod mass_ladder;
pub mod nfw_utils;
pub mod observational;
pub mod optimizer;
#[cfg(feature = "argmin-optimizers")]
pub mod optimizer_argmin;
pub mod orthoplex_crystal;
pub mod orthoplex_diffusion;
pub mod sersic;
pub mod slit_projection;
pub mod spectral;
pub mod tov;
pub mod voudon_friedmann;

pub use gravastar::{
    AnisotropicParams, AnisotropicStabilityResult, GravastarConfig, GravastarSolution,
    PolytropicEos, StabilityResult, TovState, anisotropic_stability_test,
    polytropic_stability_sweep, solve_gravastar,
};

pub use spectral::{
    SpectralAnalysisResult, analyze_k_minus_3_origin, calcagni_spectral_dimension,
    cdt_spectral_dimension, k_minus_3_spectrum, kolmogorov_spectrum, kraichnan_enstrophy_spectrum,
    parisi_sourlas_effective_dimension, parisi_sourlas_spectrum_exponent,
};

pub use bounce::{
    BounceParams,
    BounceResult,
    BounceState,
    C_KM_S,
    FitResult,
    OMEGA_B_H2,
    SyntheticBaoData,
    // Synthetic data and fitting pipeline
    SyntheticSnData,
    Z_STAR,
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
};

pub use dimensional_geometry::{ball_volume, sample_dimensional_range, unit_sphere_surface_area};

pub use observational::{
    CMB_SHIFT_R_ERR, CMB_SHIFT_R_OBS, CcMeasurement, FsigMeasurement, ModelComparison,
    ObsFitResult, RealBaoData, RealSnData, SIGMA8_PLANCK, bao_data_point_count, chi2_bao_real,
    chi2_cc, chi2_cmb_shift, chi2_fsig8, chi2_sn_real, compare_models, compute_growth_batch,
    compute_growth_fsig8, compute_precision_matrix, cosmic_chronometer_data, desi_to_real_bao,
    extract_cov_submatrix, filter_pantheon_data, filter_pantheon_data_with_indices, fit_real_data,
    growth_rate_data, set_sn_precision_from_cov,
};

pub use distances::{
    angular_diameter_distance, comoving_distance, dm_excess_to_redshift, dm_to_comoving,
    macquart_dm_cosmic, radec_to_cartesian,
};

pub use eos::{
    GAMMA_NONREL_DEGENERATE, GAMMA_RADIATION, GAMMA_STIFF, GAMMA_ULTRAREL_DEGENERATE, Polytrope,
    gamma_from_index, polytropic_index,
};

pub use tov::{
    MassRadiusPoint, NeutronStarProfile, combined_tidal_deformability, integrate_neutron_star,
    mass_radius_relation, tidal_deformability, tidal_love_number_k2, tov_maximum_mass,
};

pub use homotopy_bridge::{
    HomotopyStressEnergy, ObstructionSweepResult, SweepConfig, homotopy_lambda,
    homotopy_stress_energy, solve_gravastar_homotopy, sweep_obstruction_coupling,
};

pub use halo::{
    CDG2_DM_FRACTION_MIN, CDG2_STELLAR_MASS_SOLAR, DELTA_C, PERSEUS_Z, bbks_transfer,
    linear_growth_factor, nfw_density, nfw_enclosed_mass, nfw_velocity_dispersion,
    press_schechter_mass_function, press_schechter_orthoplex, sigma_mass, udg_abundance_ratio,
};

pub use nfw_utils::{
    NfwParams, concentration_mass_relation, nfw_density_from_params, nfw_enclosed_mass_from_params,
    nfw_params_from_mass, nfw_r200_kpc, nfw_rho_s_from_c, rho_crit_kpc_at_z,
};

pub use flrw::{
    FlatLCDM, PLANCK18_H0, PLANCK18_OMEGA_B, PLANCK18_OMEGA_M, PLANCK18_SOUND_HORIZON,
    PLANCK18_T_CMB, age_at_z, apply_redshift_to_wavelength, deceleration_parameter,
    distance_duality_deviation, lookback_time, redshift_flux_dimming, universe_age,
    verify_distance_duality, wavelength_to_redshift, z_equality,
};

pub use cmb_axis::VoudonCmbAnalyzer;
pub use deka_voudon_cmb::{
    CosmicWebGenerator, DekaVoudonCmbAnalyzer, MultipoleResult, angular_separation_degrees,
    extract_multipoles,
};
pub use voudon_friedmann::VoudonFriedmann;

pub use harmonic_halos::{
    HarmonicHaloConfig, default_box_kite_phases, default_phases, harmonic_halo_force_modulation,
    harmonic_halo_modulation, v_circ_nfw, v_circ_with_halos,
};

pub use harmonic_stacking::{
    CdDimensionParams, EigenmodeStackResult, NormalizedPoint, NormalizedResiduals, StackingConfig,
    StackingResult, detection_threshold, eigenmode_stack, predicted_wavenumbers,
    predicted_wavenumbers_cd, stack_residuals,
};

pub use sersic::{
    SersicLbmConfig, box_counting_fractal_dim, box_counting_fractal_dim_f32,
    box_counting_fractal_dim_threshold, otsu_threshold, otsu_threshold_f32, sersic_bn,
    sersic_deprojected_density, sersic_profile_2d, sersic_to_lbm_density, sersic_total_luminosity,
};

pub use cdg2_mapping::{
    DarkHaloFalsificationResult, evaluate_cdg2_consistency, spectral_dim_at_cdg2,
    zd_bottleneck_volume_fraction,
};

pub use optimizer::{NelderMeadConfig, bounded_nelder_mead, bounded_nelder_mead_single};
#[cfg(feature = "argmin-optimizers")]
pub use optimizer_argmin::{bounded_nelder_mead_argmin, fit_nfw_profile_argmin};

pub use orthoplex_diffusion::{
    OrthoplexComparison, OrthoplexFitResult, OrthoplexParams, chi2_bao_orthoplex,
    chi2_cc_orthoplex, chi2_cmb_shift_orthoplex, chi2_fsig8_orthoplex, chi2_sn_orthoplex,
    chi2_udg_orthoplex, cmb_shift_parameter_orthoplex, compare_orthoplex, compare_orthoplex_all,
    compare_orthoplex_fixed_beta, dark_energy_density_ratio, diffusion_time,
    distance_modulus_orthoplex, fit_orthoplex_model, fit_orthoplex_model_fixed_beta,
    heat_kernel_k22, hubble_e_orthoplex, luminosity_distance_orthoplex, profile_likelihood_alpha,
    spectral_dimension_k22, w_of_z_table, w_orthoplex,
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

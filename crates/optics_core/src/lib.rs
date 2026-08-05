//! optics_core: Optical simulation with GRIN ray tracing, TCMT, phase retrieval, and more.
//!
//! This crate provides:
//! - RK4 integration of the ray equation dT/ds = (grad(n) - (T.grad(n))T)/n
//! - Complex refractive index support with Beer-Lambert attenuation
//! - Central-difference gradient estimation
//! - Temporal Coupled-Mode Theory (TCMT) for resonant cavities
//! - Kerr nonlinear response with optical bistability analysis
//! - Weighted Gerchberg-Saxton (WGS) phase retrieval for holographic beam shaping
//! - Zernike polynomial aberration correction
//!
//! # Literature
//! - CSC KTH (2011): "Ray Tracing in Gradient-Index Media"
//! - Leonhardt & Philbin (2006): Transformation optics
//! - Born & Wolf: Principles of Optics, Ch. 3
//! - Liu et al., Opt. Express 21(20), 23687 (2013): TCMT + Kerr nonlinearity
//! - Suh et al., IEEE JQE 40, 1511 (2004): TCMT foundations
//! - Di Leonardo et al., Opt. Express 15, 1913 (2007): WGS algorithm
//! - Manetsch et al., arXiv:2403.12021 (2025): Large-scale tweezer arrays

pub mod absorber_benchmark;
pub mod absorber_pareto;
#[cfg(any(feature = "gpu", feature = "vulkan", feature = "cubecl"))]
pub mod algebraic_lensing_gpu;
pub mod bessel;
pub mod entropy_trap;
pub mod fano_tcmt;
pub mod grin;
pub mod mie_cylinder;
pub mod mie_poles;
pub mod multi_resonator;
pub mod optics_algebra_bridge;
pub mod p2b_fit;
pub mod phase_retrieval;
pub mod sfwm;
pub mod sfwm_source;
pub mod tcmt;

pub use grin::{
    AbsorbingGrinMedium, GrinFiber, GrinMedium, HomogeneousMedium, Ray, RayState, RayTraceResult,
    Vec3, central_difference_gradient, rk4_step, rk4_step_absorbing, trace_ray,
    trace_ray_absorbing,
};

#[cfg(any(feature = "gpu", feature = "vulkan", feature = "cubecl"))]
pub use algebraic_lensing_gpu::GpuVec3;
#[cfg(feature = "cubecl")]
pub use algebraic_lensing_gpu::{
    AlgebraicLensingCubecl, AlgebraicLensingCubeclConfig, trace_rays_cpu_reference_cubecl,
};
#[cfg(feature = "gpu")]
pub use algebraic_lensing_gpu::{AlgebraicLensingGpu, AlgebraicLensingGpuConfig};
#[cfg(feature = "vulkan")]
pub use algebraic_lensing_gpu::{
    AlgebraicLensingVulkan, AlgebraicLensingVulkanConfig, trace_rays_cpu_reference,
};

pub use tcmt::{
    CavityState,
    CouplingRegime,
    HysteresisResult,
    HysteresisTrace,
    InputField,
    KerrCavity,
    // Normalized cubic solver (Liu et al. 2013, Eq. 5)
    NormalizedSteadyState,
    SteadyStateResult,
    // First-class engine module: errors and hysteresis detection
    TcmtError,
    TcmtSolver,
    // Thermal dynamics (Johnson et al. 2006, Carmon et al. 2004)
    ThermalCavity,
    ThermalCavityState,
    ThermalRegime,
    ThermalSteadyStateResult,
    ThermalTcmtSolver,
    TurningPoint,
    TurningPointBranch,
    bistability_bounds,
    denormalize_energy,
    find_turning_points,
    find_turning_points_physical,
    hysteresis_width,
    linear_transmission,
    normalize_parameters,
    solve_normalized_cubic,
    solve_normalized_cubic_batch,
    thermal_regime,
    trace_hysteresis_loop,
    validate_cavity,
};

pub use multi_resonator::{
    MultiResonatorState, MultiResonatorSystem, MultiResonatorTrace, ResonatorChannel,
};

pub use entropy_trap::{EntropyTrapResult, absorption_spectrum, detect_entropy_trap};

pub use phase_retrieval::{
    InitialPhase, TargetSpot, WgsConfig, WgsResult, gs_continuous, wgs_discrete, zernike_phase,
};

pub use optics_algebra_bridge::{
    JonesVector, MaxwellField, MuellerMatrix, OpticalElement, OpticalSystem, PolarizationState,
};

pub use sfwm::{
    SfwmDominanceResult,
    SfwmMaterialParams,
    SfwmRateResult,
    SubstrateSfwmParams,
    SubstrateSfwmResult,
    WavevectorMismatches,
    // Son & Chekhova (2026) full reproduction
    cascaded_amplitude_sq,
    cascaded_amplitude_sq_with_dk,
    // Original API
    coherence_length,
    direct_amplitude_sq,
    direct_amplitude_sq_with_dk,
    g2_sfwm_model,
    g2_spdc_model,
    maker_fringe_sweep,
    phase_matching_function,
    photon_number_sfwm,
    photon_number_spdc,
    polarization_dependence,
    rate_ratio,
    rate_ratio_with_dk,
    rate_sweep,
    rate_sweep_with_dk,
    sfwm_dominance_check,
    substrate_sfwm_contribution,
    thickness_sweep,
};

pub use sfwm_source::{
    SfwmSourceAmplitudes, SfwmSourceError, SfwmSourceParameters, SfwmSourceRates,
    SourceCoherenceAnchors, SourceMismatchAudit, SourceWavevectorMismatches, WavevectorMismatch,
    evaluate_source_case, source_amplitudes, source_rates,
};

pub use bessel::{
    bessel_j, bessel_j_prime, bessel_y, bessel_y_prime, hankel_1, hankel_1_prime, hankel_2,
    hankel_2_prime,
};

pub use fano_tcmt::{
    ChannelAmplitudes, ChannelConstraintResiduals, ChannelCrossSections, ChannelEvaluation,
    ChannelObservableResiduals, CrossSections, DimensionlessFanoChannel, FanoChannel,
    FanoChannelError, FanoChannelParameters, FanoDrudeParams, SourceCouplingParameters,
    drude_epsilon, evaluate_channel, evaluate_dimensionless_channel, evaluate_source_constraints,
    fano_cross_sections_normalized, fano_q, fano_reflection, multi_channel_cross_sections,
    normalized_fano_c_abs, normalized_fano_c_ext, normalized_fano_c_sct, scattering_coefficient,
    try_drude_epsilon, try_fano_reflection, try_multi_channel_evaluations,
    try_scattering_coefficient,
};

pub use mie_cylinder::{
    ChannelResult, ConcentricCylinder, CylinderLayer, CylindricalPolarization,
    InterfaceContinuityResidual, MaterialRole, MieError, MieObservableResiduals, MieResult,
    extract_fano_params, mie_mdm_sweep, mie_scattering, mie_sweep, ruan_fan_mdm_fig4,
    ruan_fan_mdm_fig5, scattering_coefficient_l, try_mie_scattering, try_mie_sweep,
    try_scattering_channel, try_update_metal_epsilon, update_metal_epsilon,
};

pub use mie_poles::{
    ComplexFrequency, ComplexPole, PoleError, PoleGeometry, RootAttempt, RootRectangle, RootSearch,
    StableRootCount, complex_drude_epsilon, outgoing_determinant, refine_root, root_seed_grid,
    search_roots, stable_root_count, uniform_metal_reflection,
};

pub use p2b_fit::{
    ComplexSample, FitError, FitParameters, FitStartResult, OnePoleFitReport, OnePoleParameters,
    TcmtFitReport, background_only, fit_one_pole, fit_tcmt, one_pole_error, one_pole_max_error,
    one_pole_value, tcmt_error, tcmt_jacobian_singular_values, tcmt_max_error, tcmt_reflection,
    tcmt_scattering,
};

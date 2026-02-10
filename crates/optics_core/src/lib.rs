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

pub mod grin;
pub mod optics_algebra_bridge;
pub mod phase_retrieval;
pub mod sfwm;
pub mod tcmt;

pub use grin::{
    central_difference_gradient, rk4_step, rk4_step_absorbing, trace_ray, trace_ray_absorbing,
    AbsorbingGrinMedium, GrinFiber, GrinMedium, HomogeneousMedium, Ray, RayState, RayTraceResult,
    Vec3,
};

pub use tcmt::{
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
};

pub use phase_retrieval::{
    gs_continuous, wgs_discrete, zernike_phase, InitialPhase, TargetSpot, WgsConfig, WgsResult,
};

pub use optics_algebra_bridge::{
    JonesVector, MaxwellField, MuellerMatrix, OpticalElement, OpticalSystem, PolarizationState,
};

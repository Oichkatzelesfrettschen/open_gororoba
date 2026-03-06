//! gr_core: General relativity computations in Rust.
//!
//! This crate provides:
//! - Generic `SpacetimeMetric` trait for defining spacetimes
//! - Numerical Christoffel symbols, Riemann tensor, Ricci tensor, curvature invariants
//! - Schwarzschild spacetime with exact closed-form Christoffel symbols
//! - Kerr spacetime with exact Christoffels, geodesic integration, and shadow computation
//! - Coordinate transformations (BL, MKS, Cartesian, Kerr-Schild)
//! - Hawking radiation and black hole thermodynamics
//! - Penrose process and Blandford-Znajek energy extraction
//! - Novikov-Thorne thin disk accretion model
//! - Astrophysical constants and unit conversions
//!
//! # Module organization
//!
//! - `metric` -- Generic spacetime trait and numerical curvature computation
//! - `schwarzschild` -- Schwarzschild black hole (exact Christoffels, ISCO, potentials)
//! - `kerr` -- Kerr black hole (exact Christoffels, geodesics, shadow, ISCO, ergosphere)
//! - `kerr_newman` -- Kerr-Newman (rotating charged BH): horizons, EM field, ISCO
//! - `kerr_de_sitter` -- Kerr-de Sitter (rotating BH + Lambda): triple horizon structure
//! - `coordinates` -- BL, MKS, Cartesian, and Kerr-Schild coordinate transforms
//! - `hawking` -- Hawking radiation, BH thermodynamics, entropy, evaporation
//! - `penrose` -- Penrose process, Blandford-Znajek, superradiance
//! - `novikov_thorne` -- Thin disk accretion: efficiency, temperature, flux
//! - `synchrotron` -- Synchrotron radiation from relativistic electrons
//! - `doppler` -- Relativistic Doppler effect, beaming, aberration
//! - `absorption` -- SSA, free-free, Compton absorption + radiative transfer
//! - `scattering` -- Thomson, Rayleigh, Mie scattering + albedo, asymmetry
//! - `gravitational_waves` -- GW strain, chirp mass, inspiral waveforms, QNM
//! - `spectral_bands` -- Observational bands (EHT, ALMA, V-band, Chandra), filters, magnitudes
//! - `null_constraint` -- Null geodesic constraint preservation and renormalization
//! - `energy_conserving` -- Energy-conserving geodesic integration (RK4 + Hamiltonian correction)
//! - `constants` -- Astrophysical constants (CGS, natural units, conversions)
//! - `ppn_constraints` -- PPN parameters, experimental bounds, falsification gates (Rodal 2025)
//! - `scalar_tensor` -- Brans-Dicke field equations, junction conditions, frame transformations
//! - `area_quantization` -- LQG area spectrum, Barbero-Immirzi parameter (Barbero 1995, Immirzi 1997)
//! - `photon_graviton_tcmt` -- Photon-graviton TCMT: maps worldline QFT amplitude to resonant scattering (Ruan-Fan 2009, Ahmadiniaz et al. 2026, Maksimov 2025)
//! - `quantum_inequalities` -- Ford-Roman QI bounds, Pfenning-Ford warp analysis, energy conditions
//!
//! # Literature
//! - Bardeen (1973): Black Holes, Les Houches
//! - Chandrasekhar (1983): The Mathematical Theory of Black Holes
//! - Misner, Thorne, Wheeler (1973): Gravitation
//! - Teo (2003): Gen. Relativ. Gravit. 35, 1909
//! - Hawking (1974): Nature 248, 30
//! - Penrose (1969): Gravitational Collapse: The Role of GR
//! - Novikov & Thorne (1973): Black Holes (Les Astres Occlus)
//! - Page & Thorne (1974): ApJ 191, 499
//! - Rybicki & Lightman (1979): Radiative Processes in Astrophysics
//! - Bohren & Huffman (1983): Absorption and Scattering of Light by Small Particles
//! - Klein & Nishina (1929): Z. Phys. 52, 853
//! - Begelman, Blandford, Rees (1984): Rev. Mod. Phys. 56, 255
//! - Peters & Mathews (1963): Phys. Rev. 131, 435
//! - Blanchet (2014): Living Reviews in Relativity
//! - Newman et al. (1965): J. Math. Phys. 6, 918
//! - Carter (1968): Phys. Rev. 174, 1559
//! - Carter (1973): Black hole equilibrium states with Lambda
//! - Griffiths & Podolsky (2009): Exact Space-Times in Einstein's GR
//! - Chan et al. (2013): GRay -- GPU ray tracing in relativistic spacetimes
//! - Rodal (2025): arXiv:2507.09724v2 -- Metamaterial gravitational coupling
//! - Will (2014): Living Rev. Relativity 17, 4 -- PPN formalism
//! - Brans & Dicke (1961): Phys. Rev. 124, 925 -- Scalar-tensor gravity
//! - Barbero (1995): Phys. Rev. D 51, 5507 -- Real Ashtekar variables
//! - Immirzi (1997): Class. Quantum Grav. 14, L177 -- Quantum gravity and Regge calculus
//! - Ford & Roman (1996): Phys. Rev. D 53, 5496 -- Quantum inequalities
//! - Pfenning & Ford (1997): Class. Quantum Grav. 14, 1743 -- Warp drive energy

pub mod absorption;
pub mod acoustic_metric;
pub mod adm;
pub mod adm_algebra_bridge;
pub mod area_quantization;
pub mod constants;
pub mod coordinates;
pub mod cosmology_algebra_bridge;
pub mod doppler;
pub mod energy_conserving;
pub mod cd_ladder_force;
pub mod chingon_frame_dragging;
pub mod forces;
pub mod gravitational_waves;
pub mod lyapunov;
pub mod hawking;
pub mod kerr;
pub mod kerr_de_sitter;
pub mod kerr_newman;
pub mod lattice_hawking;
pub mod metric;
pub mod nbody_integration;
pub mod novikov_thorne;
pub mod null_constraint;
pub mod penrose;
pub mod photon_graviton;
pub mod photon_graviton_tcmt;
pub mod ppn_constraints;
pub mod quantum_inequalities;
pub mod scalar_tensor;
pub mod scattering;
pub mod schwarzschild;
pub mod sedenion_geodesic;
pub mod spacetime_algebra;
pub mod spatial;
pub mod spectral_bands;
pub mod synchrotron;
pub mod warp_metric;

// Re-export primary types from each module
pub use kerr::{
    GeodesicResult, GeodesicState, Kerr, ShadowResult, geodesic_rhs, impact_parameters,
    kerr_metric_quantities, photon_orbit_radius, shadow_boundary, shadow_ray_traced,
    trace_null_geodesic,
};

pub use metric::{
    ChristoffelComponents, CurvatureResult, MetricComponents, Minkowski, SpacetimeMetric,
    full_curvature,
};

pub use schwarzschild::Schwarzschild;

pub use energy_conserving::{
    ConservedQuantities, FullGeodesicState, apply_constraint_correction, compute_angular_momentum,
    compute_carter_constant, compute_energy, energy_conserving_step, extract_conserved_quantities,
    geodesic_acceleration, integrate_energy_conserving, relative_angular_momentum_drift,
    relative_energy_drift, rk4_geodesic_step,
};

pub use null_constraint::{
    ConstraintStats, adaptive_tolerance, constraint_drift_bound, global_drift_bound, is_null,
    is_timelike, mass_shell_constraint, needs_renormalization, null_constraint,
    renormalize_massive, renormalize_null, renormalize_null_diagonal, renormalize_null_kerr,
};

pub use cosmology_algebra_bridge::{
    ConservationLaw, CosmologicalParameters, DarkEnergyModel, EnergyMomentumTensor, FLRWMetric,
    FriedmannSolver, Redshift,
};

pub use ppn_constraints::{
    ALPHA1_BOUND, ALPHA2_BOUND, ALPHA3_BOUND, CASSINI_GAMMA_DEVIATION_BOUND,
    CASSINI_OMEGA_BD_LOWER, GPB_GEODETIC_PRECISION, LIGO_O3_GRAVITON_MASS_EV,
    MICROSCOPE_EOTVOS_BOUND, NANOGRAV_GRAVITON_MASS_EV, PPNConstraintReport, XI_BOUND,
    check_all_ppn_constraints, dipole_radiation_power, graviton_compton_wavelength,
    massive_graviton_velocity_ratio, nordtvedt_parameter_bd, perihelion_precession_per_orbit,
    ppn_gamma_bd, satisfies_alpha1_bd, satisfies_alpha2_bd, satisfies_alpha3_bd,
    satisfies_cassini_gamma, satisfies_gpb_geodetic, satisfies_microscope_wep_bd, satisfies_xi_bd,
};

pub use scalar_tensor::{BransDickeParams, JunctionCondition, fifth_force_alpha};

pub use area_quantization::{
    GAMMA_BG, GAMMA_NZJ, area_gap_ratio, area_quantum, area_spectrum, entropy_correction_factor,
    entropy_lqg, immirzi_parameter_bg, immirzi_parameter_nzj, minimum_area,
};

pub use quantum_inequalities::{
    M_OBSERVABLE_UNIVERSE_G, NegativeEnergyBudget, alcubierre_energy_density, check_qi_satisfied,
    dominant_energy_condition, lorentzian_sampling, null_energy_condition,
    pfenning_ford_energy_ratio, quantum_inequality_bound_4d, strong_energy_condition,
    warp_bubble_total_energy, warp_wall_thickness_bound, weak_energy_condition,
};

pub use adm::{AdmConstraints, AdmDecomposition, ExtrinsicCurvatureData, decompose_metric};

pub use warp_metric::{NacelleWarpBubble, NacelleWarpParams};

pub use nbody_integration::{BodyState, NBodySystem};

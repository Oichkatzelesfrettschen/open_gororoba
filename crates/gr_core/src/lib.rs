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
//! - `gravitational_waves` -- GW strain, chirp mass, inspiral waveforms, QNM
//! - `null_constraint` -- Null geodesic constraint preservation and renormalization
//! - `energy_conserving` -- Energy-conserving geodesic integration (RK4 + Hamiltonian correction)
//! - `constants` -- Astrophysical constants (CGS, natural units, conversions)
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
//! - Begelman, Blandford, Rees (1984): Rev. Mod. Phys. 56, 255
//! - Peters & Mathews (1963): Phys. Rev. 131, 435
//! - Blanchet (2014): Living Reviews in Relativity
//! - Newman et al. (1965): J. Math. Phys. 6, 918
//! - Carter (1968): Phys. Rev. 174, 1559
//! - Carter (1973): Black hole equilibrium states with Lambda
//! - Griffiths & Podolsky (2009): Exact Space-Times in Einstein's GR
//! - Chan et al. (2013): GRay -- GPU ray tracing in relativistic spacetimes

pub mod absorption;
pub mod constants;
pub mod coordinates;
pub mod doppler;
pub mod energy_conserving;
pub mod gravitational_waves;
pub mod hawking;
pub mod kerr;
pub mod kerr_de_sitter;
pub mod kerr_newman;
pub mod metric;
pub mod null_constraint;
pub mod novikov_thorne;
pub mod penrose;
pub mod schwarzschild;
pub mod synchrotron;

// Re-export primary types from each module
pub use kerr::{
    kerr_metric_quantities, photon_orbit_radius, impact_parameters,
    shadow_boundary, GeodesicState, geodesic_rhs, trace_null_geodesic,
    GeodesicResult, ShadowResult, shadow_ray_traced, Kerr,
};

pub use metric::{
    SpacetimeMetric, MetricComponents, ChristoffelComponents,
    CurvatureResult, full_curvature,
};

pub use schwarzschild::Schwarzschild;

pub use energy_conserving::{
    ConservedQuantities, FullGeodesicState,
    compute_energy, compute_angular_momentum, compute_carter_constant,
    extract_conserved_quantities, geodesic_acceleration,
    rk4_geodesic_step, apply_constraint_correction,
    energy_conserving_step, integrate_energy_conserving,
    relative_energy_drift, relative_angular_momentum_drift,
};

pub use null_constraint::{
    null_constraint, is_null, mass_shell_constraint, is_timelike,
    renormalize_null, renormalize_null_diagonal, renormalize_null_kerr,
    renormalize_massive, constraint_drift_bound, global_drift_bound,
    adaptive_tolerance, needs_renormalization, ConstraintStats,
};

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
//! - `coordinates` -- BL, MKS, Cartesian, and Kerr-Schild coordinate transforms
//! - `hawking` -- Hawking radiation, BH thermodynamics, entropy, evaporation
//! - `penrose` -- Penrose process, Blandford-Znajek, superradiance
//! - `novikov_thorne` -- Thin disk accretion: efficiency, temperature, flux
//! - `synchrotron` -- Synchrotron radiation from relativistic electrons
//! - `doppler` -- Relativistic Doppler effect, beaming, aberration
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

pub mod constants;
pub mod coordinates;
pub mod doppler;
pub mod hawking;
pub mod kerr;
pub mod metric;
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

//! gr_core: General relativity computations in Rust.
//!
//! This crate provides:
//! - Generic `SpacetimeMetric` trait for defining spacetimes
//! - Numerical Christoffel symbols, Riemann tensor, Ricci tensor, curvature invariants
//! - Schwarzschild spacetime with exact closed-form Christoffel symbols
//! - Kerr spacetime with null geodesic integration and shadow computation
//! - Astrophysical constants and unit conversions
//!
//! # Module organization
//!
//! - `metric` -- Generic spacetime trait and numerical curvature computation
//! - `schwarzschild` -- Schwarzschild black hole (exact Christoffels, ISCO, potentials)
//! - `kerr` -- Kerr black hole geodesics and shadow (Bardeen curve, ray-tracing)
//! - `constants` -- Astrophysical constants (CGS, natural units, conversions)
//!
//! # Literature
//! - Bardeen (1973): Black Holes, Les Houches
//! - Chandrasekhar (1983): The Mathematical Theory of Black Holes
//! - Misner, Thorne, Wheeler (1973): Gravitation
//! - Teo (2003): Gen. Relativ. Gravit. 35, 1909

pub mod constants;
pub mod kerr;
pub mod metric;
pub mod schwarzschild;

// Re-export primary types from each module
pub use kerr::{
    kerr_metric_quantities, photon_orbit_radius, impact_parameters,
    shadow_boundary, GeodesicState, geodesic_rhs, trace_null_geodesic,
    GeodesicResult,
};

pub use metric::{
    SpacetimeMetric, MetricComponents, ChristoffelComponents,
    CurvatureResult, full_curvature,
};

pub use schwarzschild::Schwarzschild;

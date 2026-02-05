//! gr_core: General relativity - Kerr black hole geodesics and shadows.
//!
//! This crate provides:
//! - Kerr metric in Boyer-Lindquist coordinates
//! - Null geodesic integration in Mino time
//! - Black hole shadow computation (Bardeen curve)
//! - Photon orbit calculations
//!
//! # Literature
//! - Bardeen (1973): Black Holes, Les Houches
//! - Chandrasekhar (1983): The Mathematical Theory of Black Holes
//! - Teo (2003): Gen. Relativ. Gravit. 35, 1909

pub mod kerr;

pub use kerr::{
    kerr_metric_quantities, photon_orbit_radius, impact_parameters,
    shadow_boundary, GeodesicState, geodesic_rhs, trace_null_geodesic,
    GeodesicResult,
};

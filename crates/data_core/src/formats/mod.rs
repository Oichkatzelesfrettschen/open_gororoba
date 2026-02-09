//! File format parsers for specialized data formats.
//!
//! These handle formats beyond standard CSV that appear in geophysical
//! and astronomical data archives.

pub mod gfc;
pub mod mcmc_chain;
pub mod pantheon_dat;
pub mod tap;

pub use gfc::{
    actual_max_degree, parse_gfc, validate_gfc_degrees, GravityCoefficient, GravityField,
};

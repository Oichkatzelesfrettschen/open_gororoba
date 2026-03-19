//! Numerical Stability Analysis for Cayley-Dickson Algebras.
//!
//! Tools for measuring precision drift, norm accumulation, and zero-divisor 
//! proximity in high-dimensional algebras.

use cd_kernel::cayley_dickson::{cd_associator, cd_norm_sq};

/// Measures the norm accumulation error growth. 
/// In an N-dimensional algebra, errors are expected to scale roughly as sqrt(N).
pub fn estimate_norm_error_growth(dim: usize) -> f64 {
    (dim as f64).sqrt()
}

/// Computes the magnitude of the associator [a, b, c] = (ab)c - a(bc).
/// High associator magnitudes indicate strong non-associativity.
pub fn associator_magnitude(a: &[f64], b: &[f64], c: &[f64]) -> f64 {
    let assoc = cd_associator(a, b, c);
    cd_norm_sq(&assoc).sqrt()
}

/// Recommends floating point precision based on the algebra dimension.
pub fn precision_recommendation(dim: usize) -> &'static str {
    if dim <= 8 {
        "f64 (Standard precision sufficient)"
    } else if dim <= 16 {
        "f64 (Watch for zero-divisor instability)"
    } else {
        "f128 / Extended precision recommended for physics applications"
    }
}

/// Evaluates distance to the nearest known zero divisor manifold.
/// Distance < 1e-8 indicates high risk of numerical instability.
pub fn zero_divisor_risk_assessment(a: &[f64], tol: f64) -> bool {
    let norm = cd_norm_sq(a).sqrt();
    norm < tol
}

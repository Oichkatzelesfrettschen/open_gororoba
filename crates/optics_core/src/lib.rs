//! optics_core: Gradient-Index (GRIN) ray tracing with Beer-Lambert absorption.
//!
//! This crate provides:
//! - RK4 integration of the ray equation dT/ds = (grad(n) - (T.grad(n))T)/n
//! - Complex refractive index support with Beer-Lambert attenuation
//! - Central-difference gradient estimation
//!
//! # Literature
//! - CSC KTH (2011): "Ray Tracing in Gradient-Index Media"
//! - Leonhardt & Philbin (2006): Transformation optics
//! - Born & Wolf: Principles of Optics, Ch. 3

pub mod grin;

pub use grin::{
    Ray, RayState, GrinMedium, AbsorbingGrinMedium,
    rk4_step, rk4_step_absorbing, central_difference_gradient,
    trace_ray, trace_ray_absorbing, RayTraceResult,
};

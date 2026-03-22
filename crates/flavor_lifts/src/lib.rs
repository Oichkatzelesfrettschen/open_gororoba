//! Flavor lift layer: maps algebraic/incidence data into flavor operator space.
//!
//! This crate isolates the **lift** -- the map from the 42D assessor space
//! (or V_6 complement) into 3x3 symmetric mass matrices. The lift is the
//! missing architectural piece between the algebraic family geometry
//! (interleaved S_3-sedenion framework) and the physical PMNS/CKM matrices.
//!
//! The key distinction: the algebraic family geometry is one thing,
//! but the map from algebraic structure to flavor operators is another.
//! `TensorElementLift` is the first tested 42->6 map that actually succeeds.
//!
//! # Modules
//!
//! - `lift`: The `FlavorLift` trait and all implementations
//! - `angles`: PMNS angle extraction from unitary matrices
//! - `optimizer`: Constrained direction computation and Gauss-Newton 2D optimizer
//! - `cp`: CP violation pipeline scaffolding (CP-A and CP-B)

pub mod angles;
pub mod cp;
pub mod lift;
pub mod optimizer;

pub use angles::extract_pmns_angles;
pub use lift::{
    AssessorToFlavorMap, DirectOffDiagonalLift, FlavorLift, PsiEquivariantLift, TensorElementLift,
    apply_v6_perturbation,
};
pub use optimizer::{
    compute_constrained_atmospheric_direction, compute_constrained_solar_direction,
    gauss_newton_2d,
};

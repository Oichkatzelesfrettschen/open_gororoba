//! Flavor lift layer: maps algebraic/incidence data into flavor operator space.
//!
//! # Purpose
//!
//! This crate isolates the **lift** -- the map from the 42D assessor space
//! (or its 6D V_6 complement) into 3x3 symmetric mass matrices.  The lift is
//! the missing architectural piece between the algebraic family geometry
//! (interleaved S_3-sedenion framework, Gresnigt/Gourlay 2019/2026) and the
//! physical PMNS/CKM matrices.
//!
//! The key distinction the project has made explicit:
//!
//! > The algebraic family geometry is one thing, but the map from algebraic
//! > structure to flavor operators is another.
//!
//! [`TensorElementLift`] is the first tested 42->6 map that actually succeeds
//! (C-1478).  All prior 42->3 lifts hit the rank-2 Jacobian lock (C-1476).
//!
//! # Architecture
//!
//! ```text
//! 42D assessor space
//!      |
//!      v
//!  V_6 basis (6 x 42, from SVD of X*(I - P_BC))
//!      |
//!      v                         +----> CP-A: phase-only complexification
//!  FlavorLift trait              |
//!   .lift(v, m)  ---------> 3x3 mass matrix --> eigendecompose --> PMNS/CKM
//!                                |
//!                                +----> CP-B: cross-sector Gram/rephasing
//! ```
//!
//! # Why a separate crate
//!
//! `neutrino_sector.rs` grew to 10,000+ lines because the lift logic, angle
//! extraction, and optimization were embedded in the same file as the selector
//! scan infrastructure.  This crate extracts the lift layer so that:
//!
//! 1. New CP machinery (A4) is built here, not in the god-file.
//! 2. The FlavorLift trait is testable in isolation from the scan.
//! 3. The crate boundary enforces the lift/scan separation.
//!
//! # Modules
//!
//! | Module | Contents | Key types |
//! |--------|----------|-----------|
//! | [`lift`] | FlavorLift trait + 4 implementations | [`TensorElementLift`], [`DirectOffDiagonalLift`] |
//! | [`angles`] | PMNS angle extraction, PDG reference data | [`extract_pmns_angles`], [`angles::Pdg2024`] |
//! | [`optimizer`] | Constrained V_6 directions, Gauss-Newton 2D | [`gauss_newton_2d`] |
//! | [`cp`] | CP pipeline scaffolding (CP-A / CP-B) | [`cp::CpResult`], [`cp::CpPipeline`] |
//! | [`structurable_bridge`] | Value-level middle-tier handoff and snapshots | [`apply_structurable_bridge`], [`sample_structurable_bridge`] |
//!
//! # Relationship to other crates
//!
//! - Depends on [`cd_kernel`] for `gourlay_psi` (used by [`PsiEquivariantLift`]).
//! - Depends on `gororoba_structurable` for value-level ternary bridge inputs.
//! - Depends on `faer` for 3x3 matrix operations and `nalgebra` for V_6 basis.
//! - Used by `algebra_experimental::neutrino_sector` for PMNS construction.
//! - Does NOT depend on the selector scan infrastructure (that stays in
//!   `neutrino_sector.rs`).

pub mod angles;
pub mod basis;
pub mod cp;
pub mod lift;
pub mod optimizer;
pub mod structurable_bridge;

pub use angles::{Pdg2024, extract_pmns_angles};
pub use basis::{extract_v6_basis, extract_vk_basis};
pub use lift::{
    AssessorToFlavorMap, DirectOffDiagonalLift, FlavorLift, PsiEquivariantLift, TensorElementLift,
    apply_v6_perturbation,
};
pub use optimizer::{
    compute_constrained_atmospheric_direction, compute_constrained_solar_direction, gauss_newton_2d,
};
pub use structurable_bridge::{
    StructurableBridgeSnapshot, apply_structurable_bridge, sample_structurable_bridge,
};

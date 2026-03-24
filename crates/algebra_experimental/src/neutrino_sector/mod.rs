//! Neutrino Sector: PMNS Mixing Matrix from Sedenion Signed Friction
//!
//! Derives the PMNS (Pontecorvo-Maki-Nakagawa-Sakata) mixing matrix from the
//! same signed-friction framework used for the CKM matrix. The PMNS matrix
//! relates neutrino mass eigenstates to flavor eigenstates:
//!
//!   U_PMNS = U_charged^dagger * U_neutrino
//!
//! where U_charged diagonalizes the charged lepton mass matrix and
//! U_neutrino diagonalizes the neutrino mass matrix.
//!
//! # Key differences from quark sector
//!
//! - PMNS angles are LARGE: theta_23 ~ 49 deg, theta_12 ~ 33 deg, theta_13 ~ 8.5 deg
//! - CKM angles are SMALL: theta_12 ~ 13 deg, theta_23 ~ 2.4 deg, theta_13 ~ 0.2 deg
//! - This asymmetry may arise from using different selector pairs in the
//!   sedenion algebra for the lepton vs quark sectors.
//!
//! # Module structure (Phase C split)
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`jk_action`] | `GeneratorType`, `classify_generator`, `apply_jk_full_16d` |
//! | [`pmns`] | `PmnsResult`, PMNS construction functions, chi2/pulls |
//! | [`basis`] | `extract_v6_basis`, `extract_vk_basis` |
//! | [`cp_scan`] | `CpScanResult`, `CpScanContext`, scan pipeline |
//! | [`hermitian`] | `hermitian_3x3_eig`, Cardano eigensolver |

pub mod basis;
pub mod branch_transport;
pub mod cp_scan;
pub mod hermitian;
pub mod jk_action;
pub mod pmns;

// ---------------------------------------------------------------------------
// Re-exports: preserve the flat public API surface of the original file
// ---------------------------------------------------------------------------

pub use jk_action::{GeneratorType, apply_jk_full_16d, classify_generator};

pub use pmns::{
    PmnsResult, chi_squared_pmns, compute_pmns, construct_casimir_baseline,
    construct_pmns_matrices, construct_pmns_matrices_offdiag, construct_pmns_matrices_two_param,
    construct_pmns_matrices_v6_modulated, extract_cp_phase, jarlskog_from_real_pmns, pmns_pulls,
};

pub use basis::{extract_v6_basis, extract_vk_basis};
pub use branch_transport::{
    BranchMapReport, BranchMapRow, BranchWallReport, BranchWallRow, GradientFrame, LoopReport,
    LoopStep, LoopSummary, PathScanReport, PathScanRow, V6ProbeArtifacts, V6ProbeSummary,
    alignment, compute_branch_map, compute_branch_walls, compute_gradient_frame,
    compute_loop_transport, compute_path_scan, default_alpha_ch_values, default_alpha_nu_values,
    default_probe_artifacts, fixed_alpha_ch_scan_points, fixed_alpha_nu_scan_points, perm_label,
    stable_branch_loop_points, summarize_probe_artifacts, wall_crossing_loop_points,
};

pub use cp_scan::{
    CpNelderMeadCost, CpScanBuffers, CpScanContext, CpScanResult, evaluate_cp_scan_point,
    evaluate_cp_scan_point_cardano, extract_delta_cp_invariant, refine_cp_nelder_mead,
    refine_cp_nelder_mead_r,
};

pub use hermitian::{
    C2, cconj, cmul, hermitian_3x3_eig, hermitian_3x3_eig_hybrid, pmns_from_hermitian_pair,
};

// extract_pmns_angles and Pdg2024 are re-exported from flavor_lifts::angles.
pub use flavor_lifts::{Pdg2024, extract_pmns_angles};

// Lift layer: re-exported from flavor_lifts (preserved from original).
pub use flavor_lifts::{
    AssessorToFlavorMap, DirectOffDiagonalLift, FlavorLift, PsiEquivariantLift, TensorElementLift,
    apply_v6_perturbation, compute_constrained_atmospheric_direction,
    compute_constrained_solar_direction, gauss_newton_2d,
};

// Test-only re-exports so grandchild test submodules can reach these via
// `use super::super::*;` without needing the full sub-module path.
#[cfg(test)]
pub(crate) use basis::extract_vk_basis_nalgebra;
#[cfg(test)]
pub(crate) use pmns::assemble_lepton_baseline;

#[cfg(test)]
mod tests;

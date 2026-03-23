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
pub mod cp_scan;
pub mod hermitian;
pub mod jk_action;
pub mod pmns;

// ---------------------------------------------------------------------------
// Re-exports: preserve the flat public API surface of the original file
// ---------------------------------------------------------------------------

pub use jk_action::{GeneratorType, classify_generator, apply_jk_full_16d};

pub use pmns::{
    PmnsResult,
    construct_pmns_matrices,
    construct_pmns_matrices_offdiag,
    extract_cp_phase,
    jarlskog_from_real_pmns,
    chi_squared_pmns,
    pmns_pulls,
    compute_pmns,
    construct_casimir_baseline,
    construct_pmns_matrices_two_param,
    construct_pmns_matrices_v6_modulated,
};

pub use basis::{extract_v6_basis, extract_vk_basis};

pub use cp_scan::{
    CpScanResult,
    extract_delta_cp_invariant,
    CpScanContext,
    CpScanBuffers,
    evaluate_cp_scan_point,
    evaluate_cp_scan_point_cardano,
    CpNelderMeadCost,
    refine_cp_nelder_mead,
    refine_cp_nelder_mead_r,
};

pub use hermitian::{hermitian_3x3_eig, hermitian_3x3_eig_hybrid, pmns_from_hermitian_pair};
pub use hermitian::{C2, cmul, cconj};

// extract_pmns_angles and Pdg2024 are re-exported from flavor_lifts::angles.
pub use flavor_lifts::{extract_pmns_angles, Pdg2024};

// Lift layer: re-exported from flavor_lifts (preserved from original).
pub use flavor_lifts::{
    AssessorToFlavorMap, DirectOffDiagonalLift, FlavorLift, PsiEquivariantLift,
    TensorElementLift, apply_v6_perturbation, compute_constrained_atmospheric_direction,
    compute_constrained_solar_direction, gauss_newton_2d,
};

#[cfg(test)]
mod tests;

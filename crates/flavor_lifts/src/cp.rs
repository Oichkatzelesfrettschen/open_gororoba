//! CP violation pipeline scaffolding.
//!
//! Two materially different CP constructions exist. They are NOT "one result
//! with different conventions" -- they are different pipelines producing
//! different J_CP and delta_CP values.
//!
//! **CP-A: Phase-only complexification in J_k basis**
//!   M_ij -> |M_ij| * exp(i * alpha_CP * phi_ij)
//!   Preserves magnitudes. Current result: |J_CP| ~ 8.5e-3, delta ~ 165 deg.
//!   Angles within 1.5% of PDG. (C-1494)
//!
//! **CP-B: Cross-sector Gram / rephasing pipeline**
//!   Computes Gram matrix between charged and neutrino friction profiles,
//!   extracts quartet phases, builds complex mass matrix from Gram structure.
//!   Current result: |J_CP| ~ 3.28e-2, delta ~ -90 deg. (C-1497)
//!
//! **Closure criterion**: CP is "closed" ONLY when one chosen pipeline
//! simultaneously:
//!   (a) preserves the angle fit (theta_12/13/23 within 5% of baseline),
//!   (b) yields nonzero J_CP (> 1e-4),
//!   (c) produces a rephasing-invariant PMNS delta_CP after fixing the
//!       flavor/eigenstate convention.
//!
//! **Critical conceptual note**: The quartet phase extracted from the
//! cross-sector Gram matrix is NOT the physical PMNS delta_CP. The rephased
//! PMNS delta_CP depends on both phases and moduli.

// CP-A and CP-B pipeline implementations will be added in A4a/A4b.
// This module currently provides the shared types and closure criterion.

/// Result from a CP pipeline evaluation.
#[derive(Debug, Clone)]
pub struct CpResult {
    /// Jarlskog invariant magnitude.
    pub j_cp: f64,
    /// Physical PMNS delta_CP in degrees (rephasing-invariant).
    pub delta_cp_deg: f64,
    /// PMNS mixing angles (theta_12, theta_13, theta_23) in degrees.
    pub angles_deg: (f64, f64, f64),
    /// Which pipeline produced this result.
    pub pipeline: CpPipeline,
}

/// Which CP pipeline was used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpPipeline {
    /// Phase-only complexification in J_k basis (C-1494).
    PhaseOnly,
    /// Cross-sector Gram / rephasing pipeline (C-1497).
    GramRephasing,
}

/// Check if a CP result meets the closure criterion.
///
/// Returns `true` if all three conditions are satisfied:
/// (a) angles within `angle_tol_pct` of baseline,
/// (b) |J_CP| > `j_cp_min`,
/// (c) delta_CP is finite (not NaN/Inf).
pub fn meets_closure_criterion(
    result: &CpResult,
    baseline_angles: (f64, f64, f64),
    angle_tol_pct: f64,
    j_cp_min: f64,
) -> bool {
    let (t12, t13, t23) = result.angles_deg;
    let (b12, b13, b23) = baseline_angles;

    let angle_ok = ((t12 - b12).abs() / b12 * 100.0) < angle_tol_pct
        && ((t13 - b13).abs() / b13 * 100.0) < angle_tol_pct
        && ((t23 - b23).abs() / b23 * 100.0) < angle_tol_pct;

    let j_ok = result.j_cp.abs() > j_cp_min;
    let delta_ok = result.delta_cp_deg.is_finite();

    angle_ok && j_ok && delta_ok
}

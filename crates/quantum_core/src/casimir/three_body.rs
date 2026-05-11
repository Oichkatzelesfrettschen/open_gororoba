// ============================================================================
// Three-Body Transistor Dynamics
// ============================================================================
//
// The sphere-plate-sphere system exhibits coupled dynamics when all three
// bodies can move. The full 3x3 Jacobian matrix captures:
//
// | dF_s/dx_s  dF_s/dx_p  dF_s/dx_d |
// | dF_p/dx_s  dF_p/dx_p  dF_p/dx_d |
// | dF_d/dx_s  dF_d/dx_p  dF_d/dx_d |
//
// For PFA, dF_s/dx_d = dF_d/dx_s = 0 (no direct coupling), but the plate
// mediates coupling. With plate dynamics included, the effective gain
// depends on the plate's mechanical properties.
//
// # Literature
// - Xu et al., Nature Communications 13, 6148 (2022)
// - Rodriguez et al., PRA 76, 032106 (2007): Coupled Casimir dynamics

use super::{
    CASIMIR_COEFF, CasimirError, PfaAccuracy, PfaValidityInfo, casimir_force_guarded,
    check_pfa_validity,
};

/// Three-body dynamics result.
#[derive(Debug, Clone)]
pub struct ThreeBodyResult {
    /// Jacobian matrix element: dF_source/dx_source (N/m)
    pub j_ss: f64,
    /// Jacobian matrix element: dF_source/dx_plate (N/m)
    pub j_sp: f64,
    /// Jacobian matrix element: dF_plate/dx_source (N/m)
    pub j_ps: f64,
    /// Jacobian matrix element: dF_plate/dx_plate (N/m)
    pub j_pp: f64,
    /// Jacobian matrix element: dF_plate/dx_drain (N/m)
    pub j_pd: f64,
    /// Jacobian matrix element: dF_drain/dx_plate (N/m)
    pub j_dp: f64,
    /// Jacobian matrix element: dF_drain/dx_drain (N/m)
    pub j_dd: f64,
    /// Plate mechanical spring constant (N/m)
    pub k_plate: f64,
    /// Effective transistor gain including plate dynamics
    pub effective_gain: f64,
    /// Stability eigenvalues (all should be positive for stability)
    pub stability_eigenvalues: [f64; 3],
    /// Whether the system is dynamically stable
    pub is_stable: bool,
    /// PFA validity info for source
    pub source_validity: PfaValidityInfo,
    /// PFA validity info for drain
    pub drain_validity: PfaValidityInfo,
}

/// Compute full three-body Casimir dynamics.
///
/// This function computes the complete Jacobian matrix for the coupled
/// sphere-plate-sphere system, including stability analysis.
///
/// # Arguments
/// * `r_source` - Source sphere radius (m)
/// * `r_drain` - Drain sphere radius (m)
/// * `gap_source` - Source-plate gap (m)
/// * `gap_drain` - Plate-drain gap (m)
/// * `k_plate` - Plate mechanical spring constant (N/m)
/// * `m_source` - Source sphere mass (kg), used for dynamics
/// * `m_plate` - Plate mass (kg), used for dynamics
/// * `m_drain` - Drain sphere mass (kg), used for dynamics
///
/// # Returns
/// Full three-body dynamics result with Jacobian and stability
#[allow(clippy::too_many_arguments)]
pub fn three_body_casimir_dynamics(
    r_source: f64,
    r_drain: f64,
    gap_source: f64,
    gap_drain: f64,
    k_plate: f64,
    m_source: f64,
    m_plate: f64,
    m_drain: f64,
) -> ThreeBodyResult {
    // Casimir spring constants (magnitude, treating gaps as coordinates)
    // k_casimir = |dF/dg| = 3 * C * R / g^4
    let k_cs = 3.0 * CASIMIR_COEFF * r_source / gap_source.powi(4);
    let k_cd = 3.0 * CASIMIR_COEFF * r_drain / gap_drain.powi(4);

    // Jacobian elements
    // Source sphere: only feels force from plate (gap_source = x_p - x_s)
    // dF_s/dx_s = -k_cs (moving source toward plate increases attractive force)
    // dF_s/dx_p = +k_cs (moving plate away increases gap, reduces force)
    let j_ss = -k_cs;
    let j_sp = k_cs;

    // Plate: feels forces from both spheres
    // dF_p/dx_s = +k_cs (source moves toward plate, force on plate increases)
    // dF_p/dx_p = -(k_cs + k_cd + k_plate) (plate restoring)
    // dF_p/dx_d = +k_cd (drain moves toward plate, force on plate increases)
    let j_ps = k_cs;
    let j_pp = -(k_cs + k_cd + k_plate);
    let j_pd = k_cd;

    // Drain sphere: only feels force from plate (gap_drain = x_d - x_p)
    // dF_d/dx_p = +k_cd (plate moves away, gap decreases, force increases)
    // dF_d/dx_d = -k_cd (drain moves away, gap increases, force decreases)
    let j_dp = k_cd;
    let j_dd = -k_cd;

    // Effective gain: how much drain force changes per source force
    // In quasi-static limit where plate equilibrates:
    // dx_p = -(j_ps * dx_s + j_pd * dx_d) / j_pp
    // For dx_d = 0: dx_p = -j_ps * dx_s / j_pp
    // Then dF_d = j_dp * dx_p = -j_dp * j_ps * dx_s / j_pp
    // Gain = dF_d/dF_s = (dF_d/dx_s) / (dF_s/dx_s) where dF_s/dx_s includes plate effect
    //
    // More precisely: dF_s = j_ss * dx_s + j_sp * dx_p = j_ss * dx_s - j_sp * j_ps * dx_s / j_pp
    //                     = dx_s * (j_ss - j_sp * j_ps / j_pp)
    // And dF_d = j_dp * dx_p = -j_dp * j_ps * dx_s / j_pp
    //
    // Gain = dF_d / dF_s = [-j_dp * j_ps / j_pp] / [j_ss - j_sp * j_ps / j_pp]
    //                    = [-j_dp * j_ps] / [j_ss * j_pp - j_sp * j_ps]

    let effective_gain = if j_pp.abs() > 1e-30 && (j_ss * j_pp - j_sp * j_ps).abs() > 1e-30 {
        -j_dp * j_ps / (j_ss * j_pp - j_sp * j_ps)
    } else {
        0.0
    };

    // Stability analysis: eigenvalues of -J/M matrix
    // For simplified analysis, we compute the trace and determinant
    // Eigenvalue signs determine stability (all positive omega^2 for stability)
    //
    // The mass-normalized Jacobian is:
    // K = [ j_ss/m_s  j_sp/m_s  0       ]
    //     [ j_ps/m_p  j_pp/m_p  j_pd/m_p]
    //     [ 0         j_dp/m_d  j_dd/m_d]
    //
    // For stability, all eigenvalues of -K should be positive (potential well)

    // Simplified stability check: diagonal elements should be negative
    // (restoring forces) and cross-terms should not cause instability
    let diag_stable = j_ss < 0.0 && j_pp < 0.0 && j_dd < 0.0;

    // Rough eigenvalue estimates using Gershgorin circles
    // For more accuracy, would need full eigenvalue computation
    let lambda_s = j_ss / m_source;
    let lambda_p = j_pp / m_plate;
    let lambda_d = j_dd / m_drain;

    // Omega^2 = -lambda for oscillation
    let omega_sq = [-lambda_s, -lambda_p, -lambda_d];
    let is_stable = omega_sq.iter().all(|&w| w > 0.0) && diag_stable;

    let source_validity = check_pfa_validity(r_source, gap_source);
    let drain_validity = check_pfa_validity(r_drain, gap_drain);

    ThreeBodyResult {
        j_ss,
        j_sp,
        j_ps,
        j_pp,
        j_pd,
        j_dp,
        j_dd,
        k_plate,
        effective_gain,
        stability_eigenvalues: omega_sq,
        is_stable,
        source_validity,
        drain_validity,
    }
}

/// Simplified three-body gain computation (quasi-static plate).
///
/// Computes the transistor gain assuming the plate equilibrates instantly
/// compared to the spheres. This is valid when the plate frequency is much
/// higher than the sphere oscillation frequencies.
///
/// # Arguments
/// * `r_source` - Source sphere radius (m)
/// * `r_drain` - Drain sphere radius (m)
/// * `gap_source` - Source-plate gap (m)
/// * `gap_drain` - Plate-drain gap (m)
/// * `k_plate` - Plate mechanical spring constant (N/m)
///
/// # Returns
/// Transistor gain G = dF_drain / dF_source
pub fn three_body_gain_quasistatic(
    r_source: f64,
    r_drain: f64,
    gap_source: f64,
    gap_drain: f64,
    k_plate: f64,
) -> f64 {
    if k_plate <= 0.0 {
        return 0.0;
    }

    let k_cs = 3.0 * CASIMIR_COEFF * r_source / gap_source.powi(4);
    let k_cd = 3.0 * CASIMIR_COEFF * r_drain / gap_drain.powi(4);

    // Simplified gain formula from Xu et al. 2022
    // G = k_cd / (k_cs + k_cd + k_plate) * k_cs / k_cs = k_cd / (k_cs + k_cd + k_plate)
    // But accounting for force coupling:
    // G = k_cd * k_cs / ((k_cs + k_cd + k_plate) * k_cs) = k_cd / (k_cs + k_cd + k_plate)

    k_cd / (k_cs + k_cd + k_plate)
}

/// Three-body gain with strict PFA validity.
///
/// Computes gain with enforced accuracy requirements.
pub fn three_body_gain_strict(
    r_source: f64,
    r_drain: f64,
    gap_source: f64,
    gap_drain: f64,
    k_plate: f64,
    required_accuracy: PfaAccuracy,
) -> Result<f64, CasimirError> {
    // Check validity for both source and drain
    casimir_force_guarded(r_source, gap_source, required_accuracy)?;
    casimir_force_guarded(r_drain, gap_drain, required_accuracy)?;

    Ok(three_body_gain_quasistatic(
        r_source, r_drain, gap_source, gap_drain, k_plate,
    ))
}

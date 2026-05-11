//! Additivity approximation and first-order nonadditivity corrections
//! for the sphere-plate-sphere Casimir system.
//!
//! The pairwise additivity model treats the total force as the sum of
//! independent sphere-plate PFA interactions. Nonadditivity corrections
//! arise from multiple scattering and derivative-expansion beyond PFA
//! (Bimonte 2012, Fosco et al. 2024).

use super::{CASIMIR_COEFF, PfaAccuracy, casimir_force_pfa};

/// Result of the additivity approximation for sphere-plate-sphere forces.
///
/// The additivity approximation treats the total Casimir force as the sum of
/// independent pairwise sphere-plate interactions. This is exact in PFA at
/// leading order, but nonadditivity corrections arise from:
/// 1. Multiple scattering between the three bodies
/// 2. Geometric effects beyond PFA
/// 3. Material dispersion
///
/// # Literature
/// Xu et al., Nature Communications 13, 6148 (2022), Section "Methods"
#[derive(Debug, Clone)]
pub struct AdditivityResult {
    /// Force on source sphere from plate (N)
    pub force_source: f64,
    /// Force on drain sphere from plate (N)
    pub force_drain: f64,
    /// Source gap (m)
    pub gap_source: f64,
    /// Drain gap (m)
    pub gap_drain: f64,
    /// PFA accuracy level achieved
    pub pfa_accuracy: PfaAccuracy,
    /// Estimated nonadditivity error fraction (0 if not computed)
    pub nonadditivity_error: f64,
    /// Warning if thermal correction may be needed (gap > 1 um)
    pub thermal_warning: bool,
}

/// Compute sphere-plate-sphere forces using the additivity approximation.
///
/// This is the explicit "level 0" model from Xu et al. 2022: the total force
/// is the sum of pairwise sphere-plate forces, with no three-body corrections.
///
/// # Arguments
/// * `r_source` - Source sphere radius (m)
/// * `r_drain` - Drain sphere radius (m)
/// * `gap_source` - Source-plate gap (m)
/// * `gap_drain` - Plate-drain gap (m)
///
/// # Returns
/// AdditivityResult with forces and validity diagnostics
pub fn force_sps_additive(
    r_source: f64,
    r_drain: f64,
    gap_source: f64,
    gap_drain: f64,
) -> AdditivityResult {
    let force_source = casimir_force_pfa(r_source, gap_source);
    let force_drain = casimir_force_pfa(r_drain, gap_drain);

    // Determine achieved PFA accuracy (use the worse of the two)
    let pfa_accuracy = [
        PfaAccuracy::OnePercent,
        PfaAccuracy::FivePercent,
        PfaAccuracy::TenPercent,
        PfaAccuracy::Qualitative,
    ]
    .into_iter()
    .find(|acc| acc.is_valid(r_source, gap_source) && acc.is_valid(r_drain, gap_drain))
    .unwrap_or(PfaAccuracy::Qualitative);

    // Thermal warning if gap > 1 um (thermal wavelength at 300K is ~7.6 um)
    let thermal_warning = gap_source > 1e-6 || gap_drain > 1e-6;

    AdditivityResult {
        force_source,
        force_drain,
        gap_source,
        gap_drain,
        pfa_accuracy,
        nonadditivity_error: 0.0, // Placeholder for future scattering corrections
        thermal_warning,
    }
}

/// Cross-coupling derivative for transistor gain calculation.
///
/// Computes dF_drain/dx_source analytically from the additivity approximation.
/// In pure additivity, this is zero (spheres don't directly couple).
/// With plate flexibility, coupling arises mechanically.
///
/// # Arguments
/// * `r_source` - Source sphere radius (m)
/// * `r_drain` - Drain sphere radius (m)
/// * `gap_source` - Source-plate gap (m)
/// * `gap_drain` - Plate-drain gap (m)
/// * `plate_spring` - Plate mechanical spring constant (N/m), 0 for rigid plate
///
/// # Returns
/// (dF_source/dx_source, dF_drain/dx_drain, dF_drain/dx_source)
pub fn cross_coupling_additive(
    r_source: f64,
    r_drain: f64,
    gap_source: f64,
    gap_drain: f64,
    plate_spring: f64,
) -> (f64, f64, f64) {
    // Casimir spring constants: dF/dx = 3*C*R/d^4 (restoring toward larger gap)
    let k_source = 3.0 * CASIMIR_COEFF * r_source / gap_source.powi(4);
    let k_drain = 3.0 * CASIMIR_COEFF * r_drain / gap_drain.powi(4);

    // Cross-coupling through plate flexibility
    // When source moves dx toward plate:
    //   - Plate displaces by dx_plate = F_source / k_plate
    //   - Drain gap changes by dx_plate
    //   - dF_drain = k_drain * dx_plate
    // So dF_drain/dx_source = k_source * k_drain / k_plate
    let cross = if plate_spring > 0.0 {
        k_source * k_drain / plate_spring
    } else {
        0.0
    };

    (k_source, k_drain, cross)
}

/// Transistor gain from the additivity approximation.
///
/// G = (dF_drain/dx_source) / (dF_source/dx_source) = k_drain / k_plate
///
/// # Arguments
/// * `r_drain` - Drain sphere radius (m)
/// * `gap_drain` - Plate-drain gap (m)
/// * `plate_spring` - Plate mechanical spring constant (N/m)
///
/// # Returns
/// Transistor gain (dimensionless)
pub fn transistor_gain_additive(r_drain: f64, gap_drain: f64, plate_spring: f64) -> f64 {
    if plate_spring <= 0.0 {
        return 0.0;
    }
    let k_drain = 3.0 * CASIMIR_COEFF * r_drain / gap_drain.powi(4);
    k_drain / plate_spring
}

/// First-order nonadditivity correction for the sphere-plate-sphere system.
///
/// Three-body Casimir forces are generically nonadditive.  The leading
/// correction to the additive (pairwise-PFA) result comes from the derivative
/// expansion beyond PFA (Bimonte 2012, Fosco et al. 2024).
///
/// For a single sphere-plate interaction with perfect conductors, the
/// next-to-leading-order correction to PFA is:
///
///   delta_F_i / F_PFA_i = (1/3) * (d_i / R_i)
///
/// where d_i is the gap and R_i the sphere radius.  This captures curvature
/// effects that PFA misses.  Higher orders enter as O((d/R)^2) and involve
/// plate thickness; we neglect those here.
///
/// For the three-body system, we return the sum of both single-sphere
/// corrections (fractional shifts applied to the respective PFA forces,
/// then summed as an absolute force correction in Newtons).
///
/// # Arguments
/// * `r_source` - Source sphere radius (m)
/// * `r_drain` - Drain sphere radius (m)
/// * `gap_source` - Source-plate gap (m)
/// * `gap_drain` - Plate-drain gap (m)
/// * `plate_thickness` - Plate thickness (m), enters at O(d/R)^2 (unused at this order)
///
/// # Returns
/// Absolute nonadditivity correction (N).  Sign convention: positive means
/// the total force magnitude is *larger* than the additive PFA prediction
/// (beyond-PFA curvature corrections reduce the gap-to-radius ratio error
/// and typically increase force magnitude for d/R < 1).
///
/// # Literature
/// - Bimonte, G., PRD 86, 046008 (2012): Beyond-PFA derivative expansion
/// - Fosco, C. D. et al., Physics 6(1), 20 (2024): Derivative expansion review
/// - Rahi et al., PRD 80, 085021 (2009): Scattering approach to Casimir
pub fn nonadditivity_correction(
    r_source: f64,
    r_drain: f64,
    gap_source: f64,
    gap_drain: f64,
    _plate_thickness: f64, // enters at O(d/R)^2 -- documented, not used at first order
) -> f64 {
    // Beyond-PFA coefficient for perfect conductors (derivative expansion).
    // Bimonte (2012) Eq. 3.11: alpha_1 = 1/3 for Dirichlet (perfect conductor).
    const ALPHA_1: f64 = 1.0 / 3.0;

    // PFA force for each sphere-plate pair: F = -C * R / d^3
    let f_pfa_source = casimir_force_pfa(r_source, gap_source);
    let f_pfa_drain = casimir_force_pfa(r_drain, gap_drain);

    // Fractional correction: delta_F_i = |F_PFA_i| * alpha_1 * (d_i / R_i)
    // Take absolute values since PFA forces are negative (attractive).
    let delta_source = f_pfa_source.abs() * ALPHA_1 * (gap_source / r_source);
    let delta_drain = f_pfa_drain.abs() * ALPHA_1 * (gap_drain / r_drain);

    delta_source + delta_drain
}

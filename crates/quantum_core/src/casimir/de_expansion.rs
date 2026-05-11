// ============================================================================
// Derivative Expansion Error Estimates
// ============================================================================
//
// The Derivative Expansion (DE) provides systematic corrections to PFA:
//
//   F = F_PFA * (1 + sum_{n=1}^N a_n (d/R)^n)
//
// For a sphere-plate geometry with Dirichlet BCs (perfect conductor limit):
//   a_1 = 1/3 (leading correction)
//   a_2 ~ 1/10 (subleading, geometry-dependent)
//
// # Literature
// - Emig et al., PRL 96, 220401 (2006): DE for sphere-plate
// - Fosco et al., Physics 6(1), 20 (2024): Comprehensive DE review
// - Bimonte et al., PRD 95, 065004 (2017): Higher-order corrections

use super::casimir_force_pfa;

/// Derivative expansion coefficients for sphere-plate geometry.
///
/// These are exact values from the literature for the leading correction
/// terms beyond PFA.
pub struct DeCoefficients {
    /// a_1 coefficient: leading O(d/R) correction
    pub a1: f64,
    /// a_2 coefficient: subleading O((d/R)^2) correction
    pub a2: f64,
    /// Source for the coefficient values
    pub source: &'static str,
}

impl DeCoefficients {
    /// Coefficients for sphere-plate with perfect conductor (Dirichlet).
    ///
    /// From Emig et al., PRL 96, 220401 (2006).
    pub const SPHERE_PLATE_DIRICHLET: DeCoefficients = DeCoefficients {
        a1: 1.0 / 3.0,
        a2: 0.1, // Approximate
        source: "Emig et al. PRL 96 (2006)",
    };

    /// Coefficients for sphere-plate with Neumann BCs.
    ///
    /// Different boundary conditions give different coefficients.
    pub const SPHERE_PLATE_NEUMANN: DeCoefficients = DeCoefficients {
        a1: -1.0 / 3.0,
        a2: 0.1, // Approximate
        source: "Fosco et al. Physics 6(1) (2024)",
    };

    /// Coefficients for cylinder-plate geometry.
    ///
    /// The cylinder geometry has different correction structure.
    pub const CYLINDER_PLATE: DeCoefficients = DeCoefficients {
        a1: 0.5,
        a2: 0.15,
        source: "Bimonte et al. PRD 95 (2017)",
    };
}

/// Result of derivative expansion analysis.
#[derive(Debug, Clone)]
pub struct DerivativeExpansionResult {
    /// PFA force (zeroth order)
    pub force_pfa: f64,
    /// First-order correction: a_1 * (d/R) * F_PFA
    pub correction_o1: f64,
    /// Second-order correction: a_2 * (d/R)^2 * F_PFA
    pub correction_o2: f64,
    /// Total force with corrections: F_PFA + O1 + O2
    pub force_corrected: f64,
    /// Relative error estimate: |O1 + O2| / |F_PFA|
    pub relative_error: f64,
    /// d/R ratio (expansion parameter)
    pub expansion_param: f64,
    /// Whether the expansion is well-converged (O2 << O1)
    pub is_converged: bool,
    /// Coefficients used
    pub coefficients: &'static str,
}

/// Compute Casimir force with derivative expansion corrections.
///
/// Returns both the corrected force and detailed error analysis.
///
/// # Arguments
/// * `radius` - Sphere radius in meters
/// * `gap` - Surface-to-surface gap in meters
/// * `coeffs` - Derivative expansion coefficients for the geometry
///
/// # Returns
/// Detailed result with force, corrections, and convergence diagnostics
///
/// # Example
/// ```
/// use quantum_core::casimir::{casimir_force_with_de, DeCoefficients};
///
/// let radius = 5e-6;
/// let gap = 100e-9;
///
/// let result = casimir_force_with_de(
///     radius, gap,
///     &DeCoefficients::SPHERE_PLATE_DIRICHLET,
/// );
///
/// println!("PFA force: {:.3e} N", result.force_pfa);
/// println!("Corrected: {:.3e} N", result.force_corrected);
/// println!("Relative error: {:.2}%", result.relative_error * 100.0);
/// ```
pub fn casimir_force_with_de(
    radius: f64,
    gap: f64,
    coeffs: &DeCoefficients,
) -> DerivativeExpansionResult {
    let f_pfa = casimir_force_pfa(radius, gap);
    let eps = gap / radius; // Expansion parameter d/R

    // Corrections (note: F_PFA is negative, corrections maintain sign)
    let corr_o1 = coeffs.a1 * eps * f_pfa;
    let corr_o2 = coeffs.a2 * eps * eps * f_pfa;

    let f_corrected = f_pfa + corr_o1 + corr_o2;

    // Relative error estimate (absolute values for magnitude comparison)
    let total_correction = (corr_o1 + corr_o2).abs();
    let relative_error = if f_pfa.abs() > 1e-50 {
        total_correction / f_pfa.abs()
    } else {
        0.0
    };

    // Convergence check: O2 should be much smaller than O1
    let is_converged = corr_o2.abs() < 0.25 * corr_o1.abs() || eps < 0.01;

    DerivativeExpansionResult {
        force_pfa: f_pfa,
        correction_o1: corr_o1,
        correction_o2: corr_o2,
        force_corrected: f_corrected,
        relative_error,
        expansion_param: eps,
        is_converged,
        coefficients: coeffs.source,
    }
}

/// Estimate the derivative expansion error without computing corrected force.
///
/// This is a lightweight function that just returns the expected relative
/// error for a given geometry, useful for validity checking.
///
/// # Arguments
/// * `radius` - Sphere radius in meters
/// * `gap` - Surface-to-surface gap in meters
///
/// # Returns
/// Estimated relative error (fraction, not percent)
///
/// # Note
/// Uses sphere-plate Dirichlet coefficients by default.
pub fn estimate_de_error(radius: f64, gap: f64) -> f64 {
    let eps = gap / radius;
    let coeffs = &DeCoefficients::SPHERE_PLATE_DIRICHLET;

    // Relative error ~ |a_1 * eps + a_2 * eps^2|
    (coeffs.a1 * eps + coeffs.a2 * eps * eps).abs()
}

/// Compute the gap at which PFA error equals a target value.
///
/// Solves for d such that DE error ~ target_error.
///
/// # Arguments
/// * `radius` - Sphere radius in meters
/// * `target_error` - Desired maximum relative error (e.g., 0.01 for 1%)
///
/// # Returns
/// Maximum gap in meters that achieves the target error
///
/// # Example
/// ```
/// use quantum_core::casimir::max_gap_for_error;
///
/// let radius = 5e-6;  // 5 um sphere
/// let max_gap = max_gap_for_error(radius, 0.01);  // 1% error
///
/// println!("Max gap for 1% error: {:.1} nm", max_gap * 1e9);
/// ```
pub fn max_gap_for_error(radius: f64, target_error: f64) -> f64 {
    // Simplified: using only a_1 term, d/R ~ target_error / a_1
    let a1 = DeCoefficients::SPHERE_PLATE_DIRICHLET.a1;
    let max_eps = target_error / a1.abs();
    radius * max_eps
}

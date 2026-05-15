// ============================================================================
// PFA Validity Guard System + Spring Constant Accuracy Tracking
// ============================================================================
//
// The Proximity Force Approximation (PFA) becomes unreliable when the gap d
// approaches or exceeds the radius R. This module provides strict validity
// guards that enforce geometric constraints and return detailed diagnostics.
// The spring-constant section adds the stricter d/R requirements needed to
// track the ~4x error amplification from differentiating the PFA force.
//
// # Critical Thresholds (Emig et al. PRL 96, 220401)
//
// | Accuracy | R/d ratio | d/R ratio   |
// |----------|-----------|-------------|
// | 1%       | > 132     | < 0.00755   |
// | 5%       | > 26      | < 0.038     |
// | 10%      | > 13      | < 0.077     |
// | Qual.    | > 5       | < 0.2       |
//
// # Literature
// - Emig et al., PRL 96, 220401 (2006)
// - Fosco et al., Physics 6(1), 20 (2024): Derivative expansion review

use thiserror::Error;

use super::{CASIMIR_COEFF, PfaAccuracy, casimir_force_pfa};

/// Errors arising from Casimir force calculations.
#[derive(Debug, Clone, Error)]
pub enum CasimirError {
    /// PFA validity violated: geometry outside approximation regime.
    #[error(
        "PFA validity violated: d/R = {d_over_r:.6} exceeds {accuracy:?} threshold of {threshold:.6}"
    )]
    PfaViolation {
        /// Actual d/R ratio in the geometry
        d_over_r: f64,
        /// Required accuracy level that was violated
        accuracy: PfaAccuracy,
        /// Threshold d/R for the requested accuracy
        threshold: f64,
        /// Detailed validity information
        info: PfaValidityInfo,
    },

    /// Gap is non-positive (surfaces overlapping or touching).
    #[error("Invalid gap: {gap:.3e} m (must be positive)")]
    InvalidGap {
        /// The offending gap value
        gap: f64,
    },

    /// Radius is non-positive.
    #[error("Invalid radius: {radius:.3e} m (must be positive)")]
    InvalidRadius {
        /// The offending radius value
        radius: f64,
    },
}

/// Detailed information about PFA validity for a given geometry.
#[derive(Debug, Clone)]
pub struct PfaValidityInfo {
    /// Sphere radius (m)
    pub radius: f64,
    /// Gap distance (m)
    pub gap: f64,
    /// R/d ratio (larger = better PFA validity)
    pub r_over_d: f64,
    /// d/R ratio (smaller = better PFA validity)
    pub d_over_r: f64,
    /// Achieved accuracy level
    pub achieved_accuracy: PfaAccuracy,
    /// Whether 1% accuracy is achieved
    pub one_percent_valid: bool,
    /// Whether 5% accuracy is achieved
    pub five_percent_valid: bool,
    /// Whether 10% accuracy is achieved
    pub ten_percent_valid: bool,
    /// Whether qualitative validity is achieved
    pub qualitative_valid: bool,
    /// Estimated relative error from PFA (based on d/R)
    pub estimated_error: f64,
}

impl PfaValidityInfo {
    /// Check if the geometry satisfies the given accuracy requirement.
    pub fn satisfies(&self, accuracy: PfaAccuracy) -> bool {
        match accuracy {
            PfaAccuracy::OnePercent => self.one_percent_valid,
            PfaAccuracy::FivePercent => self.five_percent_valid,
            PfaAccuracy::TenPercent => self.ten_percent_valid,
            PfaAccuracy::Qualitative => self.qualitative_valid,
        }
    }
}

/// Check PFA validity and return detailed diagnostics.
///
/// This function provides comprehensive information about whether PFA
/// is valid for a given sphere-plate geometry, including the achieved
/// accuracy level and estimated error.
///
/// # Arguments
/// * `radius` - Sphere radius in meters
/// * `gap` - Surface-to-surface gap in meters
///
/// # Returns
/// Detailed validity information
///
/// # Example
/// ```
/// use quantum_core::casimir::{check_pfa_validity, PfaAccuracy};
///
/// let radius = 5e-6;  // 5 um sphere
/// let gap = 100e-9;   // 100 nm gap
///
/// let info = check_pfa_validity(radius, gap);
/// // R/d = 50: NOT 1% accurate (needs > 132), but IS 5% accurate (needs > 26)
/// assert!(!info.one_percent_valid);
/// assert!(info.five_percent_valid);
/// ```
pub fn check_pfa_validity(radius: f64, gap: f64) -> PfaValidityInfo {
    let r_over_d = if gap > 0.0 { radius / gap } else { 0.0 };
    let d_over_r = if radius > 0.0 {
        gap / radius
    } else {
        f64::INFINITY
    };

    let one_percent_valid = r_over_d > 132.0;
    let five_percent_valid = r_over_d > 26.0;
    let ten_percent_valid = r_over_d > 13.0;
    let qualitative_valid = r_over_d > 5.0;

    // Determine achieved accuracy level
    let achieved_accuracy = if one_percent_valid {
        PfaAccuracy::OnePercent
    } else if five_percent_valid {
        PfaAccuracy::FivePercent
    } else if ten_percent_valid {
        PfaAccuracy::TenPercent
    } else {
        PfaAccuracy::Qualitative
    };

    // Estimate relative error: leading correction ~ d/R
    // From derivative expansion: error ~ (d/R) + O((d/R)^2)
    let estimated_error = d_over_r;

    PfaValidityInfo {
        radius,
        gap,
        r_over_d,
        d_over_r,
        achieved_accuracy,
        one_percent_valid,
        five_percent_valid,
        ten_percent_valid,
        qualitative_valid,
        estimated_error,
    }
}

/// Compute Casimir force with strict PFA validity guard.
///
/// This function enforces PFA validity at the specified accuracy level,
/// returning an error if the geometry violates the constraint.
///
/// # Arguments
/// * `radius` - Sphere radius in meters
/// * `gap` - Surface-to-surface gap in meters
/// * `required_accuracy` - Minimum accuracy level required
///
/// # Returns
/// * `Ok(force)` - Force in Newtons if geometry is valid
/// * `Err(CasimirError::PfaViolation)` - If geometry violates PFA at required accuracy
///
/// # Example
/// ```
/// use quantum_core::casimir::{casimir_force_guarded, PfaAccuracy};
///
/// let radius = 5e-6;
/// let gap = 30e-9;  // R/d ~ 167, satisfies 1% (needs > 132)
///
/// match casimir_force_guarded(radius, gap, PfaAccuracy::OnePercent) {
///     Ok(force) => println!("Force: {:.3e} N", force),
///     Err(e) => eprintln!("PFA violated: {}", e),
/// }
/// ```
pub fn casimir_force_guarded(
    radius: f64,
    gap: f64,
    required_accuracy: PfaAccuracy,
) -> Result<f64, CasimirError> {
    // Validate inputs
    if radius <= 0.0 {
        return Err(CasimirError::InvalidRadius { radius });
    }
    if gap <= 0.0 {
        return Err(CasimirError::InvalidGap { gap });
    }

    // Check PFA validity
    let info = check_pfa_validity(radius, gap);
    if !info.satisfies(required_accuracy) {
        let threshold = 1.0 / required_accuracy.min_r_over_d();
        return Err(CasimirError::PfaViolation {
            d_over_r: info.d_over_r,
            accuracy: required_accuracy,
            threshold,
            info,
        });
    }

    // Compute force
    Ok(casimir_force_pfa(radius, gap))
}

/// Compute Casimir force with validity info (non-failing version).
///
/// Unlike `casimir_force_guarded`, this always computes the force but
/// returns detailed validity information alongside. Useful when you
/// want to compute the force regardless of validity but still need
/// diagnostics.
///
/// # Arguments
/// * `radius` - Sphere radius in meters
/// * `gap` - Surface-to-surface gap in meters
///
/// # Returns
/// Tuple of (force in Newtons, validity info)
pub fn casimir_force_with_validity(radius: f64, gap: f64) -> (f64, PfaValidityInfo) {
    let force = casimir_force_pfa(radius, gap);
    let info = check_pfa_validity(radius, gap);
    (force, info)
}

/// Spring constant with strict PFA validity guard.
///
/// The Casimir spring constant k = dF/dx = 3 * C * R / d^4 where
/// C = pi^3 * hbar * c / 360.
///
/// This function requires stricter PFA validity for spring constant
/// calculations because the derivative amplifies errors.
///
/// # Arguments
/// * `radius` - Sphere radius in meters
/// * `gap` - Surface-to-surface gap in meters
/// * `required_accuracy` - Minimum accuracy level (OnePercent recommended)
///
/// # Returns
/// * `Ok(k)` - Spring constant in N/m if geometry is valid
/// * `Err(CasimirError)` - If geometry violates PFA
pub fn casimir_spring_constant_guarded(
    radius: f64,
    gap: f64,
    required_accuracy: PfaAccuracy,
) -> Result<f64, CasimirError> {
    // For spring constant, we recommend stricter requirements
    // because k ~ 1/d^4 amplifies errors
    casimir_force_guarded(radius, gap, required_accuracy)?;

    // k = dF/dx = 3 * C * R / d^4
    let k = 3.0 * CASIMIR_COEFF * radius / (gap * gap * gap * gap);
    Ok(k)
}

/// Error amplification factor for spring constant vs force.
///
/// Since k ~ dF/dd ~ R/d^4 while F ~ R/d^3, the relative error on k
/// is approximately 4x larger than the relative error on F for the
/// same geometry.
pub const SPRING_CONSTANT_ERROR_FACTOR: f64 = 4.0;

/// Stricter accuracy requirement for spring constant computation.
///
/// Maps a target spring constant accuracy to the required force accuracy,
/// accounting for error amplification from differentiation.
///
/// # Arguments
/// * `k_accuracy` - Desired accuracy for spring constant
///
/// # Returns
/// Required accuracy for force computation (one level stricter)
impl PfaAccuracy {
    /// Get the stricter requirement for spring constant/gain computations.
    ///
    /// Spring constant errors are amplified ~4x relative to force errors,
    /// so we require stricter PFA validity.
    pub fn stricter_for_derivative(&self) -> PfaAccuracy {
        match self {
            PfaAccuracy::Qualitative => PfaAccuracy::TenPercent,
            PfaAccuracy::TenPercent => PfaAccuracy::FivePercent,
            PfaAccuracy::FivePercent => PfaAccuracy::OnePercent,
            PfaAccuracy::OnePercent => PfaAccuracy::OnePercent, // Already strictest
        }
    }

    /// Compute the effective accuracy when computing spring constant.
    ///
    /// Returns the actual accuracy achieved for k given the geometry.
    pub fn effective_for_spring_constant(&self, radius: f64, gap: f64) -> Option<PfaAccuracy> {
        // Spring constant needs stricter geometry
        let info = check_pfa_validity(radius, gap);
        let stricter = self.stricter_for_derivative();
        if info.satisfies(stricter) {
            Some(*self)
        } else if info.satisfies(*self) {
            // Geometry passes for force but not for spring constant
            // Return the degraded accuracy
            match self {
                PfaAccuracy::OnePercent => Some(PfaAccuracy::FivePercent),
                PfaAccuracy::FivePercent => Some(PfaAccuracy::TenPercent),
                PfaAccuracy::TenPercent => Some(PfaAccuracy::Qualitative),
                PfaAccuracy::Qualitative => None,
            }
        } else {
            None
        }
    }
}

/// Spring constant with automatic stricter accuracy enforcement.
///
/// This function automatically applies the stricter d/R requirements
/// needed for spring constant accuracy (error amplified ~4x vs force).
///
/// # Arguments
/// * `radius` - Sphere radius in meters
/// * `gap` - Surface-to-surface gap in meters
/// * `target_k_accuracy` - Desired accuracy for the spring constant result
///
/// # Returns
/// * `Ok(k)` - Spring constant in N/m if geometry supports target accuracy
/// * `Err(CasimirError)` - If geometry cannot achieve target accuracy for k
///
/// # Example
/// ```
/// use quantum_core::casimir::{spring_constant_strict, PfaAccuracy};
///
/// // For 1% spring constant accuracy, we need stricter geometry
/// let radius = 20e-6;  // Larger sphere
/// let gap = 100e-9;
///
/// match spring_constant_strict(radius, gap, PfaAccuracy::FivePercent) {
///     Ok(k) => println!("k = {:.3e} N/m", k),
///     Err(e) => println!("Geometry insufficient: {}", e),
/// }
/// ```
pub fn spring_constant_strict(
    radius: f64,
    gap: f64,
    target_k_accuracy: PfaAccuracy,
) -> Result<f64, CasimirError> {
    // Use stricter force accuracy to achieve target k accuracy
    let force_accuracy = target_k_accuracy.stricter_for_derivative();
    casimir_force_guarded(radius, gap, force_accuracy)?;

    let k = 3.0 * CASIMIR_COEFF * radius / (gap * gap * gap * gap);
    Ok(k)
}

/// Transistor gain with strict PFA validity enforcement.
///
/// The transistor gain G = k_drain / k_plate requires accurate spring
/// constant computation, so this enforces stricter d/R requirements.
///
/// # Arguments
/// * `r_drain` - Drain sphere radius (m)
/// * `gap_drain` - Plate-drain gap (m)
/// * `plate_spring` - Plate mechanical spring constant (N/m)
/// * `required_accuracy` - Minimum accuracy for the gain result
///
/// # Returns
/// * `Ok(gain)` - Transistor gain if geometry supports accuracy
/// * `Err(CasimirError)` - If geometry insufficient
pub fn transistor_gain_strict(
    r_drain: f64,
    gap_drain: f64,
    plate_spring: f64,
    required_accuracy: PfaAccuracy,
) -> Result<f64, CasimirError> {
    if plate_spring <= 0.0 {
        return Ok(0.0);
    }

    // Gain depends on k_drain, which needs stricter accuracy
    let k_drain = spring_constant_strict(r_drain, gap_drain, required_accuracy)?;
    Ok(k_drain / plate_spring)
}

/// Detailed spring constant result with accuracy diagnostics.
#[derive(Debug, Clone)]
pub struct SpringConstantResult {
    /// Computed spring constant (N/m)
    pub k: f64,
    /// Achieved accuracy level
    pub achieved_accuracy: PfaAccuracy,
    /// Estimated relative error on k
    pub estimated_k_error: f64,
    /// Whether the result meets the requested accuracy
    pub meets_requirement: bool,
    /// Validity information
    pub validity: PfaValidityInfo,
}

/// Compute spring constant with full diagnostics.
///
/// Returns detailed information about accuracy and errors.
pub fn spring_constant_with_diagnostics(
    radius: f64,
    gap: f64,
    target_accuracy: PfaAccuracy,
) -> SpringConstantResult {
    let validity = check_pfa_validity(radius, gap);
    let k = 3.0 * CASIMIR_COEFF * radius / (gap * gap * gap * gap);

    // Compute effective accuracy for spring constant
    let effective_acc = target_accuracy.effective_for_spring_constant(radius, gap);
    let achieved_accuracy = effective_acc.unwrap_or(PfaAccuracy::Qualitative);

    // Estimated error on k is ~4x the force error
    let estimated_k_error = SPRING_CONSTANT_ERROR_FACTOR * validity.estimated_error;

    let meets_requirement = effective_acc.is_some()
        && matches!(
            (target_accuracy, achieved_accuracy),
            (PfaAccuracy::OnePercent, PfaAccuracy::OnePercent)
                | (
                    PfaAccuracy::FivePercent,
                    PfaAccuracy::OnePercent | PfaAccuracy::FivePercent
                )
                | (
                    PfaAccuracy::TenPercent,
                    PfaAccuracy::OnePercent | PfaAccuracy::FivePercent | PfaAccuracy::TenPercent
                )
                | (PfaAccuracy::Qualitative, _)
        );

    SpringConstantResult {
        k,
        achieved_accuracy,
        estimated_k_error,
        meets_requirement,
        validity,
    }
}

// ============================================================================
// First-Class Engine Module: Error Types and Hysteresis Detection
// ============================================================================
//
// Structured TcmtError enumeration plus hysteresis analysis for the Kerr
// bistable cubic. Includes turning-point detection (saddle-node
// bifurcations), normalized-to-physical conversion, hysteresis-loop
// tracing (up-sweep and down-sweep), and the cavity-parameter validator.

use thiserror::Error;

use super::{KerrCavity, solve_normalized_cubic};

/// Error types for TCMT operations.
///
/// Structured errors enable robust error handling in engine-level abstractions
/// and control systems integration.
#[derive(Debug, Clone, Error)]
pub enum TcmtError {
    /// Cavity parameters are physically invalid.
    #[error("Invalid cavity: {reason}")]
    InvalidCavity { reason: String },

    /// Input power exceeds safe operating range.
    #[error("Power out of range: {power:.3e} W exceeds limit {limit:.3e} W")]
    PowerOutOfRange { power: f64, limit: f64 },

    /// Detuning is outside the bistable window.
    #[error("Detuning {detuning:.3e} rad/s outside bistable window [{min:.3e}, {max:.3e}]")]
    DetuningOutOfRange { detuning: f64, min: f64, max: f64 },

    /// Solver failed to converge.
    #[error("Convergence failure after {iterations} iterations (tolerance: {tolerance:.3e})")]
    ConvergenceFailure { iterations: usize, tolerance: f64 },

    /// Bistability not achievable with current parameters.
    #[error("Bistability not achievable: {reason}")]
    NoBistability { reason: String },

    /// Thermal runaway detected.
    #[error("Thermal runaway: temperature exceeded {max_temp:.1} K")]
    ThermalRunaway { max_temp: f64 },
}

/// A turning point (saddle-node bifurcation) on the S-curve.
///
/// At turning points, the system jumps between branches. The up-sweep and
/// down-sweep turning points define the hysteresis window.
#[derive(Debug, Clone, Copy)]
pub struct TurningPoint {
    /// Normalized input power u^2 at the turning point.
    pub u_squared: f64,

    /// Normalized intracavity energy y at the turning point.
    pub y: f64,

    /// Physical input power (W) if cavity parameters known.
    pub power: Option<f64>,

    /// Physical stored energy (J) if cavity parameters known.
    pub energy: Option<f64>,

    /// Transmission at the turning point.
    pub transmission: f64,

    /// Branch type: "lower" (increasing power) or "upper" (decreasing power).
    pub branch: TurningPointBranch,
}

/// Branch classification for turning points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurningPointBranch {
    /// Lower branch: encountered when increasing power (jump up).
    Lower,
    /// Upper branch: encountered when decreasing power (jump down).
    Upper,
}

/// Result of hysteresis analysis for a bistable system.
///
/// Contains both turning points and derived quantities for control applications.
#[derive(Debug, Clone)]
pub struct HysteresisResult {
    /// Lower turning point (jump up on increasing power).
    pub turning_lower: TurningPoint,

    /// Upper turning point (jump down on decreasing power).
    pub turning_upper: TurningPoint,

    /// Hysteresis width in normalized power units.
    pub width_normalized: f64,

    /// Hysteresis width in physical power (W), if known.
    pub width_power: Option<f64>,

    /// Energy gap between branches at midpoint power.
    pub energy_contrast: f64,

    /// Normalized detuning used for analysis.
    pub omega: f64,

    /// Whether the system is in the bistable regime.
    pub is_bistable: bool,

    /// Stability margin: how far inside the bistable window.
    pub stability_margin: f64,
}

/// Find the turning points (saddle-node bifurcations) of the S-curve.
///
/// For the normalized cubic f(y) = y^3 - 2*Omega*y^2 + (Omega^2 + 1)*y - u^2 = 0,
/// turning points occur where df/dy = 0, i.e., at the local extrema of u^2(y).
///
/// # Arguments
/// * `omega` - Normalized detuning (must be |omega| > sqrt(3) for bistability)
///
/// # Returns
/// - `Ok(HysteresisResult)` with both turning points if bistable
/// - `Err(TcmtError::NoBistability)` if detuning is below critical
///
/// # Literature
/// - Liu et al., Opt. Express 21(20), 23687 (2013), Section 2.2
pub fn find_turning_points(omega: f64) -> Result<HysteresisResult, TcmtError> {
    let omega_crit = 3.0_f64.sqrt();

    if omega.abs() <= omega_crit {
        return Err(TcmtError::NoBistability {
            reason: format!(
                "|Omega| = {:.4} <= sqrt(3) = {:.4}; detuning too small for bistability",
                omega.abs(),
                omega_crit
            ),
        });
    }

    // Find y values at turning points: 3*y^2 - 4*Omega*y + (Omega^2 + 1) = 0
    // Discriminant: 16*Omega^2 - 12*(Omega^2 + 1) = 4*Omega^2 - 12 = 4*(Omega^2 - 3)
    let disc = omega * omega - 3.0;
    let sqrt_disc = disc.sqrt();

    // Use sign-correct formulas for omega > 0 (symmetric for omega < 0)
    let omega_abs = omega.abs();
    // y1 is at the local MAXIMUM of u^2(y), y2 at the local MINIMUM
    let y1 = (2.0 * omega_abs - sqrt_disc) / 3.0; // Lower y value
    let y2 = (2.0 * omega_abs + sqrt_disc) / 3.0; // Higher y value

    // Compute u^2 at turning points
    let u2_at = |y: f64| y * ((y - omega_abs).powi(2) + 1.0);
    let u_sq_at_y1 = u2_at(y1); // This is the local MAXIMUM (higher power)
    let u_sq_at_y2 = u2_at(y2); // This is the local MINIMUM (lower power)

    // Compute transmission at turning points
    let trans_at = |y: f64| {
        let eff_det = omega_abs - y;
        let denom = eff_det * eff_det + 1.0;
        eff_det * eff_det / denom
    };

    // The "lower" turning point is where system JUMPS UP (at lower power threshold)
    // The "upper" turning point is where system JUMPS DOWN (at higher power threshold)
    // Lower power threshold occurs at y2 (local min of u^2)
    // Higher power threshold occurs at y1 (local max of u^2)

    let turning_lower = TurningPoint {
        u_squared: u_sq_at_y2, // Lower power (local min)
        y: y2,                 // Higher y (where jump up happens)
        power: None,
        energy: None,
        transmission: trans_at(y2),
        branch: TurningPointBranch::Lower,
    };

    let turning_upper = TurningPoint {
        u_squared: u_sq_at_y1, // Higher power (local max)
        y: y1,                 // Lower y (where jump down happens)
        power: None,
        energy: None,
        transmission: trans_at(y1),
        branch: TurningPointBranch::Upper,
    };

    // Hysteresis width: difference between upper and lower power thresholds
    let width_normalized = u_sq_at_y1 - u_sq_at_y2;

    // Energy contrast at midpoint
    let u_sq_mid = (u_sq_at_y1 + u_sq_at_y2) / 2.0;
    let mid_solutions = solve_normalized_cubic(u_sq_mid, omega_abs);
    let energy_contrast = if mid_solutions.y_solutions.len() >= 2 {
        let y_min = mid_solutions
            .y_solutions
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let y_max = mid_solutions
            .y_solutions
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        y_max - y_min
    } else {
        0.0
    };

    // Stability margin: ratio of current width to critical
    let stability_margin = (omega.abs() - omega_crit) / omega_crit;

    Ok(HysteresisResult {
        turning_lower,
        turning_upper,
        width_normalized,
        width_power: None,
        energy_contrast,
        omega: omega_abs,
        is_bistable: true,
        stability_margin,
    })
}

/// Find turning points with physical units attached.
///
/// Converts normalized turning point results to physical quantities
/// using the cavity parameters.
pub fn find_turning_points_physical(
    cavity: &KerrCavity,
    detuning: f64,
) -> Result<HysteresisResult, TcmtError> {
    let g = cavity.gamma_total() / 2.0;
    let gamma_e = cavity.gamma_external();
    let gamma_k = cavity.gamma_kerr();

    if gamma_k.abs() < 1e-30 {
        return Err(TcmtError::NoBistability {
            reason: "Zero Kerr coefficient".to_string(),
        });
    }

    let omega = detuning / g;
    let mut result = find_turning_points(omega)?;

    // Convert to physical units
    // u^2 = 2 * gamma_e * gamma_kerr * P / g^3
    // P = u^2 * g^3 / (2 * gamma_e * gamma_kerr)
    let scale = g.powi(3) / (2.0 * gamma_e * gamma_k.abs());

    result.turning_lower.power = Some(result.turning_lower.u_squared * scale);
    result.turning_upper.power = Some(result.turning_upper.u_squared * scale);

    // y = gamma_kerr * |a|^2 / g
    // |a|^2 = y * g / gamma_kerr
    let energy_scale = g / gamma_k.abs();
    result.turning_lower.energy = Some(result.turning_lower.y * energy_scale);
    result.turning_upper.energy = Some(result.turning_upper.y * energy_scale);

    result.width_power = Some(result.width_normalized * scale);

    Ok(result)
}

/// Compute hysteresis width in normalized units.
///
/// Quick convenience function that returns just the width.
pub fn hysteresis_width(omega: f64) -> Option<f64> {
    find_turning_points(omega).ok().map(|r| r.width_normalized)
}

/// Trace a complete hysteresis loop by sweeping power up then down.
///
/// Returns (power_values, upper_branch_energies, lower_branch_energies)
/// where each branch is None outside the bistable window.
///
/// # Arguments
/// * `omega` - Normalized detuning
/// * `n_points` - Number of points in each sweep direction
/// * `power_range` - (min_u_sq, max_u_sq) range to sweep
pub fn trace_hysteresis_loop(
    omega: f64,
    n_points: usize,
    power_range: (f64, f64),
) -> HysteresisTrace {
    let (u_min, u_max) = power_range;
    let step = (u_max - u_min) / (n_points - 1).max(1) as f64;

    let mut powers = Vec::with_capacity(n_points);
    let mut up_sweep = Vec::with_capacity(n_points);
    let mut down_sweep = vec![None; n_points];

    // Up-sweep: start on lower branch (index 0), jump when it disappears
    let mut up_branch_idx = 0_usize;
    for i in 0..n_points {
        let u_sq = u_min + i as f64 * step;
        powers.push(u_sq);

        let result = solve_normalized_cubic(u_sq, omega);
        let mut sorted: Vec<_> = result.y_solutions.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if sorted.is_empty() {
            up_sweep.push(None);
        } else if sorted.len() == 1 {
            up_sweep.push(Some(sorted[0]));
            up_branch_idx = 0;
        } else {
            // Multi-solution region: try to stay on current branch
            if up_branch_idx < sorted.len() {
                up_sweep.push(Some(sorted[up_branch_idx]));
            } else {
                // Branch disappeared: jump to highest available (the jump UP)
                up_branch_idx = sorted.len() - 1;
                up_sweep.push(Some(sorted[up_branch_idx]));
            }
        }
    }

    // Down-sweep: start on upper branch (highest index), jump when it disappears
    // Track: was the previous point (higher power) in single-solution regime?
    let mut was_single_solution = true;
    let mut down_branch_idx = 0_usize;

    for i in (0..n_points).rev() {
        let u_sq = powers[i];

        let result = solve_normalized_cubic(u_sq, omega);
        let mut sorted: Vec<_> = result.y_solutions.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if sorted.is_empty() {
            down_sweep[i] = None;
            was_single_solution = true;
        } else if sorted.len() == 1 {
            down_sweep[i] = Some(sorted[0]);
            down_branch_idx = 0;
            was_single_solution = true;
        } else {
            // Multi-solution region
            if was_single_solution {
                // Just entered from single-solution: start on UPPER branch
                down_branch_idx = sorted.len() - 1;
            }

            if down_branch_idx < sorted.len() {
                down_sweep[i] = Some(sorted[down_branch_idx]);
            } else {
                // Our branch disappeared (went below lower turning point): jump DOWN
                down_branch_idx = 0;
                down_sweep[i] = Some(sorted[down_branch_idx]);
            }
            was_single_solution = false;
        }
    }

    HysteresisTrace {
        powers,
        up_sweep,
        down_sweep,
        omega,
    }
}

/// A traced hysteresis loop with up-sweep and down-sweep branches.
#[derive(Debug, Clone)]
pub struct HysteresisTrace {
    /// Normalized power values u^2.
    pub powers: Vec<f64>,

    /// Energies following up-sweep (lower->upper branch).
    pub up_sweep: Vec<Option<f64>>,

    /// Energies following down-sweep (upper->lower branch).
    pub down_sweep: Vec<Option<f64>>,

    /// Normalized detuning used.
    pub omega: f64,
}

impl HysteresisTrace {
    /// Check if the trace shows actual hysteresis (up != down in some region).
    pub fn has_hysteresis(&self) -> bool {
        self.up_sweep
            .iter()
            .zip(self.down_sweep.iter())
            .any(|(up, down)| match (up, down) {
                (Some(u), Some(d)) => (u - d).abs() > 1e-10,
                _ => false,
            })
    }

    /// Find the power range where hysteresis occurs.
    pub fn hysteresis_power_range(&self) -> Option<(f64, f64)> {
        let mut first = None;
        let mut last = None;

        for (i, (up, down)) in self.up_sweep.iter().zip(self.down_sweep.iter()).enumerate() {
            if let (Some(u), Some(d)) = (up, down)
                && (u - d).abs() > 1e-10
            {
                if first.is_none() {
                    first = Some(self.powers[i]);
                }
                last = Some(self.powers[i]);
            }
        }

        match (first, last) {
            (Some(f), Some(l)) => Some((f, l)),
            _ => None,
        }
    }
}

/// Validate cavity parameters and return error if invalid.
pub fn validate_cavity(cavity: &KerrCavity) -> Result<(), TcmtError> {
    if cavity.omega_0 <= 0.0 {
        return Err(TcmtError::InvalidCavity {
            reason: "omega_0 must be positive".to_string(),
        });
    }
    if cavity.q_intrinsic <= 0.0 || cavity.q_external <= 0.0 {
        return Err(TcmtError::InvalidCavity {
            reason: "Quality factors must be positive".to_string(),
        });
    }
    if cavity.v_eff <= 0.0 {
        return Err(TcmtError::InvalidCavity {
            reason: "Mode volume must be positive".to_string(),
        });
    }
    if cavity.n_linear <= 0.0 {
        return Err(TcmtError::InvalidCavity {
            reason: "Refractive index must be positive".to_string(),
        });
    }
    Ok(())
}

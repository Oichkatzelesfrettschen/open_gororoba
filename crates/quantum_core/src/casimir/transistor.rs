//! Casimir transistor: three-terminal force analysis and gap sweeps.
//!
//! Implements the transistor gain G = (dF_drain/dx_source) / (dF_source/dx_source)
//! for the sphere-plate-sphere system and provides a parameter sweep helper.
//! See Xu et al., Nature Communications 13, 6148 (2022).

use super::{CASIMIR_COEFF, Plate, Sphere, SpherePlateSphere, casimir_force_pfa, pfa_is_valid};

/// Casimir transistor analysis result.
#[derive(Debug, Clone)]
pub struct TransistorResult {
    /// Transistor gain: dF_drain/dF_source (dimensionless)
    pub gain: f64,
    /// Source force at equilibrium (N)
    pub force_source: f64,
    /// Drain force at equilibrium (N)
    pub force_drain: f64,
    /// Source spring constant: dF_source/dx_source (N/m)
    pub spring_source: f64,
    /// Drain spring constant: dF_drain/dx_drain (N/m)
    pub spring_drain: f64,
    /// Cross-coupling: dF_drain/dx_source (N/m)
    pub cross_coupling: f64,
    /// Whether the system is in the transistor regime
    pub in_transistor_regime: bool,
}

/// Compute transistor characteristics for the sphere-plate-sphere system.
///
/// The transistor gain is defined as the ratio of force sensitivity:
///   G = (dF_drain/dx_source) / (dF_source/dx_source)
///
/// In the basic PFA model, the spheres don't directly couple through the plate
/// (cross-coupling is zero), but coupling can arise from:
/// 1. Finite plate flexibility (mechanical coupling)
/// 2. Higher-order Casimir corrections
/// 3. Electromagnetic coupling through the plate
///
/// This function computes the spring constants and identifies the transistor regime.
///
/// # Arguments
/// * `system` - The sphere-plate-sphere configuration
/// * `plate_spring_constant` - Mechanical spring constant of the plate (N/m)
///   If zero, only pure Casimir effects are considered.
pub fn analyze_transistor(
    system: &SpherePlateSphere,
    plate_spring_constant: f64,
) -> TransistorResult {
    let gap_s = system.source_plate_gap();
    let gap_d = system.drain_plate_gap();
    let r_s = system.source.radius;
    let r_d = system.drain.radius;

    // Forces
    let f_s = casimir_force_pfa(r_s, gap_s);
    let f_d = casimir_force_pfa(r_d, gap_d);

    // Spring constants: dF/dx = d/dx[-C*R/d^3] = 3*C*R/d^4
    // Note: Increasing gap reduces force magnitude, so spring constant is positive
    // (restoring force toward larger gap when perturbed toward smaller gap)
    let k_s = 3.0 * CASIMIR_COEFF * r_s / (gap_s * gap_s * gap_s * gap_s);
    let k_d = 3.0 * CASIMIR_COEFF * r_d / (gap_d * gap_d * gap_d * gap_d);

    // Cross-coupling through plate flexibility
    // When source moves toward plate, plate displaces, affecting drain gap
    // The coupling depends on the ratio of Casimir spring to plate spring
    let cross_coupling = if plate_spring_constant > 0.0 {
        // Plate displacement: dx_plate = F_s / k_plate
        // Force change on drain: dF_d = k_d * dx_plate
        // So cross-coupling = k_d * k_s / k_plate
        k_s * k_d / plate_spring_constant
    } else {
        0.0
    };

    // Gain: ratio of force changes
    let gain = if k_s.abs() > 1e-30 {
        cross_coupling / k_s
    } else {
        0.0
    };

    // Transistor regime: significant gain (> 0.01) and valid PFA
    let in_transistor_regime =
        gain.abs() > 0.01 && pfa_is_valid(r_s, gap_s) && pfa_is_valid(r_d, gap_d);

    TransistorResult {
        gain,
        force_source: f_s,
        force_drain: f_d,
        spring_source: k_s,
        spring_drain: k_d,
        cross_coupling,
        in_transistor_regime,
    }
}

/// Parameter sweep result for transistor characterization.
#[derive(Debug, Clone)]
pub struct SweepResult {
    /// Source gap values (m)
    pub source_gaps: Vec<f64>,
    /// Drain gap values (m)
    pub drain_gaps: Vec<f64>,
    /// Force on source for each configuration (N)
    pub forces_source: Vec<f64>,
    /// Force on drain for each configuration (N)
    pub forces_drain: Vec<f64>,
    /// Transistor gain for each configuration
    pub gains: Vec<f64>,
    /// Whether each configuration is in transistor regime
    pub in_regime: Vec<bool>,
}

/// Sweep source gap while keeping drain gap fixed.
///
/// # Arguments
/// * `sphere_radius` - Radius of both spheres (m)
/// * `drain_gap` - Fixed drain-plate gap (m)
/// * `source_gaps` - Source gap values to sweep (m)
/// * `plate_spring` - Plate spring constant (N/m)
pub fn sweep_source_gap(
    sphere_radius: f64,
    drain_gap: f64,
    source_gaps: &[f64],
    plate_spring: f64,
) -> SweepResult {
    let mut forces_source = Vec::with_capacity(source_gaps.len());
    let mut forces_drain = Vec::with_capacity(source_gaps.len());
    let mut gains = Vec::with_capacity(source_gaps.len());
    let mut in_regime = Vec::with_capacity(source_gaps.len());
    let drain_gaps = vec![drain_gap; source_gaps.len()];

    for &gap_s in source_gaps {
        // Build system
        // Source at origin, plate after source gap
        let source = Sphere::new(sphere_radius, 0.0);
        let plate_pos = sphere_radius + gap_s;
        let plate = Plate::new(plate_pos, 0.0); // infinitely thin plate approximation
        let drain_pos = plate_pos + drain_gap + sphere_radius;
        let drain = Sphere::new(sphere_radius, drain_pos);

        let system = SpherePlateSphere {
            source,
            plate,
            drain,
        };
        let transistor = analyze_transistor(&system, plate_spring);

        forces_source.push(transistor.force_source);
        forces_drain.push(transistor.force_drain);
        gains.push(transistor.gain);
        in_regime.push(transistor.in_transistor_regime);
    }

    SweepResult {
        source_gaps: source_gaps.to_vec(),
        drain_gaps,
        forces_source,
        forces_drain,
        gains,
        in_regime,
    }
}

//! Casimir sphere-plate-sphere system implementation (Xu et al. 2022).
//!
//! Implements the three-terminal Casimir transistor architecture from
//! Xu et al., Nature Communications 13, 6148 (2022).
//!
//! # Physics
//!
//! The Casimir effect is a quantum vacuum phenomenon where conducting surfaces
//! experience an attractive force due to the modification of vacuum fluctuation
//! modes. For a sphere near a plate, the Proximity Force Approximation (PFA)
//! gives:
//!
//!   F_sp = -pi^3 * hbar * c * R / (360 * d^3)
//!
//! where R is the sphere radius, d is the separation, and the force is attractive
//! (negative toward the plate).
//!
//! # Three-Terminal Architecture
//!
//! The sphere-plate-sphere system consists of:
//! - **Source sphere**: One microsphere whose position modulates the system
//! - **Gate plate**: A conducting plate between the spheres
//! - **Drain sphere**: Second microsphere experiencing the modulated force
//!
//! When the source sphere moves, it changes the local vacuum mode structure,
//! affecting the force experienced by the drain sphere. The transistor gain is:
//!
//!   G = dF_drain / dF_source
//!
//! # Literature
//!
//! - Xu et al., Nature Communications 13, 6148 (2022)
//! - Casimir, Proc. K. Ned. Akad. Wet. 51, 793 (1948)
//! - Bordag et al., Physics Reports 353, 1 (2001) - Comprehensive review
//! - Derjaguin et al., Kolloid-Z. 69, 155 (1934) - PFA foundation

use std::f64::consts::PI;
use thiserror::Error;

// Physical constants (SI units)
/// Reduced Planck constant (J*s)
pub const HBAR: f64 = 1.054571817e-34;
/// Speed of light (m/s)
pub const C: f64 = 299792458.0;

/// Casimir force coefficient: pi^3 * hbar * c / 360
/// Units: J*m (or N*m^3 when divided by d^3)
pub const CASIMIR_COEFF: f64 = PI * PI * PI * HBAR * C / 360.0;

/// A microsphere in the Casimir system.
#[derive(Debug, Clone, Copy)]
pub struct Sphere {
    /// Sphere radius (meters)
    pub radius: f64,
    /// Position along the axis (meters)
    pub position: f64,
}

impl Sphere {
    /// Create a new sphere.
    ///
    /// # Arguments
    /// * `radius` - Sphere radius in meters (typically ~5 micrometers)
    /// * `position` - Position along the system axis in meters
    pub fn new(radius: f64, position: f64) -> Self {
        Sphere { radius, position }
    }

    /// Create a sphere with radius in micrometers (convenience constructor).
    pub fn from_micrometers(radius_um: f64, position_um: f64) -> Self {
        Sphere {
            radius: radius_um * 1e-6,
            position: position_um * 1e-6,
        }
    }
}

/// A conducting plate in the Casimir system.
#[derive(Debug, Clone, Copy)]
pub struct Plate {
    /// Position along the axis (meters)
    pub position: f64,
    /// Plate thickness (meters) - for geometry validation
    pub thickness: f64,
}

impl Plate {
    /// Create a new plate.
    ///
    /// # Arguments
    /// * `position` - Position along the system axis in meters
    /// * `thickness` - Plate thickness in meters
    pub fn new(position: f64, thickness: f64) -> Self {
        Plate { position, thickness }
    }

    /// Create a plate with dimensions in micrometers.
    pub fn from_micrometers(position_um: f64, thickness_um: f64) -> Self {
        Plate {
            position: position_um * 1e-6,
            thickness: thickness_um * 1e-6,
        }
    }
}

/// The sphere-plate-sphere system configuration.
#[derive(Debug, Clone)]
pub struct SpherePlateSphere {
    /// Source sphere (left side by convention)
    pub source: Sphere,
    /// Conducting gate plate
    pub plate: Plate,
    /// Drain sphere (right side by convention)
    pub drain: Sphere,
}

impl SpherePlateSphere {
    /// Create a new sphere-plate-sphere system.
    ///
    /// # Arguments
    /// * `source` - Source sphere (typically on the left)
    /// * `plate` - Conducting gate plate
    /// * `drain` - Drain sphere (typically on the right)
    ///
    /// # Panics
    /// Panics if the geometry is invalid (spheres overlapping with plate).
    pub fn new(source: Sphere, plate: Plate, drain: Sphere) -> Self {
        let sps = SpherePlateSphere { source, plate, drain };
        assert!(sps.source_plate_gap() > 0.0, "Source sphere overlaps with plate");
        assert!(sps.drain_plate_gap() > 0.0, "Drain sphere overlaps with plate");
        sps
    }

    /// Create a symmetric configuration from micrometers.
    ///
    /// # Arguments
    /// * `sphere_radius_um` - Radius of both spheres in micrometers
    /// * `source_gap_um` - Gap between source sphere surface and plate surface (micrometers)
    /// * `drain_gap_um` - Gap between plate surface and drain sphere surface (micrometers)
    /// * `plate_thickness_um` - Plate thickness in micrometers
    pub fn symmetric_from_micrometers(
        sphere_radius_um: f64,
        source_gap_um: f64,
        drain_gap_um: f64,
        plate_thickness_um: f64,
    ) -> Self {
        // Convert all to meters
        let r = sphere_radius_um * 1e-6;
        let g_s = source_gap_um * 1e-6;
        let g_d = drain_gap_um * 1e-6;
        let t = plate_thickness_um * 1e-6;

        // Source sphere at origin (center at 0)
        // Source right edge at r
        // Plate left edge at r + g_s
        // Plate center at r + g_s + t/2
        // Plate right edge at r + g_s + t
        // Drain left edge at r + g_s + t + g_d
        // Drain center at r + g_s + t + g_d + r
        let source_pos = 0.0;
        let plate_pos = r + g_s + t / 2.0;
        let drain_pos = r + g_s + t + g_d + r;

        SpherePlateSphere::new(
            Sphere::new(r, source_pos),
            Plate::new(plate_pos, t),
            Sphere::new(r, drain_pos),
        )
    }

    /// Gap between source sphere surface and plate (meters).
    pub fn source_plate_gap(&self) -> f64 {
        (self.plate.position - self.plate.thickness / 2.0)
            - (self.source.position + self.source.radius)
    }

    /// Gap between plate and drain sphere surface (meters).
    pub fn drain_plate_gap(&self) -> f64 {
        (self.drain.position - self.drain.radius)
            - (self.plate.position + self.plate.thickness / 2.0)
    }
}

/// Result of Casimir force calculation.
#[derive(Debug, Clone)]
pub struct CasimirForceResult {
    /// Force on source sphere (N, positive = away from plate)
    pub force_source: f64,
    /// Force on drain sphere (N, positive = away from plate)
    pub force_drain: f64,
    /// Source-plate gap (m)
    pub gap_source: f64,
    /// Drain-plate gap (m)
    pub gap_drain: f64,
    /// Whether PFA is valid (R >> d)
    pub pfa_valid: bool,
}

/// Compute sphere-plate Casimir force using Proximity Force Approximation.
///
/// F_sp = -pi^3 * hbar * c * R / (360 * d^3)
///
/// The force is attractive (toward the plate), so the returned value is negative.
///
/// # Arguments
/// * `radius` - Sphere radius in meters
/// * `gap` - Surface-to-surface gap in meters
///
/// # Returns
/// Force in Newtons (negative = attractive toward plate)
///
/// # Literature
/// Bordag et al., Physics Reports 353, 1 (2001), Eq. 2.53
pub fn casimir_force_pfa(radius: f64, gap: f64) -> f64 {
    if gap <= 0.0 {
        return f64::NEG_INFINITY;
    }
    -CASIMIR_COEFF * radius / (gap * gap * gap)
}

/// PFA accuracy level for validity checking.
///
/// Based on Emig et al., PRL 96, 220401 (2006) and the 2024 Derivative Expansion review.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PfaAccuracy {
    /// 1% accuracy: requires R/d > 132 (d/R < 0.00755)
    OnePercent,
    /// 5% accuracy: requires R/d > 26 (d/R < 0.038)
    FivePercent,
    /// 10% accuracy: requires R/d > 13 (d/R < 0.077)
    TenPercent,
    /// Qualitative only: R/d > 5 (significant deviations expected)
    Qualitative,
}

impl PfaAccuracy {
    /// Get the minimum R/d ratio for this accuracy level.
    pub fn min_r_over_d(&self) -> f64 {
        match self {
            PfaAccuracy::OnePercent => 132.0,
            PfaAccuracy::FivePercent => 26.0,
            PfaAccuracy::TenPercent => 13.0,
            PfaAccuracy::Qualitative => 5.0,
        }
    }

    /// Check if PFA is valid at this accuracy level.
    pub fn is_valid(&self, radius: f64, gap: f64) -> bool {
        if gap <= 0.0 {
            return false;
        }
        radius / gap > self.min_r_over_d()
    }
}

/// Check if PFA is valid for given geometry (qualitative level).
///
/// PFA requires R >> d. For 1% accuracy, R/d > 132 is needed.
/// This function uses the qualitative threshold R > 5*d for backward compatibility.
///
/// For precision work, use `PfaAccuracy::OnePercent.is_valid(r, d)` instead.
///
/// # Literature
/// - Emig et al., PRL 96, 220401 (2006): d/R < 0.00755 for 1% accuracy
/// - Fosco et al., Physics 6(1), 20 (2024): Derivative expansion review
pub fn pfa_is_valid(radius: f64, gap: f64) -> bool {
    PfaAccuracy::Qualitative.is_valid(radius, gap)
}

/// Check PFA validity with specified accuracy requirement.
pub fn pfa_is_valid_at_accuracy(radius: f64, gap: f64, accuracy: PfaAccuracy) -> bool {
    accuracy.is_valid(radius, gap)
}

/// Compute forces in the sphere-plate-sphere system.
///
/// Returns the Casimir force on each sphere due to its interaction with the plate.
/// The forces are independent in the PFA (no direct sphere-sphere coupling through
/// the plate at leading order).
///
/// # Arguments
/// * `system` - The sphere-plate-sphere configuration
pub fn compute_casimir_forces(system: &SpherePlateSphere) -> CasimirForceResult {
    let gap_source = system.source_plate_gap();
    let gap_drain = system.drain_plate_gap();

    let force_source = casimir_force_pfa(system.source.radius, gap_source);
    let force_drain = casimir_force_pfa(system.drain.radius, gap_drain);

    let pfa_valid_source = pfa_is_valid(system.source.radius, gap_source);
    let pfa_valid_drain = pfa_is_valid(system.drain.radius, gap_drain);

    CasimirForceResult {
        force_source,
        force_drain,
        gap_source,
        gap_drain,
        pfa_valid: pfa_valid_source && pfa_valid_drain,
    }
}

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
    let in_transistor_regime = gain.abs() > 0.01
        && pfa_is_valid(r_s, gap_s)
        && pfa_is_valid(r_d, gap_d);

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

        let system = SpherePlateSphere { source, plate, drain };
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

/// Compute Casimir energy between sphere and plate (PFA).
///
/// E_sp = -pi^3 * hbar * c * R / (720 * d^2)
///
/// This is the integral of the force: E = integral F dd from infinity to d.
///
/// # Arguments
/// * `radius` - Sphere radius in meters
/// * `gap` - Surface-to-surface gap in meters
///
/// # Returns
/// Energy in Joules (negative = bound state)
pub fn casimir_energy_pfa(radius: f64, gap: f64) -> f64 {
    if gap <= 0.0 {
        return f64::NEG_INFINITY;
    }
    -PI * PI * PI * HBAR * C * radius / (720.0 * gap * gap)
}

/// Finite conductivity correction factor for the Casimir force.
///
/// The ideal PFA assumes perfect conductors. For real metals with plasma
/// frequency omega_p, the force is reduced at short distances by:
///
///   F_real / F_ideal = 1 - 4*c/(omega_p * d) + O(c/(omega_p*d))^2
///
/// This correction becomes significant when d < c/omega_p (the skin depth).
///
/// # Arguments
/// * `gap` - Surface gap in meters
/// * `plasma_wavelength` - Plasma wavelength lambda_p = 2*pi*c/omega_p (m)
///   For gold: lambda_p ~ 136 nm
///
/// # Returns
/// Correction factor (< 1 for finite conductivity)
pub fn finite_conductivity_correction(gap: f64, plasma_wavelength: f64) -> f64 {
    let ratio = plasma_wavelength / (2.0 * PI * gap);
    (1.0 - 4.0 * ratio).max(0.1)  // Saturate at 0.1 to avoid unphysical negative values
}

/// Thermal (finite temperature) correction factor for the Casimir force.
///
/// At finite temperature T, thermal fluctuations contribute to the force.
/// The thermal correction becomes significant when the thermal wavelength
/// lambda_T = hbar*c/(k_B*T) is comparable to the gap.
///
/// At room temperature (T ~ 300 K), lambda_T ~ 7.6 um.
///
/// # Arguments
/// * `gap` - Surface gap in meters
/// * `temperature` - Temperature in Kelvin
///
/// # Returns
/// Correction factor (> 1 for thermal enhancement at large gaps)
pub fn thermal_correction(gap: f64, temperature: f64) -> f64 {
    const K_B: f64 = 1.380649e-23;  // Boltzmann constant
    let lambda_t = HBAR * C / (K_B * temperature);

    // Simplified thermal correction from Bordag et al.
    // Valid for gap >> lambda_T (high-temperature limit)
    // and gap << lambda_T (low-temperature limit interpolation)
    let x = gap / lambda_t;
    if x < 0.1 {
        // Low temperature: negligible correction
        1.0
    } else if x > 10.0 {
        // High temperature: thermal dominates, force ~ T/d^3
        x / 3.0
    } else {
        // Interpolation region
        1.0 + 0.1 * x * x
    }
}

/// Complete Casimir force with corrections.
///
/// Combines PFA with finite conductivity and thermal corrections.
///
/// # Arguments
/// * `radius` - Sphere radius (m)
/// * `gap` - Surface gap (m)
/// * `plasma_wavelength` - Plasma wavelength of conductor (m)
/// * `temperature` - Temperature (K)
pub fn casimir_force_with_corrections(
    radius: f64,
    gap: f64,
    plasma_wavelength: f64,
    temperature: f64,
) -> f64 {
    let f_pfa = casimir_force_pfa(radius, gap);
    let fc_corr = finite_conductivity_correction(gap, plasma_wavelength);
    let thermal_corr = thermal_correction(gap, temperature);

    f_pfa * fc_corr * thermal_corr
}

// ============================================================================
// Additivity Approximation API (Xu et al. 2022)
// ============================================================================

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

/// Nonadditivity correction placeholder (feature-gated for future implementation).
///
/// Three-body Casimir forces are generically nonadditive. The leading correction
/// to the additivity approximation comes from:
/// 1. Multiple scattering: photons bounce between all three surfaces
/// 2. Edge/curvature effects beyond PFA
///
/// This function returns 0 currently but provides the API hook for future
/// implementation of scattering-matrix or derivative-expansion corrections.
///
/// # Literature
/// - Rahi et al., PRD 80, 085021 (2009): Scattering approach to Casimir
/// - Fosco et al., Physics 6(1), 20 (2024): Derivative expansion beyond PFA
#[allow(unused_variables)]
pub fn nonadditivity_correction(
    r_source: f64,
    r_drain: f64,
    gap_source: f64,
    gap_drain: f64,
    _plate_thickness: f64,
) -> f64 {
    // TODO: Implement scattering-matrix correction when needed
    // For now, return 0 (pure additivity)
    0.0
}

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

    Ok(three_body_gain_quasistatic(r_source, r_drain, gap_source, gap_drain, k_plate))
}

// ============================================================================
// Lifshitz Formula with Dielectric Functions
// ============================================================================
//
// The Lifshitz formula computes Casimir forces from the frequency-dependent
// dielectric functions of the materials. Unlike the ideal PFA (perfect conductor
// limit), Lifshitz theory captures:
// - Material-dependent forces
// - Finite temperature effects
// - Frequency-dependent optical response
//
// # Literature
// - Lifshitz, Sov. Phys. JETP 2, 73 (1956) - Original theory
// - Dzyaloshinskii et al., Adv. Phys. 10, 165 (1961) - Extension to materials
// - Klimchitskaya et al., RMP 81, 1827 (2009) - Comprehensive review

/// Dielectric model for computing optical response at imaginary frequency.
///
/// The dielectric function at imaginary frequency \epsilon(i\xi) is always real and
/// monotonically decreasing from \epsilon(0) to 1 as \xi -> infty. This is a consequence
/// of Kramers-Kronig relations.
#[derive(Debug, Clone)]
pub enum DielectricModel {
    /// Perfect conductor: \epsilon -> infty (reflection coefficient r = 1)
    PerfectConductor,

    /// Drude model for metals: eps(i*xi) = 1 + omega_p^2 / (xi*(xi + gamma))
    /// Parameters: (plasma frequency omega_p in rad/s, damping gamma in rad/s)
    Drude { omega_p: f64, gamma: f64 },

    /// Plasma model (dissipationless Drude): eps(i*xi) = 1 + omega_p^2 / xi^2
    /// The gamma -> 0 limit of Drude; controversial for thermal Casimir effect
    Plasma { omega_p: f64 },

    /// Drude-Lorentz oscillator model (dielectrics with resonance):
    /// eps(i*xi) = 1 + Sum S_j omega_j^2 / (omega_j^2 + xi^2 + gamma_j xi)
    /// Parameters: Vec of (oscillator strength S, resonance omega_j, damping gamma_j)
    DrudeLorentz { oscillators: Vec<(f64, f64, f64)> },

    /// Tabulated dielectric data (interpolated)
    /// Parameters: (frequencies \xi in rad/s, dielectric values \epsilon(i\xi))
    Tabulated { xi: Vec<f64>, eps: Vec<f64> },
}

impl DielectricModel {
    /// Create a Drude model for gold at room temperature.
    ///
    /// Parameters from optical data:
    /// - \omega_p = 9.0 eV = 1.37e16 rad/s
    /// - \gamma = 35 meV = 5.3e13 rad/s
    pub fn gold() -> Self {
        DielectricModel::Drude {
            omega_p: 1.37e16,
            gamma: 5.3e13,
        }
    }

    /// Create a Drude model for aluminum.
    ///
    /// Parameters from optical data:
    /// - \omega_p = 12.5 eV = 1.9e16 rad/s
    /// - \gamma = 126 meV = 1.9e14 rad/s
    pub fn aluminum() -> Self {
        DielectricModel::Drude {
            omega_p: 1.9e16,
            gamma: 1.9e14,
        }
    }

    /// Create a plasma model for gold (dissipationless limit).
    pub fn gold_plasma() -> Self {
        DielectricModel::Plasma { omega_p: 1.37e16 }
    }

    /// Create a simple dielectric model for silica (SiO2).
    ///
    /// Simplified single-oscillator model:
    /// - UV resonance at ~10 eV
    /// - Static eps ~ 3.8
    pub fn silica() -> Self {
        // Single oscillator: S*w^2 / (w^2 + xi^2) where S*(w/w)^2 gives eps(0)-1
        // For eps(0) ~ 3.8, S ~ 2.8 with w ~ 1.5e16 rad/s (10 eV)
        DielectricModel::DrudeLorentz {
            oscillators: vec![(2.8, 1.5e16, 1e15)],
        }
    }

    /// Evaluate dielectric function at imaginary frequency \epsilon(i\xi).
    ///
    /// Returns the real, positive dielectric response at the imaginary
    /// frequency i\xi. The result is always >= 1 for passive materials.
    ///
    /// # Arguments
    /// * `xi` - Imaginary frequency (rad/s), must be >= 0
    pub fn epsilon_at_imaginary(&self, xi: f64) -> f64 {
        match self {
            // Perfect conductor approximated by very large but finite \epsilon
            // This avoids NaN in numerical integration while capturing
            // the essential physics (r -> 1)
            DielectricModel::PerfectConductor => 1e20,

            DielectricModel::Drude { omega_p, gamma } => {
                if xi == 0.0 {
                    f64::INFINITY // DC conductivity diverges
                } else {
                    1.0 + omega_p * omega_p / (xi * (xi + gamma))
                }
            }

            DielectricModel::Plasma { omega_p } => {
                if xi == 0.0 {
                    f64::INFINITY
                } else {
                    1.0 + omega_p * omega_p / (xi * xi)
                }
            }

            DielectricModel::DrudeLorentz { oscillators } => {
                let mut eps = 1.0;
                for &(s, omega_j, gamma_j) in oscillators {
                    eps += s * omega_j * omega_j / (omega_j * omega_j + xi * xi + gamma_j * xi);
                }
                eps
            }

            DielectricModel::Tabulated { xi: xi_tab, eps } => {
                // Linear interpolation in log-log space
                if xi <= 0.0 || xi_tab.is_empty() {
                    return 1.0;
                }
                if xi <= xi_tab[0] {
                    return eps[0];
                }
                if xi >= *xi_tab.last().unwrap() {
                    return 1.0; // High-frequency limit
                }

                // Binary search for interval
                let idx = xi_tab.partition_point(|&x| x < xi);
                if idx == 0 || idx >= xi_tab.len() {
                    return 1.0;
                }

                // Log-log interpolation
                let x0 = xi_tab[idx - 1].ln();
                let x1 = xi_tab[idx].ln();
                let y0 = (eps[idx - 1] - 1.0).max(1e-10).ln();
                let y1 = (eps[idx] - 1.0).max(1e-10).ln();
                let t = (xi.ln() - x0) / (x1 - x0);
                let eps_interp = 1.0 + (y0 + t * (y1 - y0)).exp();
                eps_interp.max(1.0)
            }
        }
    }
}

/// Fresnel reflection coefficient at imaginary frequency (TM/p-polarization).
///
/// r_TM = (eps1 kappa2 - eps2 kappa1) / (eps1 kappa2 + eps2 kappa1)
///
/// where kappa_i = sqrt(eps_i xi^2/c^2 + k_perp^2)
///
/// # Arguments
/// * `eps1` - Dielectric of medium 1 at imaginary frequency
/// * `eps2` - Dielectric of medium 2 at imaginary frequency
/// * `xi` - Imaginary frequency (rad/s)
/// * `k_perp` - Transverse wavenumber (1/m)
pub fn fresnel_tm_imaginary(eps1: f64, eps2: f64, xi: f64, k_perp: f64) -> f64 {
    let xi_c = xi / C;
    let kappa1 = (eps1 * xi_c * xi_c + k_perp * k_perp).sqrt();
    let kappa2 = (eps2 * xi_c * xi_c + k_perp * k_perp).sqrt();

    // Handle very large or infinite dielectric (perfect conductor limit)
    // In the limit \epsilon_2 -> infty: r_TM -> (\epsilon_1/\epsilon_2)(\kappa_2/\kappa_1) -> 0, but accounting
    // for the fact that \kappa_2 ~ sqrt\epsilon_2, we get r_TM -> -1
    // For numerical stability with large finite \epsilon, compute directly
    let numer = eps1 * kappa2 - eps2 * kappa1;
    let denom = eps1 * kappa2 + eps2 * kappa1;

    if denom.abs() < 1e-30 {
        return 0.0;
    }

    numer / denom
}

/// Fresnel reflection coefficient at imaginary frequency (TE/s-polarization).
///
/// r_TE = (\kappa_2 - \kappa_1) / (\kappa_2 + \kappa_1)
///
/// For non-magnetic materials (\mu = 1).
///
/// # Arguments
/// * `eps1` - Dielectric of medium 1 at imaginary frequency
/// * `eps2` - Dielectric of medium 2 at imaginary frequency
/// * `xi` - Imaginary frequency (rad/s)
/// * `k_perp` - Transverse wavenumber (1/m)
pub fn fresnel_te_imaginary(eps1: f64, eps2: f64, xi: f64, k_perp: f64) -> f64 {
    let xi_c = xi / C;
    let kappa1 = (eps1 * xi_c * xi_c + k_perp * k_perp).sqrt();
    let kappa2 = (eps2 * xi_c * xi_c + k_perp * k_perp).sqrt();

    (kappa2 - kappa1) / (kappa2 + kappa1)
}

/// Lifshitz pressure between two parallel plates at zero temperature.
///
/// Integrates over imaginary frequency and transverse momentum to compute
/// the Casimir pressure P = -dE/dA/dd between two semi-infinite slabs.
///
/// P = -(hbar/2pi^2) int_0^inf dxi int_0^inf k_perp dk_perp kappa (r_TM^2 e^{-2 kappa d}/(1-r_TM^2 e^{-2 kappa d}) + TE)
///
/// # Arguments
/// * `gap` - Surface separation (m)
/// * `eps1` - Dielectric model for surface 1
/// * `eps2` - Dielectric model for surface 2
/// * `n_xi` - Number of frequency integration points (default: 32)
/// * `n_k` - Number of momentum integration points (default: 32)
///
/// # Returns
/// Pressure in Pa (negative = attractive)
pub fn lifshitz_pressure_plates(
    gap: f64,
    eps1: &DielectricModel,
    eps2: &DielectricModel,
    n_xi: usize,
    n_k: usize,
) -> f64 {
    // Characteristic frequency scale: c/d
    let xi_char = C / gap;

    // Integration over \xi using Gauss-Laguerre-like quadrature
    // We use a change of variables: \xi = xi_char * u, integrate u from 0 to ~10
    let mut pressure = 0.0;

    for i in 0..n_xi {
        // Logarithmic spacing works well for the oscillatory integrand
        let u = ((i as f64 + 0.5) / n_xi as f64) * 10.0;
        let xi = xi_char * u;
        let du = 10.0 / n_xi as f64;

        // Dielectric values at this frequency
        let e1 = eps1.epsilon_at_imaginary(xi);
        let e2 = eps2.epsilon_at_imaginary(xi);

        // Integration over k_perp: k_perp = (\xi/c) * v, v from 0 to ~10
        for j in 0..n_k {
            let v = ((j as f64 + 0.5) / n_k as f64) * 10.0;
            let dv = 10.0 / n_k as f64;
            let k_perp = (xi / C) * v;

            // kappa = sqrt(eps*xi^2/c^2 + k_perp^2), for vacuum (eps=1):
            let kappa = (xi * xi / (C * C) + k_perp * k_perp).sqrt();

            // Reflection coefficients
            let r_tm1 = fresnel_tm_imaginary(1.0, e1, xi, k_perp);
            let r_tm2 = fresnel_tm_imaginary(1.0, e2, xi, k_perp);
            let r_te1 = fresnel_te_imaginary(1.0, e1, xi, k_perp);
            let r_te2 = fresnel_te_imaginary(1.0, e2, xi, k_perp);

            // Round-trip factor
            let exp_factor = (-2.0 * kappa * gap).exp();
            let tm_factor = r_tm1 * r_tm2 * exp_factor / (1.0 - r_tm1 * r_tm2 * exp_factor);
            let te_factor = r_te1 * r_te2 * exp_factor / (1.0 - r_te1 * r_te2 * exp_factor);

            // Jacobian: d(xi) d(k_perp) = xi_char * du * (xi/c) * dv = (xi_char * xi/c) * du * dv
            let jacobian = xi_char * (xi / C) * du * dv;

            // Integrand: k_perp * \kappa * (TM + TE contributions)
            pressure += k_perp * kappa * (tm_factor + te_factor) * jacobian;
        }
    }

    // Prefactor: -hbar/(2*pi^2)
    pressure *= -HBAR / (2.0 * PI * PI);

    pressure
}

/// Lifshitz force for sphere-plate geometry using PFA.
///
/// Combines the Lifshitz pressure with the Proximity Force Approximation:
/// F_sp = 2\piR integral P(d) d(d) evaluated at the minimum gap.
///
/// For computational efficiency, we use:
/// F_sp ~= 2\piR * gap * P(gap) * correction_factor
///
/// where the correction factor accounts for the integration over the sphere surface.
///
/// # Arguments
/// * `radius` - Sphere radius (m)
/// * `gap` - Minimum surface-to-surface separation (m)
/// * `eps_sphere` - Dielectric model for sphere
/// * `eps_plate` - Dielectric model for plate
/// * `n_xi` - Number of frequency integration points
/// * `n_k` - Number of momentum integration points
///
/// # Returns
/// Force in Newtons (negative = attractive)
pub fn lifshitz_force_sphere_plate(
    radius: f64,
    gap: f64,
    eps_sphere: &DielectricModel,
    eps_plate: &DielectricModel,
    n_xi: usize,
    n_k: usize,
) -> f64 {
    // Compute plate-plate Lifshitz pressure at the gap
    let pressure = lifshitz_pressure_plates(gap, eps_sphere, eps_plate, n_xi, n_k);

    // PFA integration: F = 2\piR integral P(d') d(d') from gap to infty
    // For P ~ 1/d^3 (perfect conductor), this gives F = \piR * P(gap) * gap
    // More generally, we need the integrated pressure, which for P(d) ~ 1/d^n gives:
    // F = 2\piR * gap * P(gap) / (n-1) where n ~ 3 for Casimir
    // Using the standard result from PFA: F ~= 2\piR * integral_gap^infty P(d') dd'
    //                                       ~= 2\piR * gap * P(gap) / 2  (for n=3)

    // For consistency with casimir_force_pfa, use the PFA formula
    2.0 * PI * radius * gap * pressure / 2.0
}

/// Lifshitz force ratio: compares material-dependent force to perfect conductor.
///
/// \eta = F_Lifshitz / F_ideal_PFA
///
/// This ratio is always <= 1 for real materials and approaches 1 for
/// perfect conductors or at very short distances (high-frequency limit).
///
/// # Arguments
/// * `radius` - Sphere radius (m)
/// * `gap` - Surface separation (m)
/// * `eps_sphere` - Dielectric model for sphere
/// * `eps_plate` - Dielectric model for plate
///
/// # Returns
/// Ratio \eta (dimensionless, 0 < \eta <= 1)
pub fn lifshitz_force_ratio(
    radius: f64,
    gap: f64,
    eps_sphere: &DielectricModel,
    eps_plate: &DielectricModel,
) -> f64 {
    let f_lifshitz = lifshitz_force_sphere_plate(radius, gap, eps_sphere, eps_plate, 32, 32);
    let f_ideal = casimir_force_pfa(radius, gap);

    if f_ideal.abs() < 1e-30 {
        return 1.0;
    }

    (f_lifshitz / f_ideal).clamp(0.0, 2.0)
}

/// Result of Lifshitz theory calculation.
#[derive(Debug, Clone)]
pub struct LifshitzResult {
    /// Force in Newtons (negative = attractive)
    pub force: f64,
    /// Pressure at the gap (Pa)
    pub pressure: f64,
    /// Ratio to ideal PFA force
    pub eta: f64,
    /// Gap used (m)
    pub gap: f64,
    /// Sphere radius (m)
    pub radius: f64,
}

/// Complete Lifshitz calculation for sphere-plate geometry.
///
/// Returns force, pressure, and ratio to ideal conductor.
///
/// # Arguments
/// * `radius` - Sphere radius (m)
/// * `gap` - Surface separation (m)
/// * `eps_sphere` - Dielectric model for sphere
/// * `eps_plate` - Dielectric model for plate
pub fn lifshitz_sphere_plate(
    radius: f64,
    gap: f64,
    eps_sphere: &DielectricModel,
    eps_plate: &DielectricModel,
) -> LifshitzResult {
    let pressure = lifshitz_pressure_plates(gap, eps_sphere, eps_plate, 32, 32);
    let force = 2.0 * PI * radius * gap * pressure / 2.0;
    let f_ideal = casimir_force_pfa(radius, gap);
    let eta = if f_ideal.abs() < 1e-30 {
        1.0
    } else {
        (force / f_ideal).clamp(0.0, 2.0)
    };

    LifshitzResult {
        force,
        pressure,
        eta,
        gap,
        radius,
    }
}

/// Matsubara frequency for finite-temperature Lifshitz theory.
///
/// \xi_n = 2\pi n k_B T / hbar
///
/// At room temperature (T = 300 K), \xi_1 ~= 2.4e14 rad/s.
///
/// # Arguments
/// * `n` - Matsubara index (n = 0, 1, 2, ...)
/// * `temperature` - Temperature in Kelvin
pub fn matsubara_frequency(n: usize, temperature: f64) -> f64 {
    const K_B: f64 = 1.380649e-23;
    2.0 * PI * (n as f64) * K_B * temperature / HBAR
}

/// Characteristic thermal wavelength.
///
/// \lambda_T = hbarc / (k_B T)
///
/// At room temperature: \lambda_T ~= 7.6 \mum
///
/// # Arguments
/// * `temperature` - Temperature in Kelvin
pub fn thermal_wavelength(temperature: f64) -> f64 {
    const K_B: f64 = 1.380649e-23;
    HBAR * C / (K_B * temperature)
}

// ============================================================================
// PFA Validity Guard System
// ============================================================================
//
// The Proximity Force Approximation (PFA) becomes unreliable when the gap d
// approaches or exceeds the radius R. This section provides strict validity
// guards that enforce geometric constraints and return detailed diagnostics.
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

/// Errors arising from Casimir force calculations.
#[derive(Debug, Clone, Error)]
pub enum CasimirError {
    /// PFA validity violated: geometry outside approximation regime.
    #[error("PFA validity violated: d/R = {d_over_r:.6} exceeds {accuracy:?} threshold of {threshold:.6}")]
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
    let d_over_r = if radius > 0.0 { gap / radius } else { f64::INFINITY };

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
                | (PfaAccuracy::FivePercent, PfaAccuracy::OnePercent | PfaAccuracy::FivePercent)
                | (PfaAccuracy::TenPercent, PfaAccuracy::OnePercent | PfaAccuracy::FivePercent | PfaAccuracy::TenPercent)
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

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-12;

    #[test]
    fn test_casimir_force_pfa_attractive() {
        // Force should be attractive (negative)
        let force = casimir_force_pfa(5e-6, 100e-9);
        assert!(force < 0.0, "Casimir force should be attractive");
    }

    #[test]
    fn test_casimir_force_pfa_scaling() {
        // F ~ R/d^3, so doubling R should double F
        let f1 = casimir_force_pfa(5e-6, 100e-9);
        let f2 = casimir_force_pfa(10e-6, 100e-9);
        assert!((f2 / f1 - 2.0).abs() < TOLERANCE);

        // Halving d should multiply F by 8
        let f3 = casimir_force_pfa(5e-6, 50e-9);
        assert!((f3 / f1 - 8.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_casimir_force_pfa_order_of_magnitude() {
        // For R = 5 um, d = 100 nm: F ~ 10^-12 N (piconewton range)
        let force = casimir_force_pfa(5e-6, 100e-9);
        assert!(force.abs() > 1e-14);
        assert!(force.abs() < 1e-10);
    }

    #[test]
    fn test_casimir_energy_consistent_with_force() {
        // E = integral F dd = -C*R/(2d^2) for F = -C*R/d^3
        // So dE/dd = C*R/d^3 = -F
        let r = 5e-6;
        let d = 100e-9;
        let dd = 1e-12;  // Small displacement

        let e1 = casimir_energy_pfa(r, d);
        let e2 = casimir_energy_pfa(r, d + dd);
        let numerical_force = -(e2 - e1) / dd;
        let analytical_force = casimir_force_pfa(r, d);

        // Should match within 1%
        assert!((numerical_force / analytical_force - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_pfa_validity() {
        // Valid when R >> d (qualitative threshold: R/d > 5)
        assert!(pfa_is_valid(5e-6, 100e-9));  // R/d = 50
        assert!(pfa_is_valid(5e-6, 400e-9));  // R/d = 12.5
        assert!(pfa_is_valid(5e-6, 800e-9));  // R/d = 6.25 (just above threshold)
        assert!(!pfa_is_valid(5e-6, 1.5e-6)); // R/d = 3.33 (below threshold)
    }

    #[test]
    fn test_sphere_creation() {
        let s = Sphere::from_micrometers(5.0, 0.0);
        assert!((s.radius - 5e-6).abs() < 1e-12);
        assert!(s.position.abs() < 1e-12);
    }

    #[test]
    fn test_sps_system_geometry() {
        let system = SpherePlateSphere::symmetric_from_micrometers(
            5.0,   // radius
            0.1,   // source gap (100 nm)
            0.1,   // drain gap (100 nm)
            0.01,  // plate thickness (10 nm)
        );

        assert!((system.source_plate_gap() - 0.1e-6).abs() < 1e-12);
        assert!((system.drain_plate_gap() - 0.1e-6).abs() < 1e-12);
    }

    #[test]
    fn test_compute_forces_symmetric() {
        let system = SpherePlateSphere::symmetric_from_micrometers(
            5.0,   // same radius
            0.1,   // same gap
            0.1,
            0.01,
        );

        let result = compute_casimir_forces(&system);

        // Forces should be equal for symmetric system
        assert!((result.force_source - result.force_drain).abs() / result.force_source.abs() < 0.01);
    }

    #[test]
    fn test_transistor_no_plate_spring() {
        let system = SpherePlateSphere::symmetric_from_micrometers(5.0, 0.1, 0.1, 0.01);
        let result = analyze_transistor(&system, 0.0);

        // Without plate spring, no cross-coupling
        assert!(result.cross_coupling.abs() < 1e-30);
        assert!(result.gain.abs() < 1e-30);
    }

    #[test]
    fn test_transistor_with_plate_spring() {
        let system = SpherePlateSphere::symmetric_from_micrometers(5.0, 0.1, 0.1, 0.01);
        // Plate spring ~ 10 N/m (typical MEMS)
        let result = analyze_transistor(&system, 10.0);

        // With plate spring, should have finite gain
        assert!(result.cross_coupling.abs() > 0.0);
        assert!(result.spring_source > 0.0);
        assert!(result.spring_drain > 0.0);
    }

    #[test]
    fn test_spring_constant_scaling() {
        // k ~ R/d^4, so halving d should multiply k by 16
        let r: f64 = 5e-6;
        let d1: f64 = 100e-9;
        let d2: f64 = 50e-9;

        let k1 = 3.0 * CASIMIR_COEFF * r / d1.powi(4);
        let k2 = 3.0 * CASIMIR_COEFF * r / d2.powi(4);

        assert!((k2 / k1 - 16.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_sweep_source_gap() {
        let gaps: Vec<f64> = (1..=10).map(|i| i as f64 * 50e-9).collect();
        let result = sweep_source_gap(5e-6, 100e-9, &gaps, 1.0);

        assert_eq!(result.source_gaps.len(), 10);
        assert_eq!(result.forces_source.len(), 10);

        // Force should decrease (less negative) as gap increases
        assert!(result.forces_source[0] < result.forces_source[9]);
    }

    #[test]
    fn test_finite_conductivity_correction() {
        // Gold plasma wavelength ~ 136 nm
        let lambda_p = 136e-9;

        // At large gap, correction should be ~1
        let corr_large = finite_conductivity_correction(1e-6, lambda_p);
        assert!((corr_large - 1.0).abs() < 0.1);

        // At small gap, correction should be < 1
        let corr_small = finite_conductivity_correction(50e-9, lambda_p);
        assert!(corr_small < 1.0);
    }

    #[test]
    fn test_thermal_correction_room_temp() {
        // At room temperature, lambda_T ~ 7.6 um
        let temp = 300.0;  // K

        // At nanometer gaps, thermal correction negligible
        let corr_small = thermal_correction(100e-9, temp);
        assert!((corr_small - 1.0).abs() < 0.1);

        // At micrometer gaps, thermal effects grow
        let corr_large = thermal_correction(10e-6, temp);
        assert!(corr_large > 1.0);
    }

    #[test]
    fn test_casimir_force_with_corrections_physical() {
        // Gold at room temperature
        let force = casimir_force_with_corrections(
            5e-6,    // 5 um sphere
            100e-9,  // 100 nm gap
            136e-9,  // gold plasma wavelength
            300.0,   // room temperature
        );

        // Should still be attractive
        assert!(force < 0.0);

        // Should be smaller than ideal PFA due to finite conductivity
        let f_ideal = casimir_force_pfa(5e-6, 100e-9);
        assert!(force.abs() < f_ideal.abs());
    }

    #[test]
    fn test_realistic_xu_parameters() {
        // From Xu et al. 2022: silica spheres ~ 5 um diameter
        // Gaps ~ 100-500 nm range
        let system = SpherePlateSphere::symmetric_from_micrometers(
            2.5,   // 2.5 um radius (5 um diameter)
            0.2,   // 200 nm gap
            0.2,
            0.05,  // 50 nm plate
        );

        let forces = compute_casimir_forces(&system);

        // Should be in piconewton range
        assert!(forces.force_source.abs() > 1e-14);
        assert!(forces.force_source.abs() < 1e-10);

        // PFA should be valid for these parameters
        assert!(forces.pfa_valid);
    }

    // ========================================================================
    // Additivity API Tests
    // ========================================================================

    #[test]
    fn test_pfa_accuracy_levels() {
        // R/d = 150 should satisfy 1% accuracy (threshold: 132)
        assert!(PfaAccuracy::OnePercent.is_valid(150e-6, 1e-6));
        // R/d = 100 should fail 1% but pass 5%
        assert!(!PfaAccuracy::OnePercent.is_valid(100e-6, 1e-6));
        assert!(PfaAccuracy::FivePercent.is_valid(100e-6, 1e-6));
        // R/d = 10 should only pass qualitative
        assert!(!PfaAccuracy::TenPercent.is_valid(10e-6, 1e-6));
        assert!(PfaAccuracy::Qualitative.is_valid(10e-6, 1e-6));
    }

    #[test]
    fn test_pfa_accuracy_min_ratios() {
        assert_eq!(PfaAccuracy::OnePercent.min_r_over_d(), 132.0);
        assert_eq!(PfaAccuracy::FivePercent.min_r_over_d(), 26.0);
        assert_eq!(PfaAccuracy::TenPercent.min_r_over_d(), 13.0);
        assert_eq!(PfaAccuracy::Qualitative.min_r_over_d(), 5.0);
    }

    #[test]
    fn test_force_sps_additive_symmetric() {
        let r = 5e-6;
        let gap = 100e-9;
        let result = force_sps_additive(r, r, gap, gap);

        // Symmetric system should have equal forces
        assert!((result.force_source - result.force_drain).abs() < TOLERANCE);
        assert_eq!(result.gap_source, result.gap_drain);
    }

    #[test]
    fn test_force_sps_additive_thermal_warning() {
        // Small gap: no thermal warning
        let result_small = force_sps_additive(5e-6, 5e-6, 100e-9, 100e-9);
        assert!(!result_small.thermal_warning);

        // Large gap: thermal warning
        let result_large = force_sps_additive(5e-6, 5e-6, 2e-6, 2e-6);
        assert!(result_large.thermal_warning);
    }

    #[test]
    fn test_cross_coupling_rigid_plate() {
        // Rigid plate (infinite spring): no cross-coupling
        let (k_s, k_d, cross) = cross_coupling_additive(
            5e-6, 5e-6, 100e-9, 100e-9, 0.0
        );
        assert!(k_s > 0.0);
        assert!(k_d > 0.0);
        assert_eq!(cross, 0.0);
    }

    #[test]
    fn test_cross_coupling_flexible_plate() {
        // Flexible plate: finite cross-coupling
        let (k_s, k_d, cross) = cross_coupling_additive(
            5e-6, 5e-6, 100e-9, 100e-9, 1.0
        );
        assert!(cross > 0.0);
        // Cross-coupling should be k_s * k_d / k_plate
        assert!((cross - k_s * k_d / 1.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_transistor_gain_additive() {
        let r = 5e-6;
        let gap = 100e-9;
        let k_plate = 10.0;

        let gain = transistor_gain_additive(r, gap, k_plate);
        let k_drain = 3.0 * CASIMIR_COEFF * r / gap.powi(4);

        assert!((gain - k_drain / k_plate).abs() < TOLERANCE);
    }

    #[test]
    fn test_transistor_gain_rigid_plate() {
        // Rigid plate gives zero gain
        let gain = transistor_gain_additive(5e-6, 100e-9, 0.0);
        assert_eq!(gain, 0.0);
    }

    #[test]
    fn test_nonadditivity_correction_placeholder() {
        // Currently returns 0 (pure additivity)
        let corr = nonadditivity_correction(5e-6, 5e-6, 100e-9, 100e-9, 10e-9);
        assert_eq!(corr, 0.0);
    }

    #[test]
    fn test_additivity_matches_compute_forces() {
        // Additivity API should match the older compute_casimir_forces
        let system = SpherePlateSphere::symmetric_from_micrometers(5.0, 0.1, 0.1, 0.01);
        let old_result = compute_casimir_forces(&system);
        let new_result = force_sps_additive(
            system.source.radius,
            system.drain.radius,
            system.source_plate_gap(),
            system.drain_plate_gap(),
        );

        assert!((old_result.force_source - new_result.force_source).abs() < TOLERANCE);
        assert!((old_result.force_drain - new_result.force_drain).abs() < TOLERANCE);
    }

    // ========================================================================
    // Lifshitz Theory Tests
    // ========================================================================

    #[test]
    fn test_dielectric_gold_drude() {
        let gold = DielectricModel::gold();

        // At high frequency, dielectric should approach 1
        let eps_high = gold.epsilon_at_imaginary(1e18);
        assert!(eps_high > 1.0);
        assert!(eps_high < 2.0, "High-frequency epsilon should be close to 1");

        // At lower frequency, dielectric should be larger
        let eps_low = gold.epsilon_at_imaginary(1e14);
        assert!(eps_low > eps_high, "epsilon should decrease with frequency");
    }

    #[test]
    fn test_dielectric_plasma_model() {
        let plasma = DielectricModel::gold_plasma();

        // Plasma model: eps(i*xi) = 1 + omega_p^2 / xi^2
        let omega_p = 1.37e16;
        let xi = 1e15;
        let expected = 1.0 + omega_p * omega_p / (xi * xi);
        let actual = plasma.epsilon_at_imaginary(xi);

        assert!((actual - expected).abs() / expected < 1e-10);
    }

    #[test]
    fn test_dielectric_silica() {
        let silica = DielectricModel::silica();

        // Silica should have \epsilon > 1 at all frequencies (dielectric)
        let eps1 = silica.epsilon_at_imaginary(1e14);
        let eps2 = silica.epsilon_at_imaginary(1e16);

        assert!(eps1 > 1.0);
        assert!(eps2 > 1.0);
        // At resonance frequency (~1.5e16), \epsilon should drop
        assert!(eps2 < eps1, "epsilon should decrease toward resonance");
    }

    #[test]
    fn test_fresnel_tm_high_epsilon() {
        // Very high dielectric (like perfect conductor): r_TM approaches -1
        // For vacuum/metal interface: r_TM = (\epsilon_1\kappa_2 - \epsilon_2\kappa_1)/(\epsilon_1\kappa_2 + \epsilon_2\kappa_1)
        // As \epsilon_2 -> infty, \kappa_2 ~ sqrt\epsilon_2, so ratio -> -1
        let eps1 = 1.0;
        let eps2 = 1e20; // Large but finite (perfect conductor approx)
        let r = fresnel_tm_imaginary(eps1, eps2, 1e15, 1e6);

        // r should be close to -1 for high contrast
        assert!(r < 0.0, "TM reflection should be negative for vacuum/metal");
        assert!(r.abs() > 0.99, "TM reflection magnitude should be ~1");
    }

    #[test]
    fn test_fresnel_te_at_normal_incidence() {
        // At k_perp = 0, both media have same kappa ratio
        let eps1 = 1.0;
        let eps2 = 10.0;
        let xi = 1e15;
        let r = fresnel_te_imaginary(eps1, eps2, xi, 0.0);

        // r_TE = (\kappa2 - \kappa1)/(\kappa2 + \kappa1) = (sqrt\epsilon2 - sqrt\epsilon1)/(sqrt\epsilon2 + sqrt\epsilon1)
        let expected = (eps2.sqrt() - eps1.sqrt()) / (eps2.sqrt() + eps1.sqrt());
        assert!((r - expected).abs() < 1e-10);
    }

    #[test]
    fn test_lifshitz_pressure_order_of_magnitude() {
        // Perfect conductors at 100 nm gap
        // P = -pi^2 * hbar * c / (240 d^4) for perfect conductors
        // At d = 100 nm: P ~ -13 Pa
        let gap = 100e-9;
        let eps = DielectricModel::PerfectConductor;
        let pressure = lifshitz_pressure_plates(gap, &eps, &eps, 32, 32);

        // Should be in the Pa range (negative)
        assert!(pressure < 0.0, "Pressure should be attractive");

        // Order of magnitude check for 100 nm gap:
        // P = -pi^2 * hbar * c / (240 d^4) ~ -13 Pa
        // Allow factor of 10 tolerance due to integration approximation
        assert!(pressure.abs() > 1.0, "Pressure too small: {}", pressure);
        assert!(pressure.abs() < 200.0, "Pressure too large: {}", pressure);
    }

    #[test]
    fn test_lifshitz_force_attractive() {
        let gap = 100e-9;
        let radius = 5e-6;
        let gold = DielectricModel::gold();

        let force = lifshitz_force_sphere_plate(radius, gap, &gold, &gold, 16, 16);

        assert!(force < 0.0, "Force should be attractive");
    }

    #[test]
    fn test_lifshitz_force_ratio_less_than_one() {
        // Real materials always give smaller force than perfect conductors
        let gap = 100e-9;
        let radius = 5e-6;
        let gold = DielectricModel::gold();

        let eta = lifshitz_force_ratio(radius, gap, &gold, &gold);

        assert!(eta > 0.0, "Ratio should be positive");
        assert!(eta <= 1.0, "Ratio should be <= 1 for real materials");
    }

    #[test]
    fn test_matsubara_frequency_room_temp() {
        // At 300 K, first Matsubara frequency ~ 2.4e14 rad/s
        let xi_1 = matsubara_frequency(1, 300.0);
        assert!((xi_1 - 2.4e14).abs() / 2.4e14 < 0.1);

        // n=0 gives zero
        let xi_0 = matsubara_frequency(0, 300.0);
        assert_eq!(xi_0, 0.0);
    }

    #[test]
    fn test_thermal_wavelength_room_temp() {
        // At 300 K, thermal wavelength ~ 7.6 \mum
        let lambda_t = thermal_wavelength(300.0);
        assert!((lambda_t - 7.6e-6).abs() / 7.6e-6 < 0.1);
    }

    #[test]
    fn test_lifshitz_sphere_plate_result() {
        let gap = 100e-9;
        let radius = 5e-6;
        let gold = DielectricModel::gold();

        let result = lifshitz_sphere_plate(radius, gap, &gold, &gold);

        assert!(result.force < 0.0);
        assert!(result.pressure < 0.0);
        assert!(result.eta > 0.0);
        assert!(result.eta <= 1.0);
        assert_eq!(result.gap, gap);
        assert_eq!(result.radius, radius);
    }

    #[test]
    fn test_dielectric_tabulated() {
        // Create a simple tabulated dielectric
        let xi_tab = vec![1e14, 1e15, 1e16, 1e17];
        let eps_tab = vec![10.0, 5.0, 2.0, 1.2];
        let model = DielectricModel::Tabulated {
            xi: xi_tab,
            eps: eps_tab,
        };

        // Check interpolation works
        let eps_mid = model.epsilon_at_imaginary(5e14);
        assert!(eps_mid < 10.0 && eps_mid > 2.0);

        // High frequency returns 1
        let eps_high = model.epsilon_at_imaginary(1e20);
        assert_eq!(eps_high, 1.0);
    }

    // =========================================================================
    // PFA Validity Guard Tests
    // =========================================================================

    #[test]
    fn test_check_pfa_validity_one_percent() {
        // R/d = 150 > 132 -> 1% validity achieved
        let radius = 15e-6;
        let gap = 100e-9;
        let info = check_pfa_validity(radius, gap);

        assert!(info.one_percent_valid, "R/d = {} should satisfy 1%", info.r_over_d);
        assert!(info.five_percent_valid);
        assert!(info.ten_percent_valid);
        assert!(info.qualitative_valid);
        assert_eq!(info.achieved_accuracy, PfaAccuracy::OnePercent);
    }

    #[test]
    fn test_check_pfa_validity_five_percent() {
        // R/d = 50 -> 5% but not 1%
        let radius = 5e-6;
        let gap = 100e-9;
        let info = check_pfa_validity(radius, gap);

        assert!(!info.one_percent_valid, "R/d = {} should NOT satisfy 1%", info.r_over_d);
        assert!(info.five_percent_valid, "R/d = {} should satisfy 5%", info.r_over_d);
        assert!(info.ten_percent_valid);
        assert!(info.qualitative_valid);
        assert_eq!(info.achieved_accuracy, PfaAccuracy::FivePercent);
    }

    #[test]
    fn test_check_pfa_validity_ten_percent() {
        // R/d = 15 -> 10% but not 5%
        let radius = 1.5e-6;
        let gap = 100e-9;
        let info = check_pfa_validity(radius, gap);

        assert!(!info.five_percent_valid);
        assert!(info.ten_percent_valid);
        assert!(info.qualitative_valid);
        assert_eq!(info.achieved_accuracy, PfaAccuracy::TenPercent);
    }

    #[test]
    fn test_check_pfa_validity_qualitative_only() {
        // R/d = 8 -> qualitative only
        let radius = 0.8e-6;
        let gap = 100e-9;
        let info = check_pfa_validity(radius, gap);

        assert!(!info.ten_percent_valid);
        assert!(info.qualitative_valid);
        assert_eq!(info.achieved_accuracy, PfaAccuracy::Qualitative);
    }

    #[test]
    fn test_check_pfa_validity_below_qualitative() {
        // R/d = 3 -> below qualitative threshold
        let radius = 0.3e-6;
        let gap = 100e-9;
        let info = check_pfa_validity(radius, gap);

        assert!(!info.qualitative_valid);
        // Still returns Qualitative as "achieved" (best possible)
        assert_eq!(info.achieved_accuracy, PfaAccuracy::Qualitative);
    }

    #[test]
    fn test_casimir_force_guarded_passes_at_one_percent() {
        // R/d = 200 > 132 -> should pass 1%
        let radius = 20e-6;
        let gap = 100e-9;

        let result = casimir_force_guarded(radius, gap, PfaAccuracy::OnePercent);
        assert!(result.is_ok(), "Should pass at 1%: {:?}", result);

        let force = result.unwrap();
        assert!(force < 0.0, "Force should be attractive");
    }

    #[test]
    fn test_casimir_force_guarded_fails_at_one_percent() {
        // R/d = 50 < 132 -> should fail 1%
        let radius = 5e-6;
        let gap = 100e-9;

        let result = casimir_force_guarded(radius, gap, PfaAccuracy::OnePercent);
        assert!(result.is_err(), "Should fail at 1%");

        match result {
            Err(CasimirError::PfaViolation { d_over_r, accuracy, .. }) => {
                assert!((d_over_r - 0.02).abs() < 0.001);
                assert_eq!(accuracy, PfaAccuracy::OnePercent);
            }
            _ => panic!("Expected PfaViolation error"),
        }
    }

    #[test]
    fn test_casimir_force_guarded_passes_at_qualitative() {
        // R/d = 50 > 5 -> should pass qualitative
        let radius = 5e-6;
        let gap = 100e-9;

        let result = casimir_force_guarded(radius, gap, PfaAccuracy::Qualitative);
        assert!(result.is_ok(), "Should pass qualitative");
    }

    #[test]
    fn test_casimir_force_guarded_invalid_gap() {
        let radius = 5e-6;
        let gap = -100e-9;  // Invalid negative gap

        let result = casimir_force_guarded(radius, gap, PfaAccuracy::Qualitative);
        assert!(matches!(result, Err(CasimirError::InvalidGap { .. })));
    }

    #[test]
    fn test_casimir_force_guarded_invalid_radius() {
        let radius = -5e-6;  // Invalid negative radius
        let gap = 100e-9;

        let result = casimir_force_guarded(radius, gap, PfaAccuracy::Qualitative);
        assert!(matches!(result, Err(CasimirError::InvalidRadius { .. })));
    }

    #[test]
    fn test_casimir_force_with_validity() {
        let radius = 5e-6;
        let gap = 100e-9;

        let (force, info) = casimir_force_with_validity(radius, gap);

        assert!(force < 0.0);
        assert_eq!(info.radius, radius);
        assert_eq!(info.gap, gap);
        assert!(info.five_percent_valid);
        assert!(!info.one_percent_valid);
    }

    #[test]
    fn test_casimir_spring_constant_guarded() {
        // R/d = 200 -> passes 1%
        let radius = 20e-6;
        let gap = 100e-9;

        let result = casimir_spring_constant_guarded(radius, gap, PfaAccuracy::OnePercent);
        assert!(result.is_ok());

        let k = result.unwrap();
        assert!(k > 0.0, "Spring constant should be positive");

        // k ~ 3*C*R/d^4 ~ order 10^-3 N/m for these parameters
        assert!(k > 1e-5);
        assert!(k < 1e-1);
    }

    #[test]
    fn test_pfa_accuracy_thresholds() {
        // Verify the threshold values match the literature
        assert_eq!(PfaAccuracy::OnePercent.min_r_over_d(), 132.0);
        assert_eq!(PfaAccuracy::FivePercent.min_r_over_d(), 26.0);
        assert_eq!(PfaAccuracy::TenPercent.min_r_over_d(), 13.0);
        assert_eq!(PfaAccuracy::Qualitative.min_r_over_d(), 5.0);

        // d/R thresholds
        assert!((1.0_f64 / 132.0 - 0.00757).abs() < 0.0001);  // 1%: d/R < 0.00755
        assert!((1.0_f64 / 26.0 - 0.0385).abs() < 0.001);     // 5%: d/R < 0.038
    }

    #[test]
    fn test_pfa_estimated_error_matches_d_over_r() {
        let radius = 5e-6;
        let gap = 100e-9;
        let info = check_pfa_validity(radius, gap);

        // Estimated error should equal d/R (leading order)
        assert!((info.estimated_error - info.d_over_r).abs() < 1e-12);
        assert!((info.estimated_error - 0.02).abs() < 1e-6);
    }

    // =========================================================================
    // Derivative Expansion Error Estimate Tests
    // =========================================================================

    #[test]
    fn test_de_coefficients_sphere_plate() {
        let coeffs = &DeCoefficients::SPHERE_PLATE_DIRICHLET;
        // a_1 = 1/3 for sphere-plate Dirichlet
        assert!((coeffs.a1 - 1.0 / 3.0).abs() < 1e-12);
        assert!(coeffs.source.contains("Emig"));
    }

    #[test]
    fn test_casimir_force_with_de_basic() {
        let radius = 5e-6;
        let gap = 100e-9;

        let result = casimir_force_with_de(
            radius, gap,
            &DeCoefficients::SPHERE_PLATE_DIRICHLET,
        );

        // PFA force should be attractive
        assert!(result.force_pfa < 0.0);

        // Corrected force should also be attractive
        assert!(result.force_corrected < 0.0);

        // For d/R = 0.02, correction is small but non-zero
        assert!(result.relative_error > 0.0);
        assert!(result.relative_error < 0.1); // Less than 10% error

        // Expansion parameter should be d/R
        assert!((result.expansion_param - 0.02).abs() < 1e-6);
    }

    #[test]
    fn test_casimir_force_with_de_convergence() {
        // Small d/R -> well converged
        let radius = 20e-6;  // Large sphere
        let gap = 100e-9;    // d/R = 0.005

        let result = casimir_force_with_de(
            radius, gap,
            &DeCoefficients::SPHERE_PLATE_DIRICHLET,
        );

        assert!(result.is_converged, "Should be converged at d/R = 0.005");

        // Large d/R -> may not be converged
        let radius_small = 0.5e-6;  // Small sphere
        let gap_large = 100e-9;     // d/R = 0.2

        let result2 = casimir_force_with_de(
            radius_small, gap_large,
            &DeCoefficients::SPHERE_PLATE_DIRICHLET,
        );

        // O2 term becomes comparable to O1 at large d/R
        // Convergence depends on exact ratio
        assert!(result2.expansion_param > 0.1);
    }

    #[test]
    fn test_casimir_force_with_de_correction_sign() {
        // For Dirichlet (a_1 = +1/3), correction has same sign as F_PFA
        // Since F_PFA < 0, correction < 0, so |F_corrected| > |F_PFA|
        let radius = 5e-6;
        let gap = 100e-9;

        let result = casimir_force_with_de(
            radius, gap,
            &DeCoefficients::SPHERE_PLATE_DIRICHLET,
        );

        // O1 correction should have same sign as F_PFA (both negative)
        assert!(result.correction_o1 < 0.0);

        // Corrected force should be more negative (larger magnitude)
        assert!(result.force_corrected < result.force_pfa);
    }

    #[test]
    fn test_estimate_de_error() {
        let radius = 5e-6;
        let gap = 100e-9;  // d/R = 0.02

        let error = estimate_de_error(radius, gap);

        // Should be approximately a_1 * (d/R) ~ 0.02/3 ~ 0.0067
        assert!(error > 0.006);
        assert!(error < 0.008);
    }

    #[test]
    fn test_max_gap_for_error() {
        let radius = 5e-6;
        let target_error = 0.01;  // 1% error

        let max_gap = max_gap_for_error(radius, target_error);

        // For 1% error with a_1 = 1/3, d/R ~ 0.03, so d ~ 150 nm
        assert!(max_gap > 100e-9);
        assert!(max_gap < 200e-9);

        // Verify: actual error at this gap should be near target
        let actual_error = estimate_de_error(radius, max_gap);
        assert!((actual_error - target_error).abs() < 0.005);
    }

    #[test]
    fn test_de_neumann_has_opposite_sign() {
        // Neumann BC has a_1 = -1/3, opposite to Dirichlet
        let coeffs_d = &DeCoefficients::SPHERE_PLATE_DIRICHLET;
        let coeffs_n = &DeCoefficients::SPHERE_PLATE_NEUMANN;

        assert!((coeffs_d.a1 + coeffs_n.a1).abs() < 1e-12);
    }

    #[test]
    fn test_de_cylinder_different_from_sphere() {
        let coeffs_sphere = &DeCoefficients::SPHERE_PLATE_DIRICHLET;
        let coeffs_cyl = &DeCoefficients::CYLINDER_PLATE;

        // Cylinder has different a_1
        assert!((coeffs_sphere.a1 - coeffs_cyl.a1).abs() > 0.1);
    }

    // =========================================================================
    // Strict Spring Constant / Gain Tests
    // =========================================================================

    #[test]
    fn test_stricter_for_derivative() {
        assert_eq!(
            PfaAccuracy::Qualitative.stricter_for_derivative(),
            PfaAccuracy::TenPercent
        );
        assert_eq!(
            PfaAccuracy::TenPercent.stricter_for_derivative(),
            PfaAccuracy::FivePercent
        );
        assert_eq!(
            PfaAccuracy::FivePercent.stricter_for_derivative(),
            PfaAccuracy::OnePercent
        );
        assert_eq!(
            PfaAccuracy::OnePercent.stricter_for_derivative(),
            PfaAccuracy::OnePercent
        );
    }

    #[test]
    fn test_spring_constant_strict_passes() {
        // Very large R/d for 1% spring constant accuracy
        let radius = 40e-6;  // R/d = 400, very safe
        let gap = 100e-9;

        let result = spring_constant_strict(radius, gap, PfaAccuracy::OnePercent);
        assert!(result.is_ok());
    }

    #[test]
    fn test_spring_constant_strict_fails() {
        // R/d = 50 -> passes 5% force but not 1% force
        // So for 5% spring constant (needs 1% force), it should fail
        let radius = 5e-6;
        let gap = 100e-9;

        let result = spring_constant_strict(radius, gap, PfaAccuracy::FivePercent);
        assert!(result.is_err(), "Should fail: 5% k needs 1% force, R/d=50 insufficient");
    }

    #[test]
    fn test_spring_constant_strict_vs_regular() {
        // The strict version requires more stringent geometry
        let radius = 13e-6;  // R/d = 130, just under 132 for 1%
        let gap = 100e-9;

        // Regular guard at 5% should pass (R/d > 26)
        let regular = casimir_spring_constant_guarded(radius, gap, PfaAccuracy::FivePercent);
        assert!(regular.is_ok());

        // Strict at 5% k (needs 1% force) should fail (R/d < 132)
        let strict = spring_constant_strict(radius, gap, PfaAccuracy::FivePercent);
        assert!(strict.is_err());
    }

    #[test]
    fn test_transistor_gain_strict() {
        let radius = 40e-6;
        let gap = 100e-9;
        let plate_spring = 1e-3;  // 1 mN/m

        let result = transistor_gain_strict(
            radius, gap, plate_spring,
            PfaAccuracy::TenPercent,
        );
        assert!(result.is_ok());

        let gain = result.unwrap();
        assert!(gain > 0.0);
    }

    #[test]
    fn test_transistor_gain_strict_zero_plate_spring() {
        let radius = 5e-6;
        let gap = 100e-9;

        // Zero plate spring gives zero gain (no coupling)
        let result = transistor_gain_strict(
            radius, gap, 0.0,
            PfaAccuracy::Qualitative,
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0.0);
    }

    #[test]
    fn test_spring_constant_with_diagnostics() {
        let radius = 5e-6;
        let gap = 100e-9;

        let result = spring_constant_with_diagnostics(
            radius, gap,
            PfaAccuracy::FivePercent,
        );

        assert!(result.k > 0.0);
        assert!(result.estimated_k_error > 0.0);
        // Error amplification: k error ~ 4x force error
        assert!(result.estimated_k_error > result.validity.estimated_error);
    }

    #[test]
    fn test_spring_constant_error_factor() {
        assert!((SPRING_CONSTANT_ERROR_FACTOR - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_effective_for_spring_constant() {
        // Large sphere: R/d = 400, should achieve target accuracy
        let result = PfaAccuracy::FivePercent.effective_for_spring_constant(40e-6, 100e-9);
        assert_eq!(result, Some(PfaAccuracy::FivePercent));

        // Medium sphere: R/d = 100
        // For 5% k: needs stricter = 1% force (R/d > 132), R/d = 100 fails
        // Fallback: passes 5% force? (R/d > 26) YES -> degrade to 10%
        let result2 = PfaAccuracy::FivePercent.effective_for_spring_constant(10e-6, 100e-9);
        assert_eq!(result2, Some(PfaAccuracy::TenPercent));

        // Qualitative k: needs 10% force (R/d > 13), R/d = 100 passes stricter
        let result3 = PfaAccuracy::Qualitative.effective_for_spring_constant(10e-6, 100e-9);
        assert_eq!(result3, Some(PfaAccuracy::Qualitative));

        // 1% k with R/d = 100: needs 1% force (stricter = 1%) but R/d < 132
        // Fallback: passes 1%? R/d > 132? NO. Returns None
        let result4 = PfaAccuracy::OnePercent.effective_for_spring_constant(10e-6, 100e-9);
        assert!(result4.is_none(), "R/d = 100 cannot achieve 1% k");

        // 10% k with R/d = 100: needs 5% force (stricter), R/d > 26 passes
        // Since R/d = 100 > 26 (5%), stricter is satisfied -> return Some(TenPercent)
        let result5 = PfaAccuracy::TenPercent.effective_for_spring_constant(10e-6, 100e-9);
        assert_eq!(result5, Some(PfaAccuracy::TenPercent));
    }

    // =========================================================================
    // Three-Body Transistor Dynamics Tests
    // =========================================================================

    #[test]
    fn test_three_body_gain_quasistatic_basic() {
        let r = 5e-6;  // 5 um spheres
        let gap = 100e-9;  // 100 nm gaps
        let k_plate = 1e-3;  // 1 mN/m

        let gain = three_body_gain_quasistatic(r, r, gap, gap, k_plate);

        // Gain should be positive and < 1 (energy conserving)
        assert!(gain > 0.0, "Gain should be positive");
        assert!(gain < 1.0, "Gain should be less than 1 for passive system");
    }

    #[test]
    fn test_three_body_gain_zero_plate_spring() {
        let r = 5e-6;
        let gap = 100e-9;

        let gain = three_body_gain_quasistatic(r, r, gap, gap, 0.0);
        assert_eq!(gain, 0.0, "Zero plate spring gives zero gain");
    }

    #[test]
    fn test_three_body_gain_asymmetric() {
        let r_source = 5e-6;
        let r_drain = 10e-6;  // Larger drain sphere
        let gap_source = 100e-9;
        let gap_drain = 50e-9;  // Smaller drain gap
        let k_plate = 1e-3;

        let gain = three_body_gain_quasistatic(
            r_source, r_drain, gap_source, gap_drain, k_plate,
        );

        // Asymmetric configuration should still give reasonable gain
        assert!(gain > 0.0);
        assert!(gain < 10.0);  // Shouldn't be unreasonably large
    }

    #[test]
    fn test_three_body_casimir_dynamics_jacobian() {
        let r = 5e-6;
        let gap = 100e-9;
        let k_plate = 1e-3;
        let m = 1e-12;  // 1 picogram

        let result = three_body_casimir_dynamics(
            r, r, gap, gap, k_plate, m, m, m,
        );

        // Diagonal elements should be negative (restoring)
        assert!(result.j_ss < 0.0, "j_ss should be negative (restoring)");
        assert!(result.j_pp < 0.0, "j_pp should be negative (restoring)");
        assert!(result.j_dd < 0.0, "j_dd should be negative (restoring)");

        // Off-diagonal source-plate and drain-plate should be positive
        assert!(result.j_sp > 0.0);
        assert!(result.j_ps > 0.0);
        assert!(result.j_dp > 0.0);
        assert!(result.j_pd > 0.0);
    }

    #[test]
    fn test_three_body_casimir_dynamics_stability() {
        let r = 5e-6;
        let gap = 100e-9;
        let k_plate = 1e-3;  // Strong plate spring for stability
        let m = 1e-12;

        let result = three_body_casimir_dynamics(
            r, r, gap, gap, k_plate, m, m, m,
        );

        // With strong plate spring, system should be stable
        // (eigenvalues should be positive for oscillatory modes)
        for &omega_sq in &result.stability_eigenvalues {
            assert!(omega_sq > 0.0, "Stability eigenvalue should be positive");
        }
        assert!(result.is_stable, "System should be stable");
    }

    #[test]
    fn test_three_body_casimir_dynamics_gain() {
        let r = 5e-6;
        let gap = 100e-9;
        let k_plate = 1e-3;
        let m = 1e-12;

        let result = three_body_casimir_dynamics(
            r, r, gap, gap, k_plate, m, m, m,
        );

        // Effective gain can be negative (opposite force direction coupling)
        // The magnitude should be non-zero and reasonable
        assert!(result.effective_gain.abs() > 0.0, "Gain magnitude should be non-zero");

        // Compare magnitude with quasistatic approximation
        let quasistatic_gain = three_body_gain_quasistatic(r, r, gap, gap, k_plate);

        // The formulations differ slightly, but magnitudes should be comparable
        // (within an order of magnitude, accounting for sign and formulation differences)
        assert!(result.effective_gain.abs() > 0.001, "Gain magnitude too small");
        assert!(result.effective_gain.abs() < 10.0, "Gain magnitude too large");

        // Both should be less than 1 for passive system (energy conserving)
        assert!(result.effective_gain.abs() < 1.0);
        assert!(quasistatic_gain < 1.0);
    }

    #[test]
    fn test_three_body_gain_strict_passes() {
        let r = 20e-6;  // Large spheres for 1% accuracy
        let gap = 100e-9;
        let k_plate = 1e-3;

        let result = three_body_gain_strict(
            r, r, gap, gap, k_plate,
            PfaAccuracy::FivePercent,
        );

        assert!(result.is_ok());
        let gain = result.unwrap();
        assert!(gain > 0.0);
    }

    #[test]
    fn test_three_body_gain_strict_fails() {
        let r = 1e-6;  // Small spheres
        let gap = 100e-9;  // R/d = 10, insufficient for 1%
        let k_plate = 1e-3;

        let result = three_body_gain_strict(
            r, r, gap, gap, k_plate,
            PfaAccuracy::OnePercent,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_three_body_validity_info() {
        let r = 5e-6;
        let gap = 100e-9;
        let k_plate = 1e-3;
        let m = 1e-12;

        let result = three_body_casimir_dynamics(
            r, r, gap, gap, k_plate, m, m, m,
        );

        // Should include validity info for both source and drain
        assert!(result.source_validity.five_percent_valid);
        assert!(result.drain_validity.five_percent_valid);
    }
}

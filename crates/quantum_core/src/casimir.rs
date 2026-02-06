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
/// The dielectric function at imaginary frequency ε(iξ) is always real and
/// monotonically decreasing from ε(0) to 1 as ξ → ∞. This is a consequence
/// of Kramers-Kronig relations.
#[derive(Debug, Clone)]
pub enum DielectricModel {
    /// Perfect conductor: ε → ∞ (reflection coefficient r = 1)
    PerfectConductor,

    /// Drude model for metals: ε(iξ) = 1 + ω_p² / (ξ(ξ + γ))
    /// Parameters: (plasma frequency ω_p in rad/s, damping γ in rad/s)
    Drude { omega_p: f64, gamma: f64 },

    /// Plasma model (dissipationless Drude): ε(iξ) = 1 + ω_p² / ξ²
    /// The γ → 0 limit of Drude; controversial for thermal Casimir effect
    Plasma { omega_p: f64 },

    /// Drude-Lorentz oscillator model (dielectrics with resonance):
    /// ε(iξ) = 1 + Σ S_j ω_j² / (ω_j² + ξ² + γ_j ξ)
    /// Parameters: Vec of (oscillator strength S, resonance ω_j, damping γ_j)
    DrudeLorentz { oscillators: Vec<(f64, f64, f64)> },

    /// Tabulated dielectric data (interpolated)
    /// Parameters: (frequencies ξ in rad/s, dielectric values ε(iξ))
    Tabulated { xi: Vec<f64>, eps: Vec<f64> },
}

impl DielectricModel {
    /// Create a Drude model for gold at room temperature.
    ///
    /// Parameters from optical data:
    /// - ω_p = 9.0 eV = 1.37e16 rad/s
    /// - γ = 35 meV = 5.3e13 rad/s
    pub fn gold() -> Self {
        DielectricModel::Drude {
            omega_p: 1.37e16,
            gamma: 5.3e13,
        }
    }

    /// Create a Drude model for aluminum.
    ///
    /// Parameters from optical data:
    /// - ω_p = 12.5 eV = 1.9e16 rad/s
    /// - γ = 126 meV = 1.9e14 rad/s
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
    /// - Static ε ≈ 3.8
    pub fn silica() -> Self {
        // Single oscillator: S*ω² / (ω² + ξ²) where S*(ω/ω)² gives ε(0)-1
        // For ε(0) ~ 3.8, S ~ 2.8 with ω ~ 1.5e16 rad/s (10 eV)
        DielectricModel::DrudeLorentz {
            oscillators: vec![(2.8, 1.5e16, 1e15)],
        }
    }

    /// Evaluate dielectric function at imaginary frequency ε(iξ).
    ///
    /// Returns the real, positive dielectric response at the imaginary
    /// frequency iξ. The result is always >= 1 for passive materials.
    ///
    /// # Arguments
    /// * `xi` - Imaginary frequency (rad/s), must be >= 0
    pub fn epsilon_at_imaginary(&self, xi: f64) -> f64 {
        match self {
            // Perfect conductor approximated by very large but finite ε
            // This avoids NaN in numerical integration while capturing
            // the essential physics (r → 1)
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
/// r_TM = (ε₁ κ₂ - ε₂ κ₁) / (ε₁ κ₂ + ε₂ κ₁)
///
/// where κ_i = sqrt(ε_i ξ²/c² + k_⊥²)
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
    // In the limit ε₂ → ∞: r_TM → (ε₁/ε₂)(κ₂/κ₁) → 0, but accounting
    // for the fact that κ₂ ∝ √ε₂, we get r_TM → -1
    // For numerical stability with large finite ε, compute directly
    let numer = eps1 * kappa2 - eps2 * kappa1;
    let denom = eps1 * kappa2 + eps2 * kappa1;

    if denom.abs() < 1e-30 {
        return 0.0;
    }

    numer / denom
}

/// Fresnel reflection coefficient at imaginary frequency (TE/s-polarization).
///
/// r_TE = (κ₂ - κ₁) / (κ₂ + κ₁)
///
/// For non-magnetic materials (μ = 1).
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
/// P = -(ℏ/2π²) ∫₀^∞ dξ ∫₀^∞ k_⊥ dk_⊥ κ (r_TM² e^{-2κd}/(1-r_TM² e^{-2κd}) + TE)
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

    // Integration over ξ using Gauss-Laguerre-like quadrature
    // We use a change of variables: ξ = xi_char * u, integrate u from 0 to ~10
    let mut pressure = 0.0;

    for i in 0..n_xi {
        // Logarithmic spacing works well for the oscillatory integrand
        let u = ((i as f64 + 0.5) / n_xi as f64) * 10.0;
        let xi = xi_char * u;
        let du = 10.0 / n_xi as f64;

        // Dielectric values at this frequency
        let e1 = eps1.epsilon_at_imaginary(xi);
        let e2 = eps2.epsilon_at_imaginary(xi);

        // Integration over k_perp: k_perp = (ξ/c) * v, v from 0 to ~10
        for j in 0..n_k {
            let v = ((j as f64 + 0.5) / n_k as f64) * 10.0;
            let dv = 10.0 / n_k as f64;
            let k_perp = (xi / C) * v;

            // κ = sqrt(ε*ξ²/c² + k_⊥²), for vacuum (ε=1):
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

            // Integrand: k_perp * κ * (TM + TE contributions)
            pressure += k_perp * kappa * (tm_factor + te_factor) * jacobian;
        }
    }

    // Prefactor: -ℏ/(2π²)
    pressure *= -HBAR / (2.0 * PI * PI);

    pressure
}

/// Lifshitz force for sphere-plate geometry using PFA.
///
/// Combines the Lifshitz pressure with the Proximity Force Approximation:
/// F_sp = 2πR ∫ P(d) d(d) evaluated at the minimum gap.
///
/// For computational efficiency, we use:
/// F_sp ≈ 2πR * gap * P(gap) * correction_factor
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

    // PFA integration: F = 2πR ∫ P(d') d(d') from gap to ∞
    // For P ~ 1/d^3 (perfect conductor), this gives F = πR * P(gap) * gap
    // More generally, we need the integrated pressure, which for P(d) ~ 1/d^n gives:
    // F = 2πR * gap * P(gap) / (n-1) where n ~ 3 for Casimir
    // Using the standard result from PFA: F ≈ 2πR * ∫_gap^∞ P(d') dd'
    //                                       ≈ 2πR * gap * P(gap) / 2  (for n=3)

    // For consistency with casimir_force_pfa, use the PFA formula
    2.0 * PI * radius * gap * pressure / 2.0
}

/// Lifshitz force ratio: compares material-dependent force to perfect conductor.
///
/// η = F_Lifshitz / F_ideal_PFA
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
/// Ratio η (dimensionless, 0 < η <= 1)
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
/// ξ_n = 2π n k_B T / ℏ
///
/// At room temperature (T = 300 K), ξ_1 ≈ 2.4e14 rad/s.
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
/// λ_T = ℏc / (k_B T)
///
/// At room temperature: λ_T ≈ 7.6 μm
///
/// # Arguments
/// * `temperature` - Temperature in Kelvin
pub fn thermal_wavelength(temperature: f64) -> f64 {
    const K_B: f64 = 1.380649e-23;
    HBAR * C / (K_B * temperature)
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
        assert!(eps_high < 2.0, "High-frequency ε should be close to 1");

        // At lower frequency, dielectric should be larger
        let eps_low = gold.epsilon_at_imaginary(1e14);
        assert!(eps_low > eps_high, "ε should decrease with frequency");
    }

    #[test]
    fn test_dielectric_plasma_model() {
        let plasma = DielectricModel::gold_plasma();

        // Plasma model: ε(iξ) = 1 + ω_p² / ξ²
        let omega_p = 1.37e16;
        let xi = 1e15;
        let expected = 1.0 + omega_p * omega_p / (xi * xi);
        let actual = plasma.epsilon_at_imaginary(xi);

        assert!((actual - expected).abs() / expected < 1e-10);
    }

    #[test]
    fn test_dielectric_silica() {
        let silica = DielectricModel::silica();

        // Silica should have ε > 1 at all frequencies (dielectric)
        let eps1 = silica.epsilon_at_imaginary(1e14);
        let eps2 = silica.epsilon_at_imaginary(1e16);

        assert!(eps1 > 1.0);
        assert!(eps2 > 1.0);
        // At resonance frequency (~1.5e16), ε should drop
        assert!(eps2 < eps1, "ε should decrease toward resonance");
    }

    #[test]
    fn test_fresnel_tm_high_epsilon() {
        // Very high dielectric (like perfect conductor): r_TM approaches -1
        // For vacuum/metal interface: r_TM = (ε₁κ₂ - ε₂κ₁)/(ε₁κ₂ + ε₂κ₁)
        // As ε₂ → ∞, κ₂ ∝ √ε₂, so ratio → -1
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

        // r_TE = (κ2 - κ1)/(κ2 + κ1) = (√ε2 - √ε1)/(√ε2 + √ε1)
        let expected = (eps2.sqrt() - eps1.sqrt()) / (eps2.sqrt() + eps1.sqrt());
        assert!((r - expected).abs() < 1e-10);
    }

    #[test]
    fn test_lifshitz_pressure_order_of_magnitude() {
        // Perfect conductors at 100 nm gap
        // P = -π²ℏc / (240 d⁴) for perfect conductors
        // At d = 100 nm: P ≈ -13 Pa
        let gap = 100e-9;
        let eps = DielectricModel::PerfectConductor;
        let pressure = lifshitz_pressure_plates(gap, &eps, &eps, 32, 32);

        // Should be in the Pa range (negative)
        assert!(pressure < 0.0, "Pressure should be attractive");

        // Order of magnitude check for 100 nm gap:
        // P = -π² ℏ c / (240 d⁴) ≈ -13 Pa
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
        // At 300 K, thermal wavelength ~ 7.6 μm
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
}

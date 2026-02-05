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

/// Check if PFA is valid for given geometry.
///
/// PFA requires R >> d. We use R > 10*d as a practical threshold.
pub fn pfa_is_valid(radius: f64, gap: f64) -> bool {
    radius > 10.0 * gap
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
        // Valid when R >> d
        assert!(pfa_is_valid(5e-6, 100e-9));  // R/d = 50
        assert!(pfa_is_valid(5e-6, 400e-9));  // R/d = 12.5
        assert!(!pfa_is_valid(5e-6, 1e-6));   // R/d = 5
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
}

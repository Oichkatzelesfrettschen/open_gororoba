//! Core types for photon-graviton mixing computations.
//!
//! All types use natural units (hbar = c = 1, m_e = 1) unless explicitly noted.
//! The field parametrization follows Bastianelli-Schubert (2005-2026) conventions.

use num_complex::Complex64;

/// Loop particle species.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopType {
    /// Charged scalar (Klein-Gordon) in the loop.
    Scalar,
    /// Charged fermion (Dirac spinor) in the loop -- physical QED.
    Spinor,
}

/// Photon linear polarization relative to the external magnetic field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhotonPolarization {
    /// Electric field parallel to the external B field.
    Parallel,
    /// Electric field perpendicular to the external B field.
    Perpendicular,
}

/// Graviton helicity state (transverse-traceless decomposition).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GravitonPolarization {
    /// Plus polarization (e_+).
    Plus,
    /// Cross polarization (e_x).
    Cross,
}

/// Combined helicity channel for the photon-graviton transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HelicityChannel {
    ParallelPlus,
    ParallelCross,
    PerpPlus,
    PerpCross,
}

impl HelicityChannel {
    /// Photon polarization component.
    pub fn photon_pol(self) -> PhotonPolarization {
        match self {
            Self::ParallelPlus | Self::ParallelCross => PhotonPolarization::Parallel,
            Self::PerpPlus | Self::PerpCross => PhotonPolarization::Perpendicular,
        }
    }

    /// Graviton polarization component.
    pub fn graviton_pol(self) -> GravitonPolarization {
        match self {
            Self::ParallelPlus | Self::PerpPlus => GravitonPolarization::Plus,
            Self::ParallelCross | Self::PerpCross => GravitonPolarization::Cross,
        }
    }

    /// All four channels.
    pub const ALL: [Self; 4] = [
        Self::ParallelPlus,
        Self::ParallelCross,
        Self::PerpPlus,
        Self::PerpCross,
    ];
}

/// Classification of the background EM field by its Lorentz invariants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldType {
    /// Pure magnetic (F.F > 0, F.*F = 0): only `a` nonzero.
    Magnetic,
    /// Pure electric (F.F < 0, F.*F = 0): only `b` nonzero.
    Electric,
    /// Crossed field (F.F = 0, F.*F = 0): null field with E perp B, |E|=|B|.
    Crossed,
    /// General field with both `a` and `b` nonzero.
    General,
}

/// Background electromagnetic field configuration.
///
/// Uses the Lorentz-invariant eigenvalue parametrization of F_{mu nu}:
///   a = magnetic-type eigenvalue (units of m_e^2/e in natural units)
///   b = electric-type eigenvalue
///
/// For a pure magnetic field B along z-hat:
///   a = eB (in natural units where m_e = 1)
///   b = 0
///
/// The photon has 4-momentum k = omega * (sin(theta), 0, cos(theta), i)
/// in Euclidean conventions.
#[derive(Debug, Clone, Copy)]
pub struct FieldConfig {
    /// Magnetic-type secular invariant (natural units).
    pub a: f64,
    /// Electric-type secular invariant (natural units).
    pub b: f64,
    /// Photon virtuality k^2 (0 for on-shell).
    pub k_squared: f64,
    /// Photon energy in natural units (omega / m_e).
    pub omega: f64,
    /// Angle between photon propagation and B field [radians].
    pub theta: f64,
}

impl FieldConfig {
    /// Construct for a pure magnetic field.
    ///
    /// # Arguments
    /// * `b_over_bcr` -- B / B_critical (dimensionless ratio)
    /// * `omega_over_m` -- photon energy / electron mass
    /// * `theta` -- angle between k and B [radians]
    pub fn pure_magnetic(b_over_bcr: f64, omega_over_m: f64, theta: f64) -> Self {
        // In natural units with m_e = 1:
        // eB = (B/B_cr) since B_cr = m_e^2/(e) = 1/e in natural units
        // and a = eB. So a = B/B_cr when we use the conventional normalization.
        Self {
            a: b_over_bcr,
            b: 0.0,
            k_squared: 0.0,
            omega: omega_over_m,
            theta,
        }
    }

    /// Construct for a pure electric field.
    pub fn pure_electric(e_over_ecr: f64, omega_over_m: f64, theta: f64) -> Self {
        Self {
            a: 0.0,
            b: e_over_ecr,
            k_squared: 0.0,
            omega: omega_over_m,
            theta,
        }
    }

    /// Classify the field type from invariants.
    pub fn field_type(&self) -> FieldType {
        let tol = 1e-15;
        match (self.a.abs() > tol, self.b.abs() > tol) {
            (true, false) => FieldType::Magnetic,
            (false, true) => FieldType::Electric,
            (false, false) => FieldType::Crossed,
            (true, true) => FieldType::General,
        }
    }

    /// The dimensionless parameter z = eB*T used in proper-time integrands.
    /// For pure magnetic field, z = a * T.
    pub fn z_param(&self, proper_time: f64) -> f64 {
        self.a * proper_time
    }
}

/// Result of a single amplitude computation.
#[derive(Debug, Clone, Copy)]
pub struct AmplitudeResult {
    /// Complex amplitude value.
    pub value: Complex64,
    /// Estimated numerical error.
    pub error_estimate: f64,
    /// Number of integrand evaluations used.
    pub n_evaluations: usize,
}

/// Full one-loop amplitude decomposed into the three diagrams.
#[derive(Debug, Clone, Copy)]
pub struct FullAmplitude {
    /// Irreducible (non-1PI vacuum polarization insertion) diagram.
    pub irreducible: Complex64,
    /// Tadpole diagram (new contribution from Part 3).
    pub tadpole: Complex64,
    /// External-leg diagram (vacuum polarization on external photon leg).
    pub external: Complex64,
    /// Total = irreducible + tadpole + external.
    pub total: Complex64,
    /// Conversion probability |M|^2 / normalization.
    pub conversion_probability: f64,
}

/// Result of magnetic dichroism computation.
#[derive(Debug, Clone, Copy)]
pub struct DichroismResult {
    /// Conversion rate for parallel photon polarization.
    pub rate_parallel: f64,
    /// Conversion rate for perpendicular photon polarization.
    pub rate_perp: f64,
    /// Ratio rate_parallel / rate_perp (4/7 at weak field for spinor QED).
    pub ratio: f64,
}

/// Result of a Ward identity check.
#[derive(Debug, Clone, Copy)]
pub struct WardCheckResult {
    /// Name of the identity being checked.
    pub identity: WardIdentity,
    /// Magnitude of the residual (should be ~ 0).
    pub residual: f64,
    /// Whether the check passes within tolerance.
    pub passes: bool,
    /// Tolerance used.
    pub tolerance: f64,
}

/// Which Ward identity is being verified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WardIdentity {
    /// Gauge: k_alpha * Gamma^{mn,a} = 0.
    Gauge,
    /// Gravitational: linearized diffeomorphism invariance.
    Gravitational,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_config_pure_magnetic() {
        let cfg = FieldConfig::pure_magnetic(0.1, 0.5, 0.3);
        assert_eq!(cfg.field_type(), FieldType::Magnetic);
        assert!((cfg.a - 0.1).abs() < 1e-15);
        assert_eq!(cfg.b, 0.0);
    }

    #[test]
    fn test_field_config_pure_electric() {
        let cfg = FieldConfig::pure_electric(0.2, 0.5, 0.3);
        assert_eq!(cfg.field_type(), FieldType::Electric);
    }

    #[test]
    fn test_field_config_crossed() {
        let cfg = FieldConfig {
            a: 0.0,
            b: 0.0,
            k_squared: 0.0,
            omega: 1.0,
            theta: 0.5,
        };
        assert_eq!(cfg.field_type(), FieldType::Crossed);
    }

    #[test]
    fn test_field_config_general() {
        let cfg = FieldConfig {
            a: 0.1,
            b: 0.2,
            k_squared: 0.0,
            omega: 1.0,
            theta: 0.5,
        };
        assert_eq!(cfg.field_type(), FieldType::General);
    }

    #[test]
    fn test_helicity_channel_decomposition() {
        assert_eq!(
            HelicityChannel::ParallelPlus.photon_pol(),
            PhotonPolarization::Parallel
        );
        assert_eq!(
            HelicityChannel::ParallelPlus.graviton_pol(),
            GravitonPolarization::Plus
        );
        assert_eq!(
            HelicityChannel::PerpCross.photon_pol(),
            PhotonPolarization::Perpendicular
        );
        assert_eq!(
            HelicityChannel::PerpCross.graviton_pol(),
            GravitonPolarization::Cross
        );
    }

    #[test]
    fn test_z_param() {
        let cfg = FieldConfig::pure_magnetic(0.1, 0.5, 0.3);
        let z = cfg.z_param(2.0);
        assert!((z - 0.2).abs() < 1e-15);
    }

    #[test]
    fn test_all_channels_count() {
        assert_eq!(HelicityChannel::ALL.len(), 4);
    }
}

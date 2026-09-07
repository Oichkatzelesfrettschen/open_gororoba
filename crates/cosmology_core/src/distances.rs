//! Cosmological distance functions and the Macquart DM-redshift relation.
//!
//! Extends the bounce module's infrastructure with comoving distances and
//! the FRB-specific DM <-> redshift conversion needed for ultrametric analysis.
//!
//! # References
//!
//! - Macquart et al. (2020), Nature 581, 391 (DM-z relation)
//! - Hogg (1999), arXiv:astro-ph/9905116 (distance measures review)
//! - Planck Collaboration VI (2020), A&A 641, A6 (cosmological parameters)

use crate::{
    bounce::{C_KM_S, hubble_e_lcdm},
    gl_integrate,
};

/// Numerical inversion ceiling; ionization history limits physical applicability separately.
pub const MAX_MACQUART_REDSHIFT: f64 = 100.0;

/// Failure to infer a bounded conditional distance from dispersion measures.
#[derive(Debug, Clone, PartialEq)]
pub enum DmInversionError {
    InvalidDm { component: &'static str },
    InvalidCosmology,
    NegativeResidual,
    OutOfRange { maximum_dm: f64 },
    NumericalFailure,
}

impl std::fmt::Display for DmInversionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDm { component } => {
                write!(formatter, "{component} DM must be finite and nonnegative")
            }
            Self::InvalidCosmology => write!(
                formatter,
                "cosmology requires finite omega_m in [0,1], omega_b in (0,1], and h0 > 0"
            ),
            Self::NegativeResidual => write!(
                formatter,
                "foreground subtraction produces negative cosmic DM"
            ),
            Self::OutOfRange { maximum_dm } => write!(
                formatter,
                "cosmic DM exceeds {maximum_dm} at the numerical redshift ceiling {MAX_MACQUART_REDSHIFT}"
            ),
            Self::NumericalFailure => write!(
                formatter,
                "cosmological evaluation exceeds finite numerical precision"
            ),
        }
    }
}

impl std::error::Error for DmInversionError {}

fn valid_cosmology(omega_m: f64, omega_b: f64, h0: f64) -> bool {
    (0.0..=1.0).contains(&omega_m)
        && omega_b.is_finite()
        && omega_b > 0.0
        && omega_b <= 1.0
        && h0.is_finite()
        && h0 > 0.0
}

fn bounded_integral(integrand: impl Fn(f64) -> f64, redshift: f64) -> f64 {
    let quadrature = gauss_quad::GaussLegendre::new(std::num::NonZeroUsize::new(20).unwrap());
    let segments = redshift.ceil().max(1.0) as u32;
    let width = redshift / f64::from(segments);
    (0..segments)
        .map(|index| {
            quadrature.integrate(
                f64::from(index) * width,
                f64::from(index + 1) * width,
                &integrand,
            )
        })
        .sum()
}

/// Planck 2018 TT,TE,EE+lowE+lensing best-fit parameters.
/// Canonical values for cosmological distance calculations.
pub mod planck2018 {
    /// Hubble constant (km/s/Mpc).
    pub const H0: f64 = 67.36;
    /// Total matter density parameter.
    pub const OMEGA_M: f64 = 0.3153;
    /// Baryon density parameter.
    pub const OMEGA_B: f64 = 0.0493;
    /// Dark energy density parameter.
    pub const OMEGA_LAMBDA: f64 = 0.6847;
    /// RMS density fluctuation at 8 Mpc/h.
    pub const SIGMA_8: f64 = 0.8111;
    /// Scalar spectral index.
    pub const N_S: f64 = 0.9649;
    /// Optical depth to reionization.
    pub const TAU: f64 = 0.0544;
}

/// Comoving distance d_C(z) in Mpc for flat Lambda-CDM.
///
/// d_C(z) = (c/H_0) * integral_0^z dz' / E(z')
///
/// This is the line-of-sight comoving distance -- the distance between
/// two objects at the same epoch that would be measured by a ruler
/// (if such a ruler could exist) between them today.
/// The interval 0 < z <= 100 uses composite quadrature shared with DM inversion.
pub fn comoving_distance(z: f64, omega_m: f64, h0: f64) -> f64 {
    if z <= 0.0 {
        return 0.0;
    }

    let integrand = |zp| 1.0 / hubble_e_lcdm(zp, omega_m);
    let integral = if z <= MAX_MACQUART_REDSHIFT {
        bounded_integral(integrand, z)
    } else {
        gl_integrate(integrand, 0.0, z, 50)
    };

    (C_KM_S / h0) * integral
}

/// Mean cosmic DM in pc/cm^3 under a constant diffuse baryon fraction.
///
/// DM_cosmic(z) = (3 * c * H_0 * Omega_b * f_d * f_e) / (8 * pi * G * m_p) *
///                integral_0^z (1+z') / E(z') dz'
///
/// Macquart et al. (2020), Eq. (2), gives f_e = 1 - Y_He/2 = 0.875 for
/// fully ionized hydrogen and doubly ionized helium with Y_He = 0.25.
/// The fixed f_d = 0.83 is a model choice for the diffuse baryon mass fraction;
/// the source permits f_d(z). With Planck 2018 parameters and the constants
/// below, the prefactor is approximately 811.944478 pc/cm^3.
///
/// The integral assumes flat matter-plus-Lambda expansion. A mean relation
/// excludes line-of-sight scatter and the Galactic and host contributions.
/// Extending the model across ionization epochs requires an electron history.
/// Invalid cosmology or redshift outside [0, 100] returns NaN. The numerical
/// interval does not establish physical validity across ionization epochs.
pub fn macquart_dm_cosmic(z: f64, omega_m: f64, omega_b: f64, h0: f64) -> f64 {
    if !(0.0..=MAX_MACQUART_REDSHIFT).contains(&z) || !valid_cosmology(omega_m, omega_b, h0) {
        return f64::NAN;
    }
    if z == 0.0 {
        return 0.0;
    }

    let h0_s = h0 * 1e5 / (3.0857e24); // H0 in s^-1 (km/s/Mpc -> 1/s)
    let c_cgs = 2.99792458e10; // cm/s
    let g_cgs = 6.67430e-8; // cm^3 / (g s^2)
    let m_p = 1.67262192e-24; // g
    let diffuse_baryon_fraction = 0.83;
    let helium_mass_fraction = 0.25;
    let electrons_per_baryon = 1.0 - helium_mass_fraction / 2.0;

    let prefactor = 3.0 * c_cgs * h0_s * omega_b * diffuse_baryon_fraction * electrons_per_baryon
        / (8.0 * std::f64::consts::PI * g_cgs * m_p);

    // Convert to pc/cm^3: 1 pc = 3.0857e18 cm
    let prefactor_pc = prefactor / 3.0857e18;

    let integral = bounded_integral(|zp| (1.0 + zp) / hubble_e_lcdm(zp, omega_m), z);

    prefactor_pc * integral
}

/// Invert the Macquart relation: DM_excess -> approximate redshift.
///
/// Uses bisection search on [`macquart_dm_cosmic`]. Input DM in pc/cm^3
/// represents the cosmic component after Galactic disk, Galactic halo and
/// observer-frame host subtraction. Macquart et al. (2020), Eq. (1), weights
/// a rest-frame host contribution by 1/(1+z); callers supply the converted term.
///
/// Returns an error for invalid inputs or DM above the value at the numerical
/// ceiling z=100. Bisection uses relative DM tolerance 1e-12, including near
/// zero; the numerical bracket supplies no physical calibration.
pub fn dm_excess_to_redshift(
    dm_excess: f64,
    omega_m: f64,
    omega_b: f64,
    h0: f64,
) -> Result<f64, DmInversionError> {
    if !valid_cosmology(omega_m, omega_b, h0) {
        return Err(DmInversionError::InvalidCosmology);
    }
    if !dm_excess.is_finite() || dm_excess < 0.0 {
        return Err(DmInversionError::InvalidDm {
            component: "cosmic",
        });
    }
    if dm_excess == 0.0 {
        return Ok(0.0);
    }
    let mut z_lo = 0.0;
    let mut z_hi = 1.0;
    loop {
        let upper_dm = macquart_dm_cosmic(z_hi, omega_m, omega_b, h0);
        if !upper_dm.is_finite() || upper_dm <= 0.0 {
            return Err(DmInversionError::NumericalFailure);
        }
        if dm_excess == upper_dm {
            return Ok(z_hi);
        }
        if dm_excess < upper_dm {
            break;
        }
        if z_hi == MAX_MACQUART_REDSHIFT {
            return Err(DmInversionError::OutOfRange {
                maximum_dm: upper_dm,
            });
        }
        z_lo = z_hi;
        z_hi = (2.0 * z_hi).min(MAX_MACQUART_REDSHIFT);
    }
    // A finite f64 exponent range needs at most 1082 halvings from z=100.
    for _ in 0..1100 {
        let z_mid = z_lo + 0.5 * (z_hi - z_lo);
        let dm_mid = macquart_dm_cosmic(z_mid, omega_m, omega_b, h0);
        if !dm_mid.is_finite() {
            return Err(DmInversionError::NumericalFailure);
        }
        if z_mid == z_lo || z_mid == z_hi {
            return Err(DmInversionError::NumericalFailure);
        }
        if dm_mid == dm_excess || (dm_mid / dm_excess - 1.0).abs() <= 1e-12 {
            return Ok(z_mid);
        }
        if dm_mid < dm_excess {
            z_lo = z_mid;
        } else {
            z_hi = z_mid;
        }
    }
    Err(DmInversionError::NumericalFailure)
}

/// Full DM -> comoving distance chain.
///
/// 1. Subtract total Galactic and observer-frame host DM contributions
/// 2. Invert Macquart relation to get redshift
/// 3. Compute comoving distance at that redshift
///
/// `dm_mw` includes Galactic disk and halo. `dm_host_observer` is already in
/// the observer frame; a rest-frame host estimate requires division by 1+z
/// before calling. All DM arguments use pc/cm^3. A fixed subtraction defines
/// a conditional distance estimate rather than a joint host/redshift inference.
///
/// Returns comoving distance in Mpc. Negative residuals are errors. The inversion
/// inherits the bracket and model assumptions of [`dm_excess_to_redshift`].
pub fn dm_to_comoving(
    dm_obs: f64,
    dm_mw: f64,
    dm_host_observer: f64,
    omega_m: f64,
    omega_b: f64,
    h0: f64,
) -> Result<f64, DmInversionError> {
    for (component, value) in [
        ("observed", dm_obs),
        ("Galactic", dm_mw),
        ("observer-frame host", dm_host_observer),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(DmInversionError::InvalidDm { component });
        }
    }
    let dm_excess = dm_obs - dm_mw - dm_host_observer;
    if dm_excess < 0.0 {
        return Err(DmInversionError::NegativeResidual);
    }
    let redshift = dm_excess_to_redshift(dm_excess, omega_m, omega_b, h0)?;
    if redshift == 0.0 {
        return Ok(0.0);
    }
    let distance = comoving_distance(redshift, omega_m, h0);
    if distance.is_finite() && distance > 0.0 {
        Ok(distance)
    } else {
        Err(DmInversionError::NumericalFailure)
    }
}

/// Angular diameter distance d_A(z) in Mpc for flat Lambda-CDM.
///
/// d_A(z) = d_C(z) / (1+z)
pub fn angular_diameter_distance(z: f64, omega_m: f64, h0: f64) -> f64 {
    if z <= 0.0 {
        return 0.0;
    }
    comoving_distance(z, omega_m, h0) / (1.0 + z)
}

/// Convert RA, Dec (degrees) + comoving distance (Mpc) to Cartesian coordinates.
///
/// Returns (x, y, z) in Mpc in a right-handed coordinate system where
/// x points toward (RA=0, Dec=0), z points toward the north celestial pole.
pub fn radec_to_cartesian(ra_deg: f64, dec_deg: f64, d_c: f64) -> (f64, f64, f64) {
    let ra = ra_deg.to_radians();
    let dec = dec_deg.to_radians();

    let x = d_c * dec.cos() * ra.cos();
    let y = d_c * dec.cos() * ra.sin();
    let z = d_c * dec.sin();

    (x, y, z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_comoving_distance_at_z0() {
        let d = comoving_distance(0.0, 0.3, 70.0);
        assert_relative_eq!(d, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_comoving_distance_increases() {
        let d1 = comoving_distance(0.5, 0.3153, 67.36);
        let d2 = comoving_distance(1.0, 0.3153, 67.36);
        let d3 = comoving_distance(2.0, 0.3153, 67.36);

        assert!(d1 > 0.0);
        assert!(d2 > d1);
        assert!(d3 > d2);
    }

    #[test]
    fn test_comoving_distance_typical_z1() {
        // At z=1 with Planck params, d_C ~ 3300 Mpc
        let d = comoving_distance(1.0, 0.3153, 67.36);
        assert!(
            d > 3000.0 && d < 3600.0,
            "d_C(z=1) = {} Mpc (expected ~3300)",
            d
        );
    }

    #[test]
    fn test_macquart_dm_at_z0() {
        let dm = macquart_dm_cosmic(0.0, 0.3153, 0.0493, 67.36);
        assert_relative_eq!(dm, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_macquart_dm_increases_with_z() {
        let dm1 = macquart_dm_cosmic(0.5, 0.3153, 0.0493, 67.36);
        let dm2 = macquart_dm_cosmic(1.0, 0.3153, 0.0493, 67.36);
        let dm3 = macquart_dm_cosmic(2.0, 0.3153, 0.0493, 67.36);

        assert!(dm1 > 0.0);
        assert!(dm2 > dm1);
        assert!(dm3 > dm2);
    }

    #[test]
    fn test_macquart_dm_typical_z1() {
        // At z=1, DM_cosmic ~ 900-1100 pc/cm^3 (Macquart+ 2020)
        let dm = macquart_dm_cosmic(1.0, 0.3153, 0.0493, 67.36);
        assert!(
            dm > 700.0 && dm < 1200.0,
            "DM_cosmic(z=1) = {} pc/cm^3 (expected ~900-1100)",
            dm
        );
    }

    #[test]
    fn test_macquart_inversion_roundtrip() {
        // DM -> z -> DM should roundtrip
        let z_true = 0.5;
        let dm = macquart_dm_cosmic(z_true, 0.3153, 0.0493, 67.36);
        let z_recovered = dm_excess_to_redshift(dm, 0.3153, 0.0493, 67.36).unwrap();

        assert_relative_eq!(z_recovered, z_true, epsilon = 1e-4);
    }

    #[test]
    fn test_macquart_inversion_high_z() {
        let z_true = 2.0;
        let dm = macquart_dm_cosmic(z_true, 0.3153, 0.0493, 67.36);
        let z_recovered = dm_excess_to_redshift(dm, 0.3153, 0.0493, 67.36).unwrap();

        assert_relative_eq!(z_recovered, z_true, epsilon = 1e-3);
    }

    #[test]
    fn test_dm_to_comoving_positive() {
        // Typical CHIME FRB: DM_obs = 500, DM_MW ~ 100, DM_host ~ 50
        let d = dm_to_comoving(500.0, 100.0, 50.0, 0.3153, 0.0493, 67.36).unwrap();
        assert!(d > 0.0, "d_C should be positive for DM_excess > 0");
        assert!(
            d < 10000.0,
            "d_C should be less than 10 Gpc for typical FRB DM"
        );
    }

    #[test]
    fn test_dm_to_comoving_zero_excess() {
        let d = dm_to_comoving(150.0, 100.0, 50.0, 0.3153, 0.0493, 67.36).unwrap();
        assert_relative_eq!(d, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_angular_diameter_distance() {
        // d_A = d_C / (1+z), so d_A < d_C for z > 0
        let d_c = comoving_distance(1.0, 0.3153, 67.36);
        let d_a = angular_diameter_distance(1.0, 0.3153, 67.36);
        assert_relative_eq!(d_a, d_c / 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_radec_to_cartesian_poles() {
        // North pole: (0, 0, d)
        let (x, y, z) = radec_to_cartesian(0.0, 90.0, 100.0);
        assert!(x.abs() < 1e-10);
        assert!(y.abs() < 1e-10);
        assert_relative_eq!(z, 100.0, epsilon = 1e-6);
    }

    #[test]
    fn test_radec_to_cartesian_origin() {
        // (RA=0, Dec=0, d) -> (d, 0, 0)
        let (x, y, z) = radec_to_cartesian(0.0, 0.0, 100.0);
        assert_relative_eq!(x, 100.0, epsilon = 1e-6);
        assert!(y.abs() < 1e-10);
        assert!(z.abs() < 1e-10);
    }

    #[test]
    fn test_planck2018_constants() {
        // Flat universe check
        let total = planck2018::OMEGA_M + planck2018::OMEGA_LAMBDA;
        assert!((total - 1.0).abs() < 0.01, "Planck flatness: {}", total);
    }
}

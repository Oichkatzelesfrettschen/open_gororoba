//! Flyby configuration constants and spacecraft database.
//!
//! Extracted from flyby_crucible.rs to support reuse across binaries.

/// Coupling constant with NFW-like 1/r^3 density scaling.
///
/// With uniform density (no scaling), alpha = 8e-14 reproduced sign patterns
/// but magnitudes were ~5x too large. With 1/r^3 gravitational focusing,
/// the effective integration path is weighted toward perigee, requiring a
/// larger alpha_0 to produce the same integrated delta-V. Calibrated to
/// NEAR's observed +13.46 mm/s.
pub const ALPHA_CHINGON: f64 = 6.0e-12;

/// Earth GM in km^3/s^2.
pub const GM_EARTH: f64 = 398600.4418;

/// Earth radius in km.
pub const R_EARTH: f64 = 6371.0;

/// Dark matter wind velocity in Galactic coordinates (km/s).
/// Sun moves at ~220 km/s toward Galactic l=90, b=0 (Cygnus direction).
/// Components: (U_toward_center, V_rotation, W_north) in Galactic frame.
pub const V_WIND_GALACTIC: [f64; 3] = [-11.1, 232.24, 7.25];

/// SOI radius for integration window: 50 Earth radii.
pub const SOI_R_EARTH: f64 = 50.0;

/// Gravitational focusing wake amplitude.
///
/// Earth's gravity deflects DM particles passing nearby, creating a
/// density enhancement (caustic) downstream in the DM wind flow.
/// The wake has a cos(theta) profile along the wind axis:
///   rho_wake(r, theta) = rho_0 * (1 + ETA_WAKE * cos(theta))
///
/// See BIB-0303 (Lundberg & Edsjo 2004) and BIB-0304 (Lee et al. 2021).
pub const ETA_WAKE: f64 = 0.10;

/// Configuration for a single flyby event.
#[derive(Debug, Clone)]
pub struct FlybyConfig {
    pub name: &'static str,
    /// Perigee altitude above Earth surface (km).
    pub perigee_alt_km: f64,
    /// Hyperbolic excess velocity v_inf (km/s).
    pub v_inf: f64,
    /// Inbound asymptotic declination (degrees, positive = North).
    pub inbound_dec_deg: f64,
    /// Inbound asymptotic right ascension (degrees).
    pub inbound_ra_deg: f64,
    /// Outbound asymptotic declination (degrees).
    pub outbound_dec_deg: f64,
    /// Outbound asymptotic right ascension (degrees).
    pub outbound_ra_deg: f64,
    /// Observed anomalous delta-V (mm/s). Positive = speed gain.
    pub observed_dv_mm_s: f64,
    /// Perigee epoch as Julian Ephemeris Date (JED/TDB).
    pub perigee_jed: f64,
}

/// Return the standard 6-spacecraft flyby database.
///
/// Data from Anderson et al. (2008) PRL 100, 091102, Table I (BIB-0297).
pub fn all_flybys() -> Vec<FlybyConfig> {
    use crate::ephemeris_loader::flyby_epochs;
    vec![
        FlybyConfig {
            name: "Galileo-I (1990-12-08)",
            perigee_alt_km: 960.0,
            v_inf: 8.949,
            inbound_dec_deg: -12.5,
            inbound_ra_deg: 263.0,
            outbound_dec_deg: -4.9,
            outbound_ra_deg: 223.0,
            observed_dv_mm_s: 3.92,
            perigee_jed: flyby_epochs::GALILEO,
        },
        FlybyConfig {
            name: "NEAR (1998-01-23)",
            perigee_alt_km: 539.0,
            v_inf: 6.851,
            inbound_dec_deg: -20.8,
            inbound_ra_deg: 280.0,
            outbound_dec_deg: 72.0,
            outbound_ra_deg: 89.0,
            observed_dv_mm_s: 13.46,
            perigee_jed: flyby_epochs::NEAR,
        },
        FlybyConfig {
            name: "Cassini (1999-08-18)",
            perigee_alt_km: 1175.0,
            v_inf: 16.01,
            inbound_dec_deg: -12.9,
            inbound_ra_deg: 257.0,
            outbound_dec_deg: -5.0,
            outbound_ra_deg: 344.0,
            observed_dv_mm_s: -2.0,
            perigee_jed: flyby_epochs::CASSINI,
        },
        FlybyConfig {
            name: "Rosetta-I (2005-03-04)",
            perigee_alt_km: 1956.0,
            v_inf: 3.863,
            inbound_dec_deg: -34.3,
            inbound_ra_deg: 247.0,
            outbound_dec_deg: -20.6,
            outbound_ra_deg: 116.0,
            observed_dv_mm_s: 1.80,
            perigee_jed: flyby_epochs::ROSETTA_I,
        },
        FlybyConfig {
            name: "MESSENGER (2005-08-02)",
            perigee_alt_km: 2347.0,
            v_inf: 4.056,
            inbound_dec_deg: 31.4,
            inbound_ra_deg: 232.0,
            outbound_dec_deg: 75.4,
            outbound_ra_deg: 174.0,
            observed_dv_mm_s: 0.02,
            perigee_jed: flyby_epochs::MESSENGER,
        },
        FlybyConfig {
            name: "Juno (2013-10-09)",
            perigee_alt_km: 559.0,
            v_inf: 9.897,
            inbound_dec_deg: -13.6,
            inbound_ra_deg: 0.0,
            outbound_dec_deg: -5.3,
            outbound_ra_deg: 345.0,
            observed_dv_mm_s: 0.0,
            perigee_jed: flyby_epochs::JUNO,
        },
    ]
}

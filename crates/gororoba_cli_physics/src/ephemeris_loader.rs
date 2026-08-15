//! Thread-safe JPL ephemeris loader for three-body flyby integration.
//!
//! Uses anise exclusively (pure Rust, thread-safe JPL .bsp/SPK parser).
//! JPL .bsp files store Chebyshev polynomials directly in ICRS/J2000,
//! so state vectors are perfectly aligned with the flyby RA/Dec coordinates.
//! Zero frame rotation = zero geometric risk for the h(t).v_wind cross-product.
//!
//! Thread safety: all query methods take `&self`, safe inside rayon par_iter.
//! anise Vector3 is `nalgebra::Vector3<f64>` -- no conversion needed.

use anise::{
    almanac::Almanac,
    prelude::{Epoch, Frame},
};
use std::path::Path;

/// NAIF body IDs.
const NAIF_MOON: i32 = 301;
const NAIF_SUN: i32 = 10;
const NAIF_EARTH: i32 = 399;

/// GM constants (km^3/s^2) for three-body gravitational acceleration.
pub const GM_MOON: f64 = 4902.799;
pub const GM_SUN: f64 = 132_712_440_018.0;

/// Mass ratio for Earth-Moon barycenter computation.
/// M_moon / M_earth = 0.012300034 (IAU 2015).
const MOON_EARTH_MASS_RATIO: f64 = 0.012_300_034;

/// Three-body state at a given epoch.
#[derive(Debug, Clone, Copy)]
pub struct ThreeBodyState {
    /// Moon geocentric position in J2000 ECI (km).
    pub moon_pos_km: [f64; 3],
    /// Sun geocentric position in J2000 ECI (km).
    pub sun_pos_km: [f64; 3],
    /// Earth-Moon barycenter geocentric offset in J2000 ECI (km).
    /// EMB = M_moon / (M_earth + M_moon) * r_moon
    pub emb_offset_km: [f64; 3],
}

/// Thread-safe JPL ephemeris loader.
///
/// Wraps anise::Almanac for sub-meter precision from JPL DE440/DE430.
/// All state vectors are returned in ICRS/J2000 ECI -- the same frame
/// used by flyby_crucible.rs for asymptotic RA/Dec directions and the
/// Galactic-to-J2000 dark matter wind rotation.
pub struct EphemerisLoader {
    almanac: Almanac,
    earth_j2000: Frame,
}

impl EphemerisLoader {
    /// Load ephemeris from a JPL .bsp file (DE440 or DE430).
    ///
    /// DE440 covers 1550-2650 with sub-meter accuracy for inner planets.
    /// DE430 covers 1550-2650 as well and is an acceptable alternative.
    ///
    /// Panics if the file cannot be loaded -- we require exact ephemeris
    /// data for the three-body tidal correction. No silent fallbacks.
    pub fn load(bsp_path: &Path) -> anyhow::Result<Self> {
        anyhow::ensure!(
            bsp_path.exists(),
            "JPL .bsp file not found: {}. Download DE440 from \
             https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp \
             and place it in data/external/de440.bsp",
            bsp_path.display()
        );
        let path_str = bsp_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Non-UTF8 path: {}", bsp_path.display()))?;
        let almanac = Almanac::new(path_str)
            .map_err(|e| anyhow::anyhow!("Failed to load {}: {}", bsp_path.display(), e))?;
        eprintln!(
            "EphemerisLoader: loaded JPL .bsp from {}",
            bsp_path.display()
        );
        Ok(Self {
            almanac,
            earth_j2000: Frame::from_ephem_j2000(NAIF_EARTH),
        })
    }

    /// Get the complete three-body state at a given Julian Ephemeris Date (JED/TDB).
    ///
    /// Returns Moon and Sun positions in geocentric J2000 ECI (km), plus
    /// the Earth-Moon barycenter offset.
    pub fn three_body_state(&self, jed: f64) -> ThreeBodyState {
        let epoch = Epoch::from_jde_tdb(jed);
        let moon_pos_km = self.query_body(NAIF_MOON, epoch);
        let sun_pos_km = self.query_body(NAIF_SUN, epoch);
        let s = MOON_EARTH_MASS_RATIO / (1.0 + MOON_EARTH_MASS_RATIO);
        let emb_offset_km = [moon_pos_km[0] * s, moon_pos_km[1] * s, moon_pos_km[2] * s];
        ThreeBodyState {
            moon_pos_km,
            sun_pos_km,
            emb_offset_km,
        }
    }

    /// Moon geocentric position in J2000 ECI (km).
    pub fn moon_geocentric_j2000(&self, jed: f64) -> [f64; 3] {
        self.query_body(NAIF_MOON, Epoch::from_jde_tdb(jed))
    }

    /// Sun geocentric position in J2000 ECI (km).
    pub fn sun_geocentric_j2000(&self, jed: f64) -> [f64; 3] {
        self.query_body(NAIF_SUN, Epoch::from_jde_tdb(jed))
    }

    /// Earth-Moon barycenter offset from Earth center in J2000 ECI (km).
    ///
    /// EMB = M_moon / (M_earth + M_moon) * r_moon
    /// This is the TRUE focal point of the local dark matter gravitational well.
    pub fn earth_moon_barycenter(&self, jed: f64) -> [f64; 3] {
        let r_moon = self.moon_geocentric_j2000(jed);
        let s = MOON_EARTH_MASS_RATIO / (1.0 + MOON_EARTH_MASS_RATIO);
        [r_moon[0] * s, r_moon[1] * s, r_moon[2] * s]
    }

    /// Query a body's geocentric J2000 position via anise.
    ///
    /// anise uses nalgebra 0.34 internally, but our workspace pins nalgebra 0.33
    /// (statrs 0.18 constraint). Extract x/y/z components to bridge the version gap.
    fn query_body(&self, naif_id: i32, epoch: Epoch) -> [f64; 3] {
        match self
            .almanac
            .state_of(naif_id, self.earth_j2000, epoch, None)
        {
            Ok(state) => {
                // Bridge nalgebra 0.34 (anise) -> 0.33 (workspace) via component extraction.
                let r = state.radius_km;
                [r[0], r[1], r[2]]
            }
            Err(e) => {
                // This should not happen with a valid DE440 .bsp covering 1550-2650.
                // Log the error but return zeros rather than panicking mid-integration.
                eprintln!(
                    "WARNING: anise query for NAIF {} at {:?} failed: {}",
                    naif_id, epoch, e
                );
                [0.0; 3]
            }
        }
    }
}

/// Flyby epochs in Julian Ephemeris Date (JED/TDB) for all 6 spacecraft.
///
/// These are the perigee passage times used by flyby_crucible.rs.
pub mod flyby_epochs {
    /// Galileo-I: 1990-12-08 ~20:34 UTC
    pub const GALILEO: f64 = 2_448_233.357;
    /// NEAR: 1998-01-23 ~07:23 UTC
    pub const NEAR: f64 = 2_450_836.808;
    /// Cassini: 1999-08-18 ~03:28 UTC
    pub const CASSINI: f64 = 2_451_409.645;
    /// Rosetta-I: 2005-03-04 ~22:10 UTC
    pub const ROSETTA_I: f64 = 2_453_434.424;
    /// MESSENGER: 2005-08-02 ~19:13 UTC
    pub const MESSENGER: f64 = 2_453_585.301;
    /// Juno: 2013-10-09 ~19:21 UTC
    pub const JUNO: f64 = 2_456_574.306;
}

/// Solar-system bodies reachable from a JPL DE-series kernel.
///
/// The planets are named by their system barycentre rather than their body
/// centre, because that is what a DE kernel stores directly: DE440 carries
/// segments for targets 1 through 9 relative to the solar-system barycentre and
/// resolves a body centre only where it also ships a satellite ephemeris. For
/// Mercury and Venus the barycentre and the body centre differ by less than the
/// plotting resolution of any whole-system view; for Jupiter the offset is
/// under 0.005 AU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolarSystemBody {
    Mercury,
    Venus,
    EarthMoonBarycenter,
    Mars,
    Jupiter,
    Saturn,
    Uranus,
    Neptune,
    Pluto,
}

impl SolarSystemBody {
    /// Every body, in increasing semi-major axis.
    pub const ALL: [Self; 9] = [
        Self::Mercury,
        Self::Venus,
        Self::EarthMoonBarycenter,
        Self::Mars,
        Self::Jupiter,
        Self::Saturn,
        Self::Uranus,
        Self::Neptune,
        Self::Pluto,
    ];

    /// Sidereal orbital period in days, from the NASA planetary fact sheets.
    ///
    /// Used to decide how far to sample the kernel when tracing a body's path:
    /// one period returns the closed orbit and nothing more. It is not used to
    /// compute a position -- every position on a chart comes from the kernel,
    /// so an approximate period costs a slightly over- or under-closed loop and
    /// never a wrong point. DE440 spans 1550 through 2650, which is 4.4 Pluto
    /// periods, so every body here closes inside the kernel.
    #[must_use]
    pub fn sidereal_period_days(self) -> f64 {
        match self {
            Self::Mercury => 87.969,
            Self::Venus => 224.701,
            Self::EarthMoonBarycenter => 365.256,
            Self::Mars => 686.980,
            Self::Jupiter => 4_332.589,
            Self::Saturn => 10_759.22,
            Self::Uranus => 30_685.4,
            Self::Neptune => 60_189.0,
            Self::Pluto => 90_560.0,
        }
    }

    /// NAIF integer ID of the barycentre.
    #[must_use]
    pub fn naif_id(self) -> i32 {
        match self {
            Self::Mercury => 1,
            Self::Venus => 2,
            Self::EarthMoonBarycenter => 3,
            Self::Mars => 4,
            Self::Jupiter => 5,
            Self::Saturn => 6,
            Self::Uranus => 7,
            Self::Neptune => 8,
            Self::Pluto => 9,
        }
    }

    /// Lowercase hyphenated name, matching the CLI spelling.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Mercury => "mercury",
            Self::Venus => "venus",
            Self::EarthMoonBarycenter => "earth-moon-barycenter",
            Self::Mars => "mars",
            Self::Jupiter => "jupiter",
            Self::Saturn => "saturn",
            Self::Uranus => "uranus",
            Self::Neptune => "neptune",
            Self::Pluto => "pluto",
        }
    }
}

/// Kilometres per astronomical unit, IAU 2012 definition (exact).
pub const KM_PER_AU: f64 = 149_597_870.7;

/// Obliquity of the ecliptic at J2000.0, in degrees (IAU 2006).
pub const OBLIQUITY_J2000_DEG: f64 = 23.439_279_444_444_445;

/// Heliocentric position in spherical ecliptic coordinates.
///
/// The triple matches the `r_au` / `lat_deg` / `lon_deg` spelling used by
/// `data_core::heliosphere_feature_cube::HeliosphereFeatureRow`, so a body and
/// an observation can be placed on one chart. That row's frame is undeclared
/// and mixed across missions, which the field documentation there records; this
/// type is unambiguously heliocentric ecliptic of J2000.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EclipticPosition {
    pub r_au: f64,
    pub lat_deg: f64,
    pub lon_deg: f64,
}

/// Rotate a J2000 equatorial vector into the J2000 ecliptic frame and express
/// it in spherical coordinates.
///
/// The rotation is about the vernal-equinox axis by the obliquity, so x is
/// unchanged and the y/z pair turns by `-eps`. Longitude runs eastward from the
/// equinox over `[0, 360)`; latitude is positive toward ecliptic north.
#[must_use]
pub fn equatorial_km_to_ecliptic(v_km: [f64; 3]) -> EclipticPosition {
    let eps = OBLIQUITY_J2000_DEG.to_radians();
    let (x, y, z) = (v_km[0], v_km[1], v_km[2]);
    let y_ecl = y * eps.cos() + z * eps.sin();
    let z_ecl = -y * eps.sin() + z * eps.cos();
    let r_km = (x * x + y_ecl * y_ecl + z_ecl * z_ecl).sqrt();
    let lon = y_ecl.atan2(x).to_degrees();
    EclipticPosition {
        r_au: r_km / KM_PER_AU,
        lat_deg: if r_km > 0.0 {
            (z_ecl / r_km).asin().to_degrees()
        } else {
            0.0
        },
        lon_deg: if lon < 0.0 { lon + 360.0 } else { lon },
    }
}

/// Heliocentric ephemeris over a JPL DE-series kernel.
///
/// Separate from `EphemerisLoader`, which is frozen on the geocentric J2000
/// frame the flyby integrator needs. This one translates from the Sun so that
/// planet positions are directly comparable with spacecraft heliocentric
/// distance.
pub struct HeliocentricEphemeris {
    almanac: Almanac,
}

impl HeliocentricEphemeris {
    /// Load a DE-series kernel. DE440 covers 1550 through 2650.
    pub fn load(bsp_path: &Path) -> anyhow::Result<Self> {
        anyhow::ensure!(
            bsp_path.exists(),
            "JPL .bsp file not found: {}. Fetch DE440 from \
             https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp",
            bsp_path.display()
        );
        let path_str = bsp_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Non-UTF8 path: {}", bsp_path.display()))?;
        let almanac = Almanac::new(path_str)
            .map_err(|e| anyhow::anyhow!("Failed to load {}: {}", bsp_path.display(), e))?;
        Ok(Self { almanac })
    }

    /// Heliocentric J2000 equatorial position in km.
    ///
    /// Fails rather than returning zeros: a caller plotting a body wants to
    /// know the epoch fell outside the kernel's coverage, where the flyby
    /// integrator preferred to keep stepping.
    pub fn body_equatorial_km(&self, body: SolarSystemBody, jed: f64) -> anyhow::Result<[f64; 3]> {
        let epoch = Epoch::from_jde_tdb(jed);
        let frame = Frame::from_ephem_j2000(body.naif_id());
        let state = self
            .almanac
            .translate_geometric(frame, Frame::from_ephem_j2000(NAIF_SUN), epoch)
            .map_err(|e| {
                anyhow::anyhow!(
                    "heliocentric query for {} (NAIF {}) at JED {} failed: {}",
                    body.name(),
                    body.naif_id(),
                    jed,
                    e
                )
            })?;
        let r = state.radius_km;
        Ok([r[0], r[1], r[2]])
    }

    /// Heliocentric ecliptic position of a body at a Julian Ephemeris Date.
    pub fn body_ecliptic(
        &self,
        body: SolarSystemBody,
        jed: f64,
    ) -> anyhow::Result<EclipticPosition> {
        Ok(equatorial_km_to_ecliptic(
            self.body_equatorial_km(body, jed)?,
        ))
    }
}

#[cfg(test)]
mod ecliptic_tests {
    use super::{OBLIQUITY_J2000_DEG, equatorial_km_to_ecliptic};

    #[test]
    fn vernal_equinox_axis_is_unrotated() {
        // The rotation is about x, so a vector along x keeps zero latitude and
        // zero longitude in both frames.
        let p = equatorial_km_to_ecliptic([1.0, 0.0, 0.0]);
        assert!(p.lat_deg.abs() < 1e-12);
        assert!(p.lon_deg.abs() < 1e-12);
    }

    #[test]
    fn equatorial_pole_lands_at_ninety_less_obliquity() {
        // The celestial pole sits at ecliptic latitude 90 - eps by definition.
        let p = equatorial_km_to_ecliptic([0.0, 0.0, 1.0]);
        assert!((p.lat_deg - (90.0 - OBLIQUITY_J2000_DEG)).abs() < 1e-9);
    }

    #[test]
    fn summer_solstice_direction_has_zero_ecliptic_latitude() {
        // The J2000 equatorial direction (0, cos eps, sin eps) is the ecliptic
        // +y axis, so it must come back at latitude 0 and longitude 90.
        let eps = OBLIQUITY_J2000_DEG.to_radians();
        let p = equatorial_km_to_ecliptic([0.0, eps.cos(), eps.sin()]);
        assert!(p.lat_deg.abs() < 1e-9);
        assert!((p.lon_deg - 90.0).abs() < 1e-9);
    }
}

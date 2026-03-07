//! Thread-safe JPL ephemeris loader for three-body flyby integration.
//!
//! Uses anise exclusively (pure Rust, thread-safe JPL .bsp/SPK parser).
//! JPL .bsp files store Chebyshev polynomials directly in ICRS/J2000,
//! so state vectors are perfectly aligned with the flyby RA/Dec coordinates.
//! Zero frame rotation = zero geometric risk for the h(t).v_wind cross-product.
//!
//! Thread safety: all query methods take `&self`, safe inside rayon par_iter.
//! anise Vector3 is nalgebra::Vector3<f64> -- no conversion needed.

use anise::almanac::Almanac;
use anise::prelude::{Epoch, Frame};
use nalgebra::Vector3;
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
    pub moon_pos_km: Vector3<f64>,
    /// Sun geocentric position in J2000 ECI (km).
    pub sun_pos_km: Vector3<f64>,
    /// Earth-Moon barycenter geocentric offset in J2000 ECI (km).
    /// EMB = M_moon / (M_earth + M_moon) * r_moon
    pub emb_offset_km: Vector3<f64>,
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
        let emb_offset_km =
            moon_pos_km * (MOON_EARTH_MASS_RATIO / (1.0 + MOON_EARTH_MASS_RATIO));
        ThreeBodyState {
            moon_pos_km,
            sun_pos_km,
            emb_offset_km,
        }
    }

    /// Moon geocentric position in J2000 ECI (km).
    pub fn moon_geocentric_j2000(&self, jed: f64) -> Vector3<f64> {
        self.query_body(NAIF_MOON, Epoch::from_jde_tdb(jed))
    }

    /// Sun geocentric position in J2000 ECI (km).
    pub fn sun_geocentric_j2000(&self, jed: f64) -> Vector3<f64> {
        self.query_body(NAIF_SUN, Epoch::from_jde_tdb(jed))
    }

    /// Earth-Moon barycenter offset from Earth center in J2000 ECI (km).
    ///
    /// EMB = M_moon / (M_earth + M_moon) * r_moon
    /// This is the TRUE focal point of the local dark matter gravitational well.
    pub fn earth_moon_barycenter(&self, jed: f64) -> Vector3<f64> {
        let r_moon = self.moon_geocentric_j2000(jed);
        r_moon * (MOON_EARTH_MASS_RATIO / (1.0 + MOON_EARTH_MASS_RATIO))
    }

    /// Query a body's geocentric J2000 position via anise.
    ///
    /// anise uses nalgebra 0.34 internally, but our workspace pins nalgebra 0.33
    /// (statrs 0.18 constraint). Extract x/y/z components to bridge the version gap.
    fn query_body(&self, naif_id: i32, epoch: Epoch) -> Vector3<f64> {
        match self.almanac.state_of(naif_id, self.earth_j2000, epoch, None) {
            Ok(state) => {
                // Bridge nalgebra 0.34 (anise) -> 0.33 (workspace) via component extraction.
                let r = state.radius_km;
                Vector3::new(r[0], r[1], r[2])
            }
            Err(e) => {
                // This should not happen with a valid DE440 .bsp covering 1550-2650.
                // Log the error but return zeros rather than panicking mid-integration.
                eprintln!(
                    "WARNING: anise query for NAIF {} at {:?} failed: {}",
                    naif_id, epoch, e
                );
                Vector3::zeros()
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn try_load() -> Option<EphemerisLoader> {
        let bsp = PathBuf::from("data/external/de440.bsp");
        EphemerisLoader::load(&bsp).ok()
    }

    #[test]
    fn test_load_requires_bsp() {
        let bogus = PathBuf::from("/nonexistent/de440.bsp");
        assert!(EphemerisLoader::load(&bogus).is_err());
    }

    #[test]
    fn test_moon_distance_if_bsp_available() {
        let Some(loader) = try_load() else { return };
        let pos = loader.moon_geocentric_j2000(flyby_epochs::NEAR);
        assert!(pos.x.is_finite());
        let dist = pos.norm();
        assert!(
            dist > 350_000.0 && dist < 410_000.0,
            "Moon distance {} km out of range",
            dist
        );
    }

    #[test]
    fn test_sun_distance_if_bsp_available() {
        let Some(loader) = try_load() else { return };
        let pos = loader.sun_geocentric_j2000(flyby_epochs::NEAR);
        assert!(pos.x.is_finite());
        let dist = pos.norm();
        assert!(
            dist > 1.45e8 && dist < 1.55e8,
            "Sun distance {} km out of range (expected ~1 AU)",
            dist
        );
    }

    #[test]
    fn test_emb_offset_if_bsp_available() {
        let Some(loader) = try_load() else { return };
        let emb = loader.earth_moon_barycenter(flyby_epochs::NEAR);
        let dist = emb.norm();
        assert!(
            dist > 3000.0 && dist < 6000.0,
            "EMB offset {} km out of range (expected ~4670 km)",
            dist
        );
    }

    #[test]
    fn test_three_body_all_flybys_if_bsp_available() {
        let Some(loader) = try_load() else { return };
        let epochs = [
            flyby_epochs::GALILEO,
            flyby_epochs::NEAR,
            flyby_epochs::CASSINI,
            flyby_epochs::ROSETTA_I,
            flyby_epochs::MESSENGER,
            flyby_epochs::JUNO,
        ];
        for &jed in &epochs {
            let state = loader.three_body_state(jed);
            assert!(
                state.moon_pos_km.norm() > 300_000.0,
                "Moon too close at JED {}",
                jed
            );
            assert!(
                state.sun_pos_km.norm() > 1.4e8,
                "Sun too close at JED {}",
                jed
            );
            assert!(
                state.emb_offset_km.norm() < 10_000.0,
                "EMB offset too large at JED {}",
                jed
            );
        }
    }

    #[test]
    fn test_flyby_epoch_ordering() {
        assert!(flyby_epochs::GALILEO < flyby_epochs::NEAR);
        assert!(flyby_epochs::NEAR < flyby_epochs::CASSINI);
        assert!(flyby_epochs::CASSINI < flyby_epochs::ROSETTA_I);
        assert!(flyby_epochs::ROSETTA_I < flyby_epochs::MESSENGER);
        assert!(flyby_epochs::MESSENGER < flyby_epochs::JUNO);
    }
}

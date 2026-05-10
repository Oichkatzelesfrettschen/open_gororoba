//! IBEX (Interstellar Boundary Explorer) ENA flux sky map parser.
//!
//! IBEX does NOT measure in-situ solar wind plasma. It detects
//! Energetic Neutral Atoms (ENAs) arriving from the heliospheric
//! boundary (termination shock, heliosheath, heliopause).
//!
//! IBEX data is used for heliospheric boundary GEOMETRY constraints,
//! not for OmniRecord time-series. The data product is a sky map of
//! ENA flux in ecliptic coordinates, integrated over 6-month orbits.
//!
//! Instruments:
//!   IBEX-Lo: 10 eV - 2 keV ENAs (Fuselier et al., 2009)
//!   IBEX-Hi: 300 eV - 6 keV ENAs (Funsten et al., 2009)
//!
//! Data product: 6-degree x 6-degree sky maps in ecliptic coordinates,
//! 6 energy channels per instrument, 6-month time bins.
//!
//! Key science: the "IBEX ribbon" -- an arc of enhanced ENA flux
//! aligned with the local interstellar magnetic field direction.
//! Constrains the interstellar B-field orientation (~3 uG, l~221, b~39).
//!
//! Source:
//!   ENA sky maps: <https://spdf.gsfc.nasa.gov/pub/data/ibex/release17/>
//!   Orbit time series: IBEX_OR_SSC via CDAWeb HAPI
//!     <https://cdaweb.gsfc.nasa.gov/hapi/info?id=IBEX_OR_SSC>

use crate::{
    fetcher::FetchError,
    parse::{parse_hapi_spacephysics_f64_or_nan, parse_hapi_time_to_ydh},
};
use csv::ReaderBuilder;
use std::collections::BTreeMap;

/// IBEX ENA sky map pixel.
#[derive(Debug, Clone)]
pub struct IbexEnaPixel {
    /// Ecliptic longitude center (deg), 0-360.
    pub ecl_lon: f64,
    /// Ecliptic latitude center (deg), -90 to +90.
    pub ecl_lat: f64,
    /// ENA flux (cm^-2 s^-1 sr^-1 keV^-1). NaN if no data.
    pub flux: f64,
    /// Statistical uncertainty on flux. NaN if unavailable.
    pub flux_err: f64,
}

/// IBEX ENA sky map for one energy channel and one orbit.
#[derive(Debug, Clone)]
pub struct IbexEnaMap {
    /// Energy channel center (keV).
    pub energy_kev: f64,
    /// Orbit number (1-based).
    pub orbit: u16,
    /// Start year of 6-month integration.
    pub year_start: u16,
    /// Instrument: "Lo" or "Hi".
    pub instrument: String,
    /// Sky map pixels (6 deg x 6 deg grid = 60 x 30 = 1800 pixels).
    pub pixels: Vec<IbexEnaPixel>,
}

/// IBEX orbit/support time-series row from CDAWeb HAPI.
#[derive(Debug, Clone)]
pub struct IbexOrbitRecord {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub radius_re: f64,
}

/// Parse IBEX ENA sky map from a CSV-like format.
///
/// Expected columns: ecl_lon, ecl_lat, flux, flux_err
/// Lines starting with '#' are comments.
pub fn parse_ibex_ena_map(
    content: &str,
    energy_kev: f64,
    orbit: u16,
    year_start: u16,
    instrument: &str,
) -> IbexEnaMap {
    let mut pixels = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 3 {
            continue;
        }

        let ecl_lon: f64 = match fields[0].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let ecl_lat: f64 = match fields[1].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let flux: f64 = fields[2].parse().unwrap_or(f64::NAN);
        let flux_err: f64 = if fields.len() > 3 {
            fields[3].parse().unwrap_or(f64::NAN)
        } else {
            f64::NAN
        };

        pixels.push(IbexEnaPixel {
            ecl_lon,
            ecl_lat,
            flux,
            flux_err,
        });
    }

    IbexEnaMap {
        energy_kev,
        orbit,
        year_start,
        instrument: instrument.to_string(),
        pixels,
    }
}

/// Parse IBEX ENA sky map from a file.
pub fn parse_ibex_ena_file(
    path: &std::path::Path,
    energy_kev: f64,
    orbit: u16,
    year_start: u16,
    instrument: &str,
) -> Result<IbexEnaMap, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("read error: {}", e)))?;
    Ok(parse_ibex_ena_map(
        &content, energy_kev, orbit, year_start, instrument,
    ))
}

/// Parse IBEX orbit/support HAPI CSV from CDAWeb.
///
/// Deduplicates to one record per (year, doy, hour) -- the first occurrence wins.
pub fn parse_ibex_orbit_hapi_csv(content: &str) -> Vec<IbexOrbitRecord> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(content.as_bytes());
    let mut hourly: BTreeMap<(u16, u16, u8), IbexOrbitRecord> = BTreeMap::new();
    for record in reader.records().flatten() {
        let Some(time) = record.get(0) else {
            continue;
        };
        let Some((year, doy, hour)) = parse_hapi_time_to_ydh(time) else {
            continue;
        };
        let radius_re = parse_hapi_spacephysics_f64_or_nan(record.get(1).unwrap_or(""));
        hourly.entry((year, doy, hour)).or_insert(IbexOrbitRecord {
            year,
            doy,
            hour,
            radius_re,
        });
    }
    hourly.into_values().collect()
}

/// Parse IBEX orbit/support data from a local file.
pub fn parse_ibex_orbit_file(path: &std::path::Path) -> Result<Vec<IbexOrbitRecord>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("read error: {}", e)))?;
    Ok(parse_ibex_orbit_hapi_csv(&content))
}

// Fetch/provider support moved to ibex_fetch (feature-gated on `fetch`).

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_ibex_sky_map_basic() {
        let data = "# IBEX-Hi ENA sky map, orbit 1, 1.1 keV\n\
                     3.0 87.0 1.5e4 500.0\n\
                     9.0 87.0 1.2e4 400.0\n\
                     3.0 81.0 2.0e4 600.0\n";
        let map = parse_ibex_ena_map(data, 1.1, 1, 2009, "Hi");
        assert_eq!(map.pixels.len(), 3);
        assert!((map.energy_kev - 1.1).abs() < 0.01);
        assert_eq!(map.orbit, 1);
        assert_eq!(map.instrument, "Hi");
        assert!((map.pixels[0].ecl_lon - 3.0).abs() < 0.01);
        assert!((map.pixels[0].ecl_lat - 87.0).abs() < 0.01);
        assert!((map.pixels[0].flux - 1.5e4).abs() < 1.0);
    }

    #[test]
    fn test_parse_ibex_ribbon_region() {
        // The IBEX ribbon has enhanced flux (~2x background)
        // at specific ecliptic coordinates
        let data = "# Ribbon region\n\
                     220.0 40.0 3.0e4 800.0\n\
                     220.0 -40.0 1.5e4 500.0\n";
        let map = parse_ibex_ena_map(data, 1.1, 1, 2009, "Hi");
        assert_eq!(map.pixels.len(), 2);
        // Ribbon pixel should have higher flux
        assert!(map.pixels[0].flux > map.pixels[1].flux);
    }

    #[test]
    fn test_parse_ibex_no_error_column() {
        let data = "3.0 87.0 1.5e4\n\
                     9.0 87.0 1.2e4\n";
        let map = parse_ibex_ena_map(data, 0.7, 5, 2011, "Lo");
        assert_eq!(map.pixels.len(), 2);
        assert!(map.pixels[0].flux_err.is_nan(), "no error column");
        assert!((map.pixels[0].flux - 1.5e4).abs() < 1.0);
    }

    #[test]
    fn test_parse_ibex_orbit_hapi_csv() {
        let data = "Time,RADIUS\n2016-01-01T00:00:27.000Z,28.66\n2016-01-01T00:15:27.000Z,28.67\n2016-01-01T01:00:27.000Z,28.70\n";
        let rows = parse_ibex_orbit_hapi_csv(data);
        assert_eq!(
            rows.len(),
            2,
            "rows are hourly-deduplicated for overlay use"
        );
        assert_eq!(rows[0].year, 2016);
        assert_eq!(rows[0].doy, 1);
        assert_eq!(rows[0].hour, 0);
        assert!((rows[0].radius_re - 28.66).abs() < 0.01);
    }
}

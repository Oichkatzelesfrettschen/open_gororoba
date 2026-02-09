//! Fermi GBM Burst Catalog parser and fetcher.
//!
//! The Fermi Gamma-ray Burst Monitor (GBM) catalog contains GRBs detected
//! by the Fermi satellite. Fetched via the HEASARC Xamin API.
//!
//! Source: https://heasarc.gsfc.nasa.gov/W3Browse/fermi/fermigbrst.html

use crate::fetcher::{download_heasarc_csv, DatasetProvider, FetchConfig, FetchError};
use std::path::{Path, PathBuf};

/// A gamma-ray burst from the Fermi GBM catalog.
///
/// Field names match HEASARC `fermigbrst` table columns.
/// Catalog docs: <https://heasarc.gsfc.nasa.gov/W3Browse/fermi/fermigbrst.html>
#[derive(Debug, Clone)]
pub struct GrbEvent {
    /// GRB name (e.g., GRB080714086).
    pub name: String,
    /// Trigger time (UTC string from HEASARC, converted from MET).
    pub trigger_time: String,
    /// Right ascension (sexagesimal string, e.g., "03 41 21.2").
    pub ra: String,
    /// Declination (sexagesimal string, e.g., "-89 00 33").
    pub dec: String,
    /// T90 duration (seconds).
    pub t90: f64,
    /// T50 duration (seconds) -- not always present.
    pub t50: f64,
    /// Fluence in 10-1000 keV band (erg/cm^2).
    pub fluence: f64,
    /// Peak flux on 64ms timescale (ph/cm^2/s).
    /// HEASARC column: `flux_64`.
    pub flux_64: f64,
    /// Peak flux on 1024ms timescale (ph/cm^2/s).
    /// HEASARC column: `flux_1024`.
    pub flux_1024: f64,
    /// Best-fit spectral model for time-integrated (fluence) spectrum.
    /// HEASARC column: `flnc_best_fitting_model`.
    pub flnc_best_fitting_model: String,
    /// Best-fit spectral model for peak flux spectrum.
    /// HEASARC column: `pflx_best_fitting_model`.
    pub pflx_best_fitting_model: String,
}

fn parse_f64(s: &str) -> f64 {
    let s = s.trim();
    if s.is_empty() || s == "null" || s == "NULL" || s == "nan" {
        return f64::NAN;
    }
    s.parse::<f64>().unwrap_or(f64::NAN)
}

/// Parse Fermi GBM burst catalog CSV.
pub fn parse_fermi_gbm_csv(path: &Path) -> Result<Vec<GrbEvent>, FetchError> {
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .has_headers(true)
        .comment(Some(b'#'))
        .from_path(path)
        .map_err(|e| FetchError::Validation(format!("CSV read error: {}", e)))?;

    let headers = reader
        .headers()
        .map_err(|e| FetchError::Validation(format!("Header read error: {}", e)))?
        .clone();

    // Use exact match to avoid `.contains()` confusion
    // (e.g., "pflx_best" matching "pflx_best_fitting_model").
    let col_exact = |name: &str| -> Option<usize> {
        headers
            .iter()
            .position(|h| h.trim().eq_ignore_ascii_case(name))
    };

    let idx_name = col_exact("name").or_else(|| col_exact("trigger_name"));
    let idx_time = col_exact("trigger_time");
    let idx_ra = col_exact("ra");
    let idx_dec = col_exact("dec");
    let idx_t90 = col_exact("t90");
    let idx_t50 = col_exact("t50");
    let idx_fluence = col_exact("fluence");
    // Real HEASARC columns for peak flux (not "pflx_best" which doesn't exist)
    let idx_flux_64 = col_exact("flux_64");
    let idx_flux_1024 = col_exact("flux_1024");
    // Real HEASARC columns for spectral model type
    let idx_flnc_model = col_exact("flnc_best_fitting_model");
    let idx_pflx_model = col_exact("pflx_best_fitting_model");

    let get_str = |record: &csv::StringRecord, idx: Option<usize>| -> String {
        idx.and_then(|i| record.get(i))
            .unwrap_or("")
            .trim()
            .to_string()
    };

    let get_f64 = |record: &csv::StringRecord, idx: Option<usize>| -> f64 {
        idx.and_then(|i| record.get(i))
            .map(parse_f64)
            .unwrap_or(f64::NAN)
    };

    let mut events = Vec::new();
    for result in reader.records() {
        let record = match result {
            Ok(r) => r,
            Err(_) => continue,
        };

        let name = get_str(&record, idx_name);
        if name.is_empty() {
            continue;
        }

        events.push(GrbEvent {
            name,
            trigger_time: get_str(&record, idx_time),
            ra: get_str(&record, idx_ra),
            dec: get_str(&record, idx_dec),
            t90: get_f64(&record, idx_t90),
            t50: get_f64(&record, idx_t50),
            fluence: get_f64(&record, idx_fluence),
            flux_64: get_f64(&record, idx_flux_64),
            flux_1024: get_f64(&record, idx_flux_1024),
            flnc_best_fitting_model: get_str(&record, idx_flnc_model),
            pflx_best_fitting_model: get_str(&record, idx_pflx_model),
        });
    }

    Ok(events)
}

/// W3Browse batch query URL for the Fermi GBM burst catalog.
///
/// HEASARC TAP only supports VOTable format (not CSV), so we use the W3Browse
/// batch endpoint which returns pipe-delimited text. The `download_heasarc_csv`
/// function converts pipe-delimited output to standard CSV.
///
/// `ResultMax=0` returns all rows; `displaymode=BatchDisplay` returns
/// pipe-delimited; `Fields=` selects specific columns.
///
/// Correct column names verified against HEASARC fermigbrst table (2026-02):
/// - `flux_64` / `flux_1024`: peak photon flux (ph/cm^2/s) on 64ms / 1024ms
/// - `flnc_best_fitting_model`: time-integrated spectral model
/// - `pflx_best_fitting_model`: peak-flux spectral model
const FERMI_GBM_URL: &str = "\
https://heasarc.gsfc.nasa.gov/db-perl/W3Browse/w3query.pl?\
tablehead=name%3Dfermigbrst&\
Action=Query&\
Coordinates=Equatorial&\
Equinox=2000&\
Radius=Default&\
NR=&\
GIFsize=0&\
Fields=name%2Ctrigger_time%2Cra%2Cdec%2Ct90%2Ct50%2Cfluence%2Cflux_64%2Cflux_1024%2Cflnc_best_fitting_model%2Cpflx_best_fitting_model&\
ResultMax=0&\
displaymode=BatchDisplay";

/// Fermi GBM burst catalog dataset provider.
pub struct FermiGbmProvider;

impl DatasetProvider for FermiGbmProvider {
    fn name(&self) -> &str {
        "Fermi GBM Burst Catalog"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("fermi_gbm_grbs.csv");
        if config.skip_existing && output.exists() {
            eprintln!("  {} already cached at {}", self.name(), output.display());
            return Ok(output);
        }
        eprintln!("  Downloading {} from HEASARC W3Browse...", self.name());
        let bytes = download_heasarc_csv(FERMI_GBM_URL, &output)?;
        eprintln!("  Saved {} bytes to {}", bytes, output.display());
        Ok(output)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("fermi_gbm_grbs.csv").exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::Path;
    use tempfile::NamedTempFile;

    fn write_temp_csv(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn test_parse_fermi_synthetic_basic() {
        let csv = "\
name,trigger_time,ra,dec,t90,t50,fluence,flux_64,flux_1024,flnc_best_fitting_model,pflx_best_fitting_model
GRB080714086,2008-07-14T02:04:12,03 41 21.2,-89 00 33,12.5,6.3,1.2e-6,3.5,2.1,pflx_band,pflx_comp
GRB090101001,2009-01-01T00:01:00,12 00 00.0,+45 30 00,0.5,0.2,5.0e-7,8.0,4.0,pflx_comp,pflx_comp
";
        let f = write_temp_csv(csv);
        let events = parse_fermi_gbm_csv(f.path()).unwrap();
        assert_eq!(events.len(), 2, "Should parse 2 GRB events");
        assert_eq!(events[0].name, "GRB080714086");
        assert!((events[0].t90 - 12.5).abs() < 0.01);
        assert_eq!(events[0].ra, "03 41 21.2");
        assert_eq!(events[0].flnc_best_fitting_model, "pflx_band");
        assert!((events[1].flux_64 - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_fermi_handles_null_values() {
        let csv = "\
name,trigger_time,ra,dec,t90,t50,fluence,flux_64,flux_1024,flnc_best_fitting_model,pflx_best_fitting_model
GRB_NULL,2010-01-01T00:00:00,,,null,null,null,null,null,,
";
        let f = write_temp_csv(csv);
        let events = parse_fermi_gbm_csv(f.path()).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].name, "GRB_NULL");
        assert!(events[0].t90.is_nan(), "null should parse as NaN");
        assert!(events[0].flux_64.is_nan(), "null should parse as NaN");
        assert!(events[0].ra.is_empty(), "empty RA should be empty string");
    }

    #[test]
    fn test_parse_fermi_skips_empty_name() {
        let csv = "\
name,trigger_time,ra,dec,t90,t50,fluence,flux_64,flux_1024,flnc_best_fitting_model,pflx_best_fitting_model
,2010-01-01T00:00:00,10 00 00,20 00 00,1.0,0.5,1e-7,2.0,1.0,band,comp
GRB_VALID,2010-02-01T00:00:00,10 00 00,20 00 00,2.0,1.0,2e-7,3.0,1.5,band,comp
";
        let f = write_temp_csv(csv);
        let events = parse_fermi_gbm_csv(f.path()).unwrap();
        assert_eq!(events.len(), 1, "Should skip row with empty name");
        assert_eq!(events[0].name, "GRB_VALID");
    }

    #[test]
    fn test_parse_fermi_alternate_column_name() {
        // Some HEASARC downloads use trigger_name instead of name
        let csv = "\
trigger_name,trigger_time,ra,dec,t90,t50,fluence,flux_64,flux_1024,flnc_best_fitting_model,pflx_best_fitting_model
GRB_ALT,2010-01-01T00:00:00,10 00 00,20 00 00,1.0,0.5,1e-7,2.0,1.0,band,comp
";
        let f = write_temp_csv(csv);
        let events = parse_fermi_gbm_csv(f.path()).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].name, "GRB_ALT");
    }

    #[test]
    fn test_parse_fermi_empty_csv() {
        let csv = "name,trigger_time,ra,dec,t90\n";
        let f = write_temp_csv(csv);
        let events = parse_fermi_gbm_csv(f.path()).unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn test_parse_fermi_gbm_csv_if_available() {
        let path = Path::new("data/external/fermi_gbm_grbs.csv");
        if !path.exists() {
            eprintln!("Skipping: Fermi GBM CSV not available (run fetch-datasets first)");
            return;
        }

        let events = parse_fermi_gbm_csv(path).expect("Failed to parse Fermi GBM CSV");
        assert!(!events.is_empty(), "Should parse at least one GRB event");

        let first = &events[0];
        assert!(!first.name.is_empty(), "Name should not be empty");
        assert!(
            first.t90.is_finite(),
            "T90 should be finite for first event"
        );
        assert!(
            first.fluence.is_finite(),
            "Fluence should be finite for first event"
        );

        let with_flux = events.iter().filter(|e| e.flux_64.is_finite()).count();
        assert!(
            with_flux > 100,
            "Should have >100 events with flux_64, got {}",
            with_flux
        );

        eprintln!(
            "Parsed {} Fermi GBM events ({} with flux_64)",
            events.len(),
            with_flux
        );
    }
}

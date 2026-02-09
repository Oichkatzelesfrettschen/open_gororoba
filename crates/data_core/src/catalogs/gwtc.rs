//! GWTC-3 gravitational wave transient catalog parser.
//!
//! Parses the GWTC-3 confident events CSV from GWOSC (Gravitational Wave
//! Open Science Center).
//!
//! Source: https://gwosc.org/eventapi/json/GWTC-3-confident/
//! Reference: Abbott et al. (2023), PRX 13, 041039

use crate::fetcher::{download_with_fallbacks, DatasetProvider, FetchConfig, FetchError};
use std::path::{Path, PathBuf};

/// A gravitational wave event from GWTC-3.
#[derive(Debug, Clone)]
pub struct GwEvent {
    pub id: String,
    pub common_name: String,
    /// Primary source mass (solar masses).
    pub mass_1_source: f64,
    pub mass_1_source_lower: f64,
    pub mass_1_source_upper: f64,
    /// Secondary source mass (solar masses).
    pub mass_2_source: f64,
    pub mass_2_source_lower: f64,
    pub mass_2_source_upper: f64,
    /// Chirp mass (solar masses).
    pub chirp_mass_source: f64,
    /// Effective spin parameter.
    pub chi_eff: f64,
    /// Luminosity distance (Mpc).
    pub luminosity_distance: f64,
    pub luminosity_distance_lower: f64,
    pub luminosity_distance_upper: f64,
    /// Cosmological redshift.
    pub redshift: f64,
    pub redshift_lower: f64,
    pub redshift_upper: f64,
    /// Network SNR.
    pub snr: f64,
    /// Probability of astrophysical origin.
    pub p_astro: f64,
    /// Total source mass (solar masses).
    pub total_mass_source: f64,
    /// Final remnant mass (solar masses).
    pub final_mass_source: f64,
}

/// Parse a float field, returning 0.0 for empty or unparseable values.
fn parse_f64(s: &str) -> f64 {
    s.trim().parse::<f64>().unwrap_or(0.0)
}

/// Parse any GWOSC event CSV (GWTC-3 confident or combined catalog).
///
/// Handles both v1 (eventapi) and v2 (api/v2) column name conventions:
/// - v1: `id`, `commonName`
/// - v2: `name`, `shortName`
///
/// Physics columns are the same in both formats.
pub fn parse_gwtc3_csv(path: &Path) -> Result<Vec<GwEvent>, FetchError> {
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .has_headers(true)
        .from_path(path)
        .map_err(|e| FetchError::Validation(format!("CSV read error: {}", e)))?;

    let headers = reader
        .headers()
        .map_err(|e| FetchError::Validation(format!("Header read error: {}", e)))?
        .clone();

    // Find column indices -- case-insensitive with fallbacks for both
    // v1 (eventapi) and v2 (api/v2) CSV formats.
    let col = |name: &str| -> Option<usize> { headers.iter().position(|h| h == name) };
    let col_ci =
        |name: &str| -> Option<usize> { headers.iter().position(|h| h.eq_ignore_ascii_case(name)) };

    // v1 has "id", v2 has "name"
    let idx_id = col("id").or_else(|| col("name"));
    // v1 has "commonName", v2 has "shortName"
    let idx_common = col("commonName").or_else(|| col("shortName"));
    let idx_m1 = col("mass_1_source");
    let idx_m1_lo = col("mass_1_source_lower");
    let idx_m1_up = col("mass_1_source_upper");
    let idx_m2 = col("mass_2_source");
    let idx_m2_lo = col("mass_2_source_lower");
    let idx_m2_up = col("mass_2_source_upper");
    let idx_chirp = col("chirp_mass_source");
    let idx_chi = col("chi_eff");
    let idx_dl = col("luminosity_distance");
    let idx_dl_lo = col("luminosity_distance_lower");
    let idx_dl_up = col("luminosity_distance_upper");
    let idx_z = col("redshift");
    let idx_z_lo = col("redshift_lower");
    let idx_z_up = col("redshift_upper");
    let idx_snr =
        col("network_matched_filter_snr").or_else(|| col_ci("network_matched_filter_snr"));
    let idx_p = col("p_astro").or_else(|| col_ci("p_astro"));
    let idx_total = col("total_mass_source").or_else(|| col_ci("total_mass_source"));
    let idx_final = col("final_mass_source").or_else(|| col_ci("final_mass_source"));

    let get = |record: &csv::StringRecord, idx: Option<usize>| -> String {
        idx.and_then(|i| record.get(i)).unwrap_or("").to_string()
    };

    let get_f64 = |record: &csv::StringRecord, idx: Option<usize>| -> f64 {
        idx.and_then(|i| record.get(i))
            .map(parse_f64)
            .unwrap_or(0.0)
    };

    let mut events = Vec::new();
    for result in reader.records() {
        let record =
            result.map_err(|e| FetchError::Validation(format!("Record parse error: {}", e)))?;

        let m1 = get_f64(&record, idx_m1);
        let m2 = get_f64(&record, idx_m2);
        let dl = get_f64(&record, idx_dl);
        let z = get_f64(&record, idx_z);

        // Skip records with missing critical data
        if m1 <= 0.0 || m2 <= 0.0 || dl <= 0.0 {
            continue;
        }

        events.push(GwEvent {
            id: get(&record, idx_id),
            common_name: get(&record, idx_common),
            mass_1_source: m1,
            mass_1_source_lower: get_f64(&record, idx_m1_lo),
            mass_1_source_upper: get_f64(&record, idx_m1_up),
            mass_2_source: m2,
            mass_2_source_lower: get_f64(&record, idx_m2_lo),
            mass_2_source_upper: get_f64(&record, idx_m2_up),
            chirp_mass_source: get_f64(&record, idx_chirp),
            chi_eff: get_f64(&record, idx_chi),
            luminosity_distance: dl,
            luminosity_distance_lower: get_f64(&record, idx_dl_lo),
            luminosity_distance_upper: get_f64(&record, idx_dl_up),
            redshift: z,
            redshift_lower: get_f64(&record, idx_z_lo),
            redshift_upper: get_f64(&record, idx_z_up),
            snr: get_f64(&record, idx_snr),
            p_astro: get_f64(&record, idx_p),
            total_mass_source: get_f64(&record, idx_total),
            final_mass_source: get_f64(&record, idx_final),
        });
    }

    Ok(events)
}

const GWTC3_URLS: &[&str] = &["https://gwosc.org/eventapi/csv/GWTC-3-confident/"];

/// GWTC-3 dataset provider (35 confident O3b events only).
pub struct Gwtc3Provider;

impl DatasetProvider for Gwtc3Provider {
    fn name(&self) -> &str {
        "GWTC-3 confident events"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("GWTC-3_confident.csv");
        download_with_fallbacks(self.name(), GWTC3_URLS, &output, config.skip_existing)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("GWTC-3_confident.csv").exists()
    }
}

/// Combined GWTC catalog URLs (all O1-O4a events, 219 unique).
///
/// The GWOSC v2 API returns a CSV with all default parameters when
/// `include-default-parameters=true` and `pagesize=500` (all events in one
/// page). The combined `GWTC` catalog deduplicates events by keeping the
/// latest catalog version for each.
const GWOSC_COMBINED_URLS: &[&str] = &[
    "https://gwosc.org/api/v2/catalogs/GWTC/events?include-default-parameters=true&format=csv&pagesize=500",
];

/// Combined GWTC dataset provider (all O1 through O4a events).
pub struct GwoscCombinedProvider;

impl DatasetProvider for GwoscCombinedProvider {
    fn name(&self) -> &str {
        "GWOSC combined GWTC (O1-O4a)"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("gwosc_all_events.csv");
        download_with_fallbacks(
            self.name(),
            GWOSC_COMBINED_URLS,
            &output,
            config.skip_existing,
        )
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("gwosc_all_events.csv").exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_parse_gwtc3_csv_if_available() {
        let path = Path::new("data/external/GWTC-3_confident.csv");
        if !path.exists() {
            eprintln!("Skipping: GWTC-3 CSV not available");
            return;
        }

        let events = parse_gwtc3_csv(path).expect("Failed to parse GWTC-3 CSV");
        assert!(!events.is_empty(), "Should parse at least one event");

        // Verify all events have positive masses and distances
        for ev in &events {
            assert!(
                ev.mass_1_source > 0.0,
                "mass_1 should be positive: {}",
                ev.id
            );
            assert!(
                ev.mass_2_source > 0.0,
                "mass_2 should be positive: {}",
                ev.id
            );
            assert!(
                ev.luminosity_distance > 0.0,
                "d_L should be positive: {}",
                ev.id
            );
        }

        eprintln!("Parsed {} GWTC-3 events", events.len());
    }

    #[test]
    fn test_parse_gwosc_combined_csv_if_available() {
        let path = Path::new("data/external/gwosc_all_events.csv");
        if !path.exists() {
            eprintln!("Skipping: GWOSC combined CSV not available");
            return;
        }

        let events = parse_gwtc3_csv(path).expect("Failed to parse GWOSC combined CSV");
        assert!(!events.is_empty(), "Should parse at least one event");

        // Combined catalog should have significantly more events than GWTC-3 alone
        eprintln!("Parsed {} combined GWOSC events", events.len());

        // Verify that events with mass data have positive values
        let with_mass: Vec<_> = events
            .iter()
            .filter(|e| e.mass_1_source > 0.0 && e.mass_2_source > 0.0)
            .collect();
        assert!(
            with_mass.len() > 50,
            "Should have >50 events with mass data, got {}",
            with_mass.len()
        );
        eprintln!("Events with valid mass data: {}", with_mass.len());
    }
}

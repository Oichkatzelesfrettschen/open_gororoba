//! CHIME/FRB catalog parser (Catalog 1 and 2).
//!
//! Parses CHIME/FRB CSV catalogs with flexible column handling.
//!
//! Catalog 1: 536 events, Amiri+ (2021)
//! Catalog 2: 4539 events, CHIME/FRB Collaboration (2025)

use crate::fetcher::{download_with_fallbacks, DatasetProvider, FetchConfig, FetchError};
use std::path::{Path, PathBuf};

/// A single FRB event from the CHIME catalog.
#[derive(Debug, Clone)]
pub struct FrbEvent {
    pub tns_name: String,
    /// Repeater source name (empty for non-repeaters).
    pub repeater_name: String,
    /// Right ascension (degrees).
    pub ra: f64,
    /// Declination (degrees).
    pub dec: f64,
    /// Galactic longitude (degrees).
    pub gl: f64,
    /// Galactic latitude (degrees).
    pub gb: f64,
    /// Bonsai DM (pc/cm^3).
    pub bonsai_dm: f64,
    /// DM from fitting (pc/cm^3).
    pub dm_fitb: f64,
    /// DM excess over NE2001 MW model (pc/cm^3).
    pub dm_exc_ne2001: f64,
    /// DM excess over YMW16 MW model (pc/cm^3).
    pub dm_exc_ymw16: f64,
    /// Bonsai S/N.
    pub bonsai_snr: f64,
    /// Flux density (Jy).
    pub flux: f64,
    /// Fluence (Jy ms).
    pub fluence: f64,
    /// Burst width from fitting (seconds).
    pub width_fitb: f64,
    /// Scattering time (seconds).
    pub scat_time: f64,
    /// MJD at 400 MHz (arrival time).
    pub mjd_400: f64,
    /// Spectral index.
    pub sp_idx: f64,
    /// High frequency bound (MHz).
    pub high_freq: f64,
    /// Low frequency bound (MHz).
    pub low_freq: f64,
    /// Peak frequency (MHz).
    pub peak_freq: f64,
    /// Whether this event is from Catalog 1.
    pub catalog1_flag: bool,
}

fn parse_f64(s: &str) -> f64 {
    let s = s.trim();
    if s.is_empty() || s == "nan" || s == "NaN" || s == "inf" || s == "-inf" {
        return f64::NAN;
    }
    s.parse::<f64>().unwrap_or(f64::NAN)
}

fn parse_bool(s: &str) -> bool {
    matches!(s.trim(), "1" | "true" | "True" | "TRUE")
}

/// Parse a CHIME/FRB catalog CSV file.
pub fn parse_chime_csv(path: &Path) -> Result<Vec<FrbEvent>, FetchError> {
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .has_headers(true)
        .from_path(path)
        .map_err(|e| FetchError::Validation(format!("CSV read error: {}", e)))?;

    let headers = reader
        .headers()
        .map_err(|e| FetchError::Validation(format!("Header read error: {}", e)))?
        .clone();

    let col = |name: &str| -> Option<usize> { headers.iter().position(|h| h == name) };

    let idx_tns = col("tns_name");
    let idx_rep = col("repeater_name");
    let idx_ra = col("ra");
    let idx_dec = col("dec");
    let idx_gl = col("gl");
    let idx_gb = col("gb");
    let idx_bonsai_dm = col("bonsai_dm");
    let idx_dm_fitb = col("dm_fitb");
    let idx_dm_ne = col("dm_exc_ne2001");
    let idx_dm_ymw = col("dm_exc_ymw16");
    let idx_snr = col("bonsai_snr");
    let idx_flux = col("flux");
    let idx_fluence = col("fluence");
    let idx_width = col("width_fitb");
    let idx_scat = col("scat_time");
    let idx_mjd = col("mjd_400");
    let idx_sp = col("sp_idx");
    let idx_hf = col("high_freq");
    let idx_lf = col("low_freq");
    let idx_pf = col("peak_freq");
    let idx_c1 = col("catalog1_flag");

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
        let record =
            result.map_err(|e| FetchError::Validation(format!("Record parse error: {}", e)))?;

        let bonsai_dm = get_f64(&record, idx_bonsai_dm);
        // Skip events with no DM
        if !bonsai_dm.is_finite() || bonsai_dm <= 0.0 {
            continue;
        }

        events.push(FrbEvent {
            tns_name: get_str(&record, idx_tns),
            repeater_name: get_str(&record, idx_rep),
            ra: get_f64(&record, idx_ra),
            dec: get_f64(&record, idx_dec),
            gl: get_f64(&record, idx_gl),
            gb: get_f64(&record, idx_gb),
            bonsai_dm,
            dm_fitb: get_f64(&record, idx_dm_fitb),
            dm_exc_ne2001: get_f64(&record, idx_dm_ne),
            dm_exc_ymw16: get_f64(&record, idx_dm_ymw),
            bonsai_snr: get_f64(&record, idx_snr),
            flux: get_f64(&record, idx_flux),
            fluence: get_f64(&record, idx_fluence),
            width_fitb: get_f64(&record, idx_width),
            scat_time: get_f64(&record, idx_scat),
            mjd_400: get_f64(&record, idx_mjd),
            sp_idx: get_f64(&record, idx_sp),
            high_freq: get_f64(&record, idx_hf),
            low_freq: get_f64(&record, idx_lf),
            peak_freq: get_f64(&record, idx_pf),
            catalog1_flag: idx_c1
                .and_then(|i| record.get(i))
                .map(parse_bool)
                .unwrap_or(false),
        });
    }

    Ok(events)
}

/// Extract repeater groups: repeater_name -> Vec<FrbEvent>, sorted by MJD.
pub fn extract_repeaters(events: &[FrbEvent]) -> Vec<(String, Vec<&FrbEvent>)> {
    use std::collections::BTreeMap;
    let mut groups: BTreeMap<String, Vec<&FrbEvent>> = BTreeMap::new();

    for ev in events {
        if !ev.repeater_name.is_empty() {
            groups.entry(ev.repeater_name.clone()).or_default().push(ev);
        }
    }

    // Sort each group by MJD
    let mut result: Vec<(String, Vec<&FrbEvent>)> = groups.into_iter().collect();
    for (_, group) in &mut result {
        group.sort_by(|a, b| a.mjd_400.partial_cmp(&b.mjd_400).unwrap());
    }

    result
}

const CATALOG1_URLS: &[&str] =
    &["https://storage.googleapis.com/chimefrb-dev.appspot.com/catalog1/chimefrbcat1.csv"];

const CATALOG2_URLS: &[&str] =
    &["https://storage.googleapis.com/chimefrb-dev.appspot.com/catalog2/chimefrbcat2.csv"];

/// CHIME Catalog 1 dataset provider.
pub struct ChimeCat1Provider;

impl DatasetProvider for ChimeCat1Provider {
    fn name(&self) -> &str {
        "CHIME/FRB Catalog 1"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("chime_frb_cat1.csv");
        download_with_fallbacks(self.name(), CATALOG1_URLS, &output, config.skip_existing)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("chime_frb_cat1.csv").exists()
    }
}

/// CHIME Catalog 2 dataset provider.
pub struct ChimeCat2Provider;

impl DatasetProvider for ChimeCat2Provider {
    fn name(&self) -> &str {
        "CHIME/FRB Catalog 2"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("chime_frb_cat2.csv");
        download_with_fallbacks(self.name(), CATALOG2_URLS, &output, config.skip_existing)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("chime_frb_cat2.csv").exists()
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
    fn test_parse_chime_synthetic_basic() {
        let csv = "\
tns_name,repeater_name,ra,dec,gl,gb,bonsai_dm,dm_fitb,dm_exc_ne2001,dm_exc_ymw16,bonsai_snr,flux,fluence,width_fitb,scat_time,mjd_400,sp_idx,high_freq,low_freq,peak_freq,catalog1_flag
FRB20180725A,,123.4,56.7,210.1,-30.2,300.5,298.1,250.0,240.0,12.5,0.5,1.2,0.003,0.001,58324.5,-2.0,800.0,400.0,600.0,1
FRB20190101B,R1,98.1,-12.3,180.0,20.0,500.0,498.0,450.0,440.0,20.0,1.0,3.5,0.001,0.0005,58484.2,-1.5,800.0,400.0,650.0,0
FRB20190201C,R1,98.2,-12.4,180.1,20.1,505.0,503.0,455.0,445.0,15.0,0.8,2.1,0.002,0.0008,58515.1,-1.8,800.0,400.0,620.0,0
";
        let f = write_temp_csv(csv);
        let events = parse_chime_csv(f.path()).unwrap();
        assert_eq!(events.len(), 3, "Should parse 3 events");
        assert_eq!(events[0].tns_name, "FRB20180725A");
        assert!((events[0].bonsai_dm - 300.5).abs() < 0.01);
        assert!(events[0].catalog1_flag);
        assert!(!events[1].catalog1_flag);
    }

    #[test]
    fn test_parse_chime_skips_zero_dm() {
        let csv = "\
tns_name,repeater_name,ra,dec,gl,gb,bonsai_dm,dm_fitb,dm_exc_ne2001,dm_exc_ymw16,bonsai_snr,flux,fluence,width_fitb,scat_time,mjd_400,sp_idx,high_freq,low_freq,peak_freq,catalog1_flag
FRB_GOOD,,10.0,20.0,30.0,40.0,100.0,99.0,80.0,70.0,10.0,0.5,1.0,0.001,0.0,58000.0,0.0,800.0,400.0,600.0,0
FRB_BAD_ZERO,,10.0,20.0,30.0,40.0,0.0,0.0,0.0,0.0,10.0,0.5,1.0,0.001,0.0,58001.0,0.0,800.0,400.0,600.0,0
FRB_BAD_NAN,,10.0,20.0,30.0,40.0,nan,nan,nan,nan,10.0,0.5,1.0,0.001,0.0,58002.0,0.0,800.0,400.0,600.0,0
";
        let f = write_temp_csv(csv);
        let events = parse_chime_csv(f.path()).unwrap();
        assert_eq!(events.len(), 1, "Should skip zero/NaN DM events");
        assert_eq!(events[0].tns_name, "FRB_GOOD");
    }

    #[test]
    fn test_parse_chime_handles_nan_fields() {
        let csv = "\
tns_name,repeater_name,ra,dec,gl,gb,bonsai_dm,dm_fitb,dm_exc_ne2001,dm_exc_ymw16,bonsai_snr,flux,fluence,width_fitb,scat_time,mjd_400,sp_idx,high_freq,low_freq,peak_freq,catalog1_flag
FRB_PARTIAL,,nan,nan,nan,nan,200.0,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,0
";
        let f = write_temp_csv(csv);
        let events = parse_chime_csv(f.path()).unwrap();
        assert_eq!(
            events.len(),
            1,
            "Event with valid DM but NaN coords should parse"
        );
        assert!(events[0].ra.is_nan());
        assert!((events[0].bonsai_dm - 200.0).abs() < 0.01);
    }

    #[test]
    fn test_extract_repeaters_groups_by_name() {
        let csv = "\
tns_name,repeater_name,ra,dec,gl,gb,bonsai_dm,dm_fitb,dm_exc_ne2001,dm_exc_ymw16,bonsai_snr,flux,fluence,width_fitb,scat_time,mjd_400,sp_idx,high_freq,low_freq,peak_freq,catalog1_flag
E1,R_alpha,10.0,20.0,30.0,40.0,100.0,99.0,80.0,70.0,10.0,0.5,1.0,0.001,0.0,58100.0,0.0,800.0,400.0,600.0,0
E2,R_alpha,10.0,20.0,30.0,40.0,100.0,99.0,80.0,70.0,10.0,0.5,1.0,0.001,0.0,58050.0,0.0,800.0,400.0,600.0,0
E3,R_beta,50.0,60.0,70.0,80.0,200.0,199.0,180.0,170.0,15.0,1.0,2.0,0.002,0.0,58200.0,0.0,800.0,400.0,600.0,0
E4,,90.0,10.0,20.0,30.0,300.0,299.0,280.0,270.0,20.0,1.5,3.0,0.003,0.0,58300.0,0.0,800.0,400.0,600.0,0
";
        let f = write_temp_csv(csv);
        let events = parse_chime_csv(f.path()).unwrap();
        assert_eq!(events.len(), 4);

        let repeaters = extract_repeaters(&events);
        assert_eq!(repeaters.len(), 2, "Should have 2 repeater sources");

        // Find R_alpha group
        let alpha = repeaters.iter().find(|(n, _)| n == "R_alpha").unwrap();
        assert_eq!(alpha.1.len(), 2, "R_alpha should have 2 bursts");
        // Verify sorted by MJD (E2 at 58050 before E1 at 58100)
        assert!(
            alpha.1[0].mjd_400 < alpha.1[1].mjd_400,
            "Repeater bursts should be sorted by MJD"
        );
    }

    #[test]
    fn test_parse_chime_empty_csv() {
        let csv = "tns_name,repeater_name,ra,dec,gl,gb,bonsai_dm\n";
        let f = write_temp_csv(csv);
        let events = parse_chime_csv(f.path()).unwrap();
        assert!(events.is_empty(), "Empty CSV should produce no events");
    }

    #[test]
    fn test_parse_chime_cat2_if_available() {
        let path = Path::new("data/external/chime_frb_cat2.csv");
        if !path.exists() {
            eprintln!("Skipping: CHIME Cat 2 not available");
            return;
        }

        let events = parse_chime_csv(path).expect("Failed to parse CHIME Cat 2");
        assert!(events.len() > 100, "Should have many FRB events");

        let repeaters = extract_repeaters(&events);
        eprintln!(
            "Parsed {} events, {} repeater sources",
            events.len(),
            repeaters.len()
        );

        let large_repeaters: Vec<_> = repeaters
            .iter()
            .filter(|(_, evts)| evts.len() >= 10)
            .collect();
        eprintln!("Repeaters with >= 10 bursts: {}", large_repeaters.len());
    }
}

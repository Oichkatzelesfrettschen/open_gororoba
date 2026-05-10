//! Wow! signal (1977) archival printout and Breakthrough Listen 6EQUJ5 follow-up.
//!
//! The Wow! signal was a strong narrowband radio signal detected on 1977-08-15
//! at the Big Ear radio telescope (Ohio State University) by Jerry Ehman during
//! a SETI survey at 1420.405 MHz (hydrogen line). The printout character sequence
//! 6EQUJ5 represents base-36 intensity values [6, 14, 26, 30, 19, 5] on a
//! 50-channel receiver.
//!
//! In 2023, the Breakthrough Listen (BL) project observed the Wow! signal
//! coordinates with the Green Bank Telescope (GBT) using the ABACAD cadence
//! strategy (ON-OFF-ON-OFF-ON-OFF) to discriminate terrestrial RFI.
//!
//! Sources:
//! - Ohio History Connection IIIF image (archival printout scan)
//! - Ehman (1997) 20th anniversary technical report
//! - Gray & Ellingsen (2002) ApJ 578 967
//! - Perez et al. (2022) RNAAS 6 197 (BL 6EQUJ5 follow-up)

use crate::fetcher::FetchError;
use std::path::Path;

/// Convert a single printout character to its base-36 intensity value.
///
/// The Big Ear used base-36 encoding: 0-9 map to 0-9, A-Z map to 10-35.
/// A space (blank) represents zero signal. Non-matching characters return None.
pub fn wow_char_to_intensity(c: char) -> Option<u32> {
    match c {
        ' ' => Some(0),
        '0'..='9' => Some(c as u32 - '0' as u32),
        'A'..='Z' => Some(c as u32 - 'A' as u32 + 10),
        _ => None,
    }
}

/// A single row from the Wow! signal printout transcription.
#[derive(Debug, Clone)]
pub struct WowPrintoutRow {
    /// Row index in the printout (0-based).
    pub row_index: usize,
    /// Channel number (1-50 on the Big Ear receiver).
    pub channel: u32,
    /// The printout character (base-36).
    pub printout_char: char,
    /// Decoded intensity value (0-35).
    pub intensity: u32,
    /// Whether this row is part of the 6EQUJ5 Wow! sequence.
    pub in_wow_sequence: bool,
}

/// Parse the Wow! signal transcription CSV.
///
/// Expected columns: `row_index,channel,printout_char,intensity,in_wow_sequence`
pub fn parse_wow_printout_csv(path: &Path) -> Result<Vec<WowPrintoutRow>, FetchError> {
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

    let col_exact = |name: &str| -> Option<usize> {
        headers
            .iter()
            .position(|h| h.trim().eq_ignore_ascii_case(name))
    };

    let idx_row = col_exact("row_index");
    let idx_ch = col_exact("channel");
    let idx_char = col_exact("printout_char");
    let idx_int = col_exact("intensity");
    let idx_wow = col_exact("in_wow_sequence");

    let mut rows = Vec::new();
    for result in reader.records() {
        let record = match result {
            Ok(r) => r,
            Err(_) => continue,
        };

        let row_index: usize = idx_row
            .and_then(|i| record.get(i))
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);

        let channel: u32 = idx_ch
            .and_then(|i| record.get(i))
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);

        let printout_char: char = idx_char
            .and_then(|i| record.get(i))
            .and_then(|s| s.trim().chars().next())
            .unwrap_or(' ');

        let intensity: u32 = idx_int
            .and_then(|i| record.get(i))
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);

        let in_wow_sequence: bool = idx_wow
            .and_then(|i| record.get(i))
            .map(|s| s.trim().eq_ignore_ascii_case("true") || s.trim() == "1")
            .unwrap_or(false);

        rows.push(WowPrintoutRow {
            row_index,
            channel,
            printout_char,
            intensity,
            in_wow_sequence,
        });
    }

    Ok(rows)
}

/// A bundle entry from the Breakthrough Listen 6EQUJ5 GBT manifest.
#[derive(Debug, Clone)]
pub struct Bl6equj5Bundle {
    /// Bundle identifier (e.g., "spliced_blc00..07_guppi_59538_39145_DIAG_6EQUJ5_1_0016").
    pub bundle_id: String,
    /// Target name (e.g., "DIAG_6EQUJ5_1").
    pub target: String,
    /// Observation sequence number.
    pub obs_num: u32,
    /// Pointing type: ON (target) or OFF (calibrator).
    pub pointing_type: String,
    /// ABACAD cadence position (1 or 2).
    pub cadence: u32,
    /// Data product tier ("filterbank" or "hires").
    pub product_tier: String,
}

/// Parse the BL 6EQUJ5 GBT manifest CSV.
///
/// Expected columns: `bundle_id,target,obs_num,pointing_type,cadence,product_tier,...`
pub fn parse_bl_manifest_csv(path: &Path) -> Result<Vec<Bl6equj5Bundle>, FetchError> {
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

    let col_exact = |name: &str| -> Option<usize> {
        headers
            .iter()
            .position(|h| h.trim().eq_ignore_ascii_case(name))
    };

    let idx_bundle = col_exact("bundle_id");
    let idx_target = col_exact("target");
    let idx_obs = col_exact("obs_num");
    let idx_pt = col_exact("pointing_type");
    let idx_cad = col_exact("cadence");
    let idx_tier = col_exact("product_tier");

    let get_str = |record: &csv::StringRecord, idx: Option<usize>| -> String {
        idx.and_then(|i| record.get(i))
            .unwrap_or("")
            .trim()
            .to_string()
    };

    let mut bundles = Vec::new();
    for result in reader.records() {
        let record = match result {
            Ok(r) => r,
            Err(_) => continue,
        };

        let bundle_id = get_str(&record, idx_bundle);
        if bundle_id.is_empty() {
            continue;
        }

        let obs_num: u32 = idx_obs
            .and_then(|i| record.get(i))
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);

        let cadence: u32 = idx_cad
            .and_then(|i| record.get(i))
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);

        bundles.push(Bl6equj5Bundle {
            bundle_id,
            target: get_str(&record, idx_target),
            obs_num,
            pointing_type: get_str(&record, idx_pt),
            cadence,
            product_tier: get_str(&record, idx_tier),
        });
    }

    Ok(bundles)
}

/// Filter BL 6EQUJ5 bundles for ABACAD cadence pairs.
///
/// ABACAD = ON-OFF-ON-OFF-ON-OFF sequence. Returns only ON pointings
/// that have a matching OFF calibrator in the same cadence group.
pub fn abacad_filter(bundles: &[Bl6equj5Bundle]) -> Vec<&Bl6equj5Bundle> {
    let mut result = Vec::new();
    for b in bundles {
        if b.pointing_type == "ON" && b.cadence > 0 {
            // Check that a corresponding OFF exists in the same cadence
            let has_off = bundles.iter().any(|other| {
                other.cadence == b.cadence
                    && other.pointing_type == "OFF"
                    && other.obs_num != b.obs_num
            });
            if has_off {
                result.push(b);
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_wow_char_to_intensity_digits() {
        for d in 0..=9u32 {
            let c = char::from_digit(d, 10).unwrap();
            assert_eq!(wow_char_to_intensity(c), Some(d));
        }
    }

    #[test]
    fn test_wow_char_to_intensity_letters() {
        assert_eq!(wow_char_to_intensity('A'), Some(10));
        assert_eq!(wow_char_to_intensity('Z'), Some(35));
        assert_eq!(wow_char_to_intensity('E'), Some(14));
        assert_eq!(wow_char_to_intensity('Q'), Some(26));
        assert_eq!(wow_char_to_intensity('U'), Some(30));
        assert_eq!(wow_char_to_intensity('J'), Some(19));
    }

    #[test]
    fn test_wow_char_6equj5_sequence() {
        let wow = "6EQUJ5";
        let expected = [6, 14, 26, 30, 19, 5];
        let decoded: Vec<u32> = wow
            .chars()
            .map(|c| wow_char_to_intensity(c).unwrap())
            .collect();
        assert_eq!(decoded, expected);
    }

    #[test]
    fn test_wow_char_space_is_zero() {
        assert_eq!(wow_char_to_intensity(' '), Some(0));
    }

    #[test]
    fn test_wow_char_invalid() {
        assert_eq!(wow_char_to_intensity('!'), None);
        assert_eq!(wow_char_to_intensity('a'), None); // lowercase not used
        assert_eq!(wow_char_to_intensity('\n'), None);
    }

    fn write_temp_csv(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn test_parse_wow_printout_csv() {
        let csv = "\
row_index,channel,printout_char,intensity,in_wow_sequence
0,1,1,1,false
1,2,6,6,true
2,2,E,14,true
3,2,Q,26,true
4,2,U,30,true
5,2,J,19,true
6,2,5,5,true
";
        let f = write_temp_csv(csv);
        let rows = parse_wow_printout_csv(f.path()).unwrap();
        assert_eq!(rows.len(), 7);
        assert!(!rows[0].in_wow_sequence);
        assert!(rows[1].in_wow_sequence);
        assert_eq!(rows[3].printout_char, 'Q');
        assert_eq!(rows[3].intensity, 26);
        assert_eq!(rows[4].intensity, 30);
    }

    #[test]
    fn test_parse_wow_printout_empty() {
        let csv = "row_index,channel,printout_char,intensity,in_wow_sequence\n";
        let f = write_temp_csv(csv);
        let rows = parse_wow_printout_csv(f.path()).unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn test_parse_wow_printout_csv_if_available() {
        let path = std::path::Path::new("data/csv/wow1977_transcription.csv");
        if !path.exists() {
            eprintln!("Skipping: wow1977_transcription.csv not available");
            return;
        }
        let rows = parse_wow_printout_csv(path).expect("Failed to parse Wow! CSV");
        assert!(!rows.is_empty(), "Should have at least one row");

        let wow_rows: Vec<_> = rows.iter().filter(|r| r.in_wow_sequence).collect();
        assert_eq!(wow_rows.len(), 6, "6EQUJ5 = 6 characters");

        let wow_intensities: Vec<u32> = wow_rows.iter().map(|r| r.intensity).collect();
        assert_eq!(wow_intensities, vec![6, 14, 26, 30, 19, 5]);
    }

    #[test]
    fn test_parse_bl_manifest_csv() {
        let csv = "\
bundle_id,target,obs_num,pointing_type,cadence,product_tier
spliced_blc00_59538_39145_DIAG_6EQUJ5_1_0016,DIAG_6EQUJ5_1,16,ON,1,filterbank
spliced_blc00_59538_39500_HIP94637_0017,HIP94637,17,OFF,1,filterbank
spliced_blc00_59538_39855_DIAG_6EQUJ5_1_0018,DIAG_6EQUJ5_1,18,ON,1,filterbank
";
        let f = write_temp_csv(csv);
        let bundles = parse_bl_manifest_csv(f.path()).unwrap();
        assert_eq!(bundles.len(), 3);
        assert_eq!(bundles[0].pointing_type, "ON");
        assert_eq!(bundles[1].pointing_type, "OFF");
        assert_eq!(bundles[0].cadence, 1);
    }

    #[test]
    fn test_abacad_filter() {
        let bundles = vec![
            Bl6equj5Bundle {
                bundle_id: "on1".to_string(),
                target: "6EQUJ5".to_string(),
                obs_num: 16,
                pointing_type: "ON".to_string(),
                cadence: 1,
                product_tier: "filterbank".to_string(),
            },
            Bl6equj5Bundle {
                bundle_id: "off1".to_string(),
                target: "HIP94637".to_string(),
                obs_num: 17,
                pointing_type: "OFF".to_string(),
                cadence: 1,
                product_tier: "filterbank".to_string(),
            },
            Bl6equj5Bundle {
                bundle_id: "on2".to_string(),
                target: "6EQUJ5".to_string(),
                obs_num: 18,
                pointing_type: "ON".to_string(),
                cadence: 1,
                product_tier: "filterbank".to_string(),
            },
            Bl6equj5Bundle {
                bundle_id: "cal".to_string(),
                target: "3C295".to_string(),
                obs_num: 11,
                pointing_type: "ON".to_string(),
                cadence: 0, // calibrator, no cadence
                product_tier: "filterbank".to_string(),
            },
        ];
        let filtered = abacad_filter(&bundles);
        assert_eq!(
            filtered.len(),
            2,
            "Should return 2 ON pointings with OFF pairs"
        );
        assert_eq!(filtered[0].bundle_id, "on1");
        assert_eq!(filtered[1].bundle_id, "on2");
    }

    #[test]
    fn test_abacad_filter_no_off() {
        let bundles = vec![Bl6equj5Bundle {
            bundle_id: "on_only".to_string(),
            target: "6EQUJ5".to_string(),
            obs_num: 16,
            pointing_type: "ON".to_string(),
            cadence: 1,
            product_tier: "filterbank".to_string(),
        }];
        let filtered = abacad_filter(&bundles);
        assert!(filtered.is_empty(), "No OFF pair means no valid cadence");
    }
}

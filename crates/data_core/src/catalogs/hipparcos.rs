//! Hipparcos legacy stellar astrometry catalog.
//!
//! Hipparcos is the predecessor to Gaia and remains important for legacy
//! cross-calibration and long-baseline astrometry comparisons.
//!
//! Source: CDS catalog I/239
//! https://cdsarc.cds.unistra.fr/ftp/I/239/
//!
//! The CDS FTP layout changed (2025): `/ftp/cats/I/239/` -> `/ftp/I/239/`.
//! The `.gz` variant is no longer served; we download the uncompressed `.dat`.

use crate::fetcher::{download_with_fallbacks, DatasetProvider, FetchConfig, FetchError};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// Expected line width (characters, excluding newline) in hip_main.dat.
pub const HIPPARCOS_LINE_WIDTH: usize = 450;

/// Number of pipe delimiters per line (78 fields).
pub const HIPPARCOS_PIPE_COUNT: usize = 77;

/// Expected total rows in the full Hipparcos main catalog.
pub const HIPPARCOS_EXPECTED_ROWS: usize = 118_218;

/// Corrected CDS URLs (no `/cats/` prefix, uncompressed `.dat`).
const HIPPARCOS_URLS: &[&str] = &[
    "https://cdsarc.cds.unistra.fr/ftp/I/239/hip_main.dat",
    "https://cdsarc.u-strasbg.fr/ftp/I/239/hip_main.dat",
];

/// Count non-empty rows in a plain-text Hipparcos main catalog file.
pub fn hipparcos_row_count(path: &Path) -> Result<usize, FetchError> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut count = 0usize;
    for line in reader.lines() {
        let line = line?;
        if !line.trim().is_empty() {
            count += 1;
        }
    }
    Ok(count)
}

/// Validate the fixed-width pipe-delimited format of a Hipparcos .dat file.
///
/// Checks up to `max_lines` lines (0 = all) for:
/// - Every line starts with `H|`
/// - Consistent line width (450 chars)
/// - Consistent pipe count (77 per line)
/// - HIP number (field 2) is a valid positive integer
pub fn validate_hipparcos_format(path: &Path, max_lines: usize) -> Result<(), FetchError> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    for (i, line_result) in reader.lines().enumerate() {
        if max_lines > 0 && i >= max_lines {
            break;
        }
        let line = line_result?;
        if line.trim().is_empty() {
            continue;
        }
        let line_num = i + 1;

        // Check line starts with catalog marker
        if !line.starts_with("H|") {
            return Err(FetchError::Validation(format!(
                "Line {} does not start with 'H|': {:?}",
                line_num,
                &line[..line.len().min(20)]
            )));
        }

        // Check line width
        let width = line.len();
        if width != HIPPARCOS_LINE_WIDTH {
            return Err(FetchError::Validation(format!(
                "Line {} has width {} (expected {})",
                line_num, width, HIPPARCOS_LINE_WIDTH
            )));
        }

        // Check pipe count
        let pipes = line.chars().filter(|&c| c == '|').count();
        if pipes != HIPPARCOS_PIPE_COUNT {
            return Err(FetchError::Validation(format!(
                "Line {} has {} pipes (expected {})",
                line_num, pipes, HIPPARCOS_PIPE_COUNT
            )));
        }

        // Extract HIP number from field 2 (between first and second pipe)
        if let Some(hip_str) = line.split('|').nth(1) {
            let trimmed = hip_str.trim();
            if trimmed.parse::<u32>().is_err() {
                return Err(FetchError::Validation(format!(
                    "Line {} HIP field is not a valid integer: {:?}",
                    line_num, trimmed
                )));
            }
        }
    }
    Ok(())
}

/// Extract the HIP identifier number from a single catalog line.
pub fn parse_hip_number(line: &str) -> Option<u32> {
    line.split('|').nth(1)?.trim().parse::<u32>().ok()
}

/// Hipparcos catalog provider.
pub struct HipparcosProvider;

impl DatasetProvider for HipparcosProvider {
    fn name(&self) -> &str {
        "Hipparcos Legacy Catalog"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("hip_main.dat");
        download_with_fallbacks(self.name(), HIPPARCOS_URLS, &output, config.skip_existing)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("hip_main.dat").exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::Path;

    /// Build a synthetic Hipparcos line with the correct 450-char pipe-delimited format.
    /// Uses 78 fields separated by 77 pipes, starting with `H|`.
    fn synthetic_hip_line(hip_id: u32) -> String {
        // Field 1: catalog marker (1 char)
        // Field 2: HIP number (11 chars, right-justified)
        // Fields 3-78: filler fields to reach 450 total chars
        let mut fields: Vec<String> = Vec::with_capacity(78);
        fields.push("H".to_string());
        fields.push(format!("{:>11}", hip_id));
        // We need the remaining 76 fields. The total line width must be 450.
        // Used width so far: "H" (1) + "|" (1) + 11 + "|" (1) = 14
        // Remaining width: 450 - 14 = 436 chars for 76 fields + 75 pipes
        // 436 - 75 = 361 chars across 76 fields -> average ~4.75 chars per field
        // Use " " as a placeholder for most fields, with a few realistic ones
        fields.push(" ".to_string()); // proximity flag
        fields.push("00 00 00.22".to_string()); // RA
        fields.push("+01 05 20.4".to_string()); // Dec
        fields.push(" 9.10".to_string()); // Vmag
        fields.push(" ".to_string()); // var flag
                                      // Fill remaining 71 fields: need to reach exact width
        let partial = fields.join("|");
        let partial_len = partial.len();
        // We need 71 more fields with 71 pipes.
        // Target: 450 - partial_len - 71 (pipes) = chars across 71 fields
        let remaining_chars = 450 - partial_len - 71;
        // Distribute: 70 single-space fields + 1 padded field
        let pad_field_len = remaining_chars - 70; // 70 fields of " " = 70 chars
        let mut pad_field = " ".repeat(pad_field_len);
        // Last char can be a letter for realism
        if pad_field_len > 1 {
            pad_field.replace_range(pad_field_len - 1..pad_field_len, "S");
        }
        fields.push(pad_field);
        for _ in 0..70 {
            fields.push(" ".to_string());
        }
        let line = fields.join("|");
        assert_eq!(
            line.len(),
            HIPPARCOS_LINE_WIDTH,
            "Synthetic line width mismatch: got {}, fields={}",
            line.len(),
            fields.len()
        );
        line
    }

    fn write_synthetic_hipparcos(lines: &[String]) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        for line in lines {
            writeln!(f, "{}", line).unwrap();
        }
        f.flush().unwrap();
        f
    }

    #[test]
    fn test_validate_hipparcos_format_synthetic() {
        let lines: Vec<String> = (1..=5).map(synthetic_hip_line).collect();
        let f = write_synthetic_hipparcos(&lines);
        validate_hipparcos_format(f.path(), 0).expect("valid synthetic data should pass");
    }

    #[test]
    fn test_validate_hipparcos_rejects_wrong_marker() {
        let mut line = synthetic_hip_line(1);
        line.replace_range(0..1, "X");
        let f = write_synthetic_hipparcos(&[line]);
        let err = validate_hipparcos_format(f.path(), 0);
        assert!(err.is_err(), "Should reject line not starting with H|");
    }

    #[test]
    fn test_validate_hipparcos_rejects_wrong_width() {
        let line = synthetic_hip_line(1);
        let short = line[..400].to_string();
        let f = write_synthetic_hipparcos(&[short]);
        let err = validate_hipparcos_format(f.path(), 0);
        assert!(err.is_err(), "Should reject line with wrong width");
    }

    #[test]
    fn test_validate_hipparcos_rejects_bad_hip_number() {
        let mut line = synthetic_hip_line(1);
        // Replace HIP number field (chars 2..13) with non-numeric
        line.replace_range(2..13, "   BADVALUE");
        let f = write_synthetic_hipparcos(&[line]);
        let err = validate_hipparcos_format(f.path(), 0);
        assert!(err.is_err(), "Should reject non-numeric HIP field");
    }

    #[test]
    fn test_parse_hip_number() {
        let line = synthetic_hip_line(42);
        assert_eq!(parse_hip_number(&line), Some(42));
    }

    #[test]
    fn test_parse_hip_number_from_real_format() {
        // Realistic first line from the actual catalog (truncated to show structure)
        let line = "H|          1| |00 00 00.22|+01 05 20.4| 9.10";
        assert_eq!(parse_hip_number(line), Some(1));
    }

    #[test]
    fn test_hipparcos_row_count_synthetic() {
        let lines: Vec<String> = (1..=10).map(synthetic_hip_line).collect();
        let f = write_synthetic_hipparcos(&lines);
        let count = hipparcos_row_count(f.path()).unwrap();
        assert_eq!(count, 10);
    }

    #[test]
    fn test_hipparcos_if_available() {
        let path = Path::new("data/external/hip_main.dat");
        if !path.exists() {
            eprintln!("Skipping: Hipparcos data not available");
            return;
        }
        let rows = hipparcos_row_count(path).expect("failed to count Hipparcos rows");
        assert_eq!(
            rows, HIPPARCOS_EXPECTED_ROWS,
            "Hipparcos should have exactly {} rows",
            HIPPARCOS_EXPECTED_ROWS
        );
        // Validate format of first 100 lines
        validate_hipparcos_format(path, 100).expect("Hipparcos format validation should pass");
    }
}

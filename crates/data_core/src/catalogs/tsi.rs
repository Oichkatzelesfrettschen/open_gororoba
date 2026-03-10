//! TSIS-1 Total Solar Irradiance (TSI) measurements.
//!
//! TSIS-1 (Total and Spectral Solar Irradiance Sensor) aboard the ISS provides
//! the most accurate TSI measurements to date, with ~0.01% absolute accuracy.
//!
//! Source: LASP Interactive Solar Irradiance Datacenter (LISIRD)
//! Reference: Kopp (2021), https://doi.org/10.1007/s11207-021-01853-x

use crate::{
    fetcher::{DatasetProvider, FetchConfig, FetchError, download_with_fallbacks},
    parse::parse_f64_or_nan,
};
use hifitime::Epoch;
use std::path::{Path, PathBuf};

/// A single TSI measurement from TSIS-1.
#[derive(Debug, Clone)]
pub struct TsiMeasurement {
    /// Date as Julian Day Number.
    pub jd: f64,
    /// ISO date string (YYYY-MM-DD).
    pub date: String,
    /// Total Solar Irradiance (W/m^2).
    pub tsi: f64,
    /// TSI measurement uncertainty (W/m^2).
    pub tsi_uncertainty: f64,
}

/// Parse TSIS-1 TSI CSV data.
pub fn parse_tsi_csv(path: &Path) -> Result<Vec<TsiMeasurement>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("Read error: {}", e)))?;
    let filtered: String = content
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            !(trimmed.is_empty() || trimmed.starts_with(';'))
        })
        .collect::<Vec<_>>()
        .join("\n");

    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .has_headers(true)
        .comment(Some(b'#'))
        .from_reader(filtered.as_bytes());

    let headers = reader
        .headers()
        .map_err(|e| FetchError::Validation(format!("Header read error: {}", e)))?
        .clone();

    let col_exact = |name: &str| -> Option<usize> {
        headers
            .iter()
            .position(|header| header.trim().eq_ignore_ascii_case(name))
    };

    let idx_jd = col_exact("jd").or_else(|| col_exact("time (Julian Date)"));
    let idx_date = col_exact("date");
    let idx_tsi = col_exact("TSI").or_else(|| col_exact("tsi_1au (W/m^2)"));
    let idx_unc = col_exact("uncertainty")
        .or_else(|| col_exact("measurement_uncertainty_1au (W/m^2)"))
        .or_else(|| col_exact("instrument_accuracy_1au (W/m^2)"));

    let mut measurements = Vec::new();

    for result in reader.records() {
        let record =
            result.map_err(|e| FetchError::Validation(format!("Record parse error: {}", e)))?;
        let jd = idx_jd
            .and_then(|idx| record.get(idx))
            .map(parse_f64_or_nan)
            .unwrap_or(f64::NAN);
        let tsi = idx_tsi
            .and_then(|idx| record.get(idx))
            .map(parse_f64_or_nan)
            .unwrap_or(f64::NAN);
        let unc = idx_unc
            .and_then(|idx| record.get(idx))
            .map(parse_f64_or_nan)
            .unwrap_or(f64::NAN);

        if tsi.is_nan() || tsi <= 0.0 || !jd.is_finite() {
            continue;
        }

        let date = idx_date
            .and_then(|idx| record.get(idx))
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| {
                Epoch::from_jde_utc(jd)
                    .to_rfc3339()
                    .split('T')
                    .next()
                    .unwrap_or("")
                    .to_string()
            });

        measurements.push(TsiMeasurement {
            jd,
            date,
            tsi,
            tsi_uncertainty: unc,
        });
    }

    Ok(measurements)
}

/// Number of fields in the TsiMeasurement struct.
pub const TSI_FIELD_COUNT: usize = 4;

/// Result of comparing TSIS and SORCE TSI measurements in an overlap period.
#[derive(Debug, Clone)]
pub struct TsiOverlapResult {
    /// Number of matched day-pairs.
    pub n_matched: usize,
    /// Mean TSIS - SORCE offset (W/m^2).
    pub mean_offset: f64,
    /// RMS residual (W/m^2).
    pub rms_residual: f64,
    /// Maximum absolute difference (W/m^2).
    pub max_abs_diff: f64,
}

/// Compare TSIS and SORCE measurements by matching Julian Day (within tolerance).
///
/// The TSIS-1 instrument and SORCE TIM overlap from ~2018 to ~2020.
/// Both provide daily-averaged TSI values. Matching by JD within `jd_tol`
/// (default 0.5 days) and computing statistics validates cross-calibration.
pub fn compare_tsis_sorce(
    tsis: &[TsiMeasurement],
    sorce: &[super::sorce::SorceMeasurement],
    jd_tol: f64,
) -> TsiOverlapResult {
    let mut diffs = Vec::new();

    for t in tsis {
        if t.tsi.is_nan() {
            continue;
        }
        // Find closest SORCE measurement by JD
        let mut best_s: Option<&super::sorce::SorceMeasurement> = None;
        let mut best_dt = f64::MAX;
        for s in sorce {
            if s.tsi.is_nan() {
                continue;
            }
            let dt = (t.jd - s.jd).abs();
            if dt < best_dt {
                best_dt = dt;
                best_s = Some(s);
            }
        }
        if let Some(s) = best_s
            && best_dt <= jd_tol
        {
            diffs.push(t.tsi - s.tsi);
        }
    }

    if diffs.is_empty() {
        return TsiOverlapResult {
            n_matched: 0,
            mean_offset: f64::NAN,
            rms_residual: f64::NAN,
            max_abs_diff: f64::NAN,
        };
    }

    let n = diffs.len();
    let mean = diffs.iter().sum::<f64>() / n as f64;
    let rms = (diffs.iter().map(|d| d * d).sum::<f64>() / n as f64).sqrt();
    let max_abs = diffs.iter().map(|d| d.abs()).fold(0.0f64, f64::max);

    TsiOverlapResult {
        n_matched: n,
        mean_offset: mean,
        rms_residual: rms,
        max_abs_diff: max_abs,
    }
}

/// LASP LISIRD TSIS-1 daily TSI data URL.
const TSIS_URLS: &[&str] = &["https://lasp.colorado.edu/lisird/latis/dap/tsis_tsi_24hr.csv"];

/// TSIS-1 Total Solar Irradiance dataset provider.
pub struct TsisTsiProvider;

impl DatasetProvider for TsisTsiProvider {
    fn name(&self) -> &str {
        "TSIS-1 TSI Daily"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("tsis1_tsi_daily.csv");
        download_with_fallbacks(self.name(), TSIS_URLS, &output, config.skip_existing)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("tsis1_tsi_daily.csv").exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp_csv(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn test_parse_synthetic_tsi() {
        let csv = "\
jd,date,TSI,uncertainty
2459215.5,2021-01-01,1360.52,0.14
2459216.5,2021-01-02,1360.48,0.13
2459217.5,2021-01-03,1360.55,0.15
";
        let f = write_temp_csv(csv);
        let measurements = parse_tsi_csv(f.path()).unwrap();
        assert_eq!(measurements.len(), 3);

        let m1 = &measurements[0];
        assert!((m1.jd - 2459215.5).abs() < 0.01);
        assert_eq!(m1.date, "2021-01-01");
        assert!((m1.tsi - 1360.52).abs() < 0.01);
        assert!((m1.tsi_uncertainty - 0.14).abs() < 0.01);
    }

    #[test]
    fn test_parse_real_lisird_header_shape() {
        let csv = "\
time (Julian Date),tsi_1au (W/m^2),instrument_accuracy_1au (W/m^2),instrument_precision_1au (W/m^2),solar_standard_deviation_1au (W/m^2),measurement_uncertainty_1au (W/m^2),tsi_true_earth (W/m^2),instrument_accuracy_true_earth (W/m^2),instrument_precision_true_earth (W/m^2),solar_standard_deviation_true_earth (W/m^2),measurement_uncertainty_true_earth (W/m^2),avg_measurement_date (Julian Date),std_dev_measurement_date (days),provisional_flag
2458129.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2458129.0,0.0,0
2458130.0,1361.6251,0.1684,0.0068,0.04423,0.1741,1407.7673,0.1741,0.0068,0.04577,0.1801,2458129.716197,0.0064,0
";
        let f = write_temp_csv(csv);
        let measurements = parse_tsi_csv(f.path()).unwrap();
        assert_eq!(measurements.len(), 1, "zero-valued rows should be skipped");
        let m1 = &measurements[0];
        assert!((m1.jd - 2458130.0).abs() < 0.01);
        assert_eq!(m1.date, "2018-01-11");
        assert!((m1.tsi - 1361.6251).abs() < 0.001);
        assert!((m1.tsi_uncertainty - 0.1741).abs() < 0.001);
    }

    #[test]
    fn test_tsi_nan_rows_skipped() {
        let csv = "\
jd,date,TSI,uncertainty
2459215.5,2021-01-01,NaN,0
2459216.5,2021-01-02,1360.5,0.1
";
        let f = write_temp_csv(csv);
        let measurements = parse_tsi_csv(f.path()).unwrap();
        assert_eq!(measurements.len(), 1, "NaN TSI rows should be skipped");
        assert!((measurements[0].tsi - 1360.5).abs() < 0.01);
    }

    #[test]
    fn test_tsi_comment_lines_skipped() {
        let csv = "\
# TSIS-1 Total Solar Irradiance
; Version 03
jd,date,TSI,uncertainty
2459215.5,2021-01-01,1360.52,0.14
";
        let f = write_temp_csv(csv);
        let measurements = parse_tsi_csv(f.path()).unwrap();
        assert_eq!(measurements.len(), 1);
    }

    #[test]
    fn test_tsi_sentinel_values() {
        let csv = "\
jd,date,TSI,uncertainty
2459215.5,2021-01-01,-99,0
2459216.5,2021-01-02,-999,0
2459217.5,2021-01-03,1360.5,0.1
";
        let f = write_temp_csv(csv);
        let measurements = parse_tsi_csv(f.path()).unwrap();
        assert_eq!(
            measurements.len(),
            1,
            "sentinel values -99/-999 should be NaN -> skipped"
        );
    }

    #[test]
    fn test_tsi_field_count() {
        assert_eq!(TSI_FIELD_COUNT, 4);
    }

    #[test]
    fn test_compare_tsis_sorce_synthetic() {
        use super::super::sorce::SorceMeasurement;

        let tsis = vec![
            TsiMeasurement {
                jd: 2458300.5,
                date: "2018-07-01".into(),
                tsi: 1360.50,
                tsi_uncertainty: 0.1,
            },
            TsiMeasurement {
                jd: 2458301.5,
                date: "2018-07-02".into(),
                tsi: 1360.52,
                tsi_uncertainty: 0.1,
            },
            TsiMeasurement {
                jd: 2458302.5,
                date: "2018-07-03".into(),
                tsi: 1360.48,
                tsi_uncertainty: 0.1,
            },
        ];
        let sorce = vec![
            SorceMeasurement {
                jd: 2458300.5,
                date: "20180701".into(),
                tsi: 1360.40,
            },
            SorceMeasurement {
                jd: 2458301.5,
                date: "20180702".into(),
                tsi: 1360.45,
            },
            SorceMeasurement {
                jd: 2458302.5,
                date: "20180703".into(),
                tsi: 1360.42,
            },
        ];

        let result = compare_tsis_sorce(&tsis, &sorce, 0.5);
        assert_eq!(result.n_matched, 3);
        // Offsets: 0.10, 0.07, 0.06 -> mean = 0.0767, rms ~ 0.078
        assert!((result.mean_offset - 0.0767).abs() < 0.01);
        assert!(result.rms_residual > 0.05);
        assert!(result.rms_residual < 0.15);
        assert!((result.max_abs_diff - 0.10).abs() < 0.01);
    }

    #[test]
    fn test_compare_tsis_sorce_no_overlap() {
        use super::super::sorce::SorceMeasurement;

        let tsis = vec![TsiMeasurement {
            jd: 2460000.5,
            date: "2023-01-01".into(),
            tsi: 1360.5,
            tsi_uncertainty: 0.1,
        }];
        let sorce = vec![SorceMeasurement {
            jd: 2458000.5,
            date: "20170101".into(),
            tsi: 1360.4,
        }];

        let result = compare_tsis_sorce(&tsis, &sorce, 0.5);
        assert_eq!(result.n_matched, 0);
        assert!(result.mean_offset.is_nan());
    }
}

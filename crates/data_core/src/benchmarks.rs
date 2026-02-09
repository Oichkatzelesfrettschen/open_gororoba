//! Benchmark and quality-metric functions for data pipeline validation.
//!
//! This module provides six benchmark families covering the data pipeline:
//!
//! 1. **Parser throughput** -- rows/s for catalog CSV/dat parsers.
//! 2. **Ephemeris accuracy** -- positional residuals from reference data.
//! 3. **Gravity truncation** -- RMS convergence vs harmonic degree.
//! 4. **Magnetic coverage** -- temporal gap analysis for Swarm data.
//! 5. **Irradiance gaps** -- TSI time-series gap detection.
//! 6. **Landsat filtering** -- STAC metadata filtering throughput.
//!
//! All benchmarks work offline with synthetic data for CI. When real cached
//! datasets are available, `*_if_available` tests additionally exercise
//! the pipeline against production data.

use std::path::Path;
use std::time::Instant;

use crate::fetcher::FetchError;

// ---------------------------------------------------------------------------
// 1. Parser throughput
// ---------------------------------------------------------------------------

/// Result of a parser throughput benchmark.
#[derive(Debug, Clone)]
pub struct ThroughputResult {
    /// Parser name.
    pub parser: String,
    /// Number of rows parsed.
    pub rows: usize,
    /// Wall-clock time in seconds.
    pub elapsed_s: f64,
    /// Rows per second.
    pub rows_per_sec: f64,
}

/// Generate synthetic CHIME-like CSV with `n` rows.
pub fn synthetic_chime_csv(n: usize) -> String {
    let mut buf = String::with_capacity(n * 300);
    buf.push_str("tns_name,repeater_name,ra,decl,gl,gb,bonsai_dm,dm_fitb,dm_exc_ne2001,dm_exc_ymw16,bonsai_snr,flux,fluence,width_fitb,scat_time,mjd_400,sp_idx,high_freq,low_freq,peak_freq,catalog1\n");
    for i in 0..n {
        buf.push_str(&format!(
            "FRB20200{:05},R{},180.0,-45.0,270.0,-15.0,{:.1},{:.1},{:.1},{:.1},12.5,0.5,3.2,0.001,0.0001,59000.{},0.0,800.0,400.0,600.0,0\n",
            i, i % 100, 300.0 + i as f64 * 0.1, 290.0 + i as f64 * 0.1,
            200.0 + i as f64 * 0.05, 180.0 + i as f64 * 0.05, i
        ));
    }
    buf
}

/// Generate synthetic ATNF-like CSV with `n` rows.
pub fn synthetic_atnf_csv(n: usize) -> String {
    let mut buf = String::with_capacity(n * 200);
    buf.push_str("PSRJ;RAJ;DECJ;DM;P0;P1;BINARY;DIST;AGE;BSURF\n");
    for i in 0..n {
        buf.push_str(&format!(
            "J{:04}+{:04};12:34:56.78;-45:12:34.5;{:.2};0.{:04};1.23e-15;;{:.1};1.0e+09;1.0e+12\n",
            i / 100,
            i % 100,
            100.0 + i as f64 * 0.3,
            100 + i,
            1.0 + i as f64 * 0.01
        ));
    }
    buf
}

/// Generate synthetic Fermi GBM-like CSV with `n` rows.
pub fn synthetic_fermi_csv(n: usize) -> String {
    let mut buf = String::with_capacity(n * 250);
    buf.push_str("name,ra,dec,trigger_time,t90,t50,fluence,flux_64,flux_1024,flnc_best_fitting_model,pflx_best_fitting_model\n");
    for i in 0..n {
        buf.push_str(&format!(
            "GRB{:06}A,12 34 56,+45 12 34,2020-01-{:02}T12:00:00,{:.2},{:.2},1.23e-06,2.34e-07,1.56e-07,comp,comp\n",
            200000 + i, (i % 28) + 1, 2.0 + i as f64 * 0.01, 1.0 + i as f64 * 0.005
        ));
    }
    buf
}

/// Generate synthetic GWTC-like CSV with `n` rows.
pub fn synthetic_gwtc_csv(n: usize) -> String {
    let mut buf = String::with_capacity(n * 200);
    buf.push_str("name,mass_1_source,mass_2_source,chirp_mass_source,luminosity_distance,network_matched_filter_snr,redshift,far\n");
    for i in 0..n {
        buf.push_str(&format!(
            "GW{:06},35.{},25.{},28.{},{:.0},12.{},{:.3},1.0e-10\n",
            150914 + i,
            i % 10,
            i % 10,
            i % 10,
            400.0 + i as f64 * 10.0,
            i % 10,
            0.1 + i as f64 * 0.001
        ));
    }
    buf
}

/// Generate synthetic Pantheon-like .dat with `n` rows.
pub fn synthetic_pantheon_dat(n: usize) -> String {
    let mut buf = String::with_capacity(n * 120);
    buf.push_str("CID zHD zHDERR m_b_corr m_b_corr_err_DIAG\n");
    for i in 0..n {
        buf.push_str(&format!(
            "SN{:05} {:.4} 0.0010 {:.4} 0.15\n",
            i,
            0.01 + i as f64 * 0.001,
            20.0 + 5.0 * (0.01 + i as f64 * 0.001).log10()
        ));
    }
    buf
}

/// Generate synthetic SDSS quasar-like CSV with `n` rows.
pub fn synthetic_sdss_csv(n: usize) -> String {
    let mut buf = String::with_capacity(n * 200);
    buf.push_str("objID,ra,dec,z,plate,mjd,fiberID,psfMag_u,psfMag_g,psfMag_r,psfMag_i,psfMag_z\n");
    for i in 0..n {
        buf.push_str(&format!(
            "{},180.{:04},-5.{:04},{:.3},1234,55000,{},20.1,19.5,19.0,18.8,18.5\n",
            1000000 + i,
            i % 10000,
            i % 10000,
            0.5 + i as f64 * 0.002,
            i % 1000
        ));
    }
    buf
}

/// Benchmark parser throughput by writing synthetic data to a temp file and timing parsing.
///
/// Returns results for each catalog parser. Uses synthetic data sized at `rows_per_parser`.
pub fn benchmark_parser_throughput(
    rows_per_parser: usize,
) -> Result<Vec<ThroughputResult>, FetchError> {
    let dir = std::env::temp_dir().join("data_core_bench_parser");
    std::fs::create_dir_all(&dir).map_err(|e| FetchError::Validation(format!("mkdir: {}", e)))?;

    let mut results = Vec::new();

    // Helper: write content, time parsing, record result
    macro_rules! bench_parser {
        ($name:expr, $gen:expr, $parse:expr) => {{
            let path = dir.join(format!("{}.csv", $name));
            let content = $gen;
            std::fs::write(&path, &content)
                .map_err(|e| FetchError::Validation(format!("write: {}", e)))?;
            let start = Instant::now();
            let rows = $parse(&path)?;
            let elapsed = start.elapsed().as_secs_f64();
            results.push(ThroughputResult {
                parser: $name.to_string(),
                rows,
                elapsed_s: elapsed,
                rows_per_sec: if elapsed > 0.0 {
                    rows as f64 / elapsed
                } else {
                    f64::INFINITY
                },
            });
        }};
    }

    bench_parser!(
        "chime",
        synthetic_chime_csv(rows_per_parser),
        |p: &Path| -> Result<usize, FetchError> {
            Ok(crate::catalogs::chime::parse_chime_csv(p)?.len())
        }
    );

    bench_parser!(
        "atnf",
        synthetic_atnf_csv(rows_per_parser),
        |p: &Path| -> Result<usize, FetchError> {
            Ok(crate::catalogs::atnf::parse_atnf_csv(p)?.len())
        }
    );

    bench_parser!(
        "fermi_gbm",
        synthetic_fermi_csv(rows_per_parser),
        |p: &Path| -> Result<usize, FetchError> {
            Ok(crate::catalogs::fermi_gbm::parse_fermi_gbm_csv(p)?.len())
        }
    );

    bench_parser!(
        "gwtc",
        synthetic_gwtc_csv(rows_per_parser),
        |p: &Path| -> Result<usize, FetchError> {
            Ok(crate::catalogs::gwtc::parse_gwtc3_csv(p)?.len())
        }
    );

    bench_parser!(
        "pantheon",
        synthetic_pantheon_dat(rows_per_parser),
        |p: &Path| -> Result<usize, FetchError> {
            Ok(crate::catalogs::pantheon::parse_pantheon_dat(p)?.len())
        }
    );

    bench_parser!(
        "sdss",
        synthetic_sdss_csv(rows_per_parser),
        |p: &Path| -> Result<usize, FetchError> {
            Ok(crate::catalogs::sdss::parse_sdss_quasar_csv(p)?.len())
        }
    );

    // Cleanup
    std::fs::remove_dir_all(&dir).ok();

    Ok(results)
}

// ---------------------------------------------------------------------------
// 2. Ephemeris interpolation accuracy
// ---------------------------------------------------------------------------

/// Result of an ephemeris accuracy check.
#[derive(Debug, Clone)]
pub struct EphemerisAccuracyResult {
    /// Body name.
    pub body: String,
    /// Number of reference points checked.
    pub n_points: usize,
    /// Maximum absolute RA residual (degrees).
    pub max_ra_residual_deg: f64,
    /// Maximum absolute Dec residual (degrees).
    pub max_dec_residual_deg: f64,
    /// Maximum distance residual (AU).
    pub max_delta_residual_au: f64,
}

/// Generate a synthetic Horizons CSV response for testing.
pub fn synthetic_horizons_csv(body: &str, n: usize) -> String {
    let mut buf = String::with_capacity(n * 120);
    buf.push_str("body,jd,date,ra,dec,delta\n");
    for i in 0..n {
        let jd = 2451545.0 + i as f64 * 30.0; // J2000.0 + steps of 30 days
        let ra = (i as f64 * 12.3456) % 360.0;
        let dec = -90.0 + (i as f64 * 7.891) % 180.0;
        let delta = 0.5 + (i as f64 * 0.123) % 40.0;
        buf.push_str(&format!(
            "{},{:.1},2000-Jan-{:02},{:.6},{:.6},{:.6}\n",
            body,
            jd,
            (i % 28) + 1,
            ra,
            dec,
            delta
        ));
    }
    buf
}

/// Benchmark ephemeris parsing accuracy by comparing parsed values to reference.
///
/// Generates synthetic reference data, writes it, parses it back, and computes
/// residuals. In a real scenario this compares DE440 interpolation vs Horizons.
pub fn benchmark_ephemeris_accuracy() -> Result<Vec<EphemerisAccuracyResult>, FetchError> {
    let dir = std::env::temp_dir().join("data_core_bench_ephemeris");
    std::fs::create_dir_all(&dir).map_err(|e| FetchError::Validation(format!("mkdir: {}", e)))?;

    let bodies = ["Mercury", "Venus", "Mars", "Jupiter"];
    let n = 120; // 10 years at 30-day steps
    let mut results = Vec::new();

    for body in &bodies {
        let content = synthetic_horizons_csv(body, n);
        let path = dir.join(format!("{}_ephemeris.csv", body));
        std::fs::write(&path, &content)
            .map_err(|e| FetchError::Validation(format!("write: {}", e)))?;

        let points = crate::geophysical::jpl_ephemeris::parse_jpl_ephemeris_csv(&path)?;

        // Recompute expected values and compare
        let mut max_ra = 0.0_f64;
        let mut max_dec = 0.0_f64;
        let mut max_delta = 0.0_f64;

        for (i, pt) in points.iter().enumerate() {
            let expected_ra = (i as f64 * 12.3456) % 360.0;
            let expected_dec = -90.0 + (i as f64 * 7.891) % 180.0;
            let expected_delta = 0.5 + (i as f64 * 0.123) % 40.0;

            max_ra = max_ra.max((pt.ra - expected_ra).abs());
            max_dec = max_dec.max((pt.dec - expected_dec).abs());
            max_delta = max_delta.max((pt.delta - expected_delta).abs());
        }

        results.push(EphemerisAccuracyResult {
            body: body.to_string(),
            n_points: points.len(),
            max_ra_residual_deg: max_ra,
            max_dec_residual_deg: max_dec,
            max_delta_residual_au: max_delta,
        });
    }

    std::fs::remove_dir_all(&dir).ok();
    Ok(results)
}

// ---------------------------------------------------------------------------
// 3. Gravity-harmonic truncation error
// ---------------------------------------------------------------------------

/// Result of a gravity truncation convergence check.
#[derive(Debug, Clone)]
pub struct TruncationResult {
    /// Truncation degree.
    pub degree: u32,
    /// Number of coefficients included at this truncation.
    pub n_coefficients: usize,
    /// RMS of C_nm coefficients beyond this degree (residual power).
    pub rms_residual: f64,
}

/// Generate a synthetic GFC file with coefficients up to degree `max_deg`.
///
/// Coefficient magnitudes follow realistic Kaula's rule: |C_nm| ~ 1e-5 / n^2.
pub fn synthetic_gfc(max_deg: u32) -> String {
    let mut buf = String::with_capacity((max_deg as usize + 1).pow(2) * 80);
    buf.push_str("modelname       SyntheticTest\n");
    buf.push_str("earth_gravity_constant  3.986004415E+14\n");
    buf.push_str("radius          6.3781363E+06\n");
    buf.push_str(&format!("max_degree      {}\n", max_deg));
    buf.push_str("end_of_head ===\n");

    // C(0,0) = 1 by convention
    buf.push_str("gfc  0  0  1.000000E+00  0.000000E+00  0.0E+00  0.0E+00\n");

    for n in 1..=max_deg {
        for m in 0..=n {
            // Kaula's rule: amplitude ~ 1e-5 / n^2
            let amplitude = 1.0e-5 / (n as f64).powi(2);
            let cnm = amplitude * if (n + m) % 2 == 0 { 1.0 } else { -1.0 };
            let snm = if m == 0 { 0.0 } else { amplitude * 0.5 };
            buf.push_str(&format!(
                "gfc  {}  {}  {:.6E}  {:.6E}  0.0E+00  0.0E+00\n",
                n, m, cnm, snm
            ));
        }
    }
    buf
}

/// Compute truncation error curve for a gravity field model.
///
/// For each truncation degree in `degrees`, computes the RMS of coefficients
/// with degree > truncation_degree. This models how much gravitational signal
/// is lost by truncating the harmonic expansion.
pub fn benchmark_gravity_truncation(
    gfc_path: &Path,
    degrees: &[u32],
) -> Result<Vec<TruncationResult>, FetchError> {
    let gf = crate::formats::gfc::parse_gfc(gfc_path)?;
    crate::formats::gfc::validate_gfc_degrees(&gf)?;

    let mut results = Vec::new();

    for &trunc_deg in degrees {
        let included: Vec<_> = gf
            .coefficients
            .iter()
            .filter(|c| c.n <= trunc_deg)
            .collect();

        let excluded: Vec<_> = gf.coefficients.iter().filter(|c| c.n > trunc_deg).collect();

        let rms = if excluded.is_empty() {
            0.0
        } else {
            let sum_sq: f64 = excluded.iter().map(|c| c.cnm * c.cnm + c.snm * c.snm).sum();
            (sum_sq / excluded.len() as f64).sqrt()
        };

        results.push(TruncationResult {
            degree: trunc_deg,
            n_coefficients: included.len(),
            rms_residual: rms,
        });
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// 4. Magnetic-field sample coverage
// ---------------------------------------------------------------------------

/// Temporal coverage analysis result.
#[derive(Debug, Clone)]
pub struct CoverageResult {
    /// Total number of records.
    pub total_records: usize,
    /// Number of temporal gaps exceeding the threshold.
    pub gap_count: usize,
    /// Maximum gap duration (as timestamp-string difference; seconds if parseable).
    pub max_gap_seconds: f64,
    /// Coverage fraction: 1.0 - (total_gap_time / total_span).
    pub coverage_fraction: f64,
}

/// Parse an ISO 8601 timestamp to seconds since an arbitrary epoch.
///
/// Simplified parser for "YYYY-MM-DDThh:mm:ssZ" format. Returns seconds
/// since 2000-01-01T00:00:00Z.
fn iso8601_to_seconds(ts: &str) -> Option<f64> {
    // Expected: "2014-01-01T00:00:00Z" (20 chars minimum)
    let ts = ts.trim().trim_end_matches('Z');
    let parts: Vec<&str> = ts.split('T').collect();
    if parts.len() != 2 {
        return None;
    }

    let date_parts: Vec<&str> = parts[0].split('-').collect();
    if date_parts.len() != 3 {
        return None;
    }
    let year: f64 = date_parts[0].parse().ok()?;
    let month: f64 = date_parts[1].parse().ok()?;
    let day: f64 = date_parts[2].parse().ok()?;

    let time_parts: Vec<&str> = parts[1].split(':').collect();
    if time_parts.len() < 2 {
        return None;
    }
    let hour: f64 = time_parts[0].parse().ok()?;
    let min: f64 = time_parts[1].parse().ok()?;
    let sec: f64 = if time_parts.len() > 2 {
        time_parts[2].parse().unwrap_or(0.0)
    } else {
        0.0
    };

    // Approximate days since 2000-01-01
    let days = (year - 2000.0) * 365.25 + (month - 1.0) * 30.44 + (day - 1.0);
    Some(days * 86400.0 + hour * 3600.0 + min * 60.0 + sec)
}

/// Analyze temporal coverage of Swarm magnetic field records.
///
/// Identifies gaps where consecutive timestamps differ by more than
/// `gap_threshold_seconds`. Returns coverage statistics.
pub fn benchmark_magnetic_coverage(
    records: &[crate::geophysical::swarm::SwarmRecord],
    gap_threshold_seconds: f64,
) -> CoverageResult {
    if records.len() < 2 {
        return CoverageResult {
            total_records: records.len(),
            gap_count: 0,
            max_gap_seconds: 0.0,
            coverage_fraction: if records.is_empty() { 0.0 } else { 1.0 },
        };
    }

    let timestamps: Vec<f64> = records
        .iter()
        .filter_map(|r| iso8601_to_seconds(&r.timestamp))
        .collect();

    if timestamps.len() < 2 {
        return CoverageResult {
            total_records: records.len(),
            gap_count: 0,
            max_gap_seconds: 0.0,
            coverage_fraction: 1.0,
        };
    }

    let mut gap_count = 0;
    let mut max_gap = 0.0_f64;
    let mut total_gap = 0.0_f64;

    for pair in timestamps.windows(2) {
        let dt = pair[1] - pair[0];
        if dt > gap_threshold_seconds {
            gap_count += 1;
            max_gap = max_gap.max(dt);
            total_gap += dt - gap_threshold_seconds;
        }
    }

    let total_span = timestamps.last().unwrap() - timestamps.first().unwrap();
    let coverage = if total_span > 0.0 {
        1.0 - (total_gap / total_span)
    } else {
        1.0
    };

    CoverageResult {
        total_records: records.len(),
        gap_count,
        max_gap_seconds: max_gap,
        coverage_fraction: coverage.clamp(0.0, 1.0),
    }
}

// ---------------------------------------------------------------------------
// 5. Irradiance time-series gap detection
// ---------------------------------------------------------------------------

/// Gap detection result for a time series.
#[derive(Debug, Clone)]
pub struct GapDetectionResult {
    /// Total number of data points.
    pub total_points: usize,
    /// Expected number of days in the span (first to last JD).
    pub span_days: f64,
    /// Number of gaps exceeding the threshold.
    pub gap_count: usize,
    /// Maximum gap duration (days).
    pub max_gap_days: f64,
    /// Coverage fraction: actual points / expected points at daily cadence.
    pub coverage_fraction: f64,
}

/// Detect gaps in a TSI time series given as (JD, TSI) pairs.
///
/// A gap is any interval between consecutive measurements exceeding
/// `gap_threshold_days` (typically 1.5 for a daily series with margin).
pub fn detect_irradiance_gaps(jd_values: &[f64], gap_threshold_days: f64) -> GapDetectionResult {
    if jd_values.len() < 2 {
        return GapDetectionResult {
            total_points: jd_values.len(),
            span_days: 0.0,
            gap_count: 0,
            max_gap_days: 0.0,
            coverage_fraction: if jd_values.is_empty() { 0.0 } else { 1.0 },
        };
    }

    let span = jd_values.last().unwrap() - jd_values.first().unwrap();
    let mut gap_count = 0;
    let mut max_gap = 0.0_f64;

    for pair in jd_values.windows(2) {
        let dt = pair[1] - pair[0];
        if dt > gap_threshold_days {
            gap_count += 1;
            max_gap = max_gap.max(dt);
        }
    }

    // Expected daily cadence
    let expected_points = span.ceil() as usize + 1;
    let coverage = jd_values.len() as f64 / expected_points.max(1) as f64;

    GapDetectionResult {
        total_points: jd_values.len(),
        span_days: span,
        gap_count,
        max_gap_days: max_gap,
        coverage_fraction: coverage.min(1.0),
    }
}

/// Generate synthetic TSI JD values with known gaps for testing.
///
/// Creates a daily series from `start_jd` for `n_days`, with gaps inserted
/// at the specified day offsets.
pub fn synthetic_tsi_jd_series(start_jd: f64, n_days: usize, gap_offsets: &[usize]) -> Vec<f64> {
    let mut jds = Vec::with_capacity(n_days);
    for day in 0..n_days {
        if !gap_offsets.contains(&day) {
            jds.push(start_jd + day as f64);
        }
    }
    jds
}

// ---------------------------------------------------------------------------
// 6. Landsat STAC metadata filtering
// ---------------------------------------------------------------------------

/// Landsat filtering benchmark result.
#[derive(Debug, Clone)]
pub struct LandsatFilterResult {
    /// Number of STAC items processed.
    pub items_processed: usize,
    /// Number passing the cloud cover filter.
    pub items_passing: usize,
    /// Wall-clock time in seconds.
    pub elapsed_s: f64,
    /// Items per second.
    pub items_per_sec: f64,
}

/// Generate a synthetic STAC item JSON string.
pub fn synthetic_stac_item(id: &str, cloud_cover: f64) -> String {
    format!(
        r#"{{
  "type": "Feature",
  "stac_version": "1.0.0",
  "id": "{}",
  "geometry": {{"type": "Polygon", "coordinates": [[[0,0],[1,0],[1,1],[0,1],[0,0]]]}},
  "properties": {{
    "datetime": "2021-12-05T00:00:00Z",
    "eo:cloud_cover": {},
    "platform": "landsat-9",
    "instruments": ["oli2", "tirs2"],
    "landsat:scene_id": "LC90090242021339LGN00"
  }},
  "assets": {{
    "B1": {{"href": "https://example.com/B1.TIF", "type": "image/tiff"}},
    "B2": {{"href": "https://example.com/B2.TIF", "type": "image/tiff"}},
    "B3": {{"href": "https://example.com/B3.TIF", "type": "image/tiff"}}
  }}
}}"#,
        id, cloud_cover
    )
}

/// Benchmark Landsat STAC metadata filtering throughput.
///
/// Generates `n` synthetic STAC items with varying cloud cover, writes each to a
/// temp file, and times the extraction of cloud cover + schema validation.
pub fn benchmark_landsat_filtering(
    n: usize,
    max_cloud_cover: f64,
) -> Result<LandsatFilterResult, FetchError> {
    let dir = std::env::temp_dir().join("data_core_bench_landsat");
    std::fs::create_dir_all(&dir).map_err(|e| FetchError::Validation(format!("mkdir: {}", e)))?;

    // Write synthetic STAC items
    let mut paths = Vec::with_capacity(n);
    for i in 0..n {
        let cloud = (i as f64 / n as f64) * 100.0; // 0% to ~100%
        let content = synthetic_stac_item(&format!("LC09_TEST_{:06}", i), cloud);
        let path = dir.join(format!("item_{:06}.json", i));
        std::fs::write(&path, &content)
            .map_err(|e| FetchError::Validation(format!("write: {}", e)))?;
        paths.push(path);
    }

    // Benchmark: validate schema + extract cloud cover + filter
    let start = Instant::now();
    let mut passing = 0;
    for path in &paths {
        crate::catalogs::landsat::validate_stac_schema(path)?;
        if let Some(cc) = crate::catalogs::landsat::extract_cloud_cover(path)? {
            if cc <= max_cloud_cover {
                passing += 1;
            }
        }
    }
    let elapsed = start.elapsed().as_secs_f64();

    std::fs::remove_dir_all(&dir).ok();

    Ok(LandsatFilterResult {
        items_processed: n,
        items_passing: passing,
        elapsed_s: elapsed,
        items_per_sec: if elapsed > 0.0 {
            n as f64 / elapsed
        } else {
            f64::INFINITY
        },
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- 1. Parser throughput --

    #[test]
    fn test_synthetic_chime_csv_valid() {
        let csv = synthetic_chime_csv(100);
        let dir = std::env::temp_dir().join("bench_test_chime");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("chime.csv");
        std::fs::write(&path, &csv).unwrap();
        let events = crate::catalogs::chime::parse_chime_csv(&path).unwrap();
        assert_eq!(events.len(), 100);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_synthetic_atnf_csv_valid() {
        let csv = synthetic_atnf_csv(50);
        let dir = std::env::temp_dir().join("bench_test_atnf");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("atnf.csv");
        std::fs::write(&path, &csv).unwrap();
        let pulsars = crate::catalogs::atnf::parse_atnf_csv(&path).unwrap();
        assert_eq!(pulsars.len(), 50);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_benchmark_parser_throughput_runs() {
        let results = benchmark_parser_throughput(500).unwrap();
        assert!(results.len() >= 6, "Should have results for 6 parsers");
        for r in &results {
            assert!(r.rows > 0, "Parser {} returned 0 rows", r.parser);
            assert!(r.elapsed_s >= 0.0);
            eprintln!(
                "  {} : {} rows in {:.4}s ({:.0} rows/s)",
                r.parser, r.rows, r.elapsed_s, r.rows_per_sec
            );
        }
    }

    // -- 2. Ephemeris accuracy --

    #[test]
    fn test_benchmark_ephemeris_accuracy_runs() {
        let results = benchmark_ephemeris_accuracy().unwrap();
        assert_eq!(results.len(), 4, "Should have results for 4 bodies");
        for r in &results {
            assert_eq!(r.n_points, 120);
            // Synthetic round-trip should have zero residual (within f64 precision)
            assert!(
                r.max_ra_residual_deg < 1e-6,
                "{}: RA residual {:.2e} too large",
                r.body,
                r.max_ra_residual_deg
            );
            assert!(
                r.max_dec_residual_deg < 1e-6,
                "{}: Dec residual {:.2e} too large",
                r.body,
                r.max_dec_residual_deg
            );
            assert!(
                r.max_delta_residual_au < 1e-6,
                "{}: delta residual {:.2e} too large",
                r.body,
                r.max_delta_residual_au
            );
        }
    }

    // -- 3. Gravity truncation --

    #[test]
    fn test_benchmark_gravity_truncation_convergence() {
        let gfc_content = synthetic_gfc(36);
        let dir = std::env::temp_dir().join("bench_test_gfc");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("synthetic.gfc");
        std::fs::write(&path, &gfc_content).unwrap();

        let degrees = [4, 8, 16, 24, 36];
        let results = benchmark_gravity_truncation(&path, &degrees).unwrap();

        assert_eq!(results.len(), 5);

        // RMS residual should decrease with increasing truncation degree
        for pair in results.windows(2) {
            assert!(
                pair[1].rms_residual <= pair[0].rms_residual,
                "RMS should decrease: degree {} ({:.2e}) -> degree {} ({:.2e})",
                pair[0].degree,
                pair[0].rms_residual,
                pair[1].degree,
                pair[1].rms_residual,
            );
        }

        // At max degree, residual should be zero
        assert!(
            results.last().unwrap().rms_residual < 1e-15,
            "At max degree, residual should be zero"
        );

        eprintln!("Truncation convergence:");
        for r in &results {
            eprintln!(
                "  degree {:3}: {} coefficients, RMS residual = {:.4e}",
                r.degree, r.n_coefficients, r.rms_residual
            );
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    // -- 4. Magnetic coverage --

    #[test]
    fn test_benchmark_magnetic_coverage_no_gaps() {
        let records: Vec<crate::geophysical::swarm::SwarmRecord> = (0..100)
            .map(|i| crate::geophysical::swarm::SwarmRecord {
                timestamp: format!("2014-01-01T00:{:02}:{:02}Z", i / 60, i % 60),
                latitude: 0.0,
                longitude: 0.0,
                radius: 6871200.0,
                f_total: 48000.0,
            })
            .collect();

        let result = benchmark_magnetic_coverage(&records, 2.0);
        assert_eq!(result.total_records, 100);
        assert_eq!(result.gap_count, 0);
        assert!(result.coverage_fraction > 0.99);
    }

    #[test]
    fn test_benchmark_magnetic_coverage_with_gaps() {
        // 3 records with a 60-second gap in the middle
        let records = vec![
            crate::geophysical::swarm::SwarmRecord {
                timestamp: "2014-01-01T00:00:00Z".into(),
                latitude: 0.0,
                longitude: 0.0,
                radius: 6871200.0,
                f_total: 48000.0,
            },
            crate::geophysical::swarm::SwarmRecord {
                timestamp: "2014-01-01T00:00:01Z".into(),
                latitude: 0.0,
                longitude: 0.0,
                radius: 6871200.0,
                f_total: 48000.0,
            },
            crate::geophysical::swarm::SwarmRecord {
                timestamp: "2014-01-01T00:01:01Z".into(),
                latitude: 0.0,
                longitude: 0.0,
                radius: 6871200.0,
                f_total: 48000.0,
            },
        ];

        let result = benchmark_magnetic_coverage(&records, 5.0);
        assert_eq!(result.total_records, 3);
        assert_eq!(result.gap_count, 1, "Should detect 1 gap > 5 seconds");
        assert!(result.max_gap_seconds > 50.0, "Max gap should be ~60s");
    }

    // -- 5. Irradiance gaps --

    #[test]
    fn test_detect_irradiance_gaps_no_gaps() {
        let jds = synthetic_tsi_jd_series(2459000.0, 365, &[]);
        let result = detect_irradiance_gaps(&jds, 1.5);
        assert_eq!(result.total_points, 365);
        assert_eq!(result.gap_count, 0);
        assert!(result.coverage_fraction > 0.99);
    }

    #[test]
    fn test_detect_irradiance_gaps_with_known_gaps() {
        // Insert gaps at day 50, 51, 52 (3-day gap) and day 200 (1-day gap)
        let jds = synthetic_tsi_jd_series(2459000.0, 365, &[50, 51, 52, 200]);
        let result = detect_irradiance_gaps(&jds, 1.5);
        assert_eq!(result.total_points, 361, "365 - 4 missing days");
        // The 3-day gap creates a 4-day interval (day 49 to day 53), which is 1 gap
        // The 1-day gap at day 200 creates a 2-day interval, which is 1 gap
        assert!(result.gap_count >= 1, "Should detect at least 1 gap");
        assert!(result.max_gap_days > 2.0, "Max gap should be > 2 days");
        assert!(result.coverage_fraction < 1.0, "Coverage should be < 100%");
        eprintln!(
            "Irradiance gaps: {} points, {} gaps, max {:.1} days, coverage {:.1}%",
            result.total_points,
            result.gap_count,
            result.max_gap_days,
            result.coverage_fraction * 100.0
        );
    }

    #[test]
    fn test_detect_irradiance_gaps_empty() {
        let result = detect_irradiance_gaps(&[], 1.5);
        assert_eq!(result.total_points, 0);
        assert_eq!(result.gap_count, 0);
        assert!((result.coverage_fraction - 0.0).abs() < 1e-10);
    }

    // -- 6. Landsat filtering --

    #[test]
    fn test_synthetic_stac_item_valid() {
        let json = synthetic_stac_item("TEST_001", 25.5);
        let dir = std::env::temp_dir().join("bench_test_stac");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("item.json");
        std::fs::write(&path, &json).unwrap();

        assert!(crate::catalogs::landsat::looks_like_landsat_stac_json(&path).unwrap());
        crate::catalogs::landsat::validate_stac_schema(&path).unwrap();
        let cc = crate::catalogs::landsat::extract_cloud_cover(&path).unwrap();
        assert!((cc.unwrap() - 25.5).abs() < 0.01);
        let assets = crate::catalogs::landsat::count_stac_assets(&path).unwrap();
        assert_eq!(assets, 3, "Should find 3 asset hrefs");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_benchmark_landsat_filtering_runs() {
        let result = benchmark_landsat_filtering(100, 30.0).unwrap();
        assert_eq!(result.items_processed, 100);
        // With cloud cover from 0 to 99, ~30% should pass the <= 30.0 filter
        assert!(
            result.items_passing > 20 && result.items_passing < 40,
            "Expected ~30 items passing, got {}",
            result.items_passing
        );
        assert!(result.elapsed_s >= 0.0);
        eprintln!(
            "Landsat filtering: {}/{} passing in {:.4}s ({:.0} items/s)",
            result.items_passing, result.items_processed, result.elapsed_s, result.items_per_sec
        );
    }

    // -- Integration: real data if available --

    #[test]
    fn test_irradiance_gaps_if_tsis_available() {
        let path = std::path::Path::new("data/external/tsis_tsi_daily.csv");
        if !path.exists() {
            eprintln!("Skipping: TSIS data not available");
            return;
        }
        let measurements = crate::catalogs::tsi::parse_tsi_csv(path).unwrap();
        let jds: Vec<f64> = measurements.iter().map(|m| m.jd).collect();
        let result = detect_irradiance_gaps(&jds, 1.5);
        eprintln!(
            "TSIS: {} points over {:.0} days, {} gaps, max {:.1} days, coverage {:.1}%",
            result.total_points,
            result.span_days,
            result.gap_count,
            result.max_gap_days,
            result.coverage_fraction * 100.0
        );
        assert!(
            result.total_points > 100,
            "TSIS should have > 100 daily points"
        );
    }

    #[test]
    fn test_irradiance_gaps_if_sorce_available() {
        let path = std::path::Path::new("data/external/sorce_tsi_daily.csv");
        if !path.exists() {
            eprintln!("Skipping: SORCE data not available");
            return;
        }
        let measurements = crate::catalogs::sorce::parse_sorce_csv(path).unwrap();
        let jds: Vec<f64> = measurements.iter().map(|m| m.jd).collect();
        let result = detect_irradiance_gaps(&jds, 1.5);
        eprintln!(
            "SORCE: {} points over {:.0} days, {} gaps, max {:.1} days, coverage {:.1}%",
            result.total_points,
            result.span_days,
            result.gap_count,
            result.max_gap_days,
            result.coverage_fraction * 100.0
        );
        assert!(
            result.total_points > 100,
            "SORCE should have > 100 daily points"
        );
    }

    #[test]
    fn test_magnetic_coverage_if_swarm_available() {
        let path = std::path::Path::new("data/external/swarm_magnetic_sample.csv");
        if !path.exists() {
            eprintln!("Skipping: Swarm data not available");
            return;
        }
        let records = crate::geophysical::swarm::parse_swarm_csv(path).unwrap();
        let result = benchmark_magnetic_coverage(&records, 2.0);
        eprintln!(
            "Swarm: {} records, {} gaps > 2s, max gap {:.1}s, coverage {:.1}%",
            result.total_records,
            result.gap_count,
            result.max_gap_seconds,
            result.coverage_fraction * 100.0
        );
        assert!(result.total_records > 0, "Swarm should have records");
    }
}

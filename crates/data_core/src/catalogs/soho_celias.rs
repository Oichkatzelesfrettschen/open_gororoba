//! SOHO CELIAS Proton Monitor mission-long bundle parser.
//!
//! The mission-long bundle published by the SOHO archive is a `tar.gz`
//! containing one yearly ZIP archive per year. Each yearly ZIP contains a
//! plain-text table of CELIAS/MTOF Proton Monitor measurements at 5-minute or
//! sub-5-minute cadence along with SOHO orbit geometry.
//!
//! This module parses the bundle into typed records and provides an hourly
//! `OmniRecord` adapter using robust per-hour medians. This keeps the SOHO
//! inner-boundary lane compatible with the rest of the current heliosphere
//! stack, which is primarily hourly.

use crate::{
    catalogs::omni::OmniRecord,
    cdf_support::{
        cdf_scalar_f64_rows, cdf_type_to_unix_ms, cdf_variable_rows, cdf_vector_f64_rows,
        filename_date_yyyymmdd, read_cdf_file, ymd_to_doy,
    },
    fetcher::FetchError,
    parse::parse_f64_or_nan,
};
use chrono::{Datelike, NaiveDate, TimeZone, Timelike, Utc};
use flate2::read::GzDecoder;
use std::{
    collections::BTreeMap,
    io::{Cursor, Read},
    path::Path,
};
use tar::Archive;
use zip::ZipArchive;
const AU_MKM: f64 = 149.597_870_7;
const PROTON_MASS_KG: f64 = 1.672_621_923_69e-27;
const BOLTZMANN_J_PER_K: f64 = 1.380_649e-23;
#[derive(Debug, Clone)]
pub struct SohoCeliasRecord {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
    pub bulk_speed: f64,
    pub proton_density: f64,
    pub thermal_speed_kms: f64,
    pub proton_temperature: f64,
    pub r_au: f64,
    pub lat_deg: f64,
    pub lon_deg: f64,
}

#[cfg(feature = "fetch")]
pub use super::soho_celias_fetch::{
    SohoCeliasBundleProvider, SohoCeliasPm5MinProvider, SohoLascoDaySampleProvider,
};

#[derive(Debug, Clone)]
pub struct SohoLascoSampleRecord {
    pub file_name: String,
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub camera: String,
    pub exposure_seconds: f64,
    pub width: u16,
    pub height: u16,
}

// CELIAS text sentinels are negative floats (e.g. -1, -1.00), not strings
// like "N/A". parse_f64_or_nan from crate::parse checks string sentinels and
// would pass through -1.0 as valid data, so we use local parsers instead.
fn parse_nonnegative_or_nan(text: &str) -> f64 {
    match text.trim().parse::<f64>() {
        Ok(v) if v <= -0.5 => f64::NAN,
        Ok(v) => v,
        Err(_) => f64::NAN,
    }
}

fn parse_signed_or_nan(text: &str) -> f64 {
    text.trim().parse::<f64>().unwrap_or(f64::NAN)
}

fn proton_temperature_from_vth(thermal_speed_kms: f64) -> f64 {
    if !thermal_speed_kms.is_finite() || thermal_speed_kms <= 0.0 {
        return f64::NAN;
    }
    let speed_m_per_s = thermal_speed_kms * 1000.0;
    PROTON_MASS_KG * speed_m_per_s * speed_m_per_s / (2.0 * BOLTZMANN_J_PER_K)
}

// SOHO CELIAS text tables use two-digit years. The mission launched in 1996,
// so values >= 90 belong to 1990-1999 and values < 90 to 2000-2089.
fn parse_two_digit_year(text: &str) -> Option<u16> {
    let year = text.parse::<u16>().ok()?;
    Some(if year >= 90 { 1900 + year } else { 2000 + year })
}

fn normalize_lon_deg(lon_deg: f64) -> f64 {
    if !lon_deg.is_finite() {
        return f64::NAN;
    }
    lon_deg.rem_euclid(360.0)
}

fn sanitize_nonnegative(value: f64) -> f64 {
    if !value.is_finite() || value.abs() >= 1.0e30 || value < 0.0 {
        f64::NAN
    } else {
        value
    }
}

fn sanitize_signed(value: f64) -> f64 {
    if !value.is_finite() || value.abs() >= 1.0e30 {
        f64::NAN
    } else {
        value
    }
}

// GSE position columns in CELIAS CDFs appear in three different unit systems
// depending on the data version:
//   > 1e5  km  (raw km from the Sun; divide by 1 AU in km = 149,597,870.7 km)
//   > 10   Re  (Earth radii; multiply by Re_km = 6,371 then divide by AU_km)
//   <= 10  AU  (already in AU; use directly)
// SOHO orbits at roughly L1 (~0.01 AU from Earth ~= 1,500,000 km), so the
// thresholds separate the regimes without overlap.
fn infer_soho_r_au_from_gse(vector: Option<&[f64]>) -> f64 {
    let Some(vector) = vector else {
        return 1.0;
    };
    if vector.len() < 3 {
        return 1.0;
    }
    let norm = vector
        .iter()
        .take(3)
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if !norm.is_finite() || norm <= 0.0 {
        return 1.0;
    }
    if norm > 1.0e5 {
        norm / 149_597_870.7
    } else if norm > 10.0 {
        (norm * 6_371.0) / 149_597_870.7
    } else {
        norm
    }
}

fn cdf_row_time_components(
    epoch_rows: Option<&[Vec<cdf::types::CdfType>]>,
    idx: usize,
    fallback_year: u16,
    fallback_doy: u16,
) -> (u16, u16, u8, u8, u8) {
    if let Some(epoch_rows) = epoch_rows
        && let Some(value) = epoch_rows.get(idx).and_then(|row| row.first())
        && let Some(timestamp_ms) = cdf_type_to_unix_ms(value)
        && let Some(dt) = Utc.timestamp_millis_opt(timestamp_ms).single()
    {
        return (
            dt.year() as u16,
            dt.ordinal() as u16,
            dt.hour() as u8,
            dt.minute() as u8,
            dt.second() as u8,
        );
    }
    let minute_of_day = (idx as u32) * 5;
    (
        fallback_year,
        fallback_doy,
        (minute_of_day / 60) as u8,
        (minute_of_day % 60) as u8,
        0,
    )
}

/// Parse a yearly CELIAS text table.
pub fn parse_soho_celias_text(content: &str) -> Vec<SohoCeliasRecord> {
    let mut records = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 16 {
            continue;
        }
        let Some(year) = parse_two_digit_year(fields[0]) else {
            continue;
        };
        let day: Vec<&str> = fields[3].split(':').collect();
        if day.len() != 4 {
            continue;
        }
        let (Ok(doy), Ok(hour), Ok(minute), Ok(second)) = (
            day[0].parse::<u16>(),
            day[1].parse::<u8>(),
            day[2].parse::<u8>(),
            day[3].parse::<u8>(),
        ) else {
            continue;
        };

        let bulk_speed = parse_nonnegative_or_nan(fields[4]);
        let proton_density = parse_nonnegative_or_nan(fields[5]);
        let thermal_speed_kms = parse_nonnegative_or_nan(fields[6]);
        let range_mkm = parse_nonnegative_or_nan(fields[12]);
        let lat_deg = parse_signed_or_nan(fields[13]);
        let lon_deg = normalize_lon_deg(parse_signed_or_nan(fields[14]));

        records.push(SohoCeliasRecord {
            year,
            doy,
            hour,
            minute,
            second,
            bulk_speed,
            proton_density,
            thermal_speed_kms,
            proton_temperature: proton_temperature_from_vth(thermal_speed_kms),
            r_au: if range_mkm.is_finite() {
                range_mkm / AU_MKM
            } else {
                f64::NAN
            },
            lat_deg,
            lon_deg,
        });
    }
    records
}

/// Parse an official SOHO CELIAS daily CDF into native 5-minute-like records.
pub fn parse_soho_celias_cdf_file(path: &Path) -> Result<Vec<SohoCeliasRecord>, FetchError> {
    let cdf = read_cdf_file(path)?;
    let (file_year, file_month, file_day) = filename_date_yyyymmdd(path).ok_or_else(|| {
        FetchError::Validation(format!(
            "could not infer SOHO CELIAS date from filename {}",
            path.display()
        ))
    })?;
    let file_doy = ymd_to_doy(file_year, file_month, file_day).ok_or_else(|| {
        FetchError::Validation(format!(
            "invalid SOHO CELIAS file date {}-{:02}-{:02}",
            file_year, file_month, file_day
        ))
    })?;

    let speeds = cdf_scalar_f64_rows(&cdf, "V_p")?;
    let densities = cdf_scalar_f64_rows(&cdf, "N_p")?;
    let thermal = cdf_scalar_f64_rows(&cdf, "Vth_p")?;
    let latitudes = cdf_scalar_f64_rows(&cdf, "HG_LAT")?;
    let longitudes = cdf_scalar_f64_rows(&cdf, "HG_LONG")?;
    let gse_positions = cdf_vector_f64_rows(&cdf, "GSE_POS").ok();
    let epoch_rows = cdf_variable_rows(&cdf, "Epoch").ok();

    let row_count = *[
        speeds.len(),
        densities.len(),
        thermal.len(),
        latitudes.len(),
        longitudes.len(),
    ]
    .iter()
    .max()
    .unwrap_or(&0);
    if row_count == 0 {
        return Err(FetchError::Validation(format!(
            "SOHO CELIAS CDF {} yielded zero rows",
            path.display()
        )));
    }

    let mut records = Vec::with_capacity(row_count);
    for idx in 0..row_count {
        let (year, doy, hour, minute, second) =
            cdf_row_time_components(epoch_rows.as_deref(), idx, file_year, file_doy);
        let thermal_speed_kms = thermal
            .get(idx)
            .copied()
            .map(sanitize_nonnegative)
            .unwrap_or(f64::NAN);
        records.push(SohoCeliasRecord {
            year,
            doy,
            hour,
            minute,
            second,
            bulk_speed: speeds
                .get(idx)
                .copied()
                .map(sanitize_nonnegative)
                .unwrap_or(f64::NAN),
            proton_density: densities
                .get(idx)
                .copied()
                .map(sanitize_nonnegative)
                .unwrap_or(f64::NAN),
            thermal_speed_kms,
            proton_temperature: proton_temperature_from_vth(thermal_speed_kms),
            r_au: infer_soho_r_au_from_gse(
                gse_positions
                    .as_ref()
                    .and_then(|rows| rows.get(idx).map(Vec::as_slice)),
            ),
            lat_deg: latitudes
                .get(idx)
                .copied()
                .map(sanitize_signed)
                .unwrap_or(f64::NAN),
            lon_deg: longitudes
                .get(idx)
                .copied()
                .map(sanitize_signed)
                .map(normalize_lon_deg)
                .unwrap_or(f64::NAN),
        });
    }
    records.sort_by_key(|r| (r.year, r.doy, r.hour, r.minute, r.second));
    Ok(records)
}

pub fn parse_soho_lasco_img_hdr(content: &str) -> Vec<SohoLascoSampleRecord> {
    let mut rows = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 7 {
            continue;
        }
        let Ok(date) = NaiveDate::parse_from_str(fields[1], "%Y/%m/%d") else {
            continue;
        };
        let Ok(time) = chrono::NaiveTime::parse_from_str(fields[2], "%H:%M:%S") else {
            continue;
        };
        let timestamp = date.and_time(time);
        rows.push(SohoLascoSampleRecord {
            file_name: fields[0].to_string(),
            year: date.year() as u16,
            doy: date.ordinal() as u16,
            hour: timestamp.hour() as u8,
            camera: fields[3].to_string(),
            exposure_seconds: parse_f64_or_nan(fields[4]),
            width: fields[5].parse::<u16>().unwrap_or(0),
            height: fields[6].parse::<u16>().unwrap_or(0),
        });
    }
    rows
}

pub fn parse_soho_lasco_img_hdr_file(
    path: &Path,
) -> Result<Vec<SohoLascoSampleRecord>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|err| FetchError::Validation(format!("read error: {err}")))?;
    Ok(parse_soho_lasco_img_hdr(&content))
}

/// Parse the mission-long `tar.gz` bundle from disk.
pub fn parse_soho_celias_bundle_file(path: &Path) -> Result<Vec<SohoCeliasRecord>, FetchError> {
    let file = std::fs::File::open(path)
        .map_err(|e| FetchError::Validation(format!("Read error: {}", e)))?;
    let gz = GzDecoder::new(file);
    let mut tar = Archive::new(gz);
    let mut all_records = Vec::new();

    for entry in tar
        .entries()
        .map_err(|e| FetchError::Validation(format!("tar read error: {}", e)))?
    {
        let mut entry =
            entry.map_err(|e| FetchError::Validation(format!("tar entry error: {}", e)))?;
        let path = entry
            .path()
            .map_err(|e| FetchError::Validation(format!("tar path error: {}", e)))?;
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !name.ends_with(".zip") {
            continue;
        }
        let mut zip_bytes = Vec::new();
        entry
            .read_to_end(&mut zip_bytes)
            .map_err(|e| FetchError::Validation(format!("zip payload read error: {}", e)))?;

        let cursor = Cursor::new(zip_bytes);
        let mut zip = ZipArchive::new(cursor)
            .map_err(|e| FetchError::Validation(format!("zip open error: {}", e)))?;
        for i in 0..zip.len() {
            let mut member = zip
                .by_index(i)
                .map_err(|e| FetchError::Validation(format!("zip member error: {}", e)))?;
            let member_name = member.name().to_ascii_lowercase();
            if !member_name.ends_with(".txt") {
                continue;
            }
            let mut text = String::new();
            member
                .read_to_string(&mut text)
                .map_err(|e| FetchError::Validation(format!("text member read error: {}", e)))?;
            all_records.extend(parse_soho_celias_text(&text));
        }
    }

    all_records.sort_by_key(|r| (r.year, r.doy, r.hour, r.minute, r.second));
    Ok(all_records)
}

fn median(values: impl Iterator<Item = f64>) -> f64 {
    let mut vals: Vec<f64> = values.filter(|v| v.is_finite()).collect();
    if vals.is_empty() {
        return f64::NAN;
    }
    vals.sort_by(|a, b| a.total_cmp(b));
    let mid = vals.len() / 2;
    if vals.len().is_multiple_of(2) {
        0.5 * (vals[mid - 1] + vals[mid])
    } else {
        vals[mid]
    }
}

/// Downsample the native CELIAS cadence to hourly medians in `OmniRecord`.
pub fn soho_to_hourly_omni(records: &[SohoCeliasRecord]) -> Vec<OmniRecord> {
    let mut bins: BTreeMap<(u16, u16, u8), Vec<&SohoCeliasRecord>> = BTreeMap::new();
    for record in records {
        // Skip records where every plasma quantity is non-finite (all-fill rows).
        if !record.bulk_speed.is_finite()
            && !record.proton_density.is_finite()
            && !record.proton_temperature.is_finite()
        {
            continue;
        }
        bins.entry((record.year, record.doy, record.hour))
            .or_default()
            .push(record);
    }

    bins.into_iter()
        .map(|((year, doy, hour), rows)| OmniRecord {
            year,
            doy,
            hour,
            b_magnitude: f64::NAN,
            bx_gse: f64::NAN,
            by_gse: f64::NAN,
            bz_gse: f64::NAN,
            proton_temperature: median(rows.iter().map(|r| r.proton_temperature)),
            proton_density: median(rows.iter().map(|r| r.proton_density)),
            bulk_speed: median(rows.iter().map(|r| r.bulk_speed)),
            flow_pressure: f64::NAN,
            plasma_beta: f64::NAN,
            alfven_mach: f64::NAN,
            dst_index: f64::NAN,
            ae_index: f64::NAN,
            kp_times_10: 0,
            r_au: median(rows.iter().map(|r| r.r_au)),
            lat_deg: median(rows.iter().map(|r| r.lat_deg)),
            lon_deg: normalize_lon_deg(median(rows.iter().map(|r| r.lon_deg))),
        })
        .collect()
}

/// Preserve the native CELIAS cadence in OmniRecord-like form.
///
/// This keeps every valid CELIAS sample as an ordered boundary record for
/// high-cadence inner-heliosphere runs. Sub-hour timestamps remain available in
/// the source `SohoCeliasRecord` values and in staged CSV artifacts; this
/// adapter is for solver paths that only need ordered samples plus physical
/// quantities.
pub fn soho_to_native_omni(records: &[SohoCeliasRecord]) -> Vec<OmniRecord> {
    records
        .iter()
        .filter(|record| {
            record.bulk_speed.is_finite()
                || record.proton_density.is_finite()
                || record.proton_temperature.is_finite()
        })
        .map(|record| OmniRecord {
            year: record.year,
            doy: record.doy,
            hour: record.hour,
            b_magnitude: f64::NAN,
            bx_gse: f64::NAN,
            by_gse: f64::NAN,
            bz_gse: f64::NAN,
            proton_temperature: record.proton_temperature,
            proton_density: record.proton_density,
            bulk_speed: record.bulk_speed,
            flow_pressure: f64::NAN,
            plasma_beta: f64::NAN,
            alfven_mach: f64::NAN,
            dst_index: f64::NAN,
            ae_index: f64::NAN,
            kp_times_10: 0,
            r_au: record.r_au,
            lat_deg: record.lat_deg,
            lon_deg: record.lon_deg,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::{Compression, write::GzEncoder};
    use std::{fs::File, io::Write};
    use tar::Builder;
    use tempfile::NamedTempFile;
    use zip::write::SimpleFileOptions;

    const SAMPLE_TEXT: &str = "\
                      SOHO/CELIAS/MTOF Proton Monitor
YY MON DY DOY:HH:MM:SS   SPEED     Np     Vth    N/S   V_He      GSE_X  GSE_Y  GSE_Z  RANGE  HGLAT  HGLONG CRN(E)
96 Jan 20 020:20:18:00     446  10.14      44    1.9    466      211.2 -100.2  -10.9  145.9   -5.1   309.7   1905
96 Jan 20 020:20:23:04     445  10.16      44    1.5    465      211.2 -100.2  -10.9  145.9   -5.1   309.6   1905
96 Jan 20 020:21:18:00      -1  -1.00      -1    0.0     -1      211.2 -100.2  -10.9  145.9   -5.1   309.5   1905
";

    #[test]
    fn test_parse_soho_celias_text_basic() {
        let records = parse_soho_celias_text(SAMPLE_TEXT);
        assert_eq!(records.len(), 3);
        assert_eq!(records[0].year, 1996);
        assert_eq!(records[0].doy, 20);
        assert_eq!(records[0].hour, 20);
        assert!((records[0].bulk_speed - 446.0).abs() < 1e-12);
        assert!((records[0].proton_density - 10.14).abs() < 1e-12);
        assert!(records[2].bulk_speed.is_nan());
        assert!(records[2].proton_density.is_nan());
        assert!(records[2].proton_temperature.is_nan());
        assert!((records[0].r_au - 145.9 / AU_MKM).abs() < 1e-12);
        assert!((records[0].lat_deg - (-5.1)).abs() < 1e-12);
        assert!((records[0].lon_deg - 309.7).abs() < 1e-12);
    }

    #[test]
    fn test_soho_to_hourly_omni_median_downsample() {
        let records = parse_soho_celias_text(SAMPLE_TEXT);
        let omni = soho_to_hourly_omni(&records);
        assert_eq!(omni.len(), 1);
        let row = &omni[0];
        assert_eq!(row.year, 1996);
        assert_eq!(row.doy, 20);
        assert_eq!(row.hour, 20);
        assert!((row.bulk_speed - 445.5).abs() < 1e-12);
        assert!((row.proton_density - 10.15).abs() < 1e-12);
        assert!(row.b_magnitude.is_nan());
        assert!((row.lat_deg - (-5.1)).abs() < 1e-12);
    }

    #[test]
    fn test_soho_to_native_omni_preserves_valid_samples() {
        let records = parse_soho_celias_text(SAMPLE_TEXT);
        let omni = soho_to_native_omni(&records);
        assert_eq!(omni.len(), 2);
        assert_eq!(omni[0].year, 1996);
        assert_eq!(omni[0].doy, 20);
        assert_eq!(omni[0].hour, 20);
        assert!((omni[0].bulk_speed - 446.0).abs() < 1e-12);
        assert!((omni[1].proton_density - 10.16).abs() < 1e-12);
    }

    #[test]
    fn test_parse_soho_celias_bundle_file() {
        let temp = NamedTempFile::new().expect("temp");
        let file = File::create(temp.path()).expect("create");
        let gz = GzEncoder::new(file, Compression::default());
        let mut tar = Builder::new(gz);

        let mut zip_cursor = Cursor::new(Vec::<u8>::new());
        {
            let mut zip = zip::ZipWriter::new(&mut zip_cursor);
            zip.start_file(
                "1996_CELIAS_Proton_Monitor_5min.txt",
                SimpleFileOptions::default(),
            )
            .expect("start zip file");
            zip.write_all(SAMPLE_TEXT.as_bytes()).expect("zip write");
            zip.finish().expect("finish zip");
        }
        let zip_bytes = zip_cursor.into_inner();

        let mut header = tar::Header::new_gnu();
        header.set_size(zip_bytes.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        tar.append_data(
            &mut header,
            "1996_CELIAS_Proton_Monitor_5min.zip",
            Cursor::new(zip_bytes),
        )
        .expect("append tar member");
        tar.finish().expect("finish tar");
        let gz = tar.into_inner().expect("take gzip writer");
        let _file = gz.finish().expect("finish gzip");

        let records = parse_soho_celias_bundle_file(temp.path()).expect("parse bundle");
        assert_eq!(records.len(), 3);
        assert_eq!(records[0].year, 1996);
    }

    #[test]
    fn test_parse_soho_lasco_img_hdr() {
        let rows = parse_soho_lasco_img_hdr(
            "22932721.fts    2024/01/01  00:00:05   C2    25.1   1024   1024     20      1  Orange  Clear   Normal       0.0000  4179\n",
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].year, 2024);
        assert_eq!(rows[0].doy, 1);
        assert_eq!(rows[0].hour, 0);
        assert_eq!(rows[0].camera, "C2");
        assert!((rows[0].exposure_seconds - 25.1).abs() < 1.0e-12);
        assert_eq!(rows[0].width, 1024);
    }
}

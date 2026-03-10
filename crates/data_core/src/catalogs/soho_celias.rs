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

use crate::{catalogs::omni::OmniRecord, fetcher::FetchError};
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
        if !(record.bulk_speed.is_finite()
            || record.proton_density.is_finite()
            || record.proton_temperature.is_finite())
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
}

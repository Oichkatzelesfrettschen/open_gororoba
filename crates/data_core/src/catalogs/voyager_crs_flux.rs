//! Voyager CRS calibrated daily-flux parser.
//!
//! This parser is intentionally separate from `voyager_crs.rs`, which handles
//! legacy rate-style ASCII archives. The calibrated flux products expose proton
//! fluxes and corresponding uncertainties, typically through CDAWeb / NASA Open
//! Data exports whose column names follow the NotesV conventions:
//!   - `protonFluxN_CRS`
//!   - `error_PROTONFLUXN_CRS`
//!   - `SC_DISTANCE`
//!
//! The parser accepts CSV exports and dynamically discovers all proton-flux
//! channels present in the header.

use crate::parse::parse_f64_or_nan;
use csv::ReaderBuilder;

/// A single calibrated Voyager CRS daily-flux record.
#[derive(Clone, Debug)]
pub struct VoyagerCrsFluxRecord {
    pub spacecraft: u8,
    pub decimal_year: f64,
    pub distance_au: f64,
    pub proton_flux: Vec<f64>,
    pub proton_flux_error: Vec<f64>,
    pub fill_flag: bool,
}

fn decimal_year_from_date(date: &str) -> f64 {
    let trimmed = date.trim();
    if let Ok(value) = trimmed.parse::<f64>() {
        return value;
    }
    let parts: Vec<&str> = trimmed
        .split(['-', 'T', '/', ' '])
        .filter(|part| !part.is_empty())
        .collect();
    if parts.len() < 3 {
        return f64::NAN;
    }
    let year = parts[0].parse::<i32>().ok();
    let month = parts[1].parse::<u32>().ok();
    let day = parts[2].parse::<u32>().ok();
    let (year, month, day) = match (year, month, day) {
        (Some(year), Some(month), Some(day)) => (year, month, day),
        _ => return f64::NAN,
    };
    let month_lengths = [
        31_u32,
        if is_leap_year(year) { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    if month == 0 || month > 12 || day == 0 || day > month_lengths[(month - 1) as usize] {
        return f64::NAN;
    }
    let doy_prior: u32 = month_lengths[..(month - 1) as usize].iter().sum();
    let doy = doy_prior + day;
    let days_in_year = if is_leap_year(year) { 366.0 } else { 365.0 };
    year as f64 + (doy as f64 - 1.0) / days_in_year
}

const fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

fn parse_flux_value(value: &str) -> f64 {
    let parsed = parse_f64_or_nan(value);
    if !parsed.is_finite() || parsed.abs() > 1.0e30 || parsed <= -1.0e20 {
        f64::NAN
    } else {
        parsed
    }
}

/// Parse calibrated Voyager CRS flux CSV data.
///
/// Expected columns include a date-like column (`time_tag`, `date`, or
/// `decimal_year`), a distance column (`SC_DISTANCE` or `distance_au`), and one
/// or more proton-flux columns with matching error columns.
pub fn parse_voyager_crs_flux_csv(
    data: &str,
    spacecraft: u8,
) -> (Vec<VoyagerCrsFluxRecord>, usize) {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_reader(data.as_bytes());
    let headers = reader
        .headers()
        .unwrap_or_else(|err| panic!("failed to read CRS flux header: {err}"))
        .clone();

    let time_idx = headers
        .iter()
        .position(|header| matches!(header, "time_tag" | "date" | "datetime" | "decimal_year"))
        .unwrap_or_else(|| panic!("CRS flux CSV is missing a time column"));
    let distance_idx = headers
        .iter()
        .position(|header| matches!(header, "SC_DISTANCE" | "distance_au"))
        .unwrap_or_else(|| panic!("CRS flux CSV is missing a distance column"));

    let mut flux_columns = Vec::new();
    for (idx, header) in headers.iter().enumerate() {
        if let Some(channel_suffix) = header.strip_prefix("protonFlux") {
            if !header.ends_with("_CRS") {
                continue;
            }
            let channel_index = channel_suffix
                .trim_end_matches("_CRS")
                .parse::<usize>()
                .unwrap_or(0);
            if channel_index == 0 {
                continue;
            }
            let error_name = format!("error_PROTONFLUX{channel_index}_CRS");
            let error_idx = headers.iter().position(|candidate| candidate == error_name);
            flux_columns.push((channel_index, idx, error_idx));
        }
    }
    flux_columns.sort_by_key(|(channel_index, _, _)| *channel_index);
    assert!(
        !flux_columns.is_empty(),
        "CRS flux CSV must contain at least one protonFluxN_CRS column"
    );

    let mut records = Vec::new();
    let mut skipped = 0_usize;
    for row in reader.records() {
        let row = match row {
            Ok(row) => row,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };
        let decimal_year = decimal_year_from_date(row.get(time_idx).unwrap_or_default());
        if !decimal_year.is_finite() {
            skipped += 1;
            continue;
        }
        let distance_au = parse_flux_value(row.get(distance_idx).unwrap_or_default());
        let mut proton_flux = Vec::with_capacity(flux_columns.len());
        let mut proton_flux_error = Vec::with_capacity(flux_columns.len());
        let mut fill_flag = !distance_au.is_finite();
        for (_, flux_idx, error_idx) in &flux_columns {
            let flux = parse_flux_value(row.get(*flux_idx).unwrap_or_default());
            let error = error_idx
                .and_then(|idx| row.get(idx))
                .map(parse_flux_value)
                .unwrap_or(f64::NAN);
            if !flux.is_finite() || !error.is_finite() || error <= 0.0 {
                fill_flag = true;
            }
            proton_flux.push(flux);
            proton_flux_error.push(error);
        }
        records.push(VoyagerCrsFluxRecord {
            spacecraft,
            decimal_year,
            distance_au,
            proton_flux,
            proton_flux_error,
            fill_flag,
        });
    }

    (records, skipped)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_FLUX: &str = "\
time_tag,SC_DISTANCE,protonFlux1_CRS,error_PROTONFLUX1_CRS,protonFlux2_CRS,error_PROTONFLUX2_CRS\n\
2010-01-01,94.1,1.5,0.2,2.5,0.3\n\
2010-01-02,94.2,-1.0E31,-1.0E31,2.6,0.3\n";

    #[test]
    fn test_parse_flux_csv_discovers_channels() {
        let (records, skipped) = parse_voyager_crs_flux_csv(SAMPLE_FLUX, 1);
        assert_eq!(skipped, 0);
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].proton_flux.len(), 2);
        assert!((records[0].proton_flux[0] - 1.5).abs() < 1e-12);
        assert!((records[0].proton_flux_error[1] - 0.3).abs() < 1e-12);
    }

    #[test]
    fn test_fill_values_become_nan_and_flagged() {
        let (records, _) = parse_voyager_crs_flux_csv(SAMPLE_FLUX, 1);
        assert!(records[1].fill_flag);
        assert!(records[1].proton_flux[0].is_nan());
        assert!(records[1].proton_flux_error[0].is_nan());
    }
}

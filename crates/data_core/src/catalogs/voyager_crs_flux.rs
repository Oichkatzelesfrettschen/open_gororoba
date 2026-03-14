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

use crate::{
    fetcher::{DatasetProvider, FetchConfig, FetchError, download_to_string},
    parse::parse_f64_or_nan,
};
use csv::ReaderBuilder;
use serde_json::Value;
use std::{fs, path::PathBuf};

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

/// CDAWeb HAPI-backed daily proton/helium CRS flux provider.
pub struct VoyagerCrsFluxProvider {
    pub spacecraft: u8,
    pub year_start: u16,
    pub year_end: u16,
}

impl Default for VoyagerCrsFluxProvider {
    fn default() -> Self {
        Self {
            spacecraft: 1,
            year_start: 2020,
            year_end: 2020,
        }
    }
}

const HAPI_ROOT: &str = "https://cdaweb.gsfc.nasa.gov/hapi/data";
const HAPI_CATALOG_URL: &str = "https://cdaweb.gsfc.nasa.gov/hapi/catalog";
const HAPI_INFO_ROOT: &str = "https://cdaweb.gsfc.nasa.gov/hapi/info";
const VOYAGER_ARCHIVE_ROOT: &str = "https://voyager.gsfc.nasa.gov/yearly";

fn flux_dataset_id(spacecraft: u8) -> &'static str {
    match spacecraft {
        1 => "VOYAGER1_CRS_DAILY_FLUX",
        2 => "VOYAGER2_CRS_DAILY_FLUX",
        _ => "VOYAGER1_CRS_DAILY_FLUX",
    }
}

impl DatasetProvider for VoyagerCrsFluxProvider {
    fn name(&self) -> &str {
        match self.spacecraft {
            1 => "Voyager 1 CRS Daily Flux",
            2 => "Voyager 2 CRS Daily Flux",
            _ => "Voyager CRS Daily Flux",
        }
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("voyager_crs");
        fs::create_dir_all(&dir)?;
        let filename = format!(
            "voyager{}_crs_daily_flux_{}_{}.csv",
            self.spacecraft, self.year_start, self.year_end
        );
        let output = dir.join(filename);
        if config.skip_existing && output.exists() {
            return Ok(output);
        }
        if self.spacecraft == 2 {
            let body = fetch_archive_range_as_csv(self.spacecraft, self.year_start, self.year_end)?;
            fs::write(&output, body)?;
            return Ok(output);
        }
        let dataset_id = flux_dataset_id(self.spacecraft);
        if !hapi_catalog_contains(dataset_id)? {
            let body = fetch_archive_range_as_csv(self.spacecraft, self.year_start, self.year_end)?;
            fs::write(&output, body)?;
            return Ok(output);
        }
        let url = format!(
            "{HAPI_ROOT}?id={}&time.min={}-01-01T00:00:00Z&time.max={}-01-01T00:00:00Z&format=csv",
            dataset_id,
            self.year_start,
            self.year_end + 1
        );
        let body = download_to_string(&url)?;
        if is_hapi_no_data_response(&body) {
            return Err(FetchError::Validation(format!(
                "Dataset {dataset_id} returned no data for requested window {}..{}",
                self.year_start, self.year_end
            )));
        }
        let body = synthesize_hapi_header_if_missing(dataset_id, &body)?;
        fs::write(&output, body)?;
        Ok(output)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config
            .output_dir
            .join("voyager_crs")
            .join(format!(
                "voyager{}_crs_daily_flux_{}_{}.csv",
                self.spacecraft, self.year_start, self.year_end
            ))
            .exists()
    }
}

fn hapi_catalog_contains(dataset_id: &str) -> Result<bool, FetchError> {
    let catalog = download_to_string(HAPI_CATALOG_URL)?;
    Ok(catalog.contains(&format!("\"id\":\"{dataset_id}\"")))
}

fn archive_year_urls(spacecraft: u8, year: u16) -> Vec<String> {
    let root = format!("{VOYAGER_ARCHIVE_ROOT}/voyager-{spacecraft}/flux/P1D");
    if spacecraft == 2 && year == 2019 {
        return vec![
            format!("{root}/{year}P1Y_bhto.asc"),
            format!("{root}/{year}P1Y_phto.asc"),
        ];
    }
    vec![format!("{root}/{year}P1Y.asc")]
}

fn fetch_archive_range_as_csv(
    spacecraft: u8,
    year_start: u16,
    year_end: u16,
) -> Result<String, FetchError> {
    let mut all_records = Vec::new();
    let mut proton_channel_count = None;
    for year in year_start..=year_end {
        let mut year_records = Vec::new();
        for url in archive_year_urls(spacecraft, year) {
            let body = download_to_string(&url)?;
            let records = parse_voyager_archive_flux(&body, spacecraft)?;
            if proton_channel_count.is_none() {
                proton_channel_count = records.first().map(|record| record.proton_flux.len());
            }
            year_records.extend(records);
        }
        all_records.extend(year_records);
    }
    let channel_count = proton_channel_count.unwrap_or(0);
    if all_records.is_empty() || channel_count == 0 {
        return Err(FetchError::Validation(format!(
            "Voyager {spacecraft} archive daily-flux fetch produced no records for {year_start}..{year_end}"
        )));
    }
    render_flux_csv(&all_records, channel_count)
}

fn render_flux_csv(
    records: &[VoyagerCrsFluxRecord],
    proton_channel_count: usize,
) -> Result<String, FetchError> {
    let mut header = vec!["decimal_year".to_string(), "distance_au".to_string()];
    for channel in 1..=proton_channel_count {
        header.push(format!("PROTONFLUX{channel}"));
        header.push(format!("error_PROTONFLUX{channel}"));
    }
    let mut out = String::new();
    out.push_str(&header.join(","));
    out.push('\n');
    for record in records {
        let mut fields = vec![
            record.decimal_year.to_string(),
            record.distance_au.to_string(),
        ];
        for channel in 0..proton_channel_count {
            fields.push(
                record
                    .proton_flux
                    .get(channel)
                    .copied()
                    .unwrap_or(f64::NAN)
                    .to_string(),
            );
            fields.push(
                record
                    .proton_flux_error
                    .get(channel)
                    .copied()
                    .unwrap_or(f64::NAN)
                    .to_string(),
            );
        }
        out.push_str(&fields.join(","));
        out.push('\n');
    }
    Ok(out)
}

fn hapi_parameter_names(dataset_id: &str) -> Result<Vec<String>, FetchError> {
    let info_url = format!("{HAPI_INFO_ROOT}?id={dataset_id}");
    let info = download_to_string(&info_url)?;
    let value: Value = serde_json::from_str(&info).map_err(|err| {
        FetchError::Validation(format!("invalid HAPI info JSON for {dataset_id}: {err}"))
    })?;
    let parameters = value
        .get("parameters")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            FetchError::Validation(format!("HAPI info for {dataset_id} is missing parameters"))
        })?;
    let names = parameters
        .iter()
        .filter_map(|parameter| parameter.get("name").and_then(Value::as_str))
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    if names.is_empty() {
        return Err(FetchError::Validation(format!(
            "HAPI info for {dataset_id} exposed zero parameter names"
        )));
    }
    Ok(names)
}

fn is_hapi_no_data_response(body: &str) -> bool {
    let trimmed = body.trim_start();
    trimmed.starts_with('{')
        && trimmed.contains("\"code\": 1201")
        && trimmed.to_ascii_lowercase().contains("no data")
}

fn synthesize_hapi_header_if_missing(dataset_id: &str, body: &str) -> Result<String, FetchError> {
    let trimmed = body.trim_start();
    if trimmed.starts_with('{') {
        return Err(FetchError::Validation(format!(
            "dataset {dataset_id} returned JSON instead of tabular CSV"
        )));
    }
    let first_line = trimmed.lines().next().unwrap_or("");
    let starts_like_data = first_line
        .chars()
        .next()
        .map(|ch| ch.is_ascii_digit())
        .unwrap_or(false);
    if !starts_like_data {
        return Ok(body.to_string());
    }
    let header = hapi_parameter_names(dataset_id)?.join(",");
    Ok(format!("{header}\n{body}"))
}

fn parse_voyager_archive_flux(
    data: &str,
    spacecraft: u8,
) -> Result<Vec<VoyagerCrsFluxRecord>, FetchError> {
    let mut lines = data.lines().map(str::trim).filter(|line| !line.is_empty());
    let _spacecraft_label = lines.next().ok_or_else(|| {
        FetchError::Validation("Voyager archive payload missing spacecraft line".to_string())
    })?;
    let _start = lines.next().ok_or_else(|| {
        FetchError::Validation("Voyager archive payload missing start line".to_string())
    })?;
    let _stop = lines.next().ok_or_else(|| {
        FetchError::Validation("Voyager archive payload missing stop line".to_string())
    })?;
    let quantity_count = lines
        .next()
        .ok_or_else(|| {
            FetchError::Validation(
                "Voyager archive payload missing quantity-count line".to_string(),
            )
        })?
        .parse::<usize>()
        .map_err(|err| {
            FetchError::Validation(format!("invalid Voyager archive quantity count: {err}"))
        })?;
    let mut proton_positions = Vec::new();
    for quantity_index in 0..quantity_count {
        let descriptor = lines.next().ok_or_else(|| {
            FetchError::Validation(format!(
                "Voyager archive payload missing descriptor line {quantity_index}"
            ))
        })?;
        if descriptor.to_ascii_lowercase().contains("hydrogen flux") {
            proton_positions.push(quantity_index);
        }
    }
    if proton_positions.is_empty() {
        return Err(FetchError::Validation(
            "Voyager archive payload exposed zero hydrogen-flux channels".to_string(),
        ));
    }

    let mut records = Vec::new();
    for line in lines {
        let fields = line.split_whitespace().collect::<Vec<_>>();
        let expected = 1 + quantity_count * 2;
        if fields.len() < expected {
            continue;
        }
        let decimal_year = decimal_year_from_date(fields[0]);
        if !decimal_year.is_finite() {
            continue;
        }
        let mut proton_flux = Vec::with_capacity(proton_positions.len());
        let mut proton_flux_error = Vec::with_capacity(proton_positions.len());
        let mut fill_flag = false;
        for position in &proton_positions {
            let value_idx = 1 + position * 2;
            let error_idx = value_idx + 1;
            let flux = parse_flux_value(fields[value_idx]);
            let error = parse_flux_value(fields[error_idx]);
            if !flux.is_finite() || !error.is_finite() || error <= 0.0 {
                fill_flag = true;
            }
            proton_flux.push(flux);
            proton_flux_error.push(error);
        }
        records.push(VoyagerCrsFluxRecord {
            spacecraft,
            decimal_year,
            distance_au: f64::NAN,
            proton_flux,
            proton_flux_error,
            fill_flag,
        });
    }
    Ok(records)
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
        .position(|header| {
            let normalized = header.trim().to_ascii_lowercase();
            matches!(
                normalized.as_str(),
                "time" | "time_tag" | "date" | "datetime" | "decimal_year"
            )
        })
        .unwrap_or_else(|| panic!("CRS flux CSV is missing a time column"));
    let distance_idx = headers.iter().position(|header| {
        let normalized = header.trim().to_ascii_lowercase();
        matches!(normalized.as_str(), "sc_distance" | "distance_au")
    });

    let mut flux_columns = Vec::new();
    for (idx, header) in headers.iter().enumerate() {
        let normalized = header.trim().to_ascii_uppercase();
        if let Some(channel_suffix) = normalized.strip_prefix("PROTONFLUX") {
            let channel_index = channel_suffix
                .trim_end_matches("_CRS")
                .parse::<usize>()
                .unwrap_or(0);
            if channel_index == 0 {
                continue;
            }
            let error_name = format!("ERROR_PROTONFLUX{channel_index}");
            let error_name_crs = format!("ERROR_PROTONFLUX{channel_index}_CRS");
            let error_idx = headers.iter().position(|candidate| {
                let normalized_candidate = candidate.trim().to_ascii_uppercase();
                normalized_candidate == error_name || normalized_candidate == error_name_crs
            });
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
        let distance_au = distance_idx
            .and_then(|idx| row.get(idx))
            .map(parse_flux_value)
            .unwrap_or(f64::NAN);
        let mut proton_flux = Vec::with_capacity(flux_columns.len());
        let mut proton_flux_error = Vec::with_capacity(flux_columns.len());
        let mut fill_flag = false;
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

    const SAMPLE_HAPI_FLUX: &str = "\
Time,PROTONFLUX1,error_PROTONFLUX1,PROTONFLUX2,error_PROTONFLUX2\n\
2016-01-01T00:00:00.864Z,1.5,0.2,2.5,0.3\n\
2016-01-02T00:00:00.864Z,1.7,0.3,2.7,0.4\n";

    const SAMPLE_ARCHIVE_FLUX: &str = "\
voyager-2\n\
2016-01-01T00:00:00\n\
2016-12-31T00:00:00\n\
 3\n\
  1   3.000 -   4.600 MeV/n Hydrogen flux\n\
  2   4.600 -   6.200 MeV/n Hydrogen flux\n\
  3   3.000 -   4.600 MeV/n Helium flux\n\
2016-01-01T00:00:00 2.239e+03 8.163e+01 1.367e+03 6.381e+01 2.657e+02 2.817e+01\n\
2016-01-02T00:00:00 2.261e+03 1.192e+02 1.382e+03 9.324e+01 3.021e+02 4.364e+01\n";

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

    #[test]
    fn test_parse_hapi_style_flux_csv_without_distance() {
        let (records, skipped) = parse_voyager_crs_flux_csv(SAMPLE_HAPI_FLUX, 1);
        assert_eq!(skipped, 0);
        assert_eq!(records.len(), 2);
        assert!(records[0].distance_au.is_nan());
        assert!((records[0].proton_flux[1] - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_parse_voyager_archive_flux() {
        let records = parse_voyager_archive_flux(SAMPLE_ARCHIVE_FLUX, 2).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].proton_flux.len(), 2);
        assert!((records[0].proton_flux[0] - 2239.0).abs() < 1e-9);
        assert!((records[1].proton_flux_error[1] - 93.24).abs() < 1e-9);
    }
}

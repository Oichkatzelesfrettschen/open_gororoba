//! Fetch implementation for voyager_crs_flux. See voyager_crs_flux.rs for record types and parsers.

use super::voyager_crs_flux::{VoyagerCrsFluxRecord, parse_voyager_archive_flux};
use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, download_to_string};
use serde_json::Value;
use std::{fs, path::PathBuf};

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

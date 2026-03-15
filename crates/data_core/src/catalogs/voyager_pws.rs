//! Voyager Plasma Wave Subsystem low-rate spectral summaries via CDAWeb HAPI.
//!
//! The HAPI dataset exposes low-rate electric-field spectra for both Voyager
//! spacecraft. This parser reduces the native multi-band vectors into hourly
//! summaries that can participate in the heliosphere feature-cube lane.

use crate::{
    fetcher::{DatasetProvider, FetchConfig, FetchError, download_hapi_csv},
    parse::{parse_hapi_spacephysics_f64_or_nan, parse_hapi_time_to_ydh},
};
use csv::ReaderBuilder;
use std::{collections::BTreeMap, path::PathBuf};

const VOYAGER1_PWS_HAPI_DATASET: &str = "VG1_PWS_LR";
const VOYAGER2_PWS_HAPI_DATASET: &str = "VG2_PWS_LR";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoyagerPwsSpacecraft {
    V1,
    V2,
}

impl VoyagerPwsSpacecraft {
    fn dataset_id(self) -> &'static str {
        match self {
            Self::V1 => VOYAGER1_PWS_HAPI_DATASET,
            Self::V2 => VOYAGER2_PWS_HAPI_DATASET,
        }
    }

    fn slug(self) -> &'static str {
        match self {
            Self::V1 => "v1",
            Self::V2 => "v2",
        }
    }
}

#[derive(Debug, Clone)]
pub struct VoyagerPwsRecord {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub spectral_mean: f64,
    pub spectral_peak: f64,
    pub band_count: usize,
}

#[derive(Default)]
struct PwsAccumulator {
    spectral_sum: f64,
    spectral_count: usize,
    spectral_peak: f64,
    band_count: usize,
}

pub fn parse_voyager_pws_hapi_csv(content: &str) -> Vec<VoyagerPwsRecord> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(content.as_bytes());
    let mut hourly: BTreeMap<(u16, u16, u8), PwsAccumulator> = BTreeMap::new();
    for record in reader.records().flatten() {
        let Some(time) = record.get(0) else {
            continue;
        };
        let Some((year, doy, hour)) = parse_hapi_time_to_ydh(time) else {
            continue;
        };
        let values: Vec<f64> = record
            .iter()
            .skip(1)
            .map(parse_hapi_spacephysics_f64_or_nan)
            .filter(|value| value.is_finite() && *value >= 0.0)
            .collect();
        if values.is_empty() {
            continue;
        }
        let entry = hourly.entry((year, doy, hour)).or_default();
        entry.band_count = entry.band_count.max(values.len());
        for value in values {
            entry.spectral_sum += value;
            entry.spectral_count += 1;
            entry.spectral_peak = entry.spectral_peak.max(value);
        }
    }
    hourly
        .into_iter()
        .map(|((year, doy, hour), acc)| VoyagerPwsRecord {
            year,
            doy,
            hour,
            spectral_mean: if acc.spectral_count == 0 {
                f64::NAN
            } else {
                acc.spectral_sum / acc.spectral_count as f64
            },
            spectral_peak: if acc.spectral_peak > 0.0 {
                acc.spectral_peak
            } else {
                f64::NAN
            },
            band_count: acc.band_count,
        })
        .collect()
}

pub fn parse_voyager_pws_file(path: &std::path::Path) -> Result<Vec<VoyagerPwsRecord>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|err| FetchError::Validation(format!("read error: {err}")))?;
    Ok(parse_voyager_pws_hapi_csv(&content))
}

pub struct VoyagerPwsProvider {
    pub year_start: u16,
    pub year_end: u16,
}

impl Default for VoyagerPwsProvider {
    fn default() -> Self {
        Self {
            year_start: 2016,
            year_end: 2016,
        }
    }
}

impl DatasetProvider for VoyagerPwsProvider {
    fn name(&self) -> &str {
        "Voyager PWS Low Rate"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let root = config.output_dir.join("voyager").join("pws");
        std::fs::create_dir_all(&root)?;

        for spacecraft in [VoyagerPwsSpacecraft::V1, VoyagerPwsSpacecraft::V2] {
            let dir = root.join(spacecraft.slug());
            std::fs::create_dir_all(&dir)?;
            for year in self.year_start..=self.year_end {
                let output = dir.join(format!("{}_pws_lr_{}.csv", spacecraft.slug(), year));
                if config.skip_existing && output.exists() {
                    continue;
                }
                match download_hapi_csv(
                    spacecraft.dataset_id(),
                    &format!("{year}-01-01T00:00:00Z"),
                    &format!("{}-01-01T00:00:00Z", year + 1),
                    Some(&["Time", "electric_field"]),
                ) {
                    Ok(body) => {
                        std::fs::write(&output, body)?;
                        log::info!("saved {}", output.display());
                    }
                    Err(err) => {
                        log::warn!(
                            "failed to download {} {} via HAPI: {}",
                            spacecraft.dataset_id(),
                            year,
                            err
                        );
                    }
                }
            }
        }

        Ok(root)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        let root = config.output_dir.join("voyager").join("pws");
        std::fs::read_dir(&root)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok())
            .any(|entry| {
                let path = entry.path();
                if path.is_file() {
                    return path.extension().and_then(|value| value.to_str()) == Some("csv");
                }
                std::fs::read_dir(path)
                    .ok()
                    .into_iter()
                    .flatten()
                    .filter_map(|child| child.ok())
                    .any(|child| {
                        child.path().extension().and_then(|value| value.to_str()) == Some("csv")
                    })
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_voyager_pws_hapi_csv() {
        let csv = "Time,electric_field_0,electric_field_1,electric_field_2\n\
2016-01-01T10:14:53.491000000Z,0.0,4.682e-07,4.950e-07\n\
2016-01-01T10:15:09.491000000Z,2.058e-06,6.425e-07,7.970e-07\n";
        let rows = parse_voyager_pws_hapi_csv(csv);
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.year, 2016);
        assert_eq!(row.doy, 1);
        assert_eq!(row.hour, 10);
        assert_eq!(row.band_count, 3);
        assert!(row.spectral_mean.is_finite());
        assert!((row.spectral_peak - 2.058e-06).abs() < 1.0e-12);
    }
}

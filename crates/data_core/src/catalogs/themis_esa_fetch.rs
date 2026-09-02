//! Fetch/provider support for THEMIS ESA ion moments (density and average
//! temperature) from the CDAWeb HAPI dataset `TH{PROBE}_L2_ESA@0`.
//!
//! The cached daily files under `data/external/themis_esa/` hold the
//! unflagged full-mode ion moments `th{probe}_peif_density` (cm^-3) and
//! `th{probe}_peif_avgtemp` (eV) at the native few-minute cadence. A 2026-09-01
//! refetch of THA 2008 DOY 301 with exactly these parameters reproduced the
//! cached file byte for byte, while the quality-filtered `...Q` parameters
//! returned NaN on 282 of its 563 rows; the manifest therefore pins the
//! unflagged series, and a consumer that wants the quality mask fetches
//! `th{probe}_peif_data_quality` beside it.

use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, download_hapi_csv_raw};
use chrono::NaiveDate;
use std::{fs, path::PathBuf};

fn hapi_dataset_for_probe(probe: &str) -> String {
    format!("{}_L2_ESA@0", probe.to_uppercase())
}

fn moment_params_for_probe(probe: &str) -> [String; 2] {
    let p = probe.to_lowercase();
    [format!("{p}_peif_density"), format!("{p}_peif_avgtemp")]
}

/// THEMIS ESA ion-moment provider configuration.
pub struct ThemisEsaProvider {
    /// Probe identifier: "THA", "THB", "THC", "THD", "THE".
    pub probe: String,
    pub year: u16,
    pub doy_start: u16,
    pub doy_end: u16,
}

impl ThemisEsaProvider {
    /// Fetch each day's moments as the exact HAPI response bytes. The
    /// external manifest hashes source serialization, so no derived header
    /// is added; the beta-join reader takes the three unnamed columns
    /// time, density, temperature.
    pub fn fetch_raw(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("themis_esa");
        fs::create_dir_all(&dir)?;

        let dataset = hapi_dataset_for_probe(&self.probe);
        let [density, avgtemp] = moment_params_for_probe(&self.probe);
        let mut errors = Vec::new();

        for doy in self.doy_start..=self.doy_end {
            let date = NaiveDate::from_yo_opt(self.year as i32, doy as u32)
                .ok_or_else(|| FetchError::Validation(format!("invalid DOY {doy}")))?;
            let output = dir.join(day_filename(&self.probe, self.year, doy));
            if config.skip_existing && output.exists() {
                continue;
            }
            let next = date.succ_opt().ok_or_else(|| {
                FetchError::Validation(format!("no day follows {date} in the calendar"))
            })?;
            let t_min = format!("{date}T00:00:00Z");
            let t_max = format!("{next}T00:00:00Z");

            println!("Fetching {} ESA {} DOY {}...", self.probe, self.year, doy);
            match download_hapi_csv_raw(
                &dataset,
                &t_min,
                &t_max,
                Some(&[density.as_str(), avgtemp.as_str()]),
            ) {
                Ok(body) => fs::write(&output, body)?,
                Err(e) => {
                    eprintln!("  Warning: {} DOY {}: {}", self.probe, doy, e);
                    errors.push(format!("{} DOY {}: {e}", self.year, doy));
                }
            }
        }

        if errors.is_empty() {
            Ok(dir)
        } else {
            Err(FetchError::Validation(format!(
                "{} THEMIS ESA retrieval failure(s): {}",
                errors.len(),
                errors.join("; ")
            )))
        }
    }
}

/// `th{probe}_esa_{year}_{doy}.csv`, the name beta-join reads.
pub fn day_filename(probe: &str, year: u16, doy: u16) -> String {
    format!("{}_esa_{:04}_{:03}.csv", probe.to_lowercase(), year, doy)
}

impl DatasetProvider for ThemisEsaProvider {
    fn name(&self) -> &str {
        "THEMIS ESA L2 ion moments"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        self.fetch_raw(config)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("themis_esa").exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_and_parameters_follow_the_probe_letter() {
        assert_eq!(hapi_dataset_for_probe("tha"), "THA_L2_ESA@0");
        assert_eq!(
            moment_params_for_probe("THA"),
            [
                "tha_peif_density".to_string(),
                "tha_peif_avgtemp".to_string()
            ]
        );
    }

    #[test]
    fn day_filename_matches_the_beta_join_reader() {
        assert_eq!(day_filename("THA", 2008, 301), "tha_esa_2008_301.csv");
    }
}

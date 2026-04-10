//! Fetch/provider support for Helios 1 & 2 merged hourly data.

use crate::{
    catalogs::helios::HeliosSpacecraft,
    fetcher::{DatasetProvider, FetchConfig, FetchError, download_hapi_csv, download_to_file},
};
use std::path::PathBuf;

const HELIOS1_HAPI_DATASET: &str = "HELIOS1_COHO1HR_MERGED_MAG_PLASMA";
const HELIOS2_HAPI_DATASET: &str = "HELIOS2_COHO1HR_MERGED_MAG_PLASMA";
const HELIOS1_SPDF_BASE: &str = "https://spdf.gsfc.nasa.gov/pub/data/helios/helios1/merged/";
const HELIOS2_SPDF_BASE: &str = "https://spdf.gsfc.nasa.gov/pub/data/helios/helios2/merged/";

/// NASA Helios dataset provider.
pub struct HeliosProvider {
    /// Which spacecraft (H1 or H2).
    pub spacecraft: HeliosSpacecraft,
    /// Start year (inclusive).
    pub year_start: u16,
    /// End year (inclusive).
    pub year_end: u16,
}

impl Default for HeliosProvider {
    fn default() -> Self {
        Self {
            spacecraft: HeliosSpacecraft::H1,
            year_start: 1976,
            year_end: 1980,
        }
    }
}

impl DatasetProvider for HeliosProvider {
    fn name(&self) -> &str {
        match self.spacecraft {
            HeliosSpacecraft::H1 => "Helios 1 Merged Hourly",
            HeliosSpacecraft::H2 => "Helios 2 Merged Hourly",
        }
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let subdir = match self.spacecraft {
            HeliosSpacecraft::H1 => "helios1",
            HeliosSpacecraft::H2 => "helios2",
        };
        let dir = config.output_dir.join("helios").join(subdir);
        std::fs::create_dir_all(&dir)?;

        let base = match self.spacecraft {
            HeliosSpacecraft::H1 => HELIOS1_SPDF_BASE,
            HeliosSpacecraft::H2 => HELIOS2_SPDF_BASE,
        };
        let hapi_dataset = match self.spacecraft {
            HeliosSpacecraft::H1 => HELIOS1_HAPI_DATASET,
            HeliosSpacecraft::H2 => HELIOS2_HAPI_DATASET,
        };

        for year in self.year_start..=self.year_end {
            let asc_name = match self.spacecraft {
                HeliosSpacecraft::H1 => format!("he1_{year}.asc"),
                HeliosSpacecraft::H2 => format!("he2_{year}.asc"),
            };
            let asc_output = dir.join(&asc_name);
            let csv_name = format!("{subdir}_{year}_merged_hapi.csv");
            let csv_output = dir.join(&csv_name);
            if config.skip_existing && (asc_output.exists() || csv_output.exists()) {
                continue;
            }

            let asc_url = format!("{base}{asc_name}");
            match download_to_file(&asc_url, &asc_output) {
                Ok(_) => {
                    log::info!("saved {}", asc_name);
                    continue;
                }
                Err(e) => {
                    log::warn!(
                        "failed to download official Helios merged file {}: {}",
                        asc_url,
                        e
                    );
                }
            }

            match download_hapi_csv(
                hapi_dataset,
                &format!("{year}-01-01T00:00:00Z"),
                &format!("{}-01-01T00:00:00Z", year + 1),
                Some(&[
                    "Time",
                    "RotationNumber",
                    "heliocentricDistance",
                    "heliographicLatitude",
                    "heliographicLongitude",
                    "sepAngle",
                    "BX",
                    "BY",
                    "BZ",
                    "BR",
                    "BT",
                    "BN",
                    "B",
                    "flowSpeed",
                    "elevAngle",
                    "azimuthAngle",
                    "protonDensity",
                    "protonTemp",
                ]),
            ) {
                Ok(data) => {
                    std::fs::write(&csv_output, data)?;
                    log::info!("saved {}", csv_name);
                }
                Err(e) => {
                    log::warn!(
                        "failed to download Helios {} via HAPI fallback: {}",
                        year,
                        e
                    );
                }
            }
        }

        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        let subdir = match self.spacecraft {
            HeliosSpacecraft::H1 => "helios1",
            HeliosSpacecraft::H2 => "helios2",
        };
        let dir = config.output_dir.join("helios").join(subdir);
        let prefix = match self.spacecraft {
            HeliosSpacecraft::H1 => "he1_",
            HeliosSpacecraft::H2 => "he2_",
        };
        std::fs::read_dir(&dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok())
            .any(|entry| {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                (name.starts_with(prefix) && name.ends_with(".asc"))
                    || (name.starts_with(subdir) && name.ends_with(".csv"))
            })
    }
}

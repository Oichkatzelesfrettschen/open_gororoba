//! Fetch/provider support for Voyager 1 and 2 merged hourly and MAG 48-sec data.

use crate::{
    catalogs::voyager::{VOYAGER2_BARTOL_BASE, VoyagerSpacecraft},
    fetcher::{DatasetProvider, FetchConfig, FetchError, download_to_string},
};
use std::path::PathBuf;

const VOYAGER1_BASE: &str = "https://spdf.gsfc.nasa.gov/pub/data/voyager/voyager1/merged/";
const VOYAGER2_BASE: &str = "https://spdf.gsfc.nasa.gov/pub/data/voyager/voyager2/merged/";
const VOYAGER1_MAG48_BASE: &str =
    "https://spdf.gsfc.nasa.gov/pub/data/voyager/voyager1/magnetic_fields/ip_48s_ascii/";
const VOYAGER2_MAG48_BASE: &str =
    "https://spdf.gsfc.nasa.gov/pub/data/voyager/voyager2/magnetic_fields/ip_48s_ascii/";

/// NASA SPDF Voyager dataset provider.
pub struct VoyagerProvider {
    /// Which spacecraft (V1 or V2).
    pub spacecraft: VoyagerSpacecraft,
    /// Start year (inclusive).
    pub year_start: u16,
    /// End year (inclusive).
    pub year_end: u16,
}

impl Default for VoyagerProvider {
    fn default() -> Self {
        Self {
            spacecraft: VoyagerSpacecraft::V1,
            year_start: 1977,
            year_end: 2024,
        }
    }
}

impl DatasetProvider for VoyagerProvider {
    fn name(&self) -> &str {
        match self.spacecraft {
            VoyagerSpacecraft::V1 => "Voyager 1 Merged Hourly",
            VoyagerSpacecraft::V2 => "Voyager 2 Merged Hourly",
        }
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let subdir = match self.spacecraft {
            VoyagerSpacecraft::V1 => "voyager1",
            VoyagerSpacecraft::V2 => "voyager2",
        };
        let dir = config.output_dir.join(subdir);
        std::fs::create_dir_all(&dir)?;

        let base = match self.spacecraft {
            VoyagerSpacecraft::V1 => VOYAGER1_BASE,
            VoyagerSpacecraft::V2 => VOYAGER2_BASE,
        };

        for year in self.year_start..=self.year_end {
            let fname = format!(
                "vy{}_{}.asc",
                match self.spacecraft {
                    VoyagerSpacecraft::V1 => "1",
                    VoyagerSpacecraft::V2 => "2",
                },
                year,
            );
            let output = dir.join(&fname);
            if config.skip_existing && output.exists() {
                continue;
            }
            let url = format!("{}{}", base, fname);
            match download_to_string(&url) {
                Ok(data) => {
                    std::fs::write(&output, data)?;
                    log::info!("saved {}", fname);
                }
                Err(e) => {
                    log::warn!("SPDF failed for Voyager {}: {}", year, e);
                    // Bartol fallback: Voyager 2 only, 1977-1997
                    if matches!(self.spacecraft, VoyagerSpacecraft::V2)
                        && (1977..=1997).contains(&year)
                    {
                        let bartol_fname = format!("vy2_{}.dat", year % 100);
                        let bartol_url = format!("{}{}", VOYAGER2_BARTOL_BASE, bartol_fname);
                        match download_to_string(&bartol_url) {
                            Ok(data) => {
                                let bartol_dir = dir.join("bartol");
                                std::fs::create_dir_all(&bartol_dir)?;
                                let bartol_out = bartol_dir.join(&bartol_fname);
                                std::fs::write(&bartol_out, &data)?;
                                log::info!("Bartol fallback saved {}", bartol_fname);
                            }
                            Err(e2) => {
                                log::warn!("Bartol fallback also failed for {}: {}", year, e2);
                            }
                        }
                    }
                }
            }
        }

        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        let subdir = match self.spacecraft {
            VoyagerSpacecraft::V1 => "voyager1",
            VoyagerSpacecraft::V2 => "voyager2",
        };
        config.output_dir.join(subdir).exists()
    }
}

/// Provider for Voyager 48-second MAG high-resolution data.
pub struct VoyagerMag48Provider {
    pub spacecraft: VoyagerSpacecraft,
    pub year_start: u16,
    pub year_end: u16,
}

impl DatasetProvider for VoyagerMag48Provider {
    fn name(&self) -> &str {
        match self.spacecraft {
            VoyagerSpacecraft::V1 => "Voyager 1 MAG 48-sec",
            VoyagerSpacecraft::V2 => "Voyager 2 MAG 48-sec",
        }
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let subdir = match self.spacecraft {
            VoyagerSpacecraft::V1 => "voyager1_mag48",
            VoyagerSpacecraft::V2 => "voyager2_mag48",
        };
        let dir = config.output_dir.join(subdir);
        std::fs::create_dir_all(&dir)?;

        let base = match self.spacecraft {
            VoyagerSpacecraft::V1 => VOYAGER1_MAG48_BASE,
            VoyagerSpacecraft::V2 => VOYAGER2_MAG48_BASE,
        };
        let prefix = match self.spacecraft {
            VoyagerSpacecraft::V1 => "vy1",
            VoyagerSpacecraft::V2 => "vy2",
        };

        for year in self.year_start..=self.year_end {
            let fname = format!("{prefix}_{year}.asc");
            let output = dir.join(&fname);
            if config.skip_existing && output.exists() {
                continue;
            }
            let url = format!("{base}{fname}");
            match download_to_string(&url) {
                Ok(data) => {
                    std::fs::write(&output, data)?;
                    log::info!("saved {}", fname);
                }
                Err(e) => {
                    log::warn!("SPDF 48-sec MAG failed for {prefix} {year}: {e}");
                }
            }
        }
        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        let subdir = match self.spacecraft {
            VoyagerSpacecraft::V1 => "voyager1_mag48",
            VoyagerSpacecraft::V2 => "voyager2_mag48",
        };
        config.output_dir.join(subdir).exists()
    }
}

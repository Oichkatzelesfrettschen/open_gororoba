//! Fetch/provider support for Pioneer 10 and 11 merged hourly data.

use crate::{
    catalogs::pioneer::PioneerSpacecraft,
    fetcher::{DatasetProvider, FetchConfig, FetchError, download_to_string},
};
use std::path::PathBuf;

const PIONEER10_BASE: &str = "https://spdf.gsfc.nasa.gov/pub/data/pioneer/pioneer10/merged/";
const PIONEER11_BASE: &str = "https://spdf.gsfc.nasa.gov/pub/data/pioneer/pioneer11/merged/";

/// NASA SPDF Pioneer dataset provider.
pub struct PioneerProvider {
    pub spacecraft: PioneerSpacecraft,
    pub year_start: u16,
    pub year_end: u16,
}

impl Default for PioneerProvider {
    fn default() -> Self {
        Self {
            spacecraft: PioneerSpacecraft::P10,
            year_start: 1972,
            year_end: 1995,
        }
    }
}

impl DatasetProvider for PioneerProvider {
    fn name(&self) -> &str {
        match self.spacecraft {
            PioneerSpacecraft::P10 => "Pioneer 10 Merged Hourly",
            PioneerSpacecraft::P11 => "Pioneer 11 Merged Hourly",
        }
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let subdir = match self.spacecraft {
            PioneerSpacecraft::P10 => "pioneer10",
            PioneerSpacecraft::P11 => "pioneer11",
        };
        let dir = config.output_dir.join("pioneer").join(subdir);
        std::fs::create_dir_all(&dir)?;

        let base = match self.spacecraft {
            PioneerSpacecraft::P10 => PIONEER10_BASE,
            PioneerSpacecraft::P11 => PIONEER11_BASE,
        };
        let prefix = match self.spacecraft {
            PioneerSpacecraft::P10 => "p10",
            PioneerSpacecraft::P11 => "p11",
        };

        for year in self.year_start..=self.year_end {
            let fname = format!("{prefix}_{:02}.asc", year % 100);
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
                    log::warn!(
                        "Pioneer {} {}: {}",
                        match self.spacecraft {
                            PioneerSpacecraft::P10 => "10",
                            PioneerSpacecraft::P11 => "11",
                        },
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
            PioneerSpacecraft::P10 => "pioneer10",
            PioneerSpacecraft::P11 => "pioneer11",
        };
        config.output_dir.join("pioneer").join(subdir).exists()
    }
}

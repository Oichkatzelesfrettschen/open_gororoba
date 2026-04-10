//! Fetch implementation for swarm_mag. See swarm_mag.rs for record types and parsers.

use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, download_amda_hapi_csv};
use std::{fs, path::PathBuf};

/// AMDA dataset ID for Swarm A vector magnetometer (NEC frame).
const SWARM_A_AMDA_MAG: &str = "swarma-mag-all";

/// Swarm MAG provider via AMDA.
pub struct SwarmMagProvider {
    pub year: u16,
    pub month_start: u8,
    pub month_end: u8,
}

impl DatasetProvider for SwarmMagProvider {
    fn name(&self) -> &str {
        "Swarm A MAG (AMDA)"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("swarm");
        fs::create_dir_all(&dir)?;

        for month in self.month_start..=self.month_end {
            let t_min = format!("{:04}-{:02}-01T00:00:00Z", self.year, month);
            let (ey, em) = if month == 12 {
                (self.year + 1, 1)
            } else {
                (self.year, month + 1)
            };
            let t_max = format!("{ey:04}-{em:02}-01T00:00:00Z");

            let fname = format!("swarma_mag_{:04}_{:02}.csv", self.year, month);
            let output = dir.join(&fname);

            if config.skip_existing && output.exists() {
                continue;
            }

            println!("Fetching Swarm A MAG {:04}-{:02}...", self.year, month);

            match download_amda_hapi_csv(SWARM_A_AMDA_MAG, &t_min, &t_max, None) {
                Ok(body) => {
                    fs::write(&output, body)?;
                }
                Err(e) => {
                    eprintln!("  Warning: Swarm {:04}-{:02}: {}", self.year, month, e);
                }
            }
        }

        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("swarm").exists()
    }
}

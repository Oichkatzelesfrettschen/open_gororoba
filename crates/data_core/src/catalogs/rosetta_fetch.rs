//! Fetch implementation for rosetta. See rosetta.rs for record types and parsers.

use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, download_amda_hapi_csv};
use std::{fs, path::PathBuf};

/// AMDA dataset ID for Rosetta RPC-MAG outboard sensor, 60s resampled.
///
/// Outboard sensor (OB) is the primary science sensor (less spacecraft
/// interference). 60s resampled cadence matches minute-level analysis.
/// Coordinate frame: CSEQ (Comet-centered Solar EQuatorial).
/// Columns: Time, Bx, By, Bz (nT).
const ROSETTA_AMDA_MAG: &str = "ros-magob-rsmp";

/// Rosetta RPC-MAG dataset provider via AMDA HAPI.
pub struct RosettaMagProvider {
    pub year_start: u16,
    pub year_end: u16,
    /// Restrict to specific month range (1-12). None = all months.
    pub month_range: Option<(u8, u8)>,
}

impl Default for RosettaMagProvider {
    fn default() -> Self {
        // Rosetta at 67P: Aug 2014 - Sep 2016 (comet escort phase)
        Self {
            year_start: 2014,
            year_end: 2016,
            month_range: None,
        }
    }
}

impl DatasetProvider for RosettaMagProvider {
    fn name(&self) -> &str {
        "Rosetta RPC-MAG (AMDA)"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("rosetta");
        fs::create_dir_all(&dir)?;

        for year in self.year_start..=self.year_end {
            let (start_month, end_month) = self.month_range.unwrap_or((1, 12));

            // Respect mission timeline
            let start_month = if year == 2014 {
                start_month.max(8)
            } else {
                start_month
            };
            let end_month = if year == 2016 {
                end_month.min(9)
            } else {
                end_month
            };

            for month in start_month..=end_month {
                let t_min = format!("{year:04}-{month:02}-01T00:00:00Z");
                let (end_year, end_month_next) = if month == 12 {
                    (year + 1, 1)
                } else {
                    (year, month + 1)
                };
                let t_max = format!("{end_year:04}-{end_month_next:02}-01T00:00:00Z");

                let fname = format!("rosetta_rpcmag_{year:04}_{month:02}.csv");
                let output = dir.join(&fname);

                if config.skip_existing && output.exists() {
                    continue;
                }

                println!(
                    "Fetching Rosetta RPC-MAG {} month {} ({} to {})...",
                    year, month, t_min, t_max
                );

                match download_amda_hapi_csv(ROSETTA_AMDA_MAG, &t_min, &t_max, None) {
                    Ok(body) => {
                        fs::write(&output, body)?;
                    }
                    Err(e) => {
                        eprintln!("  Warning: Rosetta {year}-{month:02}: {e}");
                    }
                }
            }
        }

        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("rosetta").exists()
    }
}

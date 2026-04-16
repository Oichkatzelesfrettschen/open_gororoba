//! Fetch implementation for mms. See mms.rs for record types and parsers.

use crate::fetcher::{
    DatasetProvider, FetchConfig, FetchError, download_hapi_csv, download_to_string,
};
use chrono::NaiveDate;
use std::{fs, path::PathBuf};

const MMS_FGM_HAPI_DATASET: &str = "MMS1_FGM_SRVY_L2@0";

/// MMS FGM dataset provider.
pub struct MmsFgmProvider {
    pub year_start: u16,
    pub year_end: u16,
    pub doy_range: Option<(u16, u16)>,
}

impl Default for MmsFgmProvider {
    fn default() -> Self {
        Self {
            year_start: 2015,
            year_end: 2026,
            doy_range: None,
        }
    }
}

impl DatasetProvider for MmsFgmProvider {
    fn name(&self) -> &str {
        "MMS1 FGM Survey L2"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("mms");
        fs::create_dir_all(&dir)?;

        for year in self.year_start..=self.year_end {
            // Respect the startDate of the dataset (2015-09-01)
            let (start_month, start_day) = if year == 2015 { (9, 1) } else { (1, 1) };

            if let Some((start_doy, end_doy)) = self.doy_range {
                // Fine-window test: use exact DOY range without chunking
                let start_date =
                    NaiveDate::from_yo_opt(year as i32, start_doy as u32).ok_or_else(|| {
                        FetchError::Validation(format!("invalid start doy {start_doy}"))
                    })?;
                let end_date = NaiveDate::from_yo_opt(year as i32, end_doy as u32)
                    .ok_or_else(|| FetchError::Validation(format!("invalid end doy {end_doy}")))?;

                let t_min = format!("{}T00:00:00Z", start_date.format("%Y-%m-%d"));
                let t_max = format!("{}T23:59:59Z", end_date.format("%Y-%m-%d"));
                let fname = format!("mms1_fgm_srvy_l2_{year}_{start_doy}_{end_doy}.csv");
                let output = dir.join(&fname);

                if config.skip_existing && output.exists() {
                    continue;
                }

                println!(
                    "Fetching MMS1 FGM Survey L2 for {} DOY {}-{} ({} to {})...",
                    year, start_doy, end_doy, t_min, t_max
                );

                let body = download_hapi_csv(
                    MMS_FGM_HAPI_DATASET,
                    &t_min,
                    &t_max,
                    Some(&["Time", "mms1_fgm_b_gse_srvy_l2_clean"]),
                )?;
                fs::write(&output, body)?;
            } else {
                // Monthly chunking to avoid 400 errors on large requests
                for month in start_month..=12 {
                    let t_min = format!("{:04}-{:02}-{:02}T00:00:00Z", year, month, start_day);

                    // End of month or end of year
                    let (end_year, end_month) = if month == 12 {
                        (year + 1, 1)
                    } else {
                        (year, month + 1)
                    };
                    let t_max = format!("{:04}-{:02}-01T00:00:00Z", end_year, end_month);

                    // Stop if we exceed year_end
                    if year == self.year_end && month == 12 && self.year_end < 2026 {
                        // This is simple logic, could be more precise with days but months are safe chunks
                    }

                    let fname = format!("mms1_fgm_srvy_l2_{:04}_{:02}.csv", year, month);
                    let output = dir.join(&fname);

                    if config.skip_existing && output.exists() {
                        continue;
                    }

                    println!(
                        "Fetching MMS1 FGM Survey L2 for {} month {} ({} to {})...",
                        year, month, t_min, t_max
                    );

                    let body = download_hapi_csv(
                        MMS_FGM_HAPI_DATASET,
                        &t_min,
                        &t_max,
                        Some(&["Time", "mms1_fgm_b_gse_srvy_l2_clean"]),
                    )?;
                    fs::write(&output, body)?;
                }
            }
        }

        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("mms").exists()
    }
}

/// MMS SITL / GLS event catalog provider.
///
/// Downloads the scientist-in-the-loop (SITL) and ground-loop-segment (GLS)
/// burst-mode selection catalog from the MMS Science Data Center public REST
/// API.
///
/// Endpoint:
/// `https://lasp.colorado.edu/mms/sdc/public/files/api/v1/sitl/gls/csv
///  ?start_date=YYYY-MM-DDTHH:MM:SS&stop_date=YYYY-MM-DDTHH:MM:SS`
///
/// WHY: SITL selections are expert-curated boundary annotations used to
/// harden MMS ground truth beyond |B|-gradient pseudo-labels.  Caching them
/// locally avoids repeated SDC queries during ablation sweeps.
pub struct MmsSitlProvider {
    pub start_date: NaiveDate,
    pub end_date: NaiveDate,
}

const MMS_SDC_SITL_BASE: &str =
    "https://lasp.colorado.edu/mms/sdc/public/files/api/v1/sitl/gls/csv";

impl DatasetProvider for MmsSitlProvider {
    fn name(&self) -> &str {
        "MMS SITL/GLS Event Catalog"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("mms");
        fs::create_dir_all(&dir)?;

        let fname = format!(
            "mms_sitl_{}_{}.csv",
            self.start_date.format("%Y%m%d"),
            self.end_date.format("%Y%m%d")
        );
        let output = dir.join(&fname);

        if config.skip_existing && output.exists() {
            return Ok(output);
        }

        let url = format!(
            "{}?start_date={}T00:00:00&stop_date={}T23:59:59",
            MMS_SDC_SITL_BASE,
            self.start_date.format("%Y-%m-%d"),
            self.end_date.format("%Y-%m-%d"),
        );

        println!(
            "Fetching MMS SITL catalog {} to {} from SDC...",
            self.start_date, self.end_date
        );

        let body = download_to_string(&url)?;
        fs::write(&output, &body)?;
        Ok(output)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        let dir = config.output_dir.join("mms");
        let fname = format!(
            "mms_sitl_{}_{}.csv",
            self.start_date.format("%Y%m%d"),
            self.end_date.format("%Y%m%d")
        );
        dir.join(fname).exists()
    }
}

//! Fetch/provider support for STEREO-A PLASTIC and IMPACT/MAG data.

use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, download_hapi_csv};
use std::path::PathBuf;

/// STEREO-A PLASTIC dataset provider (1-hour plasma).
pub struct StereoPlasticProvider {
    pub year_start: u16,
    pub year_end: u16,
}

impl Default for StereoPlasticProvider {
    fn default() -> Self {
        Self {
            year_start: 2024,
            year_end: 2024,
        }
    }
}

impl DatasetProvider for StereoPlasticProvider {
    fn name(&self) -> &str {
        "STEREO-A PLASTIC 1-hour Plasma"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("stereo_plastic");
        std::fs::create_dir_all(&dir)?;

        for year in self.year_start..=self.year_end {
            let fname = format!("sta_l2_pla_1dmax_1hr_{year}.csv");
            let output = dir.join(&fname);
            if config.skip_existing && output.exists() {
                continue;
            }
            let body = download_hapi_csv(
                "STA_L2_PLA_1DMAX_1HR",
                &format!("{year}-01-01T00:00:00Z"),
                &format!("{}-01-01T00:00:00Z", year + 1),
                Some(&[
                    "Time",
                    "proton_number_density_1hr",
                    "proton_bulk_speed_1hr",
                    "proton_temperature_1hr",
                    "proton_Vr_HERTN_1hr",
                    "proton_Vt_HERTN_1hr",
                    "proton_Vn_HERTN_1hr",
                ]),
            )?;
            std::fs::write(&output, body)?;
            log::info!("Saved {}", fname);
        }

        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("stereo_plastic").exists()
    }
}

/// STEREO-A IMPACT/MAG dataset provider (MAGPLASMA via CDAWeb).
pub struct StereoMagProvider {
    pub year_start: u16,
    pub year_end: u16,
}

impl Default for StereoMagProvider {
    fn default() -> Self {
        Self {
            year_start: 2024,
            year_end: 2024,
        }
    }
}

impl DatasetProvider for StereoMagProvider {
    fn name(&self) -> &str {
        "STEREO-A IMPACT/MAG MAGPLASMA"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("stereo_impact");
        std::fs::create_dir_all(&dir)?;
        for year in self.year_start..=self.year_end {
            let fname = format!("sta_coho1hr_merged_mag_plasma_{year}.csv");
            let output = dir.join(&fname);
            if config.skip_existing && output.exists() {
                continue;
            }
            let body = download_hapi_csv(
                "STA_COHO1HR_MERGED_MAG_PLASMA",
                &format!("{year}-01-01T00:00:00Z"),
                &format!("{}-01-01T00:00:00Z", year + 1),
                None,
            )?;
            std::fs::write(&output, body)?;
            log::info!("Saved {}", fname);
        }
        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("stereo_impact").exists()
    }
}

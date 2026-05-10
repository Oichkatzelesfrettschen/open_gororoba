//! Fetch/provider support for Juno cruise data.

use crate::fetcher::{
    DatasetProvider, FetchConfig, FetchError, download_amda_hapi_csv, download_hapi_csv,
};
use std::path::PathBuf;

use crate::catalogs::juno::{
    merge_juno_amda, parse_juno_amda_mag, parse_juno_amda_orb, parse_juno_amda_plasma,
};

const JUNO_POSITION_HAPI_DATASET: &str = "JUNO_HELIO1HR_POSITION";

/// AMDA dataset ID for Juno JADE L5 proton moments (cruise phase).
const JUNO_AMDA_PLASMA: &str = "juno-jadel5-protmom";

/// AMDA dataset ID for Juno FGM cruise 60-min averages in RTN coordinates.
const JUNO_AMDA_MAG: &str = "juno-fgm-cruise60";

/// AMDA dataset ID for Juno cruise ephemeris.
const JUNO_AMDA_ORB: &str = "juno-cruise-all";

/// NASA SPDF Juno cruise dataset provider.
pub struct JunoCruiseProvider {
    pub year_start: u16,
    pub year_end: u16,
}

impl Default for JunoCruiseProvider {
    fn default() -> Self {
        Self {
            year_start: 2011,
            year_end: 2016,
        }
    }
}

impl DatasetProvider for JunoCruiseProvider {
    fn name(&self) -> &str {
        "Juno Cruise Merged Hourly"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("juno");
        std::fs::create_dir_all(&dir)?;

        for year in self.year_start..=self.year_end {
            let fname = format!("juno_helio1hr_position_{year}.csv");
            let output = dir.join(&fname);
            if config.skip_existing && output.exists() {
                continue;
            }
            match download_hapi_csv(
                JUNO_POSITION_HAPI_DATASET,
                &format!("{year}-01-01T00:00:00Z"),
                &format!("{}-01-01T00:00:00Z", year + 1),
                Some(&["Time", "RAD_AU", "HG_LAT", "HG_LON"]),
            ) {
                Ok(data) => {
                    std::fs::write(&output, data)?;
                    log::info!("saved {}", fname);
                }
                Err(e) => {
                    log::warn!("failed to download Juno {}: {}", year, e);
                }
            }
        }
        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("juno").exists()
    }
}

/// Juno AMDA provider -- fetches three AMDA lanes and merges them.
///
/// Falls back to this when the CDAWeb-only `JunoCruiseProvider` is blocked.
/// Note: `JunoCruiseProvider` only provides orbital data; this provider
/// delivers the full plasma+MAG picture.
pub struct JunoAmdaProvider {
    pub year_start: u16,
    pub year_end: u16,
}

impl Default for JunoAmdaProvider {
    fn default() -> Self {
        Self {
            year_start: 2011,
            year_end: 2016,
        }
    }
}

impl DatasetProvider for JunoAmdaProvider {
    fn name(&self) -> &str {
        "Juno Cruise AMDA (plasma+MAG+orbit)"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("juno").join("amda");
        std::fs::create_dir_all(&dir)?;

        for year in self.year_start..=self.year_end {
            let t_min = format!("{year}-01-01T00:00:00Z");
            let t_max = format!("{}-01-01T00:00:00Z", year + 1);
            let out_path = dir.join(format!("juno_amda_merged_{year}.csv"));
            if config.skip_existing && out_path.exists() {
                continue;
            }

            let plasma_csv = match download_amda_hapi_csv(JUNO_AMDA_PLASMA, &t_min, &t_max, None) {
                Ok(csv) => csv,
                Err(e) => {
                    log::warn!("AMDA Juno plasma {year}: {e}");
                    continue;
                }
            };
            let mag_csv = match download_amda_hapi_csv(JUNO_AMDA_MAG, &t_min, &t_max, None) {
                Ok(csv) => csv,
                Err(e) => {
                    log::warn!("AMDA Juno MAG {year}: {e}");
                    continue;
                }
            };
            let orb_csv = match download_amda_hapi_csv(JUNO_AMDA_ORB, &t_min, &t_max, None) {
                Ok(csv) => csv,
                Err(e) => {
                    log::warn!("AMDA Juno orbit {year}: {e}");
                    continue;
                }
            };

            let plasma = parse_juno_amda_plasma(&plasma_csv);
            let mag = parse_juno_amda_mag(&mag_csv);
            let orb = parse_juno_amda_orb(&orb_csv);
            let merged = merge_juno_amda(&plasma, &mag, &orb);

            let mut csv_buf = String::from(
                "year,doy,hour,distance_au,lat_deg,lon_deg,br,bt,bn,b_mag,density,speed,temperature\n",
            );
            for r in &merged {
                csv_buf.push_str(&format!(
                    "{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                    r.year,
                    r.doy,
                    r.hour,
                    r.distance_au,
                    r.lat_deg,
                    r.lon_deg,
                    r.br,
                    r.bt,
                    r.bn,
                    r.b_magnitude,
                    r.proton_density,
                    r.bulk_speed,
                    r.proton_temperature,
                ));
            }
            std::fs::write(&out_path, csv_buf)?;
            log::info!(
                "AMDA Juno {year}: merged {} hourly records -> {}",
                merged.len(),
                out_path.display()
            );
        }

        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        let dir = config.output_dir.join("juno").join("amda");
        std::fs::read_dir(&dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok())
            .any(|entry| {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                name.starts_with("juno_amda_merged_") && name.ends_with(".csv")
            })
    }
}

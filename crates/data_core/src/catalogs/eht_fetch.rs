//! Fetch implementation for eht. See eht.rs for archive utilities and record types.

use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, download_with_fallbacks};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// URL constants (verified 2026-02-07, all HTTP 200)
// ---------------------------------------------------------------------------

// M87 2018 (2024-D01-01, branch: main)
const EHT_M87_2018_CSV: &[&str] =
    &["https://github.com/eventhorizontelescope/2024-D01-01/raw/main/EHTC_M872018_csv.tgz"];
const EHT_M87_2018_UVFITS: &[&str] =
    &["https://github.com/eventhorizontelescope/2024-D01-01/raw/main/EHTC_M872018_uvfits.tgz"];
const EHT_M87_2018_TXT: &[&str] =
    &["https://github.com/eventhorizontelescope/2024-D01-01/raw/main/EHTC_M872018_txt.tgz"];

// M87 2017 -- first image (2019-D01-01, branch: master)
const EHT_M87_2017_CSV: &[&str] = &[
    "https://github.com/eventhorizontelescope/2019-D01-01/raw/master/EHTC_FirstM87Results_Apr2019_csv.tgz",
];
const EHT_M87_2017_UVFITS: &[&str] = &[
    "https://github.com/eventhorizontelescope/2019-D01-01/raw/master/EHTC_FirstM87Results_Apr2019_uvfits.tgz",
];
const EHT_M87_2017_TXT: &[&str] = &[
    "https://github.com/eventhorizontelescope/2019-D01-01/raw/master/EHTC_FirstM87Results_Apr2019_txt.tgz",
];

// Sgr A* 2017 (2022-D02-01, branch: main)
const EHT_SGRA_CSV: &[&str] = &[
    "https://github.com/eventhorizontelescope/2022-D02-01/raw/main/EHTC_FirstSgrAResults_May2022_csv.tgz",
];
const EHT_SGRA_UVFITS: &[&str] = &[
    "https://github.com/eventhorizontelescope/2022-D02-01/raw/main/EHTC_FirstSgrAResults_May2022_uvfits.tgz",
];
const EHT_SGRA_TXT: &[&str] = &[
    "https://github.com/eventhorizontelescope/2022-D02-01/raw/main/EHTC_FirstSgrAResults_May2022_txt.tgz",
];

// 3C 279 2017 (2020-D01-01, branch: master)
const EHT_3C279_CSV: &[&str] = &[
    "https://github.com/eventhorizontelescope/2020-D01-01/raw/master/EHTC_First3C279Results_May2020_csv.tgz",
];
const EHT_3C279_UVFITS: &[&str] = &[
    "https://github.com/eventhorizontelescope/2020-D01-01/raw/master/EHTC_First3C279Results_May2020_uvfits.tgz",
];
const EHT_3C279_TXT: &[&str] = &[
    "https://github.com/eventhorizontelescope/2020-D01-01/raw/master/EHTC_First3C279Results_May2020_txt.tgz",
];

// Centaurus A 2017 (2021-D03-01, branch: main)
const EHT_CENA_CSV: &[&str] = &[
    "https://github.com/eventhorizontelescope/2021-D03-01/raw/main/EHTC_CenA_data_July2021_csv.tgz",
];
const EHT_CENA_UVFITS: &[&str] = &[
    "https://github.com/eventhorizontelescope/2021-D03-01/raw/main/EHTC_CenA_data_July2021_uvfits.tgz",
];
const EHT_CENA_TXT: &[&str] = &[
    "https://github.com/eventhorizontelescope/2021-D03-01/raw/main/EHTC_CenA_data_July2021_txt.tgz",
];

// M87 2011-2013 legacy monitoring (2020-D03-01, branch: master)
const EHT_M87_LEGACY: &[&str] = &[
    "https://github.com/eventhorizontelescope/2020-D03-01/raw/master/EHTC_MonitoringM87_Sep2020.tgz",
];

// ---------------------------------------------------------------------------
// Helper: multi-format download
// ---------------------------------------------------------------------------

/// Download all format bundles (CSV, UVFITS, TXT) for an EHT source.
/// Returns the path to the UVFITS bundle (primary data product).
fn fetch_eht_multi(
    name: &str,
    dir: &Path,
    prefix: &str,
    csv_urls: &[&str],
    uvfits_urls: &[&str],
    txt_urls: &[&str],
    skip_existing: bool,
) -> Result<PathBuf, FetchError> {
    let csv_out = dir.join(format!("{prefix}_csv.tgz"));
    let uvfits_out = dir.join(format!("{prefix}_uvfits.tgz"));
    let txt_out = dir.join(format!("{prefix}_txt.tgz"));

    // UVFITS is the primary product; download it first
    download_with_fallbacks(name, uvfits_urls, &uvfits_out, skip_existing)?;

    // CSV and TXT are supplementary; log but do not fail on error
    if let Err(e) = download_with_fallbacks(name, csv_urls, &csv_out, skip_existing) {
        log::warn!("CSV bundle for {name} failed: {e}");
    }
    if let Err(e) = download_with_fallbacks(name, txt_urls, &txt_out, skip_existing) {
        log::warn!("TXT bundle for {name} failed: {e}");
    }

    Ok(uvfits_out)
}

// ---------------------------------------------------------------------------
// Providers
// ---------------------------------------------------------------------------

/// EHT M87* 2017 -- first black hole image (Papers I-VI).
/// Bands: low + high. Pipeline: EHT-HOPS. Stokes I only.
pub struct EhtM87_2017Provider;

impl DatasetProvider for EhtM87_2017Provider {
    fn name(&self) -> &str {
        "EHT M87 2017 (First Image)"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        fetch_eht_multi(
            self.name(),
            &config.output_dir,
            "eht_m87_2017",
            EHT_M87_2017_CSV,
            EHT_M87_2017_UVFITS,
            EHT_M87_2017_TXT,
            config.skip_existing,
        )
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("eht_m87_2017_uvfits.tgz").exists()
    }
}

/// EHT M87* 2018 multi-epoch follow-up.
/// Bands: b1-b4. Pipelines: EHT-HOPS + CASA rPICARD. Stokes I only.
pub struct EhtM87Provider;

impl DatasetProvider for EhtM87Provider {
    fn name(&self) -> &str {
        "EHT M87 2018 Data Bundle"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        fetch_eht_multi(
            self.name(),
            &config.output_dir,
            "eht_m87_2018",
            EHT_M87_2018_CSV,
            EHT_M87_2018_UVFITS,
            EHT_M87_2018_TXT,
            config.skip_existing,
        )
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("eht_m87_2018_uvfits.tgz").exists()
    }
}

/// EHT Sgr A* 2017 -- first Milky Way black hole image (Papers I-VI).
/// Bands: low + high. Pipelines: EHT-HOPS + CASA rPICARD. Stokes I only.
/// Includes standard, lightcurve-normalized, and 100-min optimal subarray variants.
pub struct EhtSgrAProvider;

impl DatasetProvider for EhtSgrAProvider {
    fn name(&self) -> &str {
        "EHT Sgr A* 2017 Data Bundle"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        fetch_eht_multi(
            self.name(),
            &config.output_dir,
            "eht_sgra_2017",
            EHT_SGRA_CSV,
            EHT_SGRA_UVFITS,
            EHT_SGRA_TXT,
            config.skip_existing,
        )
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("eht_sgra_2017_uvfits.tgz").exists()
    }
}

/// EHT 3C 279 2017 -- quasar jet morphology.
/// Bands: low + high. Pipeline: EHT-HOPS.
pub struct Eht3c279Provider;

impl DatasetProvider for Eht3c279Provider {
    fn name(&self) -> &str {
        "EHT 3C279 2017 Data Bundle"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        fetch_eht_multi(
            self.name(),
            &config.output_dir,
            "eht_3c279_2017",
            EHT_3C279_CSV,
            EHT_3C279_UVFITS,
            EHT_3C279_TXT,
            config.skip_existing,
        )
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("eht_3c279_2017_uvfits.tgz").exists()
    }
}

/// EHT Centaurus A 2017 -- nearby radio galaxy jet.
/// Single date (April 10, 2017). Bands: low + high. Pipeline: EHT-HOPS.
pub struct EhtCenAProvider;

impl DatasetProvider for EhtCenAProvider {
    fn name(&self) -> &str {
        "EHT Centaurus A 2017 Data Bundle"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        fetch_eht_multi(
            self.name(),
            &config.output_dir,
            "eht_cena_2017",
            EHT_CENA_CSV,
            EHT_CENA_UVFITS,
            EHT_CENA_TXT,
            config.skip_existing,
        )
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("eht_cena_2017_uvfits.tgz").exists()
    }
}

simple_provider! {
    /// EHT M87* 2011-2013 legacy monitoring data.
    /// Pre-full-array observations, amplitude-only visibility data.
    /// Single .tgz containing all epochs (24 KB).
    pub struct EhtM87LegacyProvider;
    name = "EHT M87 2011-2013 Legacy";
    output = "eht_m87_legacy_2011_2013.tgz";
    urls = EHT_M87_LEGACY;
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_names_are_unique() {
        let names: Vec<&str> = vec![
            EhtM87_2017Provider.name(),
            EhtM87Provider.name(),
            EhtSgrAProvider.name(),
            Eht3c279Provider.name(),
            EhtCenAProvider.name(),
            EhtM87LegacyProvider.name(),
        ];
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(
            names.len(),
            sorted.len(),
            "EHT provider names must be unique"
        );
    }
}

//! Fetch implementation for nanograv. See nanograv.rs for record types and parsers.

use super::nanograv::{bestfit, extract_free_spectrum_from_kde_zip, write_free_spectrum_csv};
use crate::fetcher::{
    DatasetProvider, FetchConfig, FetchError, download_to_file, download_with_fallbacks,
    extract_tar_gz, validate_not_html,
};
use std::{
    fs,
    path::{Path, PathBuf},
};

const NANOGRAV_KDE_URL: &str =
    "https://zenodo.org/api/records/10344086/files/NANOGrav15yr_KDE-FreeSpectra_v1.1.0.zip/content";

/// NANOGrav 15yr full pulsar-timing release, Zenodo record 16051178 (CC BY 4.0).
///
/// This is the public v2.1.0 release described by the collaboration as the
/// complete narrowband + wideband 15-year timing package, including TOAs,
/// timing solutions, clock files, templates, correlations, and residuals.
const NANOGRAV_TIMING_URLS: &[&str] = &[
    "https://zenodo.org/api/records/16051178/files/NANOGrav15yr_PulsarTiming_v2.1.0.tar.gz/content",
];

/// NANOGrav 15-year free spectrum data provider.
///
/// Downloads the KDE ZIP (5.3 MB) from Zenodo, extracts point estimates
/// (median + 90% credible interval) for each of 30 frequency bins, and
/// saves as a simple CSV.
pub struct NanoGrav15yrProvider;

impl DatasetProvider for NanoGrav15yrProvider {
    fn name(&self) -> &str {
        "NANOGrav 15yr Free Spectrum"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let csv_output = config.output_dir.join("nanograv_15yr_freespectrum.csv");
        if config.skip_existing && csv_output.exists() {
            log::info!("{} already cached at {}", self.name(), csv_output.display());
            return Ok(csv_output);
        }

        // Download KDE ZIP
        let zip_path = config.output_dir.join("nanograv_15yr_kde.zip");
        log::info!("Downloading {} from Zenodo...", self.name());
        match download_to_file(NANOGRAV_KDE_URL, &zip_path) {
            Ok(bytes) => {
                let data = fs::read(&zip_path)?;
                if let Err(e) = validate_not_html(&data) {
                    log::warn!("Validation failed: {}", e);
                    fs::remove_file(&zip_path).ok();
                    return Err(e);
                }
                log::info!("Saved {} bytes to {}", bytes, zip_path.display());
            }
            Err(e) => {
                log::warn!("Download failed: {}", e);
                // Fall back to hardcoded values
                log::warn!("Using hardcoded bestfit values instead");
                write_free_spectrum_csv(&bestfit::HD_FREE_SPECTRUM, &csv_output)?;
                log::info!("Wrote {} bins to {}", bestfit::N_BINS, csv_output.display());
                return Ok(csv_output);
            }
        }

        // Extract point estimates from KDE
        log::info!("Extracting free spectrum from KDE posterior densities...");
        match extract_free_spectrum_from_kde_zip(&zip_path) {
            Ok(points) => {
                write_free_spectrum_csv(&points, &csv_output)?;
                log::info!(
                    "Extracted {} frequency bins to {}",
                    points.len(),
                    csv_output.display()
                );
            }
            Err(e) => {
                log::warn!("KDE extraction failed: {}", e);
                log::warn!("Using hardcoded bestfit values instead");
                write_free_spectrum_csv(&bestfit::HD_FREE_SPECTRUM, &csv_output)?;
                log::info!("Wrote {} bins to {}", bestfit::N_BINS, csv_output.display());
            }
        }

        Ok(csv_output)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config
            .output_dir
            .join("nanograv_15yr_freespectrum.csv")
            .exists()
    }
}

/// NANOGrav 15-year full pulsar-timing data provider.
///
/// Downloads the official v2.1.0 public release tarball and extracts it under
/// `data/external/nanograv_15yr_timing/`. This is the real timing-data lane,
/// distinct from the smaller free-spectrum KDE product above.
pub struct NanoGrav15yrTimingProvider;

impl DatasetProvider for NanoGrav15yrTimingProvider {
    fn name(&self) -> &str {
        "NANOGrav 15yr Pulsar Timing"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let archive_path = config
            .output_dir
            .join("nanograv_15yr_pulsar_timing_v2.1.0.tar.gz");
        let extract_dir = config.output_dir.join("nanograv_15yr_timing");
        if config.skip_existing && nanograv_timing_tree_present(&extract_dir) {
            log::info!(
                "{} already cached at {}",
                self.name(),
                extract_dir.display()
            );
            return Ok(extract_dir);
        }

        download_with_fallbacks(
            self.name(),
            NANOGRAV_TIMING_URLS,
            &archive_path,
            config.skip_existing,
        )?;
        let extracted = extract_tar_gz(&archive_path, &extract_dir)?;
        if extracted.is_empty() {
            return Err(FetchError::Validation(format!(
                "Extracted zero paths from {}",
                archive_path.display()
            )));
        }
        if !nanograv_timing_tree_present(&extract_dir) {
            return Err(FetchError::Validation(format!(
                "Extracted {} but did not find .par/.tim timing files under {}",
                archive_path.display(),
                extract_dir.display()
            )));
        }
        Ok(extract_dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        nanograv_timing_tree_present(&config.output_dir.join("nanograv_15yr_timing"))
    }
}

fn nanograv_timing_tree_present(root: &Path) -> bool {
    fn walk(dir: &Path, saw_par: &mut bool, saw_tim: &mut bool) -> bool {
        let Ok(entries) = fs::read_dir(dir) else {
            return false;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if walk(&path, saw_par, saw_tim) && *saw_par && *saw_tim {
                    return true;
                }
                continue;
            }
            match path.extension().and_then(|ext| ext.to_str()) {
                Some("par") => *saw_par = true,
                Some("tim") => *saw_tim = true,
                _ => {}
            }
            if *saw_par && *saw_tim {
                return true;
            }
        }
        *saw_par && *saw_tim
    }

    if !root.exists() {
        return false;
    }
    let mut saw_par = false;
    let mut saw_tim = false;
    walk(root, &mut saw_par, &mut saw_tim)
}

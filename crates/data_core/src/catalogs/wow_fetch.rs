//! Fetch implementation for wow. See wow.rs for record types and parsers.

use crate::fetcher::{DatasetProvider, FetchConfig, FetchError};
use std::path::PathBuf;

/// IIIF image endpoint for the 1977 Wow! signal printout scan.
///
/// Hosted by the Ohio History Connection via CONTENTdm / IIIF.
/// Full-resolution JPEG (~500 KB), starts with FF D8 FF (passes validate_not_html).
const WOW_PRINTOUT_IIIF_URL: &str =
    "https://cdm16007.contentdm.oclc.org/iiif/2/p267401coll32:12429/full/full/0/default.jpg";

simple_provider! {
    /// Dataset provider for the 1977 Wow! signal archival printout scan.
    ///
    /// Downloads the full-resolution IIIF JPEG from the Ohio History Connection.
    pub struct WowPrintoutProvider;
    name = "Wow! Signal 1977 Printout Scan";
    output = "wow_1977_printout.jpg";
    urls = &[WOW_PRINTOUT_IIIF_URL];
}

/// Dataset provider for the BL 6EQUJ5 GBT observation manifest.
///
/// Reads from a committed CSV manifest (no HTTP crawling of BL directory listings).
pub struct Bl6equj5ManifestProvider;

impl DatasetProvider for Bl6equj5ManifestProvider {
    fn name(&self) -> &str {
        "BL 6EQUJ5 GBT Manifest"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let manifest = PathBuf::from("data/csv/bl_6equj5_gbt_manifest.csv");
        if !manifest.exists() {
            return Err(FetchError::Validation(
                "BL 6EQUJ5 manifest CSV not found at data/csv/bl_6equj5_gbt_manifest.csv. \
                 This is a committed file, not an HTTP download."
                    .to_string(),
            ));
        }
        // Copy to output dir if different
        let output = config.output_dir.join("bl_6equj5_gbt_manifest.csv");
        if config.skip_existing && output.exists() {
            return Ok(output);
        }
        if let Some(parent) = output.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::copy(&manifest, &output)?;
        Ok(output)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config
            .output_dir
            .join("bl_6equj5_gbt_manifest.csv")
            .exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fetcher::DatasetProvider;

    #[test]
    fn test_wow_provider_name() {
        let p = WowPrintoutProvider;
        assert_eq!(p.name(), "Wow! Signal 1977 Printout Scan");
    }

    #[test]
    fn test_bl_manifest_provider_name() {
        let p = Bl6equj5ManifestProvider;
        assert_eq!(p.name(), "BL 6EQUJ5 GBT Manifest");
    }
}

//! Fetch implementation for sdss. See sdss.rs for record types and parsers.

use crate::{
    fetcher::{DatasetProvider, FetchConfig, FetchError, download_to_string, validate_not_html},
    formats::tap::percent_encode_query,
};
use std::{fs, path::PathBuf};

/// SDSS SkyServer SQL query for TOP 50000 quasars.
///
/// Uses `specObjID` (not `objID`) as the identifier from SpecObj,
/// and `bestObjID` to join to PhotoObj for photometric magnitudes.
const SDSS_QUERY: &str = "\
SELECT TOP 50000 \
  s.specObjID as objid, s.ra, s.dec, s.z, s.zErr as zerr, \
  p.psfMag_u as u, p.psfMag_g as g, p.psfMag_r as r, \
  p.psfMag_i as i \
FROM SpecObj s \
JOIN PhotoObj p ON s.bestObjID = p.objID \
WHERE s.class = 'QSO' AND s.zWarning = 0 AND s.z > 0.1 \
ORDER BY s.z";

/// Build the SkyServer CSV download URL with proper percent-encoding.
fn skyserver_csv_url(query: &str) -> String {
    let encoded = percent_encode_query(query);
    format!(
        "https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch?cmd={}&format=csv",
        encoded
    )
}

/// SDSS DR18 quasar catalog dataset provider.
pub struct SdssQsoProvider;

impl DatasetProvider for SdssQsoProvider {
    fn name(&self) -> &str {
        "SDSS DR18 Quasars"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("sdss_dr18_quasars.csv");
        if config.skip_existing && output.exists() {
            log::info!("{} already cached at {}", self.name(), output.display());
            return Ok(output);
        }

        let url = skyserver_csv_url(SDSS_QUERY);
        log::info!("Downloading {} from SkyServer...", self.name());
        let body = download_to_string(&url)?;
        validate_not_html(body.as_bytes())?;

        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&output, &body)?;
        log::info!("Saved {} bytes to {}", body.len(), output.display());
        Ok(output)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("sdss_dr18_quasars.csv").exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sdss_skyserver_url() {
        let url = skyserver_csv_url("SELECT TOP 10 ra FROM SpecObj");
        assert!(url.starts_with("https://skyserver.sdss.org/dr18/"));
        assert!(url.contains("format=csv"));
        assert!(url.contains("SELECT"));
    }
}

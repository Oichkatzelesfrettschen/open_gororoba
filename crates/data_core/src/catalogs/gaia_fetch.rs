//! Fetch implementation for gaia. See gaia.rs for record types and parsers.

use crate::{
    fetcher::{DatasetProvider, FetchConfig, FetchError, validate_not_html},
    formats::tap,
};
use std::{fs, path::PathBuf};

/// ADQL query for nearby stars with radial velocities from Gaia DR3.
const GAIA_ADQL: &str = "\
SELECT TOP 50000 \
  source_id, ra, dec, parallax, parallax_error, \
  pmra, pmdec, radial_velocity, radial_velocity_error, \
  phot_g_mean_mag, bp_rp \
FROM gaiadr3.gaia_source \
WHERE radial_velocity IS NOT NULL \
  AND parallax > 5 \
  AND parallax_error/parallax < 0.1 \
ORDER BY parallax DESC";

const GAIA_TAP_BASE: &str = "https://gea.esac.esa.int/tap-server/tap";

/// Gaia DR3 nearby stars with radial velocities.
pub struct GaiaDr3Provider;

impl DatasetProvider for GaiaDr3Provider {
    fn name(&self) -> &str {
        "Gaia DR3 Nearby Stars"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("gaia_dr3_nearby.csv");
        if config.skip_existing && output.exists() {
            log::info!("{} already cached at {}", self.name(), output.display());
            return Ok(output);
        }

        log::info!("Querying {} via TAP...", self.name());
        let body = tap::tap_query(GAIA_TAP_BASE, GAIA_ADQL, "csv")?;
        validate_not_html(body.as_bytes())?;

        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&output, &body)?;
        log::info!("Saved {} bytes to {}", body.len(), output.display());
        Ok(output)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("gaia_dr3_nearby.csv").exists()
    }
}

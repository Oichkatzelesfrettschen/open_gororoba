//! Fetch implementation for hic_raa. See hic_raa.rs for record types and parsers.

use super::hic_raa::{
    atlas_jet_raa_tables, atlas_jet_v2_tables, cms_pbpb_5020_raa_tables, cms_pbpb_5020_v2_tables,
    cms_pp_5020_spectrum_table, phenix_auau_raa_tables,
};
use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, download_with_fallbacks};
use std::path::PathBuf;

// ============================================================================
// DatasetProvider
// ============================================================================

/// Provider for downloading all HIC R_AA and v2 datasets.
pub struct HicRaaProvider;

impl DatasetProvider for HicRaaProvider {
    fn name(&self) -> &str {
        "HIC R_AA + v2 datasets"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let base = config.output_dir.join("hic_raa");

        // CMS PbPb 5.02 TeV R_AA
        for table in cms_pbpb_5020_raa_tables() {
            let path = base.join(table.filename);
            download_with_fallbacks(
                table.name,
                &[table.url_primary, table.url_fallback],
                &path,
                config.skip_existing,
            )?;
        }

        // CMS pp spectrum
        let pp = cms_pp_5020_spectrum_table();
        download_with_fallbacks(
            pp.name,
            &[pp.url_primary, pp.url_fallback],
            &base.join(pp.filename),
            config.skip_existing,
        )?;

        // CMS v2
        for table in cms_pbpb_5020_v2_tables() {
            let path = base.join(table.filename);
            download_with_fallbacks(
                table.name,
                &[table.url_primary, table.url_fallback],
                &path,
                config.skip_existing,
            )?;
        }

        // ATLAS jet R_AA
        for table in atlas_jet_raa_tables() {
            let path = base.join(table.filename);
            download_with_fallbacks(
                table.name,
                &[table.url_primary, table.url_fallback],
                &path,
                config.skip_existing,
            )?;
        }

        // ATLAS jet v2
        for table in atlas_jet_v2_tables() {
            let path = base.join(table.filename);
            download_with_fallbacks(
                table.name,
                &[table.url_primary, table.url_fallback],
                &path,
                config.skip_existing,
            )?;
        }

        // PHENIX R_AA
        for table in phenix_auau_raa_tables() {
            let path = base.join(table.filename);
            download_with_fallbacks(
                table.name,
                &[table.url_primary, table.url_fallback],
                &path,
                config.skip_existing,
            )?;
        }

        Ok(base)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        let base = config.output_dir.join("hic_raa");
        // Check if at least the first CMS table exists
        base.join("cms_pbpb_5020_raa_table8.csv").exists()
    }
}

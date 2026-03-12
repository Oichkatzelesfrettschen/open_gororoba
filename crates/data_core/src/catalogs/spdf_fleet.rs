//! `SpdfMission` helper: shared parse/to-OMNI pipeline for SPDF merged datasets.
//!
//! The SPDF merged hourly ASCII format is used by many spacecraft (Voyager,
//! Pioneer, Ulysses, Juno, Cassini, New Horizons, Helios). Each spacecraft
//! has the same parse -> optional year fixup -> to-OMNI pipeline, differing
//! only in column layout, coordinate system, and 2-digit year correction.
//!
//! This module centralises that pipeline so individual spacecraft modules
//! only need to declare their parameters and delegate to `SpdfMission`.

use super::omni::OmniRecord;
use super::spdf_merged::{SpdfColumnLayout, SpdfMergedRecord, parse_spdf_merged, spdf_to_omni};
use crate::fetcher::FetchError;

/// Shared parse/to-OMNI configuration for a SPDF merged hourly dataset.
///
/// Eliminates repeated `parse_spdf_merged` + year-fixup + `spdf_to_omni` triads
/// across the seven SPDF spacecraft modules.  The `DatasetProvider` impls
/// (which have per-mission year-range fetch loops) are kept hand-written in
/// each spacecraft module.
pub struct SpdfMission {
    /// Column layout mapping for this spacecraft's merged hourly ASCII format.
    pub layout: &'static SpdfColumnLayout,
    /// Whether the B-field is in Solar Ecliptic (SE) coordinates (`true`) or
    /// RTN (Radial-Tangential-Normal) coordinates (`false`).
    pub b_is_se: bool,
    /// Optional post-parse record fixup (e.g., 2-digit -> 4-digit year).
    pub year_fixup: Option<fn(&mut SpdfMergedRecord)>,
}

impl SpdfMission {
    /// Parse merged hourly ASCII content into `SpdfMergedRecord`s, applying
    /// the optional year fixup to each record.
    pub fn parse_merged(&self, content: &str) -> Vec<SpdfMergedRecord> {
        let mut records = parse_spdf_merged(content, self.layout);
        if let Some(fixup) = self.year_fixup {
            for r in &mut records {
                fixup(r);
            }
        }
        records
    }

    /// Read and parse a merged hourly ASCII file from disk.
    pub fn parse_file(&self, path: &std::path::Path) -> Result<Vec<SpdfMergedRecord>, FetchError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| FetchError::Validation(format!("read error: {}", e)))?;
        Ok(self.parse_merged(&content))
    }

    /// Convert `SpdfMergedRecord`s to `OmniRecord`s using this mission's
    /// coordinate system.
    pub fn to_omni(&self, records: &[SpdfMergedRecord]) -> Vec<OmniRecord> {
        spdf_to_omni(records, self.b_is_se)
    }
}

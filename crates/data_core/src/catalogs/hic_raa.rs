//! Heavy-ion collision R_AA and v2 data catalogs from HEPData.
//!
//! Provides download URLs, parsing functions, and DatasetProvider
//! implementations for R_AA and v2 measurements needed for the
//! Arleo-Falmagne QGP path-length scaling reproduction.
//!
//! Datasets:
//! - ALICE Pb-Pb 5.02 TeV h+/- R_AA (HEPData record 89396)
//! - CMS Pb-Pb 5.02 TeV h+/- R_AA (INSPIRE ins1496050)
//! - CMS Pb-Pb 2.76 TeV h+/- R_AA (INSPIRE ins1088823)
//! - ATLAS Pb-Pb 5.02 TeV jet R_AA (INSPIRE ins1673184)
//! - CMS Pb-Pb 5.02 TeV v2 (INSPIRE ins1511868)
//! - CMS Pb-Pb 2.76 TeV v2 (INSPIRE ins1107658)
//! - ATLAS Pb-Pb 5.02 TeV jet v2 (INSPIRE ins1967021)
//! - PHENIX Au-Au 200 GeV pi0 R_AA (INSPIRE ins1127262)
//! - ALICE dNch/deta Pb-Pb 5.02 TeV (INSPIRE ins1507090)

use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, download_with_fallbacks};
use crate::parse::parse_f64_or_nan;
use std::path::{Path, PathBuf};

/// Parsed R_AA data point from HEPData CSV.
#[derive(Debug, Clone)]
pub struct RaaPoint {
    /// pT bin low edge (GeV).
    pub pt_lo: f64,
    /// pT bin high edge (GeV).
    pub pt_hi: f64,
    /// pT bin center (GeV).
    pub pt: f64,
    /// R_AA central value.
    pub raa: f64,
    /// Statistical error (symmetric).
    pub stat_err: f64,
    /// Systematic error (positive).
    pub syst_err_up: f64,
    /// Systematic error (negative, stored as positive).
    pub syst_err_down: f64,
}

/// Parsed v2 data point from HEPData CSV.
#[derive(Debug, Clone)]
pub struct V2Point {
    /// pT bin low edge (GeV).
    pub pt_lo: f64,
    /// pT bin high edge (GeV).
    pub pt_hi: f64,
    /// pT bin center (GeV).
    pub pt: f64,
    /// v2 central value.
    pub v2: f64,
    /// Statistical error.
    pub stat_err: f64,
    /// Systematic error.
    pub syst_err: f64,
}

/// HEPData table descriptor.
#[derive(Debug, Clone)]
pub struct HepDataTable {
    /// Human-readable name.
    pub name: &'static str,
    /// HEPData URL (primary).
    pub url_primary: &'static str,
    /// Fallback URL (alternate INSPIRE/record format).
    pub url_fallback: &'static str,
    /// Local filename within the dataset directory.
    pub filename: &'static str,
}

// ============================================================================
// ALICE Pb-Pb 5.02 TeV h+/- R_AA (HEPData record 89396)
// Already downloaded in Sprint 54B under data/external/alice_pbpb_raa/
// ============================================================================

/// Directory for ALICE Pb-Pb R_AA data.
pub const ALICE_PBPB_RAA_DIR: &str = "alice_pbpb_raa";

/// Number of centrality bins in ALICE data (Tables 1-20).
pub const ALICE_PBPB_RAA_N_CENT: usize = 20;

/// ALICE centrality bin labels matching table order.
pub const ALICE_PBPB_RAA_CENTRALITIES: &[&str] = &[
    "0-5%", "5-10%", "10-15%", "15-20%", "20-25%", "25-30%", "30-35%", "35-40%", "40-45%",
    "45-50%", "50-55%", "55-60%", "60-65%", "65-70%", "70-75%", "75-80%", "80-85%", "85-90%",
    "90-95%", "95-100%",
];

// ============================================================================
// CMS Pb-Pb 5.02 TeV h+/- R_AA (INSPIRE ins1496050)
// ============================================================================

/// CMS 5.02 TeV R_AA HEPData tables (Tables 8-15 = R_AA per centrality).
pub fn cms_pbpb_5020_raa_tables() -> Vec<HepDataTable> {
    (8..=15)
        .map(|t| {
            let cent = match t {
                8 => "0-5%",
                9 => "5-10%",
                10 => "10-30%",
                11 => "30-50%",
                12 => "50-70%",
                13 => "70-90%",
                14 => "0-10%",
                15 => "0-100%",
                _ => "unknown",
            };
            HepDataTable {
                name: Box::leak(format!("CMS PbPb 5.02 TeV R_AA {}", cent).into_boxed_str()),
                url_primary: Box::leak(
                    format!(
                        "https://www.hepdata.net/download/table/ins1496050/Table{}/csv",
                        t
                    )
                    .into_boxed_str(),
                ),
                url_fallback: Box::leak(
                    format!(
                        "https://www.hepdata.net/download/table/ins1496050/Table%20{}/csv",
                        t
                    )
                    .into_boxed_str(),
                ),
                filename: Box::leak(format!("cms_pbpb_5020_raa_table{}.csv", t).into_boxed_str()),
            }
        })
        .collect()
}

/// CMS 5.02 TeV pp reference spectrum (Table 7).
pub fn cms_pp_5020_spectrum_table() -> HepDataTable {
    HepDataTable {
        name: "CMS pp 5.02 TeV reference spectrum",
        url_primary: "https://www.hepdata.net/download/table/ins1496050/Table7/csv",
        url_fallback: "https://www.hepdata.net/download/table/ins1496050/Table%207/csv",
        filename: "cms_pp_5020_spectrum_table7.csv",
    }
}

// ============================================================================
// CMS Pb-Pb 5.02 TeV v2 (INSPIRE ins1511868)
// ============================================================================

/// CMS v2 HEPData tables.
pub fn cms_pbpb_5020_v2_tables() -> Vec<HepDataTable> {
    // Tables vary by centrality; these are the most relevant for the analysis
    (1..=6)
        .map(|t| HepDataTable {
            name: Box::leak(format!("CMS PbPb 5.02 TeV v2 Table{}", t).into_boxed_str()),
            url_primary: Box::leak(
                format!(
                    "https://www.hepdata.net/download/table/ins1511868/Table{}/csv",
                    t
                )
                .into_boxed_str(),
            ),
            url_fallback: Box::leak(
                format!(
                    "https://www.hepdata.net/download/table/ins1511868/Table%20{}/csv",
                    t
                )
                .into_boxed_str(),
            ),
            filename: Box::leak(format!("cms_pbpb_5020_v2_table{}.csv", t).into_boxed_str()),
        })
        .collect()
}

// ============================================================================
// ATLAS Pb-Pb 5.02 TeV jet R_AA (INSPIRE ins1673184)
// ============================================================================

/// ATLAS jet R_AA tables (subset: centrality-dependent).
pub fn atlas_jet_raa_tables() -> Vec<HepDataTable> {
    // Tables 1-10 cover different centralities and jet radii
    (1..=10)
        .map(|t| HepDataTable {
            name: Box::leak(format!("ATLAS PbPb 5.02 TeV jet R_AA Table{}", t).into_boxed_str()),
            url_primary: Box::leak(
                format!(
                    "https://www.hepdata.net/download/table/ins1673184/Table{}/csv",
                    t
                )
                .into_boxed_str(),
            ),
            url_fallback: Box::leak(
                format!(
                    "https://www.hepdata.net/download/table/ins1673184/Table%20{}/csv",
                    t
                )
                .into_boxed_str(),
            ),
            filename: Box::leak(format!("atlas_jet_raa_table{}.csv", t).into_boxed_str()),
        })
        .collect()
}

// ============================================================================
// ATLAS Pb-Pb 5.02 TeV jet v2 (INSPIRE ins1967021)
// ============================================================================

/// ATLAS jet v2 tables.
pub fn atlas_jet_v2_tables() -> Vec<HepDataTable> {
    (1..=6)
        .map(|t| HepDataTable {
            name: Box::leak(format!("ATLAS PbPb 5.02 TeV jet v2 Table{}", t).into_boxed_str()),
            url_primary: Box::leak(
                format!(
                    "https://www.hepdata.net/download/table/ins1967021/Table{}/csv",
                    t
                )
                .into_boxed_str(),
            ),
            url_fallback: Box::leak(
                format!(
                    "https://www.hepdata.net/download/table/ins1967021/Table%20{}/csv",
                    t
                )
                .into_boxed_str(),
            ),
            filename: Box::leak(format!("atlas_jet_v2_table{}.csv", t).into_boxed_str()),
        })
        .collect()
}

// ============================================================================
// PHENIX Au-Au 200 GeV pi0 R_AA (INSPIRE ins1127262)
// ============================================================================

/// PHENIX pi0 R_AA tables.
pub fn phenix_auau_raa_tables() -> Vec<HepDataTable> {
    (1..=8)
        .map(|t| HepDataTable {
            name: Box::leak(format!("PHENIX AuAu 200 GeV pi0 R_AA Table{}", t).into_boxed_str()),
            url_primary: Box::leak(
                format!(
                    "https://www.hepdata.net/download/table/ins1127262/Table{}/csv",
                    t
                )
                .into_boxed_str(),
            ),
            url_fallback: Box::leak(
                format!(
                    "https://www.hepdata.net/download/table/ins1127262/Table%20{}/csv",
                    t
                )
                .into_boxed_str(),
            ),
            filename: Box::leak(format!("phenix_auau_raa_table{}.csv", t).into_boxed_str()),
        })
        .collect()
}

// ============================================================================
// Parsing
// ============================================================================

/// Parse a HEPData CSV file for R_AA data.
///
/// HEPData CSV format (after header rows starting with #):
///   pT_low, pT_high, R_AA, stat+, stat-, syst+, syst-
///
/// This parser is tolerant of varying column counts and header formats.
pub fn parse_raa_csv(path: &Path) -> Result<Vec<RaaPoint>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

    let mut points = Vec::new();
    let mut in_data = false;

    for line in content.lines() {
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') || line.starts_with('*') {
            // Detect data section start (after header lines)
            if line.contains("PT")
                || line.contains("pT")
                || line.contains("RAA")
                || line.contains("R_AA")
            {
                in_data = true;
            }
            continue;
        }

        // Try to parse as data
        let fields: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if fields.len() < 3 {
            continue;
        }

        let pt_lo = parse_f64_or_nan(fields[0]);
        let pt_hi = parse_f64_or_nan(fields[1]);
        let raa = parse_f64_or_nan(fields[2]);

        if pt_lo.is_nan() || pt_hi.is_nan() || raa.is_nan() {
            // Might be header row; try next line
            if !in_data {
                in_data = true;
            }
            continue;
        }

        let stat_err = if fields.len() > 3 {
            parse_f64_or_nan(fields[3]).abs()
        } else {
            0.0
        };

        let syst_up = if fields.len() > 5 {
            parse_f64_or_nan(fields[5]).abs()
        } else if fields.len() > 4 {
            parse_f64_or_nan(fields[4]).abs()
        } else {
            0.0
        };

        let syst_down = if fields.len() > 6 {
            parse_f64_or_nan(fields[6]).abs()
        } else {
            syst_up
        };

        points.push(RaaPoint {
            pt_lo,
            pt_hi,
            pt: 0.5 * (pt_lo + pt_hi),
            raa,
            stat_err,
            syst_err_up: syst_up,
            syst_err_down: syst_down,
        });
    }

    if points.is_empty() {
        return Err(format!("No R_AA data points found in {}", path.display()));
    }

    Ok(points)
}

/// Parse a HEPData CSV file for v2 data.
pub fn parse_v2_csv(path: &Path) -> Result<Vec<V2Point>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

    let mut points = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('*') {
            continue;
        }

        let fields: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if fields.len() < 3 {
            continue;
        }

        let pt_lo = parse_f64_or_nan(fields[0]);
        let pt_hi = parse_f64_or_nan(fields[1]);
        let v2 = parse_f64_or_nan(fields[2]);

        if pt_lo.is_nan() || pt_hi.is_nan() || v2.is_nan() {
            continue;
        }

        let stat_err = if fields.len() > 3 {
            parse_f64_or_nan(fields[3]).abs()
        } else {
            0.0
        };

        let syst_err = if fields.len() > 5 {
            parse_f64_or_nan(fields[5]).abs()
        } else if fields.len() > 4 {
            parse_f64_or_nan(fields[4]).abs()
        } else {
            0.0
        };

        points.push(V2Point {
            pt_lo,
            pt_hi,
            pt: 0.5 * (pt_lo + pt_hi),
            v2,
            stat_err,
            syst_err,
        });
    }

    if points.is_empty() {
        return Err(format!("No v2 data points found in {}", path.display()));
    }

    Ok(points)
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cms_raa_table_count() {
        let tables = cms_pbpb_5020_raa_tables();
        assert_eq!(tables.len(), 8, "Expected 8 CMS R_AA tables (Tables 8-15)");
    }

    #[test]
    fn test_cms_v2_table_count() {
        let tables = cms_pbpb_5020_v2_tables();
        assert_eq!(tables.len(), 6, "Expected 6 CMS v2 tables");
    }

    #[test]
    fn test_atlas_jet_raa_table_count() {
        let tables = atlas_jet_raa_tables();
        assert_eq!(tables.len(), 10, "Expected 10 ATLAS jet R_AA tables");
    }

    #[test]
    fn test_url_format() {
        let tables = cms_pbpb_5020_raa_tables();
        for table in &tables {
            assert!(
                table.url_primary.starts_with("https://"),
                "URL should start with https://: {}",
                table.url_primary
            );
            assert!(
                table.url_primary.ends_with("/csv"),
                "URL should end with /csv: {}",
                table.url_primary
            );
        }
    }

    #[test]
    fn test_alice_centrality_count() {
        assert_eq!(ALICE_PBPB_RAA_CENTRALITIES.len(), ALICE_PBPB_RAA_N_CENT);
    }
}

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
// ALICE Xe-Xe 5.44 TeV h+/- R_AA (HEPData record ins1672790, Table 3)
// PLB 788 (2019) 166-179
// Original HEPData Table 3 contains all centralities in one file.
// Split locally into per-centrality files alice_xexe_raa_table{1..8}.csv.
// Tables 1-8 (local): 0-5%, 5-10%, 10-20%, 20-30%, 30-40%, 40-50%, 50-60%, 60-70%
// ============================================================================

/// Directory for ALICE Xe-Xe R_AA data.
pub const ALICE_XEXE_RAA_DIR: &str = "alice_xexe_raa";

/// ALICE Xe-Xe 5.44 TeV R_AA centrality labels (Tables 1-8).
pub const ALICE_XEXE_RAA_CENTRALITIES: &[&str] = &[
    "0-5%", "5-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%",
];

/// ALICE Xe-Xe 5.44 TeV R_AA HEPData tables.
pub fn alice_xexe_5440_raa_tables() -> Vec<HepDataTable> {
    (1..=8)
        .map(|t| {
            let cent = ALICE_XEXE_RAA_CENTRALITIES[t - 1];
            HepDataTable {
                name: Box::leak(
                    format!("ALICE XeXe 5.44 TeV R_AA {}", cent).into_boxed_str(),
                ),
                url_primary: Box::leak(
                    format!(
                        "https://www.hepdata.net/download/table/ins1672790/Table{}/csv",
                        t
                    )
                    .into_boxed_str(),
                ),
                url_fallback: Box::leak(
                    format!(
                        "https://www.hepdata.net/download/table/ins1672790/Table%20{}/csv",
                        t
                    )
                    .into_boxed_str(),
                ),
                filename: Box::leak(format!("alice_xexe_raa_table{}.csv", t).into_boxed_str()),
            }
        })
        .collect()
}

// ============================================================================
// Parsing
// ============================================================================

/// Column layout detected from CSV headers.
#[derive(Debug, Clone, Copy)]
struct RaaColumnLayout {
    /// Index of pT bin low edge.
    col_pt_lo: usize,
    /// Index of pT bin high edge.
    col_pt_hi: usize,
    /// Index of R_AA central value.
    col_raa: usize,
    /// Index of stat error (positive).
    col_stat: usize,
    /// Index of syst error (positive).
    col_syst_up: usize,
    /// Index of syst error (negative).
    col_syst_down: usize,
}

impl RaaColumnLayout {
    /// 10-col HEPData ALICE format: center, LOW, HIGH, R_AA, stat+, stat-, syst+, syst-, norm+, norm-
    const HEPDATA_10: Self = Self {
        col_pt_lo: 1,
        col_pt_hi: 2,
        col_raa: 3,
        col_stat: 4,
        col_syst_up: 6,
        col_syst_down: 7,
    };

    /// Simple 7-col format: pt_lo, pt_hi, R_AA, stat+, stat-, syst+, syst-
    const SIMPLE_7: Self = Self {
        col_pt_lo: 0,
        col_pt_hi: 1,
        col_raa: 2,
        col_stat: 3,
        col_syst_up: 5,
        col_syst_down: 6,
    };
}

/// Detect R_AA column layout from a CSV header line.
///
/// Scans comma-separated column names for keywords like LOW, HIGH,
/// R_{AA}/RAA/R_AA to determine which field index holds each quantity.
/// Falls back to positional heuristics based on column count.
fn detect_raa_layout(header: &str) -> RaaColumnLayout {
    let cols: Vec<String> = header
        .split(',')
        .map(|s| s.trim().to_uppercase())
        .collect();

    // Try keyword-based detection
    let find_col = |keywords: &[&str]| -> Option<usize> {
        cols.iter().position(|c| keywords.iter().any(|kw| c.contains(kw)))
    };

    let lo = find_col(&["LOW", "PT_LO", "PT_LOW"]);
    let hi = find_col(&["HIGH", "PT_HI", "PT_HIGH"]);
    let raa = find_col(&["R_{AA}", "R_AA", "RAA"]);

    if let (Some(lo_i), Some(hi_i), Some(raa_i)) = (lo, hi, raa) {
        // Find stat and syst columns relative to R_AA
        let stat_i = raa_i + 1;
        let syst_up_i = if cols.len() > raa_i + 3 { raa_i + 3 } else { raa_i + 2 };
        let syst_down_i = if cols.len() > syst_up_i + 1 { syst_up_i + 1 } else { syst_up_i };
        return RaaColumnLayout {
            col_pt_lo: lo_i,
            col_pt_hi: hi_i,
            col_raa: raa_i,
            col_stat: stat_i,
            col_syst_up: syst_up_i,
            col_syst_down: syst_down_i,
        };
    }

    // Fallback: use column count heuristic
    if cols.len() >= 10 {
        RaaColumnLayout::HEPDATA_10
    } else {
        RaaColumnLayout::SIMPLE_7
    }
}

/// Parse a HEPData CSV file for R_AA data.
///
/// Supports two common HEPData CSV layouts via header-sniffing:
///
/// - **10-col ALICE format**: center, LOW, HIGH, R_{AA}, stat+, stat-, syst+, syst-, norm+, norm-
/// - **7-col simple format**: pT_low, pT_high, R_AA, stat+, stat-, syst+, syst-
///
/// The parser detects the layout from column header keywords (LOW, HIGH,
/// R_{AA}/R_AA/RAA) when present, falling back to column-count heuristics.
pub fn parse_raa_csv(path: &Path) -> Result<Vec<RaaPoint>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

    let mut points = Vec::new();
    let mut layout: Option<RaaColumnLayout> = None;

    for line in content.lines() {
        let line = line.trim();

        // Skip empty lines and comment-only lines
        if line.is_empty() || line.starts_with('#') || line.starts_with('*') {
            continue;
        }

        let fields: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if fields.len() < 3 {
            continue;
        }

        // Header detection: if the first field does not parse as a number,
        // this is the column-name header row.
        if layout.is_none() {
            if parse_f64_or_nan(fields[0]).is_nan() {
                layout = Some(detect_raa_layout(line));
                continue;
            }
            // No header row found -- use column count heuristic
            layout = Some(if fields.len() >= 10 {
                RaaColumnLayout::HEPDATA_10
            } else {
                RaaColumnLayout::SIMPLE_7
            });
        }

        let lay = layout.unwrap();

        let pt_lo = parse_f64_or_nan(fields.get(lay.col_pt_lo).unwrap_or(&""));
        let pt_hi = parse_f64_or_nan(fields.get(lay.col_pt_hi).unwrap_or(&""));
        let raa = parse_f64_or_nan(fields.get(lay.col_raa).unwrap_or(&""));

        if pt_lo.is_nan() || pt_hi.is_nan() || raa.is_nan() {
            continue;
        }

        let stat_err = fields
            .get(lay.col_stat)
            .map(|s| parse_f64_or_nan(s).abs())
            .unwrap_or(0.0);

        let syst_up = fields
            .get(lay.col_syst_up)
            .map(|s| parse_f64_or_nan(s).abs())
            .unwrap_or(0.0);

        let syst_down = fields
            .get(lay.col_syst_down)
            .map(|s| parse_f64_or_nan(s).abs())
            .unwrap_or(syst_up);

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

/// Column layout for v2 CSVs.
#[derive(Debug, Clone, Copy)]
struct V2ColumnLayout {
    col_pt_lo: usize,
    col_pt_hi: usize,
    col_v2: usize,
    col_stat: usize,
    col_syst: usize,
}

/// Detect v2 column layout from a CSV header line.
fn detect_v2_layout(header: &str) -> V2ColumnLayout {
    let cols: Vec<String> = header
        .split(',')
        .map(|s| s.trim().to_uppercase())
        .collect();

    let find_col = |keywords: &[&str]| -> Option<usize> {
        cols.iter().position(|c| keywords.iter().any(|kw| c.contains(kw)))
    };

    let lo = find_col(&["LOW", "PT_LO", "PT_LOW"]);
    let hi = find_col(&["HIGH", "PT_HI", "PT_HIGH"]);
    let v2 = find_col(&["V2", "V_{2}"]);

    if let (Some(lo_i), Some(hi_i), Some(v2_i)) = (lo, hi, v2) {
        let stat_i = v2_i + 1;
        let syst_i = if cols.len() > v2_i + 3 { v2_i + 3 } else { v2_i + 2 };
        return V2ColumnLayout {
            col_pt_lo: lo_i,
            col_pt_hi: hi_i,
            col_v2: v2_i,
            col_stat: stat_i,
            col_syst: syst_i,
        };
    }

    // Fallback: column-count heuristic
    if cols.len() >= 10 {
        // 10-col: center, LOW, HIGH, v2, stat+, stat-, syst+, syst-, ...
        V2ColumnLayout { col_pt_lo: 1, col_pt_hi: 2, col_v2: 3, col_stat: 4, col_syst: 6 }
    } else {
        // Simple: pt_lo, pt_hi, v2, stat, syst
        V2ColumnLayout { col_pt_lo: 0, col_pt_hi: 1, col_v2: 2, col_stat: 3, col_syst: 4 }
    }
}

/// Parse a HEPData CSV file for v2 data.
///
/// Uses the same header-sniffing approach as `parse_raa_csv` to detect
/// column layout from header keywords (LOW, HIGH, V2/V_{2}).
pub fn parse_v2_csv(path: &Path) -> Result<Vec<V2Point>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

    let mut points = Vec::new();
    let mut layout: Option<V2ColumnLayout> = None;

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('*') {
            continue;
        }

        let fields: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if fields.len() < 3 {
            continue;
        }

        // Header detection
        if layout.is_none() {
            if parse_f64_or_nan(fields[0]).is_nan() {
                layout = Some(detect_v2_layout(line));
                continue;
            }
            layout = Some(if fields.len() >= 10 {
                V2ColumnLayout { col_pt_lo: 1, col_pt_hi: 2, col_v2: 3, col_stat: 4, col_syst: 6 }
            } else {
                V2ColumnLayout { col_pt_lo: 0, col_pt_hi: 1, col_v2: 2, col_stat: 3, col_syst: 4 }
            });
        }

        let lay = layout.unwrap();

        let pt_lo = parse_f64_or_nan(fields.get(lay.col_pt_lo).unwrap_or(&""));
        let pt_hi = parse_f64_or_nan(fields.get(lay.col_pt_hi).unwrap_or(&""));
        let v2 = parse_f64_or_nan(fields.get(lay.col_v2).unwrap_or(&""));

        if pt_lo.is_nan() || pt_hi.is_nan() || v2.is_nan() {
            continue;
        }

        let stat_err = fields
            .get(lay.col_stat)
            .map(|s| parse_f64_or_nan(s).abs())
            .unwrap_or(0.0);

        let syst_err = fields
            .get(lay.col_syst)
            .map(|s| parse_f64_or_nan(s).abs())
            .unwrap_or(0.0);

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

    #[test]
    fn test_parse_raa_csv_10col_alice_format() {
        // Real ALICE HEPData 10-col format: center, LOW, HIGH, R_{AA}, stat+, stat-, syst+, syst-, norm+, norm-
        let csv = "\
#: table_doi: 10.17182/hepdata.89396.v1/t1
#: name: Table 1
p_{T} (GeV/c),p_{T} (GeV/c) LOW,p_{T} (GeV/c) HIGH,R_{AA},stat +,stat -,syst +,syst -,norm +,norm -
0.175,0.15,0.2,0.185347,4.29029e-05,-4.29029e-05,0.00964405,-0.00964405,0.00759329,-0.00759329
0.225,0.2,0.25,0.194939,3.08507e-05,-3.08507e-05,0.00792924,-0.00792924,0.00798624,-0.00798624
";
        let mut f = tempfile::NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut f, csv.as_bytes()).unwrap();
        std::io::Write::flush(&mut f).unwrap();

        let pts = parse_raa_csv(f.path()).unwrap();
        assert_eq!(pts.len(), 2, "Should parse 2 data rows");
        // pt_lo should be column 1 (LOW), not column 0 (center)
        assert!(
            (pts[0].pt_lo - 0.15).abs() < 1e-6,
            "pt_lo should be 0.15 (LOW), got {}",
            pts[0].pt_lo
        );
        assert!(
            (pts[0].pt_hi - 0.2).abs() < 1e-6,
            "pt_hi should be 0.2 (HIGH), got {}",
            pts[0].pt_hi
        );
        assert!(
            (pts[0].raa - 0.185347).abs() < 1e-4,
            "raa should be 0.185347, got {}",
            pts[0].raa
        );
        assert!(
            (pts[0].syst_err_up - 0.00964405).abs() < 1e-6,
            "syst_err_up should be 0.00964405, got {}",
            pts[0].syst_err_up
        );
    }

    #[test]
    fn test_parse_raa_csv_simple_7col_format() {
        let csv = "\
pT_lo,pT_hi,R_AA,stat+,stat-,syst+,syst-
5.0,6.0,0.35,0.02,-0.02,0.05,-0.05
6.0,8.0,0.40,0.03,-0.03,0.06,-0.06
";
        let mut f = tempfile::NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut f, csv.as_bytes()).unwrap();
        std::io::Write::flush(&mut f).unwrap();

        let pts = parse_raa_csv(f.path()).unwrap();
        assert_eq!(pts.len(), 2);
        assert!((pts[0].pt_lo - 5.0).abs() < 1e-6);
        assert!((pts[0].pt_hi - 6.0).abs() < 1e-6);
        assert!((pts[0].raa - 0.35).abs() < 1e-6);
    }

    #[test]
    fn test_parse_raa_csv_no_header_10col() {
        // 10-col with no header: should detect via column count
        let csv = "\
0.175,0.15,0.2,0.185347,4.29029e-05,-4.29029e-05,0.00964405,-0.00964405,0.00759329,-0.00759329
";
        let mut f = tempfile::NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut f, csv.as_bytes()).unwrap();
        std::io::Write::flush(&mut f).unwrap();

        let pts = parse_raa_csv(f.path()).unwrap();
        assert_eq!(pts.len(), 1);
        assert!(
            (pts[0].pt_lo - 0.15).abs() < 1e-6,
            "10-col without header should map col 1 to pt_lo"
        );
        assert!(
            (pts[0].raa - 0.185347).abs() < 1e-4,
            "10-col without header should map col 3 to raa"
        );
    }

    #[test]
    fn test_parse_raa_csv_real_alice_file() {
        let path = std::path::Path::new("data/external/alice_pbpb_raa/table_1.csv");
        if !path.exists() {
            eprintln!("Skipping: ALICE data not available");
            return;
        }
        let pts = parse_raa_csv(path).unwrap();
        assert!(pts.len() > 10, "Should parse many R_AA points");
        // First point in 0-5% centrality: R_AA should be < 1 (suppression)
        assert!(
            pts[0].raa < 1.0 && pts[0].raa > 0.0,
            "R_AA should be in (0,1), got {}",
            pts[0].raa
        );
        // pt_lo should be small pT values, not R_AA values
        assert!(
            pts[0].pt_lo < 1.0,
            "pt_lo should be < 1 GeV for first bin, got {}",
            pts[0].pt_lo
        );
    }

    #[test]
    fn test_parse_v2_csv_with_header() {
        let csv = "\
pT_lo,pT_hi,v2,stat,syst
1.0,2.0,0.05,0.01,0.005
2.0,3.0,0.08,0.015,0.007
";
        let mut f = tempfile::NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut f, csv.as_bytes()).unwrap();
        std::io::Write::flush(&mut f).unwrap();

        let pts = parse_v2_csv(f.path()).unwrap();
        assert_eq!(pts.len(), 2);
        assert!((pts[0].v2 - 0.05).abs() < 1e-6);
    }
}

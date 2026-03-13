//! LoTSS (LOFAR Two-metre Sky Survey) source catalog parser.
//!
//! Supports three releases:
//! - DR1: ~325K sources, Shimwell et al. 2019, A&A 622 A1
//!   https://lofar-surveys.org/public/LoTSS_DR1_v1.1.srl.fits
//! - DR2: ~4.4M sources, Shimwell et al. 2022, A&A 659 A1
//!   https://lofar-surveys.org/public/LoTSS_DR2_v110.srl.fits
//! - DR3: ~13.7M sources (Feb 2026), 88% northern sky at 144 MHz, 6" resolution
//!   VO Cone Search: https://vo.astron.nl/lotss_dr3/q/src_cone/scs.xml
//!
//! DR1/DR2 bulk catalogs are FITS BINTABLE; DR3 is accessed via VO Cone Search
//! returning VOTable XML or FITS depending on the requested format.
//!
//! Column correspondence (DR1/DR2 and DR3 share the same names in the
//! source-catalogue extensions):
//!   Source_Name, RA, DEC, Total_flux, E_Total_flux, Peak_flux, Isl_rms,
//!   Spectral_index, E_spectral_index, Resolved, S_Code, Maj, Min, PA
//!
//! Feature-gated behind `data_core/fits`.

use crate::fetcher::FetchError;
use std::{collections::HashMap, ops::Range, path::Path};

/// LoTSS catalog release identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoTSSRelease {
    DR1,
    DR2,
    DR3,
}

impl LoTSSRelease {
    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            LoTSSRelease::DR1 => "DR1",
            LoTSSRelease::DR2 => "DR2",
            LoTSSRelease::DR3 => "DR3",
        }
    }
}

/// A single radio source from a LoTSS catalog.
#[derive(Debug, Clone)]
pub struct LoTSSSource {
    /// IAU source name (e.g. "ILTJ000001.1+123456").
    pub source_name: String,
    /// Right ascension (degrees, J2000).
    pub ra_deg: f64,
    /// Declination (degrees, J2000).
    pub dec_deg: f64,
    /// Integrated 144 MHz flux density (mJy).
    pub flux_mjy: f32,
    /// Uncertainty on integrated flux (mJy).
    pub flux_err_mjy: f32,
    /// Peak flux density (mJy/beam).
    pub peak_flux: f32,
    /// Local RMS noise (mJy/beam).
    pub local_rms: f32,
    /// In-band spectral index (None if not measured).
    pub spectral_index: Option<f32>,
    /// Uncertainty on spectral index.
    pub spectral_index_err: Option<f32>,
    /// True if the source is spatially resolved.
    pub resolved: bool,
    /// Morphology code: 'S' single, 'C' overlapping Gaussians, 'M' multi-component.
    pub structure_code: char,
    /// Major axis FWHM (arcsec).
    pub maj_arcsec: f32,
    /// Minor axis FWHM (arcsec).
    pub min_arcsec: f32,
    /// Position angle of major axis (degrees, N through E).
    pub pa_deg: f32,
    /// Catalog release this source comes from.
    pub release: LoTSSRelease,
}

/// Columns commonly present across LoTSS releases.
///
/// DR2 masked source catalogs omit some value-added columns like
/// `Spectral_index` and `Resolved`, so those are treated as optional by the
/// FITS loader even though DR3 cone-search VOTables usually include them.
const LOTSS_FITS_COLUMNS: &[&str] = &[
    "Source_Name",
    "RA",
    "DEC",
    "Total_flux",
    "E_Total_flux",
    "Peak_flux",
    "Isl_rms",
    "S_Code",
    "Maj",
    "Min",
    "PA",
    "Spectral_index",
    "E_spectral_index",
    "Resolved",
];

/// Load LoTSS sources from a FITS BINTABLE file (DR1 or DR2 bulk download).
///
/// The FITS file should be the `*.srl.fits` source-list file distributed by
/// lofar-surveys.org.
#[cfg(feature = "fits")]
pub fn load_from_fits(
    path: &Path,
    release: LoTSSRelease,
) -> Result<Vec<LoTSSSource>, FetchError> {
    let row_count = lotss_fits_row_count(path)?;
    load_from_fits_range(path, release, 0..row_count)
}

/// Count rows in the first BINTABLE HDU of a LoTSS FITS catalog.
#[cfg(feature = "fits")]
pub fn lotss_fits_row_count(path: &Path) -> Result<usize, FetchError> {
    use fitsio::{FitsFile, hdu::HduInfo};

    let mut fptr = FitsFile::open(path)
        .map_err(|e| FetchError::Validation(format!("FITS open {}: {}", path.display(), e)))?;

    let num_hdus = {
        let mut count = 0usize;
        for _ in fptr.iter() {
            count += 1;
        }
        count
    };

    let mut table_idx: Option<usize> = None;
    let mut column_names: Vec<String> = Vec::new();
    let mut row_count: Option<usize> = None;

    for idx in 1..num_hdus {
        let hdu = fptr
            .hdu(idx)
            .map_err(|e| FetchError::Validation(format!("hdu {}: {}", idx, e)))?;
        if let HduInfo::TableInfo {
            column_descriptions,
            num_rows,
            ..
        } = hdu.info
        {
            table_idx = Some(idx);
            column_names = column_descriptions.iter().map(|d| d.name.clone()).collect();
            row_count = Some(num_rows);
            break;
        }
    }

    let table_idx = table_idx.ok_or_else(|| {
        FetchError::Validation("No BINTABLE HDU found in FITS file".to_string())
    })?;

    let _ = table_idx;
    let _ = column_names;
    row_count.ok_or_else(|| FetchError::Validation("No row count found in FITS table".to_string()))
}

/// Load a range of LoTSS sources from a FITS BINTABLE file.
#[cfg(feature = "fits")]
pub fn load_from_fits_range(
    path: &Path,
    release: LoTSSRelease,
    row_range: Range<usize>,
) -> Result<Vec<LoTSSSource>, FetchError> {
    use fitsio::{FitsFile, hdu::HduInfo};

    let mut fptr = FitsFile::open(path)
        .map_err(|e| FetchError::Validation(format!("FITS open {}: {}", path.display(), e)))?;

    let num_hdus = {
        let mut count = 0usize;
        for _ in fptr.iter() {
            count += 1;
        }
        count
    };

    let mut table_idx: Option<usize> = None;
    let mut column_names: Vec<String> = Vec::new();
    let mut table_row_count: Option<usize> = None;

    for idx in 1..num_hdus {
        let hdu = fptr
            .hdu(idx)
            .map_err(|e| FetchError::Validation(format!("hdu {}: {}", idx, e)))?;
        if let HduInfo::TableInfo {
            column_descriptions,
            num_rows,
            ..
        } = hdu.info
        {
            table_idx = Some(idx);
            column_names = column_descriptions.iter().map(|d| d.name.clone()).collect();
            table_row_count = Some(num_rows);
            break;
        }
    }

    let table_idx = table_idx.ok_or_else(|| {
        FetchError::Validation("No BINTABLE HDU found in FITS file".to_string())
    })?;
    let table_row_count = table_row_count.unwrap_or(0);
    if row_range.start > row_range.end {
        return Err(FetchError::Validation(format!(
            "Invalid FITS row range {}..{}",
            row_range.start, row_range.end
        )));
    }
    let bounded_range = row_range.start.min(table_row_count)..row_range.end.min(table_row_count);
    if bounded_range.is_empty() {
        return Ok(Vec::new());
    }

    let resolve = |want: &str| -> Result<String, FetchError> {
        column_names
            .iter()
            .find(|name| name.eq_ignore_ascii_case(want))
            .cloned()
            .ok_or_else(|| {
                FetchError::Validation(format!(
                    "Required FITS column '{}' not found in {}",
                    want,
                    path.display()
                ))
            })
    };
    let resolve_optional = |want: &str| -> Option<String> {
        column_names
            .iter()
            .find(|name| name.eq_ignore_ascii_case(want))
            .cloned()
    };

    let source_name_col = resolve("Source_Name")?;
    let ra_col = resolve("RA")?;
    let dec_col = resolve("DEC")?;
    let total_flux_col = resolve("Total_flux")?;
    let e_total_flux_col = resolve("E_Total_flux")?;
    let peak_flux_col = resolve("Peak_flux")?;
    let isl_rms_col = resolve("Isl_rms")?;
    let s_code_col = resolve("S_Code")?;
    let maj_col = resolve("Maj")?;
    let min_col = resolve("Min")?;
    let pa_col = resolve("PA")?;
    let spectral_index_col = resolve_optional("Spectral_index");
    let e_spectral_index_col = resolve_optional("E_spectral_index");
    let resolved_col = resolve_optional("Resolved");

    let hdu = fptr
        .hdu(table_idx)
        .map_err(|e| FetchError::Validation(format!("hdu {}: {}", table_idx, e)))?;

    let source_name: Vec<String> = hdu
        .read_col_range(&mut fptr, &source_name_col, &bounded_range)
        .map_err(|e| FetchError::Validation(format!("col {}: {}", source_name_col, e)))?;
    let ra: Vec<f64> = hdu
        .read_col_range(&mut fptr, &ra_col, &bounded_range)
        .map_err(|e| FetchError::Validation(format!("col {}: {}", ra_col, e)))?;
    let dec: Vec<f64> = hdu
        .read_col_range(&mut fptr, &dec_col, &bounded_range)
        .map_err(|e| FetchError::Validation(format!("col {}: {}", dec_col, e)))?;
    let flux_mjy: Vec<f32> = hdu
        .read_col_range(&mut fptr, &total_flux_col, &bounded_range)
        .map_err(|e| FetchError::Validation(format!("col {}: {}", total_flux_col, e)))?;
    let flux_err_mjy: Vec<f32> = hdu
        .read_col_range(&mut fptr, &e_total_flux_col, &bounded_range)
        .map_err(|e| FetchError::Validation(format!("col {}: {}", e_total_flux_col, e)))?;
    let peak_flux: Vec<f32> = hdu
        .read_col_range(&mut fptr, &peak_flux_col, &bounded_range)
        .map_err(|e| FetchError::Validation(format!("col {}: {}", peak_flux_col, e)))?;
    let local_rms: Vec<f32> = hdu
        .read_col_range(&mut fptr, &isl_rms_col, &bounded_range)
        .map_err(|e| FetchError::Validation(format!("col {}: {}", isl_rms_col, e)))?;
    let s_code: Vec<String> = hdu
        .read_col_range(&mut fptr, &s_code_col, &bounded_range)
        .map_err(|e| FetchError::Validation(format!("col {}: {}", s_code_col, e)))?;
    let maj_arcsec: Vec<f32> = hdu
        .read_col_range(&mut fptr, &maj_col, &bounded_range)
        .map_err(|e| FetchError::Validation(format!("col {}: {}", maj_col, e)))?;
    let min_arcsec: Vec<f32> = hdu
        .read_col_range(&mut fptr, &min_col, &bounded_range)
        .map_err(|e| FetchError::Validation(format!("col {}: {}", min_col, e)))?;
    let pa_deg: Vec<f32> = hdu
        .read_col_range(&mut fptr, &pa_col, &bounded_range)
        .map_err(|e| FetchError::Validation(format!("col {}: {}", pa_col, e)))?;

    let len = ra.len();
    let spectral_index: Vec<f32> = if let Some(col) = spectral_index_col.as_ref() {
        hdu.read_col_range(&mut fptr, col, &bounded_range)
            .map_err(|e| FetchError::Validation(format!("col {}: {}", col, e)))?
    } else {
        vec![f32::NAN; len]
    };
    let spectral_index_err: Vec<f32> = if let Some(col) = e_spectral_index_col.as_ref() {
        hdu.read_col_range(&mut fptr, col, &bounded_range)
            .map_err(|e| FetchError::Validation(format!("col {}: {}", col, e)))?
    } else {
        vec![f32::NAN; len]
    };
    let resolved_raw: Vec<i32> = if let Some(col) = resolved_col.as_ref() {
        hdu.read_col_range(&mut fptr, col, &bounded_range)
            .map_err(|e| FetchError::Validation(format!("col {}: {}", col, e)))?
    } else {
        vec![0; len]
    };
    let mut sources = Vec::with_capacity(len);

    for idx in 0..len {
        let source = source_name
            .get(idx)
            .map(|s| s.trim().to_string())
            .unwrap_or_default();
        if source.is_empty() {
            continue;
        }
        let structure_code = s_code
            .get(idx)
            .and_then(|s| s.trim().chars().next())
            .unwrap_or('S');
        let spectral = spectral_index
            .get(idx)
            .copied()
            .filter(|v| v.is_finite());
        let spectral_err = spectral_index_err
            .get(idx)
            .copied()
            .filter(|v| v.is_finite());

        sources.push(LoTSSSource {
            source_name: source,
            ra_deg: *ra.get(idx).unwrap_or(&f64::NAN),
            dec_deg: *dec.get(idx).unwrap_or(&f64::NAN),
            flux_mjy: *flux_mjy.get(idx).unwrap_or(&f32::NAN),
            flux_err_mjy: *flux_err_mjy.get(idx).unwrap_or(&f32::NAN),
            peak_flux: *peak_flux.get(idx).unwrap_or(&f32::NAN),
            local_rms: *local_rms.get(idx).unwrap_or(&f32::NAN),
            spectral_index: spectral,
            spectral_index_err: spectral_err,
            resolved: resolved_raw.get(idx).copied().unwrap_or_default() != 0,
            structure_code,
            maj_arcsec: *maj_arcsec.get(idx).unwrap_or(&f32::NAN),
            min_arcsec: *min_arcsec.get(idx).unwrap_or(&f32::NAN),
            pa_deg: *pa_deg.get(idx).unwrap_or(&f32::NAN),
            release,
        });
    }

    Ok(sources)
}

/// Load LoTSS sources from a VOTable XML string (DR3 VO Cone Search response).
///
/// The `xml` argument should be the raw response body from a VO Cone Search
/// request to `https://vo.astron.nl/lotss_dr3/q/src_cone/scs.xml`.
#[cfg(feature = "fits")]
pub fn load_from_votable(xml: &str, release: LoTSSRelease) -> Result<Vec<LoTSSSource>, FetchError> {
    use crate::formats::votable::parse_votable;

    let rows = parse_votable(xml, LOTSS_FITS_COLUMNS)?;
    let mut sources = Vec::with_capacity(rows.len());

    for row in rows {
        if let Some(src) = parse_row_from_votable(&row, release) {
            sources.push(src);
        }
    }

    Ok(sources)
}

// ---- Internal helpers --------------------------------------------------------

#[cfg(feature = "fits")]
fn parse_row_from_votable(
    row: &HashMap<String, String>,
    release: LoTSSRelease,
) -> Option<LoTSSSource> {
    let get = |key: &str| -> &str {
        row.get(key).map(|s| s.as_str()).unwrap_or("")
    };
    let parse_f32 = |key: &str| -> f32 {
        get(key).parse::<f32>().unwrap_or(f32::NAN)
    };
    let parse_f64 = |key: &str| -> f64 {
        get(key).parse::<f64>().unwrap_or(f64::NAN)
    };
    let parse_opt_f32 = |key: &str| -> Option<f32> {
        let v: f32 = get(key).parse().ok()?;
        if v.is_finite() { Some(v) } else { None }
    };

    let source_name = get("SOURCE_NAME").to_string();
    if source_name.is_empty() {
        return None;
    }

    let structure_code = get("S_CODE").chars().next().unwrap_or('S');
    let resolved_str = get("RESOLVED");
    let resolved = resolved_str.eq_ignore_ascii_case("R") || resolved_str == "1" || resolved_str == "true";

    Some(LoTSSSource {
        source_name,
        ra_deg: parse_f64("RA"),
        dec_deg: parse_f64("DEC"),
        flux_mjy: parse_f32("TOTAL_FLUX"),
        flux_err_mjy: parse_f32("E_TOTAL_FLUX"),
        peak_flux: parse_f32("PEAK_FLUX"),
        local_rms: parse_f32("ISL_RMS"),
        spectral_index: parse_opt_f32("SPECTRAL_INDEX"),
        spectral_index_err: parse_opt_f32("E_SPECTRAL_INDEX"),
        resolved,
        structure_code,
        maj_arcsec: parse_f32("MAJ"),
        min_arcsec: parse_f32("MIN"),
        pa_deg: parse_f32("PA"),
        release,
    })
}

#[cfg(all(test, feature = "fits"))]
mod tests {
    use super::*;

    #[test]
    fn release_labels() {
        assert_eq!(LoTSSRelease::DR1.label(), "DR1");
        assert_eq!(LoTSSRelease::DR2.label(), "DR2");
        assert_eq!(LoTSSRelease::DR3.label(), "DR3");
    }

    #[test]
    fn parse_votable_minimal() {
        let xml = r#"<?xml version="1.0"?>
<VOTABLE version="1.4">
  <RESOURCE>
    <TABLE>
      <FIELD name="Source_Name" datatype="char"/>
      <FIELD name="RA" datatype="double"/>
      <FIELD name="DEC" datatype="double"/>
      <FIELD name="Total_flux" datatype="float"/>
      <FIELD name="E_Total_flux" datatype="float"/>
      <FIELD name="Peak_flux" datatype="float"/>
      <FIELD name="Isl_rms" datatype="float"/>
      <FIELD name="Spectral_index" datatype="float"/>
      <FIELD name="E_spectral_index" datatype="float"/>
      <FIELD name="Resolved" datatype="char"/>
      <FIELD name="S_Code" datatype="char"/>
      <FIELD name="Maj" datatype="float"/>
      <FIELD name="Min" datatype="float"/>
      <FIELD name="PA" datatype="float"/>
      <DATA>
        <TABLEDATA>
          <TR>
            <TD>ILTJ000001.1+123456</TD><TD>0.0042</TD><TD>12.5820</TD>
            <TD>3.4</TD><TD>0.1</TD><TD>3.1</TD><TD>0.08</TD>
            <TD>-0.8</TD><TD>0.1</TD><TD>R</TD><TD>S</TD>
            <TD>6.1</TD><TD>5.2</TD><TD>45.0</TD>
          </TR>
        </TABLEDATA>
      </DATA>
    </TABLE>
  </RESOURCE>
</VOTABLE>"#;

        let sources = load_from_votable(xml, LoTSSRelease::DR3).unwrap();
        assert_eq!(sources.len(), 1);
        let s = &sources[0];
        assert_eq!(s.source_name, "ILTJ000001.1+123456");
        assert!((s.ra_deg - 0.0042).abs() < 1e-6);
        assert!(s.resolved);
        assert_eq!(s.structure_code, 'S');
        assert!((s.flux_mjy - 3.4).abs() < 0.01);
        assert_eq!(s.release, LoTSSRelease::DR3);
    }
}

//! Generic FITS binary table (BINTABLE HDU) reader.
//!
//! LoTSS DR1 and DR2 primary catalogs are distributed as FITS BINTABLE
//! extensions.  This module provides a column-selective reader that returns
//! rows as `HashMap<String, FitsValue>` so catalog parsers can consume them
//! without depending on the full fitsio API.
//!
//! Feature-gated behind `data_core/fits` to avoid linking cfitsio on hosts
//! that lack the native library.
//!
//! Reference: FITS Standard 4.0 (NOST 100-2.0), Sec. 7.3 (BINTABLE extension).

use crate::fetcher::FetchError;
use std::{collections::HashMap, path::Path};

/// A typed value extracted from a single FITS BINTABLE cell.
#[derive(Debug, Clone, PartialEq)]
pub enum FitsValue {
    Str(String),
    F32(f32),
    F64(f64),
    I32(i32),
    I64(i64),
    Bool(bool),
}

impl FitsValue {
    /// Return the inner `f64` if this is `F32` or `F64`, else `None`.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            FitsValue::F64(v) => Some(*v),
            FitsValue::F32(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// Return the inner `f32` if this is `F32`, else `None`.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            FitsValue::F32(v) => Some(*v),
            _ => None,
        }
    }

    /// Return the inner `&str` if this is `Str`, else `None`.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            FitsValue::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Return the inner bool if this is `Bool`, else `None`.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            FitsValue::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

/// Read selected columns from the first BINTABLE extension HDU of a FITS file.
///
/// `columns` is a slice of column names (case-insensitive match against TTYPE
/// keywords).  Pass an empty slice to read **all** columns.
///
/// Returns one `HashMap` per row; keys are the normalised (trimmed, uppercased)
/// TTYPE column names.
///
/// # Strategy
/// Reads each column as a typed `Vec<T>` in one cfitsio call, then transposes
/// to row-oriented HashMaps.  This minimises cfitsio round-trips vs. cell-by-cell
/// reads.
///
/// # Errors
/// Returns `FetchError::Validation` on FITS I/O or column-access failures.
#[cfg(feature = "fits")]
pub fn read_fits_table(
    path: &Path,
    columns: &[&str],
) -> Result<Vec<HashMap<String, FitsValue>>, FetchError> {
    use fitsio::{FitsFile, hdu::HduInfo, tables::ColumnDataType};

    let mut fptr = FitsFile::open(path)
        .map_err(|e| FetchError::Validation(format!("FITS open {}: {}", path.display(), e)))?;

    // Find the first table HDU (extension HDU 1+).
    // HDU 0 is always the PRIMARY (image/empty) HDU; tables start at index 1.
    let num_hdus = {
        let mut count = 0usize;
        for _ in fptr.iter() {
            count += 1;
        }
        count
    };

    let mut table_hdu_idx: Option<usize> = None;
    for idx in 1..num_hdus {
        let hdu = fptr
            .hdu(idx)
            .map_err(|e| FetchError::Validation(format!("hdu {}: {}", idx, e)))?;
        if matches!(hdu.info, HduInfo::TableInfo { .. }) {
            table_hdu_idx = Some(idx);
            break;
        }
    }

    let table_idx = table_hdu_idx.ok_or_else(|| {
        FetchError::Validation("No BINTABLE HDU found in FITS file".to_string())
    })?;

    let table_hdu = fptr
        .hdu(table_idx)
        .map_err(|e| FetchError::Validation(format!("hdu {}: {}", table_idx, e)))?;

    let (all_descs, num_rows) = match table_hdu.info {
        HduInfo::TableInfo {
            ref column_descriptions,
            num_rows,
        } => (column_descriptions.clone(), num_rows),
        _ => return Err(FetchError::Validation("Expected TableInfo HDU".to_string())),
    };

    // Map uppercase name -> index for quick lookup.
    let name_to_idx: HashMap<String, usize> = all_descs
        .iter()
        .enumerate()
        .map(|(i, d)| (d.name.trim().to_uppercase(), i))
        .collect();

    // Resolve which columns to extract (preserve input order; empty = all).
    let want: Vec<(String, usize)> = if columns.is_empty() {
        all_descs
            .iter()
            .enumerate()
            .map(|(i, d)| (d.name.trim().to_uppercase(), i))
            .collect()
    } else {
        columns
            .iter()
            .filter_map(|c| {
                let key = c.trim().to_uppercase();
                name_to_idx.get(&key).map(|&i| (key, i))
            })
            .collect()
    };

    // Build column data vectors (one Vec<FitsValue> per column).
    // We need a fresh HDU borrow for each read_col call.
    let mut col_data: Vec<(String, Vec<FitsValue>)> = Vec::with_capacity(want.len());

    for (col_name, col_idx) in &want {
        let desc = &all_descs[*col_idx];
        let fits_name = desc.name.trim();
        let typ = desc.data_type.typ;

        // Re-acquire the HDU handle per call -- fitsio FitsHdu borrows fptr mutably.
        let hdu = fptr
            .hdu(table_idx)
            .map_err(|e| FetchError::Validation(format!("hdu re-acquire: {}", e)))?;

        let values: Vec<FitsValue> = match typ {
            ColumnDataType::Float => {
                let v: Vec<f32> = hdu
                    .read_col(&mut fptr, fits_name)
                    .map_err(|e| FetchError::Validation(format!("col {}: {}", fits_name, e)))?;
                v.into_iter().map(FitsValue::F32).collect()
            }
            ColumnDataType::Double => {
                let v: Vec<f64> = hdu
                    .read_col(&mut fptr, fits_name)
                    .map_err(|e| FetchError::Validation(format!("col {}: {}", fits_name, e)))?;
                v.into_iter().map(FitsValue::F64).collect()
            }
            ColumnDataType::Int => {
                let v: Vec<i32> = hdu
                    .read_col(&mut fptr, fits_name)
                    .map_err(|e| FetchError::Validation(format!("col {}: {}", fits_name, e)))?;
                v.into_iter().map(FitsValue::I32).collect()
            }
            ColumnDataType::Long => {
                let v: Vec<i64> = hdu
                    .read_col(&mut fptr, fits_name)
                    .map_err(|e| FetchError::Validation(format!("col {}: {}", fits_name, e)))?;
                v.into_iter().map(FitsValue::I64).collect()
            }
            // Bool (FITS TLOGICAL 'L') -- read as i32; cfitsio maps T->1, F->0.
            ColumnDataType::Bool => {
                let v: Vec<i32> = hdu
                    .read_col(&mut fptr, fits_name)
                    .map_err(|e| FetchError::Validation(format!("col {} (bool as i32): {}", fits_name, e)))?;
                v.into_iter().map(|x| FitsValue::Bool(x != 0)).collect()
            }
            // Short integer columns.
            ColumnDataType::Short => {
                let v: Vec<i32> = hdu
                    .read_col(&mut fptr, fits_name)
                    .map_err(|e| FetchError::Validation(format!("col {} (short as i32): {}", fits_name, e)))?;
                v.into_iter().map(FitsValue::I32).collect()
            }
            // String, Text, and any other character-based columns.
            _ => {
                let v: Vec<String> = hdu
                    .read_col(&mut fptr, fits_name)
                    .map_err(|e| FetchError::Validation(format!("col {} (str): {}", fits_name, e)))?;
                v.into_iter()
                    .map(|s| FitsValue::Str(s.trim().to_string()))
                    .collect()
            }
        };

        col_data.push((col_name.clone(), values));
    }

    // Transpose column vectors to row HashMaps.
    let mut rows: Vec<HashMap<String, FitsValue>> = (0..num_rows)
        .map(|_| HashMap::with_capacity(col_data.len()))
        .collect();

    for (col_name, values) in col_data {
        for (row_idx, val) in values.into_iter().enumerate() {
            if row_idx < rows.len() {
                rows[row_idx].insert(col_name.clone(), val);
            }
        }
    }

    Ok(rows)
}

#[cfg(all(test, feature = "fits"))]
mod tests {
    use super::*;

    #[test]
    fn fits_value_conversions() {
        assert_eq!(FitsValue::F32(1.5).as_f64(), Some(1.5_f64));
        assert_eq!(FitsValue::F64(2.0).as_f64(), Some(2.0_f64));
        assert_eq!(FitsValue::Str("hi".to_string()).as_str(), Some("hi"));
        assert_eq!(FitsValue::Bool(true).as_bool(), Some(true));
        assert_eq!(FitsValue::I32(7).as_f64(), None);
    }
}

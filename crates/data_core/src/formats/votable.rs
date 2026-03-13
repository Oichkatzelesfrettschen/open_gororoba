//! Lightweight VOTable XML parser.
//!
//! VOTable is the IVOA standard response format for VO Cone Search and TAP
//! queries.  LoTSS DR3 returns VOTable XML for cone-search requests against
//! the ASTRON VO service.  This module provides a minimal pull parser that
//! extracts tabular data without loading a full DOM.
//!
//! Only the TABLEDATA serialization is supported (binary VOTable is a separate
//! format not used by ASTRON for LoTSS).
//!
//! Reference: IVOA VOTable 1.5 (2023), Sec. 4.7 (TABLEDATA).
//!
//! Feature-gated behind `data_core/fits` alongside the FITS reader.

use crate::fetcher::FetchError;
use std::collections::HashMap;

#[cfg(feature = "fits")]
use quick_xml::{
    Reader,
    events::Event,
};

/// Parse a VOTable XML document and extract selected columns.
///
/// `columns` is a slice of column names to extract (case-insensitive match
/// against `<FIELD name="...">` attributes).  Pass an empty slice to extract
/// **all** columns.
///
/// Returns one `HashMap` per data row; keys are the normalised (trimmed,
/// uppercased) FIELD names.  Cell values are returned as strings; callers are
/// responsible for parsing numeric types.
///
/// # Errors
/// Returns `FetchError::Validation` on XML parse failures or structural
/// violations of the VOTable schema.
#[cfg(feature = "fits")]
pub fn parse_votable(
    xml: &str,
    columns: &[&str],
) -> Result<Vec<HashMap<String, String>>, FetchError> {
    let want: Vec<String> = columns
        .iter()
        .map(|c| c.trim().to_uppercase())
        .collect();

    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);

    // Phase 1: collect FIELD names in declaration order.
    let mut field_names: Vec<String> = Vec::new();
    let mut in_table_data = false;
    let mut current_row: Option<Vec<String>> = None;
    let mut current_td: Option<String> = None;
    let mut rows: Vec<HashMap<String, String>> = Vec::new();

    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                let tag = std::str::from_utf8(e.name().as_ref())
                    .unwrap_or("")
                    .to_uppercase();
                match tag.as_str() {
                    "FIELD" => {
                        if !in_table_data {
                            let field_name = e
                                .attributes()
                                .filter_map(|a| a.ok())
                                .find(|a| {
                                    std::str::from_utf8(a.key.as_ref())
                                        .map(|k| k.eq_ignore_ascii_case("name"))
                                        .unwrap_or(false)
                                })
                                .and_then(|a| {
                                    a.unescape_value()
                                        .ok()
                                        .map(|v| v.trim().to_uppercase())
                                })
                                .unwrap_or_default();
                            field_names.push(field_name);
                        }
                    }
                    "TABLEDATA" => {
                        in_table_data = true;
                    }
                    "TR" if in_table_data => {
                        current_row = Some(Vec::with_capacity(field_names.len()));
                    }
                    "TD" if in_table_data => {
                        current_td = Some(String::new());
                    }
                    _ => {}
                }
            }
            Ok(Event::Empty(ref e)) => {
                let tag = std::str::from_utf8(e.name().as_ref())
                    .unwrap_or("")
                    .to_uppercase();
                if tag == "FIELD" && !in_table_data {
                    let field_name = e
                        .attributes()
                        .filter_map(|a| a.ok())
                        .find(|a| {
                            std::str::from_utf8(a.key.as_ref())
                                .map(|k| k.eq_ignore_ascii_case("name"))
                                .unwrap_or(false)
                        })
                        .and_then(|a| {
                            a.unescape_value()
                                .ok()
                                .map(|v| v.trim().to_uppercase())
                        })
                        .unwrap_or_default();
                    field_names.push(field_name);
                }
                if tag == "TD" && in_table_data {
                    // Empty <TD/> -- push empty string into current row.
                    if let Some(row) = current_row.as_mut() {
                        row.push(String::new());
                    }
                }
            }
            Ok(Event::End(ref e)) => {
                let tag = std::str::from_utf8(e.name().as_ref())
                    .unwrap_or("")
                    .to_uppercase();
                match tag.as_str() {
                    "TD" if in_table_data => {
                        if let (Some(row), Some(td)) = (current_row.as_mut(), current_td.take()) {
                            row.push(td);
                        }
                    }
                    "TR" if in_table_data => {
                        if let Some(cells) = current_row.take() {
                            let mut record = HashMap::with_capacity(cells.len());
                            for (idx, cell) in cells.into_iter().enumerate() {
                                if let Some(col) = field_names.get(idx) {
                                    if want.is_empty() || want.contains(col) {
                                        record.insert(col.clone(), cell);
                                    }
                                }
                            }
                            rows.push(record);
                        }
                    }
                    "TABLEDATA" => {
                        in_table_data = false;
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(ref e)) if in_table_data && current_td.is_some() => {
                if let Ok(text) = e.decode() {
                    if let Some(td) = current_td.as_mut() {
                        td.push_str(&text);
                    }
                }
            }
            Ok(Event::CData(ref e)) if in_table_data && current_td.is_some() => {
                if let Ok(text) = e.decode() {
                    if let Some(td) = current_td.as_mut() {
                        td.push_str(&text);
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(FetchError::Validation(format!("VOTable XML parse error: {}", e)));
            }
            _ => {}
        }
        buf.clear();
    }

    Ok(rows)
}

/// Extract FIELD names from a VOTable header (the `<FIELD name="...">` elements
/// in declaration order).  Useful for inspecting column layout before parsing.
#[cfg(feature = "fits")]
pub fn votable_field_names(xml: &str) -> Result<Vec<String>, FetchError> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);

    let mut names: Vec<String> = Vec::new();
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) | Ok(Event::Empty(ref e)) => {
                let tag = std::str::from_utf8(e.name().as_ref())
                    .unwrap_or("")
                    .to_uppercase();
                if tag == "FIELD" {
                    let field_name = e
                        .attributes()
                        .filter_map(|a| a.ok())
                        .find(|a| {
                            std::str::from_utf8(a.key.as_ref())
                                .map(|k| k.eq_ignore_ascii_case("name"))
                                .unwrap_or(false)
                        })
                        .and_then(|a| {
                            a.unescape_value()
                                .ok()
                                .map(|v| v.trim().to_uppercase())
                        })
                        .unwrap_or_default();
                    names.push(field_name);
                }
                if tag == "TABLEDATA" {
                    break;
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(FetchError::Validation(format!("VOTable XML parse error: {}", e)));
            }
            _ => {}
        }
        buf.clear();
    }

    Ok(names)
}

#[cfg(all(test, feature = "fits"))]
mod tests {
    use super::*;

    const MINIMAL_VOTABLE: &str = r#"<?xml version="1.0"?>
<VOTABLE version="1.4">
  <RESOURCE>
    <TABLE>
      <FIELD name="Source_Name" datatype="char"/>
      <FIELD name="RA" datatype="double"/>
      <FIELD name="DEC" datatype="double"/>
      <FIELD name="Total_flux" datatype="float"/>
      <DATA>
        <TABLEDATA>
          <TR><TD>ILT J000001+123456</TD><TD>0.0042</TD><TD>12.5820</TD><TD>3.4</TD></TR>
          <TR><TD>ILT J000002+234567</TD><TD>0.0421</TD><TD>23.7650</TD><TD>7.1</TD></TR>
        </TABLEDATA>
      </DATA>
    </TABLE>
  </RESOURCE>
</VOTABLE>"#;

    #[test]
    fn parse_all_columns() {
        let rows = parse_votable(MINIMAL_VOTABLE, &[]).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["SOURCE_NAME"], "ILT J000001+123456");
        assert_eq!(rows[0]["RA"], "0.0042");
        assert_eq!(rows[1]["TOTAL_FLUX"], "7.1");
    }

    #[test]
    fn parse_selected_columns() {
        let rows = parse_votable(MINIMAL_VOTABLE, &["Source_Name", "RA"]).unwrap();
        assert_eq!(rows.len(), 2);
        assert!(rows[0].contains_key("SOURCE_NAME"));
        assert!(rows[0].contains_key("RA"));
        assert!(!rows[0].contains_key("TOTAL_FLUX"));
    }

    #[test]
    fn field_names_roundtrip() {
        let names = votable_field_names(MINIMAL_VOTABLE).unwrap();
        assert_eq!(names, ["SOURCE_NAME", "RA", "DEC", "TOTAL_FLUX"]);
    }
}

//! Fallback PDF extraction using pure-Rust crates (lopdf + pdf-extract).
//!
//! This backend requires NO native C library, making it suitable for
//! CI runners, WASM, and other constrained environments.
//! Text extraction uses pdf-extract; image extraction uses lopdf directly.
//! Does NOT support positioned text or page rendering.

use std::path::Path;

use crate::{DocpipeError, ImageFormat, PageText, PdfImage, Result};

/// Extract all text from a PDF file, segmented by page.
pub fn extract_text(path: &Path) -> Result<Vec<PageText>> {
    let bytes = std::fs::read(path)?;
    let full_text = pdf_extract::extract_text_from_mem(&bytes)
        .map_err(|e| DocpipeError::PdfParse(format!("{e}")))?;

    // pdf-extract returns one big string; split by form-feed (page break)
    let pages: Vec<PageText> = if full_text.contains('\u{000C}') {
        full_text
            .split('\u{000C}')
            .enumerate()
            .map(|(i, text)| PageText {
                page_num: i + 1,
                text: text.trim().to_string(),
            })
            .filter(|p| !p.text.is_empty())
            .collect()
    } else {
        vec![PageText {
            page_num: 1,
            text: full_text.trim().to_string(),
        }]
    };

    if pages.is_empty() {
        return Err(DocpipeError::Extraction(
            "no text extracted from PDF (fallback)".into(),
        ));
    }

    Ok(pages)
}

/// Extract text as a single concatenated string.
pub fn extract_full_text(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path)?;
    pdf_extract::extract_text_from_mem(&bytes).map_err(|e| DocpipeError::PdfParse(format!("{e}")))
}

/// Get page count using lopdf.
pub fn page_count(path: &Path) -> Result<usize> {
    let doc = lopdf::Document::load(path).map_err(|e| DocpipeError::PdfParse(format!("{e}")))?;
    Ok(doc.get_pages().len())
}

/// Extract images from a PDF using lopdf object traversal.
pub fn extract_images(path: &Path) -> Result<Vec<PdfImage>> {
    let doc = lopdf::Document::load(path).map_err(|e| DocpipeError::PdfParse(format!("{e}")))?;

    let mut images = Vec::new();
    let pages = doc.get_pages();

    for (&page_num, &page_id) in &pages {
        let page_dict = match doc.get_dictionary(page_id) {
            Ok(d) => d,
            Err(_) => continue,
        };

        let resources = match page_dict.get(b"Resources") {
            Ok(r) => match doc.dereference(r) {
                Ok((_, obj)) => obj.clone(),
                Err(_) => continue,
            },
            Err(_) => continue,
        };

        let resources_dict = match resources.as_dict() {
            Ok(d) => d,
            Err(_) => continue,
        };

        let xobjects = match resources_dict.get(b"XObject") {
            Ok(x) => match doc.dereference(x) {
                Ok((_, obj)) => obj.clone(),
                Err(_) => continue,
            },
            Err(_) => continue,
        };

        let xobj_dict = match xobjects.as_dict() {
            Ok(d) => d,
            Err(_) => continue,
        };

        for (_name, xobj_ref) in xobj_dict.iter() {
            let (obj_id, xobj) = match doc.dereference(xobj_ref) {
                Ok(pair) => pair,
                Err(_) => continue,
            };

            let stream = match xobj.as_stream() {
                Ok(s) => s,
                Err(_) => continue,
            };

            let subtype = stream
                .dict
                .get(b"Subtype")
                .ok()
                .and_then(|s| s.as_name().ok())
                .unwrap_or(b"");
            if subtype != b"Image" {
                continue;
            }

            let width = stream
                .dict
                .get(b"Width")
                .ok()
                .and_then(|w| w.as_i64().ok())
                .unwrap_or(0) as u32;
            let height = stream
                .dict
                .get(b"Height")
                .ok()
                .and_then(|h| h.as_i64().ok())
                .unwrap_or(0) as u32;

            let filter_name: &[u8] = stream
                .dict
                .get(b"Filter")
                .ok()
                .and_then(|f| f.as_name().ok())
                .unwrap_or(b"");

            let format = if filter_name == b"DCTDecode" {
                ImageFormat::Jpeg
            } else if filter_name == b"FlateDecode" {
                ImageFormat::Png
            } else {
                ImageFormat::Raw
            };

            let data = if let Some(obj_id) = obj_id {
                match doc.get_object(obj_id) {
                    Ok(lopdf::Object::Stream(ref s)) => s.content.clone(),
                    _ => stream.content.clone(),
                }
            } else {
                stream.content.clone()
            };

            if !data.is_empty() && width > 0 && height > 0 {
                images.push(PdfImage {
                    page_num: page_num as usize,
                    data,
                    format,
                    width,
                    height,
                });
            }
        }
    }

    Ok(images)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn test_pdf_path() -> Option<PathBuf> {
        let path = PathBuf::from("../../papers/pdf/de_marrais_2000_math0011260.pdf");
        if path.exists() {
            Some(path)
        } else {
            None
        }
    }

    #[test]
    fn test_fallback_extract_text() {
        let Some(path) = test_pdf_path() else {
            eprintln!("skipping: test PDF not found");
            return;
        };
        let pages = extract_text(&path).expect("fallback extraction should succeed");
        assert!(!pages.is_empty());
    }

    #[test]
    fn test_fallback_page_count() {
        let Some(path) = test_pdf_path() else {
            eprintln!("skipping: test PDF not found");
            return;
        };
        let count = page_count(&path).expect("page count should succeed");
        assert!(count > 0);
    }

    #[test]
    fn test_fallback_extract_images() {
        let Some(path) = test_pdf_path() else {
            eprintln!("skipping: test PDF not found");
            return;
        };
        let result = extract_images(&path);
        assert!(result.is_ok());
    }
}

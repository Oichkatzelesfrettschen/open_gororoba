//! PDF extraction dispatch layer.
//!
//! Routes to pdfium_backend (primary) or fallback_backend depending on
//! which features are compiled. The pdfium backend provides richer output
//! (positioned text, native image extraction, page rendering) while
//! the fallback works in pure Rust with no native dependencies.

use std::path::Path;

use crate::{PageText, PdfImage, Result};

// DocpipeError only needed when no backend is enabled (compile-time error messages).
#[cfg(not(any(feature = "pdfium", feature = "fallback")))]
use crate::DocpipeError;

/// Extract text from each page of a PDF.
///
/// Uses pdfium when available, falls back to pdf-extract/lopdf.
pub fn extract_text(path: &Path) -> Result<Vec<PageText>> {
    #[cfg(feature = "pdfium")]
    {
        match crate::pdfium_backend::extract_text(path) {
            Ok(pages) => Ok(pages),
            Err(_pdfium_err) => {
                #[cfg(feature = "fallback")]
                {
                    crate::fallback_backend::extract_text(path)
                }
                #[cfg(not(feature = "fallback"))]
                {
                    Err(_pdfium_err)
                }
            }
        }
    }

    #[cfg(all(feature = "fallback", not(feature = "pdfium")))]
    {
        crate::fallback_backend::extract_text(path)
    }

    #[cfg(not(any(feature = "pdfium", feature = "fallback")))]
    {
        let _ = path;
        Err(DocpipeError::Extraction(
            "no PDF backend enabled: compile with 'pdfium' or 'fallback' feature".into(),
        ))
    }
}

/// Extract text as a single concatenated string.
pub fn extract_full_text(path: &Path) -> Result<String> {
    #[cfg(feature = "pdfium")]
    {
        match crate::pdfium_backend::extract_full_text(path) {
            Ok(text) => Ok(text),
            Err(_pdfium_err) => {
                #[cfg(feature = "fallback")]
                {
                    crate::fallback_backend::extract_full_text(path)
                }
                #[cfg(not(feature = "fallback"))]
                {
                    Err(_pdfium_err)
                }
            }
        }
    }

    #[cfg(all(feature = "fallback", not(feature = "pdfium")))]
    {
        crate::fallback_backend::extract_full_text(path)
    }

    #[cfg(not(any(feature = "pdfium", feature = "fallback")))]
    {
        let _ = path;
        Err(DocpipeError::Extraction("no PDF backend enabled".into()))
    }
}

/// Get the page count from a PDF.
pub fn page_count(path: &Path) -> Result<usize> {
    #[cfg(feature = "pdfium")]
    {
        match crate::pdfium_backend::page_count(path) {
            Ok(count) => Ok(count),
            Err(_pdfium_err) => {
                #[cfg(feature = "fallback")]
                {
                    crate::fallback_backend::page_count(path)
                }
                #[cfg(not(feature = "fallback"))]
                {
                    Err(_pdfium_err)
                }
            }
        }
    }

    #[cfg(all(feature = "fallback", not(feature = "pdfium")))]
    {
        crate::fallback_backend::page_count(path)
    }

    #[cfg(not(any(feature = "pdfium", feature = "fallback")))]
    {
        let _ = path;
        Err(DocpipeError::Extraction("no PDF backend enabled".into()))
    }
}

/// Extract images from a PDF.
pub fn extract_images(path: &Path) -> Result<Vec<PdfImage>> {
    #[cfg(feature = "pdfium")]
    {
        match crate::pdfium_backend::extract_images(path) {
            Ok(images) => Ok(images),
            Err(_pdfium_err) => {
                #[cfg(feature = "fallback")]
                {
                    crate::fallback_backend::extract_images(path)
                }
                #[cfg(not(feature = "fallback"))]
                {
                    Err(_pdfium_err)
                }
            }
        }
    }

    #[cfg(all(feature = "fallback", not(feature = "pdfium")))]
    {
        crate::fallback_backend::extract_images(path)
    }

    #[cfg(not(any(feature = "pdfium", feature = "fallback")))]
    {
        let _ = path;
        Err(DocpipeError::Extraction("no PDF backend enabled".into()))
    }
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
    fn test_extract_text_returns_nonempty() {
        let Some(path) = test_pdf_path() else {
            eprintln!("skipping: test PDF not found");
            return;
        };
        let pages = extract_text(&path).expect("extraction should succeed");
        assert!(!pages.is_empty(), "should extract at least one page");
        assert!(
            !pages[0].text.is_empty(),
            "first page should have text content"
        );
    }

    #[test]
    fn test_page_count() {
        let Some(path) = test_pdf_path() else {
            eprintln!("skipping: test PDF not found");
            return;
        };
        let count = page_count(&path).expect("page count should succeed");
        assert!(count > 0, "PDF should have at least one page");
    }

    #[test]
    fn test_extract_full_text() {
        let Some(path) = test_pdf_path() else {
            eprintln!("skipping: test PDF not found");
            return;
        };
        let text = extract_full_text(&path).expect("full text extraction should succeed");
        assert!(text.len() > 100, "full text should be substantial");
    }
}

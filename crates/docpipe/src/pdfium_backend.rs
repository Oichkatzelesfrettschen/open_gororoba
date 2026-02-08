//! Primary PDF extraction backend using pdfium-render.
//!
//! pdfium-render wraps Google's PDFium (Chrome's PDF engine), providing
//! high-fidelity text extraction with character positions and font info,
//! native image extraction from page objects, and page rendering to bitmaps.
//! Requires libpdfium.so at runtime (available as `libpdfium-nojs` on Arch).

use std::path::Path;

use pdfium_render::prelude::*;

use crate::{DocpipeError, ImageFormat, PageText, PdfImage, PositionedText, Result};

/// Initialize the Pdfium library.
///
/// Tries system library path first, then current directory.
fn init_pdfium() -> Result<Pdfium> {
    let bindings = Pdfium::bind_to_system_library()
        .or_else(|_| {
            Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path("./"))
        })
        .map_err(|e| DocpipeError::PdfParse(format!("failed to load pdfium: {e}")))?;
    Ok(Pdfium::new(bindings))
}

/// Extract text from each page of a PDF using pdfium.
pub fn extract_text(path: &Path) -> Result<Vec<PageText>> {
    let pdfium = init_pdfium()?;
    let doc = pdfium
        .load_pdf_from_file(path, None)
        .map_err(|e| DocpipeError::PdfParse(format!("{e}")))?;

    let mut pages = Vec::new();
    for (i, page) in doc.pages().iter().enumerate() {
        let text_page = page
            .text()
            .map_err(|e| DocpipeError::PdfParse(format!("page {}: {e}", i + 1)))?;
        let page_text = text_page.all();
        if !page_text.trim().is_empty() {
            pages.push(PageText {
                page_num: i + 1,
                text: page_text,
            });
        }
    }

    if pages.is_empty() {
        return Err(DocpipeError::Extraction(
            "no text extracted from PDF via pdfium".into(),
        ));
    }

    Ok(pages)
}

/// Extract text as a single concatenated string.
pub fn extract_full_text(path: &Path) -> Result<String> {
    let pages = extract_text(path)?;
    Ok(pages
        .iter()
        .map(|p| p.text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n"))
}

/// Extract text with character-level positioning information.
///
/// This is a pdfium-exclusive capability -- the pure-Rust fallback
/// cannot provide this level of detail. Uses page text objects to
/// get bounding boxes and font sizes.
pub fn extract_positioned_text(path: &Path) -> Result<Vec<PositionedText>> {
    let pdfium = init_pdfium()?;
    let doc = pdfium
        .load_pdf_from_file(path, None)
        .map_err(|e| DocpipeError::PdfParse(format!("{e}")))?;

    let mut segments = Vec::new();
    for (page_idx, page) in doc.pages().iter().enumerate() {
        for object in page.objects().iter() {
            if let Some(text_obj) = object.as_text_object() {
                let text: String = text_obj.text();
                if text.trim().is_empty() {
                    continue;
                }
                let bounds = object.bounds().map_err(|e| {
                    DocpipeError::PdfParse(format!("bounds error: {e}"))
                })?;
                let font_size: f32 = text_obj
                    .scaled_font_size()
                    .value;
                segments.push(PositionedText {
                    page_num: page_idx + 1,
                    text,
                    x: bounds.left().value as f64,
                    y: bounds.bottom().value as f64,
                    width: (bounds.right().value - bounds.left().value) as f64,
                    height: (bounds.top().value - bounds.bottom().value) as f64,
                    font_size: font_size as f64,
                });
            }
        }
    }

    Ok(segments)
}

/// Get page count via pdfium.
pub fn page_count(path: &Path) -> Result<usize> {
    let pdfium = init_pdfium()?;
    let doc = pdfium
        .load_pdf_from_file(path, None)
        .map_err(|e| DocpipeError::PdfParse(format!("{e}")))?;
    Ok(doc.pages().len() as usize)
}

/// Extract images embedded in the PDF using pdfium page objects.
///
/// Iterates over each page's object tree looking for image objects,
/// extracts them with their rendered bitmap data as PNG.
pub fn extract_images(path: &Path) -> Result<Vec<PdfImage>> {
    let pdfium = init_pdfium()?;
    let doc = pdfium
        .load_pdf_from_file(path, None)
        .map_err(|e| DocpipeError::PdfParse(format!("{e}")))?;

    let mut images = Vec::new();

    for (page_idx, page) in doc.pages().iter().enumerate() {
        for object in page.objects().iter() {
            if let Some(image_obj) = object.as_image_object() {
                // Get the rendered bitmap from pdfium as DynamicImage
                let img_result: std::result::Result<::image::DynamicImage, _> =
                    image_obj.get_raw_image();
                if let Ok(dyn_image) = img_result {
                    let width = dyn_image.width();
                    let height = dyn_image.height();
                    // Encode as PNG bytes
                    let mut buf = Vec::new();
                    let mut cursor = std::io::Cursor::new(&mut buf);
                    if dyn_image
                        .write_to(&mut cursor, ::image::ImageFormat::Png)
                        .is_ok()
                    {
                        images.push(PdfImage {
                            page_num: page_idx + 1,
                            data: buf,
                            format: ImageFormat::Png,
                            width,
                            height,
                        });
                    }
                }
            }
        }
    }

    Ok(images)
}

/// Render a single page to a PNG bitmap.
///
/// Useful for visual comparison, OCR preprocessing, or thumbnail generation.
pub fn render_page_to_png(path: &Path, page_num: usize, width: u16) -> Result<Vec<u8>> {
    let pdfium = init_pdfium()?;
    let doc = pdfium
        .load_pdf_from_file(path, None)
        .map_err(|e| DocpipeError::PdfParse(format!("{e}")))?;

    let page = doc
        .pages()
        .get(page_num.saturating_sub(1) as u16)
        .map_err(|e| DocpipeError::PdfParse(format!("page {page_num}: {e}")))?;

    let config = PdfRenderConfig::new()
        .set_target_width(width as i32)
        .set_maximum_height(4000);

    let bitmap = page
        .render_with_config(&config)
        .map_err(|e| DocpipeError::PdfParse(format!("render error: {e}")))?;

    let dyn_image: ::image::DynamicImage = bitmap.as_image();
    let mut buf = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut buf);
    dyn_image
        .write_to(&mut cursor, ::image::ImageFormat::Png)
        .map_err(|e| DocpipeError::Extraction(format!("PNG encode error: {e}")))?;

    Ok(buf)
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
    fn test_pdfium_extract_text() {
        let Some(path) = test_pdf_path() else {
            eprintln!("skipping: test PDF not found");
            return;
        };
        let pages = extract_text(&path).expect("pdfium text extraction should succeed");
        assert!(!pages.is_empty(), "should extract at least one page");
        let lower = pages[0].text.to_lowercase();
        assert!(
            lower.contains("box") || lower.contains("sedenion") || lower.contains("zero"),
            "text should contain expected CD keywords"
        );
    }

    #[test]
    fn test_pdfium_page_count() {
        let Some(path) = test_pdf_path() else {
            eprintln!("skipping: test PDF not found");
            return;
        };
        let count = page_count(&path).expect("page count should succeed");
        assert!(count > 0, "PDF should have at least one page");
    }

    #[test]
    fn test_pdfium_extract_images() {
        let Some(path) = test_pdf_path() else {
            eprintln!("skipping: test PDF not found");
            return;
        };
        // Should not panic even if no images found (math papers often have none)
        let result = extract_images(&path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_pdfium_positioned_text() {
        let Some(path) = test_pdf_path() else {
            eprintln!("skipping: test PDF not found");
            return;
        };
        let segments = extract_positioned_text(&path).expect("positioned text should succeed");
        assert!(!segments.is_empty(), "should get positioned text segments");
        // Verify we get font size > 0
        assert!(
            segments.iter().any(|s| s.font_size > 0.0),
            "at least some segments should have font size info"
        );
    }

    #[test]
    fn test_pdfium_render_page() {
        let Some(path) = test_pdf_path() else {
            eprintln!("skipping: test PDF not found");
            return;
        };
        let png_data = render_page_to_png(&path, 1, 800).expect("page render should succeed");
        assert!(png_data.len() > 100, "rendered PNG should have substantial data");
        // PNG magic bytes
        assert_eq!(&png_data[..4], b"\x89PNG");
    }
}

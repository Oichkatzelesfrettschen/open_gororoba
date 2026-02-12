//! TOML output serialization for extracted papers.
//!
//! Converts an ExtractedPaper into structured TOML with sections for
//! metadata, sections, equations, tables, and figures.

use std::path::Path;

use crate::{pdf, table, text, DocpipeError, ExtractedPaper, Figure, PaperMetadata, Result};

/// Extract a paper from a PDF and produce the full ExtractedPaper struct.
///
/// This is the main entry point for the extraction pipeline:
/// 1. Extract raw text via pdf-extract
/// 2. Parse metadata from first page
/// 3. Split into sections
/// 4. Extract equations
/// 5. Detect tables
/// 6. Extract images (metadata only in TOML)
pub fn extract_paper(
    pdf_path: &Path,
    metadata_override: Option<PaperMetadata>,
) -> Result<ExtractedPaper> {
    // Some PDFs are image-only/scanned and may not yield text; keep extracting
    // figures/tables metadata so we still emit a structured artifact.
    let pages = pdf::extract_text(pdf_path).unwrap_or_default();
    let full_text = pages
        .iter()
        .map(|p| p.text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");

    let metadata = metadata_override.unwrap_or_else(|| {
        let first = pages.first().map(|p| p.text.as_str()).unwrap_or("");
        text::extract_metadata(first)
    });

    let sections = if full_text.trim().is_empty() {
        Vec::new()
    } else {
        text::split_sections(&full_text)
    };
    let equations = if full_text.trim().is_empty() {
        Vec::new()
    } else {
        text::extract_equations(&full_text)
    };
    let tables = if pages.is_empty() {
        Vec::new()
    } else {
        table::detect_tables(&pages)
    };

    // Image extraction: just record metadata, don't embed raw bytes in TOML
    let images = pdf::extract_images(pdf_path).unwrap_or_default();
    let figures: Vec<Figure> = images
        .iter()
        .enumerate()
        .map(|(i, img)| Figure {
            label: Some(format!("fig:{}", i + 1)),
            caption: None,
            page_num: img.page_num,
            image_path: None, // populated later when saving to disk
        })
        .collect();

    if full_text.trim().is_empty() && figures.is_empty() {
        return Err(DocpipeError::Extraction(
            "no text or figures extracted from PDF".to_string(),
        ));
    }

    Ok(ExtractedPaper {
        metadata,
        sections,
        equations,
        tables,
        figures,
        full_text,
    })
}

/// Serialize an ExtractedPaper to TOML string.
pub fn paper_to_toml(paper: &ExtractedPaper) -> Result<String> {
    toml::to_string_pretty(paper).map_err(DocpipeError::from)
}

/// Extract a paper and write the TOML output to a file.
pub fn extract_and_write(
    pdf_path: &Path,
    output_dir: &Path,
    metadata_override: Option<PaperMetadata>,
) -> Result<ExtractedPaper> {
    let mut paper = extract_paper(pdf_path, metadata_override)?;

    // Persist extracted figure assets and wire relative image paths into paper.toml.
    let images = pdf::extract_images(pdf_path).unwrap_or_default();
    if !images.is_empty() {
        let image_dir = output_dir.join("images");
        let saved_paths = crate::image::save_images(&images, &image_dir)?;
        for (figure, image_path) in paper.figures.iter_mut().zip(saved_paths.iter()) {
            let relative = image_path
                .strip_prefix(output_dir)
                .unwrap_or(image_path)
                .to_string_lossy()
                .replace('\\', "/");
            figure.image_path = Some(relative);
        }
    }

    let toml_str = paper_to_toml(&paper)?;

    std::fs::create_dir_all(output_dir)?;
    let toml_path = output_dir.join("paper.toml");
    std::fs::write(&toml_path, &toml_str)?;

    // Write equations to separate .tex file if any exist
    if !paper.equations.is_empty() {
        let mut tex = String::new();
        for (i, eq) in paper.equations.iter().enumerate() {
            if eq.display {
                tex.push_str(&format!("% Equation {}\n", i + 1));
                if let Some(ref label) = eq.label {
                    tex.push_str(&format!("% Label: {label}\n"));
                }
                tex.push_str("\\begin{equation}\n");
                tex.push_str(&eq.latex);
                tex.push_str("\n\\end{equation}\n\n");
            }
        }
        if !tex.is_empty() {
            std::fs::write(output_dir.join("equations.tex"), &tex)?;
        }
    }

    // Write tables as CSV files.
    for (i, tbl) in paper.tables.iter().enumerate() {
        let mut csv_content = tbl.headers.join(",");
        csv_content.push('\n');
        for row in &tbl.rows {
            csv_content.push_str(&row.join(","));
            csv_content.push('\n');
        }
        std::fs::write(
            output_dir.join(format!("table_{}.csv", i + 1)),
            &csv_content,
        )?;
    }

    Ok(paper)
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
    fn test_extract_paper_roundtrip() {
        let Some(path) = test_pdf_path() else {
            eprintln!("skipping: test PDF not found");
            return;
        };
        let paper = extract_paper(&path, None).expect("extraction should succeed");
        assert!(!paper.full_text.is_empty());
        assert!(!paper.sections.is_empty());

        let toml_str = paper_to_toml(&paper).expect("TOML serialization should succeed");
        assert!(toml_str.contains("[metadata]"));
        assert!(toml_str.contains("title"));
    }

    #[test]
    fn test_extract_and_write() {
        let Some(path) = test_pdf_path() else {
            eprintln!("skipping: test PDF not found");
            return;
        };
        let tmp = tempfile::tempdir().unwrap();
        let paper =
            extract_and_write(&path, tmp.path(), None).expect("extract_and_write should succeed");

        assert!(tmp.path().join("paper.toml").exists());
        assert!(!paper.full_text.is_empty());
    }
}

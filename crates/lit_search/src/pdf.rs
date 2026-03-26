//! Scientific PDF extraction with math support (Rust port).
//!
//! Uses unpdf for high-quality LaTeX extraction and layout-aware parsing.
//! Falls back to basic text extraction if needed.

use anyhow::{Context, Result};
use std::path::Path;

pub struct PdfExtractor;

impl PdfExtractor {
    /// Extract PDF content to Markdown with math preservation.
    pub fn extract_to_markdown(path: impl AsRef<Path>) -> Result<String> {
        let path = path.as_ref();
        if !path.exists() {
            anyhow::bail!("PDF file not found: {:?}", path);
        }

        let markdown = unpdf::to_markdown(path)
            .with_context(|| format!("Failed to convert PDF to markdown: {:?}", path))?;

        Ok(markdown)
    }

    /// Batch extract multiple PDFs.
    pub fn batch_extract(paths: &[impl AsRef<Path>]) -> Vec<(String, Result<String>)> {
        paths
            .iter()
            .map(|p| {
                let path_str = p.as_ref().to_string_lossy().to_string();
                (path_str, Self::extract_to_markdown(p))
            })
            .collect()
    }
}

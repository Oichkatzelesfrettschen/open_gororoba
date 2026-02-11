//! docpipe: PDF text/image extraction and structured TOML output pipeline.
//!
//! Uses pdfium-render (Chrome's PDF engine) as the primary backend for
//! high-fidelity text extraction, image extraction, and page rendering.
//! Falls back to pure-Rust lopdf/pdf-extract when compiled with
//! `--features fallback` for environments without libpdfium.

#[cfg(feature = "pdfium")]
pub mod pdfium_backend;

#[cfg(feature = "fallback")]
pub mod fallback_backend;

pub mod equation_catalog;
pub mod image;
pub mod pdf;
pub mod table;
pub mod text;
pub mod toml_out;

use serde::{Deserialize, Serialize};

/// Errors that can occur during document processing.
#[derive(Debug, thiserror::Error)]
pub enum DocpipeError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("PDF parsing error: {0}")]
    PdfParse(String),

    #[error("TOML serialization error: {0}")]
    TomlSerialize(#[from] toml::ser::Error),

    #[error("CSV parse error: {0}")]
    Csv(#[from] csv::Error),

    #[error("extraction error: {0}")]
    Extraction(String),
}

pub type Result<T> = std::result::Result<T, DocpipeError>;

/// Text content from a single PDF page.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageText {
    pub page_num: usize,
    pub text: String,
}

/// A text segment with positional information (pdfium-only).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionedText {
    pub page_num: usize,
    pub text: String,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub font_size: f64,
}

/// An extracted image from a PDF.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdfImage {
    pub page_num: usize,
    #[serde(skip)]
    pub data: Vec<u8>,
    pub format: ImageFormat,
    pub width: u32,
    pub height: u32,
}

/// Image format extracted from PDF streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageFormat {
    Jpeg,
    Png,
    Raw,
}

/// A detected table in the text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Table {
    pub page_num: usize,
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

/// An extracted LaTeX equation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Equation {
    pub label: Option<String>,
    pub latex: String,
    pub display: bool,
}

/// A section of the paper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Section {
    pub number: Option<String>,
    pub title: String,
    pub text: String,
}

/// Metadata about the paper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaperMetadata {
    pub title: String,
    pub authors: Vec<String>,
    pub arxiv: Option<String>,
    pub year: Option<u32>,
    #[serde(rename = "abstract")]
    pub abstract_text: Option<String>,
}

/// A figure reference extracted from the paper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Figure {
    pub label: Option<String>,
    pub caption: Option<String>,
    pub page_num: usize,
    pub image_path: Option<String>,
}

/// Complete extracted paper content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedPaper {
    pub metadata: PaperMetadata,
    pub sections: Vec<Section>,
    pub equations: Vec<Equation>,
    pub tables: Vec<Table>,
    pub figures: Vec<Figure>,
    pub full_text: String,
}

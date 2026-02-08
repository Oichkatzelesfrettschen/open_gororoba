//! Image extraction convenience re-exports.
//!
//! The actual implementation lives in pdfium_backend or fallback_backend.
//! This module provides the dispatch via pdf::extract_images().
//! With pdfium, images are extracted as rendered bitmaps (PNG).
//! With the fallback, images are extracted as raw streams (JPEG/FlateDecode/Raw).

use std::path::Path;

use crate::{PdfImage, Result};

/// Save extracted images to disk.
///
/// Returns the list of file paths written.
pub fn save_images(images: &[PdfImage], output_dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    std::fs::create_dir_all(output_dir)?;
    let mut paths = Vec::new();

    for (i, img) in images.iter().enumerate() {
        let ext = match img.format {
            crate::ImageFormat::Jpeg => "jpg",
            crate::ImageFormat::Png => "png",
            crate::ImageFormat::Raw => "raw",
        };
        let filename = format!("image_p{}_{}.{}", img.page_num, i + 1, ext);
        let path = output_dir.join(filename);
        std::fs::write(&path, &img.data)?;
        paths.push(path);
    }

    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_save_images_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = save_images(&[], tmp.path()).unwrap();
        assert!(paths.is_empty());
    }
}

//! Shared HTTP download, SHA-256 validation, and disk caching infrastructure.
//!
//! Generalizes the fetch pattern from fetch_chime_frb.rs into reusable functions
//! that all catalog modules share.

use sha2::{Digest, Sha256};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum FetchError {
    #[error("HTTP request failed for {url}: {source}")]
    HttpError {
        url: String,
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("HTTP {status} from {url}")]
    HttpStatus { url: String, status: u16 },
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("Validation failed: {0}")]
    Validation(String),
    #[error("All download URLs exhausted for {dataset}")]
    AllUrlsFailed { dataset: String },
}

/// Configuration for dataset fetching.
#[derive(Debug, Clone)]
pub struct FetchConfig {
    /// Root directory for downloaded data (default: data/external).
    pub output_dir: PathBuf,
    /// Skip download if file already exists.
    pub skip_existing: bool,
    /// Verify SHA-256 checksums when available.
    pub verify_checksums: bool,
}

impl Default for FetchConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("data/external"),
            skip_existing: true,
            verify_checksums: true,
        }
    }
}

/// Download a URL to a file, returning the number of bytes written.
pub fn download_to_file(url: &str, path: &Path) -> Result<u64, FetchError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let response = ureq::get(url)
        .header("Accept", "text/csv,text/plain,application/octet-stream,*/*")
        .header("User-Agent", "gororoba-fetch/0.1 (research)")
        .call()
        .map_err(|e| FetchError::HttpError {
            url: url.to_string(),
            source: Box::new(e),
        })?;

    let status = response.status();
    if status != 200 {
        return Err(FetchError::HttpStatus {
            url: url.to_string(),
            status: status.into(),
        });
    }

    let mut reader = response.into_body().into_reader();
    let mut file = fs::File::create(path)?;
    let bytes = io::copy(&mut reader, &mut file)?;
    Ok(bytes)
}

/// Download a URL as a string (for small text files).
pub fn download_to_string(url: &str) -> Result<String, FetchError> {
    let response = ureq::get(url)
        .header("Accept", "text/csv,text/plain,*/*")
        .header("User-Agent", "gororoba-fetch/0.1 (research)")
        .call()
        .map_err(|e| FetchError::HttpError {
            url: url.to_string(),
            source: Box::new(e),
        })?;

    let status = response.status();
    if status != 200 {
        return Err(FetchError::HttpStatus {
            url: url.to_string(),
            status: status.into(),
        });
    }

    let body = response
        .into_body()
        .read_to_string()
        .map_err(|e| FetchError::Io(io::Error::other(e)))?;

    Ok(body)
}

/// Compute SHA-256 hash of a file, returning hex string.
pub fn compute_sha256(path: &Path) -> Result<String, io::Error> {
    let data = fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&data);
    let hash = hasher.finalize();
    Ok(format!("{:x}", hash))
}

/// Validate that data is not an HTML error page.
pub fn validate_not_html(data: &[u8]) -> Result<(), FetchError> {
    let prefix = std::str::from_utf8(&data[..data.len().min(512)]).unwrap_or("");
    let lower = prefix.to_lowercase();

    if lower.contains("<!doctype") || lower.contains("<html") || lower.contains("<script") {
        return Err(FetchError::Validation(
            "Response is HTML, not data".to_string(),
        ));
    }
    Ok(())
}

/// Try multiple URLs in sequence, returning the first successful download.
pub fn download_with_fallbacks(
    dataset_name: &str,
    urls: &[&str],
    output_path: &Path,
    skip_existing: bool,
) -> Result<PathBuf, FetchError> {
    if skip_existing && output_path.exists() {
        eprintln!(
            "  {} already exists at {}, skipping",
            dataset_name,
            output_path.display()
        );
        return Ok(output_path.to_path_buf());
    }

    for url in urls {
        eprintln!("  Downloading {} from {}...", dataset_name, url);
        match download_to_file(url, output_path) {
            Ok(bytes) => {
                // Validate it's not HTML
                let data = fs::read(output_path)?;
                if let Err(e) = validate_not_html(&data) {
                    eprintln!("  Validation failed: {}", e);
                    fs::remove_file(output_path).ok();
                    continue;
                }
                eprintln!("  Saved {} bytes to {}", bytes, output_path.display());
                if let Ok(hash) = compute_sha256(output_path) {
                    eprintln!("  SHA256: {}", hash);
                }
                return Ok(output_path.to_path_buf());
            }
            Err(e) => {
                eprintln!("  Failed: {}", e);
            }
        }
    }

    Err(FetchError::AllUrlsFailed {
        dataset: dataset_name.to_string(),
    })
}

/// Extract a tar.gz archive to a directory, returning the list of extracted paths.
pub fn extract_tar_gz(archive: &Path, output_dir: &Path) -> Result<Vec<PathBuf>, FetchError> {
    let file = fs::File::open(archive)?;
    let gz = flate2::read::GzDecoder::new(file);
    let mut tar = tar::Archive::new(gz);

    fs::create_dir_all(output_dir)?;

    let mut extracted = Vec::new();
    for entry in tar.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.to_path_buf();
        let dest = output_dir.join(&path);
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        entry.unpack(&dest)?;
        extracted.push(dest);
    }

    Ok(extracted)
}

/// Convert HEASARC pipe-delimited text to standard CSV.
///
/// W3Browse `displaymode=BatchDisplay` returns pipe-delimited rows wrapped in
/// `BatchStart`/`BatchEnd` markers with a `+---+---+` separator between
/// headers and data. This function strips those markers, converts pipes to
/// commas, and trims whitespace from each cell value.
pub fn convert_pipe_to_csv(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    for line in input.lines() {
        let trimmed = line.trim();
        // Skip empty lines, HTML fragments, batch markers, and separators
        if trimmed.is_empty()
            || trimmed.starts_with('<')
            || trimmed.starts_with("BatchStart")
            || trimmed.starts_with("BatchEnd")
            || trimmed.starts_with('+')
        {
            continue;
        }
        // Strip leading/trailing pipe and convert interior pipes to commas
        let cleaned = trimmed.trim_start_matches('|').trim_end_matches('|');
        // Trim whitespace from each cell value
        let csv_line: String = cleaned
            .split('|')
            .map(|cell| cell.trim())
            .collect::<Vec<_>>()
            .join(",");
        if !csv_line.is_empty() {
            output.push_str(&csv_line);
            output.push('\n');
        }
    }
    output
}

/// Download from HEASARC and convert pipe-delimited output to CSV.
pub fn download_heasarc_csv(url: &str, path: &Path) -> Result<u64, FetchError> {
    let body = download_to_string(url)?;
    validate_not_html(body.as_bytes())?;
    let csv_data = convert_pipe_to_csv(&body);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, &csv_data)?;
    Ok(csv_data.len() as u64)
}

/// Trait for dataset providers that can fetch and parse their data.
pub trait DatasetProvider {
    /// Human-readable name of the dataset.
    fn name(&self) -> &str;

    /// Download the dataset to the output directory.
    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError>;

    /// Check whether the dataset is already downloaded.
    fn is_cached(&self, config: &FetchConfig) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_not_html_accepts_csv() {
        let data = b"col1,col2,col3\n1,2,3\n4,5,6\n";
        assert!(validate_not_html(data).is_ok());
    }

    #[test]
    fn test_validate_not_html_rejects_html() {
        let data = b"<!DOCTYPE html><html><body>Error</body></html>";
        assert!(validate_not_html(data).is_err());
    }

    #[test]
    fn test_validate_not_html_rejects_script() {
        let data = b"<script>window.location = '/login'</script>";
        assert!(validate_not_html(data).is_err());
    }

    #[test]
    fn test_fetch_config_default() {
        let cfg = FetchConfig::default();
        assert_eq!(cfg.output_dir, PathBuf::from("data/external"));
        assert!(cfg.skip_existing);
        assert!(cfg.verify_checksums);
    }

    #[test]
    fn test_pipe_to_csv_basic() {
        let input = "name|ra|dec|\nGRB080714|123.4|56.7|\nGRB090101|98.1|-12.3|\n";
        let csv = convert_pipe_to_csv(input);
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], "name,ra,dec");
        assert_eq!(lines[1], "GRB080714,123.4,56.7");
        assert_eq!(lines[2], "GRB090101,98.1,-12.3");
    }

    #[test]
    fn test_pipe_to_csv_skips_html() {
        let input = "<html>\nname|value|\nfoo|bar|\n";
        let csv = convert_pipe_to_csv(input);
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "name,value");
    }

    #[test]
    fn test_pipe_to_csv_empty_input() {
        let csv = convert_pipe_to_csv("");
        assert!(csv.is_empty());
    }

    #[test]
    fn test_pipe_to_csv_batch_display_format() {
        // Real W3Browse BatchDisplay output has markers and separators
        let input = "\
BatchStart\n\
|name        |ra   |dec  |\n\
+------------+-----+-----+\n\
|GRB080714086|12.34|56.78|\n\
|GRB090101   |98.10|-12.3|\n\
BatchEnd\n";
        let csv = convert_pipe_to_csv(input);
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(
            lines.len(),
            3,
            "Should have header + 2 data rows, got: {:?}",
            lines
        );
        assert_eq!(lines[0], "name,ra,dec");
        assert_eq!(lines[1], "GRB080714086,12.34,56.78");
        assert_eq!(lines[2], "GRB090101,98.10,-12.3");
    }
}

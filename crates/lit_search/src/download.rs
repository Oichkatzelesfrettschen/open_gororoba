//! PDF download via aria2c or reqwest fallback.
//!
//! Shells out to `aria2c` for batch downloads with resume support.
//! Falls back to single-file reqwest if aria2c is not installed.

use crate::models::Paper;
use std::path::{Path, PathBuf};

/// Result of a download attempt.
#[derive(Debug)]
pub struct DownloadResult {
    pub paper_id: String,
    pub title: String,
    pub path: Option<PathBuf>,
    pub error: Option<String>,
}

/// Check if aria2c is available on the system.
pub async fn aria2c_available() -> bool {
    tokio::process::Command::new("aria2c")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .await
        .is_ok_and(|s| s.success())
}

/// Download PDFs for papers that have pdf_url set.
///
/// Uses aria2c for batch downloads when available, reqwest otherwise.
/// Returns one `DownloadResult` per paper that had a pdf_url.
pub async fn download_pdfs(
    papers: &[Paper],
    output_dir: &Path,
    client: &reqwest::Client,
) -> Vec<DownloadResult> {
    let downloadable: Vec<&Paper> = papers
        .iter()
        .filter(|p| !p.pdf_url.is_empty())
        .collect();

    if downloadable.is_empty() {
        tracing::info!("No papers with PDF URLs to download");
        return Vec::new();
    }

    std::fs::create_dir_all(output_dir).ok();

    if aria2c_available().await {
        download_via_aria2c(&downloadable, output_dir).await
    } else {
        tracing::info!("aria2c not found, using reqwest fallback (no resume support)");
        download_via_reqwest(&downloadable, output_dir, client).await
    }
}

/// Batch download via aria2c.
///
/// Writes a temporary input file with one URL per line, then invokes
/// aria2c with resume, conditional-get, and parallel connections.
async fn download_via_aria2c(
    papers: &[&Paper],
    output_dir: &Path,
) -> Vec<DownloadResult> {
    let input_path = output_dir.join(".aria2c_input.txt");
    let mut input_content = String::new();

    // Build the input file: URL\n  out=filename\n
    let mut file_map: Vec<(String, String)> = Vec::new();
    for paper in papers {
        let filename = sanitize_filename(&paper.cite_key(), &paper.pdf_url);
        input_content.push_str(&paper.pdf_url);
        input_content.push('\n');
        input_content.push_str(&format!("  out={filename}\n"));
        file_map.push((paper.paper_id.clone(), filename));
    }

    if let Err(e) = std::fs::write(&input_path, &input_content) {
        tracing::error!("Failed to write aria2c input file: {e}");
        return papers
            .iter()
            .map(|p| DownloadResult {
                paper_id: p.paper_id.clone(),
                title: p.title.clone(),
                path: None,
                error: Some(format!("Failed to write input file: {e}")),
            })
            .collect();
    }

    let status = tokio::process::Command::new("aria2c")
        .arg("--input-file")
        .arg(&input_path)
        .arg("--dir")
        .arg(output_dir)
        .arg("--continue=true")        // Resume partial downloads
        .arg("--auto-file-renaming=false")
        .arg("--conditional-get=true")  // HTTP If-Modified-Since
        .arg("--max-concurrent-downloads=4")
        .arg("--max-connection-per-server=2")
        .arg("--retry-wait=3")
        .arg("--max-tries=3")
        .arg("--console-log-level=warn")
        .arg("--summary-interval=0")
        .status()
        .await;

    // Clean up input file
    std::fs::remove_file(&input_path).ok();

    let success = status.is_ok_and(|s| s.success());
    if !success {
        tracing::warn!("aria2c exited with non-zero status (some downloads may have failed)");
    }

    // Check which files actually exist
    let mut results = Vec::new();
    for (idx, (paper_id, filename)) in file_map.iter().enumerate() {
        let path = output_dir.join(filename);
        if path.exists() && std::fs::metadata(&path).is_ok_and(|m| m.len() > 1000) {
            results.push(DownloadResult {
                paper_id: paper_id.clone(),
                title: papers[idx].title.clone(),
                path: Some(path),
                error: None,
            });
        } else {
            results.push(DownloadResult {
                paper_id: paper_id.clone(),
                title: papers[idx].title.clone(),
                path: None,
                error: Some("Download failed or file too small".into()),
            });
        }
    }

    results
}

/// Single-file download via reqwest (fallback).
async fn download_via_reqwest(
    papers: &[&Paper],
    output_dir: &Path,
    client: &reqwest::Client,
) -> Vec<DownloadResult> {
    let mut results = Vec::new();

    for paper in papers {
        let filename = sanitize_filename(&paper.cite_key(), &paper.pdf_url);
        let path = output_dir.join(&filename);

        // Skip if already downloaded
        if path.exists() && std::fs::metadata(&path).is_ok_and(|m| m.len() > 1000) {
            tracing::info!("Skipping (exists): {filename}");
            results.push(DownloadResult {
                paper_id: paper.paper_id.clone(),
                title: paper.title.clone(),
                path: Some(path),
                error: None,
            });
            continue;
        }

        match client.get(&paper.pdf_url).send().await {
            Ok(resp) if resp.status().is_success() => {
                match resp.bytes().await {
                    Ok(bytes) if bytes.len() > 1000 => {
                        if let Err(e) = std::fs::write(&path, &bytes) {
                            results.push(DownloadResult {
                                paper_id: paper.paper_id.clone(),
                                title: paper.title.clone(),
                                path: None,
                                error: Some(format!("Write error: {e}")),
                            });
                        } else {
                            tracing::info!("Downloaded: {filename} ({} bytes)", bytes.len());
                            results.push(DownloadResult {
                                paper_id: paper.paper_id.clone(),
                                title: paper.title.clone(),
                                path: Some(path),
                                error: None,
                            });
                        }
                    }
                    Ok(bytes) => {
                        results.push(DownloadResult {
                            paper_id: paper.paper_id.clone(),
                            title: paper.title.clone(),
                            path: None,
                            error: Some(format!("Response too small ({} bytes)", bytes.len())),
                        });
                    }
                    Err(e) => {
                        results.push(DownloadResult {
                            paper_id: paper.paper_id.clone(),
                            title: paper.title.clone(),
                            path: None,
                            error: Some(format!("Download error: {e}")),
                        });
                    }
                }
            }
            Ok(resp) => {
                results.push(DownloadResult {
                    paper_id: paper.paper_id.clone(),
                    title: paper.title.clone(),
                    path: None,
                    error: Some(format!("HTTP {}", resp.status())),
                });
            }
            Err(e) => {
                results.push(DownloadResult {
                    paper_id: paper.paper_id.clone(),
                    title: paper.title.clone(),
                    path: None,
                    error: Some(format!("Request error: {e}")),
                });
            }
        }

        // Rate limit between downloads
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }

    results
}

/// Generate a safe filename from cite_key + URL extension.
fn sanitize_filename(cite_key: &str, url: &str) -> String {
    // Academic papers are universally PDF; detect alternative extensions
    // only for explicitly non-PDF URLs (rare in practice).
    let ext = url
        .rsplit('.')
        .next()
        .filter(|e| e.len() <= 4 && e.chars().all(|c| c.is_ascii_alphanumeric()))
        .unwrap_or("pdf");

    let safe: String = cite_key
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect();

    let name = if safe.is_empty() {
        "unknown_paper".to_string()
    } else {
        safe
    };

    format!("{name}.{ext}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize_filename("smith2024geometry", "https://arxiv.org/pdf/2401.01234"), "smith2024geometry.pdf");
        assert_eq!(sanitize_filename("o'malley2023", "http://example.com/paper.pdf"), "o_malley2023.pdf");
        assert_eq!(sanitize_filename("", "http://x.pdf"), "unknown_paper.pdf");
    }
}

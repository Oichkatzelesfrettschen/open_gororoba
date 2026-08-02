//! Shared HTTP download, SHA-256 validation, and disk caching infrastructure.
//!
//! Generalizes the fetch pattern from fetch_chime_frb.rs into reusable functions
//! that all catalog modules share.

#[cfg(feature = "fetch")]
use crate::download_stack::{DownloadStack, TransferRequest};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{self, Read},
    path::{Path, PathBuf},
};
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

#[cfg(not(feature = "fetch"))]
fn fetch_feature_disabled(url: &str) -> FetchError {
    FetchError::Validation(format!(
        "fetch feature disabled; rebuild data_core with `--features fetch` to transfer {url}"
    ))
}

/// Download a URL to a file, returning the number of bytes written.
///
/// Routes through the standardized universal download stack so existing
/// dataset providers inherit the shared backend policy automatically.
pub fn download_to_file(url: &str, path: &Path) -> Result<u64, FetchError> {
    #[cfg(not(feature = "fetch"))]
    {
        let _ = path;
        Err(fetch_feature_disabled(url))
    }
    #[cfg(feature = "fetch")]
    {
        let stack = DownloadStack::default();
        let request = TransferRequest::download(url.to_string(), path.to_path_buf());
        let result = stack
            .recover(&request)
            .map_err(|source| FetchError::HttpError {
                url: url.to_string(),
                source: Box::new(source),
            })?;
        Ok(result.bytes)
    }
}

/// Download a URL as a string (for small text files).
pub fn download_to_string(url: &str) -> Result<String, FetchError> {
    #[cfg(not(feature = "fetch"))]
    {
        Err(fetch_feature_disabled(url))
    }
    #[cfg(feature = "fetch")]
    {
        let stack = DownloadStack::default();
        stack
            .fetch_text(&TransferRequest::probe(url.to_string()))
            .map_err(|source| FetchError::HttpError {
                url: url.to_string(),
                source: Box::new(source),
            })
    }
}

/// Download a URL as a string with a custom timeout.
pub fn download_to_string_with_timeout(
    url: &str,
    timeout: std::time::Duration,
) -> Result<String, FetchError> {
    #[cfg(not(feature = "fetch"))]
    {
        let _ = timeout;
        Err(fetch_feature_disabled(url))
    }
    #[cfg(feature = "fetch")]
    {
        let stack = DownloadStack::default().with_timeout(timeout);
        stack
            .fetch_text(&TransferRequest::probe(url.to_string()))
            .map_err(|source| FetchError::HttpError {
                url: url.to_string(),
                source: Box::new(source),
            })
    }
}

const HAPI_ROOT: &str = "https://cdaweb.gsfc.nasa.gov/hapi/data";
const HAPI_INFO_ROOT: &str = "https://cdaweb.gsfc.nasa.gov/hapi/info";

#[derive(Debug, Clone)]
struct HapiParameterDef {
    name: String,
    size: Vec<usize>,
}

/// Return true when a HAPI response encodes "no data for time range".
pub fn is_hapi_no_data_response(body: &str) -> bool {
    let Ok(value) = serde_json::from_str::<Value>(body) else {
        return false;
    };
    let Some(object) = value.as_object() else {
        return false;
    };
    let Some(status) = object.get("status").and_then(Value::as_object) else {
        return false;
    };
    let code = status.get("code").and_then(Value::as_u64);
    let message = status.get("message").and_then(Value::as_str);
    object.contains_key("HAPI")
        && code == Some(1201)
        && message
            .map(|message| message.to_ascii_lowercase().contains("no data"))
            .unwrap_or(false)
}

fn hapi_data_url(
    dataset_id: &str,
    time_min: &str,
    time_max: &str,
    parameters: Option<&[&str]>,
) -> String {
    let mut url =
        format!("{HAPI_ROOT}?id={dataset_id}&time.min={time_min}&time.max={time_max}&format=csv");
    if let Some(parameters) = parameters
        && !parameters.is_empty()
    {
        url.push_str("&parameters=");
        url.push_str(&parameters.join(","));
    }
    url
}

fn hapi_parameter_defs(dataset_id: &str) -> Result<Vec<HapiParameterDef>, FetchError> {
    let info_url = format!("{HAPI_INFO_ROOT}?id={dataset_id}");
    let info = download_to_string(&info_url)?;
    let value: Value = serde_json::from_str(&info).map_err(|err| {
        FetchError::Validation(format!("invalid HAPI info JSON for {dataset_id}: {err}"))
    })?;
    let parameters = value
        .get("parameters")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            FetchError::Validation(format!("HAPI info for {dataset_id} is missing parameters"))
        })?;
    let defs = parameters
        .iter()
        .filter_map(|parameter| {
            let name = parameter.get("name").and_then(Value::as_str)?.to_string();
            let size = parameter
                .get("size")
                .and_then(Value::as_array)
                .map(|values| {
                    values
                        .iter()
                        .filter_map(Value::as_u64)
                        .map(|value| value as usize)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            Some(HapiParameterDef { name, size })
        })
        .collect::<Vec<_>>();
    if defs.is_empty() {
        return Err(FetchError::Validation(format!(
            "HAPI info for {dataset_id} exposed zero parameter names"
        )));
    }
    Ok(defs)
}

fn expand_hapi_parameter_columns(def: &HapiParameterDef) -> Vec<String> {
    let width = if def.size.is_empty() {
        1
    } else {
        def.size
            .iter()
            .copied()
            .fold(1_usize, |acc, value| acc.saturating_mul(value.max(1)))
    };
    if width <= 1 {
        return vec![def.name.clone()];
    }
    (0..width)
        .map(|index| format!("{}_{}", def.name, index))
        .collect()
}

fn hapi_parameter_names(
    dataset_id: &str,
    requested: Option<&[&str]>,
) -> Result<Vec<String>, FetchError> {
    let defs = hapi_parameter_defs(dataset_id)?;
    let mut names = Vec::new();
    if let Some(requested) = requested {
        for requested_name in requested {
            let Some(def) = defs.iter().find(|def| def.name == *requested_name) else {
                return Err(FetchError::Validation(format!(
                    "HAPI info for {dataset_id} is missing requested parameter {requested_name}"
                )));
            };
            names.extend(expand_hapi_parameter_columns(def));
        }
    } else {
        for def in &defs {
            names.extend(expand_hapi_parameter_columns(def));
        }
    }
    Ok(names)
}

fn synthesize_hapi_header_if_missing(
    dataset_id: &str,
    body: &str,
    requested: Option<&[&str]>,
) -> Result<String, FetchError> {
    let trimmed = body.trim_start();
    if trimmed.starts_with('{') {
        return Err(FetchError::Validation(format!(
            "dataset {dataset_id} returned JSON instead of tabular CSV"
        )));
    }
    let first_line = trimmed.lines().next().unwrap_or("");
    let starts_like_data = first_line
        .chars()
        .next()
        .map(|ch| ch.is_ascii_digit())
        .unwrap_or(false);
    if !starts_like_data {
        return Ok(body.to_string());
    }
    let header = hapi_parameter_names(dataset_id, requested)?.join(",");
    Ok(format!("{header}\n{body}"))
}

/// Download raw CSV bytes from the CDAWeb HAPI endpoint.
///
/// The returned string preserves the server serialization, including whether
/// CDAWeb supplied a CSV header. No-data status documents and other JSON
/// responses fail before they can become cache payloads.
pub fn download_hapi_csv_raw(
    dataset_id: &str,
    time_min: &str,
    time_max: &str,
    parameters: Option<&[&str]>,
) -> Result<String, FetchError> {
    let url = hapi_data_url(dataset_id, time_min, time_max, parameters);
    let body = download_to_string(&url)?;
    if is_hapi_no_data_response(&body) {
        return Err(FetchError::Validation(format!(
            "dataset {dataset_id} returned no data for requested window {time_min}..{time_max}"
        )));
    }
    if body.trim_start().starts_with('{') {
        return Err(FetchError::Validation(format!(
            "dataset {dataset_id} returned JSON instead of tabular CSV"
        )));
    }
    Ok(body)
}

/// Download CSV rows from the CDAWeb HAPI endpoint and synthesize a header.
pub fn download_hapi_csv(
    dataset_id: &str,
    time_min: &str,
    time_max: &str,
    parameters: Option<&[&str]>,
) -> Result<String, FetchError> {
    let body = download_hapi_csv_raw(dataset_id, time_min, time_max, parameters)?;
    synthesize_hapi_header_if_missing(dataset_id, &body, parameters)
}

/// Download HAPI CSV to a file using the file-based download path.
///
/// Routes through `download_to_file` which supports aria2c fallback for
/// large transfers (MMS FGM SRVY: ~140 MB/day at 16 Hz). The HAPI header
/// is synthesized from the info endpoint and prepended to the saved file.
pub fn download_hapi_csv_to_file(
    dataset_id: &str,
    time_min: &str,
    time_max: &str,
    parameters: Option<&[&str]>,
    output: &Path,
) -> Result<u64, FetchError> {
    let url = hapi_data_url(dataset_id, time_min, time_max, parameters);

    // Download raw CSV to a temp file first
    let tmp = output.with_extension("csv.tmp");
    download_to_file(&url, &tmp)?;

    // Read, check for no-data, synthesize header, write final
    let body = fs::read_to_string(&tmp)?;
    if is_hapi_no_data_response(&body) {
        let _ = fs::remove_file(&tmp);
        return Err(FetchError::Validation(format!(
            "dataset {dataset_id} returned no data for requested window {time_min}..{time_max}"
        )));
    }
    let with_header = synthesize_hapi_header_if_missing(dataset_id, &body, parameters)?;
    let bytes = with_header.len() as u64;
    fs::write(output, with_header)?;
    let _ = fs::remove_file(&tmp);
    Ok(bytes)
}

/// Datasets known to produce large per-day downloads (>50 MB) that should
/// route through the file-based path with aria2c fallback.
///
/// Canonical source: registry/data_servers.toml, server id=cdaweb_hapi,
/// field `large_datasets`. This hot-path const is a performance cache;
/// the data_servers_xref gate (plan P6A.S5.T4 follow-up) enforces that
/// this array matches the registry entry. Edit BOTH when the list
/// changes; the gate will catch drift.
const LARGE_HAPI_DATASETS: &[&str] = &[
    "MMS1_FGM_SRVY_L2",
    "MMS2_FGM_SRVY_L2",
    "MMS3_FGM_SRVY_L2",
    "MMS4_FGM_SRVY_L2",
    "MESSENGER_MAG_RTN",
    "MVN_MAG_L2-SUNSTATE-1SEC",
    "C1_CP_FGM_5VPS",
    "C2_CP_FGM_5VPS",
    "C3_CP_FGM_5VPS",
    "C4_CP_FGM_5VPS",
];

/// Auto-routing HAPI download: uses file-based path (aria2c fallback) for
/// known large datasets, string-based path for everything else.
///
/// WHY: High-cadence mission data (MMS 16 Hz, MESSENGER 1 Hz, Cluster 5 VPS)
/// produces >100 MB/day CSV. The default reqwest 120s timeout is insufficient.
/// The file-based path supports aria2c with resume, retry, and multi-connection.
pub fn download_hapi_csv_auto(
    dataset_id: &str,
    time_min: &str,
    time_max: &str,
    parameters: Option<&[&str]>,
    output: Option<&Path>,
) -> Result<String, FetchError> {
    let base_id = dataset_id.split('@').next().unwrap_or(dataset_id);
    let is_large = LARGE_HAPI_DATASETS
        .iter()
        .any(|&prefix| base_id.starts_with(prefix));

    if is_large {
        if let Some(path) = output {
            download_hapi_csv_to_file(dataset_id, time_min, time_max, parameters, path)?;
            fs::read_to_string(path).map_err(FetchError::Io)
        } else {
            // No output path given for a large dataset -- fall back to string
            // with extended timeout via download_to_string_with_timeout
            let timeout = std::time::Duration::from_secs(600);
            let url = hapi_data_url(dataset_id, time_min, time_max, parameters);
            let body = download_to_string_with_timeout(&url, timeout)?;
            if is_hapi_no_data_response(&body) {
                return Err(FetchError::Validation(format!(
                    "dataset {dataset_id} returned no data for {time_min}..{time_max}"
                )));
            }
            synthesize_hapi_header_if_missing(dataset_id, &body, parameters)
        }
    } else {
        download_hapi_csv(dataset_id, time_min, time_max, parameters)
    }
}

/// AMDA/CDPP HAPI service root (European mirror, same protocol as CDAWeb HAPI).
///
/// WHY a separate constant: AMDA uses a different origin and its CSV responses
/// always include a header row, so we skip the CDAWeb info-endpoint round-trip
/// that `download_hapi_csv` uses for header synthesis.
const AMDA_HAPI_ROOT: &str = "https://amda.irap.omp.eu/service/hapi/data";

/// Download CSV rows from the AMDA HAPI endpoint.
///
/// AMDA's HAPI v2 CSV responses include a header row by default, so no
/// info-endpoint query or header synthesis is required.  The caller receives
/// the raw CSV string (including the header line) and is responsible for
/// parsing column positions from the header.
///
/// Use this as a second fallback when both the SPDF direct download and the
/// CDAWeb HAPI mirror are unavailable.  AMDA dataset IDs differ from CDAWeb
/// IDs; consult <https://amda.irap.omp.eu/> for the dataset catalogue.
pub fn download_amda_hapi_csv(
    dataset_id: &str,
    time_min: &str,
    time_max: &str,
    parameters: Option<&[&str]>,
) -> Result<String, FetchError> {
    let mut url = format!(
        "{AMDA_HAPI_ROOT}?id={dataset_id}&time.min={time_min}&time.max={time_max}&format=csv"
    );
    if let Some(parameters) = parameters
        && !parameters.is_empty()
    {
        url.push_str("&parameters=");
        url.push_str(&parameters.join(","));
    }
    let body = download_to_string(&url)?;
    if is_hapi_no_data_response(&body) {
        return Err(FetchError::Validation(format!(
            "AMDA dataset {dataset_id} returned no data for {time_min}..{time_max}"
        )));
    }
    Ok(body)
}

/// Request parameters for [`fetch_daily_hapi_csv_range`].
///
/// Groups the HAPI-specific fields that are orthogonal to the infrastructure
/// concerns in [`FetchConfig`] (output root, skip-existing flag, checksum).
pub struct DailyHapiFetchRequest<'a> {
    /// Subdirectory under `FetchConfig::output_dir` to write daily CSV files.
    pub subdir: &'a str,
    /// Filename prefix; each file is named `{prefix}_{YYYY}_{DOY:03}.csv`.
    pub file_prefix: &'a str,
    /// Human-readable label used in progress messages.
    pub log_label: &'a str,
    /// HAPI dataset identifier passed to the CDAWeb endpoint.
    pub dataset_id: &'a str,
    /// Calendar year to download.
    pub year: u16,
    /// First day of year (1-based) in the download range, inclusive.
    pub doy_start: u16,
    /// Last day of year (1-based) in the download range, inclusive.
    pub doy_end: u16,
    /// HAPI parameters filter; `None` downloads all parameters.
    pub parameters: Option<&'a [&'a str]>,
}

/// Materialize one HAPI dataset into per-day CSV files for a contiguous DOY range.
///
/// This captures the common "daily chunk -> download -> write CSV" pattern used
/// by several low-level heliophysics/planetary magnetic-field catalogs while
/// leaving parser/model semantics in the catalog modules themselves.
pub fn fetch_daily_hapi_csv_range(
    config: &FetchConfig,
    req: &DailyHapiFetchRequest<'_>,
) -> Result<PathBuf, FetchError> {
    let dir = config.output_dir.join(req.subdir);
    fs::create_dir_all(&dir)?;

    for doy in req.doy_start..=req.doy_end {
        let date = chrono::NaiveDate::from_yo_opt(req.year as i32, doy as u32)
            .ok_or_else(|| FetchError::Validation(format!("invalid DOY {doy}")))?;
        let fname = format!("{}_{:04}_{:03}.csv", req.file_prefix, req.year, doy);
        let output = dir.join(&fname);

        if config.skip_existing && output.exists() {
            continue;
        }

        let t_min = format!("{date}T00:00:00Z");
        let t_max = format!("{date}T23:59:59Z");

        println!("Fetching {} {} DOY {doy}...", req.log_label, req.year);

        match download_hapi_csv(req.dataset_id, &t_min, &t_max, req.parameters) {
            Ok(body) => {
                fs::write(&output, body)?;
            }
            Err(e) => {
                eprintln!("  Warning: {} DOY {doy}: {e}", req.log_label);
            }
        }
    }

    Ok(dir)
}

/// Compute SHA-256 hash of a file, returning hex string.
pub fn compute_sha256(path: &Path) -> Result<String, io::Error> {
    let mut hasher = Sha256::new();
    let mut file = fs::File::open(path)?;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let bytes = file.read(&mut buffer)?;
        if bytes == 0 {
            break;
        }
        hasher.update(&buffer[..bytes]);
    }
    let hash = hasher.finalize();
    Ok(hash.iter().map(|byte| format!("{byte:02x}")).collect())
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
        log::info!(
            "{} already exists at {}, skipping",
            dataset_name,
            output_path.display()
        );
        return Ok(output_path.to_path_buf());
    }

    for url in urls {
        log::info!("Downloading {} from {}...", dataset_name, url);
        match download_to_file(url, output_path) {
            Ok(bytes) => {
                // Validate it's not HTML
                let data = fs::read(output_path)?;
                if let Err(e) = validate_not_html(&data) {
                    log::warn!("Validation failed: {}", e);
                    fs::remove_file(output_path).ok();
                    continue;
                }
                log::info!("Saved {} bytes to {}", bytes, output_path.display());
                if let Ok(hash) = compute_sha256(output_path) {
                    log::debug!("SHA256: {}", hash);
                }
                return Ok(output_path.to_path_buf());
            }
            Err(e) => {
                log::warn!("Failed: {}", e);
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

    #[test]
    fn test_expand_hapi_parameter_columns_scalar() {
        let def = HapiParameterDef {
            name: "RAD_AU".to_string(),
            size: Vec::new(),
        };
        assert_eq!(expand_hapi_parameter_columns(&def), vec!["RAD_AU"]);
    }

    #[test]
    fn test_expand_hapi_parameter_columns_vector() {
        let def = HapiParameterDef {
            name: "BGSEc".to_string(),
            size: vec![3],
        };
        assert_eq!(
            expand_hapi_parameter_columns(&def),
            vec!["BGSEc_0", "BGSEc_1", "BGSEc_2"]
        );
    }

    #[test]
    fn hapi_no_data_response_requires_hapi_status_shape() {
        let marker = "{\n  \"HAPI\": \"2.0\",\n  \"status\": {\"code\": 1201, \"message\": \"OK - no data for time range\"}\n}\n";
        assert!(is_hapi_no_data_response(marker));
        assert!(is_hapi_no_data_response(
            "{\"HAPI\":\"2.0\",\"status\":{\"code\":1201,\"message\":\"no data\"}}"
        ));
        assert!(!is_hapi_no_data_response(
            "{\"status\":{\"code\":1201,\"message\":\"no data\"}}"
        ));
        assert!(!is_hapi_no_data_response("{\"pages\":12}"));
    }
}

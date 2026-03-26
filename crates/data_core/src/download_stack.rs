use backon::{BlockingRetryable, ExponentialBuilder};
use reqwest::{
    blocking::{Client, Response},
    header::{
        ACCEPT, ACCEPT_RANGES, CONTENT_LENGTH, CONTENT_RANGE, HeaderMap, HeaderName, HeaderValue,
        RANGE, USER_AGENT,
    },
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fmt, fs,
    io::{self, Read, Write},
    net::{TcpStream, ToSocketAddrs},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use thiserror::Error;
use ureq::ResponseExt;
use url::Url;

pub const DEFAULT_USER_AGENT: &str = "gororoba-download-stack/0.1 (research)";
pub const DEFAULT_PROBE_BYTES: usize = 1024;
const DEFAULT_RESUME_CHUNK_BYTES: u64 = 64 * 1024 * 1024;
const DEFAULT_REQWEST_RETRY_ATTEMPTS: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransferKind {
    Probe,
    Download,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DownloadBackend {
    Auto,
    Reqwest,
    RsyncCli,
    CurlCli,
    WgetCli,
    Aria2Cli,
    Ureq,
}

impl DownloadBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Reqwest => "reqwest",
            Self::RsyncCli => "rsync-cli",
            Self::CurlCli => "curl-cli",
            Self::WgetCli => "wget-cli",
            Self::Aria2Cli => "aria2-cli",
            Self::Ureq => "ureq",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim() {
            "auto" => Some(Self::Auto),
            "reqwest" => Some(Self::Reqwest),
            "rsync-cli" => Some(Self::RsyncCli),
            "curl-cli" => Some(Self::CurlCli),
            "wget-cli" => Some(Self::WgetCli),
            "aria2-cli" => Some(Self::Aria2Cli),
            "ureq" => Some(Self::Ureq),
            _ => None,
        }
    }
}

impl fmt::Display for DownloadBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone)]
pub struct TransferRequest {
    pub url: String,
    pub output_path: Option<PathBuf>,
    pub probe_bytes: usize,
    pub backend: DownloadBackend,
    pub note: Option<String>,
    pub headers: Vec<(String, String)>,
}

impl TransferRequest {
    pub fn probe(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            output_path: None,
            probe_bytes: DEFAULT_PROBE_BYTES,
            backend: DownloadBackend::Auto,
            note: None,
            headers: Vec::new(),
        }
    }

    pub fn download(url: impl Into<String>, output_path: impl Into<PathBuf>) -> Self {
        Self {
            url: url.into(),
            output_path: Some(output_path.into()),
            probe_bytes: DEFAULT_PROBE_BYTES,
            backend: DownloadBackend::Auto,
            note: None,
            headers: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TransferResult {
    pub backend: DownloadBackend,
    pub kind: TransferKind,
    pub requested_url: String,
    pub final_url: Option<String>,
    pub http_code: Option<u16>,
    pub content_type: Option<String>,
    pub bytes: u64,
    pub sha256: Option<String>,
    pub is_pdf: bool,
    pub output_path: Option<PathBuf>,
    pub note: String,
}

#[derive(Debug, Clone)]
struct HttpDownloadProbe {
    final_url: String,
    status: u16,
    content_type: Option<String>,
    total_bytes: Option<u64>,
    supports_ranges: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EndpointSurface {
    Unknown,
    RsyncModule,
    StaticFile,
    StaticFileRange,
    Votable,
    TapSync,
    DirectoryIndex,
}

impl EndpointSurface {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::RsyncModule => "rsync_module",
            Self::StaticFile => "static_file",
            Self::StaticFileRange => "static_file_range",
            Self::Votable => "votable",
            Self::TapSync => "tap_sync",
            Self::DirectoryIndex => "directory_index",
        }
    }
}

impl fmt::Display for EndpointSurface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointCapabilities {
    pub requested_url: String,
    pub final_url: Option<String>,
    pub scheme: String,
    pub host: Option<String>,
    pub surface: EndpointSurface,
    pub content_type: Option<String>,
    pub http_code: Option<u16>,
    pub content_length: Option<u64>,
    pub supports_ranges: bool,
    pub rsync_reachable: bool,
}

impl TransferResult {
    pub fn to_ledger_row(&self, id: impl Into<String>) -> DownloadLedgerRow {
        let note = match self.final_url.as_deref() {
            Some(final_url) if final_url != self.requested_url => {
                format!("{}; url_effective={final_url}", self.note)
            }
            _ => self.note.clone(),
        };
        DownloadLedgerRow {
            id: id.into(),
            url: self.requested_url.clone(),
            http_code: self
                .http_code
                .map(|code| code.to_string())
                .unwrap_or_default(),
            content_type: self.content_type.clone().unwrap_or_default(),
            bytes: self.bytes,
            sha256: self.sha256.clone().unwrap_or_default(),
            is_pdf: if self.is_pdf { "yes" } else { "no" }.to_string(),
            note,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TransferAttempt {
    pub backend: DownloadBackend,
    pub succeeded: bool,
    pub failure_class: Option<String>,
    pub http_code: Option<u16>,
    pub content_type: Option<String>,
    pub bytes: u64,
    pub sha256: Option<String>,
    pub is_pdf: bool,
    pub final_url: Option<String>,
    pub note: String,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone)]
pub struct TransferTrace {
    pub route: DownloadRoute,
    pub capabilities: Option<EndpointCapabilities>,
    pub terminal_result: Option<TransferResult>,
    pub attempts: Vec<TransferAttempt>,
    pub final_error: Option<String>,
}

impl TransferTrace {
    pub fn into_result(self, url: &str) -> Result<TransferResult, TransferError> {
        if let Some(result) = self.terminal_result {
            return Ok(result);
        }
        if self.route.backends.is_empty() {
            return Err(TransferError::UnsupportedScheme {
                url: url.to_string(),
            });
        }
        let messages = self
            .attempts
            .iter()
            .filter_map(|attempt| attempt.error_message.clone())
            .collect::<Vec<_>>();
        Err(TransferError::AllBackendsFailed {
            url: url.to_string(),
            messages: if messages.is_empty() {
                vec![
                    self.final_error
                        .unwrap_or_else(|| "all backends exhausted".to_string()),
                ]
            } else {
                messages
            },
        })
    }
}

fn classify_transfer_error(err: &TransferError) -> String {
    match err {
        TransferError::UnsupportedScheme { .. } => "unsupported_scheme".to_string(),
        TransferError::MissingBackendTool { .. } => "tool_missing".to_string(),
        TransferError::Io(_) => "io_error".to_string(),
        TransferError::Reqwest { source, .. } => {
            if source.is_timeout() {
                "transport_timeout".to_string()
            } else if source.is_connect() {
                "transport_connect".to_string()
            } else if source.is_redirect() {
                "transport_redirect".to_string()
            } else {
                "transport_reqwest".to_string()
            }
        }
        TransferError::Ureq { .. } => "transport_ureq".to_string(),
        TransferError::BackendFailure {
            backend: _,
            url: _,
            message,
        } => classify_backend_failure_message(message),
        TransferError::MissingOutputPath => "missing_output_path".to_string(),
        TransferError::AllBackendsFailed { .. } => "all_backends_failed".to_string(),
        TransferError::InvalidHeaderValue { .. } => "invalid_header_value".to_string(),
        TransferError::InvalidHeaderName { .. } => "invalid_header_name".to_string(),
        TransferError::PolicyConfig { .. } => "policy_config".to_string(),
    }
}

fn classify_backend_failure_message(message: &str) -> String {
    let lower = message.trim().to_ascii_lowercase();
    if lower.starts_with("http 403") {
        return "http_403".to_string();
    }
    if lower.starts_with("http 404") {
        return "http_404".to_string();
    }
    if lower.starts_with("http 429") {
        return "http_429".to_string();
    }
    if lower.starts_with("http 401") {
        return "http_401".to_string();
    }
    if lower.starts_with("http 5") {
        return "http_5xx".to_string();
    }
    if lower.starts_with("http 4") {
        return "http_4xx".to_string();
    }
    if lower.contains("not available on path") {
        return "tool_missing".to_string();
    }
    if lower.contains("timed out") || lower.contains("timeout") {
        return "transport_timeout".to_string();
    }
    if lower.contains("resolve host")
        || lower.contains("could not resolve")
        || lower.contains("dns")
    {
        return "transport_dns".to_string();
    }
    if lower.contains("connection")
        || lower.contains("tls")
        || lower.contains("ssl")
        || lower.contains("certificate")
    {
        return "transport_connect".to_string();
    }
    if lower.contains("aria2 probe mode is not implemented") {
        return "probe_not_supported".to_string();
    }
    "backend_failure".to_string()
}

#[derive(Debug, Clone)]
pub struct DownloadRoute {
    pub kind: TransferKind,
    pub scheme: String,
    pub host: Option<String>,
    pub backends: Vec<DownloadBackend>,
    pub retry_class: RetryClass,
    pub policy_name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DownloadLedgerRow {
    pub id: String,
    pub url: String,
    pub http_code: String,
    pub content_type: String,
    pub bytes: u64,
    pub sha256: String,
    pub is_pdf: String,
    pub note: String,
}

impl DownloadLedgerRow {
    pub fn header() -> &'static str {
        "id\turl\thttp_code\tcontent_type\tbytes\tsha256\tis_pdf\tnote"
    }

    pub fn to_tsv_line(&self) -> String {
        [
            escape_tsv_field(&self.id),
            escape_tsv_field(&self.url),
            escape_tsv_field(&self.http_code),
            escape_tsv_field(&self.content_type),
            self.bytes.to_string(),
            escape_tsv_field(&self.sha256),
            escape_tsv_field(&self.is_pdf),
            escape_tsv_field(&self.note),
        ]
        .join("\t")
    }
}

#[derive(Debug, Clone)]
pub struct DownloadStack {
    user_agent: String,
    timeout: Duration,
    host_policies: Vec<HostRoutingPolicy>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetryClass {
    DefaultHttp,
    CurlFirst,
    ProbeFirst,
    Aria2Download,
    FtpFamily,
    BlockedHost,
}

impl RetryClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DefaultHttp => "default_http",
            Self::CurlFirst => "curl_first",
            Self::ProbeFirst => "probe_first",
            Self::Aria2Download => "aria2_download",
            Self::FtpFamily => "ftp_family",
            Self::BlockedHost => "blocked_host",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostRoutingPolicy {
    pub name: String,
    pub host_suffix: String,
    pub retry_class: RetryClass,
    pub probe_backends: Vec<DownloadBackend>,
    pub download_backends: Vec<DownloadBackend>,
    pub note: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HostPolicyRegistry {
    #[serde(default, rename = "policy")]
    pub policies: Vec<HostRoutingPolicy>,
}

#[derive(Debug, Error)]
pub enum TransferError {
    #[error("unsupported URL scheme for {url}")]
    UnsupportedScheme { url: String },
    #[error("tool backend {backend} is not available on PATH")]
    MissingBackendTool { backend: DownloadBackend },
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("reqwest error for {url}: {source}")]
    Reqwest {
        url: String,
        #[source]
        source: reqwest::Error,
    },
    #[error("ureq error for {url}: {message}")]
    Ureq { url: String, message: String },
    #[error("backend {backend} failed for {url}: {message}")]
    BackendFailure {
        backend: DownloadBackend,
        url: String,
        message: String,
    },
    #[error("no output path provided for download request")]
    MissingOutputPath,
    #[error("all backends exhausted for {url}: {messages:?}")]
    AllBackendsFailed { url: String, messages: Vec<String> },
    #[error("header value rejected for {name}")]
    InvalidHeaderValue { name: String },
    #[error("header name rejected: {name}")]
    InvalidHeaderName { name: String },
    #[error("host policy registry error for {path}: {message}")]
    PolicyConfig { path: String, message: String },
}

impl DownloadStack {
    pub fn new() -> Self {
        Self {
            user_agent: DEFAULT_USER_AGENT.to_string(),
            timeout: Duration::from_secs(120),
            host_policies: default_host_policies(),
        }
    }

    pub fn with_user_agent(mut self, user_agent: impl Into<String>) -> Self {
        self.user_agent = user_agent.into();
        self
    }

    pub fn with_host_policies(mut self, host_policies: Vec<HostRoutingPolicy>) -> Self {
        self.host_policies = host_policies;
        self
    }

    pub fn route(&self, request: &TransferRequest, kind: TransferKind) -> DownloadRoute {
        self.route_with_capabilities(request, kind, None)
    }

    pub fn route_with_capabilities(
        &self,
        request: &TransferRequest,
        kind: TransferKind,
        capabilities: Option<&EndpointCapabilities>,
    ) -> DownloadRoute {
        if request.backend != DownloadBackend::Auto {
            return DownloadRoute {
                kind,
                scheme: parse_url_scheme(&request.url),
                host: parse_url_host(&request.url),
                backends: vec![request.backend],
                retry_class: RetryClass::DefaultHttp,
                policy_name: None,
            };
        }

        if let Some(capabilities) = capabilities {
            return route_from_capabilities(capabilities, kind);
        }

        let scheme = parse_url_scheme(&request.url);
        let host = parse_url_host(&request.url);
        if let Some(policy) = host
            .as_deref()
            .and_then(|host| best_matching_policy(&self.host_policies, host))
        {
            let backends = match kind {
                TransferKind::Probe => policy.probe_backends.clone(),
                TransferKind::Download => policy.download_backends.clone(),
            };
            return DownloadRoute {
                kind,
                scheme,
                host,
                backends,
                retry_class: policy.retry_class,
                policy_name: Some(policy.name.clone()),
            };
        }
        let difficult_host = host.as_deref().map(is_curl_first_host).unwrap_or(false);

        let backends = match scheme.as_str() {
            "ftp" | "ftps" | "sftp" => match kind {
                TransferKind::Probe => vec![DownloadBackend::CurlCli],
                TransferKind::Download => vec![DownloadBackend::CurlCli, DownloadBackend::Aria2Cli],
            },
            "rsync" => match kind {
                TransferKind::Probe => vec![DownloadBackend::RsyncCli],
                TransferKind::Download => vec![DownloadBackend::RsyncCli],
            },
            "http" | "https" => match kind {
                TransferKind::Probe if difficult_host => vec![
                    DownloadBackend::CurlCli,
                    DownloadBackend::Reqwest,
                    DownloadBackend::Ureq,
                ],
                TransferKind::Probe => vec![
                    DownloadBackend::Reqwest,
                    DownloadBackend::CurlCli,
                    DownloadBackend::Ureq,
                ],
                TransferKind::Download if difficult_host => vec![
                    DownloadBackend::CurlCli,
                    DownloadBackend::Reqwest,
                    DownloadBackend::WgetCli,
                    DownloadBackend::Aria2Cli,
                ],
                TransferKind::Download => vec![
                    DownloadBackend::Reqwest,
                    DownloadBackend::CurlCli,
                    DownloadBackend::WgetCli,
                    DownloadBackend::Aria2Cli,
                ],
            },
            _ => Vec::new(),
        };

        DownloadRoute {
            kind,
            scheme,
            host,
            backends,
            retry_class: if difficult_host {
                RetryClass::CurlFirst
            } else if matches!(kind, TransferKind::Probe) {
                RetryClass::ProbeFirst
            } else {
                RetryClass::DefaultHttp
            },
            policy_name: None,
        }
    }

    pub fn detect_capabilities(
        &self,
        request: &TransferRequest,
    ) -> Result<EndpointCapabilities, TransferError> {
        let scheme = parse_url_scheme(&request.url);
        let host = parse_url_host(&request.url);
        if scheme == "rsync" {
            return Ok(EndpointCapabilities {
                requested_url: request.url.clone(),
                final_url: Some(request.url.clone()),
                scheme,
                host,
                surface: EndpointSurface::RsyncModule,
                content_type: guess_content_type_from_url(&request.url),
                http_code: None,
                content_length: None,
                supports_ranges: false,
                rsync_reachable: true,
            });
        }

        if !matches!(scheme.as_str(), "http" | "https") {
            return Ok(EndpointCapabilities {
                requested_url: request.url.clone(),
                final_url: None,
                scheme,
                host,
                surface: EndpointSurface::Unknown,
                content_type: None,
                http_code: None,
                content_length: None,
                supports_ranges: false,
                rsync_reachable: false,
            });
        }

        let client = self.client()?;
        let headers = self.build_headers(request, Some((0_u64, 0_u64)))?;
        let response = self.send_reqwest_with_retry(&client, request, headers)?;
        let status = response.status().as_u16();
        let headers = response.headers().clone();
        let final_url = response.url().to_string();
        let content_type = content_type_from_headers(&headers);
        let supports_ranges = status == 206
            || headers
                .get(ACCEPT_RANGES)
                .and_then(|value| value.to_str().ok())
                .map(|value| value.eq_ignore_ascii_case("bytes"))
                .unwrap_or(false);
        let content_length = parse_total_bytes(&headers, status);
        let surface = classify_endpoint_surface(
            request.url.as_str(),
            final_url.as_str(),
            content_type.as_deref(),
            supports_ranges,
        );
        drop(response);

        let rsync_reachable = host
            .as_deref()
            .map(probe_rsync_reachability)
            .unwrap_or(false);

        Ok(EndpointCapabilities {
            requested_url: request.url.clone(),
            final_url: Some(final_url),
            scheme,
            host,
            surface,
            content_type,
            http_code: Some(status),
            content_length,
            supports_ranges,
            rsync_reachable,
        })
    }

    pub fn probe(&self, request: &TransferRequest) -> Result<TransferResult, TransferError> {
        self.probe_with_trace(request).into_result(&request.url)
    }

    pub fn recover(&self, request: &TransferRequest) -> Result<TransferResult, TransferError> {
        self.recover_with_trace(request).into_result(&request.url)
    }

    pub fn probe_with_trace(&self, request: &TransferRequest) -> TransferTrace {
        self.execute_with_trace(request, TransferKind::Probe)
    }

    pub fn recover_with_trace(&self, request: &TransferRequest) -> TransferTrace {
        self.execute_with_trace(request, TransferKind::Download)
    }

    pub fn fetch_text(&self, request: &TransferRequest) -> Result<String, TransferError> {
        let headers = self.build_headers(request, None)?;
        let client = self.client()?;
        let response = self.send_reqwest_with_retry(&client, request, headers)?;
        let final_url = response.url().to_string();
        let status = response.status().as_u16();
        if !(200..300).contains(&status) {
            return Err(TransferError::BackendFailure {
                backend: DownloadBackend::Reqwest,
                url: request.url.clone(),
                message: format!("HTTP {status} from {final_url}"),
            });
        }
        response.text().map_err(|source| TransferError::Reqwest {
            url: request.url.clone(),
            source,
        })
    }

    fn execute_with_trace(&self, request: &TransferRequest, kind: TransferKind) -> TransferTrace {
        let capabilities = if request.backend == DownloadBackend::Auto {
            self.detect_capabilities(request).ok()
        } else {
            None
        };
        let route = self.route_with_capabilities(request, kind, capabilities.as_ref());
        if route.backends.is_empty() {
            return TransferTrace {
                route,
                capabilities,
                terminal_result: None,
                attempts: Vec::new(),
                final_error: Some(format!("unsupported URL scheme for {}", request.url)),
            };
        }

        let mut attempts = Vec::new();
        let route_clone = route.clone();
        for backend in route.backends.iter().copied() {
            match self.execute_backend(request, kind, backend) {
                Ok(result) => {
                    attempts.push(TransferAttempt {
                        backend,
                        succeeded: true,
                        failure_class: None,
                        http_code: result.http_code,
                        content_type: result.content_type.clone(),
                        bytes: result.bytes,
                        sha256: result.sha256.clone(),
                        is_pdf: result.is_pdf,
                        final_url: result.final_url.clone(),
                        note: result.note.clone(),
                        error_message: None,
                    });
                    return TransferTrace {
                        route: route_clone,
                        capabilities,
                        terminal_result: Some(result),
                        attempts,
                        final_error: None,
                    };
                }
                Err(err) => attempts.push(TransferAttempt {
                    backend,
                    succeeded: false,
                    failure_class: Some(classify_transfer_error(&err)),
                    http_code: None,
                    content_type: None,
                    bytes: 0,
                    sha256: None,
                    is_pdf: false,
                    final_url: None,
                    note: compose_note(&request.note, backend, kind),
                    error_message: Some(err.to_string()),
                }),
            }
        }

        TransferTrace {
            route: route_clone,
            capabilities,
            terminal_result: None,
            attempts,
            final_error: Some(format!("all backends exhausted for {}", request.url)),
        }
    }

    fn execute_backend(
        &self,
        request: &TransferRequest,
        kind: TransferKind,
        backend: DownloadBackend,
    ) -> Result<TransferResult, TransferError> {
        match backend {
            DownloadBackend::Reqwest => self.execute_reqwest(request, kind),
            DownloadBackend::RsyncCli => self.execute_rsync_cli(request, kind),
            DownloadBackend::CurlCli => self.execute_curl_cli(request, kind),
            DownloadBackend::WgetCli => self.execute_wget_cli(request, kind),
            DownloadBackend::Aria2Cli => self.execute_aria2_cli(request, kind),
            DownloadBackend::Ureq => self.execute_ureq(request, kind),
            DownloadBackend::Auto => unreachable!("auto backend must be routed before execution"),
        }
    }

    fn execute_reqwest(
        &self,
        request: &TransferRequest,
        kind: TransferKind,
    ) -> Result<TransferResult, TransferError> {
        let client = self.client()?;
        match kind {
            TransferKind::Probe => self.execute_reqwest_probe(request, &client),
            TransferKind::Download => self.execute_reqwest_download(request, &client),
        }
    }

    fn execute_reqwest_probe(
        &self,
        request: &TransferRequest,
        client: &Client,
    ) -> Result<TransferResult, TransferError> {
        let probe_end = request.probe_bytes.saturating_sub(1) as u64;
        let headers = self.build_headers(request, Some((0_u64, probe_end)))?;
        let mut response = self.send_reqwest_with_retry(client, request, headers)?;

        let final_url = response.url().to_string();
        let status = response.status().as_u16();
        let content_type = content_type_from_headers(response.headers());
        let destination = ephemeral_download_path("reqwest_probe", None);
        let bytes = write_response_to_path(&mut response, &destination)?;
        let prefix = read_prefix(&destination, request.probe_bytes)?;
        let is_pdf = looks_like_pdf(content_type.as_deref(), &prefix);
        let sha256 = Some(sha256_file(&destination)?);
        fs::remove_file(&destination).ok();

        if !status_is_success(status) {
            return Err(TransferError::BackendFailure {
                backend: DownloadBackend::Reqwest,
                url: request.url.clone(),
                message: format!("HTTP {status} from {final_url}"),
            });
        }

        Ok(TransferResult {
            backend: DownloadBackend::Reqwest,
            kind: TransferKind::Probe,
            requested_url: request.url.clone(),
            final_url: Some(final_url),
            http_code: Some(status),
            content_type,
            bytes,
            sha256,
            is_pdf,
            output_path: None,
            note: compose_note(&request.note, DownloadBackend::Reqwest, TransferKind::Probe),
        })
    }

    fn execute_reqwest_download(
        &self,
        request: &TransferRequest,
        client: &Client,
    ) -> Result<TransferResult, TransferError> {
        let destination = request
            .output_path
            .clone()
            .ok_or(TransferError::MissingOutputPath)?;
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?;
        }

        let probe = self.probe_reqwest_download(request, client)?;
        if !status_is_success(probe.status) {
            return Err(TransferError::BackendFailure {
                backend: DownloadBackend::Reqwest,
                url: request.url.clone(),
                message: format!("HTTP {} from {}", probe.status, probe.final_url),
            });
        }

        let mut note = compose_note(
            &request.note,
            DownloadBackend::Reqwest,
            TransferKind::Download,
        );
        let mut local_bytes = fs::metadata(&destination)
            .map(|meta| meta.len())
            .unwrap_or(0);

        if probe.supports_ranges
            && let Some(total_bytes) = probe.total_bytes
        {
            if local_bytes > total_bytes {
                fs::remove_file(&destination).ok();
                local_bytes = 0;
                note.push_str("; removed oversized local partial before range resume");
            } else if local_bytes == total_bytes && total_bytes > 0 {
                let prefix = read_prefix(&destination, request.probe_bytes)?;
                let is_pdf = looks_like_pdf(probe.content_type.as_deref(), &prefix);
                let sha256 = Some(sha256_file(&destination)?);
                note.push_str("; local file already matched remote content-length");
                return Ok(TransferResult {
                    backend: DownloadBackend::Reqwest,
                    kind: TransferKind::Download,
                    requested_url: request.url.clone(),
                    final_url: Some(probe.final_url),
                    http_code: Some(probe.status),
                    content_type: probe.content_type,
                    bytes: total_bytes,
                    sha256,
                    is_pdf,
                    output_path: Some(destination),
                    note,
                });
            }

            if local_bytes > 0 {
                note.push_str(&format!("; resumed from byte {}", local_bytes));
            } else {
                note.push_str("; ranged download");
            }

            while local_bytes < total_bytes {
                self.download_reqwest_range_chunk(
                    request,
                    client,
                    &destination,
                    total_bytes,
                    DEFAULT_RESUME_CHUNK_BYTES,
                )?;
                local_bytes = fs::metadata(&destination)?.len();
            }

            let prefix = read_prefix(&destination, request.probe_bytes)?;
            let is_pdf = looks_like_pdf(probe.content_type.as_deref(), &prefix);
            let sha256 = Some(sha256_file(&destination)?);
            return Ok(TransferResult {
                backend: DownloadBackend::Reqwest,
                kind: TransferKind::Download,
                requested_url: request.url.clone(),
                final_url: Some(probe.final_url),
                http_code: Some(probe.status),
                content_type: probe.content_type,
                bytes: fs::metadata(&destination)?.len(),
                sha256,
                is_pdf,
                output_path: Some(destination),
                note,
            });
        }

        if destination.exists() {
            fs::remove_file(&destination).ok();
        }
        note.push_str("; full-body reqwest fallback (range metadata unavailable)");
        let bytes = self.download_reqwest_full(request, client, &destination)?;
        let prefix = read_prefix(&destination, request.probe_bytes)?;
        let is_pdf = looks_like_pdf(probe.content_type.as_deref(), &prefix);
        let sha256 = Some(sha256_file(&destination)?);

        Ok(TransferResult {
            backend: DownloadBackend::Reqwest,
            kind: TransferKind::Download,
            requested_url: request.url.clone(),
            final_url: Some(probe.final_url),
            http_code: Some(probe.status),
            content_type: probe.content_type,
            bytes,
            sha256,
            is_pdf,
            output_path: Some(destination),
            note,
        })
    }

    fn probe_reqwest_download(
        &self,
        request: &TransferRequest,
        client: &Client,
    ) -> Result<HttpDownloadProbe, TransferError> {
        let headers = self.build_headers(request, Some((0_u64, 0_u64)))?;
        let response = self.send_reqwest_with_retry(client, request, headers)?;
        let status = response.status().as_u16();
        let headers = response.headers().clone();
        let final_url = response.url().to_string();
        let content_type = content_type_from_headers(&headers);
        let supports_ranges = status == 206
            || headers
                .get(ACCEPT_RANGES)
                .and_then(|value| value.to_str().ok())
                .map(|value| value.eq_ignore_ascii_case("bytes"))
                .unwrap_or(false);
        let total_bytes = parse_total_bytes(&headers, status);
        drop(response);

        Ok(HttpDownloadProbe {
            final_url,
            status,
            content_type,
            total_bytes,
            supports_ranges,
        })
    }

    fn send_reqwest_with_retry(
        &self,
        client: &Client,
        request: &TransferRequest,
        headers: HeaderMap,
    ) -> Result<Response, TransferError> {
        let url = request.url.clone();
        let retry = ExponentialBuilder::default()
            .with_min_delay(Duration::from_millis(250))
            .with_max_delay(Duration::from_secs(5))
            .with_max_times(DEFAULT_REQWEST_RETRY_ATTEMPTS);

        (|| {
            let response = client
                .get(&url)
                .headers(headers.clone())
                .send()
                .map_err(|source| TransferError::Reqwest {
                    url: url.clone(),
                    source,
                })?;
            let status = response.status().as_u16();
            if should_retry_http_status(status) {
                return Err(TransferError::BackendFailure {
                    backend: DownloadBackend::Reqwest,
                    url: url.clone(),
                    message: format!("HTTP {status} from {}", response.url()),
                });
            }
            Ok(response)
        })
        .retry(retry)
        .sleep(std::thread::sleep)
        .when(should_retry_transfer_error)
        .call()
    }

    fn download_reqwest_range_chunk(
        &self,
        request: &TransferRequest,
        client: &Client,
        destination: &Path,
        total_bytes: u64,
        chunk_bytes: u64,
    ) -> Result<(), TransferError> {
        let current_bytes = fs::metadata(destination)
            .map(|meta| meta.len())
            .unwrap_or(0);
        if current_bytes >= total_bytes {
            return Ok(());
        }

        let end = (current_bytes + chunk_bytes)
            .saturating_sub(1)
            .min(total_bytes.saturating_sub(1));
        let headers = self.build_headers(request, Some((current_bytes, end)))?;
        let mut response = self.send_reqwest_with_retry(client, request, headers)?;
        let status = response.status().as_u16();
        if status != 206 && !(status == 200 && current_bytes == 0) {
            return Err(TransferError::BackendFailure {
                backend: DownloadBackend::Reqwest,
                url: request.url.clone(),
                message: format!(
                    "Expected partial content for bytes {}-{}, got HTTP {} from {}",
                    current_bytes,
                    end,
                    status,
                    response.url()
                ),
            });
        }

        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(destination)?;
        copy_response_to_writer(&mut response, &mut file)?;
        file.flush()?;
        Ok(())
    }

    fn download_reqwest_full(
        &self,
        request: &TransferRequest,
        client: &Client,
        destination: &Path,
    ) -> Result<u64, TransferError> {
        let temp_path = destination.with_extension("part");
        if temp_path.exists() {
            fs::remove_file(&temp_path).ok();
        }
        let headers = self.build_headers(request, None)?;
        let mut response = self.send_reqwest_with_retry(client, request, headers)?;
        let status = response.status().as_u16();
        let final_url = response.url().to_string();
        if !status_is_success(status) {
            fs::remove_file(&temp_path).ok();
            return Err(TransferError::BackendFailure {
                backend: DownloadBackend::Reqwest,
                url: request.url.clone(),
                message: format!("HTTP {status} from {final_url}"),
            });
        }
        let bytes = write_response_to_path(&mut response, &temp_path)?;
        fs::rename(&temp_path, destination)?;
        Ok(bytes)
    }

    fn execute_ureq(
        &self,
        request: &TransferRequest,
        kind: TransferKind,
    ) -> Result<TransferResult, TransferError> {
        let mut builder = ureq::get(&request.url)
            .header(ACCEPT.as_str(), "*/*")
            .header(USER_AGENT.as_str(), &self.user_agent);
        for (name, value) in &request.headers {
            builder = builder.header(name, value);
        }
        if kind == TransferKind::Probe {
            let probe_end = request.probe_bytes.saturating_sub(1);
            builder = builder.header(RANGE.as_str(), &format!("bytes=0-{probe_end}"));
        }
        let response = builder.call().map_err(|err| TransferError::Ureq {
            url: request.url.clone(),
            message: err.to_string(),
        })?;

        let status = response.status().as_u16();
        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        let final_url = response.get_uri().to_string();
        let destination = match kind {
            TransferKind::Probe => ephemeral_download_path("ureq_probe", None),
            TransferKind::Download => request
                .output_path
                .clone()
                .ok_or(TransferError::MissingOutputPath)?,
        };
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut reader = response.into_body().into_reader();
        let mut file = fs::File::create(&destination)?;
        let bytes = io::copy(&mut reader, &mut file)?;
        let prefix = read_prefix(&destination, request.probe_bytes)?;
        let is_pdf = looks_like_pdf(content_type.as_deref(), &prefix);
        let sha256 = Some(sha256_file(&destination)?);
        if kind == TransferKind::Probe {
            fs::remove_file(&destination).ok();
        }
        if !status_is_success(status) {
            if kind == TransferKind::Download {
                fs::remove_file(&destination).ok();
            }
            return Err(TransferError::BackendFailure {
                backend: DownloadBackend::Ureq,
                url: request.url.clone(),
                message: format!("HTTP {status} from {final_url}"),
            });
        }

        Ok(TransferResult {
            backend: DownloadBackend::Ureq,
            kind,
            requested_url: request.url.clone(),
            final_url: Some(final_url),
            http_code: Some(status),
            content_type,
            bytes,
            sha256,
            is_pdf,
            output_path: (kind == TransferKind::Download).then_some(destination),
            note: compose_note(&request.note, DownloadBackend::Ureq, kind),
        })
    }

    fn execute_rsync_cli(
        &self,
        request: &TransferRequest,
        kind: TransferKind,
    ) -> Result<TransferResult, TransferError> {
        ensure_tool_available("rsync", DownloadBackend::RsyncCli)?;
        let destination = match kind {
            TransferKind::Probe => ephemeral_download_path("rsync_probe", None),
            TransferKind::Download => request
                .output_path
                .clone()
                .ok_or(TransferError::MissingOutputPath)?,
        };
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?;
        }

        let output = Command::new("rsync")
            .args([
                "--partial",
                "--append-verify",
                "--times",
                "--contimeout=30",
                "--timeout=120",
            ])
            .arg(&request.url)
            .arg(&destination)
            .output()?;

        if !output.status.success() {
            if kind == TransferKind::Probe {
                fs::remove_file(&destination).ok();
            }
            return Err(TransferError::BackendFailure {
                backend: DownloadBackend::RsyncCli,
                url: request.url.clone(),
                message: String::from_utf8_lossy(&output.stderr).trim().to_string(),
            });
        }

        let bytes = fs::metadata(&destination)
            .map(|meta| meta.len())
            .unwrap_or(0);
        let prefix = read_prefix(&destination, request.probe_bytes)?;
        let content_type = guess_content_type_from_path(&destination);
        let is_pdf = looks_like_pdf(content_type.as_deref(), &prefix);
        let sha256 = Some(sha256_file(&destination)?);
        if kind == TransferKind::Probe {
            fs::remove_file(&destination).ok();
        }

        Ok(TransferResult {
            backend: DownloadBackend::RsyncCli,
            kind,
            requested_url: request.url.clone(),
            final_url: Some(request.url.clone()),
            http_code: None,
            content_type,
            bytes,
            sha256,
            is_pdf,
            output_path: (kind == TransferKind::Download).then_some(destination),
            note: compose_note(&request.note, DownloadBackend::RsyncCli, kind),
        })
    }

    fn execute_curl_cli(
        &self,
        request: &TransferRequest,
        kind: TransferKind,
    ) -> Result<TransferResult, TransferError> {
        ensure_tool_available("curl", DownloadBackend::CurlCli)?;
        let destination = match kind {
            TransferKind::Probe => ephemeral_download_path("curl_probe", None),
            TransferKind::Download => request
                .output_path
                .clone()
                .ok_or(TransferError::MissingOutputPath)?,
        };
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut command = Command::new("curl");
        command.args([
            "--silent",
            "--show-error",
            "--location",
            "--output",
            &destination.to_string_lossy(),
            "--user-agent",
            &self.user_agent,
            "--write-out",
            "\n%{http_code}\t%{content_type}\t%{url_effective}",
        ]);
        if kind == TransferKind::Probe {
            let probe_end = request.probe_bytes.saturating_sub(1);
            command.args(["--range", &format!("0-{probe_end}")]);
        } else {
            command.args(["--continue-at", "-"]);
        }
        for (name, value) in &request.headers {
            command.args(["--header", &format!("{name}: {value}")]);
        }
        command.arg(&request.url);

        let output = command.output()?;
        if !output.status.success() {
            if kind == TransferKind::Probe {
                fs::remove_file(&destination).ok();
            }
            return Err(TransferError::BackendFailure {
                backend: DownloadBackend::CurlCli,
                url: request.url.clone(),
                message: String::from_utf8_lossy(&output.stderr).trim().to_string(),
            });
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let (http_code, content_type, final_url) = parse_cli_metadata(&stdout);
        let bytes = fs::metadata(&destination)
            .map(|meta| meta.len())
            .unwrap_or(0);
        let prefix = read_prefix(&destination, request.probe_bytes)?;
        let is_pdf = looks_like_pdf(content_type.as_deref(), &prefix);
        let sha256 = Some(sha256_file(&destination)?);
        if kind == TransferKind::Probe {
            fs::remove_file(&destination).ok();
        }
        if let Some(code) = http_code
            && !status_is_success(code)
        {
            if kind == TransferKind::Probe {
                fs::remove_file(&destination).ok();
            }
            return Err(TransferError::BackendFailure {
                backend: DownloadBackend::CurlCli,
                url: request.url.clone(),
                message: format!("HTTP {code}"),
            });
        }

        Ok(TransferResult {
            backend: DownloadBackend::CurlCli,
            kind,
            requested_url: request.url.clone(),
            final_url,
            http_code,
            content_type,
            bytes,
            sha256,
            is_pdf,
            output_path: (kind == TransferKind::Download).then_some(destination),
            note: compose_note(&request.note, DownloadBackend::CurlCli, kind),
        })
    }

    fn execute_wget_cli(
        &self,
        request: &TransferRequest,
        kind: TransferKind,
    ) -> Result<TransferResult, TransferError> {
        ensure_tool_available("wget", DownloadBackend::WgetCli)?;
        let destination = match kind {
            TransferKind::Probe => ephemeral_download_path("wget_probe", None),
            TransferKind::Download => request
                .output_path
                .clone()
                .ok_or(TransferError::MissingOutputPath)?,
        };
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut command = Command::new("wget");
        command.args([
            "--quiet",
            "--output-document",
            &destination.to_string_lossy(),
            "--user-agent",
            &self.user_agent,
            "--server-response",
            "--max-redirect=10",
            "--tries=5",
            "--waitretry=5",
            "--timeout=120",
        ]);
        if kind == TransferKind::Probe {
            command.args([
                "--header",
                &format!("Range: bytes=0-{}", request.probe_bytes.saturating_sub(1)),
            ]);
        } else {
            command.arg("--continue");
        }
        for (name, value) in &request.headers {
            command.args(["--header", &format!("{name}: {value}")]);
        }
        command.arg(&request.url);
        let output = command.output()?;
        if !output.status.success() {
            if kind == TransferKind::Probe {
                fs::remove_file(&destination).ok();
            }
            return Err(TransferError::BackendFailure {
                backend: DownloadBackend::WgetCli,
                url: request.url.clone(),
                message: String::from_utf8_lossy(&output.stderr).trim().to_string(),
            });
        }

        let stderr = String::from_utf8_lossy(&output.stderr);
        let http_code = parse_last_http_status(&stderr);
        let content_type = parse_last_content_type(&stderr);
        let final_url = parse_last_location(&stderr).or_else(|| Some(request.url.clone()));
        let bytes = fs::metadata(&destination)
            .map(|meta| meta.len())
            .unwrap_or(0);
        let prefix = read_prefix(&destination, request.probe_bytes)?;
        let is_pdf = looks_like_pdf(content_type.as_deref(), &prefix);
        let sha256 = Some(sha256_file(&destination)?);
        if kind == TransferKind::Probe {
            fs::remove_file(&destination).ok();
        }
        if let Some(code) = http_code
            && !status_is_success(code)
        {
            if kind == TransferKind::Probe {
                fs::remove_file(&destination).ok();
            }
            return Err(TransferError::BackendFailure {
                backend: DownloadBackend::WgetCli,
                url: request.url.clone(),
                message: format!("HTTP {code}"),
            });
        }

        Ok(TransferResult {
            backend: DownloadBackend::WgetCli,
            kind,
            requested_url: request.url.clone(),
            final_url,
            http_code,
            content_type,
            bytes,
            sha256,
            is_pdf,
            output_path: (kind == TransferKind::Download).then_some(destination),
            note: compose_note(&request.note, DownloadBackend::WgetCli, kind),
        })
    }

    fn execute_aria2_cli(
        &self,
        request: &TransferRequest,
        kind: TransferKind,
    ) -> Result<TransferResult, TransferError> {
        ensure_tool_available("aria2c", DownloadBackend::Aria2Cli)?;
        if kind == TransferKind::Probe {
            return Err(TransferError::BackendFailure {
                backend: DownloadBackend::Aria2Cli,
                url: request.url.clone(),
                message:
                    "aria2 probe mode is not implemented; route probes through reqwest or curl"
                        .to_string(),
            });
        }
        let destination = request
            .output_path
            .clone()
            .ok_or(TransferError::MissingOutputPath)?;
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?;
        }
        let file_name = destination
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| TransferError::BackendFailure {
                backend: DownloadBackend::Aria2Cli,
                url: request.url.clone(),
                message: "output path must end with a UTF-8 file name".to_string(),
            })?;
        let directory = destination
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));

        let probe_result = self
            .execute_reqwest(
                &TransferRequest {
                    output_path: None,
                    ..request.clone()
                },
                TransferKind::Probe,
            )
            .ok();

        let output = Command::new("aria2c")
            .args([
                "--allow-overwrite=true",
                "--auto-file-renaming=false",
                "--continue=true",
                "--file-allocation=none",
                "--max-concurrent-downloads=1",
                "--max-connection-per-server=2",
                "--split=2",
                "--min-split-size=64M",
                "--max-tries=5",
                "--retry-wait=5",
                "--timeout=120",
                "--connect-timeout=30",
                "--summary-interval=0",
                "--dir",
                &directory.to_string_lossy(),
                "--out",
                file_name,
                "--user-agent",
                &self.user_agent,
                &request.url,
            ])
            .output()?;

        if !output.status.success() {
            return Err(TransferError::BackendFailure {
                backend: DownloadBackend::Aria2Cli,
                url: request.url.clone(),
                message: String::from_utf8_lossy(&output.stderr).trim().to_string(),
            });
        }

        let bytes = fs::metadata(&destination)
            .map(|meta| meta.len())
            .unwrap_or(0);
        let prefix = read_prefix(&destination, request.probe_bytes)?;
        let content_type = probe_result
            .as_ref()
            .and_then(|result| result.content_type.clone())
            .or_else(|| guess_content_type_from_path(&destination));
        let is_pdf = looks_like_pdf(content_type.as_deref(), &prefix);
        let sha256 = Some(sha256_file(&destination)?);
        Ok(TransferResult {
            backend: DownloadBackend::Aria2Cli,
            kind,
            requested_url: request.url.clone(),
            final_url: probe_result
                .as_ref()
                .and_then(|result| result.final_url.clone())
                .or_else(|| Some(request.url.clone())),
            http_code: probe_result.as_ref().and_then(|result| result.http_code),
            content_type,
            bytes,
            sha256,
            is_pdf,
            output_path: Some(destination),
            note: compose_note(&request.note, DownloadBackend::Aria2Cli, kind),
        })
    }

    fn build_headers(
        &self,
        request: &TransferRequest,
        range: Option<(u64, u64)>,
    ) -> Result<HeaderMap, TransferError> {
        let mut headers = HeaderMap::new();
        headers.insert(ACCEPT, HeaderValue::from_static("*/*"));
        headers.insert(
            USER_AGENT,
            HeaderValue::from_str(&self.user_agent).map_err(|_| {
                TransferError::InvalidHeaderValue {
                    name: USER_AGENT.as_str().to_string(),
                }
            })?,
        );
        if let Some((start, end)) = range {
            headers.insert(
                RANGE,
                HeaderValue::from_str(&format!("bytes={start}-{end}")).map_err(|_| {
                    TransferError::InvalidHeaderValue {
                        name: RANGE.as_str().to_string(),
                    }
                })?,
            );
        }
        for (name, value) in &request.headers {
            let header_name = HeaderName::from_bytes(name.as_bytes())
                .map_err(|_| TransferError::InvalidHeaderName { name: name.clone() })?;
            let header_value = HeaderValue::from_str(value)
                .map_err(|_| TransferError::InvalidHeaderValue { name: name.clone() })?;
            headers.insert(header_name, header_value);
        }
        Ok(headers)
    }

    fn client(&self) -> Result<Client, TransferError> {
        Client::builder()
            .timeout(self.timeout)
            .redirect(reqwest::redirect::Policy::limited(10))
            .build()
            .map_err(|source| TransferError::Reqwest {
                url: "reqwest-client".to_string(),
                source,
            })
    }
}

impl Default for DownloadStack {
    fn default() -> Self {
        Self::new()
    }
}

pub fn load_host_policy_registry(path: &Path) -> Result<Vec<HostRoutingPolicy>, TransferError> {
    let text = fs::read_to_string(path).map_err(TransferError::Io)?;
    let registry: HostPolicyRegistry =
        toml::from_str(&text).map_err(|err| TransferError::PolicyConfig {
            path: path.display().to_string(),
            message: err.to_string(),
        })?;
    Ok(registry.policies)
}

fn default_host_policies() -> Vec<HostRoutingPolicy> {
    vec![
        HostRoutingPolicy {
            name: "arxiv".to_string(),
            host_suffix: "arxiv.org".to_string(),
            retry_class: RetryClass::CurlFirst,
            probe_backends: vec![
                DownloadBackend::CurlCli,
                DownloadBackend::Reqwest,
                DownloadBackend::Ureq,
            ],
            download_backends: vec![
                DownloadBackend::CurlCli,
                DownloadBackend::Reqwest,
                DownloadBackend::WgetCli,
                DownloadBackend::Aria2Cli,
            ],
            note: Some("arXiv PDF endpoints respond well to ranged curl probes".to_string()),
        },
        HostRoutingPolicy {
            name: "core".to_string(),
            host_suffix: "core.ac.uk".to_string(),
            retry_class: RetryClass::CurlFirst,
            probe_backends: vec![
                DownloadBackend::Reqwest,
                DownloadBackend::CurlCli,
                DownloadBackend::Ureq,
            ],
            download_backends: vec![
                DownloadBackend::Reqwest,
                DownloadBackend::CurlCli,
                DownloadBackend::WgetCli,
            ],
            note: Some("CORE frequently redirects to fileserver mirrors before terminal status".to_string()),
        },
        HostRoutingPolicy {
            name: "lofar_surveys".to_string(),
            host_suffix: "lofar-surveys.org".to_string(),
            retry_class: RetryClass::Aria2Download,
            probe_backends: vec![
                DownloadBackend::Reqwest,
                DownloadBackend::CurlCli,
                DownloadBackend::Ureq,
            ],
            download_backends: vec![
                DownloadBackend::Aria2Cli,
                DownloadBackend::Reqwest,
                DownloadBackend::CurlCli,
                DownloadBackend::WgetCli,
            ],
            note: Some(
                "LoTSS bulk FITS downloads support HTTP byte ranges; prefer aria2 for gentle segmented resume, then reqwest range-resume before curl/wget"
                    .to_string(),
            ),
        },
        HostRoutingPolicy {
            name: "astron_vo".to_string(),
            host_suffix: "vo.astron.nl".to_string(),
            retry_class: RetryClass::ProbeFirst,
            probe_backends: vec![
                DownloadBackend::Reqwest,
                DownloadBackend::CurlCli,
                DownloadBackend::Ureq,
            ],
            download_backends: vec![
                DownloadBackend::Reqwest,
                DownloadBackend::CurlCli,
                DownloadBackend::WgetCli,
            ],
            note: Some(
                "VO cone-search responses are small XML payloads; reqwest first with retry, shell fallbacks only if needed"
                    .to_string(),
            ),
        },
        HostRoutingPolicy {
            name: "sciencedirect".to_string(),
            host_suffix: "sciencedirect.com".to_string(),
            retry_class: RetryClass::CurlFirst,
            probe_backends: vec![
                DownloadBackend::CurlCli,
                DownloadBackend::Reqwest,
                DownloadBackend::Ureq,
            ],
            download_backends: vec![
                DownloadBackend::CurlCli,
                DownloadBackend::Reqwest,
                DownloadBackend::Aria2Cli,
            ],
            note: Some("Publisher hosts are curl-first because redirects and content negotiation are finicky".to_string()),
        },
        HostRoutingPolicy {
            name: "springer".to_string(),
            host_suffix: "springer.com".to_string(),
            retry_class: RetryClass::CurlFirst,
            probe_backends: vec![
                DownloadBackend::CurlCli,
                DownloadBackend::Reqwest,
                DownloadBackend::Ureq,
            ],
            download_backends: vec![
                DownloadBackend::CurlCli,
                DownloadBackend::Reqwest,
                DownloadBackend::WgetCli,
            ],
            note: Some("Springer family hosts are curl-first because of article/PDF redirect chains".to_string()),
        },
        HostRoutingPolicy {
            name: "link-springer".to_string(),
            host_suffix: "link.springer.com".to_string(),
            retry_class: RetryClass::CurlFirst,
            probe_backends: vec![
                DownloadBackend::CurlCli,
                DownloadBackend::Reqwest,
                DownloadBackend::Ureq,
            ],
            download_backends: vec![
                DownloadBackend::CurlCli,
                DownloadBackend::Reqwest,
                DownloadBackend::WgetCli,
            ],
            note: Some("Direct Springer article host override".to_string()),
        },
        HostRoutingPolicy {
            name: "nasa_cdaweb".to_string(),
            host_suffix: "cdaweb.gsfc.nasa.gov".to_string(),
            retry_class: RetryClass::DefaultHttp,
            probe_backends: vec![
                DownloadBackend::Reqwest,
                DownloadBackend::CurlCli,
                DownloadBackend::Ureq,
            ],
            download_backends: vec![
                DownloadBackend::Reqwest,
                DownloadBackend::CurlCli,
                DownloadBackend::WgetCli,
                DownloadBackend::Aria2Cli,
            ],
            note: Some("NASA CDAWeb HAPI and direct download endpoints".to_string()),
        },
        HostRoutingPolicy {
            name: "ftp-family".to_string(),
            host_suffix: "ftp.invalid".to_string(),
            retry_class: RetryClass::FtpFamily,
            probe_backends: vec![DownloadBackend::CurlCli],
            download_backends: vec![DownloadBackend::CurlCli, DownloadBackend::Aria2Cli],
            note: Some("Scheme-driven ftp fallback baseline".to_string()),
        },
    ]
}

fn best_matching_policy<'a>(
    policies: &'a [HostRoutingPolicy],
    host: &str,
) -> Option<&'a HostRoutingPolicy> {
    policies
        .iter()
        .filter(|policy| host_matches_suffix(host, &policy.host_suffix))
        .max_by_key(|policy| policy.host_suffix.len())
}

fn route_from_capabilities(
    capabilities: &EndpointCapabilities,
    kind: TransferKind,
) -> DownloadRoute {
    let backends = match (capabilities.surface, kind) {
        (EndpointSurface::RsyncModule, _) => vec![DownloadBackend::RsyncCli],
        (EndpointSurface::TapSync | EndpointSurface::Votable, TransferKind::Probe) => vec![
            DownloadBackend::Reqwest,
            DownloadBackend::CurlCli,
            DownloadBackend::Ureq,
        ],
        (EndpointSurface::TapSync | EndpointSurface::Votable, TransferKind::Download) => vec![
            DownloadBackend::Reqwest,
            DownloadBackend::CurlCli,
            DownloadBackend::WgetCli,
        ],
        (EndpointSurface::StaticFileRange, TransferKind::Download) => vec![
            DownloadBackend::Aria2Cli,
            DownloadBackend::Reqwest,
            DownloadBackend::CurlCli,
            DownloadBackend::WgetCli,
        ],
        (EndpointSurface::StaticFileRange, TransferKind::Probe) => vec![
            DownloadBackend::Reqwest,
            DownloadBackend::CurlCli,
            DownloadBackend::Ureq,
        ],
        (EndpointSurface::DirectoryIndex, TransferKind::Probe) => vec![
            DownloadBackend::Reqwest,
            DownloadBackend::CurlCli,
            DownloadBackend::Ureq,
        ],
        (EndpointSurface::DirectoryIndex, TransferKind::Download) => vec![
            DownloadBackend::Reqwest,
            DownloadBackend::CurlCli,
            DownloadBackend::WgetCli,
        ],
        (EndpointSurface::StaticFile | EndpointSurface::Unknown, TransferKind::Probe) => vec![
            DownloadBackend::Reqwest,
            DownloadBackend::CurlCli,
            DownloadBackend::Ureq,
        ],
        (EndpointSurface::StaticFile | EndpointSurface::Unknown, TransferKind::Download) => vec![
            DownloadBackend::Reqwest,
            DownloadBackend::CurlCli,
            DownloadBackend::WgetCli,
            DownloadBackend::Aria2Cli,
        ],
    };

    DownloadRoute {
        kind,
        scheme: capabilities.scheme.clone(),
        host: capabilities.host.clone(),
        backends,
        retry_class: match capabilities.surface {
            EndpointSurface::StaticFileRange => RetryClass::Aria2Download,
            EndpointSurface::TapSync
            | EndpointSurface::Votable
            | EndpointSurface::DirectoryIndex => RetryClass::ProbeFirst,
            EndpointSurface::RsyncModule => RetryClass::FtpFamily,
            EndpointSurface::StaticFile | EndpointSurface::Unknown => RetryClass::DefaultHttp,
        },
        policy_name: Some(format!("capability:{}", capabilities.surface)),
    }
}

fn classify_endpoint_surface(
    requested_url: &str,
    final_url: &str,
    content_type: Option<&str>,
    supports_ranges: bool,
) -> EndpointSurface {
    let requested_lower = requested_url.to_ascii_lowercase();
    let content_type_lower = content_type.unwrap_or_default().to_ascii_lowercase();
    if requested_lower.starts_with("rsync://") {
        return EndpointSurface::RsyncModule;
    }

    let Some(parsed) = Url::parse(final_url)
        .ok()
        .or_else(|| Url::parse(requested_url).ok())
    else {
        return if supports_ranges {
            EndpointSurface::StaticFileRange
        } else {
            EndpointSurface::Unknown
        };
    };

    let path = parsed.path().to_ascii_lowercase();
    let query = parsed.query().unwrap_or_default().to_ascii_lowercase();
    if path.contains("/tap")
        || query.contains("request=doquery")
        || query.contains("lang=adql")
        || query.contains("adql=")
    {
        return EndpointSurface::TapSync;
    }
    if content_type_lower.contains("votable")
        || path.ends_with(".vot")
        || path.ends_with(".votable")
        || path.ends_with(".xml")
        || query.contains("format=votable")
        || query.contains("format=votable/td")
        || (query.contains("ra=") && query.contains("dec=") && query.contains("sr="))
    {
        return EndpointSurface::Votable;
    }
    if content_type_lower.contains("html") && (path.ends_with('/') || !path.contains('.')) {
        return EndpointSurface::DirectoryIndex;
    }
    if supports_ranges {
        EndpointSurface::StaticFileRange
    } else if !content_type_lower.is_empty() {
        EndpointSurface::StaticFile
    } else {
        EndpointSurface::Unknown
    }
}

fn guess_content_type_from_url(url: &str) -> Option<String> {
    let lower = url.to_ascii_lowercase();
    if lower.ends_with(".pdf") {
        Some("application/pdf".to_string())
    } else if lower.ends_with(".xml") || lower.ends_with(".vot") || lower.ends_with(".votable") {
        Some("application/x-votable+xml".to_string())
    } else if lower.ends_with(".csv") {
        Some("text/csv".to_string())
    } else if lower.ends_with(".json") {
        Some("application/json".to_string())
    } else {
        None
    }
}

fn probe_rsync_reachability(host: &str) -> bool {
    let address = format!("{host}:873");
    let mut addrs = match address.to_socket_addrs() {
        Ok(addrs) => addrs,
        Err(_) => return false,
    };
    let Some(sockaddr) = addrs.next() else {
        return false;
    };
    TcpStream::connect_timeout(&sockaddr, Duration::from_millis(500)).is_ok()
}

fn host_matches_suffix(host: &str, suffix: &str) -> bool {
    host == suffix || host.ends_with(&format!(".{suffix}"))
}

fn write_response_to_path(response: &mut Response, path: &Path) -> Result<u64, TransferError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::File::create(path)?;
    copy_response_to_writer(response, &mut file)
}

fn content_type_from_headers(headers: &reqwest::header::HeaderMap) -> Option<String> {
    headers
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string)
}

fn read_prefix(path: &Path, limit: usize) -> Result<Vec<u8>, TransferError> {
    let mut file = fs::File::open(path)?;
    let mut buffer = vec![0_u8; limit.max(16)];
    let bytes = file.read(&mut buffer)?;
    buffer.truncate(bytes);
    Ok(buffer)
}

fn looks_like_pdf(content_type: Option<&str>, prefix: &[u8]) -> bool {
    content_type
        .map(|value| value.to_ascii_lowercase().contains("pdf"))
        .unwrap_or(false)
        || prefix.starts_with(b"%PDF-")
}

fn compose_note(
    base_note: &Option<String>,
    backend: DownloadBackend,
    kind: TransferKind,
) -> String {
    let kind_label = match kind {
        TransferKind::Probe => "probe",
        TransferKind::Download => "download",
    };
    match base_note {
        Some(note) if !note.trim().is_empty() => {
            format!("{note}; standardized {kind_label} via {backend}")
        }
        _ => format!("standardized {kind_label} via {backend}"),
    }
}

fn status_is_success(status: u16) -> bool {
    (200..300).contains(&status)
}

fn should_retry_http_status(status: u16) -> bool {
    matches!(status, 408 | 425 | 429) || (500..600).contains(&status)
}

fn should_retry_transfer_error(err: &TransferError) -> bool {
    match err {
        TransferError::Reqwest { source, .. } => {
            source.is_timeout() || source.is_connect() || source.is_request() || source.is_body()
        }
        TransferError::Io(_) => true,
        TransferError::BackendFailure { message, .. } => {
            let lower = message.to_ascii_lowercase();
            lower.starts_with("http 408")
                || lower.starts_with("http 425")
                || lower.starts_with("http 429")
                || lower.starts_with("http 5")
                || lower.contains("connection")
                || lower.contains("timed out")
                || lower.contains("timeout")
                || lower.contains("reset")
                || lower.contains("temporary")
                || lower.contains("interrupted")
        }
        _ => false,
    }
}

fn copy_response_to_writer<W: Write>(
    response: &mut Response,
    writer: &mut W,
) -> Result<u64, TransferError> {
    response
        .copy_to(writer)
        .map_err(|source| TransferError::Reqwest {
            url: response.url().to_string(),
            source,
        })
}

fn parse_total_bytes(headers: &reqwest::header::HeaderMap, status: u16) -> Option<u64> {
    if status == 206 {
        headers
            .get(CONTENT_RANGE)
            .and_then(|value| value.to_str().ok())
            .and_then(parse_total_bytes_from_content_range)
    } else {
        headers
            .get(CONTENT_LENGTH)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<u64>().ok())
    }
}

fn parse_total_bytes_from_content_range(value: &str) -> Option<u64> {
    let total = value.trim().split('/').nth(1)?;
    if total == "*" {
        None
    } else {
        total.parse::<u64>().ok()
    }
}

fn parse_cli_metadata(stdout: &str) -> (Option<u16>, Option<String>, Option<String>) {
    let metadata_line = stdout.lines().last().unwrap_or_default().trim();
    let mut parts = metadata_line.split('\t');
    let http_code = parts.next().and_then(|code| code.parse::<u16>().ok());
    let content_type = parts.next().and_then(non_empty_string);
    let final_url = parts.next().and_then(non_empty_string);
    (http_code, content_type, final_url)
}

fn parse_last_http_status(stderr: &str) -> Option<u16> {
    stderr
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            trimmed
                .strip_prefix("HTTP/")
                .and_then(|rest| rest.split_whitespace().nth(1))
                .and_then(|code| code.parse::<u16>().ok())
        })
        .next_back()
}

fn parse_last_content_type(stderr: &str) -> Option<String> {
    stderr
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            let lower = trimmed.to_ascii_lowercase();
            lower
                .strip_prefix("content-type:")
                .map(|_| trimmed["Content-Type:".len()..].trim().to_string())
        })
        .next_back()
}

fn parse_last_location(stderr: &str) -> Option<String> {
    stderr
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            let lower = trimmed.to_ascii_lowercase();
            lower
                .strip_prefix("location:")
                .map(|_| trimmed["Location:".len()..].trim().to_string())
        })
        .next_back()
}

fn guess_content_type_from_path(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?.to_ascii_lowercase();
    if name.ends_with(".pdf") {
        Some("application/pdf".to_string())
    } else if name.ends_with(".txt") {
        Some("text/plain".to_string())
    } else if name.ends_with(".csv") {
        Some("text/csv".to_string())
    } else if name.ends_with(".json") {
        Some("application/json".to_string())
    } else {
        None
    }
}

fn ensure_tool_available(tool: &str, backend: DownloadBackend) -> Result<(), TransferError> {
    let status = Command::new(tool)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    match status {
        Ok(result) if result.success() => Ok(()),
        Ok(_) | Err(_) => Err(TransferError::MissingBackendTool { backend }),
    }
}

fn sha256_file(path: &Path) -> Result<String, TransferError> {
    let mut file = fs::File::open(path)?;
    let mut buffer = [0_u8; 64 * 1024];
    let mut hasher = Sha256::new();
    loop {
        let bytes = file.read(&mut buffer)?;
        if bytes == 0 {
            break;
        }
        hasher.update(&buffer[..bytes]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn escape_tsv_field(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}

fn parse_url_scheme(url: &str) -> String {
    Url::parse(url)
        .map(|parsed| parsed.scheme().to_string())
        .unwrap_or_else(|_| "unknown".to_string())
}

fn parse_url_host(url: &str) -> Option<String> {
    Url::parse(url)
        .ok()
        .and_then(|parsed| parsed.host_str().map(str::to_string))
}

fn is_curl_first_host(host: &str) -> bool {
    const CURL_FIRST_HOSTS: &[&str] = &[
        "academic.oup.com",
        "arxiv.org",
        "cambridge.org",
        "doi.org",
        "link.springer.com",
        "onlinelibrary.wiley.com",
        "royalsocietypublishing.org",
        "sciencedirect.com",
        "springer.com",
        "www.cambridge.org",
        "www.mdpi.com",
        "www.researchgate.net",
        "www.sciencedirect.com",
    ];
    CURL_FIRST_HOSTS
        .iter()
        .any(|candidate| host == *candidate || host.ends_with(&format!(".{candidate}")))
}

fn non_empty_string(value: &str) -> Option<String> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn ephemeral_download_path(prefix: &str, extension: Option<&str>) -> PathBuf {
    let mut path = std::env::temp_dir();
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let mut file_name = format!("{prefix}_{}_{}", std::process::id(), stamp);
    if let Some(ext) = extension {
        file_name.push('.');
        file_name.push_str(ext);
    }
    path.push(file_name);
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_route_prefers_reqwest_for_default_http_probe() {
        let stack = DownloadStack::default();
        let request = TransferRequest::probe("https://example.com/file.pdf");
        let route = stack.route(&request, TransferKind::Probe);
        assert_eq!(route.backends.first(), Some(&DownloadBackend::Reqwest));
    }

    #[test]
    fn test_route_prefers_curl_for_sciencedirect_download() {
        let stack = DownloadStack::default();
        let request = TransferRequest::download(
            "https://www.sciencedirect.com/science/article/pii/S123456789",
            "target/out.pdf",
        );
        let route = stack.route(&request, TransferKind::Download);
        assert_eq!(route.backends.first(), Some(&DownloadBackend::CurlCli));
    }

    #[test]
    fn test_route_prefers_reqwest_for_lofar_download() {
        let stack = DownloadStack::default();
        let request = TransferRequest::download(
            "https://lofar-surveys.org/public/DR2/catalogues/LoTSS_DR2_v110_masked.srl.fits",
            "target/lotss_dr2.fits",
        );
        let route = stack.route(&request, TransferKind::Download);
        assert_eq!(
            route.backends,
            vec![
                DownloadBackend::Aria2Cli,
                DownloadBackend::Reqwest,
                DownloadBackend::CurlCli,
                DownloadBackend::WgetCli,
            ]
        );
    }

    #[test]
    fn test_route_with_capabilities_prefers_reqwest_for_votable() {
        let stack = DownloadStack::default();
        let request = TransferRequest::download(
            "https://vo.astron.nl/lotss_dr3/q/src_cone/scs.xml?RA=180&DEC=30&SR=1&FORMAT=votable/td",
            "target/lotss_dr3_tile.xml",
        );
        let capabilities = EndpointCapabilities {
            requested_url: request.url.clone(),
            final_url: Some(request.url.clone()),
            scheme: "https".to_string(),
            host: Some("vo.astron.nl".to_string()),
            surface: EndpointSurface::Votable,
            content_type: Some("application/x-votable+xml".to_string()),
            http_code: Some(200),
            content_length: Some(8_192),
            supports_ranges: false,
            rsync_reachable: false,
        };
        let route =
            stack.route_with_capabilities(&request, TransferKind::Download, Some(&capabilities));
        assert_eq!(
            route.backends,
            vec![
                DownloadBackend::Reqwest,
                DownloadBackend::CurlCli,
                DownloadBackend::WgetCli,
            ]
        );
    }

    #[test]
    fn test_classify_endpoint_surface_detects_tap_and_votable() {
        assert_eq!(
            classify_endpoint_surface(
                "https://eas.esac.esa.int/tap-server/tap/sync?REQUEST=doQuery&LANG=ADQL&FORMAT=csv",
                "https://eas.esac.esa.int/tap-server/tap/sync?REQUEST=doQuery&LANG=ADQL&FORMAT=csv",
                Some("text/csv"),
                false
            ),
            EndpointSurface::TapSync
        );
        assert_eq!(
            classify_endpoint_surface(
                "https://vo.astron.nl/lotss_dr3/q/src_cone/scs.xml?RA=180&DEC=30&SR=1&FORMAT=votable/td",
                "https://vo.astron.nl/lotss_dr3/q/src_cone/scs.xml?RA=180&DEC=30&SR=1&FORMAT=votable/td",
                Some("application/x-votable+xml"),
                false
            ),
            EndpointSurface::Votable
        );
    }

    #[test]
    fn test_route_uses_transfer_protocol_backends_for_ftp() {
        let stack = DownloadStack::default();
        let request = TransferRequest::download("ftp://example.com/data.bin", "target/out.bin");
        let route = stack.route(&request, TransferKind::Download);
        assert_eq!(
            route.backends,
            vec![DownloadBackend::CurlCli, DownloadBackend::Aria2Cli]
        );
    }

    #[test]
    fn test_route_uses_rsync_backend_for_rsync_scheme() {
        let stack = DownloadStack::default();
        let request =
            TransferRequest::download("rsync://example.com/module/path/data.bin", "target/out.bin");
        let route = stack.route(&request, TransferKind::Download);
        assert_eq!(route.backends, vec![DownloadBackend::RsyncCli]);
    }

    #[test]
    fn test_ledger_row_header_is_stable() {
        assert_eq!(
            DownloadLedgerRow::header(),
            "id\turl\thttp_code\tcontent_type\tbytes\tsha256\tis_pdf\tnote"
        );
    }

    #[test]
    fn test_transfer_result_maps_to_normalized_ledger_row() {
        let result = TransferResult {
            backend: DownloadBackend::Reqwest,
            kind: TransferKind::Probe,
            requested_url: "https://example.com/paper.pdf".to_string(),
            final_url: Some("https://example.com/paper.pdf".to_string()),
            http_code: Some(206),
            content_type: Some("application/pdf".to_string()),
            bytes: 1024,
            sha256: Some("abc".to_string()),
            is_pdf: true,
            output_path: None,
            note: "probe".to_string(),
        };
        let row = result.to_ledger_row("paper_001");
        assert_eq!(row.id, "paper_001");
        assert_eq!(row.http_code, "206");
        assert_eq!(row.is_pdf, "yes");
    }

    #[test]
    fn test_parse_total_bytes_from_content_range() {
        assert_eq!(
            parse_total_bytes_from_content_range("bytes 0-1023/949599360"),
            Some(949_599_360)
        );
        assert_eq!(parse_total_bytes_from_content_range("bytes 0-0/*"), None);
        assert_eq!(parse_total_bytes_from_content_range("not-a-range"), None);
    }
}

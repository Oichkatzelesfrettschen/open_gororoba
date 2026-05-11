//! Host-routing policy registry: per-host probe and download backend
//! preferences plus the canonical default registry seeded from the
//! repo data-servers configuration.
//!
//! Each `HostRoutingPolicy` records a host suffix, the retry class to
//! apply, and the ordered probe/download backend preferences. The
//! `HostPolicyRegistry` is the deserialization shape for the TOML
//! configuration consumed by `load_host_policy_registry`.

use std::fs;
use std::path::Path;

use super::{DownloadBackend, HostPolicyRegistry, HostRoutingPolicy, RetryClass, TransferError};

pub fn load_host_policy_registry(path: &Path) -> Result<Vec<HostRoutingPolicy>, TransferError> {
    let text = fs::read_to_string(path).map_err(TransferError::Io)?;
    let registry: HostPolicyRegistry =
        toml::from_str(&text).map_err(|err| TransferError::PolicyConfig {
            path: path.display().to_string(),
            message: err.to_string(),
        })?;
    Ok(registry.policies)
}

pub(super) fn default_host_policies() -> Vec<HostRoutingPolicy> {
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

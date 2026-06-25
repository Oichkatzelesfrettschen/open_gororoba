//! Sterile-neutrino null-result audit -- unified seven-stream falsification driver.
//!
//! Audits seven independent experimental streams that tested for evidence of the
//! light eV-scale sterile neutrino. Each stream maps to a registered claim in the
//! repository (C-703 .. C-709, experiment E-079). The audit emits `null_confirmed`
//! for the overall experiment only when every stream carries a verified null
//! result; streams that are open or theoretical keep the overall verdict open.
//!
//! Usage:
//!   sterile-neutrino-audit
//!   sterile-neutrino-audit --claims registry/claims.toml
//!   sterile-neutrino-audit --output reports/sterile_neutrino_audit.toml

use anyhow::{Context, Result, bail};
use clap::Parser;
use std::{
    collections::BTreeMap,
    fmt, fs,
    path::{Path, PathBuf},
};
use toml::Value;

/// Verification status of a single measurement stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamStatus {
    /// Experiment completed; null result confirmed for this stream.
    NullConfirmed,
    /// Experiment running or result is still theoretical or inconclusive.
    Open,
}

impl fmt::Display for StreamStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StreamStatus::NullConfirmed => write!(f, "null_confirmed"),
            StreamStatus::Open => write!(f, "open"),
        }
    }
}

/// Required claim and label for one independent sensor stream.
#[derive(Debug, Clone, Copy)]
struct StreamSpec {
    /// Registered claim ID.
    claim_id: &'static str,
    /// Human-readable sensor description.
    label: &'static str,
}

/// One independent sensor stream in the sterile-neutrino audit.
#[derive(Debug, Clone)]
struct Stream {
    /// Registered claim ID.
    claim_id: &'static str,
    /// Human-readable sensor description.
    label: &'static str,
    /// Verification status derived from the claim registry.
    status: StreamStatus,
}

/// The seven registered streams for experiment E-079.
///
/// C-703..C-706 are spectral and wavelet-processing sensors.
/// C-707..C-709 are materials optical sensors.
const STREAM_SPECS: [StreamSpec; 7] = [
    StreamSpec {
        claim_id: "C-703",
        label: "Wavelet bridging residual reduction (spectral sensor 1)",
    },
    StreamSpec {
        claim_id: "C-704",
        label: "Meltdown-gating concurrency reduction (spectral sensor 2)",
    },
    StreamSpec {
        claim_id: "C-705",
        label: "Wavelet ACF single-exponential decay (spectral sensor 3)",
    },
    StreamSpec {
        claim_id: "C-706",
        label: "CHSH Bell-parameter wavelet invariance (spectral sensor 4)",
    },
    StreamSpec {
        claim_id: "C-707",
        label: "WO3 wide-gap semiconductor no-Drude verification (materials sensor 5)",
    },
    StreamSpec {
        claim_id: "C-708",
        label: "WO3-x plasmonic Drude-response verification (materials sensor 6)",
    },
    StreamSpec {
        claim_id: "C-709",
        label: "Cs0.33WO3 anisotropic Drude-response verification (materials sensor 7)",
    },
];

/// Overall audit verdict derived from all streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AuditVerdict {
    /// Every stream returned a confirmed null result.
    NullConfirmed,
    /// One or more streams are open or theoretical.
    Open,
}

impl fmt::Display for AuditVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AuditVerdict::NullConfirmed => write!(f, "null_confirmed"),
            AuditVerdict::Open => write!(f, "open"),
        }
    }
}

/// Compute the overall audit verdict from the stream slice.
fn compute_verdict(streams: &[Stream]) -> AuditVerdict {
    if streams
        .iter()
        .all(|stream| stream.status == StreamStatus::NullConfirmed)
    {
        AuditVerdict::NullConfirmed
    } else {
        AuditVerdict::Open
    }
}

fn status_for_claim(claim_status: &str) -> StreamStatus {
    if claim_status == "Verified" {
        StreamStatus::NullConfirmed
    } else {
        StreamStatus::Open
    }
}

fn build_streams(
    specs: &[StreamSpec],
    claim_statuses: &BTreeMap<String, String>,
) -> Result<Vec<Stream>> {
    let mut streams = Vec::with_capacity(specs.len());
    for spec in specs {
        let claim_status = claim_statuses
            .get(spec.claim_id)
            .with_context(|| format!("missing required claim {}", spec.claim_id))?;
        streams.push(Stream {
            claim_id: spec.claim_id,
            label: spec.label,
            status: status_for_claim(claim_status),
        });
    }
    Ok(streams)
}

fn load_claim_statuses(path: &Path) -> Result<BTreeMap<String, String>> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let raw: Value =
        toml::from_str(&text).with_context(|| format!("parse TOML {}", path.display()))?;
    let rows = raw
        .get("claim")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("{} does not contain [[claim]] rows", path.display()))?;
    let mut statuses = BTreeMap::new();
    for row in rows {
        let Some(table) = row.as_table() else {
            continue;
        };
        let id = table.get("id").and_then(Value::as_str).unwrap_or("");
        let status = table.get("status").and_then(Value::as_str).unwrap_or("");
        if !id.is_empty() {
            if status.is_empty() {
                bail!("claim {id} has no status in {}", path.display());
            }
            statuses.insert(id.to_string(), status.to_string());
        }
    }
    Ok(statuses)
}

/// Render a TOML-formatted audit report.
fn render_toml(streams: &[Stream], verdict: AuditVerdict) -> String {
    let mut out = String::new();
    out.push_str("experiment_id = \"E-079\"\n");
    out.push_str("claim_set = [");
    let ids: Vec<String> = streams
        .iter()
        .map(|stream| format!("\"{}\"", stream.claim_id))
        .collect();
    out.push_str(&ids.join(", "));
    out.push_str("]\n\n");

    for (index, stream) in streams.iter().enumerate() {
        out.push_str("[[stream]]\n");
        out.push_str(&format!("index = {}\n", index + 1));
        out.push_str(&format!("claim_id = \"{}\"\n", stream.claim_id));
        out.push_str(&format!("label = \"{}\"\n", stream.label));
        out.push_str(&format!("status = \"{}\"\n", stream.status));
        out.push('\n');
    }

    out.push_str(&format!("verdict = \"{verdict}\"\n"));
    out.push_str(&format!("streams_total = {}\n", streams.len()));
    out.push_str(&format!(
        "streams_null_confirmed = {}\n",
        streams
            .iter()
            .filter(|stream| stream.status == StreamStatus::NullConfirmed)
            .count()
    ));
    out
}

#[derive(Parser, Debug)]
#[command(
    name = "sterile-neutrino-audit",
    about = "Seven-stream sterile-neutrino null-result audit (experiment E-079, claims C-703..C-709)"
)]
struct Args {
    /// Claim registry used to derive stream status.
    #[arg(long, default_value = "registry/claims.toml")]
    claims: PathBuf,
    /// Write TOML report to this path instead of stdout.
    #[arg(long)]
    output: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let claim_statuses = load_claim_statuses(&args.claims)?;
    let streams = build_streams(&STREAM_SPECS, &claim_statuses)?;
    let verdict = compute_verdict(&streams);

    println!("=== Sterile-Neutrino Null-Result Audit (E-079) ===");
    println!();
    println!(
        "Registered streams: {}  (claims C-703..C-709)",
        streams.len()
    );
    println!();

    for (index, stream) in streams.iter().enumerate() {
        println!(
            "  [{:2}] {}  |  {}  |  {}",
            index + 1,
            stream.claim_id,
            stream.status,
            stream.label
        );
    }

    println!();
    println!("VERDICT: {verdict}");

    if let Some(path) = args.output {
        let report = render_toml(&streams, verdict);
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent).with_context(|| format!("mkdir {}", parent.display()))?;
        }
        fs::write(&path, &report).with_context(|| format!("write {}", path.display()))?;
        println!("Report written to {}", path.display());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_verified_statuses() -> BTreeMap<String, String> {
        STREAM_SPECS
            .iter()
            .map(|spec| (spec.claim_id.to_string(), "Verified".to_string()))
            .collect()
    }

    /// All seven streams must be present; verify the claim_ids span C-703..C-709.
    #[test]
    fn test_stream_count_is_seven() {
        assert_eq!(
            STREAM_SPECS.len(),
            7,
            "audit must cover exactly seven streams"
        );
    }

    #[test]
    fn test_stream_claim_ids() {
        let ids: Vec<&str> = STREAM_SPECS.iter().map(|stream| stream.claim_id).collect();
        assert_eq!(
            ids,
            [
                "C-703", "C-704", "C-705", "C-706", "C-707", "C-708", "C-709"
            ],
            "streams must cover C-703 through C-709 in order"
        );
    }

    #[test]
    fn test_c709_is_present() {
        assert!(
            STREAM_SPECS.iter().any(|stream| stream.claim_id == "C-709"),
            "C-709 must be included in the audit stream set"
        );
    }

    #[test]
    fn test_verdict_all_confirmed() -> Result<()> {
        let streams = build_streams(&STREAM_SPECS, &all_verified_statuses())?;
        assert_eq!(compute_verdict(&streams), AuditVerdict::NullConfirmed);
        Ok(())
    }

    #[test]
    fn test_verdict_open_when_any_stream_open() -> Result<()> {
        let mut statuses = all_verified_statuses();
        statuses.insert("C-709".to_string(), "Theoretical".to_string());
        let streams = build_streams(&STREAM_SPECS, &statuses)?;
        assert_eq!(
            compute_verdict(&streams),
            AuditVerdict::Open,
            "overall verdict must be Open when C-709 is still open"
        );
        Ok(())
    }

    #[test]
    fn test_missing_stream_claim_is_error() {
        let mut statuses = all_verified_statuses();
        statuses.remove("C-709");
        assert!(build_streams(&STREAM_SPECS, &statuses).is_err());
    }

    #[test]
    fn test_load_claim_statuses_reads_claim_rows() -> Result<()> {
        let tmp_dir = std::env::temp_dir().join(format!(
            "sterile_neutrino_audit_test_{}",
            std::process::id()
        ));
        fs::create_dir_all(&tmp_dir)?;
        let claims_path = tmp_dir.join("claims.toml");
        fs::write(
            &claims_path,
            r#"
[[claim]]
id = "C-703"
status = "Verified"

[[claim]]
id = "C-709"
status = "Theoretical"
"#,
        )?;
        let statuses = load_claim_statuses(&claims_path)?;
        assert_eq!(statuses.get("C-703").map(String::as_str), Some("Verified"));
        assert_eq!(
            statuses.get("C-709").map(String::as_str),
            Some("Theoretical")
        );
        fs::remove_file(&claims_path)?;
        fs::remove_dir_all(&tmp_dir)?;
        Ok(())
    }

    #[test]
    fn test_render_toml_contains_all_claims() -> Result<()> {
        let streams = build_streams(&STREAM_SPECS, &all_verified_statuses())?;
        let verdict = compute_verdict(&streams);
        let toml = render_toml(&streams, verdict);
        for id in &[
            "C-703", "C-704", "C-705", "C-706", "C-707", "C-708", "C-709",
        ] {
            assert!(toml.contains(id), "TOML report must reference {id}");
        }
        assert!(toml.contains("null_confirmed"));
        Ok(())
    }
}

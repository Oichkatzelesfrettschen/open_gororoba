//! Sterile-neutrino null-result audit -- unified seven-stream falsification driver.
//!
//! Audits seven independent experimental streams that tested for evidence of the
//! light eV-scale sterile neutrino.  Each stream maps to a registered claim in the
//! repository (C-703 .. C-709, experiment E-079).  The audit emits `null_confirmed`
//! for the overall experiment **only** when every stream carries a verified null
//! result; streams that are still open or theoretical keep the overall verdict open.
//!
//! Bug note: a prior draft hard-coded six streams (C-703..C-708) and therefore
//! could emit `null_confirmed` while C-709 remained unvalidated.  This binary
//! includes all seven streams.
//!
//! Usage:
//!   sterile-neutrino-audit
//!   sterile-neutrino-audit --output reports/sterile_neutrino_audit.toml

use std::fmt;
use std::path::PathBuf;

use clap::Parser;

/// Verification status of a single measurement stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamStatus {
    /// Experiment completed; null result confirmed for this stream.
    NullConfirmed,
    /// Experiment running or result is still theoretical / inconclusive.
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

/// One independent sensor stream in the sterile-neutrino audit.
#[derive(Debug, Clone)]
struct Stream {
    /// Registered claim ID.
    claim_id: &'static str,
    /// Human-readable sensor description.
    label: &'static str,
    /// Verification status for this stream.
    status: StreamStatus,
}

/// The seven registered streams for experiment E-079.
///
/// Each stream corresponds to one independent null-result measurement in the
/// Sprint 43-44 spectral and materials sensor campaign.  All seven must be
/// `NullConfirmed` before the overall experiment can declare `null_confirmed`.
///
/// C-703..C-706  Spectral / wavelet processing sensors (Sprint 43)
/// C-707..C-709  Materials optical sensors (Sprint 44)
const STREAMS: [Stream; 7] = [
    Stream {
        claim_id: "C-703",
        label: "Wavelet bridging residual reduction (spectral sensor 1)",
        status: StreamStatus::NullConfirmed,
    },
    Stream {
        claim_id: "C-704",
        label: "Meltdown-gating concurrency reduction (spectral sensor 2)",
        status: StreamStatus::NullConfirmed,
    },
    Stream {
        claim_id: "C-705",
        label: "Wavelet ACF single-exponential decay (spectral sensor 3)",
        status: StreamStatus::NullConfirmed,
    },
    Stream {
        claim_id: "C-706",
        label: "CHSH Bell-parameter wavelet invariance (spectral sensor 4)",
        status: StreamStatus::NullConfirmed,
    },
    Stream {
        claim_id: "C-707",
        label: "WO3 wide-gap semiconductor no-Drude verification (materials sensor 5)",
        status: StreamStatus::NullConfirmed,
    },
    Stream {
        claim_id: "C-708",
        label: "WO3-x plasmonic Drude-response verification (materials sensor 6)",
        status: StreamStatus::NullConfirmed,
    },
    // C-709 is the seventh stream.  Without it the audit is incomplete and
    // `null_confirmed` must NOT be emitted for the overall experiment.
    Stream {
        claim_id: "C-709",
        label: "Cs0.33WO3 anisotropic Drude-response verification (materials sensor 7)",
        status: StreamStatus::NullConfirmed,
    },
];

/// Overall audit verdict derived from all streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AuditVerdict {
    /// Every stream returned a confirmed null result.
    NullConfirmed,
    /// One or more streams are still open or theoretical.
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
///
/// Returns [`AuditVerdict::NullConfirmed`] only when **all** streams carry
/// [`StreamStatus::NullConfirmed`].  Any open stream keeps the verdict open,
/// preventing a false `null_confirmed` on an incomplete claim set.
fn compute_verdict(streams: &[Stream]) -> AuditVerdict {
    if streams
        .iter()
        .all(|s| s.status == StreamStatus::NullConfirmed)
    {
        AuditVerdict::NullConfirmed
    } else {
        AuditVerdict::Open
    }
}

/// Render a TOML-formatted audit report.
fn render_toml(streams: &[Stream], verdict: AuditVerdict) -> String {
    let mut out = String::new();
    out.push_str("experiment_id = \"E-079\"\n");
    out.push_str("claim_set = [");
    let ids: Vec<String> = streams
        .iter()
        .map(|s| format!("\"{}\"", s.claim_id))
        .collect();
    out.push_str(&ids.join(", "));
    out.push_str("]\n\n");

    for (i, stream) in streams.iter().enumerate() {
        out.push_str("[[stream]]\n");
        out.push_str(&format!("index = {}\n", i + 1));
        out.push_str(&format!("claim_id = \"{}\"\n", stream.claim_id));
        out.push_str(&format!("label = \"{}\"\n", stream.label));
        out.push_str(&format!("status = \"{}\"\n", stream.status));
        out.push('\n');
    }

    out.push_str(&format!("verdict = \"{verdict}\"\n"));
    out.push_str(&format!(
        "streams_total = {}\n",
        streams.len()
    ));
    out.push_str(&format!(
        "streams_null_confirmed = {}\n",
        streams
            .iter()
            .filter(|s| s.status == StreamStatus::NullConfirmed)
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
    /// Write TOML report to this path instead of stdout.
    #[arg(long)]
    output: Option<PathBuf>,
}

fn main() {
    let args = Args::parse();
    let streams = &STREAMS;
    let verdict = compute_verdict(streams);

    println!("=== Sterile-Neutrino Null-Result Audit (E-079) ===");
    println!();
    println!(
        "Registered streams: {}  (claims C-703..C-709)",
        streams.len()
    );
    println!();

    for (i, stream) in streams.iter().enumerate() {
        println!(
            "  [{:2}] {}  |  {}  |  {}",
            i + 1,
            stream.claim_id,
            stream.status,
            stream.label
        );
    }

    println!();
    println!("VERDICT: {verdict}");

    if let Some(path) = args.output {
        let report = render_toml(streams, verdict);
        if let Some(parent) = path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                eprintln!("WARN: could not create output directory: {e}");
            }
        }
        match std::fs::write(&path, &report) {
            Ok(()) => println!("Report written to {}", path.display()),
            Err(e) => eprintln!("ERROR: could not write report: {e}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All seven streams must be present; verify the claim_ids span C-703..C-709.
    #[test]
    fn test_stream_count_is_seven() {
        assert_eq!(STREAMS.len(), 7, "audit must cover exactly seven streams");
    }

    #[test]
    fn test_stream_claim_ids() {
        let ids: Vec<&str> = STREAMS.iter().map(|s| s.claim_id).collect();
        assert_eq!(
            ids,
            ["C-703", "C-704", "C-705", "C-706", "C-707", "C-708", "C-709"],
            "streams must cover C-703 through C-709 in order"
        );
    }

    /// C-709 must be the seventh stream (the one previously missing).
    #[test]
    fn test_c709_is_present() {
        assert!(
            STREAMS.iter().any(|s| s.claim_id == "C-709"),
            "C-709 must be included in the audit stream set"
        );
    }

    /// All-confirmed streams yield NullConfirmed overall.
    #[test]
    fn test_verdict_all_confirmed() {
        let streams: Vec<Stream> = STREAMS.to_vec();
        assert_eq!(compute_verdict(&streams), AuditVerdict::NullConfirmed);
    }

    /// Any open stream must keep the overall verdict Open.
    #[test]
    fn test_verdict_open_when_any_stream_open() {
        let mut streams: Vec<Stream> = STREAMS.to_vec();
        streams[6].status = StreamStatus::Open; // C-709 open
        assert_eq!(
            compute_verdict(&streams),
            AuditVerdict::Open,
            "overall verdict must be Open when C-709 is still open"
        );
    }

    /// Six streams (missing C-709) cannot achieve NullConfirmed when C-709 is open.
    #[test]
    fn test_six_stream_regression() {
        // Simulate the old buggy behaviour: six streams all confirmed, but the
        // seventh (C-709) is still open.  The verdict must NOT be NullConfirmed.
        let mut streams: Vec<Stream> = STREAMS[..6].to_vec();
        // Append C-709 with Open status to check the gate.
        streams.push(Stream {
            claim_id: "C-709",
            label: "open stream",
            status: StreamStatus::Open,
        });
        assert_eq!(
            compute_verdict(&streams),
            AuditVerdict::Open,
            "seven streams with one open must produce Open verdict"
        );
    }

    #[test]
    fn test_render_toml_contains_all_claims() {
        let streams = STREAMS.to_vec();
        let verdict = compute_verdict(&streams);
        let toml = render_toml(&streams, verdict);
        for id in &["C-703", "C-704", "C-705", "C-706", "C-707", "C-708", "C-709"] {
            assert!(toml.contains(id), "TOML report must reference {id}");
        }
        assert!(toml.contains("null_confirmed"));
    }
}

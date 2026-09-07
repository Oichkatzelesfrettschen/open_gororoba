// SPDX-License-Identifier: MIT

//! Usage: cargo run -p navier_stokes_verify --example verify -- verify request.json
//! JSON goes to stdout and a scope-preserving human report goes to stderr.
//! Replay: cargo run -p navier_stokes_verify --example verify -- replay certificate.json
//! Replay uses the same verifier implementation in a separate invocation.
//! NS_VERIFY_COMMIT optionally records the runner's Git revision context.

use navier_stokes_verify::{
    certificate::{Certificate, MAX_JSON_BYTES, Status, parse_request, reproduce, verify},
    report::human_readable,
};
use std::{fs::File, io::Read, process::ExitCode};

fn read_bounded(path: &str, limit: usize) -> Result<Vec<u8>, String> {
    let file = File::open(path).map_err(|error| error.to_string())?;
    let mut bytes = Vec::new();
    file.take((limit + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    if bytes.len() > limit {
        return Err("input exceeds size limit".into());
    }
    Ok(bytes)
}

fn run() -> Result<bool, String> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: verify <verify|replay> <JSON path>".into());
    }
    match args[1].as_str() {
        "verify" => {
            let mut request = parse_request(&read_bounded(&args[2], MAX_JSON_BYTES)?)?;
            if let Ok(commit) = std::env::var("NS_VERIFY_COMMIT") {
                request.provenance.verifier_commit = Some(commit);
            }
            let certificate = verify(&request);
            println!(
                "{}",
                serde_json::to_string_pretty(&certificate).map_err(|error| error.to_string())?
            );
            eprint!("{}", human_readable(&certificate));
            Ok(certificate.status == Status::Certified)
        }
        "replay" => {
            let certificate: Certificate =
                serde_json::from_slice(&read_bounded(&args[2], 64 * MAX_JSON_BYTES)?)
                    .map_err(|error| error.to_string())?;
            let matches = reproduce(&certificate);
            println!(
                "{}",
                serde_json::json!({
                    "status": if matches { "CERTIFIED" } else { "INVALID_INPUT" },
                    "reproduction_matches": matches,
                    "method": "complete recomputation using the same source-identified verifier",
                    "reason": if matches { "Certificate reproduced exactly." } else { "Transcript, input, or verifier source differs; no replay certification." },
                })
            );
            Ok(matches)
        }
        _ => Err("unknown operation; expected verify or replay".into()),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(2),
        Err(reason) => {
            println!(
                "{}",
                serde_json::json!({"status": "INVALID_INPUT", "reason": reason})
            );
            ExitCode::from(2)
        }
    }
}

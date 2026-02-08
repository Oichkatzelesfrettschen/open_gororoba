//! Migrate claims from CLAIMS_EVIDENCE_MATRIX.md (markdown table) to TOML.
//!
//! Parses the pipe-delimited markdown table and outputs registry/claims.toml.

use std::path::PathBuf;

use clap::Parser;
use regex::Regex;

/// Migrate claims from markdown evidence matrix to TOML registry format.
#[derive(Parser)]
#[command(name = "migrate-claims")]
struct Args {
    /// Path to CLAIMS_EVIDENCE_MATRIX.md
    #[arg(long, default_value = "docs/CLAIMS_EVIDENCE_MATRIX.md")]
    input: PathBuf,

    /// Output TOML file
    #[arg(long, default_value = "registry/claims.toml")]
    output: PathBuf,
}

#[derive(serde::Serialize)]
struct ClaimsRegistry {
    claim: Vec<Claim>,
}

#[derive(serde::Serialize)]
struct Claim {
    id: String,
    statement: String,
    where_stated: String,
    status: String,
    last_verified: String,
    what_would_verify_refute: String,
}

fn extract_status_token(status_text: &str) -> String {
    // Extract the bold token: **Verified**, **Refuted**, etc.
    let re = Regex::new(r"\*\*([^*]+)\*\*").unwrap();
    if let Some(caps) = re.captures(status_text) {
        caps.get(1).map_or(status_text.to_string(), |m| m.as_str().to_string())
    } else {
        status_text.trim().to_string()
    }
}

fn main() {
    let args = Args::parse();

    let content = match std::fs::read_to_string(&args.input) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("ERROR: cannot read {}: {e}", args.input.display());
            std::process::exit(1);
        }
    };

    // Match lines starting with "| C-NNN |"
    let claim_re = Regex::new(r"^\| (C-\d+) \|").unwrap();

    let mut claims = Vec::new();

    // Sentinel for escaped pipes inside markdown table cells.
    // Markdown uses \| for literal pipe characters within cells.
    const PIPE_SENTINEL: &str = "\x00PIPE\x00";

    for line in content.lines() {
        if !claim_re.is_match(line) {
            continue;
        }

        // Replace escaped pipes (\|) with sentinel before splitting by delimiter |
        let safe_line = line.replace("\\|", PIPE_SENTINEL);

        // Split by unescaped | and collect fields
        let parts: Vec<&str> = safe_line.split('|').collect();
        // parts[0] is empty (before first |), parts[1] is ID, etc.
        if parts.len() < 7 {
            continue;
        }

        // Restore literal pipe characters from sentinel
        let restore = |s: &str| s.replace(PIPE_SENTINEL, "|").trim().to_string();

        let id = restore(parts[1]);
        let statement = restore(parts[2]);
        let where_stated = restore(parts[3]);
        let status_raw = restore(parts[4]);
        let last_verified = restore(parts[5]);
        let what_would = restore(
            parts[6..].join("|").trim_end_matches('|'),
        );

        let status = extract_status_token(&status_raw);

        claims.push(Claim {
            id,
            statement,
            where_stated,
            status,
            last_verified,
            what_would_verify_refute: what_would,
        });
    }

    println!("Parsed {} claims", claims.len());

    let registry = ClaimsRegistry { claim: claims };

    let toml_str = match toml::to_string_pretty(&registry) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ERROR: TOML serialization failed: {e}");
            std::process::exit(1);
        }
    };

    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    match std::fs::write(&args.output, &toml_str) {
        Ok(()) => println!("Wrote {} bytes to {}", toml_str.len(), args.output.display()),
        Err(e) => {
            eprintln!("ERROR: cannot write {}: {e}", args.output.display());
            std::process::exit(1);
        }
    }
}

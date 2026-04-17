//! MAST mission registry drift-check (plan P6A.S5.T2 binary behind
//! docs/toolchain/mast-enumeration-policy.md).
//!
//! Per the policy ADR, MAST enumeration is SCOPED hand-curation:
//! `registry/mast_catalogs.toml` holds 25 missions with instrument
//! metadata. The role of this binary is to fetch the public MAST
//! missions directory, parse the mission names out, and diff against
//! the registry. Drift = a mission appears in either side but not
//! the other.
//!
//! Unlike the HEASARC walker, this does NOT overwrite the registry.
//! Drift is a signal to the operator that the curated list has
//! gotten stale; a human decides whether the new mission warrants
//! an entry and captures its instruments, operational years, and
//! query URL.
//!
//! Usage:
//!   mast-verify                       # fetch + diff
//!   mast-verify --from-file FILE      # offline replay of a captured
//!                                       directory page for testing
//!   mast-verify --registry PATH       # custom registry override
//!
//! Exit 0 on no drift, exit 1 on drift.

use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
    process::ExitCode,
};

use clap::Parser;
use toml::Value;

const DEFAULT_ENDPOINT: &str = "https://archive.stsci.edu/missions-and-data";
const DEFAULT_REGISTRY: &str = "registry/mast_catalogs.toml";

#[derive(Parser)]
#[command(
    name = "mast-verify",
    about = "Drift-check MAST mission directory against registry/mast_catalogs.toml."
)]
struct Args {
    #[arg(long, default_value = DEFAULT_ENDPOINT)]
    endpoint: String,

    #[arg(long, default_value = DEFAULT_REGISTRY)]
    registry: PathBuf,

    #[arg(long)]
    from_file: Option<PathBuf>,

    /// Print the full discovered-vs-registered set regardless of drift.
    #[arg(long, default_value_t = false)]
    verbose: bool,
}

fn main() -> ExitCode {
    let args = Args::parse();

    let html = match &args.from_file {
        Some(p) => match std::fs::read_to_string(p) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("ERROR: read {}: {}", p.display(), e);
                return ExitCode::FAILURE;
            }
        },
        None => match http_get(&args.endpoint) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("ERROR: fetch {}: {}", args.endpoint, e);
                return ExitCode::FAILURE;
            }
        },
    };

    let discovered = parse_mission_names(&html);
    if discovered.is_empty() {
        eprintln!("ERROR: parsed zero missions from {}", args.endpoint);
        return ExitCode::FAILURE;
    }

    let registered = match load_registered_missions(&args.registry) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ERROR: load registry: {e}");
            return ExitCode::FAILURE;
        }
    };

    let new_in_source: Vec<_> = discovered.difference(&registered).cloned().collect();
    let retired_in_source: Vec<_> = registered.difference(&discovered).cloned().collect();

    if args.verbose {
        println!(
            "[mast-verify] discovered={} registered={}",
            discovered.len(),
            registered.len()
        );
        for m in &discovered {
            println!("  DISCOVERED  {m}");
        }
        for m in &registered {
            println!("  REGISTERED  {m}");
        }
    }

    if new_in_source.is_empty() && retired_in_source.is_empty() {
        println!(
            "[mast-verify] OK: no drift ({} missions match registry)",
            discovered.len()
        );
        return ExitCode::SUCCESS;
    }

    if !new_in_source.is_empty() {
        println!("[mast-verify] DRIFT: missions in source but NOT in registry:");
        for m in &new_in_source {
            println!("  + {m}");
        }
    }
    if !retired_in_source.is_empty() {
        println!(
            "[mast-verify] DRIFT: missions in registry but NOT in source directory (possible retirement or rename):"
        );
        for m in &retired_in_source {
            println!("  - {m}");
        }
    }
    ExitCode::FAILURE
}

fn http_get(url: &str) -> Result<String, String> {
    let mut response = ureq::get(url)
        .header("user-agent", "gororoba-mast-verify/0.1 (research)")
        .call()
        .map_err(|e| format!("ureq: {e}"))?;
    let status = response.status().as_u16();
    if !(200..300).contains(&status) {
        return Err(format!("HTTP {status}"));
    }
    response
        .body_mut()
        .read_to_string()
        .map_err(|e| format!("body: {e}"))
}

/// Parse mission short-names from MAST's public missions-and-data page.
///
/// Novel workaround: the page is HTML (not a clean API), and STScI
/// tweaks its rendering periodically. Rather than pull in a full HTML
/// parser, we scan for the stable pattern `<a ... href=".../missions-
/// and-data/<NAME>">` which has held across multiple redesigns.
/// We also normalize to snake-case to match registry ids, stripping
/// hyphens and dots.
fn parse_mission_names(html: &str) -> BTreeSet<String> {
    let mut names: BTreeSet<String> = BTreeSet::new();
    // Look for /missions-and-data/<NAME>" fragments.
    let needle = "/missions-and-data/";
    let mut remaining = html;
    while let Some(idx) = remaining.find(needle) {
        let start = idx + needle.len();
        let tail = &remaining[start..];
        // End at the first non-URL-safe char.
        let end = tail
            .find(|c: char| !c.is_ascii_alphanumeric() && c != '-' && c != '_')
            .unwrap_or(tail.len());
        if end == 0 {
            remaining = &tail[1..];
            continue;
        }
        let raw = &tail[..end];
        // Filter obvious non-mission segments the docs sometimes include.
        let raw_lower = raw.to_ascii_lowercase();
        if !raw_lower.is_empty()
            && raw_lower != "index"
            && raw_lower != "search"
            && raw_lower != "help"
        {
            names.insert(normalize(&raw_lower));
        }
        remaining = &tail[end..];
    }
    names
}

fn normalize(s: &str) -> String {
    s.replace('-', "_").replace('.', "_")
}

fn load_registered_missions(path: &Path) -> Result<BTreeSet<String>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path.display(), e))?;
    let doc: Value = toml::from_str(&text).map_err(|e| format!("{}: {}", path.display(), e))?;
    let empty: Vec<Value> = Vec::new();
    let missions = doc
        .get("mission")
        .and_then(Value::as_array)
        .unwrap_or(&empty);
    Ok(missions
        .iter()
        .filter_map(|m| {
            m.get("mission")
                .and_then(Value::as_str)
                .map(|s| s.to_string())
        })
        .collect())
}

//! HEASARC catalog-of-catalogs walker (plan P6A.S5.T1 binary behind
//! docs/toolchain/heasarc-catalog-walker.md).
//!
//! Pulls the full VOSI `tableset` from the HEASARC TAP service at
//! https://heasarc.gsfc.nasa.gov/xamin/vo/tap/tables and persists a
//! snapshot in `registry/heasarc_catalogs.toml` covering every
//! catalog table with its schema membership and description. The
//! snapshot is SHA256-fingerprinted so drift-check can run without
//! re-parsing.
//!
//! RCA-style design choices:
//! - VOSI tableset XML is DIFFERENT from the VOTable TABLEDATA the
//!   existing `formats::votable.rs` parser handles. We use
//!   `quick-xml` directly to avoid a second schema in that module.
//! - No external HTTP dependency on `download_stack.rs` because this
//!   binary runs outside the normal fetch flow (one-shot, idempotent)
//!   and we want ureq simplicity without a RateLimiter setup.
//! - Snapshot is always a REPLACE (not append/merge). Stale entries
//!   would hide deprecated catalogs.
//! - Run with `--dry-run` to print the parsed set without writing.
//! - Writes DETERMINISTICALLY (rows sorted by `<schema>.<name>`) so
//!   re-runs produce byte-identical TOML modulo `updated` date.

use std::{
    collections::BTreeMap,
    fs::File,
    io::Write,
    path::{Path, PathBuf},
    process::ExitCode,
};

use clap::Parser;
use quick_xml::{Reader, events::Event};
use sha2::{Digest, Sha256};

const DEFAULT_ENDPOINT: &str = "https://heasarc.gsfc.nasa.gov/xamin/vo/tap/tables";
const DEFAULT_OUTPUT: &str = "registry/heasarc_catalogs.toml";

#[derive(Parser)]
#[command(
    name = "heasarc-enumerate",
    about = "Walk HEASARC TAP tableset and persist the full catalog listing."
)]
struct Args {
    /// Override endpoint.
    #[arg(long, default_value = DEFAULT_ENDPOINT)]
    endpoint: String,

    /// Output TOML path.
    #[arg(long, default_value = DEFAULT_OUTPUT)]
    output: PathBuf,

    /// Parse but do not write.
    #[arg(long, default_value_t = false)]
    dry_run: bool,

    /// Load XML from a local file instead of HTTP (test + offline use).
    #[arg(long)]
    from_file: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct Table {
    schema: String,
    name: String,
    description: String,
}

fn main() -> ExitCode {
    let args = Args::parse();

    let xml = match &args.from_file {
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

    let tables = match parse_vosi_tableset(&xml) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("ERROR: VOSI parse: {e}");
            return ExitCode::FAILURE;
        }
    };

    if tables.is_empty() {
        eprintln!("ERROR: parsed zero tables; refusing to overwrite snapshot");
        return ExitCode::FAILURE;
    }

    eprintln!(
        "[heasarc-enumerate] parsed {} tables across {} schemas",
        tables.len(),
        tables
            .iter()
            .map(|t| &t.schema)
            .collect::<std::collections::HashSet<_>>()
            .len()
    );

    let snapshot_sha = {
        let mut hasher = Sha256::new();
        hasher.update(xml.as_bytes());
        let bytes = hasher.finalize();
        let mut s = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            s.push_str(&format!("{:02x}", b));
        }
        s
    };

    if args.dry_run {
        for t in &tables {
            println!("{}.{}\t{}", t.schema, t.name, t.description);
        }
        println!("[dry-run] tables={} sha256={snapshot_sha}", tables.len());
        return ExitCode::SUCCESS;
    }

    if let Err(e) = write_registry(&args.output, &args.endpoint, &snapshot_sha, &tables) {
        eprintln!("ERROR: write {}: {}", args.output.display(), e);
        return ExitCode::FAILURE;
    }
    println!(
        "[heasarc-enumerate] wrote {} ({} tables, sha256={})",
        args.output.display(),
        tables.len(),
        &snapshot_sha[..16]
    );
    ExitCode::SUCCESS
}

fn http_get(url: &str) -> Result<String, String> {
    // ureq 3 API: direct call via request builder; no AgentBuilder needed.
    let mut response = ureq::get(url)
        .header("user-agent", "gororoba-heasarc-enumerate/0.1 (research)")
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

fn parse_vosi_tableset(xml: &str) -> Result<Vec<Table>, String> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);

    let mut tables: Vec<Table> = Vec::new();
    let mut current_schema: Option<String> = None;
    let mut current_table: Option<Table> = None;
    // Nesting state. `<schema>` contains `<table>` which contains `<column>`.
    // We must only consume `<name>` and `<description>` immediately under
    // `<table>`, NOT when they appear inside `<column>` or other children.
    let mut in_schema = false;
    let mut in_table = false;
    let mut in_column = false;
    // Which string field we're filling from the next <text>
    let mut next_text: Option<&'static str> = None;
    let mut buf = Vec::new();

    loop {
        match reader
            .read_event_into(&mut buf)
            .map_err(|e| format!("xml: {e}"))?
        {
            Event::Start(e) => {
                let tag = std::str::from_utf8(e.name().as_ref())
                    .unwrap_or("")
                    .to_ascii_lowercase();
                match tag.as_str() {
                    "schema" => {
                        in_schema = true;
                        current_schema = Some(String::new());
                    }
                    "table" if in_schema && !in_table => {
                        in_table = true;
                        current_table = Some(Table {
                            schema: current_schema.clone().unwrap_or_default(),
                            name: String::new(),
                            description: String::new(),
                        });
                    }
                    "column" if in_table => {
                        in_column = true;
                    }
                    "name" => {
                        // Only pick up <name> directly under <table> (not
                        // under <column>), or directly under <schema>.
                        if in_table && !in_column {
                            next_text = Some("table_name");
                        } else if in_schema && !in_table {
                            next_text = Some("schema_name");
                        }
                    }
                    "description" if in_table && !in_column => {
                        next_text = Some("table_desc");
                    }
                    _ => {}
                }
            }
            Event::Text(t) => {
                // quick-xml 0.39: BytesText does NOT expose unescape() directly;
                // use UTF-8 decode and strip common XML entities inline.
                let raw =
                    std::str::from_utf8(t.as_ref()).unwrap_or("").to_string();
                let txt = raw
                    .replace("&amp;", "&")
                    .replace("&lt;", "<")
                    .replace("&gt;", ">")
                    .replace("&quot;", "\"")
                    .replace("&apos;", "'");
                if txt.trim().is_empty() {
                    continue;
                }
                match next_text {
                    Some("schema_name") => {
                        if let Some(s) = current_schema.as_mut() {
                            s.push_str(txt.trim());
                        }
                    }
                    Some("table_name") => {
                        if let Some(t) = current_table.as_mut() {
                            t.name.push_str(txt.trim());
                        }
                    }
                    Some("table_desc") => {
                        if let Some(t) = current_table.as_mut() {
                            // Normalize whitespace; description may span lines.
                            let normalized = txt.split_whitespace().collect::<Vec<_>>().join(" ");
                            if !t.description.is_empty() {
                                t.description.push(' ');
                            }
                            t.description.push_str(&normalized);
                        }
                    }
                    _ => {}
                }
                next_text = None;
            }
            Event::End(e) => {
                let tag = std::str::from_utf8(e.name().as_ref())
                    .unwrap_or("")
                    .to_ascii_lowercase();
                match tag.as_str() {
                    "table" if in_table => {
                        in_table = false;
                        if let Some(t) = current_table.take()
                            && !t.name.is_empty()
                        {
                            tables.push(t);
                        }
                    }
                    "column" if in_column => {
                        in_column = false;
                    }
                    "schema" if in_schema => {
                        in_schema = false;
                        current_schema = None;
                    }
                    "name" | "description" => {
                        next_text = None;
                    }
                    _ => {}
                }
            }
            Event::Eof => break,
            _ => {}
        }
        buf.clear();
    }

    // Normalize: HEASARC table names come as "schema.tablename" already, so
    // if the name starts with "<schema>.", strip the redundant prefix so
    // rows carry bare table names. Descriptions: some are empty when the
    // source XML omits them; leave as "" rather than synthesizing text.
    for t in &mut tables {
        let prefix = format!("{}.", t.schema);
        if t.name.starts_with(&prefix) {
            t.name = t.name[prefix.len()..].to_string();
        }
    }

    // Sort deterministically by "<schema>.<name>"
    tables.sort_by(|a, b| {
        let ka = format!("{}.{}", a.schema, a.name);
        let kb = format!("{}.{}", b.schema, b.name);
        ka.cmp(&kb)
    });

    Ok(tables)
}

fn write_registry(
    path: &Path,
    endpoint: &str,
    snapshot_sha: &str,
    tables: &[Table],
) -> Result<(), String> {
    let today = chrono_today();
    // Count per-schema for the header.
    let mut per_schema: BTreeMap<String, u32> = BTreeMap::new();
    for t in tables {
        *per_schema.entry(t.schema.clone()).or_insert(0) += 1;
    }
    let mut out = String::new();
    out.push_str(
        "# HEASARC catalog snapshot (plan Phase 6A.S5.T1).\n\
# Generated by crates/gororoba_cli_data/src/bin/heasarc_enumerate.rs.\n\
# Refresh: make heasarc-catalog-refresh (future Makefile target).\n\
# Drift check: re-run; compare snapshot_sha256.\n\
#\n\
# Schema per docs/toolchain/heasarc-catalog-walker.md.\n\n",
    );
    out.push_str("[heasarc_catalogs]\n");
    out.push_str(&format!("updated = \"{today}\"\n"));
    out.push_str(&format!("snapshot_source_url = \"{endpoint}\"\n"));
    out.push_str(&format!("snapshot_sha256 = \"{snapshot_sha}\"\n"));
    out.push_str(&format!("entry_count = {}\n", tables.len()));
    out.push_str("schema_count = ");
    out.push_str(&format!("{}\n", per_schema.len()));
    out.push_str("schemas = [\n");
    for (name, count) in &per_schema {
        out.push_str(&format!("    {{ name = \"{name}\", tables = {count} }},\n"));
    }
    out.push_str("]\n\n");

    for t in tables {
        let esc_desc = t.description.replace('"', "\\\"");
        out.push_str("[[catalog]]\n");
        out.push_str(&format!("schema = \"{}\"\n", t.schema));
        out.push_str(&format!("name = \"{}\"\n", t.name));
        out.push_str(&format!("description = \"{}\"\n", esc_desc));
        out.push_str(&format!("last_verified = \"{today}\"\n\n"));
    }

    let mut f = File::create(path).map_err(|e| e.to_string())?;
    f.write_all(out.as_bytes()).map_err(|e| e.to_string())?;
    Ok(())
}

fn chrono_today() -> String {
    // Avoid chrono dep; manual ISO date via SystemTime.
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Days since 1970-01-01
    let days = (secs / 86_400) as i64;
    // Civil-from-days algorithm (Howard Hinnant)
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp.wrapping_sub(9) };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{:04}-{:02}-{:02}", y, m, d)
}


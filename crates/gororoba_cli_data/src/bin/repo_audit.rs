//! repo-audit: anchored debt-baseline counter that replaces grep heuristics.
//!
//! WHY: the Stage A 2026-04-30 audit used substring grep that produced false
//! positives -- for example, the literal docstring "Zero Admitted." was
//! flagged as 5 actual Admitted statements. This binary replaces the
//! heuristics with anchored counts after stripping Rust comments and string
//! literals. The output is a TOML manifest suitable for replacing the
//! [code_quality], [formal_verification], and related sections of
//! data/output/debt_baseline_*.toml.
//!
//! Usage:
//!   cargo run --release -p gororoba_cli_data --bin repo-audit -- \
//!       --root crates --root proofs --output data/output/audit/2026-05-09/
//!
//! All counts are conservatively under-counts when in doubt, never
//! over-counts -- the rule is that a real problem is better surfaced by a
//! second tool than a phantom problem treated as real.

use anyhow::{Context, Result};
use clap::Parser;
use regex::Regex;
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Parser)]
#[command(name = "repo-audit", about = "Anchored debt-baseline counter")]
struct Args {
    /// Root directories to walk (repeat to add multiple).
    #[arg(long = "root", default_values_t = vec![String::from("crates"), String::from("proofs"), String::from("xtask")])]
    roots: Vec<String>,
    /// Output directory; the file is named repo_audit_anchored_<date>.toml.
    #[arg(long, default_value = "data/output/audit/repo_audit")]
    output_dir: PathBuf,
    /// Print results to stdout in addition to writing the file.
    #[arg(long, default_value_t = false)]
    print: bool,
}

#[derive(Default, Serialize, Debug, Clone)]
struct Counts {
    rust_files: u64,
    coq_files: u64,
    other_files: u64,
    // Rust source counts (after comment/string stripping).
    unsafe_blocks: u64,
    safety_comments: u64,
    ignore_attrs: u64,
    allow_clippy_attrs: u64,
    allow_dead_code_attrs: u64,
    todo_fixme_xxx_hack: u64,
    unimplemented_macros: u64,
    todo_macros: u64,
    unreachable_macros: u64,
    // Rocq proof counts (anchored, line-start).
    coq_admitted_strict: u64,    // ^\s*Admitted\b\.?\s*$
    coq_admit_strict: u64,       // ^\s*admit\b\.?\s*$
    coq_axiom_strict: u64,       // ^Axiom\b -- top level
    coq_axiom_indented: u64,     // ^\s+Axiom\b -- nested
    coq_parameter_strict: u64,   // ^Parameter\b
    coq_parameter_indented: u64, // ^\s+Parameter\b
}

#[derive(Serialize, Debug)]
struct AuditOutput {
    meta: Meta,
    totals: Counts,
    by_root: BTreeMap<String, Counts>,
}

#[derive(Serialize, Debug)]
struct Meta {
    generated_at: String,
    binary: String,
    roots: Vec<String>,
    method: String,
}

/// Strip block comments, line comments, and string/byte/char literals.
/// Returns the stripped source. Comments are replaced with spaces of the
/// same length to preserve line numbers for any line-anchored regex pass.
fn strip_rust(src: &str) -> String {
    let bytes = src.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            // Block comment: advance to matching */ (no nesting handling for v1).
            out.push(b' ');
            out.push(b' ');
            i += 2;
            while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                out.push(if bytes[i] == b'\n' { b'\n' } else { b' ' });
                i += 1;
            }
            if i + 1 < bytes.len() {
                out.push(b' ');
                out.push(b' ');
                i += 2;
            }
        } else if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            // Line comment.
            while i < bytes.len() && bytes[i] != b'\n' {
                out.push(b' ');
                i += 1;
            }
        } else if b == b'"' {
            // String literal (best-effort; does not handle raw strings r#"..."#).
            out.push(b' ');
            i += 1;
            while i < bytes.len() && bytes[i] != b'"' {
                if bytes[i] == b'\\' && i + 1 < bytes.len() {
                    out.push(b' ');
                    out.push(b' ');
                    i += 2;
                } else {
                    out.push(if bytes[i] == b'\n' { b'\n' } else { b' ' });
                    i += 1;
                }
            }
            if i < bytes.len() {
                out.push(b' ');
                i += 1;
            }
        } else if b == b'\'' {
            // Char literal or lifetime marker. Treat as char literal if next
            // is not an alphabetic identifier byte.
            let lookahead = bytes.get(i + 1).copied().unwrap_or(0);
            if lookahead.is_ascii_alphabetic() && bytes.get(i + 2) != Some(&b'\'') {
                // Lifetime: pass through as-is.
                out.push(b);
                i += 1;
            } else {
                out.push(b' ');
                i += 1;
                while i < bytes.len() && bytes[i] != b'\'' {
                    if bytes[i] == b'\\' && i + 1 < bytes.len() {
                        out.push(b' ');
                        out.push(b' ');
                        i += 2;
                    } else {
                        out.push(b' ');
                        i += 1;
                    }
                }
                if i < bytes.len() {
                    out.push(b' ');
                    i += 1;
                }
            }
        } else {
            out.push(b);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

struct Patterns {
    unsafe_block: Regex,
    safety_comment: Regex,
    ignore_attr: Regex,
    allow_clippy: Regex,
    allow_dead_code: Regex,
    todo_fixme: Regex,
    unimplemented_macro: Regex,
    todo_macro: Regex,
    unreachable_macro: Regex,
    coq_admitted_strict: Regex,
    coq_admit_strict: Regex,
    coq_axiom_top: Regex,
    coq_axiom_indented: Regex,
    coq_parameter_top: Regex,
    coq_parameter_indented: Regex,
}

impl Patterns {
    fn new() -> Result<Self> {
        Ok(Self {
            // Rust patterns: applied to STRIPPED source so they cannot match
            // inside comments or string literals.
            unsafe_block: Regex::new(r"\bunsafe\s*\{")?,
            ignore_attr: Regex::new(r"#\[\s*ignore\b")?,
            allow_clippy: Regex::new(r"#\[\s*allow\s*\(\s*clippy::")?,
            allow_dead_code: Regex::new(r"#\[\s*allow\s*\(\s*dead_code\s*\)")?,
            unimplemented_macro: Regex::new(r"\bunimplemented!\s*\(")?,
            todo_macro: Regex::new(r"\btodo!\s*\(")?,
            unreachable_macro: Regex::new(r"\bunreachable!\s*\(")?,
            // Comment-context patterns: applied to ORIGINAL source.
            safety_comment: Regex::new(r"(?m)^\s*(?://|/\*)\s*SAFETY\s*:")?,
            todo_fixme: Regex::new(r"(?m)^\s*(?://|/\*)\s*(?:TODO|FIXME|XXX|HACK)\b")?,
            // Rocq patterns: anchored to line start; case-sensitive.
            coq_admitted_strict: Regex::new(r"(?m)^[[:space:]]*Admitted[[:space:]]*\.\s*$")?,
            coq_admit_strict: Regex::new(r"(?m)^[[:space:]]*admit[[:space:]]*\.\s*$")?,
            coq_axiom_top: Regex::new(r"(?m)^Axiom\b")?,
            coq_axiom_indented: Regex::new(r"(?m)^[[:space:]]+Axiom\b")?,
            coq_parameter_top: Regex::new(r"(?m)^Parameter\b")?,
            coq_parameter_indented: Regex::new(r"(?m)^[[:space:]]+Parameter\b")?,
        })
    }
}

fn count_in_rust_file(src: &str, patterns: &Patterns) -> Counts {
    let stripped = strip_rust(src);
    let mut c = Counts::default();
    c.rust_files = 1;
    c.unsafe_blocks = patterns.unsafe_block.find_iter(&stripped).count() as u64;
    c.ignore_attrs = patterns.ignore_attr.find_iter(&stripped).count() as u64;
    c.allow_clippy_attrs = patterns.allow_clippy.find_iter(&stripped).count() as u64;
    c.allow_dead_code_attrs = patterns.allow_dead_code.find_iter(&stripped).count() as u64;
    c.unimplemented_macros = patterns.unimplemented_macro.find_iter(&stripped).count() as u64;
    c.todo_macros = patterns.todo_macro.find_iter(&stripped).count() as u64;
    c.unreachable_macros = patterns.unreachable_macro.find_iter(&stripped).count() as u64;
    c.safety_comments = patterns.safety_comment.find_iter(src).count() as u64;
    c.todo_fixme_xxx_hack = patterns.todo_fixme.find_iter(src).count() as u64;
    c
}

fn count_in_coq_file(src: &str, patterns: &Patterns) -> Counts {
    let mut c = Counts::default();
    c.coq_files = 1;
    c.coq_admitted_strict = patterns.coq_admitted_strict.find_iter(src).count() as u64;
    c.coq_admit_strict = patterns.coq_admit_strict.find_iter(src).count() as u64;
    c.coq_axiom_strict = patterns.coq_axiom_top.find_iter(src).count() as u64;
    c.coq_axiom_indented = patterns.coq_axiom_indented.find_iter(src).count() as u64;
    c.coq_parameter_strict = patterns.coq_parameter_top.find_iter(src).count() as u64;
    c.coq_parameter_indented = patterns.coq_parameter_indented.find_iter(src).count() as u64;
    c
}

fn merge(into: &mut Counts, from: &Counts) {
    into.rust_files += from.rust_files;
    into.coq_files += from.coq_files;
    into.other_files += from.other_files;
    into.unsafe_blocks += from.unsafe_blocks;
    into.safety_comments += from.safety_comments;
    into.ignore_attrs += from.ignore_attrs;
    into.allow_clippy_attrs += from.allow_clippy_attrs;
    into.allow_dead_code_attrs += from.allow_dead_code_attrs;
    into.todo_fixme_xxx_hack += from.todo_fixme_xxx_hack;
    into.unimplemented_macros += from.unimplemented_macros;
    into.todo_macros += from.todo_macros;
    into.unreachable_macros += from.unreachable_macros;
    into.coq_admitted_strict += from.coq_admitted_strict;
    into.coq_admit_strict += from.coq_admit_strict;
    into.coq_axiom_strict += from.coq_axiom_strict;
    into.coq_axiom_indented += from.coq_axiom_indented;
    into.coq_parameter_strict += from.coq_parameter_strict;
    into.coq_parameter_indented += from.coq_parameter_indented;
}

fn process_root(root: &Path, patterns: &Patterns) -> Result<Counts> {
    let mut total = Counts::default();
    for entry in WalkDir::new(root).follow_links(false).into_iter().filter_map(|r| r.ok()) {
        if !entry.file_type().is_file() {
            continue;
        }
        let p = entry.path();
        let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("");
        match ext {
            "rs" => {
                let src = match fs::read_to_string(p) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                merge(&mut total, &count_in_rust_file(&src, patterns));
            }
            "v" => {
                let src = match fs::read_to_string(p) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                merge(&mut total, &count_in_coq_file(&src, patterns));
            }
            _ => {
                total.other_files += 1;
            }
        }
    }
    Ok(total)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let patterns = Patterns::new()?;
    let mut by_root = BTreeMap::new();
    let mut totals = Counts::default();
    for root_str in &args.roots {
        let root = PathBuf::from(root_str);
        if !root.exists() {
            eprintln!("warn: root '{}' does not exist; skipping.", root.display());
            continue;
        }
        let counts = process_root(&root, &patterns)
            .with_context(|| format!("failed to walk root {}", root.display()))?;
        merge(&mut totals, &counts);
        by_root.insert(root_str.clone(), counts);
    }
    let now = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
    let date_only = chrono::Utc::now().format("%Y_%m_%d").to_string();
    let output = AuditOutput {
        meta: Meta {
            generated_at: now,
            binary: env!("CARGO_BIN_NAME").to_string(),
            roots: args.roots.clone(),
            method: "regex-anchored on comment-stripped Rust source; line-anchored on Rocq".to_string(),
        },
        totals,
        by_root,
    };
    let toml_text = toml::to_string_pretty(&output)?;
    fs::create_dir_all(&args.output_dir)?;
    let out_path = args
        .output_dir
        .join(format!("repo_audit_anchored_{}.toml", date_only));
    fs::write(&out_path, &toml_text)?;
    eprintln!("wrote {}", out_path.display());
    if args.print {
        print!("{}", toml_text);
    }
    Ok(())
}

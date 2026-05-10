//! repo-audit: durable anchored debt-baseline counter.
//!
//! WHY: a substring-grep audit is unreliable -- the Stage A 2026-04-30 pass
//! flagged the docstring "Zero Admitted." as 5 actual Admitted statements,
//! and "TODO" matches in prose as 142 actual code-comment placeholders.
//! This binary replaces that heuristic with anchored regex on Rust source
//! that has been pre-stripped of comments and string literals, plus
//! line-anchored Rocq patterns.
//!
//! WHAT this binary IS: a durable repo-debt counter intended to be run
//! periodically (manually, in CI, or as a Make target). Its TOML output
//! is the canonical record of "how much measurable debt does the repo
//! carry today" across these classes:
//!   - unsafe blocks (with SAFETY-comment coverage)
//!   - #[ignore] / #[allow(clippy::*)] / #[allow(dead_code)] attrs
//!   - unimplemented! / todo! / unreachable! macro calls
//!   - TODO / FIXME / XXX / HACK in code-comment context
//!   - Rocq Admitted / admit / Axiom / Parameter (strict + indented)
//!
//! WHAT this binary is NOT: a full AST-based static analyzer. A v2 using
//! `syn 2.x` would catch macro-expanded unsafe and per-item attribute
//! placement; v1 is correct for un-expanded source and is sufficient for
//! the baseline-tracking use case.
//!
//! Modes:
//!   plain       Walk the configured roots and emit a TOML snapshot.
//!   --baseline-compare PATH  Read a prior snapshot and emit a delta
//!                             section comparing it to the current count.
//!   --strict    Exit non-zero if any tracked class has grown vs the
//!                 baseline. For CI use.
//!   --print     Echo the TOML to stdout in addition to writing the file.

use anyhow::{Context, Result};
use clap::Parser;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};
use walkdir::WalkDir;

#[derive(Parser)]
#[command(
    name = "repo-audit",
    about = "Anchored debt-baseline counter (Rocq + Rust)"
)]
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
    /// Compare against a prior snapshot TOML and emit a delta block.
    #[arg(long)]
    baseline_compare: Option<PathBuf>,
    /// Exit non-zero if any debt class grew compared to the baseline.
    #[arg(long, default_value_t = false)]
    strict: bool,
    /// Maximum allowed allow_clippy_unjustified count per root. When set,
    /// the binary exits non-zero if the per-root unjustified count
    /// exceeds the threshold. Use to gate "no new unjustified clippy
    /// suppression in crates/" as a pre-push regression guard.
    /// Default: disabled (None). Suggested value: 0 (no regression
    /// allowed) once the baseline is at zero.
    #[arg(long)]
    strict_unjustified_per_root: Option<u64>,
    /// Path to canonical SQLite control plane to read revisions audit from.
    /// When supplied, the report includes a `[revisions]` block with
    /// claim/insight/experiment revision counts, top mutators, and
    /// per-field counts.
    #[arg(long)]
    sqlite: Option<PathBuf>,
}

#[derive(Default, Serialize, Deserialize, Debug, Clone)]
struct Counts {
    rust_files: u64,
    rocq_files: u64,
    other_files: u64,
    // Rust source counts (after comment/string stripping).
    unsafe_blocks: u64,
    safety_comments: u64,
    ignore_attrs: u64,
    allow_clippy_attrs: u64,
    /// Subset of `allow_clippy_attrs` that lack any justification (no
    /// trailing `// ...` comment on the same line and no `// ...` line
    /// directly above the attribute). This is the actual debt count;
    /// `allow_clippy_attrs` includes legitimate multi-cursor / matrix
    /// loop patterns where the lint is correctly silenced.
    allow_clippy_unjustified: u64,
    allow_dead_code_attrs: u64,
    todo_fixme_xxx_hack: u64,
    unimplemented_macros: u64,
    todo_macros: u64,
    unreachable_macros: u64,
    // Rocq proof counts (anchored, line-start). Rocq is the renamed Coq
    // theorem prover; field names use rocq_* to track the project's
    // canonical naming.
    rocq_admitted_strict: u64,    // ^\s*Admitted\b\.?\s*$
    rocq_admit_strict: u64,       // ^\s*admit\b\.?\s*$
    rocq_axiom_strict: u64,       // ^Axiom\b -- top level
    rocq_axiom_indented: u64,     // ^\s+Axiom\b -- nested
    rocq_parameter_strict: u64,   // ^Parameter\b
    rocq_parameter_indented: u64, // ^\s+Parameter\b
}

#[derive(Serialize, Debug)]
struct AuditOutput {
    meta: Meta,
    totals: Counts,
    by_root: BTreeMap<String, Counts>,
    #[serde(skip_serializing_if = "Option::is_none")]
    baseline_delta: Option<BaselineDelta>,
    #[serde(skip_serializing_if = "Option::is_none")]
    revisions: Option<RevisionsSummary>,
}

/// Snapshot of the canonical SQLite revisions audit trail. Synthesizes
/// repo-audit's static debt count with the dynamic mutation flow recorded
/// in claim_revisions / insight_revisions / experiment_revisions, so a
/// single audit report shows both "how much debt exists" and "how much
/// has been touched recently".
#[derive(Serialize, Debug)]
struct RevisionsSummary {
    sqlite_path: String,
    claim_revisions: u64,
    insight_revisions: u64,
    experiment_revisions: u64,
    /// Per-field counts across all three revisions tables, sorted by name.
    by_field: BTreeMap<String, u64>,
    /// Top 5 actors by revision count, sorted descending.
    top_actors: Vec<ActorCount>,
}

#[derive(Serialize, Debug)]
struct ActorCount {
    actor: String,
    revisions: u64,
}

#[derive(Deserialize, Debug)]
struct PriorSnapshot {
    totals: Counts,
}

#[derive(Serialize, Debug)]
struct BaselineDelta {
    baseline_path: String,
    grown: BTreeMap<String, i64>,
    shrunk: BTreeMap<String, i64>,
    unchanged_count: u64,
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
    rocq_admitted_strict: Regex,
    rocq_admit_strict: Regex,
    rocq_axiom_top: Regex,
    rocq_axiom_indented: Regex,
    rocq_parameter_top: Regex,
    rocq_parameter_indented: Regex,
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
            rocq_admitted_strict: Regex::new(r"(?m)^[[:space:]]*Admitted[[:space:]]*\.\s*$")?,
            rocq_admit_strict: Regex::new(r"(?m)^[[:space:]]*admit[[:space:]]*\.\s*$")?,
            rocq_axiom_top: Regex::new(r"(?m)^Axiom\b")?,
            rocq_axiom_indented: Regex::new(r"(?m)^[[:space:]]+Axiom\b")?,
            rocq_parameter_top: Regex::new(r"(?m)^Parameter\b")?,
            rocq_parameter_indented: Regex::new(r"(?m)^[[:space:]]+Parameter\b")?,
        })
    }
}

fn count_in_rust_file(src: &str, patterns: &Patterns) -> Counts {
    let stripped = strip_rust(src);
    Counts {
        rust_files: 1,
        unsafe_blocks: patterns.unsafe_block.find_iter(&stripped).count() as u64,
        ignore_attrs: patterns.ignore_attr.find_iter(&stripped).count() as u64,
        allow_clippy_attrs: patterns.allow_clippy.find_iter(&stripped).count() as u64,
        allow_clippy_unjustified: count_unjustified_allow_clippy(src),
        allow_dead_code_attrs: patterns.allow_dead_code.find_iter(&stripped).count() as u64,
        unimplemented_macros: patterns.unimplemented_macro.find_iter(&stripped).count() as u64,
        todo_macros: patterns.todo_macro.find_iter(&stripped).count() as u64,
        unreachable_macros: patterns.unreachable_macro.find_iter(&stripped).count() as u64,
        safety_comments: patterns.safety_comment.find_iter(src).count() as u64,
        todo_fixme_xxx_hack: patterns.todo_fixme.find_iter(src).count() as u64,
        ..Counts::default()
    }
}

/// Count `#[allow(clippy::*)]` attributes that lack any justification:
/// neither a trailing `// ...` comment on the same line nor a `// ...`
/// comment immediately above the attribute. Operates on the ORIGINAL
/// source (not stripped) so it can see the comments. False positives are
/// possible if the attribute is on the same line as code and nothing else
/// (rare); false negatives are possible if the comment is multiple lines
/// above the attribute (also rare and correctly flagged as unjustified).
fn count_unjustified_allow_clippy(src: &str) -> u64 {
    let attr_re = match Regex::new(r"^\s*#\[\s*allow\s*\(\s*clippy::") {
        Ok(r) => r,
        Err(_) => return 0,
    };
    let trailing_re = match Regex::new(r"#\[\s*allow\s*\([^)]+\)\s*\]\s*//") {
        Ok(r) => r,
        Err(_) => return 0,
    };
    let comment_re = match Regex::new(r"^\s*(?://|/\*)") {
        Ok(r) => r,
        Err(_) => return 0,
    };
    let lines: Vec<&str> = src.lines().collect();
    let mut unjustified = 0u64;
    for (i, line) in lines.iter().enumerate() {
        if !attr_re.is_match(line) {
            continue;
        }
        // Trailing comment on the same line?
        if trailing_re.is_match(line) {
            continue;
        }
        // Comment line directly above the attribute?
        if i > 0 && comment_re.is_match(lines[i - 1]) {
            continue;
        }
        unjustified += 1;
    }
    unjustified
}

fn count_in_rocq_file(src: &str, patterns: &Patterns) -> Counts {
    Counts {
        rocq_files: 1,
        rocq_admitted_strict: patterns.rocq_admitted_strict.find_iter(src).count() as u64,
        rocq_admit_strict: patterns.rocq_admit_strict.find_iter(src).count() as u64,
        rocq_axiom_strict: patterns.rocq_axiom_top.find_iter(src).count() as u64,
        rocq_axiom_indented: patterns.rocq_axiom_indented.find_iter(src).count() as u64,
        rocq_parameter_strict: patterns.rocq_parameter_top.find_iter(src).count() as u64,
        rocq_parameter_indented: patterns.rocq_parameter_indented.find_iter(src).count() as u64,
        ..Counts::default()
    }
}

/// Read the canonical revisions audit trail from `sqlite_path` and
/// summarize it into a RevisionsSummary. Aggregates across all three
/// revisions tables (claim, insight, experiment) for the by_field and
/// top_actors blocks. Errors propagate; callers may choose to warn-and-
/// continue rather than fail the audit.
fn read_revisions_summary(sqlite_path: &Path) -> Result<RevisionsSummary> {
    let conn = rusqlite::Connection::open_with_flags(
        sqlite_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )?;
    let count_table = |table: &str| -> Result<u64> {
        let mut stmt = conn.prepare(&format!("SELECT COUNT(*) FROM {}", table))?;
        let n: i64 = stmt.query_row([], |row| row.get(0))?;
        Ok(n as u64)
    };
    let claim_revisions = count_table("claim_revisions")?;
    let insight_revisions = count_table("insight_revisions")?;
    let experiment_revisions = count_table("experiment_revisions")?;

    let mut by_field: BTreeMap<String, u64> = BTreeMap::new();
    let aggregate_field = |conn: &rusqlite::Connection,
                           table: &str,
                           by_field: &mut BTreeMap<String, u64>|
     -> Result<()> {
        let mut stmt = conn.prepare(&format!(
            "SELECT field_name, COUNT(*) FROM {} GROUP BY field_name",
            table
        ))?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as u64))
        })?;
        for row in rows {
            let (field, n) = row?;
            *by_field.entry(field).or_insert(0) += n;
        }
        Ok(())
    };
    aggregate_field(&conn, "claim_revisions", &mut by_field)?;
    aggregate_field(&conn, "insight_revisions", &mut by_field)?;
    aggregate_field(&conn, "experiment_revisions", &mut by_field)?;

    let mut actor_counts: BTreeMap<String, u64> = BTreeMap::new();
    let aggregate_actors = |conn: &rusqlite::Connection,
                            table: &str,
                            actor_counts: &mut BTreeMap<String, u64>|
     -> Result<()> {
        let mut stmt = conn.prepare(&format!(
            "SELECT actor, COUNT(*) FROM {} GROUP BY actor",
            table
        ))?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as u64))
        })?;
        for row in rows {
            let (actor, n) = row?;
            *actor_counts.entry(actor).or_insert(0) += n;
        }
        Ok(())
    };
    aggregate_actors(&conn, "claim_revisions", &mut actor_counts)?;
    aggregate_actors(&conn, "insight_revisions", &mut actor_counts)?;
    aggregate_actors(&conn, "experiment_revisions", &mut actor_counts)?;
    let mut top_actors: Vec<ActorCount> = actor_counts
        .into_iter()
        .map(|(actor, revisions)| ActorCount { actor, revisions })
        .collect();
    top_actors.sort_by(|a, b| b.revisions.cmp(&a.revisions).then(a.actor.cmp(&b.actor)));
    top_actors.truncate(5);

    Ok(RevisionsSummary {
        sqlite_path: sqlite_path.display().to_string(),
        claim_revisions,
        insight_revisions,
        experiment_revisions,
        by_field,
        top_actors,
    })
}

fn merge(into: &mut Counts, from: &Counts) {
    into.rust_files += from.rust_files;
    into.rocq_files += from.rocq_files;
    into.other_files += from.other_files;
    into.unsafe_blocks += from.unsafe_blocks;
    into.safety_comments += from.safety_comments;
    into.ignore_attrs += from.ignore_attrs;
    into.allow_clippy_attrs += from.allow_clippy_attrs;
    into.allow_clippy_unjustified += from.allow_clippy_unjustified;
    into.allow_dead_code_attrs += from.allow_dead_code_attrs;
    into.todo_fixme_xxx_hack += from.todo_fixme_xxx_hack;
    into.unimplemented_macros += from.unimplemented_macros;
    into.todo_macros += from.todo_macros;
    into.unreachable_macros += from.unreachable_macros;
    into.rocq_admitted_strict += from.rocq_admitted_strict;
    into.rocq_admit_strict += from.rocq_admit_strict;
    into.rocq_axiom_strict += from.rocq_axiom_strict;
    into.rocq_axiom_indented += from.rocq_axiom_indented;
    into.rocq_parameter_strict += from.rocq_parameter_strict;
    into.rocq_parameter_indented += from.rocq_parameter_indented;
}

fn process_root(root: &Path, patterns: &Patterns) -> Result<Counts> {
    let mut total = Counts::default();
    for entry in WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_map(|r| r.ok())
    {
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
                merge(&mut total, &count_in_rocq_file(&src, patterns));
            }
            _ => {
                total.other_files += 1;
            }
        }
    }
    Ok(total)
}

fn delta_field(name: &str, prev: u64, curr: u64) -> Option<(String, i64)> {
    let d = curr as i64 - prev as i64;
    if d == 0 {
        None
    } else {
        Some((name.to_string(), d))
    }
}

fn compute_delta(baseline_path: &Path, prior: &Counts, curr: &Counts) -> BaselineDelta {
    let mut grown = BTreeMap::new();
    let mut shrunk = BTreeMap::new();
    let mut unchanged = 0u64;
    let pairs: Vec<(&str, u64, u64)> = vec![
        ("unsafe_blocks", prior.unsafe_blocks, curr.unsafe_blocks),
        (
            "safety_comments",
            prior.safety_comments,
            curr.safety_comments,
        ),
        ("ignore_attrs", prior.ignore_attrs, curr.ignore_attrs),
        (
            "allow_clippy_attrs",
            prior.allow_clippy_attrs,
            curr.allow_clippy_attrs,
        ),
        (
            "allow_dead_code_attrs",
            prior.allow_dead_code_attrs,
            curr.allow_dead_code_attrs,
        ),
        (
            "todo_fixme_xxx_hack",
            prior.todo_fixme_xxx_hack,
            curr.todo_fixme_xxx_hack,
        ),
        (
            "unimplemented_macros",
            prior.unimplemented_macros,
            curr.unimplemented_macros,
        ),
        ("todo_macros", prior.todo_macros, curr.todo_macros),
        (
            "unreachable_macros",
            prior.unreachable_macros,
            curr.unreachable_macros,
        ),
        (
            "rocq_admitted_strict",
            prior.rocq_admitted_strict,
            curr.rocq_admitted_strict,
        ),
        (
            "rocq_admit_strict",
            prior.rocq_admit_strict,
            curr.rocq_admit_strict,
        ),
        (
            "rocq_axiom_strict",
            prior.rocq_axiom_strict,
            curr.rocq_axiom_strict,
        ),
        (
            "rocq_axiom_indented",
            prior.rocq_axiom_indented,
            curr.rocq_axiom_indented,
        ),
        (
            "rocq_parameter_strict",
            prior.rocq_parameter_strict,
            curr.rocq_parameter_strict,
        ),
        (
            "rocq_parameter_indented",
            prior.rocq_parameter_indented,
            curr.rocq_parameter_indented,
        ),
    ];
    for (name, prev, cur) in pairs {
        if let Some((n, d)) = delta_field(name, prev, cur) {
            if d > 0 {
                grown.insert(n, d);
            } else {
                shrunk.insert(n, d);
            }
        } else {
            unchanged += 1;
        }
    }
    BaselineDelta {
        baseline_path: baseline_path.display().to_string(),
        grown,
        shrunk,
        unchanged_count: unchanged,
    }
}

/// SAFETY-positive classes: SAFETY comments, safety_comments. Growth here
/// is improvement, not debt.
const SAFETY_POSITIVE_CLASSES: &[&str] = &["safety_comments"];

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
    let mut baseline_delta = None;
    let mut strict_failure = false;
    if let Some(baseline_path) = &args.baseline_compare {
        let prior_text = fs::read_to_string(baseline_path)
            .with_context(|| format!("read baseline {}", baseline_path.display()))?;
        let prior: PriorSnapshot = toml::from_str(&prior_text)
            .with_context(|| format!("parse baseline {}", baseline_path.display()))?;
        let delta = compute_delta(baseline_path, &prior.totals, &totals);
        if args.strict {
            for (class, growth) in &delta.grown {
                if SAFETY_POSITIVE_CLASSES.contains(&class.as_str()) {
                    continue;
                }
                eprintln!("strict: {} grew by {} since baseline", class, growth);
                strict_failure = true;
            }
        }
        baseline_delta = Some(delta);
    }
    let revisions = match args.sqlite.as_ref() {
        Some(path) => match read_revisions_summary(path) {
            Ok(summary) => Some(summary),
            Err(err) => {
                eprintln!(
                    "WARN: failed to read SQLite revisions from {}: {}",
                    path.display(),
                    err
                );
                None
            }
        },
        None => None,
    };
    let output = AuditOutput {
        meta: Meta {
            generated_at: now,
            binary: env!("CARGO_BIN_NAME").to_string(),
            roots: args.roots.clone(),
            method: "regex-anchored on comment-stripped Rust; line-anchored on Rocq (.v)"
                .to_string(),
        },
        totals,
        by_root,
        baseline_delta,
        revisions,
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
    if strict_failure {
        eprintln!("repo-audit --strict: at least one debt class grew vs baseline");
        std::process::exit(1);
    }
    if let Some(cap) = args.strict_unjustified_per_root {
        let mut cap_failures = Vec::new();
        for (root, counts) in &output.by_root {
            if counts.allow_clippy_unjustified > cap {
                cap_failures.push(format!(
                    "  {}: {} unjustified clippy allows (cap = {})",
                    root, counts.allow_clippy_unjustified, cap
                ));
            }
        }
        if !cap_failures.is_empty() {
            eprintln!(
                "repo-audit --strict-unjustified-per-root {}: cap exceeded:\n{}",
                cap,
                cap_failures.join("\n")
            );
            std::process::exit(1);
        }
    }
    Ok(())
}

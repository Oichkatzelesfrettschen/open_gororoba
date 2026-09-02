use anstyle::{AnsiColor, Style};
use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use regex::Regex;
use std::{
    collections::{BTreeSet, HashSet},
    fs,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Command, ExitCode},
    sync::mpsc,
    time::Duration,
};
use unicode_normalization::UnicodeNormalization;
use walkdir::WalkDir;

const CHARACTER_POLICY_REPLACEMENTS: &[(&str, &str)] = &[
    ("\u{2018}", "'"),
    ("\u{2019}", "'"),
    ("\u{201c}", "\""),
    ("\u{201d}", "\""),
    ("\u{2013}", "-"),
    ("\u{2014}", "--"),
    ("\u{2212}", "-"),
    ("\u{207b}", "-"),
    ("\u{2011}", "-"),
    ("\u{00b7}", "*"),
    ("\u{2202}", "\\partial"),
    ("\u{2192}", "->"),
    ("\u{21d2}", "=>"),
    ("\u{2026}", "..."),
    ("\u{00a0}", " "),
    ("\u{221e}", "infty"),
    ("\u{00d7}", "x"),
    ("\u{00b2}", "^2"),
    ("\u{00b3}", "^3"),
    ("\u{00b9}", "^1"),
    ("\u{2080}", "_0"),
    ("\u{2081}", "_1"),
    ("\u{2082}", "_2"),
    ("\u{2083}", "_3"),
    ("\u{2084}", "_4"),
    ("\u{2085}", "_5"),
    ("\u{2086}", "_6"),
    ("\u{2087}", "_7"),
    ("\u{2088}", "_8"),
    ("\u{2089}", "_9"),
    ("\u{00c5}", "Angstrom"),
    ("\u{03b1}", "\\alpha"),
    ("\u{03b2}", "\\beta"),
    ("\u{03b3}", "\\gamma"),
    ("\u{03b4}", "\\delta"),
    ("\u{03b5}", "\\epsilon"),
    ("\u{03b8}", "\\theta"),
    ("\u{03bb}", "\\lambda"),
    ("\u{03bc}", "\\mu"),
    ("\u{03c0}", "\\pi"),
    ("\u{03c8}", "\\psi"),
    ("\u{03b6}", "\\zeta"),
    ("\u{03b7}", "\\eta"),
    ("\u{03ba}", "\\kappa"),
    ("\u{03bd}", "\\nu"),
    ("\u{03be}", "\\xi"),
    ("\u{03c1}", "\\rho"),
    ("\u{03c3}", "\\sigma"),
    ("\u{03c4}", "\\tau"),
    ("\u{03c6}", "\\phi"),
    ("\u{03c7}", "\\chi"),
    ("\u{03c9}", "\\omega"),
    ("\u{0393}", "\\Gamma"),
    ("\u{0394}", "\\Delta"),
    ("\u{0398}", "\\Theta"),
    ("\u{039b}", "\\Lambda"),
    ("\u{03a3}", "\\Sigma"),
    ("\u{03a9}", "\\Omega"),
    ("\u{2206}", "\\Delta"),
    ("\u{2248}", "~="),
    ("\u{221d}", "~"),
    ("\u{221a}", "sqrt"),
    ("\u{222b}", "integral"),
    ("\u{210f}", "hbar"),
    ("\u{2295}", "\\oplus"),
    ("\u{00f6}", "o"),
    ("\u{00fc}", "u"),
    ("\u{00e4}", "a"),
    ("\u{00e9}", "e"),
    ("\u{00f1}", "n"),
];

const CHARACTER_POLICY_SKIP_DIRS: &[&str] = &[
    ".git",
    "target",
    "venv",
    "convos",
    "data",
    "logs",
    "reports",
    "__pycache__",
    "node_modules",
];

const CHARACTER_POLICY_SKIP_PATH_PREFIXES: &[&str] = &["data/external/papers"];

const CHARACTER_POLICY_SKIP_EXTS: &[&str] = &[
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".xlsx", ".zip", ".tar", ".gz", ".bz2",
    ".xz", ".7z", ".bsp", ".npy", ".npz", ".fits", ".pyc", ".so", ".o", ".a", ".rlib", ".rmeta",
    ".d", ".vo", ".vok", ".vos", ".glob",
];

const TERMINOLOGY_SKIP_EXTENSIONS: &[&str] = &[
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".pdf", ".ttf", ".otf", ".woff",
    ".woff2", ".zip", ".gz", ".tar", ".xz", ".bz2", ".zst", ".wasm", ".o", ".a", ".so", ".dylib",
    ".dll", ".pyc", ".pyo", ".class", ".mp4", ".webm", ".ogg", ".mp3", ".wav", ".h5", ".hdf5",
    ".npy", ".npz", ".vo", ".vos", ".vok", ".glob",
];

const TERMINOLOGY_SKIP_NAMES: &[&str] = &[
    "Cargo.lock",
    "terminology_gate.py",
    "markdown_payload_chunks.toml",
    "markdown_payloads.toml",
];

const TERMINOLOGY_SKIP_SUFFIXES: &[&str] = &[".backup.phase2", ".backup"];

const ROCQ_HEADER: &str = "From Stdlib Require Import String.\nRequire Import ConfineModel.\n\nOpen Scope string_scope.\n\n";

const DOCTOR_PYTHON_MODULES: &[&str] = &[
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "numba",
    "sympy",
    "networkx",
    "ripser",
    "persim",
    "astroquery",
    "gwpy",
    "requests",
    "qiskit",
    "euclid3",
    "quaternion",
    "pyquaternion",
    "mutatorMath",
    "ipfn",
    "findiff",
    "pymultinest",
    "typedunits",
    "quaternionic",
    "unicodedata2",
    "defcon",
    "fontMath",
];

const DOCTOR_BINARIES: &[&str] = &["docker", "coqc", "latexmk"];

const SOURCE_ANALYSIS_MCP_TOOLS: &[&str] = &[
    "source_analysis_catalog",
    "source_analysis_hazard_ack",
    "source_analysis_version",
    "source_search",
    "source_transform",
    "source_index",
    "source_static_analysis",
    "source_metrics",
    "binary_analysis",
    "trace_profile",
    "fuzz_instrument",
    "source_analysis_doctor",
];

const DOCS_REDIRECT_REQUIRED_FILES: &[&str] = &[
    "404.html",
    "book.html",
    "rustdoc.html",
    "index.html",
    ".nojekyll",
];

const DOCS_REDIRECT_REQUIRED_MARKERS: &[(&str, &str)] = &[
    ("404.html", "/.cache/cargo-default-target/doc"),
    ("404.html", "/cache/gate-target/doc"),
    ("404.html", "window.location.replace"),
    ("book.html", "./book/"),
    ("rustdoc.html", "./rustdoc/"),
];

const DOCS_REDIRECT_LEGACY_PREFIXES: &[&str] = &[
    "/.cache/cargo-default-target/doc",
    "/cache/cargo-default-target/doc",
    "/.cache/gate-target/doc",
    "/cache/gate-target/doc",
    "/target/docs-target/doc",
    "/target/doc",
];

const DOCS_REDIRECT_CASES: &[(&str, &str)] = &[
    (
        "/.cache/cargo-default-target/doc/pkg/struct.SomeType.html",
        "/rustdoc/pkg/struct.SomeType.html",
    ),
    ("/cache/gate-target/doc/index.html", "/rustdoc/index.html"),
    (
        "/repo/.cache/gate-target/doc/pkg/index.html",
        "/repo/rustdoc/pkg/index.html",
    ),
    ("/book", "/book/"),
    ("/rustdoc", "/rustdoc/"),
    ("/repo/book", "/repo/book/"),
    ("/repo/rustdoc", "/repo/rustdoc/"),
    ("/repo/other", "/repo/"),
];

#[derive(Parser, Debug)]
#[command(
    name = "repo-utilities",
    about = "Repo utility gates and helper transforms"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    #[command(name = "ansi-check", visible_alias = "ascii-check")]
    AnsiCheck(CharacterPolicyArgs),
    TerminologyGate(TerminologyArgs),
    Doctor,
    #[command(name = "mcp-smoke")]
    McpSmoke,
    #[command(name = "docs-redirect-check")]
    DocsRedirectCheck(DocsRedirectArgs),
    #[command(name = "rocq-prepare-confine", visible_alias = "coq-prepare-confine")]
    RocqPrepareConfine(CoqArgs),
    #[command(name = "validation-tool-paths")]
    ValidationToolPaths(ValidationToolPathsArgs),
}

/// The scan needs three roots and takes none of them from a compiled-in
/// constant: the staged tools directory, the parent of every worktree, and the
/// checkout that is running the gate.
#[derive(Args, Debug)]
struct ValidationToolPathsArgs {
    #[arg(long = "tools-dir")]
    tools_dir: PathBuf,
    #[arg(long = "worktrees-root")]
    worktrees_root: PathBuf,
    #[arg(long = "current-root")]
    current_root: PathBuf,
}

#[derive(Args, Debug)]
struct CharacterPolicyArgs {
    #[arg(long)]
    check: bool,
    #[arg(long)]
    fix: bool,
    #[arg(long)]
    strict_placeholders: bool,
    #[arg(long = "placeholder-scope-prefix")]
    placeholder_scope_prefix: Vec<String>,
    #[arg(
        long = "placeholder-allowlist",
        default_values_t = vec![
            String::from("bin/ascii_check.py"),
            String::from("bin/ascii_placeholder_cleanup.py")
        ]
    )]
    placeholder_allowlist: Vec<String>,
}

#[derive(Args, Debug)]
struct TerminologyArgs {
    #[arg(long)]
    quiet: bool,
}

#[derive(Args, Debug)]
struct CoqArgs {
    src: PathBuf,
    dst: PathBuf,
}

#[derive(Args, Debug)]
struct DocsRedirectArgs {
    #[arg(default_value = "target/site-docs")]
    docs_site_dir: PathBuf,
}

#[derive(Clone)]
struct BannedTerm {
    pattern: String,
    replacement: String,
    reason: String,
}

fn repo_root() -> PathBuf {
    repo_root::resolve!()
}

fn style_text(text: &str, color: AnsiColor, bold: bool) -> String {
    let mut style = Style::new().fg_color(Some(color.into()));
    if bold {
        style = style.bold();
    }
    format!("{}{}{}", style.render(), text, style.render_reset())
}

fn is_combining_mark(ch: char) -> bool {
    matches!(
        ch as u32,
        0x0300..=0x036F | 0x1AB0..=0x1AFF | 0x1DC0..=0x1DFF | 0x20D0..=0x20FF | 0xFE20..=0xFE2F
    )
}
fn is_emoji(ch: char) -> bool {
    let val = ch as u32;
    // Block: Emoticons, Transport, Misc Pictographs, Supplemental Pictographs, Flags, Variation Selectors
    matches!(
        val,
        0x1F600..=0x1F64F | // Emoticons
        0x1F300..=0x1F5FF | // Misc Symbols and Pictographs
        0x1F680..=0x1F6FF | // Transport and Map
        0x1F900..=0x1F9FF | // Supplemental Symbols and Pictographs
        0x1F1E6..=0x1F1FF | // Flags
        0xFE00..=0xFE0F     // Variation Selectors
    )
}

fn sanitize_text(text: &str) -> String {
    let mut current = text.to_string();
    for (src, dst) in CHARACTER_POLICY_REPLACEMENTS {
        current = current.replace(src, dst);
    }
    let normalized: String = current
        .nfkd()
        .filter(|ch| !is_combining_mark(*ch))
        .collect();
    let mut out = String::new();
    for ch in normalized.chars() {
        if ch.is_ascii() || (!is_emoji(ch) && (ch as u32) > 127) {
            out.push(ch);
        } else if is_emoji(ch) {
            out.push_str(&format!("<EMOJI+{:04X}>", ch as u32));
        } else {
            out.push_str(&format!("<U+{:04X}>", ch as u32));
        }
    }
    out
}

fn strip_ansi_sequences(text: &str) -> String {
    let csi = Regex::new(r"\x1B\[[0-?]*[ -/]*[@-~]").expect("valid ansi csi regex");
    let osc = Regex::new(r"\x1B\][^\x07]*(\x07|\x1B\\)").expect("valid ansi osc regex");
    let without_csi = csi.replace_all(text, "");
    let without_osc = osc.replace_all(&without_csi, "");
    without_osc.replace('\u{001b}', "")
}

fn iter_character_policy_files(root: &Path) -> Vec<PathBuf> {
    WalkDir::new(root)
        .into_iter()
        .filter_entry(|entry| {
            if !entry.file_type().is_dir() {
                return true;
            }
            if entry.depth() == 0 {
                return true;
            }
            let name = entry.file_name().to_string_lossy();
            !name.starts_with('.')
                && !CHARACTER_POLICY_SKIP_DIRS.contains(&name.as_ref())
                && !name.starts_with("target")
                && !name.ends_with("_venv")
        })
        .filter_map(std::result::Result::ok)
        .filter(|entry| entry.file_type().is_file())
        .map(|entry| entry.into_path())
        .collect()
}

fn run_character_policy(args: CharacterPolicyArgs) -> Result<()> {
    if args.check == args.fix {
        bail!("Pass exactly one of: --check, --fix");
    }
    let root = repo_root();
    let scope_prefixes: Vec<String> = args
        .placeholder_scope_prefix
        .into_iter()
        .map(|value| value.trim_matches('/').to_string())
        .filter(|value| !value.is_empty())
        .collect();
    let allowlist: HashSet<String> = args.placeholder_allowlist.into_iter().collect();
    let mut failures = Vec::new();
    let mut placeholder_failures = Vec::new();

    for path in iter_character_policy_files(&root) {
        let rel = path
            .strip_prefix(&root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        if CHARACTER_POLICY_SKIP_EXTS
            .iter()
            .any(|ext| rel.ends_with(ext))
        {
            continue;
        }
        if CHARACTER_POLICY_SKIP_PATH_PREFIXES
            .iter()
            .any(|prefix| rel.starts_with(prefix))
        {
            continue;
        }
        let Ok(metadata) = path.metadata() else {
            continue;
        };
        if metadata.len() > 10_000_000 {
            continue;
        }
        let Ok(raw) = fs::read(&path) else { continue };
        let Ok(text) = String::from_utf8(raw) else {
            continue;
        };
        let new_text = if args.fix {
            let sanitized = sanitize_text(&text);
            strip_ansi_sequences(&sanitized)
        } else {
            text.clone()
        };

        if let Some(bad_ch) = new_text
            .chars()
            .find(|&ch| (ch.is_control() && !matches!(ch, '\n' | '\r' | '\t')) || is_emoji(ch))
        {
            failures.push(format!("{} (first bad char: U+{:04X})", rel, bad_ch as u32));
            continue;
        }

        if args.strict_placeholders {
            let in_scope = scope_prefixes.is_empty()
                || scope_prefixes
                    .iter()
                    .any(|prefix| rel == *prefix || rel.starts_with(&format!("{prefix}/")));
            if in_scope
                && !allowlist.contains(&rel)
                && (new_text.contains("<U+") || new_text.contains("<EMOJI+"))
            {
                placeholder_failures.push(rel.clone());
                continue;
            }
        }
        if args.fix && new_text != text {
            fs::write(&path, new_text).with_context(|| format!("write {}", path.display()))?;
        }
    }

    if !failures.is_empty() {
        println!(
            "{}",
            style_text(
                "Files with disallowed characters (emojis or controls) detected:",
                AnsiColor::Red,
                true
            )
        );
        for fp in failures.iter().take(50) {
            println!("  - {fp}");
        }
        bail!("ansi check failed");
    }
    if !placeholder_failures.is_empty() {
        println!(
            "{}",
            style_text(
                "Placeholder tokens detected in strict character mode:",
                AnsiColor::Red,
                true
            )
        );
        for fp in placeholder_failures.iter().take(50) {
            println!("  - {fp}");
        }
        bail!("strict placeholder check failed");
    }
    Ok(())
}

fn git_tracked_files(root: &Path) -> Result<Vec<PathBuf>> {
    let output = Command::new("git")
        .args(["ls-files", "-z"])
        .current_dir(root)
        .output()
        .context("run git ls-files -z")?;
    if !output.status.success() {
        bail!(
            "git ls-files failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .split('\0')
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .collect())
}

fn terminology_should_skip(path: &Path) -> bool {
    if let Some(name) = path.file_name().and_then(|value| value.to_str()) {
        if TERMINOLOGY_SKIP_NAMES.contains(&name) {
            return true;
        }
        if TERMINOLOGY_SKIP_SUFFIXES
            .iter()
            .any(|suffix| name.ends_with(suffix))
        {
            return true;
        }
    }
    if let Some(ext) = path.extension().and_then(|value| value.to_str()) {
        let with_dot = format!(".{ext}");
        if TERMINOLOGY_SKIP_EXTENSIONS.contains(&with_dot.as_str()) {
            return true;
        }
    }
    let parts: Vec<String> = path
        .iter()
        .map(|part| part.to_string_lossy().to_string())
        .collect();
    parts
        .iter()
        .any(|part| part == ".git" || part == "target" || part == "__pycache__")
}

fn load_banned_terms(root: &Path) -> Result<Vec<BannedTerm>> {
    let path = root.join("registry").join("terminology_standards.toml");
    let text = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let value: toml::Value =
        toml::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
    let rows = value
        .get("banned")
        .and_then(toml::Value::as_array)
        .cloned()
        .unwrap_or_default();
    Ok(rows
        .iter()
        .filter_map(toml::Value::as_table)
        .map(|table| BannedTerm {
            pattern: table
                .get("pattern")
                .and_then(toml::Value::as_str)
                .unwrap_or_default()
                .to_string(),
            replacement: table
                .get("replacement")
                .and_then(toml::Value::as_str)
                .unwrap_or_default()
                .to_string(),
            reason: table
                .get("reason")
                .and_then(toml::Value::as_str)
                .unwrap_or_default()
                .to_string(),
        })
        .collect())
}

fn compile_banned_patterns(banned: &[BannedTerm]) -> Vec<(Regex, BannedTerm)> {
    banned
        .iter()
        .map(|entry| {
            let pattern =
                if entry.pattern == entry.pattern.to_uppercase() && entry.pattern.contains('_') {
                    regex::escape(&entry.pattern)
                } else {
                    format!("(?i:{})", regex::escape(&entry.pattern))
                };
            (
                Regex::new(&pattern).expect("valid banned regex"),
                entry.clone(),
            )
        })
        .collect()
}

fn allowlist_patterns() -> Vec<Regex> {
    [
        r"Toulouse\s*\(?1977\)?",
        r"Harary.*frustration",
        r"spin.glass\s+frustration",
        r"Zaslavsky.*frustrat",
        r"\\cite\{.*Toulouse",
        r#"^pattern\s*=\s*""#,
        r#"^replacement\s*=\s*""#,
        r#"^reason\s*=\s*""#,
    ]
    .into_iter()
    .map(|pattern| Regex::new(pattern).expect("valid allowlist regex"))
    .collect()
}

fn is_allowlisted(line: &str, allowlist: &[Regex]) -> bool {
    allowlist.iter().any(|pattern| pattern.is_match(line))
}

fn build_candidate_lines_with_rg(
    root: &Path,
    files: &[PathBuf],
    banned: &[BannedTerm],
) -> Result<Vec<(PathBuf, usize, String)>> {
    let mut candidates = Vec::new();
    let mut seen = BTreeSet::new();
    for chunk in files.chunks(256) {
        let mut command = Command::new("rg");
        command
            .args([
                "--line-number",
                "--with-filename",
                "--no-heading",
                "--color",
                "never",
                "--fixed-strings",
                "--ignore-case",
                "--no-messages",
            ])
            .current_dir(root);
        for entry in banned {
            command.arg("-e").arg(&entry.pattern);
        }
        command.arg("--");
        for file in chunk {
            command.arg(file);
        }
        let output = command.output().context("run ripgrep prefilter")?;
        if !output.status.success() && output.status.code() != Some(1) {
            bail!("{}", String::from_utf8_lossy(&output.stderr).trim());
        }
        for raw_line in String::from_utf8_lossy(&output.stdout).lines() {
            let mut parts = raw_line.splitn(3, ':');
            let (Some(path), Some(line_no), Some(line_text)) =
                (parts.next(), parts.next(), parts.next())
            else {
                continue;
            };
            let Ok(number) = line_no.parse::<usize>() else {
                continue;
            };
            let key = (path.to_string(), number, line_text.to_string());
            if seen.insert(key.clone()) {
                candidates.push((PathBuf::from(key.0), key.1, key.2));
            }
        }
    }
    Ok(candidates)
}

fn run_terminology_gate(args: TerminologyArgs) -> Result<()> {
    let root = repo_root();
    let banned = load_banned_terms(&root)?;
    if banned.is_empty() {
        return Ok(());
    }
    let compiled = compile_banned_patterns(&banned);
    let allowlist = allowlist_patterns();
    let files = git_tracked_files(&root)?;
    let scan_files: Vec<PathBuf> = files
        .into_iter()
        .filter(|path| !terminology_should_skip(path) && root.join(path).is_file())
        .collect();
    let mut violations: Vec<(PathBuf, usize, String, BannedTerm)> = Vec::new();
    let mut engine = "python".to_string();

    if Command::new("rg").arg("--version").output().is_ok() && !scan_files.is_empty() {
        match build_candidate_lines_with_rg(&root, &scan_files, &banned) {
            Ok(candidates) => {
                engine = "ripgrep".to_string();
                for (rel_path, line_no, line) in candidates {
                    if is_allowlisted(&line, &allowlist) {
                        continue;
                    }
                    for (regex, entry) in &compiled {
                        if regex.is_match(&line) {
                            violations.push((
                                rel_path.clone(),
                                line_no,
                                line.trim().to_string(),
                                entry.clone(),
                            ));
                        }
                    }
                }
            }
            Err(err) => {
                eprintln!(
                    "WARNING: rg prefilter failed ({}); falling back to python scan",
                    err
                );
            }
        }
    }

    if engine == "python" {
        for rel_path in &scan_files {
            let Ok(text) = fs::read_to_string(root.join(rel_path)) else {
                continue;
            };
            for (line_no, line) in text.lines().enumerate() {
                if is_allowlisted(line, &allowlist) {
                    continue;
                }
                for (regex, entry) in &compiled {
                    if regex.is_match(line) {
                        violations.push((
                            rel_path.clone(),
                            line_no + 1,
                            line.trim().to_string(),
                            entry.clone(),
                        ));
                    }
                }
            }
        }
    }

    if !violations.is_empty() {
        if !args.quiet {
            println!(
                "{}",
                style_text(
                    &format!("FAIL: {} terminology violation(s) found:", violations.len()),
                    AnsiColor::Red,
                    true
                )
            );
            println!();
            for (rel_path, line_no, line_text, entry) in &violations {
                println!("  {}:{}", rel_path.display(), line_no);
                println!("    found:   {:?}", entry.pattern);
                println!("    replace: {:?}", entry.replacement);
                println!("    reason:  {}", entry.reason);
                let display = if line_text.len() > 120 {
                    format!("{}...", &line_text[..120])
                } else {
                    line_text.clone()
                };
                println!("    line:    {display}");
                println!();
            }
        }
        bail!("terminology violations found");
    }

    if !args.quiet {
        println!(
            "{}",
            style_text(
                &format!(
                    "OK: terminology gate passed ({} banned patterns, {} files scanned, engine={}).",
                    compiled.len(),
                    scan_files.len(),
                    engine
                ),
                AnsiColor::Green,
                true,
            )
        );
    }
    Ok(())
}

fn select_python() -> Option<String> {
    for candidate in ["python3", "python"] {
        if Command::new(candidate).arg("--version").output().is_ok() {
            return Some(candidate.to_string());
        }
    }
    None
}

fn run_doctor() -> Result<()> {
    println!("gemini-experiments doctor");
    let python = select_python();
    let python_version = if let Some(py) = &python {
        let output = Command::new(py)
            .args(["-c", "import sys; print(sys.version.split()[0])"])
            .output()
            .with_context(|| format!("run {py} for version"))?;
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    } else {
        "MISSING".to_string()
    };
    println!("python: {python_version}");
    println!();
    println!("Python modules:");
    for module in DOCTOR_PYTHON_MODULES {
        let ok = if let Some(py) = &python {
            Command::new(py)
                .args(["-c", &format!("import importlib.util, sys; sys.exit(0 if importlib.util.find_spec({module:?}) is not None else 1)")])
                .status()
                .map(|status| status.success())
                .unwrap_or(false)
        } else {
            false
        };
        println!("- {module}: {}", if ok { "OK" } else { "MISSING" });
    }
    println!();
    println!("System binaries:");
    for binary in DOCTOR_BINARIES {
        let ok = Command::new("sh")
            .args(["-lc", &format!("command -v {binary} >/dev/null 2>&1")])
            .status()
            .map(|status| status.success())
            .unwrap_or(false);
        println!("- {binary}: {}", if ok { "OK" } else { "MISSING" });
    }
    println!();
    println!("Next steps:");
    println!("- Core: `make test`");
    println!("- Native BLAS candidates: `make doctor-blas`");
    println!("- Optional requirements: `REQUIREMENTS.md`");
    Ok(())
}

fn wait_for_jsonrpc_id(
    rx: &mpsc::Receiver<Result<String, String>>,
    request_id: i64,
    timeout: Duration,
) -> Result<serde_json::Value> {
    loop {
        let line = rx
            .recv_timeout(timeout)
            .with_context(|| format!("timed out waiting for JSON-RPC response id {request_id}"))?
            .map_err(|err| anyhow::anyhow!(err))?;
        let value: serde_json::Value =
            serde_json::from_str(&line).with_context(|| format!("parse MCP response: {line}"))?;
        if value.get("id").and_then(serde_json::Value::as_i64) == Some(request_id) {
            return Ok(value);
        }
    }
}

fn run_mcp_smoke() -> Result<()> {
    let mut child = Command::new("source-analysis-mcp")
        .arg("serve")
        .current_dir(repo_root())
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .context("start source-analysis-mcp serve")?;

    let stdout = child.stdout.take().context("capture MCP stdout")?;
    let stderr = child.stderr.take().context("capture MCP stderr")?;
    let mut stdin = child.stdin.take().context("capture MCP stdin")?;
    let (tx, rx) = mpsc::channel::<Result<String, String>>();
    let stdout_tx = tx.clone();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let _ = stdout_tx.send(line.map_err(|err| format!("read MCP stdout: {err}")));
        }
    });
    std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            if !line.trim().is_empty() {
                eprintln!("source-analysis-mcp stderr: {line}");
            }
        }
    });

    let initialize = serde_json::json!({
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {
                "name": "open-gororoba-repo-utilities",
                "version": env!("CARGO_PKG_VERSION")
            }
        },
        "jsonrpc": "2.0",
        "id": 0
    });
    writeln!(stdin, "{initialize}").context("send MCP initialize")?;
    stdin.flush().context("flush MCP initialize")?;
    let init_response = wait_for_jsonrpc_id(&rx, 0, Duration::from_secs(10))?;
    if init_response.get("error").is_some() {
        bail!("MCP initialize failed: {init_response}");
    }

    let initialized = serde_json::json!({
        "method": "notifications/initialized",
        "jsonrpc": "2.0"
    });
    let list_tools = serde_json::json!({
        "method": "tools/list",
        "jsonrpc": "2.0",
        "id": 1
    });
    writeln!(stdin, "{initialized}").context("send MCP initialized notification")?;
    writeln!(stdin, "{list_tools}").context("send MCP tools/list")?;
    stdin.flush().context("flush MCP tools/list")?;

    let tools_response = wait_for_jsonrpc_id(&rx, 1, Duration::from_secs(10))?;
    if tools_response.get("error").is_some() {
        bail!("MCP tools/list failed: {tools_response}");
    }
    let tools = tools_response
        .pointer("/result/tools")
        .and_then(serde_json::Value::as_array)
        .context("MCP tools/list response missing result.tools array")?;
    let actual: BTreeSet<String> = tools
        .iter()
        .filter_map(|tool| tool.get("name").and_then(serde_json::Value::as_str))
        .map(str::to_string)
        .collect();
    let expected: BTreeSet<String> = SOURCE_ANALYSIS_MCP_TOOLS
        .iter()
        .map(|tool| (*tool).to_string())
        .collect();
    if actual != expected {
        let missing: Vec<_> = expected.difference(&actual).cloned().collect();
        let extra: Vec<_> = actual.difference(&expected).cloned().collect();
        bail!("source-analysis MCP tool mismatch; missing={missing:?} extra={extra:?}");
    }

    drop(stdin);
    let _ = child.kill();
    let _ = child.wait();
    println!(
        "OK: source-analysis MCP smoke listed {} expected tools.",
        SOURCE_ANALYSIS_MCP_TOOLS.len()
    );
    Ok(())
}

fn run_rocq_prepare_confine(args: CoqArgs) -> Result<()> {
    let text =
        fs::read_to_string(&args.src).with_context(|| format!("read {}", args.src.display()))?;
    let mut output = String::from(ROCQ_HEADER);
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("Theorem ") {
            output.push_str("Axiom ");
            output.push_str(rest);
        } else {
            output.push_str(line);
        }
        output.push('\n');
    }
    if let Some(parent) = args.dst.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    fs::write(&args.dst, output).with_context(|| format!("write {}", args.dst.display()))?;
    println!("Wrote {}", args.dst.display());
    Ok(())
}

fn docs_redirect_root(path: &str) -> String {
    let first_segment = path
        .strip_prefix('/')
        .unwrap_or(path)
        .split('/')
        .next()
        .unwrap_or_default();
    if !first_segment.is_empty() && first_segment != "book" && first_segment != "rustdoc" {
        format!("/{first_segment}/")
    } else {
        "/".to_string()
    }
}

fn simulate_docs_redirect(path: &str) -> String {
    let mut root = docs_redirect_root(path);
    for prefix in DOCS_REDIRECT_LEGACY_PREFIXES {
        if let Some(prefix_index) = path.find(prefix) {
            let prefix_part = &path[..prefix_index];
            root = if !prefix_part.is_empty() && prefix_part != "/" {
                format!("{}/", prefix_part.trim_end_matches('/'))
            } else {
                "/".to_string()
            };
            let suffix = &path[prefix_index + prefix.len()..];
            return format!("{root}rustdoc{suffix}");
        }
    }

    let book = format!("{root}book");
    let book_slash = format!("{root}book/");
    if path == root || path == book || path == book_slash {
        return book_slash;
    }

    let rustdoc = format!("{root}rustdoc");
    let rustdoc_slash = format!("{root}rustdoc/");
    if path == rustdoc || path == rustdoc_slash {
        return rustdoc_slash;
    }

    root
}

fn require_docs_file(docs_site_dir: &Path, relative_path: &str) -> Result<()> {
    let path = docs_site_dir.join(relative_path);
    if !path.is_file() {
        bail!("required docs artifact missing: {}", path.display());
    }
    Ok(())
}

fn require_docs_marker(docs_site_dir: &Path, relative_path: &str, marker: &str) -> Result<()> {
    let path = docs_site_dir.join(relative_path);
    let text = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    if !text.contains(marker) {
        bail!("expected marker missing in {}: {}", path.display(), marker);
    }
    Ok(())
}

fn run_docs_redirect_check(args: DocsRedirectArgs) -> Result<()> {
    for relative_path in DOCS_REDIRECT_REQUIRED_FILES {
        require_docs_file(&args.docs_site_dir, relative_path)?;
    }
    for (relative_path, marker) in DOCS_REDIRECT_REQUIRED_MARKERS {
        require_docs_marker(&args.docs_site_dir, relative_path, marker)?;
    }
    for (input, expected) in DOCS_REDIRECT_CASES {
        let output = simulate_docs_redirect(input);
        if output != *expected {
            bail!("redirect mismatch for '{input}': expected '{expected}', got '{output}'");
        }
        println!("OK: {input} -> {output}");
    }
    println!("OK: docs redirect checks passed.");
    Ok(())
}

/// Fail when a staged validation binary names a worktree that is gone. The
/// binary would abort at runtime on the first path it resolves through that
/// name, and the stamp cannot see it: the stamp gates the rebuild decision,
/// while the bytes come from the shared Cargo build-dir.
fn run_validation_tool_paths(args: ValidationToolPathsArgs) -> Result<()> {
    let current = args
        .current_root
        .canonicalize()
        .unwrap_or_else(|_| args.current_root.clone());
    let hits = repo_utilities::validation_tool_paths::scan_tools_dir(
        &args.tools_dir,
        &args.worktrees_root,
        &current,
    )
    .with_context(|| format!("scanning {}", args.tools_dir.display()))?;
    if hits.is_empty() {
        println!(
            "OK: no staged validation tool names a vanished worktree under {}.",
            args.worktrees_root.display()
        );
        return Ok(());
    }
    for hit in &hits {
        eprintln!(
            "[validation-tool-paths] {} embeds {}, under the removed checkout {}",
            hit.binary.display(),
            hit.embedded,
            hit.vanished_root
        );
    }
    bail!(
        "{} staged validation tool path reference(s) point at a removed worktree; run `make validation-tools-rebuild`",
        hits.len()
    );
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Commands::AnsiCheck(args) => run_character_policy(args),
        Commands::TerminologyGate(args) => run_terminology_gate(args),
        Commands::Doctor => run_doctor(),
        Commands::McpSmoke => run_mcp_smoke(),
        Commands::DocsRedirectCheck(args) => run_docs_redirect_check(args),
        Commands::RocqPrepareConfine(args) => run_rocq_prepare_confine(args),
        Commands::ValidationToolPaths(args) => run_validation_tool_paths(args),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            let msg = err.to_string();
            if msg != "ansi check failed"
                && msg != "strict placeholder check failed"
                && msg != "terminology violations found"
            {
                eprintln!("{err:#}");
            }
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn docs_redirect_cases_match_legacy_policy() {
        for (input, expected) in DOCS_REDIRECT_CASES {
            assert_eq!(simulate_docs_redirect(input), *expected, "input={input}");
        }
    }

    #[test]
    fn docs_redirect_check_accepts_minimal_site() {
        let temp_dir = std::env::temp_dir().join(format!(
            "gororoba_docs_redirect_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time after unix epoch")
                .as_nanos()
        ));
        fs::create_dir_all(&temp_dir).expect("create temp docs site");
        fs::write(
            temp_dir.join("404.html"),
            "/.cache/cargo-default-target/doc\n/cache/gate-target/doc\nwindow.location.replace\n",
        )
        .expect("write 404.html");
        fs::write(temp_dir.join("book.html"), "./book/").expect("write book.html");
        fs::write(temp_dir.join("rustdoc.html"), "./rustdoc/").expect("write rustdoc.html");
        fs::write(temp_dir.join("index.html"), "index").expect("write index.html");
        fs::write(temp_dir.join(".nojekyll"), "").expect("write .nojekyll");

        run_docs_redirect_check(DocsRedirectArgs {
            docs_site_dir: temp_dir.clone(),
        })
        .expect("minimal docs redirect site passes");

        let _ = fs::remove_dir_all(temp_dir);
    }
}

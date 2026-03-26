use anstyle::{AnsiColor, Style};
use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use regex::Regex;
use std::{
    collections::{BTreeSet, HashSet},
    fs,
    path::{Path, PathBuf},
    process::{Command, ExitCode},
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
    #[command(name = "rocq-prepare-confine", visible_alias = "coq-prepare-confine")]
    RocqPrepareConfine(CoqArgs),
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

#[derive(Clone)]
struct BannedTerm {
    pattern: String,
    replacement: String,
    reason: String,
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crate must be nested under repo/crates")
        .to_path_buf()
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
    matches!(
        val,
        0x1F600..=0x1F64F | // Emoticons
        0x1F300..=0x1F5FF | // Misc Symbols and Pictographs
        0x1F680..=0x1F6FF | // Transport and Map
        0x2600..=0x26FF   | // Misc Symbols
        0x2700..=0x27BF   | // Dingbats
        0xFE00..=0xFE0F   | // Variation Selectors
        0x1F900..=0x1F9FF | // Supplemental Symbols and Pictographs
        0x1F1E6..=0x1F1FF   // Flags
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

fn has_disallowed_characters(text: &str) -> bool {
    text.chars().any(|ch| {
        if ch.is_control() && !matches!(ch, '\n' | '\r' | '\t') {
            return true;
        }
        if is_emoji(ch) {
            return true;
        }
        false
    })
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
        if let Some(bad_ch) = new_text.chars().find(|&ch| {
            (ch.is_control() && !matches!(ch, '\n' | '\r' | '\t')) || is_emoji(ch)
        }) {
            failures.push(format!("{} (first bad char: U+{:04X})", rel, bad_ch as u32));
            continue;
        }
        if args.strict_placeholders {
            let in_scope = scope_prefixes.is_empty()
                || scope_prefixes
                    .iter()
                    .any(|prefix| rel == *prefix || rel.starts_with(&format!("{prefix}/")));
            if in_scope && !allowlist.contains(&rel) && (new_text.contains("<U+") || new_text.contains("<EMOJI+")) {
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

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Commands::AnsiCheck(args) => run_character_policy(args),
        Commands::TerminologyGate(args) => run_terminology_gate(args),
        Commands::Doctor => run_doctor(),
        Commands::RocqPrepareConfine(args) => run_rocq_prepare_confine(args),
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

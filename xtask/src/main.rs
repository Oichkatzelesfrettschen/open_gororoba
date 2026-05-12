use anstyle::{AnsiColor, Color, Style};
use anyhow::{Context, Result, bail};
use chrono::{Local, SecondsFormat};
use clap::Parser;
use provenance_store::ProvenanceStore;
use rusqlite::Connection;
use serde::Serialize;
use std::{
    collections::BTreeMap,
    env, fs,
    fs::OpenOptions,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};
use tempfile::tempdir;
use verified_core::topology::HardwareTopology;
use walkdir::WalkDir;

const HEADER_STYLE: Style = Style::new()
    .fg_color(Some(Color::Ansi(AnsiColor::Cyan)))
    .bold();
const OK_STYLE: Style = Style::new()
    .fg_color(Some(Color::Ansi(AnsiColor::Green)))
    .bold();
const FAIL_STYLE: Style = Style::new()
    .fg_color(Some(Color::Ansi(AnsiColor::Red)))
    .bold();
const WARN_STYLE: Style = Style::new().fg_color(Some(Color::Ansi(AnsiColor::Yellow)));
const INFO_STYLE: Style = Style::new().fg_color(Some(Color::Ansi(AnsiColor::Blue)));
const RESET: &str = "\x1b[0m";

#[derive(Debug, Serialize)]
struct SchemaSnapshot {
    generated_from: String,
    object_count: usize,
    objects: Vec<SchemaObject>,
}

#[derive(Debug, Serialize)]
struct SchemaObject {
    schema: String,
    name: String,
    object_type: String,
    column_count: i64,
    without_rowid: bool,
    strict: bool,
    columns: Vec<SchemaColumn>,
    foreign_keys: Vec<SchemaForeignKey>,
    indexes: Vec<SchemaIndex>,
}

#[derive(Debug, Serialize)]
struct SchemaColumn {
    cid: i64,
    name: String,
    declared_type: String,
    not_null: bool,
    default_value: Option<String>,
    primary_key_position: i64,
    hidden: i64,
}

#[derive(Debug, Serialize)]
struct SchemaForeignKey {
    id: i64,
    seq: i64,
    ref_table: String,
    from_column: String,
    to_column: String,
    on_update: String,
    on_delete: String,
    match_kind: String,
}

#[derive(Debug, Serialize)]
struct SchemaIndex {
    seq: i64,
    name: String,
    unique: bool,
    origin: String,
    partial: bool,
    columns: Vec<SchemaIndexColumn>,
}

#[derive(Debug, Serialize)]
struct SchemaIndexColumn {
    seqno: i64,
    cid: i64,
    name: Option<String>,
    descending: bool,
    collation: String,
    key: bool,
}

#[derive(Debug, Serialize)]
struct HostProfile {
    physical_core_ids: Vec<usize>,
    physical_core_count: usize,
    l3_cache_bytes: usize,
    l3_safe_working_set_bytes: usize,
    worker_budget: usize,
    cargo_jobs: usize,
    rayon_threads: usize,
    rust_test_threads: usize,
    nextest_test_threads: usize,
    pytest_workers: usize,
}

const INLINE_TEST_MARKERS: &[&str] = &["#[test]", "#[cfg(test)]", "mod tests"];
const GATE_AUDIT_TAIL_LINE_COUNT: usize = 20;

#[derive(Debug, Serialize)]
struct GateAuditStepRecord {
    name: String,
    exit_code: i32,
    log: String,
}

#[derive(Parser, Debug)]
#[command(
    name = "local-nextest-plan",
    about = "Run a package-aware grouped local nextest plan"
)]
struct LocalNextestCli {
    #[arg(long)]
    build_jobs: String,
    #[arg(long)]
    test_threads: String,
    #[arg(long, default_value = "")]
    filterset: String,
    #[arg(long)]
    timing_json_out: Option<PathBuf>,
    /// Which test kinds to run. `lib` (default) runs only library unit
    /// tests; `all` runs lib + integration tests. Integration test
    /// binaries are the dominant link-time cost in the local pre-push
    /// gate (~4m30s for 441 binaries on this workspace), so the
    /// default skips them. CI on PR open should use `all`.
    #[arg(long, default_value = "lib")]
    kinds: String,
    packages: Vec<String>,
}

/// Tier-5B (2026-05-12): `cargo xtask gate-local` -- pre-push gate
/// driver with structured timing output.
///
/// Wraps the three Make sub-targets (check, rust-regression-scoped,
/// governance-gate-readonly) and records per-phase elapsed time +
/// exit code to a JSONL timing log under
/// `data/output/audit/<YYYY-MM-DD>/gate-timing-<unix-ts>.jsonl`.
///
/// This is the replacement for the inline shell loop in `gate-local`.
/// Currently the Makefile target still drives end-to-end orchestration;
/// the xtask version is an opt-in via `make gate-local-xtask`.
#[derive(Parser, Debug)]
#[command(name = "gate-local", about = "Run pre-push gate with structured timing JSONL output")]
struct GateLocalCli {
    /// Path to the workspace-routing binary cache (default:
    /// .cache/gate-target/gate-tools/workspace-routing).
    #[arg(long)]
    routing_bin: Option<PathBuf>,
    /// Write timing JSONL to this path. Default:
    /// data/output/audit/<YYYY-MM-DD>/gate-timing-<unix-ts>.jsonl
    #[arg(long)]
    timing_json: Option<PathBuf>,
    /// Force run rust-regression-scoped regardless of routing.
    #[arg(long)]
    force_rust: bool,
    /// Force run make check regardless of routing.
    #[arg(long)]
    force_check: bool,
    /// Force run governance-gate-readonly regardless of routing.
    #[arg(long)]
    force_governance: bool,
}

/// `cargo xtask gate-tools-status` -- inspect cached gate-tool binary
/// freshness.
///
/// Prints, for each cached binary under
/// `.cache/gate-target/gate-tools/`, its mtime, size, and whether its
/// declared Makefile source dependencies are newer (would trigger
/// rebuild on next gate-local invocation).
#[derive(Parser, Debug)]
#[command(name = "gate-tools-status", about = "Inspect cached gate-tool binaries and source-dep freshness")]
struct GateToolsStatusCli {
    /// Override the gate-tools cache root.
    /// Default: $(REPO_CARGO_TARGET_DIR)/gate-tools = .cache/gate-target/gate-tools.
    #[arg(long)]
    tools_dir: Option<PathBuf>,
}

/// `cargo xtask gate-timing-summary` -- aggregate recent gate-local
/// timing JSONL files into a per-phase stats table.
///
/// Reads files under `data/output/audit/<YYYY-MM-DD>/gate-timing-*.jsonl`
/// and computes count/mean/median/p95/max/min for each phase's
/// `elapsed_sec`.
#[derive(Parser, Debug)]
#[command(name = "gate-timing-summary", about = "Aggregate gate-local timing JSONL")]
struct GateTimingSummaryCli {
    /// Look-back window in days. Default 30.
    #[arg(long, default_value_t = 30)]
    since_days: u64,
    /// Root directory for timing JSONL files.
    /// Default: data/output/audit/
    #[arg(long)]
    audit_root: Option<PathBuf>,
    /// Filter to a single phase name (cache-check, check,
    /// rust-regression-scoped, governance-gate-readonly).
    #[arg(long)]
    phase: Option<String>,
    /// Output format: `table` (default) or `json`.
    #[arg(long, default_value = "table")]
    format: String,
    /// Show the last N raw run lines for each phase. Default 0 = none.
    #[arg(long, default_value_t = 0)]
    last: usize,
}

/// `cargo xtask gate-timing-regression-check` -- detect regressions
/// vs a baseline.
///
/// Loads recent gate-timing JSONL files (same source as
/// gate-timing-summary), computes per-phase median over baseline,
/// then checks the latest run's elapsed against `baseline_median *
/// threshold`. Non-zero exit if any phase regressed.
#[derive(Parser, Debug)]
#[command(
    name = "gate-timing-regression-check",
    about = "Compare latest gate timing to baseline median; fail on regression"
)]
struct GateTimingRegressionCheckCli {
    /// Baseline look-back in days. Default 14.
    #[arg(long, default_value_t = 14)]
    baseline_days: u64,
    /// Threshold multiplier: regression if elapsed > median * threshold.
    /// Default 2.0.
    #[arg(long, default_value_t = 2.0)]
    threshold: f64,
    /// Minimum baseline sample count to enable check. Default 5.
    #[arg(long, default_value_t = 5)]
    min_samples: usize,
    /// Audit root override.
    #[arg(long)]
    audit_root: Option<PathBuf>,
}

#[derive(Parser, Debug)]
#[command(
    name = "sparse-profile",
    about = "Run reproducible Nsight profiling for the sparse 1024^3 CUDA benchmark"
)]
struct SparseProfileCli {
    #[arg(long, default_value = "gpu_sparse_1024")]
    bench: String,
    #[arg(long, default_value = "both")]
    mode: String,
    #[arg(long, default_value = "reports/nsight")]
    output_dir: PathBuf,
    #[arg(long, default_value_t = false)]
    run_ncu: bool,
    #[arg(long, default_value_t = 5)]
    ncu_launch_skip: u32,
    #[arg(long, default_value_t = 3)]
    ncu_launch_count: u32,
}

#[derive(Parser, Debug)]
#[command(
    name = "gpu-profile",
    about = "Run diffable GPU benchmark/profile sweeps for sparse CUDA workloads"
)]
struct GpuProfileCli {
    #[arg(long, default_value = "gpu_sparse_1024")]
    bench: String,
    #[arg(long, default_value = "both")]
    mode: String,
    #[arg(long, default_value = "reports/gpu_profiles")]
    output_dir: PathBuf,
    #[arg(long, value_delimiter = ',')]
    tile_bytes: Vec<u64>,
    #[arg(long, default_value_t = false)]
    run_nsys: bool,
    #[arg(long, default_value_t = false)]
    run_ncu: bool,
}

#[derive(Debug, Serialize)]
struct SparseProfileRunRecord {
    mode: String,
    nsys_report: Option<PathBuf>,
    ncu_csv: Option<PathBuf>,
    skipped_reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct SparseProfileManifest {
    bench: String,
    binary: Option<PathBuf>,
    generated_at: String,
    nsys_available: bool,
    ncu_available: bool,
    runs: Vec<SparseProfileRunRecord>,
}

#[derive(Debug, Serialize)]
struct GpuProfileSweepRow {
    bench: String,
    mode: String,
    tile_bytes: Option<u64>,
    elapsed_seconds: Option<f64>,
    throughput_mlups: Option<f64>,
    effective_glups: Option<f64>,
    stdout_path: PathBuf,
    nsys_report: Option<PathBuf>,
    ncu_csv: Option<PathBuf>,
    skipped_reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct GpuProfileSweepManifest {
    bench: String,
    generated_at: String,
    rows: Vec<GpuProfileSweepRow>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PackagePlan {
    has_lib_tests: bool,
    tests: Vec<String>,
}

#[derive(Debug)]
struct TimingRecorder {
    output_path: Option<PathBuf>,
    total_start: Instant,
    run_count: u64,
    skip_count: u64,
}

impl TimingRecorder {
    fn new(output_path: Option<PathBuf>) -> Self {
        Self {
            output_path,
            total_start: Instant::now(),
            run_count: 0,
            skip_count: 0,
        }
    }

    fn write(&self, value: serde_json::Value) -> Result<()> {
        let Some(path) = &self.output_path else {
            return Ok(());
        };
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create timing output directory {}", parent.display()))?;
        }
        let mut handle = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("open timing output {}", path.display()))?;
        writeln!(handle, "{}", serde_json::to_string(&value)?)
            .with_context(|| format!("write timing output {}", path.display()))?;
        Ok(())
    }

    fn record_skip(&mut self, package: &str, reason: &str) -> Result<()> {
        self.skip_count += 1;
        self.write(serde_json::json!({
            "kind": "skip",
            "package": package,
            "reason": reason,
        }))
    }

    fn record_run(
        &mut self,
        packages: &[String],
        targets: &serde_json::Value,
        command: &[String],
        returncode: i32,
        elapsed_sec: f64,
    ) -> Result<()> {
        self.run_count += 1;
        self.write(serde_json::json!({
            "kind": "run",
            "packages": packages,
            "targets": targets,
            "command": command,
            "returncode": returncode,
            "elapsed_sec": elapsed_sec,
        }))
    }

    fn record_summary(&self, returncode: i32) -> Result<()> {
        self.write(serde_json::json!({
            "kind": "summary",
            "run_count": self.run_count,
            "skip_count": self.skip_count,
            "returncode": returncode,
            "total_elapsed_sec": self.total_start.elapsed().as_secs_f64(),
        }))
    }
}

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let Some(command) = args.next() else {
        println!(
            "{HEADER_STYLE}usage: cargo run -p xtask -- <db-docs|host-profile|local-nextest-plan|gate-local|gate-tools-status|gate-timing-summary|gate-timing-regression-check|gate-audit|audit-deep|registry-emit-all-mirrors|sparse-profile|gpu-profile|ci-route|ascii-check|ascii-cleanup|coq-stub|convos-chunk|terminology-gate|cpd-file-list|worker-budget> [args]{RESET}"
        );
        return Ok(());
    };
    match command.as_str() {
        "db-docs" => run_db_docs(args.any(|arg| arg == "--check")),
        "host-profile" => {
            let mut format = "shell".to_string();
            let mut iter = args.peekable();
            while let Some(arg) = iter.next() {
                match arg.as_str() {
                    "--format" => {
                        let Some(value) = iter.next() else {
                            bail!("host-profile --format requires a value");
                        };
                        format = value;
                    }
                    other => bail!("unknown host-profile argument: {other}"),
                }
            }
            run_host_profile(&format)
        }
        "local-nextest-plan" => run_local_nextest_plan(LocalNextestCli::try_parse_from(
            std::iter::once("local-nextest-plan".to_string()).chain(args),
        )?),
        "gate-local" => {
            let cli = GateLocalCli::try_parse_from(
                std::iter::once("gate-local".to_string()).chain(args),
            )?;
            let exit_code = run_gate_local(cli)?;
            if exit_code != 0 {
                std::process::exit(exit_code);
            }
            Ok(())
        }
        "gate-timing-summary" => {
            let cli = GateTimingSummaryCli::try_parse_from(
                std::iter::once("gate-timing-summary".to_string()).chain(args),
            )?;
            run_gate_timing_summary(cli)
        }
        "gate-tools-status" => {
            let cli = GateToolsStatusCli::try_parse_from(
                std::iter::once("gate-tools-status".to_string()).chain(args),
            )?;
            run_gate_tools_status(cli)
        }
        "gate-timing-regression-check" => {
            let cli = GateTimingRegressionCheckCli::try_parse_from(
                std::iter::once("gate-timing-regression-check".to_string()).chain(args),
            )?;
            let exit_code = run_gate_timing_regression_check(cli)?;
            if exit_code != 0 {
                std::process::exit(exit_code);
            }
            Ok(())
        }
        "gate-audit" => {
            let cfg = parse_gate_audit_args(args)?;
            run_gate_audit(cfg)
        }
        "audit-deep" => {
            // PH-5.A: structured audit-deep composite. Wraps the
            // Makefile audit-deep chain (rust-clippy + cargo-deny-check +
            // dep-audit + cpd-audit) but emits per-step exit codes,
            // log files, and a Markdown summary instead of unstructured
            // shell output. Mirrors the run_gate_audit reporting
            // surface so a single archival workflow can consume both.
            let cfg = parse_gate_audit_args(args)?;
            run_audit_deep(cfg)
        }
        "registry-emit-all-mirrors" => {
            // PH-5.B: replaces the 54-line shell heredoc that the
            // Makefile `registry-export-markdown` target carried. The
            // shell version invoked `registry-emit Xmirror --output Y`
            // sequentially 23 times with a hand-maintained list of
            // (mirror_kind, output_path) pairs. The xtask version owns
            // the list as Rust data so adding a new mirror is a single
            // entry and the loop is exit-coded properly. See
            // memory feedback_install_source_priority.md for the
            // Makefile -> xtask migration policy.
            run_registry_emit_all_mirrors()
        }
        "sparse-profile" => run_sparse_profile(SparseProfileCli::try_parse_from(
            std::iter::once("sparse-profile".to_string()).chain(args),
        )?),
        "gpu-profile" => run_gpu_profile(GpuProfileCli::try_parse_from(
            std::iter::once("gpu-profile".to_string()).chain(args),
        )?),
        "ci-route" => run_ci_route(CiRouteCli::try_parse_from(
            std::iter::once("ci-route".to_string()).chain(args),
        )?),
        "ascii-check" => run_ascii_check(args.any(|arg| arg == "--fix")),
        "ascii-cleanup" => run_ascii_cleanup(args.any(|arg| arg == "--fix")),
        "coq-stub" => {
            let src = args.next().expect("coq-stub requires src path");
            let dst = args.next().expect("coq-stub requires dst path");
            run_coq_stub(Path::new(&src), Path::new(&dst))
        }
        "convos-chunk" => {
            let path = args.next().expect("convos-chunk requires path");
            let lines = args.next().and_then(|s| s.parse().ok()).unwrap_or(800);
            let prefix = args.next().unwrap_or_else(|| "C1".to_string());
            run_convos_chunk(Path::new(&path), lines, &prefix)
        }
        "terminology-gate" => run_terminology_gate(args.any(|arg| arg == "--quiet")),
        "cpd-file-list" => {
            let mut output = PathBuf::from("/tmp/cpd_src_list.txt");
            let mut iter = args.peekable();
            while let Some(arg) = iter.next() {
                match arg.as_str() {
                    "--output" => {
                        let Some(value) = iter.next() else {
                            bail!("cpd-file-list --output requires a value");
                        };
                        output = PathBuf::from(value);
                    }
                    other => bail!("unknown cpd-file-list argument: {other}"),
                }
            }
            run_cpd_file_list(&output)
        }
        "worker-budget" => run_worker_budget(),
        other => bail!("unknown xtask command: {other}"),
    }
}

#[derive(serde::Deserialize)]
struct BannedTerm {
    pattern: String,
    replacement: String,
    reason: String,
}

#[derive(serde::Deserialize)]
struct TerminologyStandards {
    banned: Vec<BannedTerm>,
}

fn run_terminology_gate(quiet: bool) -> Result<()> {
    let repo_root = env::current_dir()?;
    let toml_path = repo_root.join("registry/terminology_standards.toml");
    if !toml_path.exists() {
        return Ok(());
    }

    let toml_text = fs::read_to_string(toml_path)?;
    let standards: TerminologyStandards = toml::from_str(&toml_text)?;

    let mut compiled = Vec::new();
    for entry in &standards.banned {
        let re = if entry.pattern == entry.pattern.to_uppercase() && entry.pattern.contains('_') {
            regex::Regex::new(&regex::escape(&entry.pattern))?
        } else {
            regex::RegexBuilder::new(&regex::escape(&entry.pattern))
                .case_insensitive(true)
                .build()?
        };
        compiled.push((re, entry));
    }

    let mut violations = 0;
    let skip_dirs = [".git", "target", "venv", "convos", "data", "reports"];
    let skip_exts = ["png", "jpg", "pdf", "zip", "so", "o", "h5", "npy"];

    for entry in WalkDir::new(&repo_root) {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            if skip_dirs.iter().any(|&d| name == d) {
                continue;
            }
        }
        if !path.is_file() {
            continue;
        }
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if skip_exts.contains(&ext) {
            continue;
        }

        let text = match fs::read_to_string(path) {
            Ok(t) => t,
            Err(_) => continue,
        };

        for (lineno, line) in text.lines().enumerate() {
            for (re, entry) in &compiled {
                if re.is_match(line) {
                    violations += 1;
                    if !quiet {
                        println!(
                            "  {FAIL_STYLE}[FAIL]{RESET} {}:{}: violation of pattern '{}'",
                            path.strip_prefix(&repo_root)?.display(),
                            lineno + 1,
                            entry.pattern
                        );
                        println!("    {INFO_STYLE}reason:{RESET}    {}", entry.reason);
                        println!("    {INFO_STYLE}suggested:{RESET} {}", entry.replacement);
                    }
                }
            }
        }
    }

    if violations > 0 {
        println!(
            "{FAIL_STYLE}Terminology gate failed with {} violations.{RESET}",
            violations
        );
        bail!("Terminology gate failed.");
    }

    if !quiet {
        println!("{OK_STYLE}--- Terminology Gate Passed ---{RESET}");
    }
    Ok(())
}

fn run_convos_chunk(path: &Path, chunk_lines: usize, prefix: &str) -> Result<()> {
    let content = fs::read_to_string(path)?;
    let total_lines = content.lines().count();
    let n_chunks = total_lines.div_ceil(chunk_lines);

    println!("{INFO_STYLE}path:{RESET}        {}", path.display());
    println!("{INFO_STYLE}lines:{RESET}       {}", total_lines);
    println!("{INFO_STYLE}chunk_lines:{RESET} {}", chunk_lines);
    println!("{INFO_STYLE}chunks:{RESET}      {}", n_chunks);
    println!();

    for i in 0..n_chunks {
        let start = i * chunk_lines + 1;
        let end = ((i + 1) * chunk_lines).min(total_lines);
        let chunk_id = format!("{}-{:04}", prefix, i + 1);
        println!("{HEADER_STYLE}{}:{RESET} L{}-L{}", chunk_id, start, end);
    }
    Ok(())
}

fn run_coq_stub(src: &Path, dst: &Path) -> Result<()> {
    let header = "From Stdlib Require Import String.\nRequire Import ConfineModel.\n\nOpen Scope string_scope.\n\n";
    let text = fs::read_to_string(src)?;
    let mut out = String::from(header);
    for line in text.lines() {
        if line.starts_with("Theorem ") {
            out.push_str("Axiom ");
            out.push_str(line.strip_prefix("Theorem ").unwrap_or(line));
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(dst, out)?;
    println!("  [OK]   Wrote {}", dst.display());
    Ok(())
}

fn run_ascii_check(fix: bool) -> Result<()> {
    println!("{HEADER_STYLE}--- Repo ANSI-safe UTF-8 Gate ---{RESET}");
    let repo_root = env::current_dir()?;
    let mut failures = Vec::new();

    let skip_dirs = [".git", "target", "venv", "convos", "data", "reports"];

    for entry in WalkDir::new(&repo_root) {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            if skip_dirs.iter().any(|&d| name == d) {
                continue;
            }
        }
        if !path.is_file() {
            continue;
        }

        // Skip binaries by extension
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if ["png", "jpg", "pdf", "xlsx", "zip", "so", "o"].contains(&ext) {
            continue;
        }

        let content = fs::read(path)?;
        let mut non_ascii = false;
        for &b in &content {
            if b > 127 {
                non_ascii = true;
                break;
            }
        }

        if non_ascii {
            failures.push(path.to_owned());
            let rel_path = path.strip_prefix(&repo_root)?.display();
            if !fix {
                println!("  {FAIL_STYLE}[FAIL]{RESET} {}", rel_path);
            } else {
                let mut fixed = Vec::with_capacity(content.len());
                for &b in &content {
                    if b <= 127 {
                        fixed.push(b);
                    } else {
                        fixed.push(b'?');
                    }
                }
                fs::write(path, fixed)?;
                println!("  {OK_STYLE}[FIXED]{RESET} {}", rel_path);
            }
        }
    }

    if !failures.is_empty() && !fix {
        println!(
            "{FAIL_STYLE}Found {} files with non-ASCII characters.{RESET}",
            failures.len()
        );
        bail!("ASCII check failed.");
    }

    println!("{OK_STYLE}ASCII check passed.{RESET}");
    Ok(())
}

fn run_ascii_cleanup(fix: bool) -> Result<()> {
    println!("{HEADER_STYLE}--- ASCII Placeholder Cleanup ---{RESET}");
    let repo_root = env::current_dir()?;
    let mut token_map = BTreeMap::new();
    token_map.insert("<U+00B1>", "+/-");
    token_map.insert("<U+00B7>", "*");
    token_map.insert("<U+039B>", "Lambda");
    token_map.insert("<U+0393>", "Gamma");
    token_map.insert("<U+03A3>", "Sigma");
    token_map.insert("<U+03A8>", "\\Psi");
    token_map.insert("<U+03A6>", "\\Phi");
    token_map.insert("<U+03B7>", "eta");
    token_map.insert("<U+03BD>", "\\nu");
    token_map.insert("<U+03C1>", "\\rho");
    token_map.insert("<U+03C4>", "\\tau");
    token_map.insert("<U+03C6>", "\\phi");
    token_map.insert("<U+03C9>", "omega");
    token_map.insert("<U+2020>", "dagger");
    token_map.insert("<U+2194>", "<->");
    token_map.insert("<U+2193>", "down");
    token_map.insert("<U+21A6>", "|->");
    token_map.insert("<U+2202>", "\\partial");
    token_map.insert("<U+2203>", "exists");
    token_map.insert("<U+2207>", "\\nabla");
    token_map.insert("<U+2208>", "in");
    token_map.insert("<U+2218>", "circ");
    token_map.insert("<U+221A>", "sqrt");
    token_map.insert("<U+2212>", "-");
    token_map.insert("<U+222B>", "int");
    token_map.insert("<U+223C>", "~");
    token_map.insert("<U+2248>", "approx");
    token_map.insert("<U+2243>", "approx");
    token_map.insert("<U+2264>", "<=");
    token_map.insert("<U+2265>", ">=");
    token_map.insert("<U+2282>", "subseteq");
    token_map.insert("<U+2295>", "\\oplus");
    token_map.insert("<U+2297>", "\\otimes");
    token_map.insert("<U+22C6>", "\\star");
    token_map.insert("<U+22A5>", "bot");
    token_map.insert("<U+2609>", "_sun");

    let skip_dirs = [".git", "target", "venv", "convos"];

    for entry in WalkDir::new(&repo_root) {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            if skip_dirs.iter().any(|&d| name == d) {
                continue;
            }
        }
        if !path.is_file() {
            continue;
        }

        let mut text = match fs::read_to_string(path) {
            Ok(t) => t,
            Err(_) => continue,
        };

        if !text.contains("<U+") {
            continue;
        }

        let original = text.clone();
        for (token, replacement) in &token_map {
            text = text.replace(token, replacement);
        }

        if text != original {
            let rel_path = path.strip_prefix(&repo_root)?.display();
            if fix {
                fs::write(path, &text)?;
                println!("  {OK_STYLE}[FIXED]{RESET}   {}", rel_path);
            } else {
                println!("  {WARN_STYLE}[PENDING]{RESET} {}", rel_path);
            }
        }
    }
    println!("{OK_STYLE}Cleanup complete.{RESET}");
    Ok(())
}

#[derive(Parser, Debug)]
#[command(name = "ci-route", about = "Map changed files to affected Rust crates")]
struct CiRouteCli {
    #[arg(long)]
    local: bool,
    #[arg(long)]
    base: Option<String>,
    #[arg(long)]
    verbose: bool,
}

fn run_ci_route(cli: CiRouteCli) -> Result<()> {
    let base = cli.base.unwrap_or_else(|| "HEAD~1".to_string());

    // Get changed files via git
    let output = Command::new("git")
        .args(["diff", "--name-only", &base, "HEAD"])
        .output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let files: Vec<String> = stdout.lines().map(|s| s.to_string()).collect();

    if cli.verbose {
        eprintln!("[ci-route] Changed files: {}", files.len());
    }

    // 1. Force workspace triggers
    let workspace_triggers = [
        "Cargo.toml",
        "Cargo.lock",
        "rust-toolchain.toml",
        "Makefile",
        "agents.toml",
    ];
    let mut force_workspace = false;
    for f in &files {
        if workspace_triggers.iter().any(|&t| f == t) {
            force_workspace = true;
            break;
        }
    }

    if force_workspace {
        if cli.local {
            println!("--workspace");
        } else {
            println!("::set-output name=rust_scope::--workspace");
        }
        return Ok(());
    }

    // 2. Identify affected crates
    let mut affected = std::collections::HashSet::new();
    for f in &files {
        if f.starts_with("crates/") {
            let parts: Vec<&str> = f.split('/').collect();
            if parts.len() >= 2 {
                affected.insert(parts[1].to_string());
            }
        }
    }

    if affected.is_empty() {
        if !cli.local {
            println!("::set-output name=rust_scope::");
        }
        return Ok(());
    }

    // 3. TODO: Transitive closure (requires parsing Cargo.tomls)
    // For now, emit direct affected crates
    let mut scope = String::new();
    for c in affected {
        scope.push_str(&format!("-p {} ", c));
    }

    if cli.local {
        println!("{}", scope.trim());
    } else {
        println!("::set-output name=rust_scope::{}", scope.trim());
    }

    Ok(())
}

fn run_gpu_profile(cli: GpuProfileCli) -> Result<()> {
    fs::create_dir_all(&cli.output_dir)
        .with_context(|| format!("create output directory {}", cli.output_dir.display()))?;
    run_status(
        Command::new("cargo")
            .arg("bench")
            .arg("--no-run")
            .arg("-p")
            .arg("lbm_3d_cuda")
            .arg("--bench")
            .arg(&cli.bench),
        "build gpu benchmark",
    )?;
    let binary = locate_sparse_bench_binary(&cli.bench)?;
    let nsys_available = cli.run_nsys && tool_available("nsys", &["--version"]);
    let ncu_available = cli.run_ncu && tool_available("ncu", &["--version"]);
    let tile_bytes = if cli.tile_bytes.is_empty() {
        vec![None]
    } else {
        cli.tile_bytes.into_iter().map(Some).collect()
    };
    let mut rows = Vec::new();
    for mode in sparse_profile_modes(&cli.mode)? {
        for tile in &tile_bytes {
            rows.push(run_gpu_profile_case(
                &cli.bench,
                &binary,
                &cli.output_dir,
                mode,
                *tile,
                nsys_available,
                ncu_available,
            )?);
        }
    }
    let manifest = GpuProfileSweepManifest {
        bench: cli.bench.clone(),
        generated_at: Local::now().to_rfc3339_opts(SecondsFormat::Secs, true),
        rows,
    };
    let manifest_path = cli
        .output_dir
        .join(format!("{}_sweep_manifest.json", cli.bench));
    fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)
        .with_context(|| format!("write {}", manifest_path.display()))?;
    let summary_path = cli
        .output_dir
        .join(format!("{}_sweep_summary.csv", cli.bench));
    write_gpu_profile_summary_csv(&summary_path, &manifest.rows)?;
    println!("{}", manifest_path.display());
    println!("{}", summary_path.display());
    Ok(())
}

fn run_gpu_profile_case(
    bench: &str,
    binary: &Path,
    output_dir: &Path,
    mode: &str,
    tile_bytes: Option<u64>,
    nsys_available: bool,
    ncu_available: bool,
) -> Result<GpuProfileSweepRow> {
    let label = match tile_bytes {
        Some(bytes) => format!("{}_tile{}", mode_label(mode), bytes),
        None => mode_label(mode).to_string(),
    };
    let stdout_path = output_dir.join(format!("{}_{}.stdout.txt", bench, label));
    let mut run = Command::new(binary);
    apply_sparse_profile_env(&mut run, mode, tile_bytes);
    let output = run
        .output()
        .with_context(|| format!("run benchmark {} {}", bench, label))?;
    fs::write(&stdout_path, &output.stdout)
        .with_context(|| format!("write {}", stdout_path.display()))?;
    let stdout_text = String::from_utf8_lossy(&output.stdout).to_string();
    let (elapsed_seconds, throughput_mlups, effective_glups) =
        parse_gpu_sparse_bench_stdout(&stdout_text);
    let mut row = GpuProfileSweepRow {
        bench: bench.to_string(),
        mode: mode_label(mode).to_string(),
        tile_bytes,
        elapsed_seconds,
        throughput_mlups,
        effective_glups,
        stdout_path,
        nsys_report: None,
        ncu_csv: None,
        skipped_reason: if output.status.success() {
            None
        } else {
            Some(format!("benchmark exited with status {}", output.status))
        },
    };

    if nsys_available {
        let base = output_dir.join(format!("{}_{}", bench, label));
        let mut nsys = Command::new("nsys");
        nsys.arg("profile")
            .arg("--force-overwrite=true")
            .arg("--sample=none")
            .arg("--trace=cuda,nvtx,osrt")
            .arg("-o")
            .arg(&base)
            .arg(binary);
        apply_sparse_profile_env(&mut nsys, mode, tile_bytes);
        let _ = nsys.status();
        row.nsys_report = Some(base.with_extension("nsys-rep"));
    }
    if ncu_available {
        let ncu_csv = output_dir.join(format!("{}_{}_ncu.csv", bench, label));
        let mut ncu = Command::new("ncu");
        ncu.arg("--target-processes")
            .arg("all")
            .arg("--kernel-name-base")
            .arg("demangled")
            .arg("--kernel-name")
            .arg("lbm_step_sparse_aa")
            .arg("--launch-skip")
            .arg("5")
            .arg("--launch-count")
            .arg("3")
            .arg("--section")
            .arg("SpeedOfLight")
            .arg("--section")
            .arg("LaunchStats")
            .arg("--section")
            .arg("Occupancy")
            .arg("--csv")
            .arg("--log-file")
            .arg(&ncu_csv)
            .arg(binary);
        apply_sparse_profile_env(&mut ncu, mode, tile_bytes);
        let _ = ncu.status();
        row.ncu_csv = Some(ncu_csv);
    }
    Ok(row)
}

fn apply_sparse_profile_env(command: &mut Command, mode: &str, tile_bytes: Option<u64>) {
    if mode == "managed" {
        command.env("GOROROBA_SPARSE_MEMORY_MODE", "managed");
    } else if mode == "managed-tiled" {
        command.env("GOROROBA_SPARSE_MEMORY_MODE", "managed-tiled");
    }
    if let Some(bytes) = tile_bytes {
        command.env("GOROROBA_SPARSE_TILE_BYTES", bytes.to_string());
    }
}

fn parse_gpu_sparse_bench_stdout(stdout: &str) -> (Option<f64>, Option<f64>, Option<f64>) {
    let elapsed_seconds = extract_last_float_after(stdout, "Time elapsed:");
    let throughput_mlups = extract_last_float_after(stdout, "Throughput:");
    let effective_glups = extract_last_float_after(stdout, "Effective:");
    (elapsed_seconds, throughput_mlups, effective_glups)
}

fn extract_last_float_after(text: &str, prefix: &str) -> Option<f64> {
    text.lines().find_map(|line| {
        let trimmed = line.trim();
        if !trimmed.starts_with(prefix) {
            return None;
        }
        trimmed
            .split_whitespace()
            .find_map(|token| token.parse::<f64>().ok())
    })
}

fn write_gpu_profile_summary_csv(path: &Path, rows: &[GpuProfileSweepRow]) -> Result<()> {
    let mut writer =
        csv::Writer::from_path(path).with_context(|| format!("create {}", path.display()))?;
    writer.write_record([
        "bench",
        "mode",
        "tile_bytes",
        "elapsed_seconds",
        "throughput_mlups",
        "effective_glups",
        "stdout_path",
        "nsys_report",
        "ncu_csv",
        "skipped_reason",
    ])?;
    for row in rows {
        writer.write_record([
            row.bench.as_str(),
            row.mode.as_str(),
            row.tile_bytes
                .map(|value| value.to_string())
                .unwrap_or_default()
                .as_str(),
            row.elapsed_seconds
                .map(|value| format!("{value:.6}"))
                .unwrap_or_default()
                .as_str(),
            row.throughput_mlups
                .map(|value| format!("{value:.6}"))
                .unwrap_or_default()
                .as_str(),
            row.effective_glups
                .map(|value| format!("{value:.6}"))
                .unwrap_or_default()
                .as_str(),
            row.stdout_path.to_string_lossy().as_ref(),
            row.nsys_report
                .as_ref()
                .map(|path| path.to_string_lossy().into_owned())
                .unwrap_or_default()
                .as_str(),
            row.ncu_csv
                .as_ref()
                .map(|path| path.to_string_lossy().into_owned())
                .unwrap_or_default()
                .as_str(),
            row.skipped_reason.clone().unwrap_or_default().as_str(),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn run_sparse_profile(cli: SparseProfileCli) -> Result<()> {
    let nsys_available = tool_available("nsys", &["--version"]);
    let ncu_available = tool_available("ncu", &["--version"]);

    fs::create_dir_all(&cli.output_dir)
        .with_context(|| format!("create output directory {}", cli.output_dir.display()))?;
    let mut manifest = SparseProfileManifest {
        bench: cli.bench.clone(),
        binary: None,
        generated_at: Local::now().to_rfc3339_opts(SecondsFormat::Secs, true),
        nsys_available,
        ncu_available,
        runs: Vec::new(),
    };

    if !nsys_available {
        let manifest_path = cli.output_dir.join(format!("{}_manifest.json", cli.bench));
        fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?).with_context(|| {
            format!("write sparse profile manifest {}", manifest_path.display())
        })?;
        println!("nsys not available; sparse profiling skipped without blocking the workflow");
        println!("{}", manifest_path.display());
        return Ok(());
    }

    run_status(
        Command::new("cargo")
            .arg("bench")
            .arg("--no-run")
            .arg("-p")
            .arg("lbm_3d_cuda")
            .arg("--bench")
            .arg(&cli.bench),
        "build sparse benchmark",
    )?;

    let binary = locate_sparse_bench_binary(&cli.bench)?;
    manifest.binary = Some(binary.clone());

    for mode in sparse_profile_modes(&cli.mode)? {
        let label = mode_label(mode);
        let base = cli.output_dir.join(format!("{}_{}", cli.bench, label));

        let mut nsys = Command::new("nsys");
        nsys.arg("profile")
            .arg("--force-overwrite=true")
            .arg("--sample=none")
            .arg("--trace=cuda,nvtx,osrt");
        if mode == "managed" {
            nsys.arg("--cuda-um-cpu-page-faults=true")
                .arg("--cuda-um-gpu-page-faults=true");
        }
        nsys.arg("-o").arg(&base).arg(&binary);
        if mode == "managed" {
            nsys.env("GOROROBA_SPARSE_MEMORY_MODE", "managed");
        }
        run_status(&mut nsys, &format!("nsys sparse profile ({label})"))?;

        let nsys_report = base.with_extension("nsys-rep");
        let mut record = SparseProfileRunRecord {
            mode: label.to_string(),
            nsys_report: Some(nsys_report),
            ncu_csv: None,
            skipped_reason: None,
        };

        if cli.run_ncu {
            if ncu_available {
                let ncu_csv = cli
                    .output_dir
                    .join(format!("{}_{}_ncu.csv", cli.bench, label));
                let mut ncu = Command::new("ncu");
                ncu.arg("--target-processes")
                    .arg("all")
                    .arg("--kernel-name-base")
                    .arg("demangled")
                    .arg("--kernel-name")
                    .arg("lbm_step_sparse_aa")
                    .arg("--launch-skip")
                    .arg(cli.ncu_launch_skip.to_string())
                    .arg("--launch-count")
                    .arg(cli.ncu_launch_count.to_string())
                    .arg("--section")
                    .arg("SpeedOfLight")
                    .arg("--section")
                    .arg("LaunchStats")
                    .arg("--section")
                    .arg("Occupancy")
                    .arg("--section")
                    .arg("SchedulerStats")
                    .arg("--section")
                    .arg("InstructionStats")
                    .arg("--section")
                    .arg("MemoryWorkloadAnalysis")
                    .arg("--csv")
                    .arg("--log-file")
                    .arg(&ncu_csv)
                    .arg(&binary);
                if mode == "managed" {
                    ncu.env("GOROROBA_SPARSE_MEMORY_MODE", "managed");
                }
                run_status(&mut ncu, &format!("ncu sparse profile ({label})"))?;
                record.ncu_csv = Some(ncu_csv);
            } else {
                record.skipped_reason =
                    Some("ncu not available; nsys report still generated".to_string());
            }
        }

        manifest.runs.push(record);
    }

    let manifest_path = cli.output_dir.join(format!("{}_manifest.json", cli.bench));
    fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)
        .with_context(|| format!("write sparse profile manifest {}", manifest_path.display()))?;
    println!("{}", manifest_path.display());
    Ok(())
}

fn tool_available(tool: &str, version_args: &[&str]) -> bool {
    let mut command = Command::new(tool);
    for arg in version_args {
        command.arg(arg);
    }
    command
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn run_status(command: &mut Command, context: &str) -> Result<()> {
    let status = command
        .status()
        .with_context(|| format!("spawn {context}"))?;
    if !status.success() {
        bail!("{context} failed with status {status}");
    }
    Ok(())
}

fn locate_sparse_bench_binary(bench: &str) -> Result<PathBuf> {
    let deps_dir = PathBuf::from(".cache/cargo-default-target/release/deps");
    let prefix = format!("{bench}-");
    let mut matches = Vec::new();
    for entry in fs::read_dir(&deps_dir)
        .with_context(|| format!("read benchmark directory {}", deps_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if !name.starts_with(&prefix) || name.ends_with(".d") || name.ends_with(".rlib") {
            continue;
        }
        matches.push(path);
    }
    matches.sort();
    matches.pop().with_context(|| {
        format!(
            "no benchmark binary found for {bench} in {}",
            deps_dir.display()
        )
    })
}

fn sparse_profile_modes(mode: &str) -> Result<Vec<&'static str>> {
    match mode {
        "device" => Ok(vec!["device"]),
        "managed" => Ok(vec!["managed"]),
        "managed-tiled" => Ok(vec!["managed-tiled"]),
        "both" => Ok(vec!["device", "managed"]),
        "all" => Ok(vec!["device", "managed", "managed-tiled"]),
        other => bail!("unsupported sparse-profile mode: {other}"),
    }
}

fn mode_label(mode: &str) -> &'static str {
    match mode {
        "device" => "device",
        "managed" => "managed",
        "managed-tiled" => "managed_tiled",
        _ => "unknown",
    }
}

struct GateAuditConfig {
    output_dir: Option<PathBuf>,
    // When true: only gate-ci-registry runs (no Rust compilation). Fails fast on
    // the first failing step. Useful for registry/governance-only validation
    // where the 9-minute rust-regression compile is unnecessary overhead.
    fast: bool,
}

fn parse_gate_audit_args(args: impl Iterator<Item = String>) -> Result<GateAuditConfig> {
    let mut output_dir = None;
    let mut fast = false;
    let mut iter = args.peekable();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--output-dir" => {
                let Some(value) = iter.next() else {
                    bail!("gate-audit --output-dir requires a value");
                };
                output_dir = Some(PathBuf::from(value));
            }
            "--fast" => {
                fast = true;
            }
            other => bail!("unknown gate-audit argument: {other}"),
        }
    }
    Ok(GateAuditConfig { output_dir, fast })
}

fn run_host_profile(format: &str) -> Result<()> {
    let profile = detect_host_profile();
    match format {
        "shell" => {
            println!("HOST_PHYSICAL_CORES={}", profile.physical_core_count);
            println!(
                "HOST_PHYSICAL_CORE_IDS=\"{}\"",
                join_usize(&profile.physical_core_ids)
            );
            println!("HOST_L3_CACHE_BYTES={}", profile.l3_cache_bytes);
            println!(
                "HOST_L3_SAFE_WORKING_SET_BYTES={}",
                profile.l3_safe_working_set_bytes
            );
            println!("HOST_WORKER_BUDGET={}", profile.worker_budget);
            println!("HOST_CARGO_JOBS={}", profile.cargo_jobs);
            println!("HOST_RAYON_THREADS={}", profile.rayon_threads);
            println!("HOST_RUST_TEST_THREADS={}", profile.rust_test_threads);
            println!("HOST_NEXTEST_TEST_THREADS={}", profile.nextest_test_threads);
            println!("HOST_PYTEST_WORKERS={}", profile.pytest_workers);
        }
        "json" => {
            println!("{}", serde_json::to_string_pretty(&profile)?);
        }
        "budget" => {
            println!("{}", profile.worker_budget);
        }
        other => bail!("unsupported host-profile format: {other}"),
    }
    Ok(())
}

fn run_db_docs(check_only: bool) -> Result<()> {
    let repo_root = repo_root()?;
    let outputs = generate_schema_outputs(&repo_root)?;
    write_or_check(
        &repo_root.join("db/schema.sql"),
        &outputs.schema_sql,
        check_only,
    )?;
    write_or_check(
        &repo_root.join("docs/db/schema.json"),
        &outputs.schema_json,
        check_only,
    )?;
    write_or_check(
        &repo_root.join("crates/data_core/src/registry_mirrors/db_catalog.rs"),
        &outputs.catalog_rs,
        check_only,
    )?;
    if check_only {
        println!("db-docs OK: generated schema artifacts match committed files");
    } else {
        println!(
            "db-docs OK: regenerated db/schema.sql docs/db/schema.json crates/data_core/src/registry_mirrors/db_catalog.rs"
        );
    }
    Ok(())
}

fn run_gate_audit(cfg: GateAuditConfig) -> Result<()> {
    let repo_root = repo_root()?;
    let generated_at = Local::now();
    let timestamp = generated_at.format("%Y-%m-%d/%H%M%S").to_string();
    let output_dir = match cfg.output_dir {
        Some(path) if path.is_absolute() => path,
        Some(path) => repo_root.join(path),
        None => repo_root.join("reports").join("gates").join(timestamp),
    };
    fs::create_dir_all(&output_dir).with_context(|| {
        format!(
            "create gate-audit output directory {}",
            output_dir.display()
        )
    })?;

    // Fast mode: registry-only, fails on first step failure.
    // WHY: gate-ci-rust runs rust-regression (full workspace compile + test run,
    // ~9 min). For governance/registry-only changes, that compile overhead is
    // pure waste. --fast lets developers iterate on TOML edits in ~2 min instead
    // of ~12 min, without sacrificing correctness guarantees for registry changes.
    //
    // Full mode: all three steps. Uses cargo check --workspace --tests instead of
    // cargo nextest list for the workspace-compile-check step. cargo check skips
    // LLVM codegen (~3-5x faster than nextest list on a cold cache) while still
    // verifying that every test-gated compilation unit typechecks. Since
    // rust-regression already ran all tests in gate-ci-rust, the nextest list
    // was redundant for correctness; cargo check preserves the compile-check
    // intent at lower cost.
    let commands: Vec<(&str, Vec<String>)> = if cfg.fast {
        vec![(
            "gate-ci-registry",
            vec!["make".to_string(), "gate-ci-registry".to_string()],
        )]
    } else {
        vec![
            (
                "gate-ci-registry",
                vec!["make".to_string(), "gate-ci-registry".to_string()],
            ),
            (
                "gate-ci-rust",
                vec!["make".to_string(), "gate-ci-rust".to_string()],
            ),
            (
                // cargo check skips LLVM codegen; verifies test-gated code compiles
                // without paying the nextest list compilation + linking tax.
                "workspace-check",
                vec![
                    "cargo".to_string(),
                    "check".to_string(),
                    "--workspace".to_string(),
                    "--tests".to_string(),
                ],
            ),
        ]
    };

    let cargo_home = repo_root.join(".cache").join("cargo-home");
    let cargo_target_dir = repo_root.join(".cache").join("gate-target");

    let mut summary_lines = vec![
        format!(
            "# Gate Audit ({})",
            generated_at.to_rfc3339_opts(SecondsFormat::Secs, false)
        ),
        String::new(),
        format!(
            "Output directory: `{}`",
            repo_relative(&output_dir, &repo_root)
        ),
        String::new(),
        "| Step | Exit Code | Log |".to_string(),
        "| --- | ---: | --- |".to_string(),
    ];

    let mut failures = 0usize;
    let mut step_rows = Vec::<GateAuditStepRecord>::new();

    for (name, command) in commands {
        let log_path = output_dir.join(format!("{name}.log"));
        let output = Command::new(&command[0])
            .args(&command[1..])
            .current_dir(&repo_root)
            .env("CARGO_HOME", &cargo_home)
            .env("CARGO_TARGET_DIR", &cargo_target_dir)
            .output()
            .with_context(|| format!("run {}", format_command(&command)))?;
        let exit_code = output.status.code().unwrap_or(1);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined_output = format!("{stdout}{stderr}");

        let log_text = format!(
            "# Step: {name}\n# Command: {}\n# Exit Code: {exit_code}\n\n{combined_output}",
            format_command(&command)
        );
        fs::write(&log_path, log_text)
            .with_context(|| format!("write gate-audit step log {}", log_path.display()))?;

        let log_rel = repo_relative(&log_path, &repo_root);
        summary_lines.push(format!("| `{name}` | `{exit_code}` | `{log_rel}` |"));
        step_rows.push(GateAuditStepRecord {
            name: name.to_string(),
            exit_code,
            log: log_rel,
        });

        summary_lines.push(String::new());
        summary_lines.push(format!("## {name}"));
        summary_lines.push(String::new());
        summary_lines.push(format!("Exit code: `{exit_code}`"));
        summary_lines.push(String::new());
        summary_lines.push("```text".to_string());
        summary_lines.extend(render_tail_block(
            &combined_output,
            GATE_AUDIT_TAIL_LINE_COUNT,
        ));
        summary_lines.push("```".to_string());

        if exit_code != 0 {
            failures += 1;
            // Fail fast: no point waiting for Rust compilation if registry checks
            // already failed. The developer needs to fix the registry first.
            if cfg.fast || name == "gate-ci-registry" {
                break;
            }
        }
    }

    summary_lines.push(String::new());
    summary_lines.push(if failures == 0 {
        "Gate audit passed.".to_string()
    } else {
        format!("Gate audit failed in {failures} step(s).")
    });
    summary_lines.push(String::new());
    summary_lines.push("Review the per-step logs for full output.".to_string());
    summary_lines.push(String::new());

    let summary_path = output_dir.join("summary.md");
    let summary_text = summary_lines.join("\n");
    fs::write(&summary_path, format!("{summary_text}\n"))
        .with_context(|| format!("write gate-audit summary {}", summary_path.display()))?;

    let reports_gates_root = repo_root.join("reports").join("gates");
    fs::create_dir_all(&reports_gates_root).with_context(|| {
        format!(
            "create reports/gates directory {}",
            reports_gates_root.display()
        )
    })?;
    let latest_summary_path = reports_gates_root.join("LATEST.md");
    let latest_manifest_path = reports_gates_root.join("latest.json");
    fs::write(&latest_summary_path, format!("{summary_text}\n")).with_context(|| {
        format!(
            "write latest gate-audit summary {}",
            latest_summary_path.display()
        )
    })?;
    let latest_manifest = serde_json::json!({
        "generated_at": Local::now().to_rfc3339_opts(SecondsFormat::Secs, false),
        "output_dir": repo_relative(&output_dir, &repo_root),
        "summary": repo_relative(&summary_path, &repo_root),
        "failure_count": failures,
        "steps": step_rows,
    });
    fs::write(
        &latest_manifest_path,
        format!("{}\n", serde_json::to_string_pretty(&latest_manifest)?),
    )
    .with_context(|| {
        format!(
            "write gate-audit manifest {}",
            latest_manifest_path.display()
        )
    })?;

    println!("Wrote: {}", repo_relative(&summary_path, &repo_root));
    if failures != 0 {
        bail!("gate-audit failed in {failures} step(s)");
    }
    Ok(())
}

/// PH-5.A: structured audit-deep composite lane.
///
/// # Purpose and call sites
///
/// Wraps the Makefile `audit-deep` chain (rust-clippy, cargo-deny-check,
/// dep-audit, cpd-audit) and emits per-step exit codes, log files,
/// and a Markdown summary in the same format as `run_gate_audit`. The
/// Makefile target still exists as the user-facing entry point; this
/// xtask variant adds structured archival for CI and tranche-acceptance
/// evidence.
///
/// Called from `cargo run -p xtask -- audit-deep`. The Makefile target
/// `make audit-deep-structured` will be added in a follow-up to give
/// developers a familiar Makefile entry point.
///
/// # What this owns vs delegates
///
/// Owns: invocation order, per-step log capture, Markdown summary
/// rendering, exit-code aggregation, structured TOML record assembly
/// (so downstream consumers can index runs by date / step / pass-fail).
///
/// Delegates to the existing Makefile targets:
/// - `make rust-clippy` (workspace clippy with -D warnings)
/// - `make cargo-deny-check` (license + advisory + sources policy)
/// - `make dep-audit` (cargo-audit advisory scan)
/// - `make cpd-audit` (PMD copy-paste detector)
///
/// # Why skip the same things audit-deep Makefile skips
///
/// The Makefile target skips rust-semver-check (fwht path-dep
/// resolution issue) and docs-freshness (bracket notation in math
/// docs triggers broken-intra-doc-links). The xtask wrapper preserves
/// those skips so the behavior is identical; the structured output
/// just adds reporting on top.
///
/// # Cross-references
///
/// - Sibling: [`run_gate_audit`] (this fn mirrors its reporting surface).
/// - Makefile: `audit-deep` target with full rationale comments.
/// - PH-5 roadmap: `plans/repo_debt_roadmap_2026_04_11.toml`.
fn run_audit_deep(cfg: GateAuditConfig) -> Result<()> {
    let repo_root = repo_root()?;
    let generated_at = Local::now();
    let timestamp = generated_at.format("%Y-%m-%d/%H%M%S").to_string();
    let output_dir = match cfg.output_dir {
        Some(path) if path.is_absolute() => path,
        Some(path) => repo_root.join(path),
        None => repo_root.join("reports").join("audit-deep").join(timestamp),
    };
    fs::create_dir_all(&output_dir).with_context(|| {
        format!(
            "create audit-deep output directory {}",
            output_dir.display()
        )
    })?;

    // The four Makefile steps, in the same order as `audit-deep` runs
    // them. We do NOT add semver-check or docs-freshness here; see the
    // Makefile rationale comments.
    let commands: Vec<(&str, Vec<String>)> = vec![
        (
            "rust-clippy",
            vec!["make".to_string(), "rust-clippy".to_string()],
        ),
        (
            "cargo-deny-check",
            vec!["make".to_string(), "cargo-deny-check".to_string()],
        ),
        ("dep-audit", vec!["make".to_string(), "dep-audit".to_string()]),
        ("cpd-audit", vec!["make".to_string(), "cpd-audit".to_string()]),
    ];

    let cargo_home = repo_root.join(".cache").join("cargo-home");
    let cargo_target_dir = repo_root.join(".cache").join("gate-target");

    let mut summary_lines = vec![
        format!(
            "# Audit Deep ({})",
            generated_at.to_rfc3339_opts(SecondsFormat::Secs, false)
        ),
        String::new(),
        format!(
            "Output directory: `{}`",
            repo_relative(&output_dir, &repo_root)
        ),
        String::new(),
        "| Step | Exit Code | Log |".to_string(),
        "| --- | ---: | --- |".to_string(),
    ];

    let mut failures = 0usize;
    let mut step_rows = Vec::<GateAuditStepRecord>::new();

    for (name, command) in commands {
        let log_path = output_dir.join(format!("{name}.log"));
        let output = Command::new(&command[0])
            .args(&command[1..])
            .current_dir(&repo_root)
            .env("CARGO_HOME", &cargo_home)
            .env("CARGO_TARGET_DIR", &cargo_target_dir)
            .output()
            .with_context(|| format!("run {}", format_command(&command)))?;
        let exit_code = output.status.code().unwrap_or(1);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined_output = format!("{stdout}{stderr}");

        let log_text = format!(
            "# Step: {name}\n# Command: {}\n# Exit Code: {exit_code}\n\n{combined_output}",
            format_command(&command)
        );
        fs::write(&log_path, log_text)
            .with_context(|| format!("write audit-deep step log {}", log_path.display()))?;

        let log_rel = repo_relative(&log_path, &repo_root);
        summary_lines.push(format!("| `{name}` | `{exit_code}` | `{log_rel}` |"));
        step_rows.push(GateAuditStepRecord {
            name: name.to_string(),
            exit_code,
            log: log_rel,
        });

        if exit_code != 0 {
            failures += 1;
        }
    }

    let summary_path = output_dir.join("SUMMARY.md");
    fs::write(&summary_path, summary_lines.join("\n")).with_context(|| {
        format!(
            "write audit-deep summary {}",
            summary_path.display()
        )
    })?;

    // Structured TOML record (parallel to the SUMMARY.md) so downstream
    // tooling can index runs without parsing Markdown.
    let toml_path = output_dir.join("audit_deep.toml");
    let record = serde_json::json!({
        "generated_at": generated_at.to_rfc3339_opts(SecondsFormat::Secs, false),
        "failures": failures,
        "steps": step_rows,
    });
    fs::write(&toml_path, toml::to_string_pretty(&record)?).with_context(|| {
        format!("write audit-deep toml {}", toml_path.display())
    })?;

    println!("Wrote: {}", repo_relative(&summary_path, &repo_root));
    if failures != 0 {
        bail!("audit-deep failed in {failures} step(s)");
    }
    Ok(())
}

/// PH-5.B: structured replacement for the 54-line `registry-export-markdown`
/// Makefile heredoc.
///
/// # Purpose and call sites
///
/// The Makefile target `registry-export-markdown` historically invoked
/// `registry-emit Xmirror --output Y` 23 times sequentially with a
/// hand-maintained list of (mirror_kind, output_path) pairs. Adding a
/// new mirror meant editing the Makefile in two places (the cargo build
/// list and the invocation list), and any failure mid-chain produced a
/// confusing partial-state.
///
/// This xtask command owns the list as a Rust array, builds the
/// `registry-emit` binary once, then loops with proper error
/// propagation. Adding a new mirror is a single tuple in the
/// `MIRRORS` array below.
///
/// Called from `cargo run -p xtask -- registry-emit-all-mirrors`. The
/// Makefile target `registry-export-markdown` should now delegate
/// to this xtask command for the per-mirror loop (the
/// `registry-refresh registry-build` prerequisites still live in
/// the Makefile).
///
/// # Why an array, not a TOML manifest?
///
/// The list rarely changes (~23 entries) and adding one is editing a
/// single tuple. A TOML manifest would add deserialization overhead
/// and one more file to keep in sync. The array is the source of
/// truth; the Makefile no longer encodes any of it.
///
/// # What this owns vs delegates
///
/// Owns: the (mirror_kind, output_path) list, the build-then-loop
/// structure, per-step error reporting.
///
/// Delegates to `registry-emit`: actual mirror generation (existing
/// binary; this command just invokes it).
///
/// # Cross-references
///
/// - Memory: `feedback_install_source_priority.md` (Makefile -> xtask
///   migration policy).
/// - Sibling: [`run_audit_deep`] (same xtask reporting pattern).
/// - Replaced shell: Makefile line 1495 `registry-export-markdown`.
fn run_registry_emit_all_mirrors() -> Result<()> {
    let repo_root = repo_root()?;
    let cargo_home = repo_root.join(".cache").join("cargo-home");
    let cargo_target_dir = repo_root.join(".cache").join("gate-target");

    // The 23 mirror kinds. Format: (registry-emit subcommand, output
    // path relative to crates/data_core/src/registry_mirrors/).
    //
    // # Why this list is hand-maintained vs derived
    //
    // The registry-emit binary's own enum is the structural truth, but
    // not every variant has a single canonical output path -- e.g.
    // `requirements-legacy` emits multiple markdown files, not one
    // Rust mirror. Keeping the list local to this xtask lets us be
    // explicit about which mirrors fan out to a Rust file vs which
    // emit markdown elsewhere.
    const MIRRORS: &[(&str, &str)] = &[
        ("insights-mirror", "insights_registry_mirror.rs"),
        ("claims-mirror", "claims_registry_mirror.rs"),
        ("bibliography-mirror", "bibliography_registry_mirror.rs"),
        ("experiments-mirror", "experiments_registry_mirror.rs"),
        ("theorems-mirror", "theorems_registry_mirror.rs"),
        ("roadmap-mirror", "roadmap_registry_mirror.rs"),
        ("todo-mirror", "todo_registry_mirror.rs"),
        ("next-actions-mirror", "next_actions_registry_mirror.rs"),
        ("navigator-mirror", "navigator_registry_mirror.rs"),
        ("entrypoint-docs-mirror", "entrypoint_docs_registry_mirror.rs"),
        ("requirements-mirror", "requirements_registry_mirror.rs"),
        (
            "knowledge-migration-plan-mirror",
            "knowledge_migration_plan_registry_mirror.rs",
        ),
        (
            "markdown-governance-mirror",
            "markdown_governance_registry_mirror.rs",
        ),
        ("claims-tasks-mirror", "claims_tasks_registry_mirror.rs"),
        ("claims-domains-mirror", "claims_domains_registry_mirror.rs"),
        ("claim-tickets-mirror", "claim_tickets_registry_mirror.rs"),
        ("external-sources-mirror", "external_sources_registry_mirror.rs"),
        ("book-docs-mirror", "book_docs_registry_mirror.rs"),
        (
            "data-artifact-narratives-mirror",
            "data_artifact_narratives_registry_mirror.rs",
        ),
        (
            "reports-narratives-mirror",
            "reports_narratives_registry_mirror.rs",
        ),
        ("docs-convos-mirror", "docs_convos_registry_mirror.rs"),
        (
            "docs-root-narratives-mirror",
            "docs_root_narratives_registry_mirror.rs",
        ),
        (
            "research-narratives-mirror",
            "research_narratives_registry_mirror.rs",
        ),
    ];

    // Step 1: build registry-emit + markdown-registry once. Use the
    // release-gate profile (matches the Makefile heredoc; both binaries
    // are stable so we don't need full release LTO).
    println!("Building registry-emit + markdown-registry ...");
    let build_status = Command::new("cargo")
        .args([
            "build",
            "--profile",
            "release-gate",
            "-p",
            "gororoba_cli_data",
            "--bin",
            "registry-emit",
            "--bin",
            "markdown-registry",
        ])
        .current_dir(&repo_root)
        .env("CARGO_HOME", &cargo_home)
        .env("CARGO_TARGET_DIR", &cargo_target_dir)
        .status()
        .context("invoke cargo build for registry-emit + markdown-registry")?;
    if !build_status.success() {
        bail!(
            "build of registry-emit + markdown-registry failed: {}",
            build_status
        );
    }

    let registry_emit = cargo_target_dir.join("release-gate").join("registry-emit");
    let mirror_dir = repo_root
        .join("crates")
        .join("data_core")
        .join("src")
        .join("registry_mirrors");

    // Step 2: invoke registry-emit Xmirror --output Y for each entry.
    // Fail fast on the first error -- a stale mirror is worse than a
    // partial emission because the dependent crate's `include!` macros
    // would then read inconsistent state.
    let mut applied = 0usize;
    for (kind, output_name) in MIRRORS {
        let output_path = mirror_dir.join(output_name);
        print!("  emit {} -> {} ... ", kind, output_name);
        let status = Command::new(&registry_emit)
            .args([kind, "--output"])
            .arg(&output_path)
            .current_dir(&repo_root)
            .env("CARGO_HOME", &cargo_home)
            .env("CARGO_TARGET_DIR", &cargo_target_dir)
            .status()
            .with_context(|| format!("invoke registry-emit {}", kind))?;
        if !status.success() {
            bail!(
                "registry-emit {} failed (exit {:?}); aborting fan-out",
                kind,
                status.code()
            );
        }
        println!("ok");
        applied += 1;
    }

    println!(
        "registry-emit-all-mirrors: {} / {} mirrors emitted.",
        applied,
        MIRRORS.len()
    );
    Ok(())
}

fn run_local_nextest_plan(cli: LocalNextestCli) -> Result<()> {
    let exit_code = local_nextest_plan(cli)?;
    if exit_code != 0 {
        bail!("local-nextest-plan failed with exit code {exit_code}");
    }
    Ok(())
}

fn local_nextest_plan(cli: LocalNextestCli) -> Result<i32> {
    let root = repo_root()?;
    let mut timing = TimingRecorder::new(cli.timing_json_out);
    let mut package_plans = BTreeMap::<String, PackagePlan>::new();
    let mut lib_packages = Vec::<String>::new();
    let mut test_packages = Vec::<String>::new();

    for package in &cli.packages {
        let Some(plan) = package_plan(&root, package)? else {
            let reason = "no inline lib tests and no integration tests";
            println!("[local-nextest] skip {package}: {reason}");
            timing.record_skip(package, reason)?;
            continue;
        };
        if plan.has_lib_tests {
            lib_packages.push(package.clone());
        }
        if !plan.tests.is_empty() {
            test_packages.push(package.clone());
        }
        package_plans.insert(package.clone(), plan);
    }

    let mut exit_code = 0;
    if !lib_packages.is_empty() {
        let command = build_local_nextest_command(
            &lib_packages,
            true,
            false,
            &cli.build_jobs,
            &cli.test_threads,
            &cli.filterset,
        );
        let targets = lib_packages
            .iter()
            .map(|package| (package.clone(), serde_json::json!(["lib"])))
            .collect::<serde_json::Map<String, serde_json::Value>>();
        exit_code = run_local_nextest_command(
            &root,
            &lib_packages,
            &command,
            &serde_json::Value::Object(targets),
            &mut timing,
        )?;
        if exit_code != 0 {
            timing.record_summary(exit_code)?;
            return Ok(exit_code);
        }
    }
    let run_integration_tests = matches!(cli.kinds.as_str(), "all" | "tests");
    if !run_integration_tests && !test_packages.is_empty() {
        println!(
            "[local-nextest] skip integration-test phase for {} packages (kinds={}). Run with --kinds all to include.",
            test_packages.len(),
            cli.kinds
        );
    }
    if run_integration_tests && !test_packages.is_empty() {
        let command = build_local_nextest_command(
            &test_packages,
            false,
            true,
            &cli.build_jobs,
            &cli.test_threads,
            &cli.filterset,
        );
        let targets = test_packages
            .iter()
            .map(|package| {
                let selected = package_plans
                    .get(package)
                    .map(|plan| {
                        if plan.tests.is_empty() {
                            vec!["tests".to_string()]
                        } else {
                            plan.tests
                                .iter()
                                .map(|name| format!("test:{name}"))
                                .collect::<Vec<_>>()
                        }
                    })
                    .unwrap_or_else(|| vec!["tests".to_string()]);
                (package.clone(), serde_json::json!(selected))
            })
            .collect::<serde_json::Map<String, serde_json::Value>>();
        exit_code = run_local_nextest_command(
            &root,
            &test_packages,
            &command,
            &serde_json::Value::Object(targets),
            &mut timing,
        )?;
        if exit_code != 0 {
            timing.record_summary(exit_code)?;
            return Ok(exit_code);
        }
    }

    timing.record_summary(exit_code)?;
    Ok(exit_code)
}

fn package_root(root: &Path, package: &str) -> PathBuf {
    root.join("crates").join(package)
}

fn has_library(root: &Path, package: &str) -> bool {
    package_root(root, package)
        .join("src")
        .join("lib.rs")
        .is_file()
}

fn has_inline_tests(root: &Path, package: &str) -> Result<bool> {
    let src_root = package_root(root, package).join("src");
    if !src_root.is_dir() {
        return Ok(false);
    }
    let bin_root = src_root.join("bin");
    for entry in WalkDir::new(&src_root)
        .into_iter()
        .filter_map(std::result::Result::ok)
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
            continue;
        }
        if path.starts_with(&bin_root) {
            continue;
        }
        let text = fs::read_to_string(path)
            .with_context(|| format!("read Rust source {}", path.display()))?;
        for raw_line in text.lines() {
            let line = raw_line.trim();
            if line.starts_with("//") || line.starts_with("/*") || line.starts_with('*') {
                continue;
            }
            if INLINE_TEST_MARKERS
                .iter()
                .any(|marker| line.starts_with(marker))
            {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn integration_tests(root: &Path, package: &str) -> Result<Vec<String>> {
    let tests_dir = package_root(root, package).join("tests");
    if !tests_dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut tests = Vec::new();
    for entry in fs::read_dir(&tests_dir)
        .with_context(|| format!("read tests directory {}", tests_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_file()
            && path.extension().and_then(|ext| ext.to_str()) == Some("rs")
            && let Some(stem) = path.file_stem().and_then(|stem| stem.to_str())
        {
            tests.push(stem.to_string());
        }
    }
    tests.sort();
    Ok(tests)
}

fn package_plan(root: &Path, package: &str) -> Result<Option<PackagePlan>> {
    let has_lib = has_library(root, package);
    let has_lib_tests = has_lib && has_inline_tests(root, package)?;
    let tests = integration_tests(root, package)?;
    if !has_lib_tests && tests.is_empty() {
        return Ok(None);
    }
    Ok(Some(PackagePlan {
        has_lib_tests,
        tests,
    }))
}

fn build_local_nextest_command(
    packages: &[String],
    run_lib: bool,
    run_tests: bool,
    build_jobs: &str,
    test_threads: &str,
    filterset: &str,
) -> Vec<String> {
    let mut command = vec![
        "cargo".to_string(),
        "nextest".to_string(),
        "run".to_string(),
        "--build-jobs".to_string(),
        build_jobs.to_string(),
        "--test-threads".to_string(),
        test_threads.to_string(),
    ];
    if run_lib {
        command.push("--lib".to_string());
    }
    if run_tests {
        command.push("--tests".to_string());
    }
    for package in packages {
        command.push("-p".to_string());
        command.push(package.clone());
    }
    if !filterset.is_empty() {
        command.push("-E".to_string());
        command.push(filterset.to_string());
    }
    command
}

fn run_local_nextest_command(
    root: &Path,
    packages: &[String],
    command: &[String],
    targets: &serde_json::Value,
    timing: &mut TimingRecorder,
) -> Result<i32> {
    let targets_object = targets
        .as_object()
        .expect("selected targets must be a JSON object");
    for package in packages {
        let joined = targets_object
            .get(package)
            .and_then(|value| value.as_array())
            .map(|entries| {
                entries
                    .iter()
                    .filter_map(|entry| entry.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_else(|| "(none)".to_string());
        println!("[local-nextest] run {package}: {joined}");
    }
    let start = Instant::now();
    let status = Command::new(&command[0])
        .args(&command[1..])
        .current_dir(root)
        .status()
        .with_context(|| format!("run {}", command.join(" ")))?;
    let code = status.code().unwrap_or(1);
    timing.record_run(
        packages,
        targets,
        command,
        code,
        start.elapsed().as_secs_f64(),
    )?;
    Ok(code)
}

struct GeneratedSchemaOutputs {
    schema_sql: String,
    schema_json: String,
    catalog_rs: String,
}

fn generate_schema_outputs(repo_root: &Path) -> Result<GeneratedSchemaOutputs> {
    let temp_dir = tempdir().context("create temporary directory for schema docs")?;
    let db_path = temp_dir.path().join("schema.sqlite3");
    let _store = ProvenanceStore::open(&db_path)?;
    drop(_store);
    let conn = Connection::open(&db_path).context("open temporary schema sqlite database")?;
    conn.pragma_update(None, "foreign_keys", "ON")
        .context("enable foreign_keys for schema introspection")?;

    let schema_sql = render_schema_sql(&conn)?;
    let snapshot = introspect_schema(&conn)?;
    let schema_json = serde_json::to_string_pretty(&snapshot)?;
    let catalog_rs = render_catalog_rustdoc(&snapshot);
    let _ = repo_root;
    Ok(GeneratedSchemaOutputs {
        schema_sql,
        schema_json,
        catalog_rs,
    })
}

fn render_schema_sql(conn: &Connection) -> Result<String> {
    let mut stmt = conn.prepare(
        "SELECT sql
         FROM sqlite_schema
         WHERE sql IS NOT NULL
           AND name NOT LIKE 'sqlite_%'
           AND name NOT IN (
               SELECT name
               FROM pragma_table_list
               WHERE schema = 'main' AND type = 'shadow'
           )
         ORDER BY tbl_name, type DESC, name",
    )?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
    let mut sql_blocks = Vec::new();
    for row in rows {
        let mut sql = row?;
        if !sql.trim_end().ends_with(';') {
            sql.push(';');
        }
        sql_blocks.push(sql);
    }
    let mut out = String::new();
    out.push_str("-- GENERATED FILE. DO NOT EDIT.\n");
    out.push_str("-- Canonical source: db/migrations/*.sql\n");
    out.push_str("-- Regenerate with: cargo run -p xtask -- db-docs\n\n");
    out.push_str(&sql_blocks.join("\n\n"));
    out.push('\n');
    Ok(out)
}

fn introspect_schema(conn: &Connection) -> Result<SchemaSnapshot> {
    let mut stmt = conn.prepare(
        "SELECT schema, name, type, ncol, wr, strict
         FROM pragma_table_list
         WHERE schema = 'main'
           AND name NOT LIKE 'sqlite_%'
           AND type != 'shadow'
         ORDER BY type, name",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, i64>(3)?,
            row.get::<_, i64>(4)?,
            row.get::<_, i64>(5)?,
        ))
    })?;
    let mut objects = Vec::new();
    for row in rows {
        let (schema, name, object_type, ncol, wr, strict) = row?;
        objects.push(SchemaObject {
            schema,
            name: name.clone(),
            object_type,
            column_count: ncol,
            without_rowid: wr != 0,
            strict: strict != 0,
            columns: schema_columns(conn, &name)?,
            foreign_keys: schema_foreign_keys(conn, &name)?,
            indexes: schema_indexes(conn, &name)?,
        });
    }
    Ok(SchemaSnapshot {
        generated_from: "db/migrations/*.sql via SQLite pragma introspection".to_string(),
        object_count: objects.len(),
        objects,
    })
}

fn schema_columns(conn: &Connection, table_name: &str) -> Result<Vec<SchemaColumn>> {
    let sql = format!(
        "SELECT cid, name, type, \"notnull\", dflt_value, pk, hidden
         FROM pragma_table_xinfo('{}')
         ORDER BY cid",
        table_name.replace('\'', "''")
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], |row| {
        Ok(SchemaColumn {
            cid: row.get(0)?,
            name: row.get(1)?,
            declared_type: row.get::<_, Option<String>>(2)?.unwrap_or_default(),
            not_null: row.get::<_, i64>(3)? != 0,
            default_value: row.get(4)?,
            primary_key_position: row.get(5)?,
            hidden: row.get(6)?,
        })
    })?;
    let mut columns = Vec::new();
    for row in rows {
        columns.push(row?);
    }
    Ok(columns)
}

fn schema_foreign_keys(conn: &Connection, table_name: &str) -> Result<Vec<SchemaForeignKey>> {
    let sql = format!(
        "SELECT id, seq, \"table\", \"from\", \"to\", on_update, on_delete, match
         FROM pragma_foreign_key_list('{}')
         ORDER BY id, seq",
        table_name.replace('\'', "''")
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], |row| {
        Ok(SchemaForeignKey {
            id: row.get(0)?,
            seq: row.get(1)?,
            ref_table: row.get(2)?,
            from_column: row.get(3)?,
            to_column: row.get(4)?,
            on_update: row.get(5)?,
            on_delete: row.get(6)?,
            match_kind: row.get(7)?,
        })
    })?;
    let mut keys = Vec::new();
    for row in rows {
        keys.push(row?);
    }
    Ok(keys)
}

fn schema_indexes(conn: &Connection, table_name: &str) -> Result<Vec<SchemaIndex>> {
    let sql = format!(
        "SELECT seq, name, \"unique\", origin, partial
         FROM pragma_index_list('{}')
         ORDER BY seq",
        table_name.replace('\'', "''")
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, i64>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, i64>(4)?,
        ))
    })?;
    let mut indexes = Vec::new();
    for row in rows {
        let (seq, name, unique, origin, partial) = row?;
        indexes.push(SchemaIndex {
            seq,
            columns: schema_index_columns(conn, &name)?,
            name,
            unique: unique != 0,
            origin,
            partial: partial != 0,
        });
    }
    Ok(indexes)
}

fn schema_index_columns(conn: &Connection, index_name: &str) -> Result<Vec<SchemaIndexColumn>> {
    let sql = format!(
        "SELECT seqno, cid, name, desc, coll, key
         FROM pragma_index_xinfo('{}')
         ORDER BY seqno",
        index_name.replace('\'', "''")
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], |row| {
        Ok(SchemaIndexColumn {
            seqno: row.get(0)?,
            cid: row.get(1)?,
            name: row.get(2)?,
            descending: row.get::<_, i64>(3)? != 0,
            collation: row.get::<_, Option<String>>(4)?.unwrap_or_default(),
            key: row.get::<_, i64>(5)? != 0,
        })
    })?;
    let mut columns = Vec::new();
    for row in rows {
        columns.push(row?);
    }
    Ok(columns)
}

fn render_catalog_rustdoc(snapshot: &SchemaSnapshot) -> String {
    let raw = render_catalog_markdown_raw(snapshot);
    let mut rustdoc = String::new();
    for line in raw.lines() {
        if line.is_empty() {
            rustdoc.push_str("//!\n");
        } else {
            rustdoc.push_str("//! ");
            rustdoc.push_str(line);
            rustdoc.push('\n');
        }
    }
    rustdoc
}

fn render_catalog_markdown_raw(snapshot: &SchemaSnapshot) -> String {
    let mut out = String::new();
    out.push_str("<!-- AUTO-GENERATED: DO NOT EDIT -->\n");
    out.push_str("<!-- Source of truth: db/schema.sql -->\n");
    out.push_str(
        "<!-- Generated from: db/migrations/*.sql via cargo run -p xtask -- db-docs -->\n\n",
    );
    out.push_str("# Database Catalog\n\n");
    out.push_str("Generated file. Do not edit.\n\n");
    out.push_str("- Source of truth: `db/schema.sql`\n");
    out.push_str("- Canonical migrations: `db/migrations/*.sql`\n");
    out.push_str("- Regenerate with: `cargo run -p xtask -- db-docs`\n");
    out.push_str(&format!("- Objects: `{}`\n\n", snapshot.object_count));

    for object in &snapshot.objects {
        out.push_str(&format!(
            "## `{}` ({})\n\n",
            object.name, object.object_type
        ));
        out.push_str(&format!(
            "- Strict: `{}`\n- Without rowid: `{}`\n- Declared columns: `{}`\n\n",
            object.strict, object.without_rowid, object.column_count
        ));
        out.push_str("| cid | name | type | not null | default | pk | hidden |\n");
        out.push_str("| --- | --- | --- | --- | --- | --- | --- |\n");
        for column in &object.columns {
            let default_value = column.default_value.as_deref().unwrap_or("");
            out.push_str(&format!(
                "| {} | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |\n",
                column.cid,
                column.name,
                column.declared_type,
                column.not_null,
                default_value.replace('|', "\\|"),
                column.primary_key_position,
                column.hidden
            ));
        }
        out.push('\n');
        if !object.foreign_keys.is_empty() {
            out.push_str("Foreign keys:\n\n");
            out.push_str("| id | seq | table | from | to | on update | on delete | match |\n");
            out.push_str("| --- | --- | --- | --- | --- | --- | --- | --- |\n");
            for fk in &object.foreign_keys {
                out.push_str(&format!(
                    "| {} | {} | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |\n",
                    fk.id,
                    fk.seq,
                    fk.ref_table,
                    fk.from_column,
                    fk.to_column,
                    fk.on_update,
                    fk.on_delete,
                    fk.match_kind
                ));
            }
            out.push('\n');
        }
        if !object.indexes.is_empty() {
            out.push_str("Indexes:\n\n");
            out.push_str("| seq | name | unique | origin | partial | columns |\n");
            out.push_str("| --- | --- | --- | --- | --- | --- |\n");
            for index in &object.indexes {
                let cols = index
                    .columns
                    .iter()
                    .map(|col| {
                        let label = col.name.clone().unwrap_or_else(|| "<expr>".to_string());
                        if col.descending {
                            format!("{label} desc")
                        } else {
                            label
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                out.push_str(&format!(
                    "| {} | `{}` | `{}` | `{}` | `{}` | `{}` |\n",
                    index.seq, index.name, index.unique, index.origin, index.partial, cols
                ));
            }
            out.push('\n');
        }
    }

    out
}

fn detect_host_profile() -> HostProfile {
    let topo = HardwareTopology::current();
    let physical_core_count = topo.physical_core_ids.len().max(1);
    HostProfile {
        physical_core_ids: topo.physical_core_ids.clone(),
        physical_core_count,
        l3_cache_bytes: topo.l3_cache_bytes,
        l3_safe_working_set_bytes: topo.l3_safe_working_set_bytes,
        worker_budget: physical_core_count,
        cargo_jobs: physical_core_count,
        rayon_threads: physical_core_count,
        rust_test_threads: physical_core_count,
        nextest_test_threads: physical_core_count,
        pytest_workers: physical_core_count,
    }
}

fn join_usize(items: &[usize]) -> String {
    items
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn format_command(parts: &[String]) -> String {
    parts.join(" ")
}

fn render_tail_block(text: &str, tail_line_count: usize) -> Vec<String> {
    let lines = text.lines().collect::<Vec<_>>();
    if lines.is_empty() {
        return vec!["(no log output)".to_string()];
    }
    let start = lines.len().saturating_sub(tail_line_count);
    let mut tail = Vec::new();
    if start > 0 {
        tail.push(format!("... ({} earlier line(s) omitted)", start));
    }
    tail.extend(lines[start..].iter().map(|line| (*line).to_string()));
    tail
}

fn repo_relative(path: &Path, repo_root: &Path) -> String {
    path.strip_prefix(repo_root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn write_or_check(path: &Path, content: &str, check_only: bool) -> Result<()> {
    if check_only {
        let existing = fs::read_to_string(path)
            .with_context(|| format!("read existing generated file {}", path.display()))?;
        if existing != content {
            bail!(
                "generated schema artifact drift detected for {}; run cargo run -p xtask -- db-docs",
                path.display()
            );
        }
        return Ok(());
    }

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create output directory {}", parent.display()))?;
    }
    fs::write(path, content).with_context(|| format!("write generated file {}", path.display()))?;
    Ok(())
}

// ---- CPD file list generator ----
//
// WHY: The Makefile `_CPD_REGEN_LIST` variable used `find crates -name '*.rs'`
//      combined with Make `foreach` expansions to build per-file and per-dir
//      exclusion flags.  That approach is fragile (ordering-sensitive, temp-file
//      race condition under parallel Make, opaque exclusion logic).  This
//      implementation owns the exclusion policy in Rust with deterministic
//      output and explicit error handling.
//
// WHAT: Walks `crates/` from the repo root, filters *.rs files, applies the
//       same exclusion list that the Makefile maintained, and writes one
//       absolute path per line to the output file.
//
// HOW: `cargo run -p xtask -- cpd-file-list [--output <path>]`
//      Default output: /tmp/cpd_src_list.txt (same default as the old Makefile var).

// Directories excluded from CPD scans.  These are auto-generated mirrors
// with zero hand-written logic; scanning them is noise, not signal.
const CPD_EXCLUDE_DIRS: &[&str] = &["crates/data_core/src/registry_mirrors"];

// Individual files excluded from CPD scans.  Transcribed reference datasets
// or auto-generated code that happens not to live under an excluded directory.
const CPD_EXCLUDE_FILES: &[&str] = &[
    "crates/materials_core/src/optical_database.rs",
    "crates/materials_core/src/crystal_symmetry.rs",
];

fn run_cpd_file_list(output: &Path) -> Result<()> {
    let repo_root = repo_root()?;
    let crates_dir = repo_root.join("crates");

    let mut paths: Vec<String> = Vec::new();

    for entry in WalkDir::new(&crates_dir)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if path.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }

        // Compute the path relative to repo root for exclusion checks.
        let rel = path
            .strip_prefix(&repo_root)
            .unwrap_or(path)
            .to_str()
            .unwrap_or("")
            .replace('\\', "/");

        // Check excluded directories.
        let in_excluded_dir = CPD_EXCLUDE_DIRS.iter().any(|excl| rel.starts_with(excl));
        if in_excluded_dir {
            continue;
        }

        // Check excluded individual files.
        let is_excluded_file = CPD_EXCLUDE_FILES.iter().any(|excl| rel == *excl);
        if is_excluded_file {
            continue;
        }

        // Store the absolute path string (PMD cpd --file-list expects absolute paths).
        paths.push(
            path.canonicalize()
                .unwrap_or_else(|_| path.to_path_buf())
                .to_string_lossy()
                .into_owned(),
        );
    }

    paths.sort();

    if let Some(parent) = output.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("create output directory {}", parent.display()))?;
    }
    fs::write(output, paths.join("\n") + "\n")
        .with_context(|| format!("write CPD file list to {}", output.display()))?;

    eprintln!(
        "cpd-file-list: wrote {} paths to {}",
        paths.len(),
        output.display()
    );
    Ok(())
}

// ---- Worker budget ----
//
// WHY: scripts/detect_worker_budget.sh used a 60-line chain of nproc /
//      getconf / sysctl / lscpu / /proc/cpuinfo fallbacks plus awk to compute
//      nproc/2.  std::thread::available_parallelism() covers all platforms in
//      one call.  This subcommand is the preferred non-Makefile consumer.
//
// NOTE: The Makefile still uses `$(shell sh scripts/detect_worker_budget.sh)`
//       for the WORKER_BUDGET variable because that variable is evaluated at
//       Make parse time, before any cargo compilation step runs.  The shell
//       script is therefore retained as a zero-overhead fallback for that
//       specific context.  All other callers should use this subcommand.
//
// HOW: `cargo run -p xtask -- worker-budget`
//      Prints a single integer: available_parallelism / 2, minimum 1.

fn run_worker_budget() -> Result<()> {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let budget = (threads / 2).max(1);
    println!("{budget}");
    Ok(())
}

fn repo_root() -> Result<PathBuf> {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .context("resolve repository root from xtask manifest directory")
}

// ===========================================================================
// gate-local xtask driver (Tier-5B, 2026-05-12)
// ===========================================================================

/// Routing flags emitted by workspace-routing CLI (parsed from stderr lines
/// like `[ci-routing] run_rust=True`).
#[derive(Debug, Default)]
struct RoutingFlags {
    run_rust: bool,
    run_governance: bool,
    run_check: bool,
    scope: String,
}

fn parse_routing_flags(stderr: &str, stdout: &str) -> RoutingFlags {
    let mut flags = RoutingFlags {
        scope: stdout.trim().to_string(),
        ..Default::default()
    };
    for line in stderr.lines() {
        let Some(rest) = line.strip_prefix("[ci-routing] ") else {
            continue;
        };
        let Some((key, val)) = rest.split_once('=') else {
            continue;
        };
        match key {
            "run_rust" => flags.run_rust = val == "True",
            "run_governance" => flags.run_governance = val == "True",
            "run_check" => flags.run_check = val == "True",
            _ => {}
        }
    }
    flags
}

/// Run a make sub-target, streaming stdout/stderr to the parent terminal,
/// and return (exit_code, elapsed_seconds).
fn run_make_target(root: &Path, target: &str, env: &[(&str, &str)]) -> Result<(i32, f64)> {
    let start = Instant::now();
    let mut command = std::process::Command::new("make");
    command.current_dir(root);
    command.arg(target);
    for (key, value) in env {
        command.env(key, value);
    }
    let status = command
        .status()
        .with_context(|| format!("spawn make {target}"))?;
    let elapsed = start.elapsed().as_secs_f64();
    Ok((status.code().unwrap_or(-1), elapsed))
}

fn run_gate_local(cli: GateLocalCli) -> Result<i32> {
    let root = repo_root()?;
    let routing_bin = cli.routing_bin.unwrap_or_else(|| {
        root.join(".cache/gate-target/gate-tools/workspace-routing")
    });

    // Default timing path: data/output/audit/<date>/gate-timing-<ts>.jsonl
    let timing_path = cli.timing_json.unwrap_or_else(|| {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let date = chrono::Local::now().format("%Y-%m-%d").to_string();
        root.join("data/output/audit")
            .join(date)
            .join(format!("gate-timing-{now}.jsonl"))
    });

    let timing = TimingRecorder::new(Some(timing_path.clone()));

    // Routing decision via cached workspace-routing binary.
    let mut flags = RoutingFlags {
        run_rust: true,
        run_governance: true,
        run_check: true,
        scope: "--workspace".to_string(),
    };
    let routing_start = Instant::now();
    if routing_bin.exists() {
        let output = std::process::Command::new(&routing_bin)
            .arg("--local")
            .arg("--verbose")
            .current_dir(&root)
            .output()
            .with_context(|| format!("run workspace-routing at {}", routing_bin.display()))?;
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        eprint!("{}", stderr);
        flags = parse_routing_flags(&stderr, &stdout);
    } else {
        eprintln!(
            "[gate-local] WARNING: workspace-routing cache not found at {}; running full workspace",
            routing_bin.display()
        );
    }
    timing.write(serde_json::json!({
        "kind": "routing",
        "elapsed_sec": routing_start.elapsed().as_secs_f64(),
        "run_rust": flags.run_rust,
        "run_governance": flags.run_governance,
        "run_check": flags.run_check,
        "scope": flags.scope,
    }))?;

    let mut total_exit = 0;

    // Phase: cache-check (always, near-instant when memoized).
    let (code, elapsed) = run_make_target(&root, "cache-check", &[])?;
    timing.write(serde_json::json!({
        "kind": "phase",
        "phase": "cache-check",
        "exit_code": code,
        "elapsed_sec": elapsed,
    }))?;
    if code != 0 {
        return Ok(code);
    }

    // Phase: make check.
    if flags.run_check || cli.force_check {
        let (code, elapsed) = run_make_target(&root, "check", &[])?;
        timing.write(serde_json::json!({
            "kind": "phase",
            "phase": "check",
            "exit_code": code,
            "elapsed_sec": elapsed,
        }))?;
        if code != 0 {
            return Ok(code);
        }
    } else {
        eprintln!("[gate-local] SKIP: no check-relevant (non-Rust) file changes detected.");
        timing.write(serde_json::json!({
            "kind": "skip",
            "phase": "check",
            "reason": "run_check=False",
        }))?;
    }

    // Phase: rust-regression-scoped.
    if flags.run_rust || cli.force_rust {
        let scope = if flags.scope.is_empty() {
            "--workspace".to_string()
        } else {
            flags.scope.clone()
        };
        let env_pairs: Vec<(&str, &str)> = vec![
            ("RUST_SCOPE", scope.as_str()),
            ("RUST_RUN_HEAVY", "0"),
        ];
        let (code, elapsed) = run_make_target(&root, "rust-regression-scoped", &env_pairs)?;
        timing.write(serde_json::json!({
            "kind": "phase",
            "phase": "rust-regression-scoped",
            "exit_code": code,
            "elapsed_sec": elapsed,
            "scope": scope,
        }))?;
        if code != 0 {
            total_exit = code;
        }
    } else {
        eprintln!("[gate-local] SKIP: no Rust-relevant changes detected.");
        timing.write(serde_json::json!({
            "kind": "skip",
            "phase": "rust-regression-scoped",
            "reason": "run_rust=False",
        }))?;
    }

    // Phase: governance-gate-readonly.
    if flags.run_governance || cli.force_governance {
        let (code, elapsed) = run_make_target(&root, "governance-gate-readonly", &[])?;
        timing.write(serde_json::json!({
            "kind": "phase",
            "phase": "governance-gate-readonly",
            "exit_code": code,
            "elapsed_sec": elapsed,
        }))?;
        if code != 0 && total_exit == 0 {
            total_exit = code;
        }
    } else {
        eprintln!("[gate-local] SKIP: no governance-relevant changes detected.");
        timing.write(serde_json::json!({
            "kind": "skip",
            "phase": "governance-gate-readonly",
            "reason": "run_governance=False",
        }))?;
    }

    timing.record_summary(total_exit)?;
    eprintln!(
        "[gate-local] xtask driver complete. Timing JSONL at {}",
        timing_path.display()
    );
    Ok(total_exit)
}

// ===========================================================================
// gate-timing-summary + gate-timing-regression-check (Tier-5C, 2026-05-12)
// ===========================================================================

/// One phase record parsed out of a gate-timing-*.jsonl file.
#[derive(Debug, Clone)]
struct PhaseRecord {
    file_mtime_secs: u64,
    phase: String,
    elapsed_sec: f64,
    exit_code: i64,
}

fn default_audit_root() -> Result<PathBuf> {
    Ok(repo_root()?.join("data/output/audit"))
}

/// Walk audit_root for date directories within `since_days`, collect
/// gate-timing-*.jsonl phase records.
fn collect_phase_records(audit_root: &Path, since_days: u64) -> Result<Vec<PhaseRecord>> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .context("system clock before unix epoch")?
        .as_secs();
    let cutoff = now.saturating_sub(since_days * 86_400);

    let mut records = Vec::new();
    if !audit_root.exists() {
        return Ok(records);
    }
    for entry in fs::read_dir(audit_root)
        .with_context(|| format!("read audit root {}", audit_root.display()))?
    {
        let entry = entry?;
        let date_dir = entry.path();
        if !date_dir.is_dir() {
            continue;
        }
        for jsonl in fs::read_dir(&date_dir)
            .with_context(|| format!("read date dir {}", date_dir.display()))?
        {
            let jsonl = jsonl?;
            let path = jsonl.path();
            let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if !name.starts_with("gate-timing-") || !name.ends_with(".jsonl") {
                continue;
            }
            let meta = jsonl.metadata()?;
            let mtime_secs = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);
            if mtime_secs < cutoff {
                continue;
            }
            let text = fs::read_to_string(&path)
                .with_context(|| format!("read {}", path.display()))?;
            for line in text.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let value: serde_json::Value = match serde_json::from_str(line) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if value.get("kind").and_then(|v| v.as_str()) != Some("phase") {
                    continue;
                }
                let phase = match value.get("phase").and_then(|v| v.as_str()) {
                    Some(p) => p.to_string(),
                    None => continue,
                };
                let elapsed_sec = match value.get("elapsed_sec").and_then(|v| v.as_f64()) {
                    Some(e) => e,
                    None => continue,
                };
                let exit_code = value
                    .get("exit_code")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                records.push(PhaseRecord {
                    file_mtime_secs: mtime_secs,
                    phase,
                    elapsed_sec,
                    exit_code,
                });
            }
        }
    }
    records.sort_by_key(|r| r.file_mtime_secs);
    Ok(records)
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[derive(Debug, serde::Serialize)]
struct PhaseStats {
    phase: String,
    count: usize,
    mean_sec: f64,
    median_sec: f64,
    p95_sec: f64,
    max_sec: f64,
    min_sec: f64,
    last_sec: f64,
}

fn compute_phase_stats(records: &[PhaseRecord]) -> Vec<PhaseStats> {
    let mut by_phase: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut last_by_phase: BTreeMap<String, f64> = BTreeMap::new();
    for r in records {
        by_phase
            .entry(r.phase.clone())
            .or_default()
            .push(r.elapsed_sec);
        last_by_phase.insert(r.phase.clone(), r.elapsed_sec);
    }
    let mut out = Vec::new();
    for (phase, mut elapsed) in by_phase {
        elapsed.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let count = elapsed.len();
        let sum: f64 = elapsed.iter().sum();
        let mean = sum / count as f64;
        let median = percentile(&elapsed, 0.5);
        let p95 = percentile(&elapsed, 0.95);
        let max = *elapsed.last().unwrap_or(&0.0);
        let min = *elapsed.first().unwrap_or(&0.0);
        let last = last_by_phase.get(&phase).copied().unwrap_or(0.0);
        out.push(PhaseStats {
            phase,
            count,
            mean_sec: mean,
            median_sec: median,
            p95_sec: p95,
            max_sec: max,
            min_sec: min,
            last_sec: last,
        });
    }
    out
}

fn run_gate_timing_summary(cli: GateTimingSummaryCli) -> Result<()> {
    let audit_root = match cli.audit_root {
        Some(p) => p,
        None => default_audit_root()?,
    };
    let mut records = collect_phase_records(&audit_root, cli.since_days)?;
    if let Some(filter) = &cli.phase {
        records.retain(|r| &r.phase == filter);
    }

    if records.is_empty() {
        eprintln!(
            "[gate-timing-summary] no records under {} within last {} days",
            audit_root.display(),
            cli.since_days
        );
        return Ok(());
    }

    let stats = compute_phase_stats(&records);

    match cli.format.as_str() {
        "json" => {
            println!("{}", serde_json::to_string_pretty(&stats)?);
        }
        "table" => {
            println!(
                "{:<28} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
                "phase", "count", "mean", "median", "p95", "min", "max", "last"
            );
            println!("{:-<108}", "");
            for s in &stats {
                println!(
                    "{:<28} {:>6} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3}",
                    s.phase,
                    s.count,
                    s.mean_sec,
                    s.median_sec,
                    s.p95_sec,
                    s.min_sec,
                    s.max_sec,
                    s.last_sec,
                );
            }
            if cli.last > 0 {
                let recent: Vec<&PhaseRecord> =
                    records.iter().rev().take(cli.last * stats.len()).collect();
                println!();
                println!("Last {} raw records (newest first):", cli.last * stats.len());
                for r in recent {
                    println!(
                        "  {}  {:<28}  {:>10.3}s  exit={}",
                        chrono::DateTime::<chrono::Local>::from(
                            std::time::UNIX_EPOCH
                                + std::time::Duration::from_secs(r.file_mtime_secs)
                        )
                        .format("%Y-%m-%d %H:%M"),
                        r.phase,
                        r.elapsed_sec,
                        r.exit_code,
                    );
                }
            }
        }
        other => bail!("unsupported --format: {other} (expected table|json)"),
    }
    Ok(())
}

/// One entry in the gate-tools-status table.
struct ToolStatusEntry {
    name: &'static str,
    cached_path: PathBuf,
    /// Source files whose mtime would trigger rebuild.
    source_deps: Vec<PathBuf>,
    /// True when the entry is a transient runtime artifact whose
    /// missing/absent state is the EXPECTED steady-state (e.g.,
    /// gate-local.lock is written at gate start and removed by trap
    /// at gate end -- absence means no gate is running).
    runtime_artifact: bool,
}

fn format_age(now: u64, then: u64) -> String {
    if now <= then {
        return "future".to_string();
    }
    let secs = now - then;
    if secs < 60 {
        format!("{}s ago", secs)
    } else if secs < 3600 {
        format!("{}m ago", secs / 60)
    } else if secs < 86400 {
        format!("{}h ago", secs / 3600)
    } else {
        format!("{}d ago", secs / 86400)
    }
}

fn run_gate_tools_status(cli: GateToolsStatusCli) -> Result<()> {
    let root = repo_root()?;
    let tools_dir = cli
        .tools_dir
        .unwrap_or_else(|| root.join(".cache/gate-target/gate-tools"));

    let entries = vec![
        ToolStatusEntry {
            name: "workspace-routing",
            cached_path: tools_dir.join("workspace-routing"),
            source_deps: vec![
                root.join("crates/gororoba_cli_data/src/bin/workspace_routing.rs"),
                root.join("crates/gororoba_cli_data/Cargo.toml"),
            ],
            runtime_artifact: false,
        },
        ToolStatusEntry {
            name: "host-profile.sh",
            cached_path: tools_dir.join("host-profile.sh"),
            source_deps: vec![
                root.join("xtask/src/main.rs"),
                root.join("xtask/Cargo.toml"),
            ],
            runtime_artifact: false,
        },
        ToolStatusEntry {
            name: "xtask",
            cached_path: tools_dir.join("xtask"),
            source_deps: vec![
                root.join("xtask/src/main.rs"),
                root.join("xtask/Cargo.toml"),
            ],
            runtime_artifact: false,
        },
        ToolStatusEntry {
            name: "markdown-registry",
            cached_path: tools_dir.join("markdown-registry"),
            source_deps: vec![
                root.join("crates/gororoba_cli_data/src/bin/markdown_registry.rs"),
                root.join("crates/gororoba_cli_data/Cargo.toml"),
            ],
            runtime_artifact: false,
        },
        ToolStatusEntry {
            name: "governance-verify",
            cached_path: tools_dir.join("governance-verify"),
            source_deps: vec![
                root.join("crates/gororoba_cli_data/src/bin/governance_verify.rs"),
                root.join("crates/gororoba_cli_data/Cargo.toml"),
            ],
            runtime_artifact: false,
        },
        ToolStatusEntry {
            name: "integrity-resolution",
            cached_path: tools_dir.join("integrity-resolution"),
            source_deps: vec![
                root.join("crates/gororoba_cli_data/src/bin/integrity_resolution.rs"),
                root.join("crates/gororoba_cli_data/Cargo.toml"),
            ],
            runtime_artifact: false,
        },
        ToolStatusEntry {
            name: "cache-check.last",
            cached_path: tools_dir.join("cache-check.last"),
            source_deps: vec![],
            runtime_artifact: true,
        },
        ToolStatusEntry {
            name: "gate-local.lock",
            cached_path: tools_dir.join("gate-local.lock"),
            source_deps: vec![],
            runtime_artifact: true,
        },
    ];

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    println!("Gate tools cache at {}", tools_dir.display());
    println!();
    println!("tool                       size_kb          mtime          age  status");
    println!("------------------------------------------------------------------------------");

    let mut any_stale = false;
    for entry in &entries {
        if !entry.cached_path.exists() {
            let status = if entry.runtime_artifact {
                "absent (runtime artifact, expected when idle)"
            } else {
                any_stale = true;
                "MISSING"
            };
            println!(
                "{:<22} {:>10} {:>14} {:>12}  {}",
                entry.name, "-", "-", "-", status
            );
            continue;
        }
        let meta = fs::metadata(&entry.cached_path)?;
        let size_kb = meta.len() / 1024;
        let mtime_secs = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let mtime_str = chrono::DateTime::<chrono::Local>::from(
            std::time::UNIX_EPOCH + std::time::Duration::from_secs(mtime_secs),
        )
        .format("%m-%d %H:%M")
        .to_string();

        // Check source deps: if any is newer than cached binary, status=STALE.
        let mut newest_dep: Option<u64> = None;
        let mut newest_dep_name: Option<String> = None;
        for dep in &entry.source_deps {
            if let Ok(dep_meta) = fs::metadata(dep)
                && let Ok(modified) = dep_meta.modified()
                && let Ok(dep_secs) = modified.duration_since(std::time::UNIX_EPOCH)
            {
                let dep_secs = dep_secs.as_secs();
                if dep_secs > newest_dep.unwrap_or(0) {
                    newest_dep = Some(dep_secs);
                    newest_dep_name = Some(
                        dep.file_name()
                            .and_then(|s| s.to_str())
                            .unwrap_or("?")
                            .to_string(),
                    );
                }
            }
        }
        let status = match newest_dep {
            Some(dep_t) if dep_t > mtime_secs => {
                any_stale = true;
                format!(
                    "STALE (newer: {})",
                    newest_dep_name.as_deref().unwrap_or("?")
                )
            }
            _ if entry.source_deps.is_empty() => "ok (runtime artifact)".to_string(),
            _ => "ok".to_string(),
        };
        println!(
            "{:<22} {:>10} {:>14} {:>12}  {}",
            entry.name,
            size_kb,
            mtime_str,
            format_age(now, mtime_secs),
            status,
        );
    }

    println!();
    if any_stale {
        println!("Some tools are stale or missing. Run `make gate-tools` to refresh.");
    } else {
        println!("All cached gate tools fresh.");
    }
    Ok(())
}

fn run_gate_timing_regression_check(cli: GateTimingRegressionCheckCli) -> Result<i32> {
    let audit_root = match cli.audit_root {
        Some(p) => p,
        None => default_audit_root()?,
    };
    let records = collect_phase_records(&audit_root, cli.baseline_days)?;
    if records.is_empty() {
        eprintln!(
            "[gate-timing-regression-check] no records under {} within last {} days; skipping",
            audit_root.display(),
            cli.baseline_days
        );
        return Ok(0);
    }

    // Group by phase, last is "latest", rest is baseline.
    let mut by_phase: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut latest_by_phase: BTreeMap<String, f64> = BTreeMap::new();
    for r in &records {
        by_phase
            .entry(r.phase.clone())
            .or_default()
            .push(r.elapsed_sec);
        latest_by_phase.insert(r.phase.clone(), r.elapsed_sec);
    }

    let mut regressed = false;
    println!(
        "{:<28} {:>10} {:>10} {:>10} {:>10} {:>8}",
        "phase", "baseline_n", "median", "threshold", "latest", "status"
    );
    println!("{:-<90}", "");
    for (phase, mut elapsed) in by_phase {
        let latest = latest_by_phase.get(&phase).copied().unwrap_or(0.0);
        // Drop latest from baseline samples for cleaner comparison.
        if let Some(pos) = elapsed
            .iter()
            .rposition(|v| (*v - latest).abs() < f64::EPSILON)
        {
            elapsed.remove(pos);
        }
        if elapsed.len() < cli.min_samples {
            println!(
                "{:<28} {:>10} {:>10} {:>10} {:>10.3} {:>8}",
                phase,
                elapsed.len(),
                "n/a",
                "n/a",
                latest,
                "warmup"
            );
            continue;
        }
        elapsed.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = percentile(&elapsed, 0.5);
        let thresh = median * cli.threshold;
        let status = if latest > thresh { "FAIL" } else { "ok" };
        if latest > thresh {
            regressed = true;
        }
        println!(
            "{:<28} {:>10} {:>10.3} {:>10.3} {:>10.3} {:>8}",
            phase,
            elapsed.len(),
            median,
            thresh,
            latest,
            status,
        );
    }

    if regressed {
        eprintln!(
            "[gate-timing-regression-check] FAIL: at least one phase regressed >{:.2}x median",
            cli.threshold
        );
        Ok(2)
    } else {
        eprintln!("[gate-timing-regression-check] OK: no regressions detected");
        Ok(0)
    }
}

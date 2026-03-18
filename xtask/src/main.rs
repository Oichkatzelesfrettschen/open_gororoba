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
    packages: Vec<String>,
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
        bail!(
            "usage: cargo run -p xtask -- <db-docs|host-profile|local-nextest-plan|gate-audit|sparse-profile|gpu-profile> [args]"
        );
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
        "gate-audit" => run_gate_audit(parse_gate_audit_args(args)?),
        "sparse-profile" => run_sparse_profile(SparseProfileCli::try_parse_from(
            std::iter::once("sparse-profile".to_string()).chain(args),
        )?),
        "gpu-profile" => run_gpu_profile(GpuProfileCli::try_parse_from(
            std::iter::once("gpu-profile".to_string()).chain(args),
        )?),
        other => bail!("unknown xtask command: {other}"),
    }
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
    let summary_path = cli.output_dir.join(format!("{}_sweep_summary.csv", cli.bench));
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
    let (elapsed_seconds, throughput_mlups, effective_glups) = parse_gpu_sparse_bench_stdout(&stdout_text);
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
    text.lines()
        .find_map(|line| {
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
    let mut writer = csv::Writer::from_path(path)
        .with_context(|| format!("create {}", path.display()))?;
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
        let manifest_path = cli
            .output_dir
            .join(format!("{}_manifest.json", cli.bench));
        fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?).with_context(
            || format!("write sparse profile manifest {}", manifest_path.display()),
        )?;
        println!(
            "nsys not available; sparse profiling skipped without blocking the workflow"
        );
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

    let manifest_path = cli
        .output_dir
        .join(format!("{}_manifest.json", cli.bench));
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
    command.status().map(|status| status.success()).unwrap_or(false)
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
    matches
        .pop()
        .with_context(|| format!("no benchmark binary found for {bench} in {}", deps_dir.display()))
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

fn parse_gate_audit_args(args: impl Iterator<Item = String>) -> Result<Option<PathBuf>> {
    let mut output_dir = None;
    let mut iter = args.peekable();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--output-dir" => {
                let Some(value) = iter.next() else {
                    bail!("gate-audit --output-dir requires a value");
                };
                output_dir = Some(PathBuf::from(value));
            }
            other => bail!("unknown gate-audit argument: {other}"),
        }
    }
    Ok(output_dir)
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
        &repo_root.join("docs/db/catalog.md"),
        &outputs.catalog_md,
        check_only,
    )?;
    if check_only {
        println!("db-docs OK: generated schema artifacts match committed files");
    } else {
        println!("db-docs OK: regenerated db/schema.sql docs/db/schema.json docs/db/catalog.md");
    }
    Ok(())
}

fn run_gate_audit(output_dir_override: Option<PathBuf>) -> Result<()> {
    let repo_root = repo_root()?;
    let generated_at = Local::now();
    let timestamp = generated_at.format("%Y-%m-%d/%H%M%S").to_string();
    let output_dir = match output_dir_override {
        Some(path) if path.is_absolute() => path,
        Some(path) => repo_root.join(path),
        None => repo_root.join("reports").join("gates").join(timestamp),
    };
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("create gate-audit output directory {}", output_dir.display()))?;

    let commands: Vec<(&str, Vec<String>)> = vec![
        (
            "gate-ci-registry",
            vec!["./makew".to_string(), "gate-ci-registry".to_string()],
        ),
        (
            "gate-ci-rust",
            vec!["./makew".to_string(), "gate-ci-rust".to_string()],
        ),
        (
            "nextest-list",
            vec![
                "cargo".to_string(),
                "nextest".to_string(),
                "list".to_string(),
                "--workspace".to_string(),
                "--tests".to_string(),
            ],
        ),
    ];

    let cargo_home = repo_root.join(".cache").join("cargo-home");
    let cargo_target_dir = repo_root.join(".cache").join("gate-target");

    let mut summary_lines = vec![
        format!(
            "# Gate Audit ({})",
            generated_at.to_rfc3339_opts(SecondsFormat::Secs, false)
        ),
        String::new(),
        format!("Output directory: `{}`", repo_relative(&output_dir, &repo_root)),
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
    .with_context(|| format!("write gate-audit manifest {}", latest_manifest_path.display()))?;

    println!("Wrote: {}", repo_relative(&summary_path, &repo_root));
    if failures != 0 {
        bail!("gate-audit failed in {failures} step(s)");
    }
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
    if !test_packages.is_empty() {
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
    catalog_md: String,
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
    let catalog_md = render_catalog_markdown(&snapshot);
    let _ = repo_root;
    Ok(GeneratedSchemaOutputs {
        schema_sql,
        schema_json,
        catalog_md,
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

fn render_catalog_markdown(snapshot: &SchemaSnapshot) -> String {
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

fn repo_root() -> Result<PathBuf> {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .context("resolve repository root from xtask manifest directory")
}

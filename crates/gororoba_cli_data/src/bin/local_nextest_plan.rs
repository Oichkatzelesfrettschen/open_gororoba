use anyhow::{Context, Result};
use clap::Parser;
use serde_json::json;
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    process::{Command, ExitCode},
    time::Instant,
};
use walkdir::WalkDir;

const INLINE_TEST_MARKERS: &[&str] = &["#[test]", "#[cfg(test)]", "mod tests"];

#[derive(Parser, Debug)]
#[command(
    name = "local-nextest-plan",
    about = "Run a package-aware grouped local nextest plan"
)]
struct Cli {
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
        self.write(json!({
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
        self.write(json!({
            "kind": "run",
            "packages": packages,
            "targets": targets,
            "command": command,
            "returncode": returncode,
            "elapsed_sec": elapsed_sec,
        }))
    }

    fn record_summary(&self, returncode: i32) -> Result<()> {
        self.write(json!({
            "kind": "summary",
            "run_count": self.run_count,
            "skip_count": self.skip_count,
            "returncode": returncode,
            "total_elapsed_sec": self.total_start.elapsed().as_secs_f64(),
        }))
    }
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crate must be nested under repo/crates")
        .to_path_buf()
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

fn build_command(
    packages: &[String],
    plan: &PackagePlan,
    build_jobs: &str,
    test_threads: &str,
    filterset: &str,
) -> (Vec<String>, serde_json::Value) {
    let mut command = vec![
        "cargo".to_string(),
        "nextest".to_string(),
        "run".to_string(),
        "--build-jobs".to_string(),
        build_jobs.to_string(),
        "--test-threads".to_string(),
        test_threads.to_string(),
    ];
    if plan.has_lib_tests {
        command.push("--lib".to_string());
    }
    for package in packages {
        command.push("-p".to_string());
        command.push(package.clone());
    }
    for test_name in &plan.tests {
        command.push("--test".to_string());
        command.push(test_name.clone());
    }
    if !filterset.is_empty() {
        command.push("-E".to_string());
        command.push(filterset.to_string());
    }

    let targets = packages
        .iter()
        .map(|package| {
            let mut selected = Vec::new();
            if plan.has_lib_tests {
                selected.push("lib".to_string());
            }
            selected.extend(plan.tests.iter().map(|name| format!("test:{name}")));
            (package.clone(), json!(selected))
        })
        .collect::<serde_json::Map<String, serde_json::Value>>();
    (command, serde_json::Value::Object(targets))
}

fn run_command(
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

fn run(cli: Cli) -> Result<i32> {
    let root = repo_root();
    let mut timing = TimingRecorder::new(cli.timing_json_out);
    let mut grouped_plans: Vec<(PackagePlan, Vec<String>)> = Vec::new();

    for package in &cli.packages {
        let Some(plan) = package_plan(&root, package)? else {
            let reason = "no inline lib tests and no integration tests";
            println!("[local-nextest] skip {package}: {reason}");
            timing.record_skip(package, reason)?;
            continue;
        };

        if let Some((_, packages)) = grouped_plans
            .iter_mut()
            .find(|(signature, _)| *signature == plan)
        {
            packages.push(package.clone());
        } else {
            grouped_plans.push((plan, vec![package.clone()]));
        }
    }

    let mut exit_code = 0;
    for (plan, packages) in &grouped_plans {
        let (command, targets) = build_command(
            packages,
            plan,
            &cli.build_jobs,
            &cli.test_threads,
            &cli.filterset,
        );
        exit_code = run_command(&root, packages, &command, &targets, &mut timing)?;
        if exit_code != 0 {
            timing.record_summary(exit_code)?;
            return Ok(exit_code);
        }
    }

    timing.record_summary(exit_code)?;
    Ok(exit_code)
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(code) => ExitCode::from(code as u8),
        Err(err) => {
            eprintln!("{err:?}");
            ExitCode::from(1)
        }
    }
}

use anyhow::{Context, Result, bail};
use chrono::{NaiveDate, Timelike, Utc};
use clap::Parser;
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    thread,
    time::Instant,
};

const DEFAULT_DATE_TOKEN: &str = "2026_02_14";
const DEFAULT_REPORT_TEMPLATE: &str = "reports/synthesis_execution_contract_{date_token}.toml";

#[derive(Debug, Parser)]
#[command(
    name = "synthesis-execution-contract",
    about = "Run synthesis execution contract checks and emit a TOML rollup"
)]
struct Args {
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,

    #[arg(long, default_value = DEFAULT_DATE_TOKEN)]
    date_token: String,

    #[arg(long, default_value = "")]
    report_path: String,
}

#[derive(Debug, Clone)]
struct ContractStep {
    step_id: &'static str,
    label: &'static str,
    argv: &'static [&'static str],
}

#[derive(Debug, Clone)]
struct StepResult {
    step_id: String,
    label: String,
    command: String,
    exit_code: i32,
    outcome: String,
    started_utc: String,
    finished_utc: String,
    duration_seconds: f64,
    stdout_excerpt: String,
    stderr_excerpt: String,
    skip_reason: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let repo_root = args.repo_root.canonicalize().context("resolve repo root")?;
    let (date_iso, date_token) = normalize_date_token(&args.date_token)?;
    let report_path = if args.report_path.is_empty() {
        repo_root.join(DEFAULT_REPORT_TEMPLATE.replace("{date_token}", &date_token))
    } else {
        let explicit = PathBuf::from(&args.report_path);
        if explicit.is_absolute() {
            explicit
        } else {
            repo_root.join(explicit)
        }
    };
    if let Some(parent) = report_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }

    let results = run_contract(&repo_root)?;
    let rollup_display_path = display_path(&report_path, &repo_root);
    let rendered = render_report(&date_iso, &date_token, &rollup_display_path, &results)?;
    fs::write(&report_path, rendered)
        .with_context(|| format!("write {}", report_path.display()))?;

    let failure_count = results
        .iter()
        .filter(|result| result.outcome == "fail")
        .count();
    let skipped_count = results
        .iter()
        .filter(|result| result.outcome == "skipped")
        .count();
    let overall_outcome = if failure_count == 0 && skipped_count == 0 {
        "pass"
    } else {
        "fail"
    };

    println!("synthesis-execution-contract: {}", overall_outcome);
    println!("rollup: {}", rollup_display_path);
    if overall_outcome == "pass" {
        Ok(())
    } else {
        bail!("synthesis execution contract failed")
    }
}

fn normalize_date_token(raw: &str) -> Result<(String, String)> {
    let cleaned = raw.trim();
    if cleaned.is_empty() {
        bail!("date token cannot be empty");
    }
    let parsed = NaiveDate::parse_from_str(&cleaned.replace('_', "-"), "%Y-%m-%d")
        .with_context(|| format!("invalid date token: {}", raw))?;
    Ok((
        parsed.format("%Y-%m-%d").to_string(),
        parsed.format("%Y_%m_%d").to_string(),
    ))
}

fn utc_now() -> String {
    Utc::now()
        .with_nanosecond(0)
        .unwrap_or_else(Utc::now)
        .to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

fn display_path(path: &Path, repo_root: &Path) -> String {
    path.strip_prefix(repo_root)
        .map(|value| value.to_string_lossy().replace('\\', "/"))
        .unwrap_or_else(|_| path.to_string_lossy().replace('\\', "/"))
}

fn run_contract(repo_root: &Path) -> Result<Vec<StepResult>> {
    let group_a = vec![
        ContractStep {
            step_id: "governance_gate",
            label: "governance-gate",
            argv: &["make", "governance-gate"],
        },
        ContractStep {
            step_id: "registry_acceptance_gate",
            label: "registry-acceptance-gate",
            argv: &["make", "registry-acceptance-gate"],
        },
    ];
    let group_b = vec![
        ContractStep {
            step_id: "typed_policy_strict",
            label: "registry-verify-typed-policy-error",
            argv: &["make", "registry-verify-typed-policy-error"],
        },
        ContractStep {
            step_id: "project_counter_sync_check",
            label: "project-counter-sync --check",
            argv: &[
                "cargo",
                "run",
                "--release",
                "--bin",
                "project-counter-sync",
                "--",
                "--check",
            ],
        },
    ];

    let mut handles = Vec::new();
    for step in group_a.clone() {
        let repo_root = repo_root.to_path_buf();
        handles.push(thread::spawn(move || run_one(&repo_root, &step)));
    }

    let mut results = Vec::new();
    for handle in handles {
        results.push(
            handle
                .join()
                .map_err(|_| anyhow::anyhow!("thread join failure"))??,
        );
    }
    results.sort_by_key(|result| match result.step_id.as_str() {
        "governance_gate" => 0usize,
        "registry_acceptance_gate" => 1usize,
        _ => usize::MAX,
    });

    for step in group_b {
        results.push(run_one(repo_root, &step)?);
    }
    Ok(results)
}

fn run_one(repo_root: &Path, step: &ContractStep) -> Result<StepResult> {
    let command = shell_join(step.argv);
    let started_utc = utc_now();
    let t0 = Instant::now();
    let completed = Command::new(step.argv[0])
        .args(&step.argv[1..])
        .current_dir(repo_root)
        .output()
        .with_context(|| format!("run {}", command))?;
    let duration_seconds = t0.elapsed().as_secs_f64();
    Ok(StepResult {
        step_id: step.step_id.to_string(),
        label: step.label.to_string(),
        command,
        exit_code: completed.status.code().unwrap_or(1),
        outcome: if completed.status.success() {
            "pass".to_string()
        } else {
            "fail".to_string()
        },
        started_utc,
        finished_utc: utc_now(),
        duration_seconds,
        stdout_excerpt: excerpt(&String::from_utf8_lossy(&completed.stdout), 1200),
        stderr_excerpt: excerpt(&String::from_utf8_lossy(&completed.stderr), 1200),
        skip_reason: String::new(),
    })
}

fn render_report(
    date_iso: &str,
    date_token: &str,
    rollup_path: &str,
    results: &[StepResult],
) -> Result<String> {
    let success_count = results
        .iter()
        .filter(|result| result.outcome == "pass")
        .count();
    let failure_count = results
        .iter()
        .filter(|result| result.outcome == "fail")
        .count();
    let skipped_count = results
        .iter()
        .filter(|result| result.outcome == "skipped")
        .count();
    let overall_outcome = if failure_count == 0 && skipped_count == 0 {
        "pass"
    } else {
        "fail"
    };
    let first_failure = results.iter().find(|result| result.outcome == "fail");

    let mut lines = vec![
        "# Synthesis execution contract rollup.".to_string(),
        String::new(),
        "[report]".to_string(),
        format!("id = {}", q(&format!("synthesis-execution-contract-{}", date_iso))),
        format!("date = {}", q(date_iso)),
        format!("date_token = {}", q(date_token)),
        "status = \"completed\"".to_string(),
        "target = \"synthesis-execution-contract\"".to_string(),
        format!("rollup_path = {}", q(rollup_path)),
        format!("overall_outcome = {}", q(overall_outcome)),
        format!("all_commands_passed = {}", bool_literal(overall_outcome == "pass")),
        format!("success_count = {}", success_count),
        format!("failure_count = {}", failure_count),
        format!("skipped_count = {}", skipped_count),
        String::new(),
        "[commands]".to_string(),
        "governance_gate = \"make governance-gate\"".to_string(),
        "registry_acceptance_gate = \"make registry-acceptance-gate\"".to_string(),
        "typed_policy_strict = \"make registry-verify-typed-policy-error\"".to_string(),
        "project_counter_sync_check = \"cargo run --release --bin project-counter-sync -- --check\"".to_string(),
        String::new(),
        "[summary]".to_string(),
        format!(
            "first_failure_step = {}",
            q(first_failure.map(|result| result.step_id.as_str()).unwrap_or(""))
        ),
        format!(
            "first_failure_command = {}",
            q(first_failure.map(|result| result.command.as_str()).unwrap_or(""))
        ),
        String::new(),
    ];

    for (index, result) in results.iter().enumerate() {
        lines.push("[[step]]".to_string());
        lines.push(format!("order = {}", index + 1));
        lines.push(format!("id = {}", q(&result.step_id)));
        lines.push(format!("label = {}", q(&result.label)));
        lines.push(format!("command = {}", q(&result.command)));
        lines.push(format!("exit_code = {}", result.exit_code));
        lines.push(format!("outcome = {}", q(&result.outcome)));
        lines.push(format!("started_utc = {}", q(&result.started_utc)));
        lines.push(format!("finished_utc = {}", q(&result.finished_utc)));
        lines.push(format!("duration_seconds = {:.3}", result.duration_seconds));
        if !result.skip_reason.is_empty() {
            lines.push(format!("skip_reason = {}", q(&result.skip_reason)));
        }
        lines.push(format!("stdout_excerpt = {}", q(&result.stdout_excerpt)));
        lines.push(format!("stderr_excerpt = {}", q(&result.stderr_excerpt)));
        lines.push(String::new());
    }

    let rendered = format!("{}\n", lines.join("\n"));
    assert_ascii(&rendered, "synthesis execution contract output")?;
    Ok(rendered)
}

fn excerpt(text: &str, limit: usize) -> String {
    let normalized = text.replace("\r\n", "\n").trim().to_string();
    if normalized.is_empty() {
        return String::new();
    }
    let trimmed = if normalized.len() > limit {
        format!("{}\n...[truncated]...", &normalized[..limit])
    } else {
        normalized
    };
    ascii_safe(&trimmed)
}

fn ascii_safe(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch.is_ascii() {
            out.push(ch);
        } else {
            out.push('?');
        }
    }
    out
}

fn shell_join(argv: &[&str]) -> String {
    argv.iter()
        .map(|arg| {
            if arg
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '/' | '.' | '='))
            {
                (*arg).to_string()
            } else {
                format!("'{}'", arg.replace('\'', "'\"'\"'"))
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn q(value: &str) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "\"\"".to_string())
}

fn bool_literal(value: bool) -> &'static str {
    if value { "true" } else { "false" }
}

fn assert_ascii(text: &str, context: &str) -> Result<()> {
    if !text.is_ascii() {
        bail!("non-ASCII content in {}", context);
    }
    Ok(())
}

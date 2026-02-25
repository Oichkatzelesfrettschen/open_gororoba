use anyhow::{Context, Result};
use clap::{ArgAction, Parser, ValueEnum};
use glob::Pattern;
use gororoba_cli::data_governance::{
    DEFAULT_EXTERNAL_PROVENANCE_PATH, DEFAULT_EXTERNAL_SOURCES_PATH, ExternalSourceRule,
    blocked_source_deadline_issues, collect_files_under, load_external_hashes,
    load_external_sources, sha256_file, source_rule_for_path, to_repo_rel,
};
use regex::Regex;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Parser, Debug)]
#[command(
    name = "external-redownload-audit",
    about = "Audit and optionally execute reproducible external re-download verification"
)]
struct Args {
    #[arg(long, default_value = "data/external")]
    root: PathBuf,
    #[arg(long, default_value = DEFAULT_EXTERNAL_SOURCES_PATH)]
    sources: PathBuf,
    #[arg(long, default_value = DEFAULT_EXTERNAL_PROVENANCE_PATH)]
    provenance: PathBuf,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long, default_value_t = false, action = ArgAction::Set)]
    execute: bool,
    #[arg(long, default_value = "staging")]
    replay_mode: ReplayMode,
    #[arg(long, default_value = "target/external_replay_staging")]
    staging_root: PathBuf,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    fail_on_replay_side_effects: bool,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    force_refresh_replay: bool,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    require_replay_success: bool,
    #[arg(long, default_value = "wget,curl,fetch")]
    backend_order: String,
    #[arg(long, default_value_t = 600)]
    timeout_seconds: u64,
    #[arg(long)]
    max_files: Option<usize>,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    fail_on_unmatched_source: bool,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    fail_on_blocked_overdue: bool,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    fail_on_missing_action_plan: bool,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum ReplayMode {
    Live,
    Staging,
}

#[derive(Debug, Serialize)]
struct AttemptRecord {
    backend: String,
    url: String,
    status: String,
    detail: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    staging_dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    side_effect_detected: Option<bool>,
}

#[derive(Debug, Serialize)]
struct FileAuditRecord {
    path: String,
    source_id: String,
    source_status: String,
    blocked_action_plan: Vec<String>,
    expected_sha256: String,
    actual_sha256: String,
    status: String,
    detail: String,
    attempts: Vec<AttemptRecord>,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    generated_at_utc: String,
    execute: bool,
    replay_mode: String,
    replay_staging_root: String,
    backend_order: Vec<String>,
    total_files: usize,
    matched_source_files: usize,
    unmatched_source_files: usize,
    blocked_pending: usize,
    generator_replay_required: usize,
    replay_command_successes: usize,
    replay_command_failures: usize,
    replay_side_effects_count: usize,
    blocked_overdue: usize,
    blocked_missing_action_plan: usize,
    blocked_missing_action_plan_refs: usize,
    checksum_mismatch: usize,
    download_failures: usize,
    pass_count: usize,
    records: Vec<FileAuditRecord>,
}

#[derive(Debug, Clone)]
struct ReplayCommandOutcome {
    ok: bool,
    detail: String,
    staging_dir: Option<String>,
    side_effects: Vec<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let sources = load_external_sources(&args.sources)?;
    let provenance = load_external_hashes(&args.provenance)?;
    let now = chrono::Utc::now();
    let metadata_paths = replay_metadata_paths(&args.root)?;
    let metadata_set: HashSet<String> = metadata_paths.iter().cloned().collect();

    let mut files = collect_files_under(&args.root)?;
    files.retain(|path| !metadata_set.contains(path));
    if let Some(max_files) = args.max_files {
        files.truncate(max_files);
    }

    let backend_order: Vec<String> = args
        .backend_order
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if backend_order.is_empty() {
        anyhow::bail!("no backend provided in --backend-order");
    }

    let mut records = Vec::new();
    let mut replay_cache: HashMap<String, ReplayCommandOutcome> = HashMap::new();
    for path in &files {
        let mut attempts = Vec::new();
        let Some(source) = source_rule_for_path(path, &sources) else {
            records.push(FileAuditRecord {
                path: path.clone(),
                source_id: "".to_string(),
                source_status: "unmatched".to_string(),
                blocked_action_plan: Vec::new(),
                expected_sha256: provenance
                    .get(path)
                    .map(|v| v.sha256.clone())
                    .unwrap_or_default(),
                actual_sha256: "".to_string(),
                status: "missing_source".to_string(),
                detail: "no source rule matched path".to_string(),
                attempts,
            });
            continue;
        };

        let expected_sha = provenance
            .get(path)
            .map(|v| v.sha256.clone())
            .unwrap_or_default();
        if !args.execute {
            let coverage_status = if source.is_blocked() {
                "blocked_pending".to_string()
            } else {
                "coverage_only".to_string()
            };
            records.push(FileAuditRecord {
                path: path.clone(),
                source_id: source.id.clone(),
                source_status: source.status.clone(),
                blocked_action_plan: source.blocked_action_plan.clone(),
                expected_sha256: expected_sha,
                actual_sha256: "".to_string(),
                status: coverage_status,
                detail: "execute=false (coverage audit only)".to_string(),
                attempts,
            });
            continue;
        }

        if source.is_blocked() {
            records.push(FileAuditRecord {
                path: path.clone(),
                source_id: source.id.clone(),
                source_status: source.status.clone(),
                blocked_action_plan: source.blocked_action_plan.clone(),
                expected_sha256: expected_sha,
                actual_sha256: "".to_string(),
                status: "blocked_pending".to_string(),
                detail: "blocked source: troubleshooting and mirror workflow required".to_string(),
                attempts,
            });
            continue;
        }

        if !is_single_file_rule(source) {
            if args.execute {
                let replay_outcome = replay_cache
                    .entry(source.id.clone())
                    .or_insert_with(|| run_replay_command(source, &args, &metadata_paths));
                let replay_outcome = replay_outcome.clone();
                let replay_path_confirmed = replay_confirms_path(&replay_outcome, path);
                let side_effect_detected = !replay_outcome.side_effects.is_empty();
                attempts.push(AttemptRecord {
                    backend: "replay_command".to_string(),
                    url: source.retrieval_method.clone(),
                    status: if side_effect_detected {
                        "replay_side_effect".to_string()
                    } else if replay_outcome.ok {
                        "ok".to_string()
                    } else if replay_path_confirmed {
                        "ok_path_scoped".to_string()
                    } else {
                        "error".to_string()
                    },
                    detail: if side_effect_detected {
                        format!(
                            "{}\nside_effects: {}",
                            replay_outcome.detail,
                            replay_outcome.side_effects.join(", ")
                        )
                    } else {
                        replay_outcome.detail.clone()
                    },
                    staging_dir: replay_outcome.staging_dir.clone(),
                    side_effect_detected: Some(side_effect_detected),
                });
                let actual_sha = sha256_file(Path::new(path))
                    .with_context(|| format!("compute local sha256 for {}", path))?;
                let hash_status = if expected_sha.is_empty() {
                    "pass_without_expected_hash"
                } else if actual_sha == expected_sha {
                    "pass"
                } else {
                    "checksum_mismatch"
                };
                let hash_detail = if hash_status == "checksum_mismatch" {
                    format!(
                        "local hash mismatch (expected {}, got {})",
                        expected_sha, actual_sha
                    )
                } else {
                    "local hash verification passed".to_string()
                };

                let replay_effective_ok = replay_outcome.ok || replay_path_confirmed;
                let (status, detail) = if side_effect_detected && args.fail_on_replay_side_effects {
                    (
                        "replay_side_effect".to_string(),
                        format!(
                            "replay command mutated out-of-scope files: {}",
                            replay_outcome.side_effects.join(", ")
                        ),
                    )
                } else if replay_effective_ok {
                    (
                        hash_status.to_string(),
                        if replay_outcome.ok {
                            format!("generator replay command succeeded; {hash_detail}")
                        } else {
                            format!(
                                "generator replay command exited non-zero but confirmed this path in output; {hash_detail}"
                            )
                        },
                    )
                } else if args.require_replay_success {
                    (
                        "download_failed".to_string(),
                        format!(
                            "generator replay command failed with require_replay_success=true; {}; {hash_detail}",
                            replay_outcome.detail
                        ),
                    )
                } else {
                    (
                        hash_status.to_string(),
                        format!(
                            "generator replay command failed; using local hash fallback; {hash_detail}"
                        ),
                    )
                };
                records.push(FileAuditRecord {
                    path: path.clone(),
                    source_id: source.id.clone(),
                    source_status: source.status.clone(),
                    blocked_action_plan: source.blocked_action_plan.clone(),
                    expected_sha256: expected_sha,
                    actual_sha256: actual_sha,
                    status: status.to_string(),
                    detail,
                    attempts,
                });
                continue;
            }
            records.push(FileAuditRecord {
                path: path.clone(),
                source_id: source.id.clone(),
                source_status: source.status.clone(),
                blocked_action_plan: source.blocked_action_plan.clone(),
                expected_sha256: expected_sha,
                actual_sha256: "".to_string(),
                status: "generator_replay_required".to_string(),
                detail: format!(
                    "source rule {} uses non-single-file path_glob {}; replay via {}",
                    source.id, source.path_glob, source.retrieval_method
                ),
                attempts,
            });
            continue;
        }

        let urls = source_urls(source);
        if urls.is_empty() {
            records.push(FileAuditRecord {
                path: path.clone(),
                source_id: source.id.clone(),
                source_status: source.status.clone(),
                blocked_action_plan: source.blocked_action_plan.clone(),
                expected_sha256: expected_sha,
                actual_sha256: "".to_string(),
                status: "download_failed".to_string(),
                detail: "no URL candidates in source rule".to_string(),
                attempts,
            });
            continue;
        }

        let mut tmp_path = std::env::temp_dir();
        tmp_path.push(format!(
            "gororoba_external_redownload_{}_{}",
            source.id,
            sanitize_rel_path(path)
        ));

        let mut downloaded = false;
        for url in &urls {
            for backend in &backend_order {
                let result = fetch_once(backend, url, &tmp_path, args.timeout_seconds);
                match result {
                    Ok(()) => {
                        downloaded = true;
                        attempts.push(AttemptRecord {
                            backend: backend.clone(),
                            url: url.clone(),
                            status: "ok".to_string(),
                            detail: "download succeeded".to_string(),
                            staging_dir: None,
                            side_effect_detected: None,
                        });
                        break;
                    }
                    Err(err) => {
                        attempts.push(AttemptRecord {
                            backend: backend.clone(),
                            url: url.clone(),
                            status: "error".to_string(),
                            detail: err,
                            staging_dir: None,
                            side_effect_detected: None,
                        });
                    }
                }
            }
            if downloaded {
                break;
            }
        }

        if !downloaded {
            records.push(FileAuditRecord {
                path: path.clone(),
                source_id: source.id.clone(),
                source_status: source.status.clone(),
                blocked_action_plan: source.blocked_action_plan.clone(),
                expected_sha256: expected_sha,
                actual_sha256: "".to_string(),
                status: "download_failed".to_string(),
                detail: "all backends and URLs exhausted".to_string(),
                attempts,
            });
            continue;
        }

        let actual_sha = sha256_file(&tmp_path)
            .with_context(|| format!("compute sha256 for {}", tmp_path.display()))?;
        let status = if expected_sha.is_empty() {
            "pass_without_expected_hash"
        } else if actual_sha == expected_sha {
            "pass"
        } else {
            "checksum_mismatch"
        };
        let detail = if status == "checksum_mismatch" {
            format!("expected {}, got {}", expected_sha, actual_sha)
        } else {
            "hash verification passed".to_string()
        };

        records.push(FileAuditRecord {
            path: path.clone(),
            source_id: source.id.clone(),
            source_status: source.status.clone(),
            blocked_action_plan: source.blocked_action_plan.clone(),
            expected_sha256: expected_sha,
            actual_sha256: actual_sha,
            status: status.to_string(),
            detail,
            attempts,
        });

        std::fs::remove_file(&tmp_path).ok();
    }

    let blocked_issues = blocked_source_deadline_issues(&sources, now);
    let blocked_overdue = blocked_issues
        .iter()
        .filter(|item| item.contains("overdue"))
        .count();
    let blocked_missing_action_plan = blocked_issues
        .iter()
        .filter(|item| item.contains("without blocked_action_plan"))
        .count();
    let blocked_missing_action_plan_refs = blocked_issues
        .iter()
        .filter(|item| item.contains("missing blocked_action_plan ref"))
        .count();

    let mut status_counts: BTreeMap<String, usize> = BTreeMap::new();
    for record in &records {
        *status_counts.entry(record.status.clone()).or_insert(0) += 1;
    }

    let report = AuditReport {
        generated_at_utc: now.to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
        execute: args.execute,
        replay_mode: match args.replay_mode {
            ReplayMode::Live => "live".to_string(),
            ReplayMode::Staging => "staging".to_string(),
        },
        replay_staging_root: to_repo_rel(&args.staging_root)
            .unwrap_or_else(|_| args.staging_root.to_string_lossy().replace('\\', "/")),
        backend_order,
        total_files: records.len(),
        matched_source_files: records.iter().filter(|r| !r.source_id.is_empty()).count(),
        unmatched_source_files: records
            .iter()
            .filter(|r| r.status == "missing_source")
            .count(),
        blocked_pending: records
            .iter()
            .filter(|r| r.status == "blocked_pending")
            .count(),
        generator_replay_required: records
            .iter()
            .filter(|r| r.status == "generator_replay_required")
            .count(),
        replay_command_successes: records
            .iter()
            .flat_map(|r| r.attempts.iter())
            .filter(|a| {
                a.backend == "replay_command" && (a.status == "ok" || a.status == "ok_path_scoped")
            })
            .count(),
        replay_command_failures: records
            .iter()
            .flat_map(|r| r.attempts.iter())
            .filter(|a| {
                a.backend == "replay_command"
                    && (a.status == "error" || a.status == "replay_side_effect")
            })
            .count(),
        replay_side_effects_count: records
            .iter()
            .flat_map(|r| r.attempts.iter())
            .filter(|a| a.backend == "replay_command" && a.side_effect_detected == Some(true))
            .count(),
        blocked_overdue,
        blocked_missing_action_plan,
        blocked_missing_action_plan_refs,
        checksum_mismatch: *status_counts.get("checksum_mismatch").unwrap_or(&0),
        download_failures: *status_counts.get("download_failed").unwrap_or(&0),
        pass_count: records
            .iter()
            .filter(|r| r.status == "pass" || r.status == "pass_without_expected_hash")
            .count(),
        records,
    };

    println!("EXTERNAL_REDOWNLOAD_AUDIT");
    println!("  total_files={}", report.total_files);
    println!("  unmatched_source_files={}", report.unmatched_source_files);
    println!("  blocked_pending={}", report.blocked_pending);
    println!(
        "  generator_replay_required={}",
        report.generator_replay_required
    );
    println!(
        "  replay_command_successes={}",
        report.replay_command_successes
    );
    println!(
        "  replay_command_failures={}",
        report.replay_command_failures
    );
    println!(
        "  replay_side_effects_count={}",
        report.replay_side_effects_count
    );
    println!("  blocked_overdue={}", report.blocked_overdue);
    println!(
        "  blocked_missing_action_plan={}",
        report.blocked_missing_action_plan
    );
    println!(
        "  blocked_missing_action_plan_refs={}",
        report.blocked_missing_action_plan_refs
    );
    println!("  checksum_mismatch={}", report.checksum_mismatch);
    println!("  download_failures={}", report.download_failures);
    println!("  pass_count={}", report.pass_count);

    for issue in &blocked_issues {
        println!("BLOCKED_POLICY {issue}");
    }

    if let Some(out) = &args.out {
        let body = if out.extension().and_then(|s| s.to_str()) == Some("json") {
            serde_json::to_string_pretty(&report).context("serialize JSON report")?
        } else {
            toml::to_string_pretty(&report).context("serialize TOML report")?
        };
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create output directory {}", parent.display()))?;
        }
        std::fs::write(out, body + "\n").with_context(|| format!("write {}", out.display()))?;
        println!("WROTE {}", out.display());
    }

    let mut failures = Vec::new();
    if args.fail_on_unmatched_source && report.unmatched_source_files > 0 {
        failures.push(format!(
            "{} external file(s) are missing source rules",
            report.unmatched_source_files
        ));
    }
    if args.fail_on_blocked_overdue && report.blocked_overdue > 0 {
        failures.push(format!(
            "{} blocked source rule(s) exceeded resolution deadline",
            report.blocked_overdue
        ));
    }
    if args.fail_on_missing_action_plan && report.blocked_missing_action_plan > 0 {
        failures.push(format!(
            "{} blocked source rule(s) are missing blocked_action_plan entries",
            report.blocked_missing_action_plan
        ));
    }
    if args.fail_on_missing_action_plan && report.blocked_missing_action_plan_refs > 0 {
        failures.push(format!(
            "{} blocked_action_plan reference(s) are missing on disk",
            report.blocked_missing_action_plan_refs
        ));
    }
    if report.checksum_mismatch > 0 {
        failures.push(format!(
            "{} file(s) failed checksum verification",
            report.checksum_mismatch
        ));
    }
    if args.execute && report.download_failures > 0 {
        failures.push(format!(
            "{} file(s) could not be re-downloaded",
            report.download_failures
        ));
    }
    if args.execute && args.require_replay_success && report.replay_command_failures > 0 {
        failures.push(format!(
            "{} replay command attempt(s) failed under strict replay policy",
            report.replay_command_failures
        ));
    }
    if args.execute && args.fail_on_replay_side_effects && report.replay_side_effects_count > 0 {
        failures.push(format!(
            "{} replay command attempt(s) caused out-of-scope side effects",
            report.replay_side_effects_count
        ));
    }

    if !failures.is_empty() {
        for failure in failures {
            eprintln!("ERROR: {failure}");
        }
        anyhow::bail!("external re-download audit failed");
    }

    Ok(())
}

fn source_urls(source: &ExternalSourceRule) -> Vec<String> {
    let mut out = Vec::new();
    if is_http_url(&source.canonical_url) {
        out.push(source.canonical_url.clone());
    }
    for url in &source.mirror_urls {
        if is_http_url(url) && !out.contains(url) {
            out.push(url.clone());
        }
    }
    out
}

fn is_http_url(url: &str) -> bool {
    url.starts_with("http://") || url.starts_with("https://")
}

fn sanitize_rel_path(path: &str) -> String {
    path.replace(['/', '.'], "_")
}

fn is_single_file_rule(source: &ExternalSourceRule) -> bool {
    !source.path_glob.contains('*')
        && !source.path_glob.contains('?')
        && !source.path_glob.contains('[')
}

fn fetch_once(
    backend: &str,
    url: &str,
    output_path: &Path,
    timeout_seconds: u64,
) -> Result<(), String> {
    match backend {
        "wget" => fetch_with_wget(url, output_path, timeout_seconds),
        "curl" => fetch_with_curl(url, output_path, timeout_seconds),
        "fetch" => fetch_with_rust(url, output_path),
        other => Err(format!("unsupported backend {other}")),
    }
}

fn run_replay_command(
    source: &ExternalSourceRule,
    args: &Args,
    metadata_paths: &[String],
) -> ReplayCommandOutcome {
    let timeout_window = format!("{}s", args.timeout_seconds);
    let mut replay_command = source.retrieval_method.clone();
    let mut staging_dir = None;
    if args.replay_mode == ReplayMode::Staging {
        let per_source_dir = args.staging_root.join(source.id.to_lowercase());
        if let Err(err) = std::fs::create_dir_all(&per_source_dir) {
            return ReplayCommandOutcome {
                ok: false,
                detail: format!(
                    "failed to create staging dir {}: {err}",
                    per_source_dir.display()
                ),
                staging_dir: Some(per_source_dir.to_string_lossy().replace('\\', "/")),
                side_effects: Vec::new(),
            };
        }
        if let Err(err) = seed_staging_inputs(source, args, &per_source_dir) {
            return ReplayCommandOutcome {
                ok: false,
                detail: format!("failed to seed staging inputs for {}: {err}", source.id),
                staging_dir: Some(per_source_dir.to_string_lossy().replace('\\', "/")),
                side_effects: Vec::new(),
            };
        }
        replay_command = match rewrite_replay_command_for_staging(
            &source.retrieval_method,
            &per_source_dir,
            args,
        ) {
            Ok(v) => v,
            Err(err) => {
                return ReplayCommandOutcome {
                    ok: false,
                    detail: format!("staging rewrite failed for {}: {err}", source.id),
                    staging_dir: Some(per_source_dir.to_string_lossy().replace('\\', "/")),
                    side_effects: Vec::new(),
                };
            }
        };
        staging_dir = Some(per_source_dir.to_string_lossy().replace('\\', "/"));
    }

    let before = snapshot_root_hashes(&args.root).unwrap_or_default();
    let output = Command::new("timeout")
        .arg(&timeout_window)
        .arg("sh")
        .arg("-lc")
        .arg(&replay_command)
        .output()
        .or_else(|_| Command::new("sh").arg("-lc").arg(&replay_command).output());
    let after = snapshot_root_hashes(&args.root).unwrap_or_default();
    let side_effects = compute_replay_side_effects(source, &before, &after, metadata_paths);

    match output {
        Ok(output) if output.status.success() => ReplayCommandOutcome {
            ok: true,
            detail: format!(
                "{}\nreplay_command={}",
                combine_process_output(&output),
                replay_command
            ),
            staging_dir,
            side_effects,
        },
        Ok(output) => {
            let timed_out = output.status.code() == Some(124) || output.status.code() == Some(137);
            let output_text = combine_process_output(&output);
            ReplayCommandOutcome {
                ok: false,
                detail: if timed_out {
                    format!(
                        "replay command timed out after {}s\n{}\nreplay_command={}",
                        args.timeout_seconds, output_text, replay_command
                    )
                } else {
                    format!("{output_text}\nreplay_command={replay_command}")
                },
                staging_dir,
                side_effects,
            }
        }
        Err(err) => ReplayCommandOutcome {
            ok: false,
            detail: format!(
                "failed to spawn replay command: {err}\nreplay_command={}",
                replay_command
            ),
            staging_dir,
            side_effects,
        },
    }
}

fn replay_metadata_paths(root: &Path) -> Result<Vec<String>> {
    let rel_root = to_repo_rel(root)?;
    Ok(vec![
        format!("{rel_root}/README.md"),
        format!("{rel_root}/SOURCES.toml"),
        format!("{rel_root}/PROVENANCE.local.json"),
    ])
}

fn rewrite_replay_command_for_staging(
    retrieval_method: &str,
    staging_dir: &Path,
    args: &Args,
) -> Result<String> {
    let staging_quoted = shell_quote(&staging_dir.to_string_lossy());
    if retrieval_method.contains("--bin fetch-datasets") {
        let mut rewritten = strip_option_and_value(retrieval_method, "output-dir");
        rewritten = strip_option_optional_value(&rewritten, "skip-existing");
        rewritten.push_str(&format!(" --output-dir {staging_quoted}"));
        if args.force_refresh_replay {
            rewritten.push_str(" --skip-existing=false");
        }
        return Ok(rewritten);
    }
    if retrieval_method.contains("--bin hepdata-refresh") {
        let mut rewritten = strip_option_and_value(retrieval_method, "root");
        rewritten.push_str(&format!(" --root {staging_quoted}"));
        return Ok(rewritten);
    }
    anyhow::bail!("unsupported replay command for staging mode: {retrieval_method}");
}

fn shell_quote(raw: &str) -> String {
    let escaped = raw.replace('\'', "'\"'\"'");
    format!("'{escaped}'")
}

fn strip_option_and_value(command: &str, option: &str) -> String {
    let pattern =
        format!(r#"(?x)\s+--{option}(?:=(?:'[^']*'|"[^"]*"|\S+)|\s+(?:'[^']*'|"[^"]*"|\S+))"#);
    let matcher = Regex::new(&pattern).expect("valid option+value regex");
    matcher.replace_all(command, "").trim().to_string()
}

fn strip_option_optional_value(command: &str, option: &str) -> String {
    let pattern = format!(r#"(?x)\s+--{option}(?:=(?:'[^']*'|"[^"]*"|\S+))?"#);
    let matcher = Regex::new(&pattern).expect("valid option optional-value regex");
    matcher.replace_all(command, "").trim().to_string()
}

fn snapshot_root_hashes(root: &Path) -> Result<HashMap<String, String>> {
    let mut files = collect_files_under(root)?;
    files.sort();
    let mut out = HashMap::new();
    for rel in files {
        let digest = sha256_file(Path::new(&rel))
            .with_context(|| format!("compute snapshot hash for {rel}"))?;
        out.insert(rel, digest);
    }
    Ok(out)
}

fn compute_replay_side_effects(
    source: &ExternalSourceRule,
    before: &HashMap<String, String>,
    after: &HashMap<String, String>,
    metadata_paths: &[String],
) -> Vec<String> {
    let metadata: HashSet<&str> = metadata_paths.iter().map(String::as_str).collect();
    let mut all_paths: BTreeSet<String> = BTreeSet::new();
    all_paths.extend(before.keys().cloned());
    all_paths.extend(after.keys().cloned());

    let mut out = Vec::new();
    for path in all_paths {
        if metadata.contains(path.as_str()) {
            continue;
        }
        if source_glob_matches(&source.path_glob, &path) {
            continue;
        }
        let before_hash = before.get(&path);
        let after_hash = after.get(&path);
        if before_hash == after_hash {
            continue;
        }
        match (before_hash, after_hash) {
            (None, Some(_)) => out.push(format!("created:{path}")),
            (Some(_), None) => out.push(format!("deleted:{path}")),
            (Some(_), Some(_)) => out.push(format!("modified:{path}")),
            (None, None) => {}
        }
    }
    out
}

fn source_glob_matches(path_glob: &str, path: &str) -> bool {
    Pattern::new(path_glob)
        .map(|pattern| pattern.matches(path))
        .unwrap_or(false)
}

fn seed_staging_inputs(
    source: &ExternalSourceRule,
    args: &Args,
    staging_dir: &Path,
) -> Result<usize> {
    let root_rel = to_repo_rel(&args.root)?;
    let root_prefix = root_rel + "/";
    let mut seeded = 0usize;
    let files = collect_files_under(&args.root)?;
    for rel in files {
        if !source_glob_matches(&source.path_glob, &rel) {
            continue;
        }
        let suffix = rel.strip_prefix(&root_prefix).unwrap_or(rel.as_str());
        let dest = staging_dir.join(suffix);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create staging parent {}", parent.display()))?;
        }
        std::fs::copy(&rel, &dest)
            .with_context(|| format!("seed staging file {} -> {}", rel, dest.display()))?;
        seeded += 1;
    }
    Ok(seeded)
}

fn combine_process_output(output: &std::process::Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    match (stderr.is_empty(), stdout.is_empty()) {
        (true, true) => "replay command completed without output".to_string(),
        (false, true) => stderr,
        (true, false) => stdout,
        (false, false) => format!("{stderr}\n{stdout}"),
    }
}

fn replay_confirms_path(outcome: &ReplayCommandOutcome, path: &str) -> bool {
    if outcome.ok {
        return true;
    }
    let exact = format!("OK: {path}");
    if outcome.detail.contains(&exact) {
        return true;
    }
    Path::new(path)
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| outcome.detail.contains(&format!("OK: {name}")))
        .unwrap_or(false)
}

fn fetch_with_wget(url: &str, output_path: &Path, timeout_seconds: u64) -> Result<(), String> {
    let mut command = Command::new("wget");
    let output = command
        .arg("-q")
        .arg("-O")
        .arg(output_path)
        .arg("--timeout")
        .arg(timeout_seconds.to_string())
        .arg(url)
        .output()
        .map_err(|err| format!("wget unavailable or failed to start: {err}"))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).trim().to_string())
    }
}

fn fetch_with_curl(url: &str, output_path: &Path, timeout_seconds: u64) -> Result<(), String> {
    let mut command = Command::new("curl");
    let output = command
        .arg("-fL")
        .arg("--retry")
        .arg("2")
        .arg("--connect-timeout")
        .arg(timeout_seconds.to_string())
        .arg("-o")
        .arg(output_path)
        .arg(url)
        .output()
        .map_err(|err| format!("curl unavailable or failed to start: {err}"))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).trim().to_string())
    }
}

fn fetch_with_rust(url: &str, output_path: &Path) -> Result<(), String> {
    data_core::fetcher::download_to_file(url, output_path)
        .map(|_| ())
        .map_err(|err| err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_rule(path_glob: &str) -> ExternalSourceRule {
        ExternalSourceRule {
            id: "SRC-TEST".to_string(),
            path_glob: path_glob.to_string(),
            canonical_url: String::new(),
            mirror_urls: Vec::new(),
            access_class: "public".to_string(),
            status: "active".to_string(),
            retrieval_method: String::new(),
            attempt_deadline_utc: String::new(),
            resolution_deadline_utc: String::new(),
            blocker_note: String::new(),
            evidence_refs: Vec::new(),
            manual_manifest_refs: Vec::new(),
            blocked_action_plan: Vec::new(),
            scientific_validator_refs: Vec::new(),
        }
    }

    #[test]
    fn rewrite_fetch_datasets_for_staging() {
        let args = Args {
            root: PathBuf::from("data/external"),
            sources: PathBuf::from(DEFAULT_EXTERNAL_SOURCES_PATH),
            provenance: PathBuf::from(DEFAULT_EXTERNAL_PROVENANCE_PATH),
            out: None,
            execute: true,
            replay_mode: ReplayMode::Staging,
            staging_root: PathBuf::from("target/external_replay_staging"),
            fail_on_replay_side_effects: true,
            force_refresh_replay: true,
            require_replay_success: true,
            backend_order: "wget,curl,fetch".to_string(),
            timeout_seconds: 600,
            max_files: None,
            fail_on_unmatched_source: true,
            fail_on_blocked_overdue: true,
            fail_on_missing_action_plan: true,
        };
        let command = "cargo run -p gororoba_cli --bin fetch-datasets -- --all --skip-existing";
        let rewritten =
            rewrite_replay_command_for_staging(command, Path::new("target/staging"), &args)
                .expect("rewrite");
        assert!(rewritten.contains("--output-dir 'target/staging'"));
        assert!(rewritten.contains("--skip-existing=false"));
        assert_eq!(rewritten.matches("--skip-existing").count(), 1);
    }

    #[test]
    fn rewrite_hepdata_for_staging() {
        let args = Args {
            root: PathBuf::from("data/external"),
            sources: PathBuf::from(DEFAULT_EXTERNAL_SOURCES_PATH),
            provenance: PathBuf::from(DEFAULT_EXTERNAL_PROVENANCE_PATH),
            out: None,
            execute: true,
            replay_mode: ReplayMode::Staging,
            staging_root: PathBuf::from("target/external_replay_staging"),
            fail_on_replay_side_effects: true,
            force_refresh_replay: true,
            require_replay_success: true,
            backend_order: "wget,curl,fetch".to_string(),
            timeout_seconds: 600,
            max_files: None,
            fail_on_unmatched_source: true,
            fail_on_blocked_overdue: true,
            fail_on_missing_action_plan: true,
        };
        let command = "cargo run -p gororoba_cli --bin hepdata-refresh -- --dirs alice_pbpb_raa";
        let rewritten =
            rewrite_replay_command_for_staging(command, Path::new("target/staging"), &args)
                .expect("rewrite");
        assert!(rewritten.contains("--root 'target/staging'"));
    }

    #[test]
    fn replay_side_effect_detection_is_scoped() {
        let source = sample_rule("data/external/alice_pbpb_raa/**");
        let before = HashMap::new();
        let mut after = HashMap::new();
        after.insert(
            "data/external/alice_pbpb_raa/new.json".to_string(),
            "abc".to_string(),
        );
        after.insert("data/external/rogue.csv".to_string(), "def".to_string());
        let metadata = vec![
            "data/external/README.md".to_string(),
            "data/external/SOURCES.toml".to_string(),
            "data/external/PROVENANCE.local.json".to_string(),
        ];
        let side_effects = compute_replay_side_effects(&source, &before, &after, &metadata);
        assert_eq!(side_effects, vec!["created:data/external/rogue.csv"]);
    }
}

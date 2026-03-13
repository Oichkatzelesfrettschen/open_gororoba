use clap::Parser;
use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};
use tempfile::tempdir;
use walkdir::WalkDir;

#[derive(Debug, Parser)]
#[command(name = "verify-registry-mirror-freshness")]
#[command(about = "Verify TOML-driven markdown mirrors are fresh using Rust emitters")]
struct Cli {
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    canonical_db: PathBuf,
    #[arg(long, default_value = "docs/generated")]
    out_dir: PathBuf,
    #[arg(long, default_value_t = false)]
    emit_legacy: bool,
    #[arg(long, action = clap::ArgAction::Set, default_value_t = true)]
    legacy_claims_sync: bool,
}

struct Job {
    command: &'static str,
    args: Vec<String>,
}

fn emitter_bin() -> Result<PathBuf, String> {
    let current = env::current_exe().map_err(|err| format!("current_exe: {}", err))?;
    let sibling = current.with_file_name("registry-emit");
    if sibling.exists() {
        return Ok(sibling);
    }
    Err(format!(
        "could not locate registry-emit next to {}",
        current.display()
    ))
}

fn run_job(bin: &Path, repo_root: &Path, job: &Job) -> Result<(), String> {
    let status = Command::new(bin)
        .arg(job.command)
        .args(&job.args)
        .current_dir(repo_root)
        .status()
        .map_err(|err| format!("spawn {}: {}", job.command, err))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{} failed with status {}", job.command, status))
    }
}

fn mirror_jobs(stage_root: &Path, out_dir: &Path, canonical_db: &Path) -> Vec<Job> {
    let stage_out = stage_root.join(out_dir);
    vec![
        Job {
            command: "insights-mirror",
            args: vec![
                "--canonical-db".into(),
                canonical_db.display().to_string(),
                "--output".into(),
                stage_out
                    .join("INSIGHTS_REGISTRY_MIRROR.md")
                    .display()
                    .to_string(),
            ],
        },
        Job {
            command: "claims-mirror",
            args: vec![
                "--canonical-db".into(),
                canonical_db.display().to_string(),
                "--output".into(),
                stage_out
                    .join("CLAIMS_REGISTRY_MIRROR.md")
                    .display()
                    .to_string(),
            ],
        },
        Job {
            command: "bibliography-mirror",
            args: vec![
                "--output".into(),
                stage_out
                    .join("BIBLIOGRAPHY_REGISTRY_MIRROR.md")
                    .display()
                    .to_string(),
            ],
        },
        Job {
            command: "experiments-mirror",
            args: vec![
                "--canonical-db".into(),
                canonical_db.display().to_string(),
                "--output".into(),
                stage_out
                    .join("EXPERIMENTS_REGISTRY_MIRROR.md")
                    .display()
                    .to_string(),
            ],
        },
        Job {
            command: "theorems-mirror",
            args: vec![
                "--canonical-db".into(),
                canonical_db.display().to_string(),
                "--output".into(),
                stage_out
                    .join("THEOREMS_REGISTRY_MIRROR.md")
                    .display()
                    .to_string(),
            ],
        },
        Job {
            command: "roadmap-mirror",
            args: vec![
                "--output".into(),
                stage_out
                    .join("ROADMAP_REGISTRY_MIRROR.md")
                    .display()
                    .to_string(),
            ],
        },
        Job {
            command: "todo-mirror",
            args: vec![
                "--output".into(),
                stage_out
                    .join("TODO_REGISTRY_MIRROR.md")
                    .display()
                    .to_string(),
            ],
        },
        Job {
            command: "next-actions-mirror",
            args: vec![
                "--output".into(),
                stage_out
                    .join("NEXT_ACTIONS_REGISTRY_MIRROR.md")
                    .display()
                    .to_string(),
            ],
        },
        Job {
            command: "requirements-mirror",
            args: vec![
                "--output".into(),
                stage_out
                    .join("REQUIREMENTS_REGISTRY_MIRROR.md")
                    .display()
                    .to_string(),
            ],
        },
        Job {
            command: "knowledge-migration-plan-mirror",
            args: vec![
                "--output".into(),
                stage_out
                    .join("KNOWLEDGE_MIGRATION_PLAN_REGISTRY_MIRROR.md")
                    .display()
                    .to_string(),
            ],
        },
        Job {
            command: "navigator-mirror",
            args: vec![
                "--output".into(),
                stage_out
                    .join("NAVIGATOR_REGISTRY_MIRROR.md")
                    .display()
                    .to_string(),
            ],
        },
        Job {
            command: "entrypoint-docs-mirror",
            args: vec![
                "--output".into(),
                stage_out
                    .join("ENTRYPOINT_DOCS_REGISTRY_MIRROR.md")
                    .display()
                    .to_string(),
            ],
        },
        Job {
            command: "markdown-governance-mirror",
            args: vec![
                "--output".into(),
                stage_out
                    .join("MARKDOWN_GOVERNANCE_REGISTRY_MIRROR.md")
                    .display()
                    .to_string(),
            ],
        },
        Job {
            command: "claims-tasks-mirror",
            args: vec![
                "--output".into(),
                stage_out
                    .join("CLAIMS_TASKS_REGISTRY_MIRROR.md")
                    .display()
                    .to_string(),
            ],
        },
        Job {
            command: "claims-domains-mirror",
            args: vec![
                "--output".into(),
                stage_out
                    .join("CLAIMS_DOMAINS_REGISTRY_MIRROR.md")
                    .display()
                    .to_string(),
            ],
        },
        Job {
            command: "claim-tickets-mirror",
            args: vec![
                "--output".into(),
                stage_out
                    .join("CLAIM_TICKETS_REGISTRY_MIRROR.md")
                    .display()
                    .to_string(),
            ],
        },
        Job {
            command: "external-sources-mirror",
            args: vec![
                "--output".into(),
                stage_out
                    .join("EXTERNAL_SOURCES_REGISTRY_MIRROR.md")
                    .display()
                    .to_string(),
            ],
        },
        Job {
            command: "research-narratives-mirror",
            args: vec![
                "--output".into(),
                stage_out
                    .join("RESEARCH_NARRATIVES_REGISTRY_MIRROR.md")
                    .display()
                    .to_string(),
            ],
        },
        Job {
            command: "book-docs-mirror",
            args: vec![
                "--output".into(),
                stage_out
                    .join("BOOK_DOCS_REGISTRY_MIRROR.md")
                    .display()
                    .to_string(),
            ],
        },
        Job {
            command: "docs-root-narratives-mirror",
            args: vec![
                "--output".into(),
                stage_out
                    .join("DOCS_ROOT_NARRATIVES_REGISTRY_MIRROR.md")
                    .display()
                    .to_string(),
            ],
        },
        Job {
            command: "data-artifact-narratives-mirror",
            args: vec![
                "--output".into(),
                stage_out
                    .join("DATA_ARTIFACT_NARRATIVES_REGISTRY_MIRROR.md")
                    .display()
                    .to_string(),
            ],
        },
        Job {
            command: "reports-narratives-mirror",
            args: vec![
                "--output".into(),
                stage_out
                    .join("REPORTS_NARRATIVES_REGISTRY_MIRROR.md")
                    .display()
                    .to_string(),
            ],
        },
        Job {
            command: "docs-convos-mirror",
            args: vec![
                "--output".into(),
                stage_out
                    .join("DOCS_CONVOS_REGISTRY_MIRROR.md")
                    .display()
                    .to_string(),
            ],
        },
    ]
}

fn base_legacy_jobs(stage_root: &Path) -> Vec<Job> {
    vec![
        Job {
            command: "insights-legacy",
            args: vec![
                "--output".into(),
                stage_root.join("docs/INSIGHTS.md").display().to_string(),
            ],
        },
        Job {
            command: "experiments-legacy",
            args: vec![
                "--output".into(),
                stage_root
                    .join("docs/EXPERIMENTS_PORTFOLIO_SHORTLIST.md")
                    .display()
                    .to_string(),
            ],
        },
        Job {
            command: "theorems-legacy",
            args: vec![
                "--output".into(),
                stage_root.join("docs/THEOREMS.md").display().to_string(),
            ],
        },
        Job {
            command: "roadmap-legacy",
            args: vec![
                "--output".into(),
                stage_root.join("docs/ROADMAP.md").display().to_string(),
            ],
        },
        Job {
            command: "todo-legacy",
            args: vec![
                "--output".into(),
                stage_root.join("docs/TODO.md").display().to_string(),
            ],
        },
        Job {
            command: "next-actions-legacy",
            args: vec![
                "--output".into(),
                stage_root
                    .join("docs/NEXT_ACTIONS.md")
                    .display()
                    .to_string(),
            ],
        },
        Job {
            command: "requirements-legacy",
            args: vec!["--repo-root".into(), stage_root.display().to_string()],
        },
        Job {
            command: "navigator-legacy",
            args: vec![
                "--output".into(),
                stage_root.join("NAVIGATOR.md").display().to_string(),
            ],
        },
        Job {
            command: "entrypoint-docs-legacy",
            args: vec!["--repo-root".into(), stage_root.display().to_string()],
        },
        Job {
            command: "bibliography-legacy",
            args: vec![
                "--output".into(),
                stage_root
                    .join("docs/BIBLIOGRAPHY.md")
                    .display()
                    .to_string(),
            ],
        },
    ]
}

fn claims_sync_legacy_jobs(stage_root: &Path, repo_root: &Path) -> Vec<Job> {
    vec![
        Job {
            command: "claims-matrix-legacy",
            args: vec![
                "--output".into(),
                stage_root
                    .join("docs/CLAIMS_EVIDENCE_MATRIX.md")
                    .display()
                    .to_string(),
            ],
        },
        Job {
            command: "claims-tasks-legacy",
            args: vec![
                "--output".into(),
                stage_root
                    .join("docs/CLAIMS_TASKS.md")
                    .display()
                    .to_string(),
            ],
        },
        Job {
            command: "claims-domains-legacy",
            args: vec!["--repo-root".into(), stage_root.display().to_string()],
        },
        Job {
            command: "claim-tickets-legacy",
            args: vec!["--repo-root".into(), stage_root.display().to_string()],
        },
        Job {
            command: "external-sources-legacy",
            args: vec!["--repo-root".into(), stage_root.display().to_string()],
        },
        Job {
            command: "research-narratives-legacy",
            args: vec!["--repo-root".into(), stage_root.display().to_string()],
        },
        Job {
            command: "docs-root-narratives-legacy",
            args: vec!["--repo-root".into(), stage_root.display().to_string()],
        },
        Job {
            command: "data-artifact-narratives-legacy",
            args: vec![
                "--repo-root".into(),
                stage_root.display().to_string(),
                "--artifact-index".into(),
                repo_root
                    .join("registry/artifact_scrolls.toml")
                    .display()
                    .to_string(),
            ],
        },
        Job {
            command: "monograph-legacy",
            args: vec!["--repo-root".into(), stage_root.display().to_string()],
        },
        Job {
            command: "reports-narratives-legacy",
            args: vec!["--repo-root".into(), stage_root.display().to_string()],
        },
        Job {
            command: "docs-convos-legacy",
            args: vec!["--repo-root".into(), stage_root.display().to_string()],
        },
    ]
}

fn collect_stage_files(stage_root: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    for entry in WalkDir::new(stage_root) {
        let entry = entry.map_err(|err| format!("walk {}: {}", stage_root.display(), err))?;
        if entry.file_type().is_file() {
            let relative = entry
                .path()
                .strip_prefix(stage_root)
                .map_err(|err| format!("strip prefix {}: {}", entry.path().display(), err))?;
            files.push(relative.to_path_buf());
        }
    }
    files.sort();
    Ok(files)
}

fn compare_stage_to_repo(stage_root: &Path, repo_root: &Path) -> Result<Vec<PathBuf>, String> {
    let mut stale = Vec::new();
    for relative in collect_stage_files(stage_root)? {
        let staged = stage_root.join(&relative);
        let actual = repo_root.join(&relative);
        let staged_bytes =
            fs::read(&staged).map_err(|err| format!("read {}: {}", staged.display(), err))?;
        let actual_bytes = fs::read(&actual).unwrap_or_default();
        if staged_bytes != actual_bytes {
            stale.push(relative);
        }
    }
    Ok(stale)
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<(), String> {
    for entry in WalkDir::new(src) {
        let entry = entry.map_err(|err| format!("walk {}: {}", src.display(), err))?;
        let relative = entry
            .path()
            .strip_prefix(src)
            .map_err(|err| format!("strip prefix {}: {}", entry.path().display(), err))?;
        let target = dst.join(relative);
        if entry.file_type().is_dir() {
            fs::create_dir_all(&target)
                .map_err(|err| format!("mkdir {}: {}", target.display(), err))?;
        } else {
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent)
                    .map_err(|err| format!("mkdir {}: {}", parent.display(), err))?;
            }
            fs::copy(entry.path(), &target).map_err(|err| {
                format!(
                    "copy {} -> {}: {}",
                    entry.path().display(),
                    target.display(),
                    err
                )
            })?;
        }
    }
    Ok(())
}

fn main() -> Result<(), String> {
    let cli = Cli::parse();
    let repo_root = cli
        .repo_root
        .canonicalize()
        .map_err(|err| format!("canonicalize {}: {}", cli.repo_root.display(), err))?;
    let stage_dir = tempdir().map_err(|err| format!("tempdir: {}", err))?;
    let stage_root = stage_dir.path();
    let bin = emitter_bin()?;

    for job in mirror_jobs(stage_root, &cli.out_dir, &cli.canonical_db) {
        run_job(&bin, &repo_root, &job)?;
    }

    if cli.emit_legacy {
        copy_dir_recursive(
            &repo_root.join("registry/knowledge/artifacts"),
            &stage_root.join("registry/knowledge/artifacts"),
        )?;
        for job in base_legacy_jobs(stage_root) {
            run_job(&bin, &repo_root, &job)?;
        }
        if cli.legacy_claims_sync {
            for job in claims_sync_legacy_jobs(stage_root, &repo_root) {
                run_job(&bin, &repo_root, &job)?;
            }
        }
    }

    let stale = compare_stage_to_repo(stage_root, &repo_root)?;
    if stale.is_empty() {
        println!("OK: generated registry/control-plane mirrors are fresh.");
        return Ok(());
    }

    eprintln!(
        "ERROR: generated registry/control-plane mirrors are stale. Regenerate with MARKDOWN_EXPORT=1 make registry-export-markdown."
    );
    for path in stale {
        eprintln!("{}", path.display());
    }
    Err("stale mirrors detected".to_string())
}

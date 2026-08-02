//! Verify that the active Rocq project lists every intended proof source.
//!
//! The Rocq project file is a build input, while the `theories/` and
//! `verified/` directories are the source inventory. This verifier compares
//! both directions so a source cannot silently fall outside the compiled
//! project and a project entry cannot point at a missing file.

use std::{
    collections::BTreeSet,
    fs,
    path::{Component, Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use clap::Parser;
use walkdir::WalkDir;

const DEFAULT_PROJECT_FILE: &str = "proofs/_RocqProject";

#[derive(Debug, Parser)]
#[command(
    name = "rocq-project-audit",
    about = "Verify Rocq project-file parity against the proof source inventory"
)]
struct Args {
    /// Repository root containing proofs/_RocqProject.
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,

    /// Active Rocq project file, resolved relative to --repo-root.
    #[arg(long, default_value = DEFAULT_PROJECT_FILE)]
    project_file: PathBuf,

    /// Intended proof source roots, resolved relative to --repo-root.
    #[arg(
        long = "source-root",
        value_name = "PATH",
        default_values = ["proofs/theories", "proofs/verified"]
    )]
    source_roots: Vec<PathBuf>,
}

#[derive(Debug, Eq, PartialEq)]
struct ProjectInventory {
    listed_sources: BTreeSet<String>,
    disk_sources: BTreeSet<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let repo_root = fs::canonicalize(&args.repo_root)
        .with_context(|| format!("resolve repository root {}", args.repo_root.display()))?;
    let project_file = resolve_repo_path(&repo_root, &args.project_file)?;
    let source_roots = args
        .source_roots
        .iter()
        .map(|path| resolve_repo_path(&repo_root, path))
        .collect::<Result<Vec<_>>>()?;
    let inventory = verify_project(&repo_root, &project_file, &source_roots)?;

    println!(
        "verified Rocq project parity: {} listed sources, {} inventory sources",
        inventory.listed_sources.len(),
        inventory.disk_sources.len()
    );
    Ok(())
}

fn resolve_repo_path(repo_root: &Path, path: &Path) -> Result<PathBuf> {
    let candidate = if path.is_absolute() {
        path.to_path_buf()
    } else {
        repo_root.join(path)
    };
    let resolved = fs::canonicalize(&candidate)
        .with_context(|| format!("resolve repository path {}", candidate.display()))?;
    if !resolved.starts_with(repo_root) {
        bail!(
            "path {} resolves outside repository root {}",
            candidate.display(),
            repo_root.display()
        );
    }
    Ok(resolved)
}

fn verify_project(
    repo_root: &Path,
    project_file: &Path,
    source_roots: &[PathBuf],
) -> Result<ProjectInventory> {
    let project_text = fs::read_to_string(project_file)
        .with_context(|| format!("read Rocq project file {}", project_file.display()))?;
    if !project_text.is_ascii() {
        bail!(
            "Rocq project file {} contains non-ASCII bytes",
            project_file.display()
        );
    }

    let listed_sources = parse_project_sources(repo_root, project_file, &project_text)?;
    let disk_sources = collect_source_inventory(repo_root, source_roots)?;

    let missing_from_project = disk_sources
        .difference(&listed_sources)
        .cloned()
        .collect::<Vec<_>>();
    let missing_from_disk = listed_sources
        .difference(&disk_sources)
        .cloned()
        .collect::<Vec<_>>();
    if !missing_from_project.is_empty() || !missing_from_disk.is_empty() {
        let mut message = String::from("Rocq project/source parity failed");
        if !missing_from_project.is_empty() {
            message.push_str("; source files missing from project: ");
            message.push_str(&missing_from_project.join(", "));
        }
        if !missing_from_disk.is_empty() {
            message.push_str("; project entries missing on disk: ");
            message.push_str(&missing_from_disk.join(", "));
        }
        bail!(message);
    }

    Ok(ProjectInventory {
        listed_sources,
        disk_sources,
    })
}

fn parse_project_sources(
    repo_root: &Path,
    project_file: &Path,
    project_text: &str,
) -> Result<BTreeSet<String>> {
    let project_dir = project_file
        .parent()
        .context("Rocq project file has no parent directory")?;
    let mut listed_sources = BTreeSet::new();

    for (line_index, raw_line) in project_text.lines().enumerate() {
        let line_number = line_index + 1;
        let trimmed = raw_line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with('-') {
            continue;
        }
        let entry = trimmed.split('#').next().unwrap_or_default().trim();
        if !entry.ends_with(".v") {
            bail!(
                "unsupported non-directive entry in {} at line {}: {}",
                project_file.display(),
                line_number,
                trimmed
            );
        }
        let entry_path = Path::new(entry);
        if entry_path.is_absolute()
            || entry_path
                .components()
                .any(|component| matches!(component, Component::ParentDir | Component::RootDir))
        {
            bail!(
                "Rocq project entry at {}:{} must remain relative to the project directory: {}",
                project_file.display(),
                line_number,
                entry
            );
        }

        let absolute_path = project_dir.join(entry_path);
        let resolved_path = fs::canonicalize(&absolute_path).with_context(|| {
            format!(
                "resolve Rocq project entry {} at line {}",
                entry, line_number
            )
        })?;
        if !resolved_path.starts_with(repo_root) {
            bail!(
                "Rocq project entry at {}:{} resolves outside repository root: {}",
                project_file.display(),
                line_number,
                entry
            );
        }
        if !resolved_path.is_file() {
            bail!(
                "Rocq project entry at {}:{} is not a regular file: {}",
                project_file.display(),
                line_number,
                entry
            );
        }
        let relative_path = repo_relative(repo_root, &resolved_path)?;
        if !listed_sources.insert(relative_path.clone()) {
            bail!(
                "duplicate Rocq project source at {}:{}: {}",
                project_file.display(),
                line_number,
                relative_path
            );
        }
    }

    if listed_sources.is_empty() {
        bail!(
            "Rocq project file {} lists no .v sources",
            project_file.display()
        );
    }
    Ok(listed_sources)
}

fn collect_source_inventory(
    repo_root: &Path,
    source_roots: &[PathBuf],
) -> Result<BTreeSet<String>> {
    if source_roots.is_empty() {
        bail!("at least one Rocq source root is required");
    }
    let mut disk_sources = BTreeSet::new();
    for source_root in source_roots {
        if !source_root.is_dir() {
            bail!(
                "Rocq source root is not a directory: {}",
                source_root.display()
            );
        }
        for entry in WalkDir::new(source_root).follow_links(false) {
            let entry = entry
                .with_context(|| format!("walk Rocq source root {}", source_root.display()))?;
            if !entry.file_type().is_file()
                || entry.path().extension().and_then(|ext| ext.to_str()) != Some("v")
            {
                continue;
            }
            let relative_path = repo_relative(repo_root, entry.path())?;
            if !disk_sources.insert(relative_path.clone()) {
                bail!("Rocq source roots overlap at {}", relative_path);
            }
        }
    }
    if disk_sources.is_empty() {
        bail!("Rocq source roots contain no .v files");
    }
    Ok(disk_sources)
}

fn repo_relative(repo_root: &Path, path: &Path) -> Result<String> {
    let relative = path
        .strip_prefix(repo_root)
        .with_context(|| format!("path {} is outside {}", path.display(), repo_root.display()))?;
    let relative = relative
        .to_str()
        .with_context(|| format!("path {} is not valid UTF-8", path.display()))?;
    if !relative.is_ascii() {
        bail!("path {} contains non-ASCII characters", path.display());
    }
    Ok(relative.replace('\\', "/"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn fixture(entries: &[&str]) -> (tempfile::TempDir, PathBuf, Vec<PathBuf>) {
        let directory = tempdir().expect("create fixture directory");
        let proof_root = directory.path().join("proofs");
        let theories = proof_root.join("theories");
        let verified = proof_root.join("verified");
        fs::create_dir_all(&theories).expect("create theories directory");
        fs::create_dir_all(&verified).expect("create verified directory");
        fs::write(
            theories.join("Prelude.v"),
            "Theorem prelude_ok : True.\nProof. exact I. Qed.\n",
        )
        .expect("write theory fixture");
        fs::write(
            verified.join("C-0001.v"),
            "Theorem claim_ok : True.\nProof. exact I. Qed.\n",
        )
        .expect("write verified fixture");
        let project_file = proof_root.join("_RocqProject");
        let project_body = entries.join("\n") + "\n";
        fs::write(&project_file, project_body).expect("write project fixture");
        (
            directory,
            project_file,
            vec![
                PathBuf::from("proofs/theories"),
                PathBuf::from("proofs/verified"),
            ],
        )
    }

    #[test]
    fn accepts_bidirectional_parity() {
        let (directory, project_file, source_roots) =
            fixture(&["theories/Prelude.v", "verified/C-0001.v"]);
        let inventory = verify_project(
            directory.path(),
            &project_file,
            &[
                directory.path().join(&source_roots[0]),
                directory.path().join(&source_roots[1]),
            ],
        )
        .expect("complete project should pass");
        assert_eq!(inventory.listed_sources.len(), 2);
        assert_eq!(inventory.listed_sources, inventory.disk_sources);
    }

    #[test]
    fn rejects_unlisted_source() {
        let (directory, project_file, source_roots) = fixture(&["theories/Prelude.v"]);
        let error = verify_project(
            directory.path(),
            &project_file,
            &[
                directory.path().join(&source_roots[0]),
                directory.path().join(&source_roots[1]),
            ],
        )
        .expect_err("unlisted source must fail parity");
        assert!(error.to_string().contains("verified/C-0001.v"));
    }

    #[test]
    fn rejects_duplicate_project_entry() {
        let (directory, project_file, source_roots) = fixture(&[
            "theories/Prelude.v",
            "theories/Prelude.v",
            "verified/C-0001.v",
        ]);
        let error = verify_project(
            directory.path(),
            &project_file,
            &[
                directory.path().join(&source_roots[0]),
                directory.path().join(&source_roots[1]),
            ],
        )
        .expect_err("duplicate source must fail parity");
        assert!(error.to_string().contains("duplicate Rocq project source"));
    }
}

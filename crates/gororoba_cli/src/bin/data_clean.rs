use anyhow::Result;
use clap::{Parser, ValueEnum};
use gororoba_cli::data_governance::{
    DEFAULT_GOVERNANCE_PATH, collect_files_under, git_ignored_paths, load_data_governance,
};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Scope {
    Reproducible,
    AllCleanable,
    Lane,
}

#[derive(Parser, Debug)]
#[command(
    name = "data-clean",
    about = "Rust-native cleanup for reproducible data lanes"
)]
struct Args {
    #[arg(long, default_value = DEFAULT_GOVERNANCE_PATH)]
    governance: PathBuf,
    #[arg(long, value_enum, default_value_t = Scope::Reproducible)]
    scope: Scope,
    #[arg(long)]
    lane: Option<String>,
    #[arg(long, default_value_t = true)]
    respect_gitignore: bool,
    #[arg(long)]
    apply: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let governance = load_data_governance(&args.governance)?;

    let lanes: Vec<_> = governance
        .lane
        .iter()
        .filter(|lane| lane.cleanable)
        .filter(|lane| match args.scope {
            Scope::AllCleanable => true,
            Scope::Reproducible => lane.lane_class == "generated_reproducible",
            Scope::Lane => args.lane.as_deref() == Some(lane.id.as_str()),
        })
        .collect();

    if matches!(args.scope, Scope::Lane) && lanes.is_empty() {
        anyhow::bail!(
            "requested --scope lane but no cleanable lane matched --lane {:?}",
            args.lane
        );
    }

    let mut candidate_files = Vec::new();
    for lane in &lanes {
        candidate_files.extend(collect_files_under(Path::new(&lane.root))?);
    }
    candidate_files.sort();
    candidate_files.dedup();

    let files_to_remove: Vec<String> = if args.respect_gitignore {
        let ignored = git_ignored_paths(&PathBuf::from("."), &candidate_files)?;
        candidate_files
            .into_iter()
            .filter(|path| ignored.contains(path))
            .collect()
    } else {
        candidate_files
    };

    println!("DATA_CLEAN");
    println!("  lanes={}", lanes.len());
    println!("  candidates={}", files_to_remove.len());
    println!("  mode={}", if args.apply { "apply" } else { "dry-run" });

    for path in files_to_remove.iter().take(50) {
        println!("  remove {path}");
    }
    if files_to_remove.len() > 50 {
        println!("  ... {} more", files_to_remove.len() - 50);
    }

    if !args.apply {
        return Ok(());
    }

    let mut removed_files = 0usize;
    let mut touched_dirs = BTreeSet::new();
    for rel in &files_to_remove {
        let path = PathBuf::from(rel);
        if path.exists() {
            std::fs::remove_file(&path)?;
            removed_files += 1;
            if let Some(parent) = path.parent() {
                touched_dirs.insert(parent.to_path_buf());
            }
        }
    }

    let mut dirs: Vec<PathBuf> = touched_dirs.into_iter().collect();
    dirs.sort_by_key(|p| std::cmp::Reverse(p.components().count()));
    for dir in dirs {
        remove_empty_chain(&dir)?;
    }

    println!("  removed_files={removed_files}");
    Ok(())
}

fn remove_empty_chain(start: &Path) -> Result<()> {
    let mut current = Some(start.to_path_buf());
    while let Some(path) = current {
        if !path.exists() || !path.is_dir() {
            break;
        }
        if path == Path::new("data") {
            break;
        }
        if std::fs::read_dir(&path)?.next().is_some() {
            break;
        }
        std::fs::remove_dir(&path)?;
        current = path.parent().map(Path::to_path_buf);
    }
    Ok(())
}

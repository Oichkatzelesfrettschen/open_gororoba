use clap::{Parser, ValueEnum};
use glob::glob;
use scrolls_core::{
    convert_csv_to_scroll, render_scroll_index, slugify, ConvertSpec, ScrollIndexEntry,
};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DatasetClassArg {
    CanonicalDataset,
    GeneratedArtifact,
    HoldingExternal,
    HoldingArchive,
}

impl DatasetClassArg {
    fn as_str(self) -> &'static str {
        match self {
            DatasetClassArg::CanonicalDataset => "canonical_dataset",
            DatasetClassArg::GeneratedArtifact => "generated_artifact",
            DatasetClassArg::HoldingExternal => "holding_external_csv",
            DatasetClassArg::HoldingArchive => "holding_archive_csv",
        }
    }
}

/// Convert CSV files into TOML scroll datasets and emit a canonical index.
#[derive(Debug, Parser)]
struct Args {
    /// Source glob for CSV files (alternative to --source-manifest).
    #[arg(long)]
    source_glob: Option<String>,
    /// Text manifest containing source CSV paths (one per line).
    #[arg(long)]
    source_manifest: Option<PathBuf>,
    /// Output index TOML path.
    #[arg(long)]
    out_index: PathBuf,
    /// Output directory for per-dataset TOML files.
    #[arg(long)]
    out_dir: PathBuf,
    /// Top-level index metadata table name.
    #[arg(long)]
    index_table: String,
    /// Dataset id prefix, e.g. PC/PG/AH/EH.
    #[arg(long)]
    dataset_prefix: String,
    /// Human-readable corpus label.
    #[arg(long)]
    corpus_label: String,
    /// Dataset class annotation.
    #[arg(long, value_enum, default_value_t = DatasetClassArg::CanonicalDataset)]
    dataset_class: DatasetClassArg,
}

fn load_manifest(path: &Path) -> Result<Vec<String>, String> {
    let text = fs::read_to_string(path).map_err(|err| format!("{}: {}", path.display(), err))?;
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_string)
        .collect())
}

fn collect_sources(args: &Args) -> Result<Vec<String>, String> {
    let mut paths: BTreeSet<String> = BTreeSet::new();
    if let Some(pattern) = &args.source_glob {
        let entries = glob(pattern).map_err(|err| format!("glob parse {}: {}", pattern, err))?;
        for entry in entries {
            let path = entry.map_err(|err| format!("glob match error: {}", err))?;
            paths.insert(path.to_string_lossy().to_string());
        }
    }
    if let Some(manifest) = &args.source_manifest {
        for item in load_manifest(manifest)? {
            paths.insert(item);
        }
    }
    if paths.is_empty() {
        return Err(
            "no input CSV files found; pass --source-glob and/or --source-manifest".to_string(),
        );
    }
    Ok(paths.into_iter().collect())
}

fn run(args: Args) -> Result<(), String> {
    let inputs = collect_sources(&args)?;
    fs::create_dir_all(&args.out_dir)
        .map_err(|err| format!("mkdir {}: {}", args.out_dir.display(), err))?;
    if let Some(parent) = args.out_index.parent() {
        fs::create_dir_all(parent).map_err(|err| format!("mkdir {}: {}", parent.display(), err))?;
    }

    let mut entries: Vec<ScrollIndexEntry> = Vec::new();
    for (idx, source_csv) in inputs.iter().enumerate() {
        let source_path = Path::new(source_csv);
        if !source_path.exists() {
            return Err(format!("missing source CSV: {}", source_csv));
        }
        let dataset_id = format!("{}-{:04}", args.dataset_prefix, idx + 1);
        let file_name = source_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("dataset.csv");
        let slug = slugify(file_name);
        let out_name = format!("{}_{}.toml", dataset_id, slug);
        let out_path = args.out_dir.join(out_name);
        let canonical_toml = out_path.to_string_lossy().to_string();
        let spec = ConvertSpec {
            dataset_id: &dataset_id,
            slug: &slug,
            source_csv,
            canonical_toml: &canonical_toml,
            dataset_class: args.dataset_class.as_str(),
            corpus_label: &args.corpus_label,
            migrated_by: "gororoba_cli::scrollify-csv",
        };
        let converted = convert_csv_to_scroll(source_path, &spec)
            .map_err(|err| format!("{}: {}", source_csv, err))?;
        fs::write(&out_path, converted.rendered_dataset_toml)
            .map_err(|err| format!("write {}: {}", out_path.display(), err))?;
        entries.push(converted.index_entry);
    }

    let source_descriptor = if let Some(manifest) = &args.source_manifest {
        format!("manifest:{}", manifest.display())
    } else if let Some(glob) = &args.source_glob {
        format!("glob:{}", glob)
    } else {
        "manual".to_string()
    };
    let rendered_index = render_scroll_index(
        &entries,
        &args.index_table,
        &source_descriptor,
        &args.out_dir.to_string_lossy(),
        &args.corpus_label,
        "gororoba_cli::scrollify-csv",
    );
    fs::write(&args.out_index, rendered_index)
        .map_err(|err| format!("write {}: {}", args.out_index.display(), err))?;

    println!(
        "Wrote {} with {} datasets and {} scroll TOML files in {}.",
        args.out_index.display(),
        entries.len(),
        entries.len(),
        args.out_dir.display()
    );
    Ok(())
}

fn main() {
    let args = Args::parse();
    if let Err(err) = run(args) {
        eprintln!("ERROR: {}", err);
        std::process::exit(1);
    }
}

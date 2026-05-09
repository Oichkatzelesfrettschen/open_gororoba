//! Dataset SHA256 sweep (plan P6A.S2.T3).
//!
//! Walks every directory under `data/external/` that has a corresponding
//! row in `registry/datasets.toml` and computes a DETERMINISTIC tree-level
//! digest: sha256 of a sorted list of `sha256(file) + " " + relpath`
//! lines. File-level hashes are streamed (64 KB chunks) so arbitrarily
//! large files (e.g. 38 GB `manga/`, 15 GB `things/`) never exceed
//! bounded memory.
//!
//! Parallelism:
//!   - rayon fans out across datasets (outer parallelism).
//!   - Within each dataset, files are enumerated via walkdir and hashed
//!     sequentially (inner parallelism rate-limited by I/O bandwidth).
//!
//! Output:
//!   - `--emit-toml-patch`: prints a TOML fragment suitable for diffing
//!     into `registry/datasets.toml` (one `sha256 = "..."` per id).
//!   - `--in-place`: rewrite registry/datasets.toml updating sha256
//!     fields and `bytes` where they drift.
//!   - `--report`: human-readable summary (default).
//!
//! RCA note: tree-level digests are more robust than a flat `find | xargs
//! sha256sum | sha256sum` shell pipeline because:
//!   1. Relative paths are canonicalised (sorted, stripped of prefix).
//!   2. Hidden/dotfiles are handled explicitly (included by default).
//!   3. Symlinks do not follow (prevents directory loop attacks).
//!   4. Consistent across invocations regardless of filesystem readdir
//!      ordering, which is non-deterministic on some filesystems.

use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufReader, Read, Write},
    path::{Path, PathBuf},
    process::ExitCode,
    time::Instant,
};

use clap::Parser;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use toml::Value;
use walkdir::WalkDir;

#[derive(Parser)]
#[command(
    name = "dataset-hash-sweep",
    about = "Compute deterministic per-dataset SHA256 tree-digests (plan P6A.S2.T3)."
)]
struct Args {
    /// Registry path.
    #[arg(long, default_value = "registry/datasets.toml")]
    registry: PathBuf,

    /// Data root.
    #[arg(long, default_value = "data/external")]
    data_root: PathBuf,

    /// Emit a TOML patch on stdout.
    #[arg(long, default_value_t = false)]
    emit_toml_patch: bool,

    /// Rewrite the registry in place (update sha256 and bytes).
    #[arg(long, default_value_t = false)]
    in_place: bool,

    /// Limit to a specific dataset id (repeatable).
    #[arg(long)]
    only: Vec<String>,

    /// Skip rows that already have a non-empty sha256 (resume mode).
    #[arg(long, default_value_t = false)]
    skip_existing: bool,
}

#[derive(Clone, Debug)]
struct DatasetRow {
    id: String,
    local_path: String,
}

/// Per-directory hash result: `(sha256_hex, file_count, total_bytes)`.
type HashOutcome = Result<(String, u64, u64), String>;
type DatasetHashResult = (DatasetRow, HashOutcome);

fn main() -> ExitCode {
    let args = Args::parse();

    let text = match std::fs::read_to_string(&args.registry) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("ERROR: read {}: {}", args.registry.display(), e);
            return ExitCode::FAILURE;
        }
    };
    let doc: Value = match toml::from_str(&text) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("ERROR: parse {}: {}", args.registry.display(), e);
            return ExitCode::FAILURE;
        }
    };

    let empty: Vec<Value> = Vec::new();
    let rows = doc
        .get("dataset")
        .and_then(Value::as_array)
        .unwrap_or(&empty);

    let mut to_hash: Vec<DatasetRow> = Vec::new();
    for r in rows {
        let id = match r.get("id").and_then(Value::as_str) {
            Some(s) => s.to_string(),
            None => continue,
        };
        if !args.only.is_empty() && !args.only.contains(&id) {
            continue;
        }
        let local_path = r
            .get("local_path")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let existing_sha = r
            .get("sha256")
            .and_then(Value::as_str)
            .unwrap_or("");
        if args.skip_existing && !existing_sha.is_empty() {
            continue;
        }
        to_hash.push(DatasetRow { id, local_path });
    }

    if to_hash.is_empty() {
        eprintln!("nothing to hash");
        return ExitCode::SUCCESS;
    }

    eprintln!("[hash-sweep] hashing {} datasets", to_hash.len());
    let t0 = Instant::now();

    let results: Vec<DatasetHashResult> = to_hash
        .into_par_iter()
        .map(|row| {
            let dir = Path::new(&row.local_path);
            let start = Instant::now();
            let r = hash_directory(dir);
            eprintln!(
                "  [{}] {:>10.2}s {}",
                if r.is_ok() { "ok " } else { "ERR" },
                start.elapsed().as_secs_f64(),
                row.id
            );
            (row, r)
        })
        .collect();

    eprintln!(
        "[hash-sweep] total {:.1}s across {} datasets",
        t0.elapsed().as_secs_f64(),
        results.len()
    );

    let mut by_id: BTreeMap<String, (String, u64)> = BTreeMap::new();
    let mut failures: Vec<(String, String)> = Vec::new();
    for (row, r) in &results {
        match r {
            Ok((sha, bytes, _files)) => {
                by_id.insert(row.id.clone(), (sha.clone(), *bytes));
            }
            Err(e) => failures.push((row.id.clone(), e.clone())),
        }
    }

    if args.emit_toml_patch {
        for (id, (sha, bytes)) in &by_id {
            println!("# dataset {id}: bytes={bytes}");
            println!("# sha256 = \"{sha}\"");
            println!();
        }
    }

    if args.in_place {
        if let Err(e) = rewrite_in_place(&args.registry, &text, &by_id) {
            eprintln!("ERROR: in-place rewrite: {e}");
            return ExitCode::FAILURE;
        }
        eprintln!("[hash-sweep] registry rewritten in place");
    }

    // Report
    println!(
        "[hash-sweep] {} successful, {} failed",
        by_id.len(),
        failures.len()
    );
    for (id, err) in &failures {
        println!("  FAIL {id}: {err}");
    }

    if failures.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// Return (tree_sha256_hex, total_bytes, file_count).
fn hash_directory(dir: &Path) -> Result<(String, u64, u64), String> {
    if !dir.is_dir() {
        return Err(format!("not a directory: {}", dir.display()));
    }

    // Enumerate files (sorted relative paths, no symlink following).
    let mut entries: Vec<PathBuf> = Vec::new();
    for entry in WalkDir::new(dir).follow_links(false).into_iter() {
        let entry = entry.map_err(|e| format!("walk: {e}"))?;
        if entry.file_type().is_file() {
            entries.push(entry.into_path());
        }
    }
    entries.sort();

    let mut total_bytes: u64 = 0;
    let mut tree_lines: Vec<String> = Vec::with_capacity(entries.len());

    for path in &entries {
        let rel = path.strip_prefix(dir).unwrap_or(path);
        let rel_str = rel.to_string_lossy().to_string();
        let (file_sha, size) = hash_file_streaming(path)?;
        total_bytes += size;
        // "<sha256> <relpath>" one per line; sorted by path ensures determinism
        tree_lines.push(format!("{file_sha}  {rel_str}"));
    }

    // Hash the joined tree listing
    let manifest = tree_lines.join("\n") + "\n";
    let mut hasher = Sha256::new();
    hasher.update(manifest.as_bytes());
    let tree_sha = hex_encode(&hasher.finalize());

    Ok((tree_sha, total_bytes, entries.len() as u64))
}

fn hash_file_streaming(path: &Path) -> Result<(String, u64), String> {
    let f = File::open(path).map_err(|e| format!("open {}: {}", path.display(), e))?;
    let mut reader = BufReader::with_capacity(64 * 1024, f);
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024];
    let mut total: u64 = 0;
    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| format!("read {}: {}", path.display(), e))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        total += n as u64;
    }
    Ok((hex_encode(&hasher.finalize()), total))
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

/// In-place TOML rewrite.
///
/// We do a line-oriented edit (track `id = "..."` context, then update
/// the following `sha256 = ""` and `bytes = N` lines) rather than
/// round-tripping through toml::Value to preserve comments and ordering.
fn rewrite_in_place(
    path: &Path,
    original: &str,
    updates: &BTreeMap<String, (String, u64)>,
) -> Result<(), String> {
    let mut out = String::with_capacity(original.len() + updates.len() * 72);
    let mut current_id: Option<String> = None;
    for line in original.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("id = \"") {
            current_id = trimmed
                .trim_start_matches("id = \"")
                .split('"')
                .next()
                .map(String::from);
        }
        if let Some(id) = &current_id
            && let Some((sha, bytes)) = updates.get(id)
        {
            if trimmed.starts_with("sha256 = ") {
                out.push_str(&format!("sha256 = \"{sha}\"\n"));
                continue;
            }
            if trimmed.starts_with("bytes = ") {
                out.push_str(&format!("bytes = {bytes}\n"));
                continue;
            }
        }
        out.push_str(line);
        out.push('\n');
    }
    let mut f = File::create(path).map_err(|e| format!("create {}: {}", path.display(), e))?;
    f.write_all(out.as_bytes())
        .map_err(|e| format!("write {}: {}", path.display(), e))?;
    Ok(())
}

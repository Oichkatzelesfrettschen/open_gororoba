// SPDX-License-Identifier: MIT
//
// A copied validation binary carries every path its compilation resolved:
// `env!("CARGO_MANIFEST_DIR")` expansions, debuginfo directory entries, and
// panic-location strings all land in the byte image. The shared Cargo
// build-dir hands a byte-identical workspace crate compiled in one worktree to
// every other worktree, so a binary staged under one checkout can hold another
// checkout's path. When that other checkout is removed, the binary fails at
// runtime on a path that no longer resolves.
//
// `scan_tools_dir` reads each staged binary, extracts every absolute path
// beginning at the worktrees root, and reports the ones that neither live
// under the running checkout nor exist on disk. The existence filter is what
// makes the scan tractable: debuginfo names thousands of live paths, and only
// a vanished one can break a run.

use std::collections::BTreeSet;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// One vanished path found inside one staged binary.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct VanishedPathHit {
    pub binary: PathBuf,
    /// The shallowest ancestor under the worktrees root that does not exist.
    /// A string table packs entries without separators, so an extracted run
    /// can carry a few bytes of the next entry; the ancestor is immune to
    /// that and names the removed checkout exactly.
    pub vanished_root: String,
    pub embedded: String,
}

/// Walk down from `worktrees_root` and return the first component path that
/// does not exist.
fn shallowest_missing(path: &str, worktrees_root: &Path) -> Option<String> {
    let rest = Path::new(path).strip_prefix(worktrees_root).ok()?;
    let mut probe = worktrees_root.to_path_buf();
    for component in rest.components() {
        probe.push(component);
        if !probe.exists() {
            return Some(probe.to_string_lossy().into_owned());
        }
    }
    None
}

/// A byte is part of a path run while it is printable ASCII and not a
/// separator that a path cannot contain in this repository's layout.
fn is_path_byte(b: u8) -> bool {
    matches!(b, b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9')
        || matches!(b, b'/' | b'.' | b'-' | b'_' | b'+' | b'@' | b'~' | b'=')
}

/// Extract every distinct absolute path that starts at `root` in `bytes`.
pub fn embedded_paths_under(bytes: &[u8], root: &str) -> BTreeSet<String> {
    let needle = root.as_bytes();
    let mut found = BTreeSet::new();
    if needle.is_empty() || bytes.len() < needle.len() {
        return found;
    }
    let mut i = 0usize;
    while i + needle.len() <= bytes.len() {
        if &bytes[i..i + needle.len()] != needle {
            i += 1;
            continue;
        }
        let mut end = i + needle.len();
        while end < bytes.len() && is_path_byte(bytes[end]) {
            end += 1;
        }
        // Trim a trailing separator or dot so `/a/b/` and `/a/b` collapse.
        let mut slice = &bytes[i..end];
        while matches!(slice.last(), Some(b'/') | Some(b'.')) {
            slice = &slice[..slice.len() - 1];
        }
        if slice.len() > needle.len()
            && let Ok(text) = std::str::from_utf8(slice)
        {
            found.insert(text.to_string());
        }
        i = end.max(i + 1);
    }
    found
}

/// Report the paths under `worktrees_root` that a binary embeds, that do not
/// sit under `current_root`, and that no longer exist.
pub fn scan_binary(
    binary: &Path,
    worktrees_root: &Path,
    current_root: &Path,
) -> io::Result<Vec<VanishedPathHit>> {
    let bytes = fs::read(binary)?;
    let root = format!("{}/", worktrees_root.display());
    let current = current_root.to_string_lossy().into_owned();
    let mut hits = Vec::new();
    for path in embedded_paths_under(&bytes, &root) {
        if path == current || path.starts_with(&format!("{current}/")) {
            continue;
        }
        let Some(vanished_root) = shallowest_missing(&path, worktrees_root) else {
            continue;
        };
        hits.push(VanishedPathHit {
            binary: binary.to_path_buf(),
            vanished_root,
            embedded: path,
        });
    }
    Ok(hits)
}

/// Scan every regular file directly inside `tools_dir`. Stamps and the
/// host-profile snapshot are read as bytes like the executables; a text file
/// naming a vanished worktree is the same defect.
pub fn scan_tools_dir(
    tools_dir: &Path,
    worktrees_root: &Path,
    current_root: &Path,
) -> io::Result<Vec<VanishedPathHit>> {
    let mut hits = Vec::new();
    let entries = match fs::read_dir(tools_dir) {
        Ok(entries) => entries,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(hits),
        Err(err) => return Err(err),
    };
    let mut paths: Vec<PathBuf> = Vec::new();
    for entry in entries {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            paths.push(entry.path());
        }
    }
    paths.sort();
    for path in paths {
        hits.extend(scan_binary(&path, worktrees_root, current_root)?);
    }
    Ok(hits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extraction_takes_the_longest_run_and_drops_the_bare_root() {
        let bytes = b"\x00/wt/a/registry/claims.toml\x00padding/wt/\x00";
        let found = embedded_paths_under(bytes, "/wt/");
        assert!(found.contains("/wt/a/registry/claims.toml"), "{found:?}");
        assert_eq!(found.len(), 1);
    }
}

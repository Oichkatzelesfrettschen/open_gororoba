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
    /// The shallowest directory under the worktrees root that does not exist.
    /// This is the removed checkout, or a removed directory inside a live one.
    pub vanished_root: String,
    pub embedded: String,
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

/// Decide whether an extracted path names a directory that is gone.
///
/// A linker packs string-table entries without separators, so a run extracted
/// from a binary regularly carries the first bytes of the next entry:
/// `.../enso-outlook/xtask` came back as
/// `.../enso-outlook/xtasktotal_elapsed_seccalled`. The glue can only corrupt
/// the tail, so the parent chain is tested for existence directly, and the
/// last component is tested by prefix: a live sibling whose name is a prefix
/// of the extracted component means the run is a glued form of that sibling.
/// `xtasktotal_elapsed_seccalled` clears on its `xtask` sibling, while a bare
/// checkout root such as `.../build-share-primary-cache` has no such sibling
/// and stays a hit.
fn shallowest_missing_directory(path: &str, worktrees_root: &Path) -> Option<String> {
    let rest = Path::new(path).strip_prefix(worktrees_root).ok()?;
    let components: Vec<_> = rest.components().collect();
    let (last, parents) = components.split_last()?;
    let mut probe = worktrees_root.to_path_buf();
    for component in parents {
        probe.push(component);
        if !probe.is_dir() {
            return Some(probe.to_string_lossy().into_owned());
        }
    }
    let tail = last.as_os_str().to_string_lossy().into_owned();
    let glued_from_a_live_sibling = fs::read_dir(&probe).is_ok_and(|entries| {
        entries.flatten().any(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            !name.is_empty() && tail.starts_with(&name)
        })
    });
    if glued_from_a_live_sibling {
        return None;
    }
    probe.push(last);
    Some(probe.to_string_lossy().into_owned())
}

/// Report the paths under `worktrees_root` that a binary embeds, that do not
/// sit under `current_root`, and that no live directory accounts for.
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
        let Some(vanished_root) = shallowest_missing_directory(&path, worktrees_root) else {
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
    fn a_glued_tail_clears_on_a_live_sibling_and_a_bare_root_does_not() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        std::fs::create_dir_all(root.join("live/xtask")).unwrap();
        let glued = format!("{}/live/xtasktotal_elapsed_sec", root.display());
        assert_eq!(shallowest_missing_directory(&glued, root), None);
        let gone = format!("{}/removed/crates/thing", root.display());
        assert_eq!(
            shallowest_missing_directory(&gone, root),
            Some(root.join("removed").to_string_lossy().into_owned())
        );
        // A bare checkout root with no trailing component and no live sibling
        // whose name is a prefix of it.
        let bare = format!("{}/build-share-primary-cache", root.display());
        assert_eq!(
            shallowest_missing_directory(&bare, root),
            Some(root.join("build-share-primary-cache").to_string_lossy().into_owned())
        );
    }

    #[test]
    fn extraction_takes_the_longest_run_and_drops_the_bare_root() {
        let bytes = b"\x00/wt/a/registry/claims.toml\x00padding/wt/\x00";
        let found = embedded_paths_under(bytes, "/wt/");
        assert!(found.contains("/wt/a/registry/claims.toml"), "{found:?}");
        assert_eq!(found.len(), 1);
    }
}

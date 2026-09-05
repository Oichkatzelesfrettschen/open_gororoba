# Retained calibration and external-intake sources

The source archives preserve exact retrieved bytes needed to inspect the
calibration protocol and replay the external input-admission failure after a
worktree is removed. A later download can return revised provider bytes.
The entry manifests pin the observations used by the retained findings.

| Archive | Files | Uncompressed bytes | Evidence boundary |
| --- | ---: | ---: | --- |
| `external-intake.tar.zst` | 684 | 318826464 | HAPI bodies, retry observations, native CDF, source metadata and diagnostic outputs |
| `calibration-research.tar.zst` | 24 | 2812107 | Explicitly licensed full text, factual provider metadata, catalog and research notes |

The public research archive excludes 19 source files (1844239 bytes) whose
redistribution permission remains unestablished. Exact copies and the original
43-member archive remain private in `.cache/empirical-claim-private-research`
in both the working and primary checkouts. The original archive digest is
`4cddd2b33ade6c94753ef9faaaef8eed4744d146a2dc404305206f0dd4214fdd`.
`calibration-research-private.entries` and
`calibration-research-private.sha256` retain the private member inventory.
The public/private lists partition all 43 original research files; all 727
source members remain locally preserved across the two archives and private
retention. `license-evidence.md` records the license evidence, attribution,
and complete BSD notice for the publicly retained full text.

The archives retain each member's original `.cache`-relative path. The
`*.entries` files enumerate every member; the `*.sha256` files pin individual
file contents. `archives.sha256` pins the compressed archives. Two exact path
rules in the repository `.gitattributes` store the archives through Git LFS.

The intake archive excludes the rebuildable `cdf-inspect` executable
(4870576 bytes); its 882-byte Rust source remains archived. The private
`.cache/artifact-retrieval-private-headers` directory stays outside both
archives. Its raw authentication and cookie fields must remain private.
The intake `.headers` files were separately scanned for cookie and
authorization field names; that scan found zero matches. The archives retain
failed source requests and challenged publisher responses as observations;
retention does not establish document identity or admit those responses as
scientific evidence.

## Restore into an empty directory

Run from the checkout root after Git LFS has materialized both archives.
`--keep-old-files` refuses replacement of any existing member. Restoration
uses a new directory so preexisting input files remain outside the extraction
boundary.

```bash
set -eu
archive_root="$(pwd)/data/output/audit/staples-causal-validation/source-archives"
sha256sum --check "$archive_root/archives.sha256"
mkdir -p .cache
restore_root="$(mktemp -d .cache/source-archive-restore.XXXXXXXX)"
for bundle in external-intake calibration-research; do
    tar --zstd --extract --keep-old-files --no-same-owner \
        --file="$archive_root/$bundle.tar.zst" --directory="$restore_root"
    (cd "$restore_root" && sha256sum --check "$archive_root/$bundle.sha256")
done
```

The archive manifests use paths relative to the restoration root. Inspect
the resulting `$restore_root/.cache/` tree before selecting a replay root.
The original worktree caches remain retained until integration and cleanup
parity checks complete.

## Historical locators and replay

The external-intake manifest, per-date receipts, blocked-intake result and
byte-check log record absolute paths under their original worktree. Those
strings identify the historical capture location. Resolve their
`.cache/staples-external-intake/` suffix against the selected restoration root.
When a consumer requires literal filenames in its input manifest, create a
separate derived replay manifest containing the original manifest hash and
the explicit old-root-to-restore-root mapping. Preserve the original receipt
and body hashes. A path rewrite must not be presented as the original sealed
campaign input.

`external-native-source-rca.toml` names the native CDF and failed HAPI body
under the archived relative paths. Pass the restored paths explicitly to
`external-cdf-order-audit`. The retained `external-cdf-order-build.args` and
`external-intake-build.toml` refer to machine-specific Cargo dependency
directories and hash-qualified rlibs. Those arguments document the observed
build; a fresh cache must resolve compatible dependencies from the pinned
workspace lock and record the rebuilt executable identity. Archive retention
does not establish a portable fresh-cache build.

The research source manifest names document basenames. Resolve public names
inside `.cache/empirical-calibration-research/` under the restoration root.
Resolve private names against the locally retained private inventory; a
public checkout alone does not contain those full articles or technical
documents. The public citations, hashes and findings remain available.
The retained research summaries can include later corrections; the archive
preserves the cached versions and the primary-source bytes independently.

## Deterministic archive construction

GNU tar 1.35 and Zstandard 1.5.7 create regular-file-only archives from
lexically sorted entry lists. Tar records use GNU format, zero mtime,
numeric uid/gid zero and mode 0644. These metadata values are archive
normalization choices; the SHA256 manifests verify original file bytes.

```bash
set -eu
for bundle in external-intake calibration-research; do
    tar --format=gnu --sort=name --mtime=@0 --owner=0 --group=0 \
        --numeric-owner --mode=0644 --no-recursion --verbatim-files-from \
        -T "data/output/audit/staples-causal-validation/source-archives/$bundle.entries" \
        -I 'zstd -19 -T1' -cf ".cache/$bundle-rebuilt.tar.zst"
done
```

Build from a root containing the restored `.cache` members and the retained
entry lists. Compare rebuilt archive digests with `archives.sha256` after
accounting for the output filename. `verification.toml` records member
coverage and extraction checks for the retained archive bytes.

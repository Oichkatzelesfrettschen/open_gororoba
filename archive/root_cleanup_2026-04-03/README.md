# Root Cleanup Bundle (2026-04-03)

This archive bundles local-only root artifacts that were moved out of the
repository root during a structure audit on 2026-04-03.

Purpose:

- reduce visible root clutter without deleting user data
- preserve loose screenshots, logs, scratch scripts, and accidental folders
- avoid moving files that are still referenced by registries or code paths

Contents:

- `screenshots/`: orphaned root PNG captures with no repo references
- `logs/`: orphaned root logs with no repo references
- `scripts/`: orphaned root shell helpers with no repo references
- `tests/`: orphaned root test shell helpers with no repo references
- `text/`: orphaned root text scratch files with no repo references
- `accidental_dirs/root-tilde-dir/`: the accidental `./~` directory that had
  been sitting in the repo root

Files intentionally left in the root after this pass:

- registry-linked Markdown notes still referenced by
  `registry/markdown_owner_map.toml`
- ephemeral local markers such as `.codex`, which may be recreated by tooling

Files relocated after the initial pass:

- root `warp_ring_integration.png` moved here as a duplicate after the canonical
  artifact path was standardized to `data/artifacts/warp_ring_integration.png`
- root `perf.data` and `perf.data.old` moved into `reports/profiling/root_legacy/`

This bundle is a quarantine/archive staging area, not a new canonical source
surface.

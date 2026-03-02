#!/usr/bin/env python3
"""Migrate stale algebra_core paths to canonical split-crate locations.

Post-split crate mapping:
  - crates/algebra_core/src/analysis/*     -> crates/algebra_analysis/src/*
  - crates/algebra_core/src/experimental/* -> crates/algebra_experimental/src/*
  - crates/algebra_core/src/construction/cayley_dickson* -> crates/cd_kernel/src/cayley_dickson*
  - crates/algebra_core/src/construction/mult_table*     -> crates/cd_kernel/src/mult_table*
  - Flat legacy paths (boxkites.rs, etc.)  -> appropriate new crate
  - Rust module paths (algebra_core::analysis::*) -> algebra_analysis::*

Usage:
  python3 scripts/migrate_crate_paths.py --dry-run --verbose
  python3 scripts/migrate_crate_paths.py --apply
  python3 scripts/migrate_crate_paths.py --apply --verbose
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path rewrite rules, ordered longest-match-first within each category.
#
# Rules are plain string replacements applied in order.  The ordering
# guarantees that more-specific patterns (e.g. a full file path) are
# attempted before less-specific prefixes (e.g. a directory prefix).
# ---------------------------------------------------------------------------

PATH_REWRITES: list[tuple[str, str]] = [
    # -- Category 3: CD kernel (specific files, before directory prefixes) --
    (
        "crates/algebra_core/src/construction/cayley_dickson",
        "crates/cd_kernel/src/cayley_dickson",
    ),
    (
        "crates/algebra_core/src/construction/mult_table",
        "crates/cd_kernel/src/mult_table",
    ),
    # -- Category 1: analysis directory prefix -> algebra_analysis -----------
    ("crates/algebra_core/src/analysis/", "crates/algebra_analysis/src/"),
    # -- Category 2: experimental directory prefix -> algebra_experimental ---
    ("crates/algebra_core/src/experimental/", "crates/algebra_experimental/src/"),
    # -- Category 4: flat legacy paths (no subdirectory) --------------------
    # Only remap modules that moved; paths like clifford.rs, wheels.rs, etc.
    # stay in algebra_core (they were reorganised INTO subdirectories there).
    (
        "crates/algebra_core/src/boxkites.rs",
        "crates/algebra_analysis/src/boxkites.rs",
    ),
    (
        "crates/algebra_core/src/cayley_dickson.rs",
        "crates/cd_kernel/src/cayley_dickson.rs",
    ),
    (
        "crates/algebra_core/src/emanation.rs",
        "crates/algebra_experimental/src/emanation.rs",
    ),
    (
        "crates/algebra_core/src/grassmannian.rs",
        "crates/algebra_analysis/src/grassmannian.rs",
    ),
    (
        "crates/algebra_core/src/homotopy_algebra.rs",
        "crates/algebra_analysis/src/homotopy_algebra.rs",
    ),
    (
        "crates/algebra_core/src/reggiani.rs",
        "crates/algebra_analysis/src/reggiani.rs",
    ),
    (
        "crates/algebra_core/src/zd_graphs.rs",
        "crates/algebra_analysis/src/zd_graphs.rs",
    ),
    # -- Category 4b: bare prefix (no crates/ -- appears in Rocq comments
    #    and some TOML reference fields).  Must come AFTER the crates/-prefixed
    #    rules so that the longer match is tried first.
    #    IMPORTANT: only match specific files/dirs that moved, not the generic
    #    "algebra_core/src/construction/" prefix (some construction modules stay).
    (
        "algebra_core/src/construction/cayley_dickson",
        "cd_kernel/src/cayley_dickson",
    ),
    (
        "algebra_core/src/construction/mult_table",
        "cd_kernel/src/mult_table",
    ),
    ("algebra_core/src/analysis/", "algebra_analysis/src/"),
    ("algebra_core/src/experimental/", "algebra_experimental/src/"),
    # -- Category 5: Rust module-path syntax --------------------------------
    ("algebra_core::analysis::", "algebra_analysis::"),
    ("algebra_core::experimental::", "algebra_experimental::"),
    ("algebra_core::construction::cayley_dickson", "cd_kernel::cayley_dickson"),
    ("algebra_core::construction::mult_table", "cd_kernel::mult_table"),
]

# Marker line that indicates a file is auto-generated markdown (skip in docs/).
# TOML and .v files are always processed; only docs/*.md files are filtered.
AUTO_GEN_MARKER = "AUTO-GENERATED: DO NOT EDIT"


def collect_target_files(root: Path) -> list[Path]:
    """Return all files in scope for migration."""
    targets: list[Path] = []

    # registry/**/*.toml
    registry_dir = root / "registry"
    if registry_dir.is_dir():
        targets.extend(sorted(registry_dir.rglob("*.toml")))

    # proofs/**/*.v
    proofs_dir = root / "proofs"
    if proofs_dir.is_dir():
        targets.extend(sorted(proofs_dir.rglob("*.v")))

    # docs/**/*.md (hand-authored only -- skip AUTO-GENERATED)
    docs_dir = root / "docs"
    if docs_dir.is_dir():
        targets.extend(sorted(docs_dir.rglob("*.md")))

    # plans/**/*.toml, reports/**/*.toml, archive/**/*.toml
    for extra in ("plans", "reports", "archive"):
        extra_dir = root / extra
        if extra_dir.is_dir():
            targets.extend(sorted(extra_dir.rglob("*.toml")))

    return targets


def is_auto_generated_md(path: Path, content: str) -> bool:
    """Check whether a .md file carries the auto-generation marker."""
    if path.suffix != ".md":
        return False
    # Check only the first 512 bytes for the marker (it is always near the top).
    return AUTO_GEN_MARKER in content[:512]


def apply_rewrites(content: str) -> tuple[str, int]:
    """Apply all rewrite rules to content.  Return (new_content, change_count)."""
    total = 0
    for old, new in PATH_REWRITES:
        count = content.count(old)
        if count > 0:
            content = content.replace(old, new)
            total += count
    return content, total


def run(
    root: Path,
    *,
    dry_run: bool = True,
    verbose: bool = False,
) -> tuple[int, int, int]:
    """Run the migration.  Returns (files_changed, total_replacements, files_skipped)."""
    files = collect_target_files(root)
    files_changed = 0
    total_replacements = 0
    files_skipped = 0

    for path in files:
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            if verbose:
                print(f"  SKIP {path.relative_to(root)}: {exc}")
            files_skipped += 1
            continue

        # Skip auto-generated markdown (will be regenerated from TOML source).
        if is_auto_generated_md(path, content):
            if verbose:
                print(f"  SKIP {path.relative_to(root)}: auto-generated")
            files_skipped += 1
            continue

        new_content, count = apply_rewrites(content)
        if count == 0:
            continue

        total_replacements += count
        files_changed += 1

        rel = path.relative_to(root)
        if verbose:
            print(f"  {rel}: {count} replacement(s)")

        if not dry_run:
            path.write_text(new_content, encoding="utf-8")

    return files_changed, total_replacements, files_skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate stale algebra_core paths to split-crate canonical locations."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes without writing files.",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes in-place.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-file details.",
    )
    args = parser.parse_args()

    # Resolve project root (script lives in scripts/).
    root = Path(__file__).resolve().parent.parent
    if not (root / "Cargo.toml").is_file():
        print(f"ERROR: Cannot find project root (expected Cargo.toml at {root})")
        sys.exit(1)

    mode_label = "DRY RUN" if args.dry_run else "APPLY"
    print(f"[{mode_label}] Scanning {root} ...")
    print()

    changed, total, skipped = run(root, dry_run=args.dry_run, verbose=args.verbose)

    print()
    print(f"Files changed:  {changed}")
    print(f"Replacements:   {total}")
    print(f"Files skipped:  {skipped} (auto-generated / unreadable)")

    if args.dry_run and total > 0:
        print()
        print("Re-run with --apply to write changes.")


if __name__ == "__main__":
    main()

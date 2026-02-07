#!/usr/bin/env python3
"""
Verify provider consistency between Rust dataset registry and markdown manifest.

This check fails when provider symbols listed in:
- crates/gororoba_cli/src/bin/fetch_datasets.rs
and
- docs/external_sources/DATASET_MANIFEST.md
drift out of sync.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

RUST_REGISTRY_PATH = Path("crates/gororoba_cli/src/bin/fetch_datasets.rs")
MANIFEST_PATH = Path("docs/external_sources/DATASET_MANIFEST.md")

RUST_PROVIDER_RX = re.compile(
    r"Box::new\(\s*(?:[A-Za-z0-9_]+::)*([A-Za-z0-9_]+Provider)\s*\)"
)
MANIFEST_PROVIDER_RX = re.compile(r"\b([A-Za-z0-9_]+Provider)\b")


def _parse_markdown_cells(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return []
    if not stripped.endswith("|"):
        return []
    return [cell.strip() for cell in stripped.split("|")[1:-1]]


def _is_separator_row(cells: list[str]) -> bool:
    if not cells:
        return False
    for cell in cells:
        if not cell:
            continue
        if any(ch not in "-:" for ch in cell):
            return False
    return True


def _load_rust_provider_counts(path: Path) -> Counter[str]:
    text = path.read_text(encoding="utf-8")
    return Counter(RUST_PROVIDER_RX.findall(text))


def _load_manifest_provider_counts(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    lines = path.read_text(encoding="utf-8").splitlines()
    in_dataset_table = False
    provider_col = -1

    for line in lines:
        cells = _parse_markdown_cells(line)
        if not cells:
            in_dataset_table = False
            provider_col = -1
            continue

        if "Dataset" in cells and "Provider" in cells:
            in_dataset_table = True
            provider_col = cells.index("Provider")
            continue

        if not in_dataset_table:
            continue
        if _is_separator_row(cells):
            continue
        if provider_col < 0 or provider_col >= len(cells):
            continue

        provider_cell = cells[provider_col]
        for provider in MANIFEST_PROVIDER_RX.findall(provider_cell):
            counts[provider] += 1

    return counts


def _format_list(items: list[str]) -> str:
    return ", ".join(items) if items else "<none>"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root (default: inferred from this file path).",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    rust_path = repo_root / RUST_REGISTRY_PATH
    manifest_path = repo_root / MANIFEST_PATH

    if not rust_path.exists():
        print(f"ERROR: Missing {RUST_REGISTRY_PATH}")
        return 2
    if not manifest_path.exists():
        print(f"ERROR: Missing {MANIFEST_PATH}")
        return 2

    rust_counts = _load_rust_provider_counts(rust_path)
    manifest_counts = _load_manifest_provider_counts(manifest_path)

    rust_set = set(rust_counts)
    manifest_set = set(manifest_counts)

    missing_in_manifest = sorted(rust_set - manifest_set)
    missing_in_rust = sorted(manifest_set - rust_set)

    rust_dups = sorted(name for name, count in rust_counts.items() if count > 1)
    manifest_dups = sorted(name for name, count in manifest_counts.items() if count > 1)

    if missing_in_manifest or missing_in_rust or rust_dups or manifest_dups:
        if missing_in_manifest:
            print(
                "ERROR: providers present in rust registry but missing from dataset manifest: "
                f"{_format_list(missing_in_manifest)}"
            )
        if missing_in_rust:
            print(
                "ERROR: providers present in dataset manifest but missing from rust registry: "
                f"{_format_list(missing_in_rust)}"
            )
        if rust_dups:
            print(
                "ERROR: duplicate providers in rust registry: "
                f"{_format_list(rust_dups)}"
            )
        if manifest_dups:
            print(
                "ERROR: duplicate providers in dataset manifest: "
                f"{_format_list(manifest_dups)}"
            )
        return 2

    print(
        "OK: dataset provider registry and manifest are consistent "
        f"({len(rust_set)} providers)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Verify that backticked artifact links in docs/CLAIMS_TASKS.md resolve.

Rationale: the tasks tracker is used as an executable backlog; broken file references should
fail the smoke suite early.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from verification.markdown_table import iter_table_rows


def _parse_markdown_table_rows(text: str) -> list[list[str]]:
    return [p.cells for p in iter_table_rows(text)]


def _extract_backticked_paths(cell_text: str) -> list[str]:
    return [m.group(1).strip() for m in re.finditer(r"`([^`]+)`", cell_text)]


def _path_matches_exist(repo_root: Path, pattern: str) -> bool:
    # Treat patterns relative to repo root. Disallow absolute paths.
    if pattern.startswith("/"):
        return False
    return any(repo_root.glob(pattern))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root (default: inferred from this file path).",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    tasks_path = repo_root / "docs/CLAIMS_TASKS.md"
    if not tasks_path.exists():
        print("ERROR: Missing docs/CLAIMS_TASKS.md")
        return 2

    failures: list[str] = []
    rows = _parse_markdown_table_rows(tasks_path.read_text(encoding="utf-8"))
    for parts in rows:
        if len(parts) < 4:
            continue
        claim_id = parts[0]
        if not claim_id.startswith("C-"):
            continue
        output_cell = parts[2]
        for token in _extract_backticked_paths(output_cell):
            if "*" in token or "?" in token or "[" in token:
                if not _path_matches_exist(repo_root, token):
                    failures.append(f"{claim_id}: glob has no matches: `{token}`")
                continue
            p = repo_root / token
            if not p.exists():
                failures.append(f"{claim_id}: missing path: `{token}`")

    if failures:
        for msg in failures:
            print(f"ERROR: {msg}")
        return 2

    print("OK: claims tasks artifact links resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

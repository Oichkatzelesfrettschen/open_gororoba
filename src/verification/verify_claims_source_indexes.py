#!/usr/bin/env python3
"""
Ensure open claims are source-indexed.

Policy: any claim in `docs/CLAIMS_EVIDENCE_MATRIX.md` that is Speculative/Unverified/Partially
verified must reference at least one `docs/external_sources/*.md` index in its "Where stated"
cell. This makes it harder for narrative claims to drift without a primary-source anchor.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from verification.markdown_table import iter_table_rows


@dataclass(frozen=True)
class MatrixRow:
    claim_id: str
    where_stated: str
    status: str


def _parse_markdown_table_rows(text: str) -> list[list[str]]:
    return [p.cells for p in iter_table_rows(text)]


def _load_matrix_rows(matrix_text: str) -> list[MatrixRow]:
    out: list[MatrixRow] = []
    for parts in _parse_markdown_table_rows(matrix_text):
        if len(parts) < 5:
            continue
        claim_id = parts[0]
        where = parts[2]
        status = parts[3]
        if claim_id.startswith("C-"):
            out.append(MatrixRow(claim_id=claim_id, where_stated=where, status=status))
    return out


def _needs_source_index(status: str) -> bool:
    s = status.lower()
    if "unverified" in s:
        return True
    if "speculative" in s:
        return True
    if "partial" in s:
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root (default: inferred from this file path).",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    matrix_path = repo_root / "docs/CLAIMS_EVIDENCE_MATRIX.md"
    if not matrix_path.exists():
        print("ERROR: Missing docs/CLAIMS_EVIDENCE_MATRIX.md")
        return 2

    matrix_rows = _load_matrix_rows(matrix_path.read_text(encoding="utf-8"))
    failures: list[str] = []

    for row in matrix_rows:
        if not _needs_source_index(row.status):
            continue
        if "docs/external_sources/" not in row.where_stated:
            failures.append(
                f"{row.claim_id}: open-status claim missing docs/external_sources index"
            )

    if failures:
        for msg in failures:
            print(f"ERROR: {msg}")
        return 2

    print("OK: open claims are source-indexed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

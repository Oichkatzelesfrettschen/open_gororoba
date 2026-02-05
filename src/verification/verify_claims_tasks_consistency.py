#!/usr/bin/env python3
"""
Cross-check `docs/CLAIMS_EVIDENCE_MATRIX.md` vs `docs/CLAIMS_TASKS.md`.

Goal: prevent drift where open (unresolved) claims exist in the matrix but have
no executable backlog items, or where task status suggests the matrix status is
stale.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from verification.markdown_table import iter_table_rows


@dataclass(frozen=True)
class MatrixRow:
    claim_id: str
    status: str


def _parse_markdown_table_rows(text: str) -> list[list[str]]:
    return [p.cells for p in iter_table_rows(text)]


def _load_matrix_rows(matrix_text: str) -> list[MatrixRow]:
    out: list[MatrixRow] = []
    for parts in _parse_markdown_table_rows(matrix_text):
        if len(parts) < 4:
            continue
        claim_id = parts[0]
        status = parts[3]
        if claim_id.startswith("C-"):
            out.append(MatrixRow(claim_id=claim_id, status=status))
    return out


def _load_tasks_statuses(tasks_text: str) -> dict[str, set[str]]:
    by_claim: dict[str, set[str]] = {}
    for parts in _parse_markdown_table_rows(tasks_text):
        if len(parts) < 4:
            continue
        claim_id = parts[0]
        status = parts[3]
        if not claim_id.startswith("C-"):
            continue
        by_claim.setdefault(claim_id, set()).add(status)
    return by_claim


def _is_resolved_matrix_status(status: str) -> bool:
    # Conservative: treat any "*partial*" status as unresolved.
    s = status.lower()
    if "partial" in s:
        return False
    resolved_markers = (
        "verified",
        "refuted",
        "rejected",
        "confirmed",
        "established",
        "clarified",
        "integrated",
        "implemented",
        "tested",
        "modeled",
    )
    return any(m in s for m in resolved_markers)


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
    tasks_path = repo_root / "docs/CLAIMS_TASKS.md"

    failures: list[str] = []

    if not matrix_path.exists():
        failures.append("Missing docs/CLAIMS_EVIDENCE_MATRIX.md")
    if not tasks_path.exists():
        failures.append("Missing docs/CLAIMS_TASKS.md")
    if failures:
        for msg in failures:
            print(f"ERROR: {msg}")
        return 2

    matrix_rows = _load_matrix_rows(matrix_path.read_text(encoding="utf-8"))
    tasks_statuses = _load_tasks_statuses(tasks_path.read_text(encoding="utf-8"))

    open_claims = [r for r in matrix_rows if not _is_resolved_matrix_status(r.status)]
    for row in open_claims:
        if row.claim_id not in tasks_statuses:
            failures.append(f"Open claim missing from tasks tracker: {row.claim_id}")

    # If a claim is tracked as DONE in tasks, it should not be Unverified in the matrix.
    for claim_id, statuses in tasks_statuses.items():
        if "DONE" not in statuses:
            continue
        row = next((r for r in matrix_rows if r.claim_id == claim_id), None)
        if row is None:
            continue
        if "unverified" in row.status.lower():
            failures.append(
                f"Tasks mark DONE but matrix says Unverified: {claim_id} ({row.status})"
            )

    if failures:
        for msg in failures:
            print(f"ERROR: {msg}")
        return 2

    print("OK: claims matrix and tasks tracker consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

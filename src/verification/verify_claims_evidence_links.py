#!/usr/bin/env python3
"""
Lightweight "claims -> evidence" integrity checks.

Goal: keep docs pointing at real, runnable repo artifacts (tests, verifiers,
cached datasets) by failing fast when referenced paths drift or disappear.

This verifier is intentionally conservative: it does not try to judge whether a
claim is correct, only whether its declared evidence locations exist on disk.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

_BACKTICK_RE = re.compile(r"`([^`]+)`")
_CID_RE = re.compile(r"^\|\s*(C-\d{3})\s*\|")


def _looks_like_repo_path(token: str) -> bool:
    token = token.strip()
    if not token:
        return False
    if any(ch.isspace() for ch in token):
        return False
    if "/" in token:
        return True
    return token.endswith((".py", ".md", ".csv", ".json", ".txt"))


def _normalize_repo_path_token(token: str) -> str:
    """
    Normalize common "path-like" references used in docs.

    We primarily care about verifying that the underlying artifact exists on
    disk.  Some docs reference pytest node ids (e.g., tests/foo.py::TestBar),
    which are valid evidence pointers but are not literal filesystem paths.
    """
    token = token.strip()
    if "::" in token:
        candidate = token.split("::", 1)[0].strip()
        if _looks_like_repo_path(candidate):
            return candidate
    return token


def _iter_backtick_paths(text: str) -> list[str]:
    out: list[str] = []
    for m in _BACKTICK_RE.finditer(text):
        token = _normalize_repo_path_token(m.group(1))
        if _looks_like_repo_path(token):
            out.append(token)
    return out


def _parse_claim_ids_from_matrix(text: str) -> list[str]:
    ids: list[str] = []
    for line in text.splitlines():
        m = _CID_RE.match(line)
        if m:
            ids.append(m.group(1))
    return ids


def _find_matrix_row_line(text: str, claim_id: str) -> str | None:
    needle = f"| {claim_id} |"
    for line in text.splitlines():
        if line.startswith(needle):
            return line
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root (default: inferred from this file path).",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    failures: list[str] = []

    matrix_path = repo_root / "docs/CLAIMS_EVIDENCE_MATRIX.md"
    index_path = repo_root / "docs/VERIFIED_CLAIMS_INDEX.md"

    for doc_path in (matrix_path, index_path):
        if not doc_path.exists():
            failures.append(f"Missing required doc: {doc_path.relative_to(repo_root)}")
            continue
        text = doc_path.read_text(encoding="utf-8")
        for rel in _iter_backtick_paths(text):
            p = repo_root / rel
            if not p.exists():
                failures.append(f"Missing referenced path: {rel} (from {doc_path.name})")

    if matrix_path.exists():
        matrix_text = matrix_path.read_text(encoding="utf-8")
        ids = _parse_claim_ids_from_matrix(matrix_text)
        seen: set[str] = set()
        dups: set[str] = set()
        for claim_id in ids:
            if claim_id in seen:
                dups.add(claim_id)
            seen.add(claim_id)
        for claim_id in sorted(dups):
            failures.append(f"Duplicate claim id in matrix: {claim_id}")

        # Claim-specific guardrails for high-risk narratives.
        c007 = _find_matrix_row_line(matrix_text, "C-007")
        if c007 is not None and "gwtc3_selection_bias_control_metrics.csv" not in c007:
            failures.append(
                "C-007 must reference data/csv/gwtc3_selection_bias_control_metrics.csv"
            )

    if failures:
        for msg in failures:
            print(f"ERROR: {msg}")
        return 2

    print("OK: claims/evidence links resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

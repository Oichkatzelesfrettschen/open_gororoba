#!/usr/bin/env python3
"""
Verify that each claim row has a stable "Where stated" pointer.

Goal: make the C-001..C-427 audit mechanically tractable by ensuring every row
points at repo-local artifacts (docs/code/data/tests) rather than legacy
shorthand like "vX expY".

This verifier is intentionally conservative and only checks for the presence of
at least one stable pointer token. It does not judge correctness of the claim.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from verification.markdown_table import iter_table_rows

_CLAIM_ID_RE = re.compile(r"^C-\d{3}$")


def _where_has_stable_pointer(where_cell: str) -> bool:
    cell = where_cell.strip()
    if not cell:
        return False
    low = cell.lower()
    if "multiple docs" in low:
        return True
    if "`" in cell:
        return True
    # Allow non-backticked but still-stable path mentions.
    return any(tok in cell for tok in ("src/", "docs/", "data/", "tests/"))


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

    text = matrix_path.read_text(encoding="utf-8")
    failures: list[str] = []
    for parsed in iter_table_rows(text):
        if len(parsed.cells) != 6:
            continue
        claim_id = parsed.cells[0].strip()
        if not _CLAIM_ID_RE.fullmatch(claim_id):
            continue
        where_cell = parsed.cells[2]
        if not _where_has_stable_pointer(where_cell):
            failures.append(
                f"docs/CLAIMS_EVIDENCE_MATRIX.md:{parsed.lineno}: {claim_id}: "
                "missing/weak Where stated pointer"
            )

    if failures:
        for msg in failures[:200]:
            print(f"ERROR: {msg}")
        if len(failures) > 200:
            print(f"ERROR: ... plus {len(failures) - 200} more")
        return 2

    print("OK: claims matrix 'Where stated' pointers present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

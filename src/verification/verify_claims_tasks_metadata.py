#!/usr/bin/env python3
"""
Verify claims-tasks tracker metadata hygiene.

Enforces:
- A file-level ISO date header exists: 'Date: YYYY-MM-DD'.
- Task rows are mechanically parseable into exactly 4 columns.
- Task status is one of a small canonical token set.

Rationale: make the C-001..C-427 audit scriptable and avoid silent drift.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path

from verification.claims_metadata_schema import CANONICAL_TASK_STATUS_TOKENS
from verification.markdown_table import iter_table_rows

_CLAIM_ID_RE = re.compile(r"^C-\d{3}$")
_DATE_LINE_RE = re.compile(r"^Date:\s*(\d{4}-\d{2}-\d{2})\b")


def _verify_date_header(text: str) -> str | None:
    for line in text.splitlines():
        m = _DATE_LINE_RE.match(line.strip())
        if not m:
            continue
        date_str = m.group(1)
        try:
            dt.date.fromisoformat(date_str)
        except ValueError:
            return f"Invalid Date: header ISO date: {date_str!r}"
        return None
    return "Missing Date: header with ISO date 'YYYY-MM-DD'"


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

    text = tasks_path.read_text(encoding="utf-8")
    failures: list[str] = []

    msg = _verify_date_header(text)
    if msg:
        failures.append(msg)

    for parsed in iter_table_rows(text):
        if not parsed.cells:
            continue
        claim_id = parsed.cells[0].strip()
        if not _CLAIM_ID_RE.fullmatch(claim_id):
            continue
        if len(parsed.cells) != 4:
            failures.append(
                f"docs/CLAIMS_TASKS.md:{parsed.lineno}: {claim_id}: expected 4 columns, got "
                f"{len(parsed.cells)}"
            )
            continue
        status = parsed.cells[3].strip()
        if status not in CANONICAL_TASK_STATUS_TOKENS:
            allowed = ", ".join(CANONICAL_TASK_STATUS_TOKENS)
            failures.append(
                f"docs/CLAIMS_TASKS.md:{parsed.lineno}: {claim_id}: non-canonical task status "
                f"{status!r} (allowed: {allowed})"
            )

    if failures:
        for msg in failures[:200]:
            print(f"ERROR: {msg}")
        if len(failures) > 200:
            print(f"ERROR: ... plus {len(failures) - 200} more")
        return 2

    print("OK: claims tasks metadata hygiene")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

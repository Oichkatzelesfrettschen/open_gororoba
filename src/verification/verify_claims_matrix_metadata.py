#!/usr/bin/env python3
"""
Verify claims-matrix metadata hygiene.

Enforces:
- Table rows are mechanically parseable into 6 columns.
- Status begins with a canonical token formatted as '**Token**'.
- Last verified begins with an ISO date 'YYYY-MM-DD'.

Rationale: make C-001..C-427 audits scriptable and avoid silent drift.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path

from verification.claims_metadata_schema import CANONICAL_CLAIMS_STATUS_TOKENS

_STATUS_RE = re.compile(r"^\*\*([^*]+)\*\*")
_ISO_PREFIX_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\b")


@dataclass(frozen=True)
class ParsedRow:
    claim_id: str
    status: str
    last_verified: str


def _parse_table_line_strict(line: str) -> list[str]:
    """
    Split a markdown table row into cells.

    This assumes literal pipes inside cells are escaped as '\\|', and that the row
    begins and ends with a pipe.
    """
    if not (line.startswith("|") and line.rstrip().endswith("|")):
        raise ValueError("Row must start and end with '|'")

    # Manual scan so we can ignore escaped pipes.
    cells: list[str] = []
    buf: list[str] = []
    escaped = False
    in_code = False
    i = 0
    # Skip leading pipe.
    i += 1
    while i < len(line):
        ch = line[i]
        if escaped:
            buf.append(ch)
            escaped = False
            i += 1
            continue
        if ch == "\\":
            escaped = True
            buf.append(ch)
            i += 1
            continue
        if ch == "`":
            in_code = not in_code
            buf.append(ch)
            i += 1
            continue
        if ch == "|" and not in_code:
            cell = "".join(buf).strip()
            cells.append(cell)
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1

    # Trailing buffer should be empty because rows end with '|'.
    if buf:
        raise ValueError("Trailing non-cell content after final '|'")

    # Drop potential empty due to the final separator.
    if cells and cells[-1] == "":
        cells = cells[:-1]

    return cells


def _parse_claim_rows(matrix_text: str) -> list[ParsedRow]:
    rows: list[ParsedRow] = []
    for lineno, line in enumerate(matrix_text.splitlines(), 1):
        if not line.startswith("| C-"):
            continue
        try:
            parts = _parse_table_line_strict(line)
        except ValueError as e:
            msg = (
                f"ERROR: docs/CLAIMS_EVIDENCE_MATRIX.md:{lineno}: "
                f"unparseable row ({e}). Escape literal pipes inside cells as '\\|'."
            )
            raise SystemExit(msg) from e
        if len(parts) != 6:
            msg = (
                f"ERROR: docs/CLAIMS_EVIDENCE_MATRIX.md:{lineno}: "
                f"expected 6 columns, got {len(parts)}. "
                "Escape literal pipes inside cells as '\\|'."
            )
            raise SystemExit(msg)
        claim_id = parts[0].strip()
        status = parts[3].strip()
        last_verified = parts[4].strip()
        if not re.fullmatch(r"C-\d{3}", claim_id):
            raise SystemExit(
                f"ERROR: docs/CLAIMS_EVIDENCE_MATRIX.md:{lineno}: bad claim id: {claim_id!r}"
            )
        rows.append(ParsedRow(claim_id=claim_id, status=status, last_verified=last_verified))
    return rows


def _verify_status(row: ParsedRow) -> str | None:
    m = _STATUS_RE.match(row.status)
    if not m:
        return f"{row.claim_id}: status must begin with '**Token**' ({row.status!r})"
    token = m.group(1).strip()
    if token not in CANONICAL_CLAIMS_STATUS_TOKENS:
        allowed = ", ".join(CANONICAL_CLAIMS_STATUS_TOKENS)
        return f"{row.claim_id}: non-canonical status token {token!r} (allowed: {allowed})"
    return None


def _verify_last_verified(row: ParsedRow) -> str | None:
    m = _ISO_PREFIX_RE.match(row.last_verified)
    if not m:
        return (
            f"{row.claim_id}: last_verified must begin with ISO date YYYY-MM-DD "
            f"({row.last_verified!r})"
        )
    date_str = m.group(1)
    try:
        dt.date.fromisoformat(date_str)
    except ValueError:
        return f"{row.claim_id}: invalid ISO date prefix {date_str!r}"
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
    matrix_path = repo_root / "docs/CLAIMS_EVIDENCE_MATRIX.md"
    if not matrix_path.exists():
        print("ERROR: Missing docs/CLAIMS_EVIDENCE_MATRIX.md")
        return 2

    rows = _parse_claim_rows(matrix_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for row in rows:
        msg = _verify_status(row)
        if msg:
            failures.append(msg)
        msg = _verify_last_verified(row)
        if msg:
            failures.append(msg)

    if failures:
        for msg in failures[:200]:
            print(f"ERROR: {msg}")
        if len(failures) > 200:
            print(f"ERROR: ... plus {len(failures) - 200} more")
        return 2

    print("OK: claims matrix metadata hygiene")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Normalize legacy/hard-to-parse claim-matrix rows.

This is a metadata hygiene helper that makes the claims matrix more mechanically
tractable by:
- collapsing accidental column splits caused by unescaped '|' characters
- moving legacy inline experiment tags ("\\|v...") out of the Claim cell
- escaping any remaining literal '|' in cell content

It is intentionally conservative:
- only rewrites a small, explicit allowlist of claim IDs
- never changes the Status token or Last verified date
"""

# SCRIPT_CONTRACT: {"inferred":true,"inputs":[],"network":"forbidden","network_env":"","outputs":[],"version":1}

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from verification.markdown_table import parse_table_line_cells

ISO_PREFIX_RE = re.compile(r"^\s*\d{4}-\d{2}-\d{2}\b")
STATUS_PREFIX_RE = re.compile(r"^\s*\*\*[^*]+\*\*")


LEGACY_CLAIM_IDS = {
    # Rows with legacy inline experiment tags in the claim text (from
    # reports/claims_status_inventory.md).
    "C-068",
    "C-070",
    "C-075",
    "C-077",
    "C-087",
    "C-094",
    "C-096",
    "C-097",
    "C-100",
    "C-115",
    "C-123",
    "C-126",
    "C-128",
    "C-129",
    "C-130",
    "C-131",
    "C-132",
    "C-135",
    "C-165",
    "C-399",
}

_STATUS_SUFFIX_ALLOWLIST: dict[str, set[str]] = {
    "Verified": {"math", "data"},
    "Speculative": {"WEAK", "SUGGESTIVE"},
}


@dataclass(frozen=True)
class Parsed:
    claim_id: str
    cells: list[str]


def _escape_unescaped_pipes(text: str) -> str:
    out: list[str] = []
    escaped = False
    in_code = False
    for ch in text:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            out.append(ch)
            escaped = True
            continue
        if ch == "`":
            in_code = not in_code
            out.append(ch)
            continue
        if ch == "|" and not in_code:
            out.append("\\|")
            continue
        out.append(ch)
    return "".join(out)


def _normalize_cells_to_six(cells: list[str]) -> list[str] | None:
    """
    Attempt to coerce a lossy-split row into exactly 6 cells.

    Strategy:
    - Identify the ISO date cell from the right.
    - Assume the status cell is immediately to its left (and matches '**Token**').
    - Merge any extra splits in the left side into the Claim/Where cells.
    - Merge any extra splits in the right side into the "What would verify/refute" cell.
    """
    if len(cells) < 6:
        return None

    # Find the last_verified cell from the right.
    last_idx = None
    for i in range(len(cells) - 1, -1, -1):
        if ISO_PREFIX_RE.match(cells[i]):
            last_idx = i
            break
    if last_idx is None or last_idx < 2:
        return None

    status_idx = last_idx - 1
    if not STATUS_PREFIX_RE.match(cells[status_idx]):
        # Fall back: search left of last_idx for a status-looking cell.
        found = None
        for i in range(last_idx - 1, -1, -1):
            if STATUS_PREFIX_RE.match(cells[i]):
                found = i
                break
        if found is None:
            return None
        status_idx = found
        last_idx = status_idx + 1
        if last_idx >= len(cells) or not ISO_PREFIX_RE.match(cells[last_idx]):
            return None

    left = [c.strip() for c in cells[:status_idx]]
    status = cells[status_idx].strip()
    last_verified = cells[last_idx].strip()
    right = [c.strip() for c in cells[last_idx + 1 :]]

    if not left or not re.fullmatch(r"C-\d{3}", left[0].strip()):
        return None

    claim_id = left[0].strip()
    left_rest = left[1:]
    if not left_rest:
        claim = ""
        where = ""
    elif len(left_rest) == 1:
        claim = left_rest[0]
        where = ""
    elif len(left_rest) == 2:
        claim, where = left_rest
    else:
        # Merge any extra accidental splits into the claim cell, keep the last as where.
        claim = "\\|".join(left_rest[:-1])
        where = left_rest[-1]

    what = "\\|".join([p for p in right if p]) if right else ""
    return [claim_id, claim, where, status, last_verified, what]


def _split_legacy_inline_experiments(claim: str, where: str) -> tuple[str, str]:
    """
    Move legacy inline experiment tags out of claim text.

    Pattern: '\\|v' marks older inline notes (v1 exp..., v11 exp..., etc).
    """
    marker = "\\|v"
    if marker not in claim:
        return claim, where
    prefix, rest = claim.split(marker, 1)
    moved = "v" + rest
    prefix = prefix.rstrip()
    moved = moved.strip()
    if not moved:
        return prefix, where
    if where.strip():
        where = where.rstrip() + "; legacy: " + moved
    else:
        where = "legacy: " + moved
    return prefix, where


def _clean_status_cell(status: str) -> str:
    """
    Keep only the canonical token and (optionally) a short allowlisted suffix.

    Legacy rows sometimes leaked fragments of math/text into the Status column due
    to accidental pipe splits. Keeping these fragments makes the matrix harder to
    audit mechanically.
    """
    m = re.match(r"^\s*\*\*([^*]+)\*\*(.*)$", status)
    if not m:
        return status.strip()
    token = m.group(1).strip()
    rest = m.group(2).strip()
    if not rest:
        return f"**{token}**"

    m_paren = re.match(r"^\(([^)]*)\)\s*$", rest)
    if not m_paren:
        return f"**{token}**"
    suffix = m_paren.group(1).strip()
    allowed = _STATUS_SUFFIX_ALLOWLIST.get(token, set())
    if suffix in allowed:
        return f"**{token}** ({suffix})"
    return f"**{token}**"


def _serialize_row(cells: list[str]) -> str:
    if len(cells) != 6:
        raise ValueError("expected 6 cells")
    escaped = [_escape_unescaped_pipes(c.strip()) for c in cells]
    return "| " + " | ".join(escaped) + " |"


def _normalize_line(line: str) -> Parsed | None:
    if not line.startswith("| C-"):
        return None
    cells = parse_table_line_cells(line)
    if not cells:
        return None
    claim_id = cells[0].strip()
    if claim_id not in LEGACY_CLAIM_IDS:
        return None

    if len(cells) != 6:
        fixed = _normalize_cells_to_six(cells)
        if fixed is None:
            return None
        cells = fixed
    else:
        cells = [c.strip() for c in cells]

    claim_id = cells[0].strip()
    claim, where = cells[1], cells[2]
    claim, where = _split_legacy_inline_experiments(claim, where)
    cells[1] = claim
    cells[2] = where
    cells[3] = _clean_status_cell(cells[3])
    return Parsed(claim_id=claim_id, cells=cells)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        default="docs/CLAIMS_EVIDENCE_MATRIX.md",
        help="Path to docs/CLAIMS_EVIDENCE_MATRIX.md.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes back to the matrix file (default: dry-run).",
    )
    args = parser.parse_args()

    matrix_path = Path(args.matrix)
    text = matrix_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    changed: list[tuple[int, str, str]] = []
    for i, line in enumerate(lines):
        parsed = _normalize_line(line)
        if parsed is None:
            continue
        new_line = _serialize_row(parsed.cells)
        if new_line != line:
            changed.append((i, parsed.claim_id, new_line))

    if not changed:
        print("OK: no legacy rows needed normalization")
        return 0

    for idx, claim_id, _ in changed[:50]:
        print(f"CHANGE: {matrix_path}:{idx+1}: {claim_id}")
    if len(changed) > 50:
        print(f"NOTE: ... plus {len(changed) - 50} more")

    if not args.apply:
        print("DRY-RUN: pass --apply to write changes")
        return 2

    changed_map = {idx: new_line for idx, _, new_line in changed}
    out_lines = [changed_map.get(i, line) for i, line in enumerate(lines)]
    matrix_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"Wrote: {matrix_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

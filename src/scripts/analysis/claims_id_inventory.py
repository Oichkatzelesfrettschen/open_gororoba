#!/usr/bin/env python3
"""
Claims ID inventory report.

Purpose:
- Make claim-by-claim auditing mechanically tractable by surfacing:
  - claim ID min/max
  - duplicate claim IDs
  - gaps in the numeric sequence (within min..max)

This is a planning aid only; it does not judge correctness.
"""

# SCRIPT_CONTRACT: {"inferred":true,"inputs":[],"network":"forbidden","network_env":"","outputs":[],"version":1}

from __future__ import annotations

import argparse
import datetime as dt
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from verification.markdown_table import iter_table_rows

_CLAIM_ID_RE = re.compile(r"^C-(\d{3})$")


@dataclass(frozen=True)
class ClaimId:
    raw: str
    n: int


def _iter_claim_ids(matrix_text: str) -> list[ClaimId]:
    out: list[ClaimId] = []
    for parsed in iter_table_rows(matrix_text):
        if len(parsed.cells) != 6:
            continue
        claim_id = parsed.cells[0].strip()
        m = _CLAIM_ID_RE.fullmatch(claim_id)
        if not m:
            continue
        out.append(ClaimId(raw=claim_id, n=int(m.group(1))))
    return out


def _render_report(matrix_path: Path, ids: list[ClaimId]) -> str:
    today = dt.date.today().isoformat()
    nums = [c.n for c in ids]
    counts = Counter(c.raw for c in ids)

    out_lines: list[str] = []
    out_lines.append(f"# Claims ID Inventory ({today})")
    out_lines.append("")
    out_lines.append(f"Matrix: `{matrix_path.as_posix()}`")
    out_lines.append(f"Total claim rows parsed: {len(ids)}")
    out_lines.append("")
    if not nums:
        out_lines.append("No claim IDs parsed (expected rows like `| C-001 | ... |`).")
        out_lines.append("")
        return "\n".join(out_lines)

    min_id = min(nums)
    max_id = max(nums)
    out_lines.append(f"- Min ID: C-{min_id:03d}")
    out_lines.append(f"- Max ID: C-{max_id:03d}")
    out_lines.append("")

    dups = sorted([cid for cid, n in counts.items() if n > 1])
    out_lines.append("## Duplicate claim IDs")
    out_lines.append("")
    if not dups:
        out_lines.append("- None")
    else:
        for cid in dups:
            out_lines.append(f"- {cid} (count={counts[cid]})")
    out_lines.append("")

    present = set(nums)
    missing_nums = [n for n in range(min_id, max_id + 1) if n not in present]
    out_lines.append("## Gaps within min..max")
    out_lines.append("")
    out_lines.append("These are missing numeric IDs between the minimum and maximum.")
    out_lines.append("")
    if not missing_nums:
        out_lines.append("- None")
        out_lines.append("")
        return "\n".join(out_lines)

    # Render in compact chunks to keep the report scannable.
    chunk_size = 24
    as_ids = [f"C-{n:03d}" for n in missing_nums]
    for i in range(0, len(as_ids), chunk_size):
        chunk = ", ".join(as_ids[i : i + chunk_size])
        out_lines.append(f"- {chunk}")
    out_lines.append("")
    return "\n".join(out_lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        default="docs/CLAIMS_EVIDENCE_MATRIX.md",
        help="Path to the claims matrix markdown file.",
    )
    parser.add_argument(
        "--out",
        default="reports/claims_id_inventory.md",
        help="Output markdown report path.",
    )
    args = parser.parse_args()

    matrix_path = Path(args.matrix)
    if not matrix_path.exists():
        raise SystemExit(f"Missing matrix: {matrix_path}")

    ids = _iter_claim_ids(matrix_path.read_text(encoding="utf-8"))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render_report(matrix_path, ids), encoding="utf-8")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


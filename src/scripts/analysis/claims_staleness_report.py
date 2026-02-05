#!/usr/bin/env python3
"""
Claims staleness report.

Purpose:
- Make claim-by-claim auditing mechanically tractable by surfacing:
  - rows with unknown last-verified dates (1970-01-01 sentinel)
  - "open" claims whose last-verified date is older than a threshold

This is a planning aid only; it does not judge correctness.
"""

# SCRIPT_CONTRACT: {"inferred":true,"inputs":[],"network":"forbidden","network_env":"","outputs":[],"version":1}

from __future__ import annotations

import argparse
import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path

from verification.claims_metadata_schema import CANONICAL_CLAIMS_STATUS_TOKENS
from verification.markdown_table import iter_table_rows

_STATUS_RE = re.compile(r"^\*\*([^*]+)\*\*")


@dataclass(frozen=True)
class Row:
    claim_id: str
    status_token: str
    last_verified: dt.date | None
    claim_short: str


def _strip_md(text: str) -> str:
    return text.replace("`", "").replace("**", "").replace("*", "").strip()


def _shorten(text: str, limit: int = 110) -> str:
    text = _strip_md(text).replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _parse_status_token(status_cell: str) -> str:
    m = _STATUS_RE.match(status_cell.strip())
    token = m.group(1).strip() if m else ""
    if token in CANONICAL_CLAIMS_STATUS_TOKENS:
        return token
    return "Other"


def _parse_iso_date_prefix(last_verified_cell: str) -> dt.date | None:
    head = last_verified_cell.strip().split()[0] if last_verified_cell.strip() else ""
    try:
        return dt.date.fromisoformat(head)
    except ValueError:
        return None


def _is_claim_id(text: str) -> bool:
    return bool(re.fullmatch(r"C-\d{3}", text.strip()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        default="docs/CLAIMS_EVIDENCE_MATRIX.md",
        help="Path to the claims matrix markdown file.",
    )
    parser.add_argument(
        "--out",
        default="reports/claims_staleness_report.md",
        help="Output markdown report path.",
    )
    parser.add_argument(
        "--stale-days",
        type=int,
        default=30,
        help="Age threshold (days) for staleness classification.",
    )
    args = parser.parse_args()

    matrix_path = Path(args.matrix)
    if not matrix_path.exists():
        raise SystemExit(f"Missing matrix: {matrix_path}")

    text = matrix_path.read_text(encoding="utf-8")
    rows: list[Row] = []
    for parsed in iter_table_rows(text):
        if len(parsed.cells) != 6:
            continue
        claim_id = parsed.cells[0].strip()
        if not _is_claim_id(claim_id):
            continue
        claim_cell = parsed.cells[1]
        status_cell = parsed.cells[3]
        last_verified_cell = parsed.cells[4]
        rows.append(
            Row(
                claim_id=claim_id,
                status_token=_parse_status_token(status_cell),
                last_verified=_parse_iso_date_prefix(last_verified_cell),
                claim_short=_shorten(claim_cell),
            )
        )

    today = dt.date.today()
    stale_days = args.stale_days
    open_tokens = {
        "Unverified",
        "Partially verified",
        "Speculative",
        "Modeled",
        "Literature",
        "Theoretical",
        "Clarified",
    }

    unknown = [r for r in rows if r.last_verified is None or r.last_verified == dt.date(1970, 1, 1)]
    stale_open = [
        r
        for r in rows
        if r.status_token in open_tokens
        and r.last_verified is not None
        and r.last_verified != dt.date(1970, 1, 1)
        and (today - r.last_verified).days >= stale_days
    ]

    def age_days(d: dt.date) -> int:
        return (today - d).days

    unknown_sorted = sorted(unknown, key=lambda r: (r.claim_id,))
    stale_open_sorted = sorted(
        stale_open,
        key=lambda r: (age_days(r.last_verified or today), r.claim_id),
        reverse=True,
    )

    out_lines: list[str] = []
    out_lines.append(f"# Claims Staleness Report ({today.isoformat()})")
    out_lines.append("")
    out_lines.append(f"Matrix: `{matrix_path.as_posix()}`")
    out_lines.append(f"Total claims parsed: {len(rows)}")
    out_lines.append(f"Stale threshold (days): {stale_days}")
    out_lines.append("")

    out_lines.append("## Unknown last-verified dates")
    out_lines.append("")
    out_lines.append(
        "Rows with missing/invalid last-verified dates, or the sentinel '1970-01-01 (unknown)'."
    )
    out_lines.append("")
    if not unknown_sorted:
        out_lines.append("- None")
        out_lines.append("")
    else:
        for r in unknown_sorted:
            out_lines.append(f"- {r.claim_id} ({r.status_token}): {r.claim_short}")
        out_lines.append("")

    out_lines.append("## Stale open claims")
    out_lines.append("")
    out_lines.append(
        "Open claims whose last-verified date is older than the threshold."
    )
    out_lines.append("")
    if not stale_open_sorted:
        out_lines.append("- None")
        out_lines.append("")
    else:
        for r in stale_open_sorted:
            assert r.last_verified is not None
            age = age_days(r.last_verified)
            out_lines.append(
                f"- {r.claim_id} ({r.status_token}, age_days={age}): {r.claim_short}"
            )
        out_lines.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

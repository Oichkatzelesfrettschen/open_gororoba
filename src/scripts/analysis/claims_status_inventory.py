#!/usr/bin/env python3
"""
Claims status inventory report.

Generates a small, ASCII-only snapshot of the claims matrix for planning:
- counts by canonical status token
- lists of "open" claims (Unverified / Partially verified / Speculative / Modeled /
  Literature / Theoretical / Clarified)

This script does not judge correctness; it only summarizes declared states.
"""

# SCRIPT_CONTRACT: {"inferred":true,"inputs":[],"network":"forbidden","network_env":"","outputs":[],"version":1}

from __future__ import annotations

import argparse
import datetime as dt
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from verification.claims_metadata_schema import CANONICAL_CLAIMS_STATUS_TOKENS
from verification.markdown_table import iter_table_rows


@dataclass(frozen=True)
class ClaimRow:
    claim_id: str
    status_token: str
    last_verified: str
    claim_short: str


_STATUS_RE = re.compile(r"^\*\*([^*]+)\*\*")


def _strip_md(text: str) -> str:
    # Conservative: good enough for short summaries.
    return text.replace("`", "").replace("**", "").replace("*", "").strip()


def _shorten(text: str, limit: int = 110) -> str:
    text = _strip_md(text).replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _parse_rows(md_text: str) -> list[ClaimRow]:
    out: list[ClaimRow] = []
    for parsed in iter_table_rows(md_text):
        if not parsed.cells:
            continue
        if len(parsed.cells) != 6:
            # Matrix formatting is enforced by a verifier; skip partials here.
            continue
        claim_id = parsed.cells[0].strip()
        if not re.fullmatch(r"C-\d{3}", claim_id):
            continue

        claim_cell = parsed.cells[1]
        status_cell = parsed.cells[3].strip()
        last_verified_cell = parsed.cells[4].strip()

        m = _STATUS_RE.match(status_cell)
        status_token = m.group(1).strip() if m else ""
        if status_token not in CANONICAL_CLAIMS_STATUS_TOKENS:
            status_token = "Other"

        # Last-verified begins with ISO date; take the prefix token for sorting.
        last_verified = last_verified_cell.split()[0] if last_verified_cell else ""

        out.append(
            ClaimRow(
                claim_id=claim_id,
                status_token=status_token,
                last_verified=last_verified,
                claim_short=_shorten(claim_cell),
            )
        )
    return out


@dataclass(frozen=True)
class HygieneAnomalies:
    missing_where_paths: list[str]
    legacy_inline_experiments: list[str]


def _collect_hygiene_anomalies(md_text: str) -> HygieneAnomalies:
    missing_where_paths: list[str] = []
    legacy_inline_experiments: list[str] = []

    for parsed in iter_table_rows(md_text):
        if not parsed.cells:
            continue
        claim_id = parsed.cells[0].strip()
        if not re.fullmatch(r"C-\d{3}", claim_id):
            continue
        if len(parsed.cells) != 6:
            # A format error is handled by the verifier; skip here.
            continue

        claim_cell = parsed.cells[1]
        where_cell = parsed.cells[2]

        # Heuristic: a "Where stated" cell should contain at least one stable pointer
        # (repo path in backticks or a clear path prefix).
        where_has_path = (
            "`" in where_cell
            or "src/" in where_cell
            or "docs/" in where_cell
            or "data/" in where_cell
        )
        if not where_has_path and "multiple docs" not in where_cell:
            missing_where_paths.append(claim_id)

        # Legacy patterns where experiment notes were embedded into the claim cell.
        if "\\|v" in claim_cell:
            legacy_inline_experiments.append(claim_id)

    return HygieneAnomalies(
        missing_where_paths=missing_where_paths,
        legacy_inline_experiments=legacy_inline_experiments,
    )


def _date_key(date_str: str) -> tuple[int, int, int]:
    # Treat missing/invalid dates as very old.
    try:
        d = dt.date.fromisoformat(date_str)
        return (d.year, d.month, d.day)
    except ValueError:
        return (0, 0, 0)

def _chunks(items: list[str], n: int) -> list[list[str]]:
    if n <= 0:
        raise ValueError("chunk size must be positive")
    return [items[i : i + n] for i in range(0, len(items), n)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        default="docs/CLAIMS_EVIDENCE_MATRIX.md",
        help="Path to the claims matrix markdown file.",
    )
    parser.add_argument(
        "--out",
        default="reports/claims_status_inventory.md",
        help="Output markdown report path.",
    )
    args = parser.parse_args()

    matrix_path = Path(args.matrix)
    if not matrix_path.exists():
        raise SystemExit(f"Missing matrix: {matrix_path}")

    rows = _parse_rows(matrix_path.read_text(encoding="utf-8"))
    if not rows:
        raise SystemExit("No claim rows parsed. Matrix format may have changed.")

    matrix_text = matrix_path.read_text(encoding="utf-8")
    hygiene = _collect_hygiene_anomalies(matrix_text)

    counts = Counter(r.status_token for r in rows)
    open_tokens = {
        "Unverified",
        "Partially verified",
        "Speculative",
        "Modeled",
        "Literature",
        "Theoretical",
        "Clarified",
    }
    open_rows = [r for r in rows if r.status_token in open_tokens]
    open_rows_sorted = sorted(open_rows, key=lambda r: (_date_key(r.last_verified), r.claim_id))

    today = dt.date.today().isoformat()
    out_lines: list[str] = []
    out_lines.append(f"# Claims Status Inventory ({today})")
    out_lines.append("")
    out_lines.append(f"Matrix: `{matrix_path.as_posix()}`")
    out_lines.append(f"Total claims parsed: {len(rows)}")
    out_lines.append("")
    out_lines.append("## Counts by canonical status token")
    out_lines.append("")
    for token in CANONICAL_CLAIMS_STATUS_TOKENS:
        v = counts.get(token, 0)
        if v:
            out_lines.append(f"- {token}: {v}")
    if counts.get("Other", 0):
        out_lines.append(f"- Other: {counts['Other']}")
    out_lines.append("")
    out_lines.append("## Open claims (oldest first)")
    out_lines.append("")
    out_lines.append(
        "This list includes claims marked Unverified, Partially verified, Speculative, "
        "Modeled, Literature, Theoretical, or Clarified."
    )
    out_lines.append("")
    for r in open_rows_sorted:
        lv = r.last_verified or "UNKNOWN_DATE"
        out_lines.append(f"- {r.claim_id} ({r.status_token}, last_verified={lv}): {r.claim_short}")
    out_lines.append("")
    out_lines.append("## Parsing notes")
    out_lines.append("")
    out_lines.append("- This report is a planning snapshot, not evidence.")
    out_lines.append("- If a matrix row includes extra '|' characters, parsing may skip it.")
    out_lines.append("")
    out_lines.append("## Metadata hygiene notes")
    out_lines.append("")
    out_lines.append(
        "These are *formatting* and *traceability* issues that make claim-by-claim audits harder."
    )
    out_lines.append("")
    missing_where_count = len(hygiene.missing_where_paths)
    out_lines.append(f"- Rows with missing/weak `Where stated` pointers: {missing_where_count}")
    out_lines.append(
        f"- Rows with legacy inline experiment tags in claim text (contains \\\\|v): "
        f"{len(hygiene.legacy_inline_experiments)}"
    )
    out_lines.append("")
    if hygiene.missing_where_paths:
        out_lines.append("- Missing `Where stated` pointers (all):")
        missing_sorted = sorted(hygiene.missing_where_paths)
        for chunk in _chunks(missing_sorted, 12):
            out_lines.append("  - " + ", ".join(chunk))
        out_lines.append("")
    if hygiene.legacy_inline_experiments:
        out_lines.append("- Example legacy inline experiment tags (first 20):")
        out_lines.append("  - " + ", ".join(hygiene.legacy_inline_experiments[:20]))
        out_lines.append("")
    out_lines.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

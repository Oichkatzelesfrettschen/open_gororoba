#!/usr/bin/env python3
"""
Claims status-contradiction report.

Purpose:
- Detect rows where the status cell contains multiple canonical status tokens.

Example of a high-signal anomaly:
  **Verified** (Unverified ... )

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

_LEADING_TOKEN_RE = re.compile(r"^\*\*([^*]+)\*\*")


@dataclass(frozen=True)
class Hit:
    claim_id: str
    leading_token: str
    other_tokens: tuple[str, ...]
    status_cell: str


def _token_pattern(token: str) -> re.Pattern[str]:
    # Match canonical tokens as whole words, allowing whitespace.
    escaped = re.escape(token)
    # Convert escaped spaces to flexible whitespace.
    escaped = escaped.replace(r"\ ", r"\s+")
    return re.compile(rf"\b{escaped}\b", flags=re.IGNORECASE)


_TOKEN_PATTERNS: dict[str, re.Pattern[str]] = {
    t: _token_pattern(t) for t in CANONICAL_CLAIMS_STATUS_TOKENS
}


def _is_claim_id(text: str) -> bool:
    return bool(re.fullmatch(r"C-\d{3}", text.strip()))


def _find_contradictions(matrix_text: str) -> list[Hit]:
    hits: list[Hit] = []
    for parsed in iter_table_rows(matrix_text):
        if len(parsed.cells) != 6:
            continue
        claim_id = parsed.cells[0].strip()
        if not _is_claim_id(claim_id):
            continue
        status_cell = parsed.cells[3].strip()
        m = _LEADING_TOKEN_RE.match(status_cell)
        if not m:
            continue
        leading = m.group(1).strip()
        if leading not in CANONICAL_CLAIMS_STATUS_TOKENS:
            continue

        remainder = status_cell[m.end() :].strip()
        found: list[str] = []
        for token in CANONICAL_CLAIMS_STATUS_TOKENS:
            if token == leading:
                continue
            if _TOKEN_PATTERNS[token].search(remainder):
                found.append(token)
        if found:
            hits.append(
                Hit(
                    claim_id=claim_id,
                    leading_token=leading,
                    other_tokens=tuple(found),
                    status_cell=status_cell,
                )
            )
    return hits


def _render_report(matrix_path: Path, hits: list[Hit]) -> str:
    today = dt.date.today().isoformat()
    out_lines: list[str] = []
    out_lines.append(f"# Claims Status Contradictions ({today})")
    out_lines.append("")
    out_lines.append(f"Matrix: `{matrix_path.as_posix()}`")
    out_lines.append(f"Hits: {len(hits)}")
    out_lines.append("")
    out_lines.append(
        "This report flags rows where the status cell contains multiple canonical status tokens."
    )
    out_lines.append("")
    if not hits:
        out_lines.append("No contradictions found.")
        out_lines.append("")
        return "\n".join(out_lines)

    out_lines.append("## Hits")
    out_lines.append("")
    for h in sorted(hits, key=lambda x: x.claim_id):
        other = ", ".join(h.other_tokens)
        out_lines.append(f"- {h.claim_id}: leading={h.leading_token}, also_mentions=[{other}]")
        out_lines.append(f"  - status: `{h.status_cell}`")
    out_lines.append("")
    out_lines.append("## Notes")
    out_lines.append("")
    out_lines.append(
        "- Not every multi-token row is logically inconsistent, but they tend to be confusing."
    )
    out_lines.append(
        "- Prefer one canonical token + a parenthetical that does not reuse other tokens."
    )
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
        default="reports/claims_status_contradictions.md",
        help="Output markdown report path.",
    )
    args = parser.parse_args()

    matrix_path = Path(args.matrix)
    if not matrix_path.exists():
        raise SystemExit(f"Missing matrix: {matrix_path}")

    hits = _find_contradictions(matrix_path.read_text(encoding="utf-8"))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render_report(matrix_path, hits), encoding="utf-8")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

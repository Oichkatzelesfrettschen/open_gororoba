#!/usr/bin/env python3
"""
Claims bold-token inventory report.

Purpose:
- Identify bold tokens (Markdown **TOKEN**) present in the claims matrix that are
  *not* canonical status tokens.

Rationale:
- The status column is strictly canonicalized by verifiers, but legacy rows may
  embed bold markers like **CONFIRMED** inside other cells (often as escaped-pipe
  delimited mini-fields). This report makes cleanup tractable without changing
  semantics.

This script does not judge correctness; it only inventories formatting tokens.
"""

# SCRIPT_CONTRACT: {"inferred":true,"inputs":[],"network":"forbidden","network_env":"","outputs":[],"version":1}

from __future__ import annotations

import argparse
import datetime as dt
import re
from collections import Counter, defaultdict
from pathlib import Path

from verification.claims_metadata_schema import CANONICAL_CLAIMS_STATUS_TOKENS
from verification.markdown_table import iter_table_rows

_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")


def _strip_md(text: str) -> str:
    return text.replace("`", "").replace("**", "").replace("*", "").strip()


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
        default="reports/claims_bold_tokens_inventory.md",
        help="Output markdown report path.",
    )
    args = parser.parse_args()

    matrix_path = Path(args.matrix)
    if not matrix_path.exists():
        raise SystemExit(f"Missing matrix: {matrix_path}")

    matrix_text = matrix_path.read_text(encoding="utf-8")
    piped_token_to_claims: dict[str, set[str]] = defaultdict(set)
    other_token_to_claims: dict[str, set[str]] = defaultdict(set)

    for parsed in iter_table_rows(matrix_text):
        if len(parsed.cells) != 6:
            continue
        claim_id = parsed.cells[0].strip()
        if not _is_claim_id(claim_id):
            continue
        for cell in parsed.cells:
            for m in _BOLD_RE.finditer(cell):
                t = _strip_md(m.group(1))
                if not t:
                    continue
                if t in CANONICAL_CLAIMS_STATUS_TOKENS:
                    continue
                # Legacy "mini-field" tags are often written as escaped-pipe delimited:
                #   ... \\|**CONFIRMED**\\| ...
                is_piped = cell[max(0, m.start() - 2) : m.start()] == "\\|"
                if is_piped:
                    piped_token_to_claims[t].add(claim_id)
                else:
                    other_token_to_claims[t].add(claim_id)

    piped_counts = Counter({t: len(cids) for t, cids in piped_token_to_claims.items()})
    other_counts = Counter({t: len(cids) for t, cids in other_token_to_claims.items()})

    today = dt.date.today().isoformat()
    out_lines: list[str] = []
    out_lines.append(f"# Claims Bold-Token Inventory ({today})")
    out_lines.append("")
    out_lines.append(f"Matrix: `{matrix_path.as_posix()}`")
    out_lines.append(f"Non-canonical bold tokens: {len(piped_counts) + len(other_counts)}")
    out_lines.append("")
    out_lines.append("## Escaped-pipe tags (by frequency)")
    out_lines.append("")
    if not piped_counts:
        out_lines.append("- None")
        out_lines.append("")
    else:
        for token, n in piped_counts.most_common():
            claims = sorted(piped_token_to_claims[token])
            sample = ", ".join(claims[:24])
            suffix = "" if len(claims) <= 24 else f", ... (+{len(claims) - 24})"
            out_lines.append(f"- {token}: {n} claim(s) ({sample}{suffix})")
        out_lines.append("")

    out_lines.append("## Other bold spans (curated)")
    out_lines.append("")
    if not other_counts:
        out_lines.append("- None")
        out_lines.append("")
    else:
        # Avoid dumping sentence-length bold spans; keep this report scannable.
        max_len = 48
        max_words = 7
        kept_tokens = [
            t for t in other_counts
            if len(t) <= max_len and len(t.split()) <= max_words
        ]
        kept_counts = Counter({t: other_counts[t] for t in kept_tokens})
        skipped = len(other_counts) - len(kept_counts)
        if kept_counts:
            for token, n in kept_counts.most_common():
                claims = sorted(other_token_to_claims[token])
                sample = ", ".join(claims[:24])
                suffix = "" if len(claims) <= 24 else f", ... (+{len(claims) - 24})"
                out_lines.append(f"- {token}: {n} claim(s) ({sample}{suffix})")
        else:
            out_lines.append("- None (all bold spans were long)")
        if skipped:
            out_lines.append("")
            out_lines.append(f"Skipped long bold spans: {skipped}")
        out_lines.append("")

    out_lines.append("## Notes")
    out_lines.append("")
    out_lines.append("- Canonical status tokens are intentionally excluded.")
    out_lines.append(
        "- If tokens are legacy inline fields, prefer moving them to the notes file or to "
        "`docs/CLAIMS_TASKS.md` with explicit task linkage."
    )
    out_lines.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Build a single, canonical, machine-parseable claims matrix.

Why:
- `docs/CLAIMS_EVIDENCE_MATRIX.md` historically accumulated multiple tables with
  varying schemas, which makes automated audits brittle.
- This script extracts all claim IDs C-001..C-427 from a notes file and emits a
  uniform 6-column table with canonical Status tokens and ISO date prefixes.

Inputs:
- A "notes" markdown file that may contain multiple tables mentioning `C-XXX`.

Outputs:
- A canonical `docs/CLAIMS_EVIDENCE_MATRIX.md` with one table:
  | ID | Claim | Where stated | Status | Last verified | What would verify/refute it |
"""

# SCRIPT_CONTRACT: {"inferred":true,"inputs":[],"network":"forbidden","network_env":"","outputs":[],"version":1}

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

CANONICAL_STATUS_TOKENS = (
    "Verified",
    "Partially verified",
    "Unverified",
    "Speculative",
    "Modeled",
    "Literature",
    "Theoretical",
    "Not supported",
    "Refuted",
    "Clarified",
    "Established",
)

_DATE_RE = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")
_CID_RE = re.compile(r"^\|\s*(C-\d{3})\s*\|")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")

_TOKEN_NORMALIZATION: list[tuple[str, str]] = [
    ("verified", "Verified"),
    ("confirmed", "Verified"),
    ("refuted", "Refuted"),
    ("not supported", "Not supported"),
    ("rejected", "Not supported"),
    ("partial", "Partially verified"),
    ("unverified", "Unverified"),
    ("speculative", "Speculative"),
    ("suggestive", "Speculative"),
    ("weak", "Speculative"),
    ("modeled", "Modeled"),
    ("modelled", "Modeled"),
    ("literature", "Literature"),
    ("theoretical", "Theoretical"),
    ("clarified", "Clarified"),
    ("established", "Established"),
]


@dataclass(frozen=True)
class CanonicalRow:
    claim_id: str
    claim: str
    where_stated: str
    status: str
    last_verified: str
    what_would_verify: str


def _escape_cell_pipes(text: str) -> str:
    # Normalize any already-escaped pipes to a single backslash, then escape any
    # remaining literal pipes. This avoids creating '\\\\|' which does not escape
    # the pipe in common markdown-table parsers.
    text = re.sub(r"\\+\|", r"\\|", text)
    return re.sub(r"(?<!\\)\|", r"\\|", text)


def _assert_ascii(text: str, context: str) -> None:
    bad = [ch for ch in text if ord(ch) > 127]
    if not bad:
        return
    sample = "".join(sorted(set(bad)))[:20]
    raise SystemExit(f"ERROR: Non-ASCII output in {context}: {sample!r}")


def _extract_first_date(text: str) -> str:
    m = _DATE_RE.search(text)
    return m.group(0) if m else ""


def _normalize_status(raw: str) -> str:
    s = raw.strip()
    label = s
    tail = ""

    if s.startswith("**") and "**" in s[2:]:
        end = s.find("**", 2)
        label = s[2:end]
        tail = s[end + 2 :].strip()
    else:
        # If the row includes bold tokens elsewhere, prefer them.
        bolds = _BOLD_RE.findall(s)
        if bolds:
            label = bolds[0]

    label_stripped = label.strip()
    low = label_stripped.lower()

    token = None
    for needle, mapped in _TOKEN_NORMALIZATION:
        if needle in low:
            token = mapped
            break
    if token is None:
        token = "Unverified" if not low else "Speculative"

    note = ""
    if token == "Verified":
        rest = re.sub(r"(?i)\b(verified|confirmed)\b", "", label_stripped).strip()
        if rest:
            note = rest
    elif token == "Partially verified":
        rest = re.sub(r"(?i)\b(partial|partially verified)\b", "", label_stripped).strip()
        if rest:
            note = rest
    elif token == "Not supported":
        rest = re.sub(r"(?i)\b(not supported|rejected)\b", "", label_stripped).strip()
        note = f"rejected {rest}".strip() if rest else "rejected"
    elif token == "Refuted":
        rest = re.sub(r"(?i)\brefuted\b", "", label_stripped).strip()
        if rest:
            note = rest
    else:
        rest = re.sub(r"(?i)\b" + re.escape(token) + r"\b", "", label_stripped).strip()
        if rest:
            note = rest

    if tail:
        note = (note + " " + tail).strip() if note else tail

    note = note.strip()
    if note and not (note.startswith("(") and note.endswith(")")):
        note = f"({note})"

    if token not in CANONICAL_STATUS_TOKENS:
        raise SystemExit(f"BUG: produced non-canonical token: {token}")
    return f"**{token}**{(' ' + note) if note else ''}"


def _parse_row_cells_naive(line: str) -> list[str]:
    # Tolerant split: does not understand escapes. We'll handle "extra pipes"
    # by merging later based on the active table schema.
    return [c.strip() for c in line.strip().split("|")[1:-1]]


def _parse_table_line_cells(line: str) -> list[str]:
    """
    Parse a markdown-table row into cells, respecting escaped pipes ('\\|') and
    code spans (backticks).
    """
    if not (line.startswith("|") and line.rstrip().endswith("|")):
        return []

    cells: list[str] = []
    buf: list[str] = []
    escaped = False
    in_code = False

    i = 1  # skip leading pipe
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
            cells.append("".join(buf).strip())
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1

    if cells and cells[-1] == "":
        cells = cells[:-1]
    return cells


def _merge_to_n_cells(cells: list[str], n: int, merge_index: int) -> list[str]:
    """
    If cells has more than n entries, merge extras into cells[merge_index]
    using a literal '|' join.
    """
    if len(cells) <= n:
        return cells
    if merge_index < 0 or merge_index >= n:
        raise ValueError("merge_index out of range")
    # Keep left side up to merge_index, merge middle, keep right side to fit n.
    left = cells[:merge_index]
    right_count = n - merge_index - 1
    right = cells[-right_count:] if right_count > 0 else []
    middle = cells[merge_index : len(cells) - len(right)]
    merged = "|".join(middle).strip()
    out = left + [merged] + right
    return out


def _score_candidate(row: CanonicalRow) -> int:
    score = 0
    if "docs/external_sources/" in row.where_stated:
        score += 10
    if "`" in row.where_stated or "src/" in row.where_stated or "docs/" in row.where_stated:
        score += 5
    if row.last_verified and not row.last_verified.startswith("1970-01-01"):
        score += 2
    if row.claim and len(row.claim) > 30:
        score += 1
    if row.status.startswith("**") and "**" in row.status[2:]:
        score += 1
    return score


def _render_canonical(rows: list[CanonicalRow]) -> str:
    out: list[str] = []
    out.append("# Claims / Evidence Matrix (Canonical)")
    out.append("")
    out.append("This file is the machine-parseable source of truth for claims C-001..C-427.")
    out.append("The expanded narrative and legacy tables live in the corresponding notes file.")
    out.append("")
    out.append("Legend (canonical tokens):")
    out.append("- **Verified**: reproducible computation or first-party source cited.")
    out.append("- **Partially verified**: evidence exists; gaps remain (e.g., provenance).")
    out.append("- **Unverified**: no reliable evidence yet (needs sourcing + tests).")
    out.append("- **Speculative**: conjecture; may remain unverified but must be labeled.")
    out.append("- **Modeled**: toy model or simulation (not necessarily physically validated).")
    out.append("- **Literature**: literature claim (not yet reproduced in-repo).")
    out.append("- **Theoretical**: theory/blueprint claim (no experimental validation).")
    out.append("- **Not supported**: negative result or insufficient signal in-repo.")
    out.append("- **Refuted**: contradicted by strong evidence or established constraints.")
    out.append("- **Clarified**: corrected/renamed to avoid false statements; may still be open.")
    out.append("- **Established**: standard background (cite sources; may not be unit-tested).")
    out.append("")
    out.append("Metadata policy:")
    out.append("- Status cells begin with `**Token**` from the canonical set above.")
    out.append("- Last verified begins with ISO date `YYYY-MM-DD` (use `1970-01-01` if unknown).")
    out.append("")
    out.append(
        "| ID | Claim | Where stated | Status | Last verified | "
        "What would verify/refute it |"
    )
    out.append("|---:|---|---|---|---|---|")
    for r in rows:
        out.append(
            f"| {r.claim_id} | {r.claim} | {r.where_stated} | {r.status} | "
            f"{r.last_verified} | {r.what_would_verify} |"
        )
    out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True, help="Input notes markdown path.")
    parser.add_argument(
        "--out", dest="out_path", required=True, help="Output canonical matrix path."
    )
    parser.add_argument(
        "--range",
        dest="id_range",
        default="1-408",
        help="Inclusive claim id range, e.g. 1-408.",
    )
    args = parser.parse_args()

    m = re.fullmatch(r"(\d+)-(\d+)", args.id_range.strip())
    if not m:
        raise SystemExit("ERROR: --range must be like 1-408")
    start_id = int(m.group(1))
    end_id = int(m.group(2))
    if start_id < 0 or end_id < start_id:
        raise SystemExit("ERROR: invalid --range")

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    text = in_path.read_text(encoding="utf-8")

    # Track the most recent ISO date seen in surrounding text as a fallback.
    context_date = ""
    # Track the current table schema based on the latest header line.
    header_cells: list[str] | None = None

    best_by_id: dict[str, CanonicalRow] = {}
    best_score: dict[str, int] = {}

    for line in text.splitlines():
        d = _extract_first_date(line)
        if d:
            context_date = d

        if line.startswith("|") and not line.lstrip().startswith("|---"):
            # Capture table headers to interpret subsequent C-rows.
            header_candidate = _parse_table_line_cells(line)
            if header_candidate and any(c.strip() == "ID" for c in header_candidate):
                header_cells = header_candidate
                continue

        m_cid = _CID_RE.match(line)
        if not m_cid:
            continue
        claim_id = m_cid.group(1)

        cells = _parse_table_line_cells(line)
        if not cells:
            cells = _parse_row_cells_naive(line)
        if header_cells is None:
            # Best-effort default: assume the canonical schema.
            header_cells = [
                "ID",
                "Claim",
                "Where stated",
                "Status",
                "Last verified",
                "What would verify/refute it",
            ]

        expected = len(header_cells)
        # Heuristic merge: pipes most often appear in Claim/Notes, so merge into the 2nd column.
        if len(cells) != expected:
            merge_idx = 1 if expected >= 2 else 0
            cells = _merge_to_n_cells(cells, n=expected, merge_index=merge_idx)

        # If we still do not match, skip (but keep going).
        if len(cells) != expected:
            continue

        colnames = [c.lower() for c in header_cells]
        claim = ""
        where = ""
        status_raw = ""
        last_verified = ""
        what = ""

        if "where stated" in colnames and "what would verify/refute it" in colnames:
            # Canonical schema.
            claim = cells[1]
            where = cells[2]
            status_raw = cells[3]
            last_verified = cells[4] if expected >= 6 else context_date
            what = cells[5] if expected >= 6 else cells[4]
        elif "claim tested" in colnames and "verdict" in colnames and "key metric" in colnames:
            # Experiment-results schema.
            claim = cells[colnames.index("claim tested")]
            where = cells[colnames.index("experiment")]
            status_raw = cells[colnames.index("verdict")]
            key_metric = cells[colnames.index("key metric")]
            notes = cells[colnames.index("notes")]
            last_verified = _extract_first_date(line) or context_date or "1970-01-01 (unknown)"
            what = f"Key metric: {key_metric}. Notes: {notes}".strip()
        elif "status" in colnames and "what would verify/refute it" in colnames:
            # Legacy 5-col schema (no explicit last-verified).
            claim = cells[colnames.index("claim")]
            where = cells[colnames.index("where stated")]
            status_raw = cells[colnames.index("status")]
            last_verified = _extract_first_date(line) or context_date or "1970-01-01 (unknown)"
            what = cells[colnames.index("what would verify/refute it")]
        else:
            # Unknown schema: skip.
            continue

        status = _normalize_status(status_raw)
        status = _escape_cell_pipes(status)
        lv = _extract_first_date(last_verified) or _extract_first_date(line) or context_date
        if not lv:
            lv = "1970-01-01 (unknown)"
        else:
            lv = lv

        row = CanonicalRow(
            claim_id=claim_id,
            claim=_escape_cell_pipes(claim.strip()),
            where_stated=_escape_cell_pipes(where.strip()),
            status=status.strip(),
            last_verified=lv.strip(),
            what_would_verify=_escape_cell_pipes(what.strip()),
        )

        # Policy backstop: open claims should always point to an external-sources
        # index, even if only to an inbox-style aggregator.
        low_status = row.status.lower()
        is_open = (
            ("unverified" in low_status)
            or ("speculative" in low_status)
            or ("partial" in low_status)
        )
        if is_open and "docs/external_sources/" not in row.where_stated:
            inbox = "`docs/external_sources/OPEN_CLAIMS_SOURCES.md`"
            where2 = row.where_stated.strip()
            where2 = inbox if not where2 else f"{where2}, {inbox}"
            row = CanonicalRow(
                claim_id=row.claim_id,
                claim=row.claim,
                where_stated=where2,
                status=row.status,
                last_verified=row.last_verified,
                what_would_verify=row.what_would_verify,
            )

        score = _score_candidate(row)
        prev = best_score.get(claim_id, -1)
        if score > prev:
            best_by_id[claim_id] = row
            best_score[claim_id] = score

    # Fill missing IDs with placeholders.
    out_rows: list[CanonicalRow] = []
    for i in range(start_id, end_id + 1):
        cid = f"C-{i:03d}"
        if cid in best_by_id:
            out_rows.append(best_by_id[cid])
            continue
        out_rows.append(
            CanonicalRow(
                claim_id=cid,
                claim="MISSING: claim text not found in notes (needs extraction).",
                where_stated="",
                status="**Unverified**",
                last_verified="1970-01-01 (unknown)",
                what_would_verify="Recover this claim from the notes file and add sources/tests.",
            )
        )

    out_text = _render_canonical(out_rows)
    _assert_ascii(out_text, context=str(out_path))
    out_path.write_text(out_text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

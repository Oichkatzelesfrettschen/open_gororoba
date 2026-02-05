#!/usr/bin/env python3
"""
Claims batch backlog report.

Goal:
- Make claim-by-claim auditing mechanically tractable by producing a focused,
  ASCII-only planning report for a contiguous claim-id range.

Non-goals:
- This script does not judge correctness.
- This script does not fetch sources (network stays out of tests).

Inputs:
- docs/CLAIMS_EVIDENCE_MATRIX.md (authoritative claim registry)
- docs/claims/CLAIMS_DOMAIN_MAP.csv (claim -> domain tags)

Output:
- reports/*.md (planning snapshot only)
"""

# SCRIPT_CONTRACT: {"inferred":true,"inputs":[],"network":"forbidden","network_env":"","outputs":[],"version":1}

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path

from verification.claims_metadata_schema import CANONICAL_CLAIMS_STATUS_TOKENS
from verification.markdown_table import iter_table_rows


@dataclass(frozen=True)
class ClaimRow:
    claim_id: str
    claim_text: str
    where_stated: str
    status_cell: str
    status_token: str
    last_verified: str
    evidence_notes: str


_STATUS_RE = re.compile(r"^\*\*([^*]+)\*\*")

_OPEN_TOKENS = {
    "Unverified",
    "Partially verified",
    "Speculative",
    "Modeled",
    "Literature",
    "Theoretical",
    "Clarified",
}


def _strip_md(text: str) -> str:
    # Conservative: good enough for short planning snippets.
    return text.replace("`", "").replace("**", "").replace("*", "").strip()


def _shorten(text: str, limit: int = 120) -> str:
    text = _strip_md(text).replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _claim_num(claim_id: str) -> int:
    # claim_id is validated upstream by regex.
    return int(claim_id.split("-")[1])


def _parse_rows(md_text: str) -> list[ClaimRow]:
    out: list[ClaimRow] = []
    for parsed in iter_table_rows(md_text):
        if not parsed.cells:
            continue
        if len(parsed.cells) != 6:
            continue
        claim_id = parsed.cells[0].strip()
        if not re.fullmatch(r"C-\d{3}", claim_id):
            continue

        claim_text = parsed.cells[1]
        where_stated = parsed.cells[2]
        status_cell = parsed.cells[3].strip()
        last_verified_cell = parsed.cells[4].strip()
        evidence_notes = parsed.cells[5]

        m = _STATUS_RE.match(status_cell)
        status_token = m.group(1).strip() if m else ""
        if status_token not in CANONICAL_CLAIMS_STATUS_TOKENS:
            status_token = "Other"

        last_verified = last_verified_cell.split()[0] if last_verified_cell else ""

        out.append(
            ClaimRow(
                claim_id=claim_id,
                claim_text=claim_text,
                where_stated=where_stated,
                status_cell=status_cell,
                status_token=status_token,
                last_verified=last_verified,
                evidence_notes=evidence_notes,
            )
        )
    return out


def _load_domain_map(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    out: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            claim_id = (row.get("claim_id") or "").strip()
            domains_raw = (row.get("domains") or "").strip()
            if not claim_id:
                continue
            domains = [d.strip() for d in domains_raw.split(";") if d.strip()]
            out[claim_id] = domains
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="docs/CLAIMS_EVIDENCE_MATRIX.md")
    parser.add_argument("--domain-map", default="docs/claims/CLAIMS_DOMAIN_MAP.csv")
    parser.add_argument(
        "--id-from",
        type=int,
        required=True,
        help="Start claim number (e.g. 1 for C-001).",
    )
    parser.add_argument(
        "--id-to",
        type=int,
        required=True,
        help="End claim number (inclusive).",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output markdown path under reports/.",
    )
    args = parser.parse_args()

    if args.id_from <= 0 or args.id_to <= 0:
        raise SystemExit("ERROR: --id-from/--id-to must be positive integers")
    if args.id_to < args.id_from:
        raise SystemExit("ERROR: --id-to must be >= --id-from")

    repo_root = Path(__file__).resolve().parents[3]
    matrix_path = (repo_root / args.matrix).resolve()
    domain_map_path = (repo_root / args.domain_map).resolve()

    if not matrix_path.exists():
        raise SystemExit(f"ERROR: missing matrix: {matrix_path}")

    out_path = Path(args.out) if args.out else Path(
        f"reports/claims_batch_backlog_C{args.id_from:03}_C{args.id_to:03}.md"
    )
    out_path = (repo_root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    domain_map = _load_domain_map(domain_map_path)
    rows = _parse_rows(matrix_path.read_text(encoding="utf-8"))
    if not rows:
        raise SystemExit("ERROR: no claim rows parsed; matrix format may have changed")

    in_range = [r for r in rows if args.id_from <= _claim_num(r.claim_id) <= args.id_to]
    in_range_sorted = sorted(in_range, key=lambda r: _claim_num(r.claim_id))
    open_rows = [r for r in in_range_sorted if r.status_token in _OPEN_TOKENS]

    today = dt.date.today().isoformat()
    range_label = f"C-{args.id_from:03}..C-{args.id_to:03}"

    lines: list[str] = []
    lines.append(f"# Claims Batch Backlog ({range_label}) ({today})")
    lines.append("")
    lines.append("Purpose: planning snapshot for claim-by-claim audits (not evidence).")
    lines.append("")
    lines.append(f"- Matrix: `{matrix_path.relative_to(repo_root).as_posix()}`")
    lines.append(f"- Domain map: `{domain_map_path.relative_to(repo_root).as_posix()}`")
    lines.append(f"- Claims in range: {len(in_range_sorted)}")
    lines.append(f"- Open claims in range: {len(open_rows)}")
    lines.append("")

    lines.append("## Open claims (in-range, oldest-first by last_verified)")
    lines.append("")
    if not open_rows:
        lines.append("- (none)")
    else:
        def date_key(s: str) -> tuple[int, int, int]:
            try:
                d = dt.date.fromisoformat(s)
                return (d.year, d.month, d.day)
            except ValueError:
                return (0, 0, 0)

        open_sorted = sorted(open_rows, key=lambda r: (date_key(r.last_verified), r.claim_id))
        for r in open_sorted:
            domains = domain_map.get(r.claim_id, [])
            domains_str = ", ".join(domains) if domains else "(none)"
            lv = r.last_verified or "UNKNOWN_DATE"
            lines.append(
                f"- {r.claim_id} ({r.status_token}, last_verified={lv}, domains={domains_str}): "
                f"{_shorten(r.claim_text, limit=110)}"
            )
    lines.append("")

    lines.append("## Details (all claims in range)")
    lines.append("")
    lines.append(
        "| Claim | Domains | Status | Last verified | Claim (short) | "
        "Where stated (short) | Evidence / notes (short) |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for r in in_range_sorted:
        domains = domain_map.get(r.claim_id, [])
        domains_str = ", ".join(domains) if domains else "(none)"
        lv = r.last_verified or "UNKNOWN_DATE"
        lines.append(
            "| "
            + " | ".join(
                [
                    r.claim_id,
                    domains_str,
                    _shorten(r.status_cell, limit=28),
                    lv,
                    _shorten(r.claim_text, limit=80),
                    _shorten(r.where_stated, limit=80),
                    _shorten(r.evidence_notes, limit=80),
                ]
            )
            + " |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {out_path.relative_to(repo_root).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

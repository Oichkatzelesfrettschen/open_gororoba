#!/usr/bin/env python3
"""
Heuristic ranking of claims for audit triage.

Inputs:
- docs/CLAIMS_EVIDENCE_MATRIX.md (authoritative claim list)
- repo text files under docs/src/tests/examples (for mention frequency)

Output:
- reports/claims_priority_ranking.md

Note: This is a planning tool, not evidence.
"""

# SCRIPT_CONTRACT: {"inferred":true,"inputs":[],"network":"forbidden","network_env":"","outputs":[],"version":1}

from __future__ import annotations

import argparse
import datetime as dt
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RankedClaim:
    claim_id: str
    status_raw: str
    last_verified: str
    mention_count: int
    score: int
    claim: str


_CID_RE = re.compile(r"\bC-\d{3}\b")
_ROW_ID_RE = re.compile(r"^\|\s*(C-\d{3})\s*\|")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_DATE_RE = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")


def _strip_md(text: str) -> str:
    return (
        text.replace("**", "")
        .replace("*", "")
        .replace("`", "")
        .strip()
    )


def _select_status_candidate(candidates: list[str]) -> str:
    if not candidates:
        return ""
    keywords = [
        "verified",
        "refuted",
        "speculative",
        "unverified",
        "partial",
        "implemented",
        "tested",
        "modeled",
        "established",
        "not supported",
    ]
    for c in candidates:
        low = c.lower()
        if any(k in low for k in keywords):
            return c
    return candidates[0]


def _parse_rows(md_text: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for line in md_text.splitlines():
        if not line.startswith("|"):
            continue
        if " ID " in line and " Claim " in line and " Status " in line:
            continue
        if line.lstrip().startswith("|---"):
            continue
        m = _ROW_ID_RE.match(line)
        if not m:
            continue
        claim_id = m.group(1)
        bolds = [_strip_md(b) for b in _BOLD_RE.findall(line)]
        status_raw = _strip_md(_select_status_candidate(bolds))
        dates = _DATE_RE.findall(line)
        last_verified = dates[0] if dates else ""
        out.append(
            {
                "claim_id": claim_id.strip(),
                "claim": _strip_md(line),
                "status_raw": status_raw,
                "last_verified": last_verified,
            }
        )
    return out


def _date_key(date_str: str) -> tuple[int, int, int]:
    try:
        d = dt.date.fromisoformat(date_str)
        return (d.year, d.month, d.day)
    except ValueError:
        return (0, 0, 0)


def _iter_text_files(repo_root: Path) -> list[Path]:
    roots = [repo_root / "docs", repo_root / "src", repo_root / "tests", repo_root / "examples"]
    exts = {".md", ".py", ".tex", ".txt"}
    ignore = {
        repo_root / "docs" / "CLAIMS_EVIDENCE_MATRIX.md",
        repo_root / "docs" / "CLAIMS_TASKS.md",
        repo_root / "docs" / "VERIFIED_CLAIMS_INDEX.md",
        repo_root / "reports" / "claims_status_inventory.md",
        repo_root / "reports" / "claims_priority_ranking.md",
    }
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p in ignore:
                continue
            if p.is_file() and p.suffix in exts:
                out.append(p)
    return out


def _count_mentions(repo_root: Path) -> Counter[str]:
    c = Counter()
    for p in _iter_text_files(repo_root):
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Skip non-UTF8 text.
            continue
        for cid in _CID_RE.findall(text):
            c[cid] += 1
    return c


def _is_closed_status(status_raw: str) -> bool:
    s = status_raw.lower()
    return ("verified" in s) or ("refuted" in s) or ("established" in s)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root (default: .).",
    )
    parser.add_argument(
        "--matrix",
        default="docs/CLAIMS_EVIDENCE_MATRIX.md",
        help="Claims matrix path (relative to repo root).",
    )
    parser.add_argument(
        "--out",
        default="reports/claims_priority_ranking.md",
        help="Output markdown report path (relative to repo root).",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    matrix_path = (repo_root / args.matrix).resolve()
    out_path = (repo_root / args.out).resolve()

    rows = _parse_rows(matrix_path.read_text(encoding="utf-8"))
    mentions = _count_mentions(repo_root)

    ranked: list[RankedClaim] = []
    for r in rows:
        claim_id = r["claim_id"]
        status_raw = r["status_raw"]
        last_verified = r["last_verified"]
        mention_count = int(mentions.get(claim_id, 0))
        closed = _is_closed_status(status_raw)
        age_key = _date_key(last_verified)
        # Oldest dates => smallest key => higher priority.
        # Encode age as a 0..1000-ish penalty.
        age_score = 0
        if age_key == (0, 0, 0):
            age_score = 400
        else:
            y, m, d = age_key
            # Treat 2026-01-01 as baseline.
            try:
                days = (dt.date.today() - dt.date(y, m, d)).days
            except ValueError:
                days = 0
            age_score = min(400, max(0, days))

        score = 0
        if not closed:
            score += 1000
        score += min(500, 20 * mention_count)
        score += age_score
        ranked.append(
            RankedClaim(
                claim_id=claim_id,
                status_raw=status_raw,
                last_verified=last_verified,
                mention_count=mention_count,
                score=score,
                claim=r["claim"],
            )
        )

    ranked.sort(key=lambda x: (-x.score, x.claim_id))

    today = dt.date.today().isoformat()
    lines: list[str] = []
    lines.append(f"# Claims Priority Ranking ({today})")
    lines.append("")
    lines.append("Heuristic ranking for audit triage. Higher score means:")
    lines.append("- status is not marked Verified/Refuted/Established, and/or")
    lines.append("- claim is referenced frequently, and/or")
    lines.append("- claim has an old or missing last_verified date.")
    lines.append("")
    lines.append("This report is for planning, not evidence.")
    lines.append("")
    lines.append("## Top 80 claims to review next")
    lines.append("")
    for rc in ranked[:80]:
        lv = rc.last_verified or "UNKNOWN_DATE"
        lines.append(
            f"- {rc.claim_id} (score={rc.score}, mentions={rc.mention_count}, "
            f"last_verified={lv}): {rc.status_raw}"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Mention counts are computed from docs/src/tests/examples only.")
    lines.append(
        "- This does not attempt to estimate scientific importance; "
        "it estimates audit risk."
    )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {out_path.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

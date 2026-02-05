#!/usr/bin/env python3
"""
Replace legacy "vX expY" shorthand in the claims matrix "Where stated" column
with stable, repo-local pointers.

This keeps the claim-by-claim audit mechanically tractable by making the
"Where stated" cell linkable and grep-friendly.
"""

# SCRIPT_CONTRACT: {"inferred":true,"inputs":[],"network":"forbidden","network_env":"","outputs":[],"version":1}

from __future__ import annotations

import argparse
import re
from pathlib import Path

from verification.markdown_table import parse_table_line_cells

_CLAIM_ID_RE = re.compile(r"^C-\d{3}$")
_TICKED_PATH_RE = re.compile(r"`([^`]+)`")


_V10_PTR = (
    "`src/scripts/analysis/cd_algebraic_experiments_v10.py`, "
    "`data/csv/cd_algebraic_experiments_v10.json`"
)
_V11_PTR = (
    "`src/scripts/analysis/cd_algebraic_experiments_v11.py`, "
    "`data/csv/cd_algebraic_experiments_v11.json`"
)
_V12_PTR = (
    "`src/scripts/analysis/cd_algebraic_experiments_v12.py`, "
    "`data/csv/cd_algebraic_experiments_v12.json`"
)
_V13_PTR = (
    "`src/scripts/analysis/cd_algebraic_experiments_v13.py`, "
    "`data/csv/cd_algebraic_experiments_v13.json`"
)
_V14_PTR = (
    "`src/scripts/analysis/cd_algebraic_experiments_v14.py`, "
    "`data/csv/cd_algebraic_experiments_v14.json`"
)


WHERE_REPLACEMENTS: dict[str, str] = {
    # v10 (tensor ranks + global property table)
    "C-110": _V10_PTR,
    "C-111": _V10_PTR,
    # v11 (high-dim convergence + ZD searches)
    "C-112": _V11_PTR,
    "C-113": _V11_PTR,
    "C-114": _V11_PTR,
    "C-115": _V11_PTR,
    "C-116": _V11_PTR,
    "C-117": _V11_PTR,
    # v12 (loop / Jordan / Bol identities + bootstraps)
    "C-118": _V12_PTR,
    "C-119": _V12_PTR,
    "C-121": _V12_PTR,
    "C-122": _V12_PTR,
    # v13 (additional identities / Lie-bracket analysis)
    "C-124": _V13_PTR,
    "C-125": _V13_PTR,
    "C-126": _V13_PTR,
    "C-127": _V13_PTR,
    # v14 (universal ratios + orthogonality + ZD pair products)
    "C-131": _V14_PTR,
    "C-133": _V14_PTR,
    "C-134": _V14_PTR,
}


def _verify_paths_exist(repo_root: Path) -> list[str]:
    failures: list[str] = []
    for cid, cell in sorted(WHERE_REPLACEMENTS.items()):
        for rel_str in _TICKED_PATH_RE.findall(cell):
            rel = Path(rel_str)
            p = (repo_root / rel).resolve()
            if not p.exists():
                failures.append(f"{cid}: missing path: {rel_str}")
    return failures


def _render_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[3]),
        help="Repository root (default: inferred from this file path).",
    )
    ap.add_argument(
        "--matrix",
        default="docs/CLAIMS_EVIDENCE_MATRIX.md",
        help="Path to claims matrix markdown (repo-relative).",
    )
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    failures = _verify_paths_exist(repo_root)
    if failures:
        for msg in failures:
            print(f"ERROR: {msg}")
        return 2

    matrix_path = repo_root / args.matrix
    if not matrix_path.exists():
        print(f"ERROR: Missing matrix: {matrix_path}")
        return 2

    lines = matrix_path.read_text(encoding="utf-8").splitlines()
    updated = 0

    for idx, line in enumerate(lines):
        if not line.startswith("| C-"):
            continue
        cells = parse_table_line_cells(line)
        if len(cells) != 6:
            continue
        cid = cells[0].strip()
        if not _CLAIM_ID_RE.fullmatch(cid):
            continue
        replacement = WHERE_REPLACEMENTS.get(cid)
        if replacement is None:
            continue
        if cells[2].strip() == replacement:
            continue
        cells[2] = replacement
        lines[idx] = _render_row(cells)
        updated += 1

    if updated:
        matrix_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Updated rows: {updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

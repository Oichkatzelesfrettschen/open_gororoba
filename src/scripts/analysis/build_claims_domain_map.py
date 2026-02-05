#!/usr/bin/env python3
"""
Build a domain mapping for claims (C-001..C-427).

This produces a mechanically parseable CSV so the claim audit can be organized
without changing the canonical matrix in `docs/CLAIMS_EVIDENCE_MATRIX.md`.

Notes:
- This is a heuristic bootstrap, not a truth claim.
- The mapping should be reviewed and refined over time.
- Output is ASCII-only.
"""

# SCRIPT_CONTRACT: {"inferred":true,"inputs":[],"network":"forbidden","network_env":"","outputs":[],"version":1}

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClaimRow:
    claim_id: str
    claim: str
    where_stated: str


def _parse_table_line_strict(line: str) -> list[str]:
    if not (line.startswith("|") and line.rstrip().endswith("|")):
        raise ValueError("Row must start and end with '|'")

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

    if buf:
        raise ValueError("Trailing non-cell content after final '|'")
    if cells and cells[-1] == "":
        cells = cells[:-1]
    return cells


def _load_claim_rows(matrix_path: Path) -> list[ClaimRow]:
    rows: list[ClaimRow] = []
    for lineno, line in enumerate(matrix_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.startswith("| C-"):
            continue
        parts = _parse_table_line_strict(line)
        if len(parts) != 6:
            raise SystemExit(
                f"ERROR: {matrix_path}:{lineno}: expected 6 columns, got {len(parts)}"
            )
        claim_id = parts[0].strip()
        if not re.fullmatch(r"C-\d{3}", claim_id):
            raise SystemExit(f"ERROR: {matrix_path}:{lineno}: bad claim id: {claim_id!r}")
        rows.append(
            ClaimRow(
                claim_id=claim_id,
                claim=parts[1].strip(),
                where_stated=parts[2].strip(),
            )
        )
    return rows


DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "coq": ("coq", "rocq", "coqc"),
    "cpp": ("c++", "cmake", "conan", "cpp/"),
    "materials": (
        "metamaterial",
        "absorber",
        "salisbury",
        "tcmt",
        "jarvis",
        "nomad",
        "aflow",
    ),
    "gravitational-waves": ("gwtc", "gwosc", "ligo", "virgo", "kagra", "gravitational wave"),
    "stellar-cartography": ("tscp", "sky", "skylocalization", "dipole", "cmb"),
    "cosmology": ("lcdm", "frw", "dark energy", "hubble", "planck", "cosmolog", "w0"),
    "holography": (
        "ads/cft",
        "holograph",
        "entanglement",
        "rt",
        "hrt",
        "modular",
        "bit thread",
        "jlms",
    ),
    "open-systems": ("lindblad", "gksl", "decoherence", "pointer", "darwinism", "open-system"),
    "tensor-networks": ("tensor-network", "tensor network", "mera", "happy", "entropy scaling"),
    "spectral": ("spectral", "dirac", "laplacian", "isospectral", "connes"),
    "datasets": ("data/external", "dataset", "fetch", "snapshot", "zenodo", "doi"),
    "visualization": ("plot", "image", "dashboard", "visualization", ".png", "3160x2820"),
    "engineering": ("optimization", "benchmark", "datacenter", "latency", "performance"),
    "legacy": ("legacy", "archive/"),
    "meta": ("falsif", "look-elsewhere", "trial factor", "decision rule", "method"),
}

ALGEBRA_KEYWORDS = (
    "cayley-dickson",
    "sedenion",
    "octonion",
    "zero divisor",
    "box-kite",
    "assessor",
    "de marrais",
    "reggiani",
    "annihilator",
    "motif",
    "wheel",
)

PRIMARY_PRIORITY: tuple[str, ...] = (
    "coq",
    "cpp",
    "materials",
    "gravitational-waves",
    "stellar-cartography",
    "cosmology",
    "holography",
    "open-systems",
    "tensor-networks",
    "spectral",
    "datasets",
    "visualization",
    "engineering",
    "legacy",
    "algebra",
    "meta",
)


def _domains_for_claim(row: ClaimRow) -> list[str]:
    text = f"{row.claim} {row.where_stated}".lower()
    matches: set[str] = set()
    for domain, kws in DOMAIN_KEYWORDS.items():
        if any(kw in text for kw in kws):
            matches.add(domain)

    has_algebra = any(kw in text for kw in ALGEBRA_KEYWORDS)
    if has_algebra:
        matches.add("algebra")

    if not matches:
        matches.add("meta")

    primary = next((d for d in PRIMARY_PRIORITY if d in matches), "meta")

    ordered: list[str] = [primary]
    for d in PRIMARY_PRIORITY:
        if d == primary:
            continue
        if d in matches:
            ordered.append(d)

    # Keep the mapping compact.
    return ordered[:3]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[3]),
        help="Repository root (default: inferred from this file path).",
    )
    parser.add_argument(
        "--out",
        default="docs/claims/CLAIMS_DOMAIN_MAP.csv",
        help="Output CSV path (default: docs/claims/CLAIMS_DOMAIN_MAP.csv).",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    matrix_path = repo_root / "docs/CLAIMS_EVIDENCE_MATRIX.md"
    out_path = repo_root / args.out

    if not matrix_path.exists():
        print("ERROR: Missing docs/CLAIMS_EVIDENCE_MATRIX.md")
        return 2

    rows = _load_claim_rows(matrix_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["claim_id", "domains"])
        for row in rows:
            domains = _domains_for_claim(row)
            writer.writerow([row.claim_id, ";".join(domains)])

    print(f"OK: wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

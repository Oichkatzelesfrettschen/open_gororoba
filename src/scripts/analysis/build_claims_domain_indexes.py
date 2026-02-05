#!/usr/bin/env python3
"""
Generate per-domain claim indexes under docs/claims/by_domain/.

Inputs:
- docs/claims/CLAIMS_DOMAIN_MAP.csv
- docs/CLAIMS_EVIDENCE_MATRIX.md

Output:
- docs/claims/INDEX.md
- docs/claims/by_domain/<domain>.md

This is deterministic and offline. Output is ASCII-only.
"""

# SCRIPT_CONTRACT: {"inferred":true,"inputs":[],"network":"forbidden","network_env":"","outputs":[],"version":1}

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

CANONICAL_DOMAINS = (
    "meta",
    "algebra",
    "spectral",
    "holography",
    "open-systems",
    "tensor-networks",
    "cosmology",
    "gravitational-waves",
    "stellar-cartography",
    "materials",
    "engineering",
    "datasets",
    "visualization",
    "cpp",
    "coq",
    "legacy",
)


@dataclass(frozen=True)
class MatrixRow:
    claim_id: str
    claim: str
    where_stated: str
    status: str
    last_verified: str


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


def _load_matrix_rows(matrix_path: Path) -> dict[str, MatrixRow]:
    out: dict[str, MatrixRow] = {}
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
        out[claim_id] = MatrixRow(
            claim_id=claim_id,
            claim=parts[1].strip(),
            where_stated=parts[2].strip(),
            status=parts[3].strip(),
            last_verified=parts[4].strip(),
        )
    return out


def _load_domain_map(map_path: Path) -> dict[str, list[str]]:
    by_claim: dict[str, list[str]] = {}
    with map_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["claim_id", "domains"]:
            raise SystemExit(
                f"ERROR: {map_path}: expected header 'claim_id,domains', got {reader.fieldnames}"
            )
        for i, row in enumerate(reader, start=2):
            claim_id = (row.get("claim_id") or "").strip()
            domains_raw = (row.get("domains") or "").strip()
            if not claim_id:
                raise SystemExit(f"ERROR: {map_path}:{i}: missing claim_id")
            domains = [d.strip() for d in domains_raw.split(";") if d.strip()]
            by_claim[claim_id] = domains
    return by_claim


def _claim_sort_key(claim_id: str) -> tuple[int, str]:
    m = re.fullmatch(r"C-(\d{3})", claim_id)
    if not m:
        return (10**9, claim_id)
    return (int(m.group(1)), claim_id)


def _write_domain_file(path: Path, domain: str, claim_rows: list[MatrixRow]) -> None:
    lines: list[str] = []
    lines.append(f"# Claims: {domain}")
    lines.append("")
    lines.append("Source: docs/claims/CLAIMS_DOMAIN_MAP.csv + docs/CLAIMS_EVIDENCE_MATRIX.md")
    lines.append("")
    lines.append(f"Count: {len(claim_rows)}")
    lines.append("")
    for row in claim_rows:
        # Include status token on the same line so the overclaim detector can treat
        # speculative rows as clearly labeled.
        lines.append(
            f"- Hypothesis {row.claim_id} ({row.status}, {row.last_verified}): {row.claim}"
        )
        lines.append(f"  - Where stated: {row.where_stated}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[3]),
        help="Repository root (default: inferred from this file path).",
    )
    parser.add_argument(
        "--map",
        default="docs/claims/CLAIMS_DOMAIN_MAP.csv",
        help="Domain map CSV path (default: docs/claims/CLAIMS_DOMAIN_MAP.csv).",
    )
    parser.add_argument(
        "--matrix",
        default="docs/CLAIMS_EVIDENCE_MATRIX.md",
        help="Claims matrix path (default: docs/CLAIMS_EVIDENCE_MATRIX.md).",
    )
    parser.add_argument(
        "--out-dir",
        default="docs/claims/by_domain",
        help="Output directory for per-domain markdown files.",
    )
    parser.add_argument(
        "--index",
        default="docs/claims/INDEX.md",
        help="Domain index markdown path (default: docs/claims/INDEX.md).",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    map_path = repo_root / args.map
    matrix_path = repo_root / args.matrix
    out_dir = repo_root / args.out_dir
    index_path = repo_root / args.index

    if not map_path.exists():
        print(f"ERROR: Missing {map_path}")
        return 2
    if not matrix_path.exists():
        print(f"ERROR: Missing {matrix_path}")
        return 2

    matrix_rows = _load_matrix_rows(matrix_path)
    domain_map = _load_domain_map(map_path)

    domain_to_claim_ids: dict[str, list[str]] = {d: [] for d in CANONICAL_DOMAINS}
    for claim_id, domains in domain_map.items():
        for domain in domains:
            if domain in domain_to_claim_ids:
                domain_to_claim_ids[domain].append(claim_id)

    out_dir.mkdir(parents=True, exist_ok=True)

    counts: list[tuple[str, int]] = []
    for domain in CANONICAL_DOMAINS:
        ids = sorted(domain_to_claim_ids.get(domain, []), key=_claim_sort_key)
        rows = [matrix_rows[cid] for cid in ids if cid in matrix_rows]
        _write_domain_file(out_dir / f"{domain}.md", domain, rows)
        counts.append((domain, len(rows)))

    index_lines: list[str] = []
    index_lines.append("# Claims by domain")
    index_lines.append("")
    index_lines.append("See also: docs/CLAIMS_DOMAIN_TAXONOMY.md")
    index_lines.append("")
    for domain, n in counts:
        index_lines.append(f"- `{domain}` ({n}): `docs/claims/by_domain/{domain}.md`")
    index_lines.append("")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("\n".join(index_lines), encoding="utf-8")

    print(f"OK: wrote {index_path} and {out_dir}/<domain>.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

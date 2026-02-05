#!/usr/bin/env python3
"""
Verify claim domain mapping coverage and hygiene.

Enforces:
- `docs/claims/CLAIMS_DOMAIN_MAP.csv` exists and covers all claim IDs in the
  canonical matrix.
- Domains are chosen from a small canonical taxonomy.
- Each claim has at least one domain.

Rationale: modularize a large claim set without modifying the canonical matrix.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

CANONICAL_DOMAINS = {
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
}


def _load_claim_ids_from_matrix(matrix_path: Path) -> set[str]:
    claim_ids: set[str] = set()
    for lineno, line in enumerate(matrix_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.startswith("| C-"):
            continue
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if not parts:
            continue
        claim_id = parts[0]
        if not re.fullmatch(r"C-\d{3}", claim_id):
            raise SystemExit(f"ERROR: {matrix_path}:{lineno}: bad claim id: {claim_id!r}")
        claim_ids.add(claim_id)
    return claim_ids


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
            if not re.fullmatch(r"C-\d{3}", claim_id):
                raise SystemExit(f"ERROR: {map_path}:{i}: bad claim_id: {claim_id!r}")
            if claim_id in by_claim:
                raise SystemExit(f"ERROR: {map_path}:{i}: duplicate claim_id: {claim_id}")
            domains = [d.strip() for d in domains_raw.split(";") if d.strip()]
            by_claim[claim_id] = domains
    return by_claim


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root (default: inferred from this file path).",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    matrix_path = repo_root / "docs/CLAIMS_EVIDENCE_MATRIX.md"
    map_path = repo_root / "docs/claims/CLAIMS_DOMAIN_MAP.csv"

    if not matrix_path.exists():
        print("ERROR: Missing docs/CLAIMS_EVIDENCE_MATRIX.md")
        return 2
    if not map_path.exists():
        print("ERROR: Missing docs/claims/CLAIMS_DOMAIN_MAP.csv")
        return 2

    matrix_ids = _load_claim_ids_from_matrix(matrix_path)
    mapped = _load_domain_map(map_path)

    failures: list[str] = []

    missing = sorted(matrix_ids - set(mapped))
    extra = sorted(set(mapped) - matrix_ids)
    if missing:
        failures.append(f"Missing mappings for {len(missing)} claims (e.g., {missing[:5]})")
    if extra:
        failures.append(f"Unknown claim IDs in mapping: {extra[:10]}")

    for claim_id, domains in mapped.items():
        if not domains:
            failures.append(f"{claim_id}: missing domains list")
            continue
        bad = [d for d in domains if d not in CANONICAL_DOMAINS]
        if bad:
            failures.append(
                f"{claim_id}: unknown domains: {bad} (see docs/CLAIMS_DOMAIN_TAXONOMY.md)"
            )

    if failures:
        for msg in failures[:200]:
            print(f"ERROR: {msg}")
        if len(failures) > 200:
            print(f"ERROR: ... plus {len(failures) - 200} more")
        return 2

    print("OK: claims domain mapping covers all claims")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

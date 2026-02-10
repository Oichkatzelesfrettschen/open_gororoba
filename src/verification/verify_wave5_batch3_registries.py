#!/usr/bin/env python3
"""
Verify Wave 5 Batch 3 registries:
- registry/conflict_markers.toml
- registry/lacunae.toml
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path


def _assert_ascii(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    bad = sorted({ch for ch in text if ord(ch) > 127})
    if bad:
        sample = "".join(bad[:20])
        raise SystemExit(f"ERROR: non-ASCII content in {path}: {sample!r}")


def _load(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root.",
    )
    parser.add_argument(
        "--conflict-path",
        default="registry/conflict_markers.toml",
        help="Conflict marker registry path.",
    )
    parser.add_argument(
        "--lacunae-path",
        default="registry/lacunae.toml",
        help="Lacunae registry path.",
    )
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    claims_path = root / "registry/claims.toml"
    conflict_path = root / args.conflict_path
    lacunae_path = root / args.lacunae_path

    for path in (claims_path, conflict_path, lacunae_path):
        if not path.exists():
            raise SystemExit(f"ERROR: missing required file: {path}")

    _assert_ascii(conflict_path)
    _assert_ascii(lacunae_path)

    claims = _load(claims_path).get("claim", [])
    conflict_raw = _load(conflict_path)
    lacunae_raw = _load(lacunae_path)

    claim_ids = {str(row.get("id", "")) for row in claims}
    markers = conflict_raw.get("marker", [])
    lacunae = lacunae_raw.get("lacuna", [])

    failures: list[str] = []

    # Conflict marker checks
    meta = conflict_raw.get("conflict_markers", {})
    if int(meta.get("marker_count", -1)) != len(markers):
        failures.append("conflict_markers marker_count metadata mismatch")
    seen_marker_ids: set[str] = set()
    severity_counts = {"high": 0, "medium": 0, "low": 0}
    for row in markers:
        mid = str(row.get("id", ""))
        if mid in seen_marker_ids:
            failures.append(f"duplicate conflict marker id: {mid}")
            break
        seen_marker_ids.add(mid)
        severity = str(row.get("severity", ""))
        if severity not in severity_counts:
            failures.append(f"invalid marker severity: {mid} -> {severity}")
        else:
            severity_counts[severity] += 1
        line_start = int(row.get("line_start", 0))
        line_end = int(row.get("line_end", 0))
        if line_start < 0 or line_end < 0 or (line_start > 0 and line_end < line_start):
            failures.append(f"invalid marker line span: {mid} ({line_start}, {line_end})")
        for cref in row.get("claim_refs", []):
            if str(cref) not in claim_ids:
                failures.append(f"marker references unknown claim id: {mid} -> {cref}")
        if not str(row.get("marker_kind", "")).strip():
            failures.append(f"marker missing marker_kind: {mid}")
        if not str(row.get("source_registry", "")).strip():
            failures.append(f"marker missing source_registry: {mid}")
        if not str(row.get("status", "")).strip():
            failures.append(f"marker missing status: {mid}")
        if not isinstance(row.get("positive_evidence", []), list):
            failures.append(f"marker positive_evidence not list: {mid}")
        if not isinstance(row.get("negative_evidence", []), list):
            failures.append(f"marker negative_evidence not list: {mid}")

    if int(meta.get("high_severity_count", -1)) != severity_counts["high"]:
        failures.append("conflict_markers high_severity_count metadata mismatch")
    if int(meta.get("medium_severity_count", -1)) != severity_counts["medium"]:
        failures.append("conflict_markers medium_severity_count metadata mismatch")
    if int(meta.get("low_severity_count", -1)) != severity_counts["low"]:
        failures.append("conflict_markers low_severity_count metadata mismatch")

    # Lacuna checks
    lmeta = lacunae_raw.get("lacunae", {})
    if int(lmeta.get("lacuna_count", -1)) != len(lacunae):
        failures.append("lacunae lacuna_count metadata mismatch")
    seen_lacuna_ids: set[str] = set()
    open_count = 0
    priority_counts = {"high": 0, "medium": 0, "low": 0}
    for row in lacunae:
        lid = str(row.get("id", ""))
        if lid in seen_lacuna_ids:
            failures.append(f"duplicate lacuna id: {lid}")
            break
        seen_lacuna_ids.add(lid)
        status = str(row.get("status", ""))
        if status == "open":
            open_count += 1
        priority = str(row.get("priority", ""))
        if priority not in priority_counts:
            failures.append(f"invalid lacuna priority: {lid} -> {priority}")
        else:
            priority_counts[priority] += 1
        for cref in row.get("claim_refs", []):
            if str(cref) not in claim_ids:
                failures.append(f"lacuna references unknown claim id: {lid} -> {cref}")
        for marker_id in row.get("related_marker_ids", []):
            if str(marker_id) not in seen_marker_ids:
                failures.append(f"lacuna references unknown conflict marker id: {lid} -> {marker_id}")
        if not str(row.get("title", "")).strip():
            failures.append(f"lacuna missing title: {lid}")
        if not str(row.get("description", "")).strip():
            failures.append(f"lacuna missing description: {lid}")
        if not str(row.get("origin", "")).strip():
            failures.append(f"lacuna missing origin: {lid}")

    if int(lmeta.get("open_count", -1)) != open_count:
        failures.append("lacunae open_count metadata mismatch")
    if int(lmeta.get("high_priority_count", -1)) != priority_counts["high"]:
        failures.append("lacunae high_priority_count metadata mismatch")
    if int(lmeta.get("medium_priority_count", -1)) != priority_counts["medium"]:
        failures.append("lacunae medium_priority_count metadata mismatch")
    if int(lmeta.get("low_priority_count", -1)) != priority_counts["low"]:
        failures.append("lacunae low_priority_count metadata mismatch")

    if failures:
        print("ERROR: Wave5 Batch3 registry verification failed.")
        for item in failures[:250]:
            print(f"- {item}")
        if len(failures) > 250:
            print(f"- ... and {len(failures) - 250} more failures")
        return 1

    print(
        "OK: Wave5 Batch3 conflict/lacuna registries verified. "
        f"markers={len(markers)} lacunae={len(lacunae)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

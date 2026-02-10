#!/usr/bin/env python3
"""
Verify that every in-scope CSV is covered by the correct TOML registry lane.

In-scope CSV zones:
- data/csv/* (project_csv)
- data/csv/legacy/* (legacy_csv)
- curated/**/*.csv (curated_csv)
- data/external/*.csv (external_csv)
- archive/**/*.csv, docs/archive/**/*.csv (archive_csv)

Coverage contracts:
- project_csv -> registry/project_csv_canonical_datasets.toml OR registry/project_csv_generated_artifacts.toml
- legacy_csv -> registry/legacy_csv_datasets.toml
- curated_csv -> registry/curated_csv_datasets.toml
- external_csv -> registry/external_csv_holding_datasets.toml
- archive_csv -> registry/archive_csv_holding_datasets.toml
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path


def _load_source_set(path: Path) -> set[str]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return {str(row.get("source_csv", "")) for row in data.get("dataset", [])}


def _zone_set(inventory: dict, zone: str) -> set[str]:
    return {
        str(row.get("path", ""))
        for row in inventory.get("document", [])
        if str(row.get("zone", "")) == zone
    }


def _summarize_mismatch(
    failures: list[str],
    label: str,
    expected: set[str],
    actual: set[str],
) -> None:
    if expected == actual:
        return
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    failures.append(f"{label}: coverage mismatch.")
    if missing:
        failures.append(f"{label}: missing={len(missing)}")
        failures.extend(f"- missing: {item}" for item in missing[:20])
    if extra:
        failures.append(f"{label}: extra={len(extra)}")
        failures.extend(f"- extra: {item}" for item in extra[:20])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root.",
    )
    parser.add_argument(
        "--inventory",
        default="registry/csv_inventory.toml",
        help="CSV inventory TOML path.",
    )
    parser.add_argument(
        "--project-canonical-index",
        default="registry/project_csv_canonical_datasets.toml",
        help="Project canonical scroll index path.",
    )
    parser.add_argument(
        "--project-generated-index",
        default="registry/project_csv_generated_artifacts.toml",
        help="Project generated scroll index path.",
    )
    parser.add_argument(
        "--legacy-index",
        default="registry/legacy_csv_datasets.toml",
        help="Legacy CSV index path.",
    )
    parser.add_argument(
        "--curated-index",
        default="registry/curated_csv_datasets.toml",
        help="Curated CSV index path.",
    )
    parser.add_argument(
        "--external-holding-index",
        default="registry/external_csv_holding_datasets.toml",
        help="External holding index path.",
    )
    parser.add_argument(
        "--archive-holding-index",
        default="registry/archive_csv_holding_datasets.toml",
        help="Archive holding index path.",
    )
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    inventory = tomllib.loads((root / args.inventory).read_text(encoding="utf-8"))

    project_zone = _zone_set(inventory, "project_csv")
    legacy_zone = _zone_set(inventory, "legacy_csv")
    curated_zone = _zone_set(inventory, "curated_csv")
    external_zone = _zone_set(inventory, "external_csv")
    archive_zone = _zone_set(inventory, "archive_csv")

    project_canonical = _load_source_set(root / args.project_canonical_index)
    project_generated = _load_source_set(root / args.project_generated_index)
    legacy_index = _load_source_set(root / args.legacy_index)
    curated_index = _load_source_set(root / args.curated_index)
    external_index = _load_source_set(root / args.external_holding_index)
    archive_index = _load_source_set(root / args.archive_holding_index)

    failures: list[str] = []
    _summarize_mismatch(
        failures,
        "project_csv",
        project_zone,
        project_canonical | project_generated,
    )
    _summarize_mismatch(failures, "legacy_csv", legacy_zone, legacy_index)
    _summarize_mismatch(failures, "curated_csv", curated_zone, curated_index)
    _summarize_mismatch(failures, "external_csv", external_zone, external_index)
    _summarize_mismatch(failures, "archive_csv", archive_zone, archive_index)

    manual_triage_in_scope = [
        str(row.get("path", ""))
        for row in inventory.get("document", [])
        if str(row.get("zone", "")) in {
            "project_csv",
            "legacy_csv",
            "curated_csv",
            "external_csv",
            "archive_csv",
        }
        and str(row.get("migration_action", "")) == "manual_triage"
    ]
    if manual_triage_in_scope:
        failures.append(
            f"in-scope CSV entries still marked manual_triage: {len(manual_triage_in_scope)}"
        )
        failures.extend(f"- manual_triage: {item}" for item in manual_triage_in_scope[:20])

    if failures:
        print("ERROR: CSV corpus coverage verification failed.")
        for item in failures[:200]:
            print(item)
        if len(failures) > 200:
            print(f"... and {len(failures) - 200} more failures")
        return 1

    print(
        "OK: CSV corpus coverage verified. "
        f"project={len(project_zone)} legacy={len(legacy_zone)} curated={len(curated_zone)} "
        f"external={len(external_zone)} archive={len(archive_zone)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

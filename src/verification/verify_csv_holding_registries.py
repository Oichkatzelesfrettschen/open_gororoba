#!/usr/bin/env python3
"""
Verify external/archive CSV holding registries and manifests.
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path


def _load_manifest(path: Path) -> set[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return {
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    }


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
        help="CSV inventory path.",
    )
    parser.add_argument(
        "--external-registry",
        default="registry/external_csv_holding.toml",
        help="External holding registry path.",
    )
    parser.add_argument(
        "--archive-registry",
        default="registry/archive_csv_holding.toml",
        help="Archive holding registry path.",
    )
    parser.add_argument(
        "--external-manifest",
        default="registry/manifests/external_csv_holding_manifest.txt",
        help="External manifest path.",
    )
    parser.add_argument(
        "--archive-manifest",
        default="registry/manifests/archive_csv_holding_manifest.txt",
        help="Archive manifest path.",
    )
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    inv = tomllib.loads((root / args.inventory).read_text(encoding="utf-8"))
    ext = tomllib.loads((root / args.external_registry).read_text(encoding="utf-8"))
    arc = tomllib.loads((root / args.archive_registry).read_text(encoding="utf-8"))

    external_inventory_paths = {
        str(row.get("path", ""))
        for row in inv.get("document", [])
        if str(row.get("zone", "")) == "external_csv"
    }
    archive_inventory_paths = {
        str(row.get("path", ""))
        for row in inv.get("document", [])
        if str(row.get("zone", "")) == "archive_csv"
    }

    external_registry_paths = {str(row.get("path", "")) for row in ext.get("dataset", [])}
    archive_registry_paths = {str(row.get("path", "")) for row in arc.get("dataset", [])}
    external_manifest_paths = _load_manifest(root / args.external_manifest)
    archive_manifest_paths = _load_manifest(root / args.archive_manifest)

    failures: list[str] = []
    if external_registry_paths != external_inventory_paths:
        failures.append("External holding registry paths do not match external_csv inventory set.")
    if archive_registry_paths != archive_inventory_paths:
        failures.append("Archive holding registry paths do not match archive_csv inventory set.")
    if external_manifest_paths != external_inventory_paths:
        failures.append("External holding manifest does not match external_csv inventory set.")
    if archive_manifest_paths != archive_inventory_paths:
        failures.append("Archive holding manifest does not match archive_csv inventory set.")

    for row in ext.get("dataset", []):
        if str(row.get("hold_status", "")) != "queued_for_scroll_conversion":
            failures.append(f"{row.get('path')}: external hold_status mismatch")
    for row in arc.get("dataset", []):
        if str(row.get("hold_status", "")) != "queued_for_scroll_conversion":
            failures.append(f"{row.get('path')}: archive hold_status mismatch")

    if failures:
        print("ERROR: CSV holding registry verification failed.")
        for item in failures[:200]:
            print(f"- {item}")
        if len(failures) > 200:
            print(f"- ... and {len(failures) - 200} more failures")
        return 1

    print(
        "OK: CSV holding registries verified. "
        f"external={len(external_inventory_paths)} archive={len(archive_inventory_paths)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

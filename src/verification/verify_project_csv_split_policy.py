#!/usr/bin/env python3
"""
Verify project_csv split policy and scroll conversion coverage.
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
        "--policy",
        default="registry/project_csv_split_policy.toml",
        help="Project CSV split policy path.",
    )
    parser.add_argument(
        "--canonical-manifest",
        default="registry/manifests/project_csv_canonical_manifest.txt",
        help="Canonical source manifest path.",
    )
    parser.add_argument(
        "--generated-manifest",
        default="registry/manifests/project_csv_generated_manifest.txt",
        help="Generated source manifest path.",
    )
    parser.add_argument(
        "--canonical-index",
        default="registry/project_csv_canonical_datasets.toml",
        help="Canonical scroll index path.",
    )
    parser.add_argument(
        "--generated-index",
        default="registry/project_csv_generated_artifacts.toml",
        help="Generated scroll index path.",
    )
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    inventory = tomllib.loads((root / args.inventory).read_text(encoding="utf-8"))
    policy = tomllib.loads((root / args.policy).read_text(encoding="utf-8"))
    canonical_index = tomllib.loads((root / args.canonical_index).read_text(encoding="utf-8"))
    generated_index = tomllib.loads((root / args.generated_index).read_text(encoding="utf-8"))

    project_paths = {
        str(row.get("path", ""))
        for row in inventory.get("document", [])
        if str(row.get("zone", "")) == "project_csv"
    }
    policy_rows = policy.get("dataset", [])
    policy_paths = {str(row.get("path", "")) for row in policy_rows}
    canonical_policy = {
        str(row.get("path", ""))
        for row in policy_rows
        if str(row.get("classification", "")) == "canonical_dataset"
    }
    generated_policy = {
        str(row.get("path", ""))
        for row in policy_rows
        if str(row.get("classification", "")) == "generated_artifact"
    }

    failures: list[str] = []

    if policy_paths != project_paths:
        missing = sorted(project_paths - policy_paths)
        extra = sorted(policy_paths - project_paths)
        if missing:
            failures.append(f"Policy missing {len(missing)} project_csv paths.")
            failures.extend(f"- missing: {item}" for item in missing[:20])
        if extra:
            failures.append(f"Policy has {len(extra)} non-project paths.")
            failures.extend(f"- extra: {item}" for item in extra[:20])

    if canonical_policy & generated_policy:
        failures.append("Policy has overlapping canonical/generated path assignments.")

    if canonical_policy | generated_policy != project_paths:
        failures.append("Policy canonical/generated partition does not cover project_csv set.")

    canonical_manifest = _load_manifest(root / args.canonical_manifest)
    generated_manifest = _load_manifest(root / args.generated_manifest)
    if canonical_manifest != canonical_policy:
        failures.append("Canonical manifest does not match policy canonical_dataset set.")
    if generated_manifest != generated_policy:
        failures.append("Generated manifest does not match policy generated_artifact set.")

    canonical_index_paths = {
        str(row.get("source_csv", "")) for row in canonical_index.get("dataset", [])
    }
    generated_index_paths = {
        str(row.get("source_csv", "")) for row in generated_index.get("dataset", [])
    }

    if canonical_index_paths != canonical_policy:
        failures.append("Canonical index source_csv set does not match policy canonical_dataset set.")
    if generated_index_paths != generated_policy:
        failures.append("Generated index source_csv set does not match policy generated_artifact set.")

    for row in canonical_index.get("dataset", []):
        if str(row.get("dataset_class", "")) != "canonical_dataset":
            failures.append(f"{row.get('source_csv')}: canonical index has wrong dataset_class")
    for row in generated_index.get("dataset", []):
        if str(row.get("dataset_class", "")) != "generated_artifact":
            failures.append(f"{row.get('source_csv')}: generated index has wrong dataset_class")

    if failures:
        print("ERROR: project_csv split policy verification failed.")
        for item in failures[:200]:
            print(f"- {item}")
        if len(failures) > 200:
            print(f"- ... and {len(failures) - 200} more failures")
        return 1

    print(
        "OK: project_csv split policy and scroll coverage verified. "
        f"canonical={len(canonical_policy)} generated={len(generated_policy)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

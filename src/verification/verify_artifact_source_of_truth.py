#!/usr/bin/env python3
"""
Verify integrity of registry/artifact_source_of_truth.toml.

Checks:
- Top-level counts match artifact rows.
- Artifact IDs and keys are unique.
- Status and minimum_requirement invariants hold.
- canonical_functional_url is present in all_links when non-empty.
- downloaded paths exist for downloaded artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tomllib


VALID_STATUSES = {
    "downloaded",
    "downloadable",
    "blocked",
    "citation_only_no_link",
    "unverified",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root.",
    )
    parser.add_argument(
        "--registry",
        default="registry/artifact_source_of_truth.toml",
        help="Path to source-of-truth registry.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    registry_path = repo_root / args.registry
    if not registry_path.exists():
        print(f"ERROR: missing registry {registry_path}")
        return 1

    data = tomllib.loads(registry_path.read_text(encoding="utf-8"))
    head = data.get("artifact_source_of_truth", {})
    coverage = data.get("coverage", {})
    artifacts = data.get("artifact", [])
    failures: list[str] = []

    if not isinstance(artifacts, list):
        failures.append("artifact table is missing or not a list")
        artifacts = []

    ids: set[str] = set()
    keys: set[str] = set()
    downloaded_count = 0
    downloadable_count = 0
    blocked_count = 0
    citation_only_count = 0
    unverified_count = 0
    missing_minimum_count = 0
    manual_count = 0
    coverage_missing_keys: list[str] = [str(v).strip() for v in coverage.get("artifacts_without_working_mirror", [])]

    for idx, art in enumerate(artifacts):
        art_id = str(art.get("id", "")).strip()
        key = str(art.get("key", "")).strip()
        status = str(art.get("status", "")).strip()
        minimum_met = bool(art.get("minimum_requirement_met", False))
        manual = bool(art.get("manual_intervention_required", False))
        all_links = [str(v).strip() for v in art.get("all_links", []) if str(v).strip()]
        working = [str(v).strip() for v in art.get("working_mirrors", []) if str(v).strip()]
        working_pdf = [str(v).strip() for v in art.get("working_pdf_mirrors", []) if str(v).strip()]
        nonworking = [str(v).strip() for v in art.get("nonworking_mirrors", []) if str(v).strip()]
        unverified = [str(v).strip() for v in art.get("unverified_mirrors", []) if str(v).strip()]
        downloaded_paths = [str(v).strip() for v in art.get("downloaded_paths", []) if str(v).strip()]
        canonical_url = str(art.get("canonical_functional_url", "")).strip()
        canonical_path = str(art.get("canonical_download_path", "")).strip()

        if not art_id:
            failures.append(f"artifact[{idx}] missing id")
        elif art_id in ids:
            failures.append(f"duplicate artifact id: {art_id}")
        else:
            ids.add(art_id)

        if not key:
            failures.append(f"{art_id or f'index {idx}'} missing key")
        elif key in keys:
            failures.append(f"duplicate artifact key: {key}")
        else:
            keys.add(key)

        if status not in VALID_STATUSES:
            failures.append(f"{art_id}: invalid status {status!r}")

        if canonical_url and canonical_url not in all_links:
            failures.append(f"{art_id}: canonical_functional_url not in all_links")

        if status == "downloaded":
            downloaded_count += 1
            if not downloaded_paths:
                failures.append(f"{art_id}: downloaded status requires downloaded_paths")
        elif status == "downloadable":
            downloadable_count += 1
        elif status == "blocked":
            blocked_count += 1
            if working:
                failures.append(f"{art_id}: blocked status but has working_mirrors")
        elif status == "citation_only_no_link":
            citation_only_count += 1
            if all_links:
                failures.append(f"{art_id}: citation_only_no_link but all_links is not empty")
        elif status == "unverified":
            unverified_count += 1

        if minimum_met != bool(working or downloaded_paths):
            failures.append(
                f"{art_id}: minimum_requirement_met mismatch with working/downloaded mirrors"
            )
        if not minimum_met:
            missing_minimum_count += 1
            if key not in coverage_missing_keys:
                failures.append(
                    f"{art_id}: missing minimum requirement but key absent from coverage.artifacts_without_working_mirror"
                )

        if manual:
            manual_count += 1

        if len(working_pdf) > len(working):
            failures.append(f"{art_id}: working_pdf_mirrors cannot exceed working_mirrors")

        if canonical_path:
            path_obj = repo_root / canonical_path
            if not path_obj.exists():
                failures.append(f"{art_id}: canonical_download_path does not exist: {canonical_path}")

        for path in downloaded_paths:
            path_obj = repo_root / path
            if not path_obj.exists():
                failures.append(f"{art_id}: downloaded path missing on disk: {path}")

        if (
            not minimum_met
            and status != "citation_only_no_link"
            and not (nonworking or unverified)
        ):
            failures.append(
                f"{art_id}: neither nonworking nor unverified mirrors recorded despite missing minimum"
            )

    expected_counts = {
        "artifact_count": len(artifacts),
        "downloaded_count": downloaded_count,
        "downloadable_count": downloadable_count,
        "blocked_count": blocked_count,
        "citation_only_no_link_count": citation_only_count,
        "unverified_count": unverified_count,
        "missing_minimum_requirement_count": missing_minimum_count,
        "manual_intervention_required_count": manual_count,
    }
    for key, expected in expected_counts.items():
        observed = int(head.get(key, -1))
        if observed != expected:
            failures.append(
                f"header {key} mismatch: header={observed} computed={expected}"
            )

    source_files = [str(v).strip() for v in head.get("source_files", []) if str(v).strip()]
    source_file_count = int(head.get("source_file_count", -1))
    if source_file_count != len(source_files):
        failures.append("header source_file_count mismatch with source_files list length")

    source_tables = [str(v).strip() for v in head.get("source_tables", []) if str(v).strip()]
    source_table_count = int(head.get("source_table_count", -1))
    if source_table_count != len(source_tables):
        failures.append("header source_table_count mismatch with source_tables list length")

    coverage_count = int(coverage.get("artifacts_without_working_mirror_count", -1))
    if coverage_count != len(coverage_missing_keys):
        failures.append(
            "coverage artifacts_without_working_mirror_count mismatch with list length"
        )
    if coverage_count != missing_minimum_count:
        failures.append(
            "coverage artifacts_without_working_mirror_count mismatch with computed missing minimum count"
        )

    if failures:
        print("ERROR: artifact source-of-truth verification failed.")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(
        "OK: artifact source-of-truth verified. "
        f"artifacts={len(artifacts)} downloaded={downloaded_count} "
        f"downloadable={downloadable_count} blocked={blocked_count} "
        f"citation_only={citation_only_count} unverified={unverified_count} "
        f"missing_minimum={missing_minimum_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

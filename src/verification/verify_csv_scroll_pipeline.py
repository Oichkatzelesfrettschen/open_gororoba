#!/usr/bin/env python3
"""
Verify unified CSV scroll pipeline registry consistency.
"""

from __future__ import annotations

import argparse
import tomllib
from collections import Counter
from pathlib import Path

EXPECTED_LANES = {
    "project_canonical",
    "project_generated",
    "external_holding",
    "archive_holding",
}


def _load(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _assert_ascii(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    bad = sorted({ch for ch in text if ord(ch) > 127})
    if bad:
        sample = "".join(bad[:20])
        raise SystemExit(f"ERROR: non-ASCII content in {path}: {sample!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root.",
    )
    parser.add_argument(
        "--pipeline",
        default="registry/csv_scroll_pipeline.toml",
        help="Pipeline registry path.",
    )
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    pipeline_path = root / args.pipeline
    if not pipeline_path.exists():
        raise SystemExit(f"ERROR: missing pipeline registry: {pipeline_path}")
    _assert_ascii(pipeline_path)

    raw = _load(pipeline_path)
    section = raw.get("csv_scroll_pipeline", {})
    lanes = raw.get("lane", [])
    refs = raw.get("dataset_ref", [])

    failures: list[str] = []
    if int(section.get("lane_count", -1)) != len(lanes):
        failures.append("lane_count metadata mismatch.")
    if int(section.get("dataset_total", -1)) != len(refs):
        failures.append("dataset_total metadata mismatch.")

    lane_names = {str(row.get("name", "")) for row in lanes}
    if lane_names != EXPECTED_LANES:
        failures.append(
            "lane name mismatch: "
            f"expected={sorted(EXPECTED_LANES)} got={sorted(lane_names)}"
        )

    refs_by_lane = Counter(str(row.get("lane_name", "")) for row in refs)

    for lane in lanes:
        lane_name = str(lane.get("name", ""))
        source_registry = str(lane.get("source_registry", ""))
        source_table = str(lane.get("source_table", ""))
        declared_count = int(lane.get("dataset_count", -1))
        manifest_path = str(lane.get("manifest_path", ""))
        canonical_dir = str(lane.get("canonical_dir", ""))

        source_abs = root / source_registry
        if not source_abs.exists():
            failures.append(f"missing source registry for lane {lane_name}: {source_registry}")
            continue

        source_raw = _load(source_abs)
        source_section = source_raw.get(source_table, {})
        source_datasets = source_raw.get("dataset", [])
        if declared_count != len(source_datasets):
            failures.append(
                "lane "
                f"{lane_name}: dataset_count mismatch {declared_count} != "
                f"{len(source_datasets)}"
            )
        if refs_by_lane.get(lane_name, 0) != len(source_datasets):
            failures.append(
                f"lane {lane_name}: dataset_ref mismatch {refs_by_lane.get(lane_name, 0)} != "
                f"{len(source_datasets)}"
            )

        expected_manifest = str(source_section.get("source_descriptor", ""))
        if expected_manifest.startswith("manifest:"):
            expected_manifest = expected_manifest.split(":", 1)[1]
        else:
            expected_manifest = ""
        if manifest_path != expected_manifest:
            failures.append(
                f"lane {lane_name}: manifest path mismatch {manifest_path!r} != "
                f"{expected_manifest!r}"
            )
        if manifest_path and not (root / manifest_path).exists():
            failures.append(f"lane {lane_name}: missing manifest file {manifest_path}")

        expected_dir = str(source_section.get("canonical_dir", ""))
        if canonical_dir != expected_dir:
            failures.append(
                f"lane {lane_name}: canonical_dir mismatch {canonical_dir!r} != {expected_dir!r}"
            )
        if canonical_dir and not (root / canonical_dir).exists():
            failures.append(f"lane {lane_name}: missing canonical dir {canonical_dir}")

    seen_ref_ids: set[str] = set()
    for row in refs:
        ref_id = str(row.get("id", ""))
        if ref_id in seen_ref_ids:
            failures.append(f"duplicate dataset_ref id: {ref_id}")
            continue
        seen_ref_ids.add(ref_id)

        source_csv = str(row.get("source_csv", ""))
        canonical_toml = str(row.get("canonical_toml", ""))
        if not source_csv:
            failures.append(f"{ref_id}: missing source_csv")
        elif not (root / source_csv).exists():
            failures.append(f"{ref_id}: missing source_csv file {source_csv}")

        if not canonical_toml:
            failures.append(f"{ref_id}: missing canonical_toml")
        elif not (root / canonical_toml).exists():
            failures.append(f"{ref_id}: missing canonical_toml file {canonical_toml}")

    if failures:
        print("ERROR: csv scroll pipeline verification failed.")
        for item in failures[:200]:
            print(f"- {item}")
        if len(failures) > 200:
            print(f"- ... and {len(failures) - 200} more failures")
        return 1

    print(
        "OK: csv scroll pipeline verified. "
        f"lanes={len(lanes)} dataset_refs={len(refs)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

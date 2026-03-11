#!/usr/bin/env python3
"""
Verify dataset label aliases used by execution-planning registries.
"""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path


DATASET_RE = re.compile(r"^(?:PC|PG|EX|AR|CU)-\d{4}$")


def _load(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _assert_ascii(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    bad = sorted({ch for ch in text if ord(ch) > 127})
    if bad:
        sample = "".join(bad[:20])
        raise SystemExit(f"ERROR: non-ASCII content in {path}: {sample!r}")


def _normalize_dataset_label(value: str) -> str:
    return " ".join(value.strip().split()).lower()


def _collect_dataset_ids(root: Path) -> set[str]:
    out: set[str] = set()
    candidate_files = [
        "registry/project_csv_canonical_datasets.toml",
        "registry/project_csv_generated_artifacts.toml",
        "registry/project_csv_generated_datasets.toml",
        "registry/external_csv_datasets.toml",
        "registry/archive_csv_datasets.toml",
        "registry/curated_csv_datasets.toml",
    ]
    for rel in candidate_files:
        path = root / rel
        if not path.exists():
            continue
        raw = _load(path)
        for row in raw.get("dataset", []):
            dataset_id = str(row.get("id", ""))
            if DATASET_RE.fullmatch(dataset_id):
                out.add(dataset_id)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root.",
    )
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    alias_path = root / "registry/dataset_label_aliases.toml"
    experiments_path = root / "registry/experiments.toml"
    lineage_path = root / "registry/experiment_lineage.toml"
    for path in (alias_path, experiments_path, lineage_path):
        if not path.exists():
            raise SystemExit(f"ERROR: missing required registry {path.relative_to(root)}")
        _assert_ascii(path)

    alias_raw = _load(alias_path)
    experiments = _load(experiments_path).get("experiment", [])
    lineages = _load(lineage_path).get("lineage", [])
    dataset_ids = _collect_dataset_ids(root)

    meta = alias_raw.get("dataset_label_aliases", {})
    rows = alias_raw.get("alias", [])
    failures: list[str] = []

    if int(meta.get("alias_count", -1)) != len(rows):
        failures.append("dataset_label_aliases alias_count metadata mismatch")

    normalized_seen: dict[str, str] = {}
    alias_labels: set[str] = set()
    for row in rows:
        alias_id = str(row.get("id", ""))
        label = str(row.get("label", "")).strip()
        label_normalized = _normalize_dataset_label(
            str(row.get("label_normalized", "")) or label
        )
        canonical_dataset_id = str(row.get("canonical_dataset_id", "")).strip()
        if not alias_id:
            failures.append("dataset_label_aliases alias row missing id")
        if not label:
            failures.append(f"dataset_label_aliases[{alias_id}] missing label")
        if not label_normalized:
            failures.append(f"dataset_label_aliases[{alias_id}] missing label_normalized")
        if not DATASET_RE.fullmatch(canonical_dataset_id):
            failures.append(
                f"dataset_label_aliases[{alias_id}] invalid canonical_dataset_id: {canonical_dataset_id}"
            )
        elif canonical_dataset_id not in dataset_ids:
            failures.append(
                f"dataset_label_aliases[{alias_id}] unknown canonical_dataset_id: {canonical_dataset_id}"
            )
        if label_normalized in normalized_seen:
            failures.append(
                "dataset_label_aliases duplicate normalized label: "
                f"{label_normalized} ({normalized_seen[label_normalized]}, {alias_id})"
            )
        else:
            normalized_seen[label_normalized] = alias_id
        alias_labels.add(label_normalized)

    for row in experiments:
        experiment_id = str(row.get("id", ""))
        for label in row.get("dataset_label_refs", []):
            normalized = _normalize_dataset_label(str(label))
            if normalized not in alias_labels:
                failures.append(
                    f"experiments[{experiment_id}] unknown dataset_label_ref: {label}"
                )

    for row in lineages:
        lineage_id = str(row.get("id", ""))
        for label in row.get("dataset_label_refs", []):
            normalized = _normalize_dataset_label(str(label))
            if normalized not in alias_labels:
                failures.append(
                    f"experiment_lineage[{lineage_id}] unknown dataset_label_ref: {label}"
                )

    if failures:
        print("ERROR: dataset label alias verification failed.")
        for item in failures[:200]:
            print(f"- {item}")
        if len(failures) > 200:
            print(f"- ... and {len(failures) - 200} more failures")
        return 1

    print(
        "OK: dataset label aliases verified. "
        f"aliases={len(rows)} experiments={len(experiments)} lineages={len(lineages)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

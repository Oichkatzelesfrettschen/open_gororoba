#!/usr/bin/env python3
"""
Verify operational-role contracts in registry/external_sources.toml and
surface-consumption declarations in registry/experiments.toml.
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

ROLE_ALLOWLIST = {
    "reference_capture",
    "provider_manifest",
    "chronology_pack_status",
    "generated_index",
    "dataset_lineage_audit",
    "claims_inbox",
    "falsification_contract",
    "claim_bridge",
    "mirror_audit",
}
TRUTH_SURFACE_ALLOWLIST = {
    "chronology_control",
    "environment_context",
    "lineage_transition",
    "observation_benchmark",
}
ROLES_REQUIRING_ARTIFACT_CONTRACTS = {
    "dataset_lineage_audit",
    "falsification_contract",
}
ANOMALY_INPUT_MARKERS = (
    "data/external/flyby_anomaly/",
    "data/external/pioneer_anomaly/",
    "data/output/anomaly/",
)


def _load(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _string_list(row: dict, key: str) -> list[str]:
    values = row.get(key, [])
    if not isinstance(values, list):
        return []
    return [str(item).strip() for item in values if str(item).strip()]


def _looks_like_anomaly_surface_experiment(row: dict) -> bool:
    for key in ("input", "run"):
        value = str(row.get(key, ""))
        if any(marker in value for marker in ANOMALY_INPUT_MARKERS):
            return True
    for key in ("output", "dataset_refs", "input_path_refs", "output_path_refs"):
        for value in _string_list(row, key):
            if any(marker in value for marker in ANOMALY_INPUT_MARKERS):
                return True
            lowered = value.lower()
            if "flyby anomaly" in lowered or "pioneer anomaly" in lowered:
                return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root.",
    )
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    external_path = root / "registry/external_sources.toml"
    experiments_path = root / "registry/experiments.toml"
    if not external_path.exists():
        raise SystemExit(f"ERROR: missing {external_path}")
    if not experiments_path.exists():
        raise SystemExit(f"ERROR: missing {experiments_path}")

    external_docs = _load(external_path).get("document", [])
    experiments = _load(experiments_path).get("experiment", [])

    docs_by_id = {str(row.get("id", "")): row for row in external_docs}
    failures: list[str] = []
    role_count = 0
    anomaly_experiment_count = 0

    for row in external_docs:
        doc_id = str(row.get("id", ""))
        role = str(row.get("operational_role", "")).strip()
        role_count += 1
        if role not in ROLE_ALLOWLIST:
            failures.append(f"external_sources[{doc_id}] invalid operational_role: {role}")

        truth_surfaces = _string_list(row, "truth_surfaces")
        unknown_truth = sorted(set(truth_surfaces) - TRUTH_SURFACE_ALLOWLIST)
        if unknown_truth:
            failures.append(
                f"external_sources[{doc_id}] invalid truth_surfaces: {', '.join(unknown_truth)}"
            )

        contract_paths = _string_list(row, "artifact_contract_paths")
        if role in ROLES_REQUIRING_ARTIFACT_CONTRACTS and not contract_paths:
            failures.append(
                f"external_sources[{doc_id}] role {role} requires artifact_contract_paths"
            )
        for rel in contract_paths:
            if not (root / rel).exists():
                failures.append(f"external_sources[{doc_id}] missing artifact_contract_path: {rel}")

        lineage = str(row.get("source_lineage_summary", "")).strip()
        if role == "dataset_lineage_audit" and not lineage:
            failures.append(
                f"external_sources[{doc_id}] dataset_lineage_audit requires source_lineage_summary"
            )
        if role == "falsification_contract" and "observation_benchmark" not in truth_surfaces:
            failures.append(
                f"external_sources[{doc_id}]"
                " falsification_contract must expose"
                " observation_benchmark"
            )

    for row in experiments:
        experiment_id = str(row.get("id", ""))
        if not _looks_like_anomaly_surface_experiment(row):
            continue
        anomaly_experiment_count += 1
        source_refs = _string_list(row, "external_source_refs")
        truth_consumption = _string_list(row, "truth_surface_consumption")
        if not source_refs:
            failures.append(
                f"experiments[{experiment_id}]"
                " anomaly surface experiment"
                " missing external_source_refs"
            )
            continue
        if not truth_consumption:
            failures.append(
                f"experiments[{experiment_id}]"
                " anomaly surface experiment"
                " missing truth_surface_consumption"
            )
            continue

        unknown_sources = [ref for ref in source_refs if ref not in docs_by_id]
        if unknown_sources:
            failures.append(
                f"experiments[{experiment_id}]"
                " unknown external_source_refs:"
                f" {', '.join(unknown_sources)}"
            )
            continue

        unknown_truth = sorted(set(truth_consumption) - TRUTH_SURFACE_ALLOWLIST)
        if unknown_truth:
            failures.append(
                f"experiments[{experiment_id}] invalid truth_surface_consumption: "
                + ", ".join(unknown_truth)
            )
            continue

        offered_surfaces: set[str] = set()
        for ref in source_refs:
            offered_surfaces.update(_string_list(docs_by_id[ref], "truth_surfaces"))
        missing_surfaces = [
            surface for surface in truth_consumption if surface not in offered_surfaces
        ]
        if missing_surfaces:
            failures.append(
                f"experiments[{experiment_id}] truth surfaces not offered by external_source_refs: "
                + ", ".join(missing_surfaces)
            )

    if failures:
        print("ERROR: external-source operational contract verification failed.")
        for item in failures:
            print(f"- {item}")
        return 1

    print(
        "OK: external-source operational contracts verified. "
        f"docs={role_count} anomaly_experiments={anomaly_experiment_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

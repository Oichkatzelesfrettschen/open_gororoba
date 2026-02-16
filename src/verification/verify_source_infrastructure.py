#!/usr/bin/env python3
"""
Verify source infrastructure lane projections against authoritative master registry.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tomllib


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
        "--infrastructure",
        default="registry/source_infrastructure.toml",
        help="Source infrastructure manifest.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    infra_path = repo_root / args.infrastructure
    if not infra_path.exists():
        print(f"ERROR: missing infrastructure manifest: {infra_path}")
        return 1

    infra = _load(infra_path)
    infra_head = infra.get("source_infrastructure", {})
    lane_defs = infra.get("lane", [])

    master_rel = str(infra_head.get("master_registry", "")).strip()
    if not master_rel:
        print("ERROR: infrastructure missing master_registry")
        return 1
    master_path = repo_root / master_rel
    if not master_path.exists():
        print(f"ERROR: missing master registry: {master_path}")
        return 1

    master = _load(master_path)
    artifacts = master.get("artifact", [])
    if not isinstance(artifacts, list):
        print("ERROR: master artifact table missing or invalid")
        return 1

    master_ids = [str(a.get("id", "")).strip() for a in artifacts]
    master_id_set = {i for i in master_ids if i}
    failures: list[str] = []

    if len(master_ids) != len(master_id_set):
        failures.append("master has duplicate or empty artifact ids")

    lane_membership: dict[str, str] = {}
    lane_total = 0

    for lane_def in lane_defs:
        lane_name = str(lane_def.get("name", "")).strip()
        lane_rel = str(lane_def.get("path", "")).strip()
        expected_count = int(lane_def.get("artifact_count", -1))
        if not lane_name or not lane_rel:
            failures.append("lane definition missing name/path")
            continue

        lane_path = repo_root / lane_rel
        if not lane_path.exists():
            failures.append(f"lane file missing: {lane_rel}")
            continue

        lane_data = _load(lane_path)
        lane_head = lane_data.get("lane", {})
        lane_artifacts = lane_data.get("artifact_ref", [])
        lane_count = len(lane_artifacts)
        lane_total += lane_count

        if int(lane_head.get("artifact_count", -1)) != lane_count:
            failures.append(f"lane header artifact_count mismatch: {lane_rel}")

        if expected_count != lane_count:
            failures.append(
                f"infrastructure artifact_count mismatch for lane {lane_name}: "
                f"infra={expected_count} lane={lane_count}"
            )

        for ref in lane_artifacts:
            aid = str(ref.get("id", "")).strip()
            if not aid:
                failures.append(f"{lane_rel}: artifact_ref missing id")
                continue
            if aid not in master_id_set:
                failures.append(f"{lane_rel}: unknown artifact id {aid}")
                continue
            existing = lane_membership.get(aid)
            if existing and existing != lane_name:
                failures.append(
                    f"artifact {aid} appears in multiple lanes: {existing}, {lane_name}"
                )
            lane_membership[aid] = lane_name

    missing_from_lanes = sorted(master_id_set - set(lane_membership.keys()))
    if missing_from_lanes:
        failures.append(
            f"{len(missing_from_lanes)} master artifacts missing lane assignment"
        )

    infra_total = int(infra_head.get("total_artifact_count", -1))
    if infra_total != len(artifacts):
        failures.append(
            f"infrastructure total_artifact_count mismatch: infra={infra_total} master={len(artifacts)}"
        )

    if lane_total != len(artifacts):
        failures.append(
            f"lane total mismatch: lane_total={lane_total} master={len(artifacts)}"
        )

    if failures:
        print("ERROR: source infrastructure verification failed")
        for item in failures:
            print(f"- {item}")
        return 1

    print(
        "OK: source infrastructure verified. "
        f"artifacts={len(artifacts)} lanes={len(lane_defs)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

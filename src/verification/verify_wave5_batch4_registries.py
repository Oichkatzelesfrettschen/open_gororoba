#!/usr/bin/env python3
"""
Verify Wave 5 Batch 4 registries:
- registry/experiments.toml
- registry/experiment_lineage.toml
- registry/roadmap.toml
- registry/todo.toml
- registry/next_actions.toml
- registry/requirements.toml
- registry/module_requirements.toml
"""

from __future__ import annotations

import argparse
import hashlib
import re
import tomllib
from pathlib import Path


CLAIM_RE = re.compile(r"^C-\d{3}$")
INSIGHT_RE = re.compile(r"^I-\d{3}$")
EXPERIMENT_RE = re.compile(r"^E-\d{3}$")
WORKSTREAM_RE = re.compile(r"^WS-[A-Z0-9-]+$")
TODO_RE = re.compile(r"^T-\d{3}$")
ACTION_RE = re.compile(r"^NA-\d{3}$")
REQ_RE = re.compile(r"^REQ-[A-Z0-9-]+$")
DATASET_RE = re.compile(r"^(?:PC|PG|EX|AR|CU)-\d{4}$")


def _assert_ascii(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    bad = sorted({ch for ch in text if ord(ch) > 127})
    if bad:
        sample = "".join(bad[:20])
        raise SystemExit(f"ERROR: non-ASCII content in {path}: {sample!r}")


def _load(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


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
        for value in raw.values():
            if not isinstance(value, list):
                continue
            for row in value:
                if not isinstance(row, dict):
                    continue
                rid = str(row.get("id", ""))
                if DATASET_RE.fullmatch(rid):
                    out.add(rid)
    return out


def _status_token(value: str) -> str:
    return value.strip().upper().replace("-", "_")


def _verify_dependencies(
    failures: list[str],
    deps: list[str],
    where: str,
    claim_ids: set[str],
    insight_ids: set[str],
    experiment_ids: set[str],
    workstream_ids: set[str],
    todo_ids: set[str],
    action_ids: set[str],
    req_ids: set[str],
) -> None:
    for dep in deps:
        dep_id = str(dep).strip()
        if not dep_id:
            continue
        if CLAIM_RE.fullmatch(dep_id):
            if dep_id not in claim_ids:
                failures.append(f"{where} unknown claim dependency: {dep_id}")
            continue
        if INSIGHT_RE.fullmatch(dep_id):
            if dep_id not in insight_ids:
                failures.append(f"{where} unknown insight dependency: {dep_id}")
            continue
        if EXPERIMENT_RE.fullmatch(dep_id):
            if dep_id not in experiment_ids:
                failures.append(f"{where} unknown experiment dependency: {dep_id}")
            continue
        if WORKSTREAM_RE.fullmatch(dep_id):
            if dep_id not in workstream_ids:
                failures.append(f"{where} unknown workstream dependency: {dep_id}")
            continue
        if TODO_RE.fullmatch(dep_id):
            if dep_id not in todo_ids:
                failures.append(f"{where} unknown todo dependency: {dep_id}")
            continue
        if ACTION_RE.fullmatch(dep_id):
            if dep_id not in action_ids:
                failures.append(f"{where} unknown action dependency: {dep_id}")
            continue
        if REQ_RE.fullmatch(dep_id):
            if dep_id not in req_ids:
                failures.append(f"{where} unknown requirements dependency: {dep_id}")
            continue
        failures.append(f"{where} malformed dependency id: {dep_id}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root.",
    )
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    required = [
        "registry/claims.toml",
        "registry/insights.toml",
        "registry/binaries.toml",
        "registry/experiments.toml",
        "registry/experiment_lineage.toml",
        "registry/roadmap.toml",
        "registry/todo.toml",
        "registry/next_actions.toml",
        "registry/requirements.toml",
        "registry/module_requirements.toml",
    ]
    for rel in required:
        path = root / rel
        if not path.exists():
            raise SystemExit(f"ERROR: missing required registry {rel}")

    for rel in required[3:]:
        _assert_ascii(root / rel)

    claims = _load(root / "registry/claims.toml").get("claim", [])
    insights = _load(root / "registry/insights.toml").get("insight", [])
    binaries = _load(root / "registry/binaries.toml").get("binary", [])
    experiments_raw = _load(root / "registry/experiments.toml")
    lineage_raw = _load(root / "registry/experiment_lineage.toml")
    roadmap_raw = _load(root / "registry/roadmap.toml")
    todo_raw = _load(root / "registry/todo.toml")
    actions_raw = _load(root / "registry/next_actions.toml")
    requirements_raw = _load(root / "registry/requirements.toml")
    module_requirements_raw = _load(root / "registry/module_requirements.toml")

    claim_ids = {str(row.get("id", "")) for row in claims}
    insight_ids = {str(row.get("id", "")) for row in insights}
    dataset_ids = _collect_dataset_ids(root)

    binary_names = {str(row.get("name", "")) for row in binaries}
    binary_experiment = {
        str(row.get("name", "")): str(row.get("experiment", ""))
        for row in binaries
        if str(row.get("name", "")).strip()
    }

    experiments_meta = experiments_raw.get("experiments", {})
    experiments = experiments_raw.get("experiment", [])
    lineages_meta = lineage_raw.get("experiment_lineage", {})
    lineages = lineage_raw.get("lineage", [])
    edges = lineage_raw.get("edge", [])
    workstreams = roadmap_raw.get("workstream", [])
    todo_items = todo_raw.get("item", [])
    actions = actions_raw.get("action", [])
    req_modules = requirements_raw.get("module", [])
    req_gaps = requirements_raw.get("coverage_gap", [])
    mr_meta = module_requirements_raw.get("module_requirements", {})
    mr_modules = module_requirements_raw.get("module", [])
    mr_commands = module_requirements_raw.get("command", [])
    mr_packages = module_requirements_raw.get("package", [])

    workstream_ids = {str(row.get("id", "")) for row in workstreams}
    todo_ids = {str(row.get("id", "")) for row in todo_items}
    action_ids = {str(row.get("id", "")) for row in actions}
    req_ids = {str(row.get("id", "")) for row in req_modules}

    failures: list[str] = []

    # W5-016 experiments normalization.
    if int(experiments_meta.get("experiment_count", -1)) != len(experiments):
        failures.append("experiments experiment_count metadata mismatch")
    if int(experiments_meta.get("deterministic_count", -1)) != sum(
        1 for row in experiments if bool(row.get("deterministic", False))
    ):
        failures.append("experiments deterministic_count metadata mismatch")
    if int(experiments_meta.get("gpu_count", -1)) != sum(
        1 for row in experiments if bool(row.get("gpu", False))
    ):
        failures.append("experiments gpu_count metadata mismatch")
    if int(experiments_meta.get("seeded_count", -1)) != sum(
        1 for row in experiments if row.get("seed", None) is not None
    ):
        failures.append("experiments seeded_count metadata mismatch")

    exp_status_allow = set(experiments_meta.get("status_allowlist", []))
    exp_ids = {str(row.get("id", "")) for row in experiments}
    lineage_ids = {str(row.get("id", "")) for row in lineages}
    seen_exp_ids: set[str] = set()
    for row in experiments:
        eid = str(row.get("id", ""))
        if eid in seen_exp_ids:
            failures.append(f"duplicate experiment id: {eid}")
            continue
        seen_exp_ids.add(eid)

        status = str(row.get("status", ""))
        if status not in exp_status_allow:
            failures.append(f"experiment[{eid}] status outside allowlist: {status}")
        if str(row.get("status_token", "")) != _status_token(status):
            failures.append(f"experiment[{eid}] status_token mismatch")

        if str(row.get("lineage_id", "")) not in lineage_ids:
            failures.append(f"experiment[{eid}] unknown lineage_id")

        run_cmd = str(row.get("run", ""))
        run_sha = str(row.get("run_command_sha256", ""))
        expected_sha = hashlib.sha256(run_cmd.encode("utf-8")).hexdigest()
        if run_sha != expected_sha:
            failures.append(f"experiment[{eid}] run_command_sha256 mismatch")

        binary = str(row.get("binary", ""))
        binary_registered = bool(row.get("binary_registered", False))
        if binary_registered and binary not in binary_names:
            failures.append(f"experiment[{eid}] binary marked registered but missing: {binary}")
        if not binary_registered and binary in binary_names:
            failures.append(f"experiment[{eid}] binary marked unregistered but exists: {binary}")

        declared = str(row.get("binary_experiment_declared", ""))
        if declared and declared != eid:
            failures.append(
                f"experiment[{eid}] binary_experiment_declared mismatch: {declared}"
            )
        if binary in binary_experiment and binary_experiment.get(binary, "") not in {"", eid}:
            failures.append(
                f"experiment[{eid}] binary registry experiment mismatch: "
                f"{binary_experiment.get(binary, '')}"
            )

        claims_refs = {str(v) for v in row.get("claim_refs", [])}
        claims_legacy = {str(v) for v in row.get("claims", [])}
        if claims_refs != claims_legacy:
            failures.append(f"experiment[{eid}] claim_refs and claims diverge")
        for cid in claims_refs:
            if cid not in claim_ids:
                failures.append(f"experiment[{eid}] unknown claim ref: {cid}")

        for did in row.get("dataset_refs", []):
            dataset_id = str(did)
            if dataset_id and dataset_id not in dataset_ids:
                failures.append(f"experiment[{eid}] unknown dataset ref: {dataset_id}")

    # W5-016 experiment lineage.
    if int(lineages_meta.get("lineage_count", -1)) != len(lineages):
        failures.append("experiment_lineage lineage_count metadata mismatch")
    if int(lineages_meta.get("edge_count", -1)) != len(edges):
        failures.append("experiment_lineage edge_count metadata mismatch")

    by_experiment: dict[str, dict] = {str(row.get("experiment_id", "")): row for row in lineages}
    for eid in exp_ids:
        if eid not in by_experiment:
            failures.append(f"missing lineage row for experiment: {eid}")

    seen_lineage_ids: set[str] = set()
    lineage_to_experiment: dict[str, str] = {}
    for row in lineages:
        lid = str(row.get("id", ""))
        eid = str(row.get("experiment_id", ""))
        if lid in seen_lineage_ids:
            failures.append(f"duplicate lineage id: {lid}")
            continue
        seen_lineage_ids.add(lid)
        lineage_to_experiment[lid] = eid
        if eid not in exp_ids:
            failures.append(f"lineage[{lid}] unknown experiment_id: {eid}")
        if str(row.get("binary", "")) not in binary_names:
            failures.append(f"lineage[{lid}] unknown binary")
        if str(row.get("run_command_sha256", "")) != hashlib.sha256(
            str(row.get("run_command", "")).encode("utf-8")
        ).hexdigest():
            failures.append(f"lineage[{lid}] run_command_sha256 mismatch")
        for cid in row.get("claim_refs", []):
            if str(cid) not in claim_ids:
                failures.append(f"lineage[{lid}] unknown claim ref: {cid}")
        for did in row.get("dataset_refs", []):
            if str(did) not in dataset_ids:
                failures.append(f"lineage[{lid}] unknown dataset ref: {did}")

    edge_id_seen: set[str] = set()
    edge_kinds = {
        "implemented_by_binary",
        "supports_claim",
        "touches_dataset",
        "consumes_path",
        "produces_path",
    }
    to_kinds = {"binary", "claim", "dataset", "path"}
    binary_edge_lineages: set[str] = set()
    for row in edges:
        edge_id = str(row.get("id", ""))
        if edge_id in edge_id_seen:
            failures.append(f"duplicate lineage edge id: {edge_id}")
            continue
        edge_id_seen.add(edge_id)
        lid = str(row.get("lineage_id", ""))
        eid = str(row.get("from_id", ""))
        to_ref = str(row.get("to_ref", ""))
        to_kind = str(row.get("to_kind", ""))
        edge_kind = str(row.get("edge_kind", ""))
        if lid not in lineage_to_experiment:
            failures.append(f"lineage edge[{edge_id}] unknown lineage_id: {lid}")
        if eid not in exp_ids:
            failures.append(f"lineage edge[{edge_id}] unknown from_id: {eid}")
        elif lid in lineage_to_experiment and lineage_to_experiment.get(lid, "") != eid:
            failures.append(
                f"lineage edge[{edge_id}] from_id does not match lineage experiment: {lid}"
            )
        if to_kind not in to_kinds:
            failures.append(f"lineage edge[{edge_id}] invalid to_kind: {to_kind}")
        if edge_kind not in edge_kinds:
            failures.append(f"lineage edge[{edge_id}] invalid edge_kind: {edge_kind}")
        if to_kind == "binary":
            if to_ref not in binary_names:
                failures.append(f"lineage edge[{edge_id}] unknown binary ref: {to_ref}")
            binary_edge_lineages.add(lid)
        elif to_kind == "claim" and to_ref not in claim_ids:
            failures.append(f"lineage edge[{edge_id}] unknown claim ref: {to_ref}")
        elif to_kind == "dataset" and to_ref not in dataset_ids:
            failures.append(f"lineage edge[{edge_id}] unknown dataset ref: {to_ref}")
        elif to_kind == "path" and not to_ref:
            failures.append(f"lineage edge[{edge_id}] empty path ref")
    for lid in seen_lineage_ids:
        if lid not in binary_edge_lineages:
            failures.append(f"lineage[{lid}] missing binary edge")

    # W5-021 roadmap/todo/next-actions schema hardening.
    roadmap_meta = roadmap_raw.get("roadmap", {})
    roadmap_status_allow = set(roadmap_meta.get("status_allowlist", []))
    roadmap_priority_allow = set(roadmap_meta.get("priority_allowlist", []))
    if int(roadmap_meta.get("workstream_count", -1)) != len(workstreams):
        failures.append("roadmap workstream_count metadata mismatch")

    for row in workstreams:
        wid = str(row.get("id", ""))
        status = str(row.get("status", ""))
        priority = str(row.get("priority", ""))
        deps = [str(v) for v in row.get("dependencies", [])]
        acceptance = row.get("acceptance_criteria", [])
        if status not in roadmap_status_allow:
            failures.append(f"workstream[{wid}] status outside allowlist: {status}")
        if priority not in roadmap_priority_allow:
            failures.append(f"workstream[{wid}] priority outside allowlist: {priority}")
        if str(row.get("status_token", "")) != _status_token(status):
            failures.append(f"workstream[{wid}] status_token mismatch")
        if not isinstance(acceptance, list) or not acceptance:
            failures.append(f"workstream[{wid}] missing acceptance_criteria")
        _verify_dependencies(
            failures,
            deps,
            f"workstream[{wid}].dependencies",
            claim_ids,
            insight_ids,
            exp_ids,
            workstream_ids,
            todo_ids,
            action_ids,
            req_ids,
        )

    todo_meta = todo_raw.get("todo", {})
    todo_status_allow = set(todo_meta.get("status_allowlist", []))
    todo_priority_allow = set(todo_meta.get("priority_allowlist", []))
    if int(todo_meta.get("item_count", -1)) != len(todo_items):
        failures.append("todo item_count metadata mismatch")
    for row in todo_items:
        tid = str(row.get("id", ""))
        status = str(row.get("status", ""))
        priority = str(row.get("priority", ""))
        deps = [str(v) for v in row.get("dependencies", [])]
        acceptance = row.get("acceptance_criteria", [])
        if status not in todo_status_allow:
            failures.append(f"todo[{tid}] status outside allowlist: {status}")
        if priority not in todo_priority_allow:
            failures.append(f"todo[{tid}] priority outside allowlist: {priority}")
        if str(row.get("status_token", "")) != _status_token(status):
            failures.append(f"todo[{tid}] status_token mismatch")
        if not isinstance(acceptance, list) or not acceptance:
            failures.append(f"todo[{tid}] missing acceptance_criteria")
        _verify_dependencies(
            failures,
            deps,
            f"todo[{tid}].dependencies",
            claim_ids,
            insight_ids,
            exp_ids,
            workstream_ids,
            todo_ids,
            action_ids,
            req_ids,
        )

    actions_meta = actions_raw.get("meta", {})
    actions_status_allow = set(actions_meta.get("status_allowlist", []))
    actions_priority_allow = set(actions_meta.get("priority_allowlist", []))
    if int(actions_meta.get("action_count", -1)) != len(actions):
        failures.append("next_actions action_count metadata mismatch")
    for row in actions:
        aid = str(row.get("id", ""))
        status = str(row.get("status", ""))
        priority = str(row.get("priority", ""))
        deps = [str(v) for v in row.get("dependencies", [])]
        acceptance = row.get("acceptance_criteria", [])
        if status not in actions_status_allow:
            failures.append(f"next_actions[{aid}] status outside allowlist: {status}")
        if priority not in actions_priority_allow:
            failures.append(f"next_actions[{aid}] priority outside allowlist: {priority}")
        if str(row.get("status_token", "")) != _status_token(status):
            failures.append(f"next_actions[{aid}] status_token mismatch")
        if not isinstance(acceptance, list) or not acceptance:
            failures.append(f"next_actions[{aid}] missing acceptance_criteria")
        _verify_dependencies(
            failures,
            deps,
            f"next_actions[{aid}].dependencies",
            claim_ids,
            insight_ids,
            exp_ids,
            workstream_ids,
            todo_ids,
            action_ids,
            req_ids,
        )

    # W5-022 requirements decomposition.
    requirements_meta = requirements_raw.get("requirements", {})
    req_status_allow = set(requirements_meta.get("status_allowlist", []))
    runtime_allow = set(requirements_meta.get("runtime_stack_allowlist", []))
    if int(requirements_meta.get("module_count", -1)) != len(req_modules):
        failures.append("requirements module_count metadata mismatch")
    if int(requirements_meta.get("coverage_gap_count", -1)) != len(req_gaps):
        failures.append("requirements coverage_gap_count metadata mismatch")

    for row in req_modules:
        rid = str(row.get("id", ""))
        status = str(row.get("status", ""))
        runtime_stack = str(row.get("runtime_stack", ""))
        if status not in req_status_allow:
            failures.append(f"requirements.module[{rid}] invalid status: {status}")
        if str(row.get("status_token", "")) != _status_token(status):
            failures.append(f"requirements.module[{rid}] status_token mismatch")
        if runtime_stack not in runtime_allow:
            failures.append(f"requirements.module[{rid}] invalid runtime_stack: {runtime_stack}")
        for dep in row.get("requires_modules", []):
            dep_id = str(dep)
            if dep_id not in req_ids:
                failures.append(f"requirements.module[{rid}] unknown requires_modules ref: {dep_id}")
        for field in ("install_targets", "verify_targets", "acceptance_criteria"):
            value = row.get(field, [])
            if not isinstance(value, list):
                failures.append(f"requirements.module[{rid}] {field} must be list")

    for row in req_gaps:
        gid = str(row.get("id", ""))
        gap_status = str(row.get("status", ""))
        if gap_status not in {"open", "in_progress", "resolved", "blocked", "deferred"}:
            failures.append(f"requirements.coverage_gap[{gid}] invalid status: {gap_status}")
        if str(row.get("status_token", "")) != _status_token(gap_status):
            failures.append(f"requirements.coverage_gap[{gid}] status_token mismatch")
        for mid in row.get("related_module_ids", []):
            module_id = str(mid)
            if module_id and module_id not in req_ids:
                failures.append(
                    f"requirements.coverage_gap[{gid}] unknown related_module_id: {module_id}"
                )

    if int(mr_meta.get("module_count", -1)) != len(mr_modules):
        failures.append("module_requirements module_count metadata mismatch")
    if int(mr_meta.get("package_count", -1)) != len(mr_packages):
        failures.append("module_requirements package_count metadata mismatch")
    if int(mr_meta.get("command_count", -1)) != len(mr_commands):
        failures.append("module_requirements command_count metadata mismatch")
    if int(mr_meta.get("python_package_count", -1)) != sum(
        1 for row in mr_packages if str(row.get("manager", "")) == "pip"
    ):
        failures.append("module_requirements python_package_count metadata mismatch")
    if int(mr_meta.get("rust_package_count", -1)) != sum(
        1 for row in mr_packages if str(row.get("manager", "")) == "cargo"
    ):
        failures.append("module_requirements rust_package_count metadata mismatch")

    mr_module_ids = {str(row.get("id", "")) for row in mr_modules}
    if mr_module_ids != req_ids:
        failures.append("module_requirements module id set differs from requirements module set")

    command_ids = {str(row.get("id", "")) for row in mr_commands}
    package_ids = {str(row.get("id", "")) for row in mr_packages}
    for row in mr_modules:
        mid = str(row.get("id", ""))
        status = str(row.get("status", ""))
        if status not in req_status_allow:
            failures.append(f"module_requirements.module[{mid}] invalid status: {status}")
        if str(row.get("status_token", "")) != _status_token(status):
            failures.append(f"module_requirements.module[{mid}] status_token mismatch")
        for dep in row.get("requires_modules", []):
            dep_id = str(dep)
            if dep_id not in mr_module_ids:
                failures.append(
                    f"module_requirements.module[{mid}] unknown requires_modules ref: {dep_id}"
                )
        for cmd_id in row.get("command_refs", []):
            if str(cmd_id) not in command_ids:
                failures.append(
                    f"module_requirements.module[{mid}] unknown command_ref: {cmd_id}"
                )
        for pkg_id in row.get("package_refs", []):
            if str(pkg_id) not in package_ids:
                failures.append(
                    f"module_requirements.module[{mid}] unknown package_ref: {pkg_id}"
                )

    for row in mr_commands:
        cid = str(row.get("id", ""))
        mid = str(row.get("module_id", ""))
        kind = str(row.get("kind", ""))
        cmd = str(row.get("command", ""))
        if mid not in mr_module_ids:
            failures.append(f"module_requirements.command[{cid}] unknown module_id: {mid}")
        if kind not in {"install", "verify"}:
            failures.append(f"module_requirements.command[{cid}] invalid kind: {kind}")
        if not cmd:
            failures.append(f"module_requirements.command[{cid}] empty command")

    for row in mr_packages:
        pid = str(row.get("id", ""))
        mid = str(row.get("module_id", ""))
        manager = str(row.get("manager", ""))
        name = str(row.get("name", ""))
        if mid not in mr_module_ids:
            failures.append(f"module_requirements.package[{pid}] unknown module_id: {mid}")
        if manager not in {"pip", "cargo"}:
            failures.append(f"module_requirements.package[{pid}] invalid manager: {manager}")
        if not name:
            failures.append(f"module_requirements.package[{pid}] empty name")

    if failures:
        print("ERROR: Wave5 Batch4 registry verification failed.")
        for item in failures[:300]:
            print(f"- {item}")
        if len(failures) > 300:
            print(f"- ... and {len(failures) - 300} more failures")
        return 1

    print(
        "OK: Wave5 Batch4 registries verified. "
        f"experiments={len(experiments)} lineages={len(lineages)} edges={len(edges)} "
        f"workstreams={len(workstreams)} todo={len(todo_items)} actions={len(actions)} "
        f"req_modules={len(req_modules)} mr_packages={len(mr_packages)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

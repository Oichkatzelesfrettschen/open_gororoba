#!/usr/bin/env python3
"""
Verify cross-registry dangling references across claims, insights,
experiments, sources, and datasets.
"""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path


CLAIM_RE = re.compile(r"\bC-\d{3}\b")
INSIGHT_RE = re.compile(r"\bI-\d{3}\b")
EXPERIMENT_RE = re.compile(r"\bE-\d{3}\b")
SOURCE_RE = re.compile(r"\bXS-\d{3}\b")
DATASET_RE = re.compile(r"\b(?:PC|PG|EX|AR|CU)-\d{4}\b")
WORKSTREAM_RE = re.compile(r"\bWS-[A-Z0-9-]+\b")
TODO_RE = re.compile(r"\bT-\d{3}\b")
ACTION_RE = re.compile(r"\bNA-\d{3}\b")
REQ_RE = re.compile(r"\bREQ-[A-Z0-9-]+\b")


def _load(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _extract_refs(text: str) -> dict[str, list[str]]:
    return {
        "claims": sorted(set(CLAIM_RE.findall(text))),
        "insights": sorted(set(INSIGHT_RE.findall(text))),
        "experiments": sorted(set(EXPERIMENT_RE.findall(text))),
        "sources": sorted(set(SOURCE_RE.findall(text))),
        "datasets": sorted(set(DATASET_RE.findall(text))),
    }


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
        rows = raw.get("dataset", [])
        for row in rows:
            rid = str(row.get("id", ""))
            if rid:
                out.add(rid)
    return out


def _record_fail(failures: list[str], label: str, value: str) -> None:
    failures.append(f"{label}: {value}")


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
        "registry/experiments.toml",
        "registry/experiment_lineage.toml",
        "registry/roadmap.toml",
        "registry/todo.toml",
        "registry/next_actions.toml",
        "registry/requirements.toml",
        "registry/module_requirements.toml",
        "registry/external_sources.toml",
        "registry/claims_atoms.toml",
        "registry/claims_evidence_edges.toml",
        "registry/provenance_sources.toml",
        "registry/narrative_paragraph_atoms.toml",
    ]
    for rel in required:
        if not (root / rel).exists():
            raise SystemExit(f"ERROR: missing required registry {rel}")

    claims = _load(root / "registry/claims.toml").get("claim", [])
    insights = _load(root / "registry/insights.toml").get("insight", [])
    experiments = _load(root / "registry/experiments.toml").get("experiment", [])
    lineage_rows = _load(root / "registry/experiment_lineage.toml").get("lineage", [])
    lineage_edges = _load(root / "registry/experiment_lineage.toml").get("edge", [])
    roadmap_rows = _load(root / "registry/roadmap.toml").get("workstream", [])
    todo_rows = _load(root / "registry/todo.toml").get("item", [])
    action_rows = _load(root / "registry/next_actions.toml").get("action", [])
    requirements_rows = _load(root / "registry/requirements.toml").get("module", [])
    module_requirements_rows = _load(root / "registry/module_requirements.toml").get("module", [])
    module_requirements_packages = _load(root / "registry/module_requirements.toml").get("package", [])
    module_requirements_commands = _load(root / "registry/module_requirements.toml").get("command", [])
    sources = _load(root / "registry/external_sources.toml").get("document", [])
    claim_atoms = _load(root / "registry/claims_atoms.toml").get("atom", [])
    claim_edges = _load(root / "registry/claims_evidence_edges.toml").get("edge", [])
    provenance_rows = _load(root / "registry/provenance_sources.toml").get("record", [])
    paragraph_rows = _load(root / "registry/narrative_paragraph_atoms.toml").get("paragraph", [])

    conflict_rows = []
    if (root / "registry/conflict_markers.toml").exists():
        conflict_rows = _load(root / "registry/conflict_markers.toml").get("marker", [])
    lacunae_rows = []
    if (root / "registry/lacunae.toml").exists():
        lacunae_rows = _load(root / "registry/lacunae.toml").get("lacuna", [])

    claim_ids = {str(row.get("id", "")) for row in claims}
    insight_ids = {str(row.get("id", "")) for row in insights}
    experiment_ids = {str(row.get("id", "")) for row in experiments}
    workstream_ids = {str(row.get("id", "")) for row in roadmap_rows}
    todo_ids = {str(row.get("id", "")) for row in todo_rows}
    action_ids = {str(row.get("id", "")) for row in action_rows}
    req_ids = {str(row.get("id", "")) for row in requirements_rows}
    module_req_ids = {str(row.get("id", "")) for row in module_requirements_rows}
    source_ids = {str(row.get("id", "")) for row in sources}
    dataset_ids = _collect_dataset_ids(root)
    marker_ids = {str(row.get("id", "")) for row in conflict_rows}
    lineage_ids = {str(row.get("id", "")) for row in lineage_rows}

    failures: list[str] = []
    counters = {
        "claims": 0,
        "insights": 0,
        "experiments": 0,
        "sources": 0,
        "datasets": 0,
    }

    def check_id(kind: str, rid: str, where: str) -> None:
        if not rid:
            return
        counters[kind] += 1
        if kind == "claims" and rid not in claim_ids:
            _record_fail(failures, where, f"unknown claim {rid}")
        elif kind == "insights" and rid not in insight_ids:
            _record_fail(failures, where, f"unknown insight {rid}")
        elif kind == "experiments" and rid not in experiment_ids:
            _record_fail(failures, where, f"unknown experiment {rid}")
        elif kind == "sources" and rid not in source_ids:
            _record_fail(failures, where, f"unknown source {rid}")
        elif kind == "datasets" and rid not in dataset_ids:
            _record_fail(failures, where, f"unknown dataset {rid}")

    def check_dependency(dep: str, where: str) -> None:
        value = str(dep).strip()
        if not value:
            return
        if CLAIM_RE.fullmatch(value):
            check_id("claims", value, where)
            return
        if INSIGHT_RE.fullmatch(value):
            check_id("insights", value, where)
            return
        if EXPERIMENT_RE.fullmatch(value):
            check_id("experiments", value, where)
            return
        if WORKSTREAM_RE.fullmatch(value):
            if value not in workstream_ids:
                _record_fail(failures, where, f"unknown workstream {value}")
            return
        if TODO_RE.fullmatch(value):
            if value not in todo_ids:
                _record_fail(failures, where, f"unknown todo {value}")
            return
        if ACTION_RE.fullmatch(value):
            if value not in action_ids:
                _record_fail(failures, where, f"unknown action {value}")
            return
        if REQ_RE.fullmatch(value):
            if value not in req_ids:
                _record_fail(failures, where, f"unknown requirement module {value}")
            return
        _record_fail(failures, where, f"malformed dependency id {value}")

    # Structured refs: insights/experiments/sources
    for row in insights:
        sid = str(row.get("id", ""))
        for cid in row.get("claims", []):
            check_id("claims", str(cid), f"insights[{sid}].claims")

    for row in experiments:
        eid = str(row.get("id", ""))
        for cid in row.get("claims", []):
            check_id("claims", str(cid), f"experiments[{eid}].claims")

    for row in sources:
        xid = str(row.get("id", ""))
        for cid in row.get("claim_refs", []):
            check_id("claims", str(cid), f"external_sources[{xid}].claim_refs")

    # Claims free-text refs
    for row in claims:
        cid = str(row.get("id", ""))
        corpus = f"{row.get('where_stated', '')} {row.get('what_would_verify_refute', '')}"
        refs = _extract_refs(corpus)
        for rid in refs["claims"]:
            check_id("claims", rid, f"claims[{cid}].text")
        for rid in refs["insights"]:
            check_id("insights", rid, f"claims[{cid}].text")
        for rid in refs["experiments"]:
            check_id("experiments", rid, f"claims[{cid}].text")
        for rid in refs["sources"]:
            check_id("sources", rid, f"claims[{cid}].text")
        for rid in refs["datasets"]:
            check_id("datasets", rid, f"claims[{cid}].text")

    # claims_atoms
    for row in claim_atoms:
        aid = str(row.get("id", ""))
        check_id("claims", str(row.get("claim_id", "")), f"claims_atoms[{aid}].claim_id")
        for field in ("cross_refs", "where_stated_refs", "verification_refs"):
            for value in row.get(field, []):
                refs = _extract_refs(str(value))
                for rid in refs["claims"]:
                    check_id("claims", rid, f"claims_atoms[{aid}].{field}")
                for rid in refs["insights"]:
                    check_id("insights", rid, f"claims_atoms[{aid}].{field}")
                for rid in refs["experiments"]:
                    check_id("experiments", rid, f"claims_atoms[{aid}].{field}")
                for rid in refs["sources"]:
                    check_id("sources", rid, f"claims_atoms[{aid}].{field}")
                for rid in refs["datasets"]:
                    check_id("datasets", rid, f"claims_atoms[{aid}].{field}")

    # claims_evidence_edges
    for row in claim_edges:
        eid = str(row.get("id", ""))
        check_id("claims", str(row.get("claim_id", "")), f"claims_evidence_edges[{eid}].claim_id")
        target_ref = str(row.get("target_ref", ""))
        refs = _extract_refs(target_ref)
        for rid in refs["claims"]:
            check_id("claims", rid, f"claims_evidence_edges[{eid}].target_ref")
        for rid in refs["insights"]:
            check_id("insights", rid, f"claims_evidence_edges[{eid}].target_ref")
        for rid in refs["experiments"]:
            check_id("experiments", rid, f"claims_evidence_edges[{eid}].target_ref")
        for rid in refs["sources"]:
            check_id("sources", rid, f"claims_evidence_edges[{eid}].target_ref")
        for rid in refs["datasets"]:
            check_id("datasets", rid, f"claims_evidence_edges[{eid}].target_ref")

    # provenance and paragraph atoms
    for row in provenance_rows:
        pid = str(row.get("id", ""))
        for cid in row.get("claim_refs", []):
            check_id("claims", str(cid), f"provenance_sources[{pid}].claim_refs")
        refs = _extract_refs(str(row.get("source_ref", "")))
        for rid in refs["sources"]:
            check_id("sources", rid, f"provenance_sources[{pid}].source_ref")
        for rid in refs["datasets"]:
            check_id("datasets", rid, f"provenance_sources[{pid}].source_ref")

    for row in paragraph_rows:
        pid = str(row.get("id", ""))
        for cid in row.get("claim_refs", []):
            check_id("claims", str(cid), f"narrative_paragraph_atoms[{pid}].claim_refs")

    # experiment lineage
    for row in lineage_rows:
        lid = str(row.get("id", ""))
        check_id("experiments", str(row.get("experiment_id", "")), f"experiment_lineage[{lid}].experiment_id")
        for cid in row.get("claim_refs", []):
            check_id("claims", str(cid), f"experiment_lineage[{lid}].claim_refs")
        for did in row.get("dataset_refs", []):
            check_id("datasets", str(did), f"experiment_lineage[{lid}].dataset_refs")

    for row in lineage_edges:
        eid = str(row.get("id", ""))
        lid = str(row.get("lineage_id", ""))
        if lid not in lineage_ids:
            _record_fail(failures, f"experiment_lineage.edge[{eid}].lineage_id", f"unknown lineage {lid}")
        check_id("experiments", str(row.get("from_id", "")), f"experiment_lineage.edge[{eid}].from_id")
        to_ref = str(row.get("to_ref", ""))
        to_kind = str(row.get("to_kind", ""))
        if to_kind == "claim":
            check_id("claims", to_ref, f"experiment_lineage.edge[{eid}].to_ref")
        elif to_kind == "dataset":
            check_id("datasets", to_ref, f"experiment_lineage.edge[{eid}].to_ref")

    # planning registries
    for row in roadmap_rows:
        wid = str(row.get("id", ""))
        for dep in row.get("dependencies", []):
            check_dependency(str(dep), f"roadmap.workstream[{wid}].dependencies")
        for cid in row.get("claims", []):
            check_id("claims", str(cid), f"roadmap.workstream[{wid}].claims")
        insight = str(row.get("insight", ""))
        if insight:
            check_id("insights", insight, f"roadmap.workstream[{wid}].insight")
        for value in row.get("evidence_refs", []):
            refs = _extract_refs(str(value))
            for rid in refs["claims"]:
                check_id("claims", rid, f"roadmap.workstream[{wid}].evidence_refs")
            for rid in refs["insights"]:
                check_id("insights", rid, f"roadmap.workstream[{wid}].evidence_refs")
            for rid in refs["experiments"]:
                check_id("experiments", rid, f"roadmap.workstream[{wid}].evidence_refs")
            for rid in refs["sources"]:
                check_id("sources", rid, f"roadmap.workstream[{wid}].evidence_refs")
            for rid in refs["datasets"]:
                check_id("datasets", rid, f"roadmap.workstream[{wid}].evidence_refs")

    for row in todo_rows:
        tid = str(row.get("id", ""))
        for dep in row.get("dependencies", []):
            check_dependency(str(dep), f"todo.item[{tid}].dependencies")

    for row in action_rows:
        aid = str(row.get("id", ""))
        for dep in row.get("dependencies", []):
            check_dependency(str(dep), f"next_actions.action[{aid}].dependencies")
        for value in row.get("evidence_refs", []):
            refs = _extract_refs(str(value))
            for rid in refs["claims"]:
                check_id("claims", rid, f"next_actions.action[{aid}].evidence_refs")
            for rid in refs["insights"]:
                check_id("insights", rid, f"next_actions.action[{aid}].evidence_refs")
            for rid in refs["experiments"]:
                check_id("experiments", rid, f"next_actions.action[{aid}].evidence_refs")
            for rid in refs["sources"]:
                check_id("sources", rid, f"next_actions.action[{aid}].evidence_refs")
            for rid in refs["datasets"]:
                check_id("datasets", rid, f"next_actions.action[{aid}].evidence_refs")

    # requirements registries
    for row in requirements_rows:
        rid = str(row.get("id", ""))
        for mid in row.get("requires_modules", []):
            module_id = str(mid)
            if module_id not in req_ids:
                _record_fail(
                    failures,
                    f"requirements.module[{rid}].requires_modules",
                    f"unknown requirement module {module_id}",
                )

    for row in module_requirements_rows:
        rid = str(row.get("id", ""))
        if rid not in req_ids:
            _record_fail(failures, "module_requirements.module", f"missing requirements module {rid}")
        for mid in row.get("requires_modules", []):
            module_id = str(mid)
            if module_id not in module_req_ids:
                _record_fail(
                    failures,
                    f"module_requirements.module[{rid}].requires_modules",
                    f"unknown module {module_id}",
                )

    package_ids = {str(row.get("id", "")) for row in module_requirements_packages}
    command_ids = {str(row.get("id", "")) for row in module_requirements_commands}
    for row in module_requirements_rows:
        rid = str(row.get("id", ""))
        for pid in row.get("package_refs", []):
            package_id = str(pid)
            if package_id not in package_ids:
                _record_fail(
                    failures,
                    f"module_requirements.module[{rid}].package_refs",
                    f"unknown package {package_id}",
                )
        for cid in row.get("command_refs", []):
            command_id = str(cid)
            if command_id not in command_ids:
                _record_fail(
                    failures,
                    f"module_requirements.module[{rid}].command_refs",
                    f"unknown command {command_id}",
                )

    for row in module_requirements_packages:
        pid = str(row.get("id", ""))
        module_id = str(row.get("module_id", ""))
        if module_id not in module_req_ids:
            _record_fail(
                failures,
                f"module_requirements.package[{pid}].module_id",
                f"unknown module {module_id}",
            )

    for row in module_requirements_commands:
        cid = str(row.get("id", ""))
        module_id = str(row.get("module_id", ""))
        if module_id not in module_req_ids:
            _record_fail(
                failures,
                f"module_requirements.command[{cid}].module_id",
                f"unknown module {module_id}",
            )

    # conflict markers + lacunae
    for row in conflict_rows:
        mid = str(row.get("id", ""))
        for cid in row.get("claim_refs", []):
            check_id("claims", str(cid), f"conflict_markers[{mid}].claim_refs")

    for row in lacunae_rows:
        lid = str(row.get("id", ""))
        for cid in row.get("claim_refs", []):
            check_id("claims", str(cid), f"lacunae[{lid}].claim_refs")
        for mid in row.get("related_marker_ids", []):
            if str(mid) not in marker_ids:
                _record_fail(failures, f"lacunae[{lid}].related_marker_ids", f"unknown marker {mid}")

    if failures:
        print("ERROR: cross-registry reference verification failed.")
        for item in failures[:300]:
            print(f"- {item}")
        if len(failures) > 300:
            print(f"- ... and {len(failures) - 300} more failures")
        return 1

    print(
        "OK: cross-registry references verified. "
        f"checks claims={counters['claims']} insights={counters['insights']} "
        f"experiments={counters['experiments']} sources={counters['sources']} "
        f"datasets={counters['datasets']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

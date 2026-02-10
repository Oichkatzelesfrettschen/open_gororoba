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
    source_ids = {str(row.get("id", "")) for row in sources}
    dataset_ids = _collect_dataset_ids(root)
    marker_ids = {str(row.get("id", "")) for row in conflict_rows}

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

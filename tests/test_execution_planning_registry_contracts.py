"""
Focused contract checks for execution-planning registry normalization.
"""

from __future__ import annotations

import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_PATH = REPO_ROOT / "registry/experiments.toml"
LINEAGE_PATH = REPO_ROOT / "registry/experiment_lineage.toml"


def _load(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_script_runner_experiments_do_not_emit_binary_edges() -> None:
    experiments_raw = _load(EXPERIMENTS_PATH)
    lineage_raw = _load(LINEAGE_PATH)

    experiments = experiments_raw["experiment"]
    lineages = {row["experiment_id"]: row for row in lineage_raw["lineage"]}
    edges = lineage_raw["edge"]

    script_runner_ids = {
        row["id"]
        for row in experiments
        if str(row.get("run", "")).startswith("python3 ") and not str(row.get("binary", ""))
    }
    assert script_runner_ids, "expected script-runner experiments with empty binary fields"

    binary_edge_experiments = {
        row["from_id"] for row in edges if row.get("to_kind") == "binary"
    }

    for experiment_id in sorted(script_runner_ids):
        experiment_row = next(row for row in experiments if row["id"] == experiment_id)
        lineage_row = lineages[experiment_id]

        assert experiment_row["binary"] == ""
        assert lineage_row["binary"] == ""
        assert experiment_id not in binary_edge_experiments
        assert (
            "Execution command is explicitly declared."
            in lineage_row["acceptance_criteria"]
        )


def test_src_provider_contract_refs_survive_lineage_projection() -> None:
    experiments_raw = _load(EXPERIMENTS_PATH)
    lineage_raw = _load(LINEAGE_PATH)

    experiment_row = next(
        row for row in experiments_raw["experiment"] if row["id"] == "E-140"
    )
    lineage_row = next(
        row for row in lineage_raw["lineage"] if row["experiment_id"] == "E-140"
    )
    source_edges = [
        row
        for row in lineage_raw["edge"]
        if row["from_id"] == "E-140" and row["to_kind"] == "source"
    ]

    assert experiment_row["binary"] == ""
    assert lineage_row["binary"] == ""
    assert experiment_row["external_source_refs"]
    assert all(
        str(value).startswith("SRC-") for value in experiment_row["external_source_refs"]
    )
    assert (
        experiment_row["external_source_refs"] == lineage_row["external_source_refs"]
    )
    assert sorted(row["to_ref"] for row in source_edges) == sorted(
        experiment_row["external_source_refs"]
    )


def test_experiment_lineage_edge_count_metadata_matches_actual_edges() -> None:
    lineage_raw = _load(LINEAGE_PATH)
    assert lineage_raw["experiment_lineage"]["edge_count"] == len(lineage_raw["edge"])

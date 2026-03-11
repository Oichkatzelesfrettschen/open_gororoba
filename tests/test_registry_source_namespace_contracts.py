"""
Focused contracts for source-reference namespaces across registry lanes.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_PATH = REPO_ROOT / "registry/experiments.toml"
EXTERNAL_SOURCES_PATH = REPO_ROOT / "registry/external_sources.toml"
SOURCE_CONTRACTS_PATH = REPO_ROOT / "data/external/SOURCES.toml"
CROSSREFS_SCRIPT = REPO_ROOT / "src/verification/verify_registry_crossrefs.py"
pytestmark = pytest.mark.smoke


def _load(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _load_crossrefs_module():
    spec = importlib.util.spec_from_file_location(
        "verify_registry_crossrefs", CROSSREFS_SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_crossrefs_verifier_script_passes() -> None:
    env = dict(os.environ)
    env["PYTHONWARNINGS"] = "error"
    proc = subprocess.run(
        [sys.executable, str(CROSSREFS_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, (
        "Verifier exited non-zero.\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )
    assert "OK: cross-registry references verified." in proc.stdout


def test_source_refs_resolve_against_their_declared_namespace() -> None:
    experiments = _load(EXPERIMENTS_PATH)["experiment"]
    dossier_ids = {
        row["id"] for row in _load(EXTERNAL_SOURCES_PATH).get("document", [])
    }
    source_contract_ids = {
        row["id"] for row in _load(SOURCE_CONTRACTS_PATH).get("source", [])
    }

    assert dossier_ids
    assert source_contract_ids

    for row in experiments:
        experiment_id = row["id"]
        refs = [str(value) for value in row.get("external_source_refs", [])]
        for ref in refs:
            if ref.startswith("XS-"):
                assert ref in dossier_ids, f"{experiment_id} missing dossier ref {ref}"
            elif ref.startswith("SRC-"):
                assert (
                    ref in source_contract_ids
                ), f"{experiment_id} missing source-contract ref {ref}"
            else:
                raise AssertionError(
                    f"{experiment_id} uses unsupported external source namespace: {ref}"
                )


def test_truth_surface_experiments_continue_to_use_xs_dossiers() -> None:
    experiments = _load(EXPERIMENTS_PATH)["experiment"]
    surface_rows = [
        row for row in experiments if row.get("truth_surface_consumption", [])
    ]

    assert surface_rows, "expected truth-surface experiments to exist"
    for row in surface_rows:
        refs = [str(value) for value in row.get("external_source_refs", [])]
        assert refs, f"{row['id']} should carry external source refs"
        assert all(
            ref.startswith("XS-") for ref in refs
        ), f"{row['id']} should use dossier refs for truth-surface contracts"


def test_wildcard_source_stems_do_not_extract_fake_contract_ids() -> None:
    module = _load_crossrefs_module()
    refs = module._extract_refs(  # type: ignore[attr-defined]
        "data/external/SOURCES.toml (SRC-EUCLID-Q1-ZENODO-*), "
        "data/external/euclid/zenodo/euclid_zenodo_manifest.json"
    )
    assert refs["sources"] == []

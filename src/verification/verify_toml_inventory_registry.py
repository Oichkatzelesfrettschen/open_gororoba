#!/usr/bin/env python3
"""
Verify Wave 4 TOML inventory coverage and role consistency.
"""

from __future__ import annotations

from pathlib import Path
import tomllib


def _expected_role(path: str) -> str:
    if path.startswith("registry/data/"):
        return "dataset_scroll"
    if path.startswith("registry/"):
        return "registry_control_plane"
    if path == "Cargo.toml":
        return "cargo_workspace_manifest"
    if path.endswith("/Cargo.toml"):
        return "cargo_crate_manifest"
    if path == ".cargo/config.toml":
        return "cargo_toolchain_config"
    if path == "pyproject.toml":
        return "python_project_config"
    if path.startswith("papers/"):
        return "papers_registry"
    return "toml_other"


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    inv_path = repo_root / "registry/toml_inventory.toml"
    md_inv_path = repo_root / "registry/markdown_inventory.toml"

    inv = tomllib.loads(inv_path.read_text(encoding="utf-8"))
    md_inv = tomllib.loads(md_inv_path.read_text(encoding="utf-8"))

    failures: list[str] = []
    summary = inv.get("toml_inventory", {})
    docs = inv.get("document", [])
    doc_paths = [str(row.get("path", "")).strip() for row in docs]
    doc_path_set = set(doc_paths)

    if int(summary.get("document_count", -1)) != len(docs):
        failures.append(
            f"document_count mismatch: {summary.get('document_count')} vs {len(docs)}"
        )
    if int(summary.get("parse_error_count", -1)) != 0:
        failures.append(f"parse_error_count={summary.get('parse_error_count')} (expected 0)")

    if len(doc_paths) != len(doc_path_set):
        failures.append("duplicate TOML paths detected in registry/toml_inventory.toml")

    for row in docs:
        path = str(row.get("path", "")).strip()
        role = str(row.get("role", "")).strip()
        parse_ok = bool(row.get("parse_ok", False))

        if not (repo_root / path).is_file():
            failures.append(f"{path}: file missing on disk")
        if not parse_ok:
            failures.append(f"{path}: parse_ok=false")

        expected = _expected_role(path)
        if role != expected:
            failures.append(f"{path}: role={role} expected={expected}")

    required_core = {
        "registry/claims.toml",
        "registry/insights.toml",
        "registry/experiments.toml",
        "registry/bibliography.toml",
        "registry/roadmap.toml",
        "registry/todo.toml",
        "registry/next_actions.toml",
        "registry/markdown_inventory.toml",
        "registry/markdown_governance.toml",
        "registry/markdown_corpus_registry.toml",
        "registry/toml_inventory.toml",
        "registry/csv_inventory.toml",
        "registry/csv_migration_scope.toml",
        "registry/wave4_roadmap.toml",
    }
    for path in sorted(required_core):
        if path not in doc_path_set:
            failures.append(f"missing required TOML inventory path: {path}")

    expected_destinations: set[str] = set()
    for row in md_inv.get("document", []):
        classification = str(row.get("classification", "")).strip()
        destination = str(row.get("toml_destination", "")).strip()
        if classification == "toml_published_markdown" and destination:
            expected_destinations.add(destination)

    for destination in sorted(expected_destinations):
        if destination not in doc_path_set:
            failures.append(f"markdown destination missing from TOML inventory: {destination}")

    if failures:
        print("ERROR: Wave 4 TOML inventory verification failed.")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("OK: Wave 4 TOML inventory coverage and role checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Verify Wave 4 markdown corpus control-plane invariants.
"""

from __future__ import annotations

import tomllib
from pathlib import Path


def _load_governance(repo_root: Path) -> tuple[set[str], set[str]]:
    gov = tomllib.loads(
        (repo_root / "registry/markdown_governance.toml").read_text(encoding="utf-8")
    )
    policy = gov.get("policy", {})
    safe_classifications = {
        str(item).strip() for item in policy.get("safe_classifications", []) if str(item).strip()
    }
    tracked_allowed_modes = {
        str(item).strip() for item in policy.get("tracked_allowed_modes", []) if str(item).strip()
    }
    tracked_allowed_paths = {
        str(item).strip().replace("\\", "/")
        for item in policy.get("tracked_allowed_paths", [])
        if str(item).strip()
    }
    for row in gov.get("document", []):
        path = str(row.get("path", "")).strip().replace("\\", "/")
        mode = str(row.get("mode", "")).strip()
        if path and mode in tracked_allowed_modes:
            tracked_allowed_paths.add(path)
    return safe_classifications, tracked_allowed_paths


def _in_policy_scope(path: str) -> bool:
    # Strict mode applies to every markdown row discovered in inventory.
    return bool(path)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    inv_path = repo_root / "registry/markdown_inventory.toml"
    corpus_path = repo_root / "registry/markdown_corpus_registry.toml"

    inv = tomllib.loads(inv_path.read_text(encoding="utf-8"))
    corpus = tomllib.loads(corpus_path.read_text(encoding="utf-8"))
    safe_classifications, allowed_tracked_markdown = _load_governance(repo_root)

    failures: list[str] = []
    docs = inv.get("document", [])
    policy = corpus.get("policy", {})

    safe_from_corpus = {
        str(item).strip() for item in policy.get("safe_classifications", []) if str(item).strip()
    }
    if safe_from_corpus and safe_from_corpus != safe_classifications:
        failures.append(
            "markdown_corpus_registry policy safe_classifications drift from markdown_governance"
        )

    allowed_from_corpus = {
        str(item).strip().replace("\\", "/")
        for item in policy.get("allowed_tracked_markdown", [])
        if str(item).strip()
    }
    if allowed_from_corpus and allowed_from_corpus != allowed_tracked_markdown:
        failures.append(
            "markdown_corpus_registry allowed_tracked_markdown drift from markdown_governance"
        )

    for row in docs:
        path = str(row.get("path", "")).strip()
        git_status = str(row.get("git_status", "")).strip()
        classification = str(row.get("classification", "")).strip()
        destination = str(row.get("toml_destination", "")).strip()

        if _in_policy_scope(path) and classification not in safe_classifications:
            failures.append(f"{path}: classification={classification} is outside safe set")

        if (
            _in_policy_scope(path)
            and git_status == "tracked"
            and path not in allowed_tracked_markdown
        ):
            failures.append(f"{path}: tracked markdown is outside allowlist")

        if _in_policy_scope(path) and classification == "toml_published_markdown":
            if not destination:
                failures.append(f"{path}: missing toml_destination")
            elif not (repo_root / destination).is_file():
                failures.append(f"{path}: toml_destination not found -> {destination}")

    if failures:
        print("ERROR: Wave 4 markdown corpus policy verification failed.")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("OK: Wave 4 markdown corpus policy matches markdown_governance and markdown_inventory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

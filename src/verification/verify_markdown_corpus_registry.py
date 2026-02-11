#!/usr/bin/env python3
"""
Verify Wave 4 markdown corpus control-plane invariants.
"""

from __future__ import annotations

from pathlib import Path
import tomllib

SAFE_CLASSIFICATIONS = {
    "toml_published_markdown",
    "toml_destination_exists_manual_markdown",
    "generated_artifact",
    "third_party_markdown",
}

ALLOWED_TRACKED_MARKDOWN: set[str] = set()


def _in_policy_scope(path: str) -> bool:
    # Strict mode applies to every markdown row discovered in inventory.
    return bool(path)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    inv_path = repo_root / "registry/markdown_inventory.toml"
    corpus_path = repo_root / "registry/markdown_corpus_registry.toml"

    inv = tomllib.loads(inv_path.read_text(encoding="utf-8"))
    corpus = tomllib.loads(corpus_path.read_text(encoding="utf-8"))

    failures: list[str] = []
    docs = inv.get("document", [])
    corpus_docs = corpus.get("document", [])
    corpus_summary = corpus.get("markdown_corpus_registry", {})

    tracked_violations = 0
    class_violations = 0
    destination_missing = 0

    for row in docs:
        path = str(row.get("path", "")).strip()
        git_status = str(row.get("git_status", "")).strip()
        classification = str(row.get("classification", "")).strip()
        destination = str(row.get("toml_destination", "")).strip()

        if _in_policy_scope(path) and classification not in SAFE_CLASSIFICATIONS:
            class_violations += 1
            failures.append(f"{path}: classification={classification} is outside safe set")

        if _in_policy_scope(path) and git_status == "tracked" and path not in ALLOWED_TRACKED_MARKDOWN:
            tracked_violations += 1
            failures.append(f"{path}: tracked markdown is outside allowlist")

        if _in_policy_scope(path) and classification == "toml_published_markdown":
            if not destination:
                destination_missing += 1
                failures.append(f"{path}: missing toml_destination")
            elif not (repo_root / destination).is_file():
                destination_missing += 1
                failures.append(f"{path}: toml_destination not found -> {destination}")

    if int(corpus_summary.get("document_count", -1)) != len(docs):
        failures.append(
            "markdown_corpus_registry.document_count mismatch: "
            f"{corpus_summary.get('document_count')} vs {len(docs)}"
        )
    if len(corpus_docs) != len(docs):
        failures.append(f"corpus document table mismatch: {len(corpus_docs)} vs {len(docs)}")

    if int(corpus_summary.get("tracked_violation_count", -1)) != tracked_violations:
        failures.append(
            "tracked_violation_count mismatch: "
            f"{corpus_summary.get('tracked_violation_count')} vs {tracked_violations}"
        )
    if int(corpus_summary.get("classification_violation_count", -1)) != class_violations:
        failures.append(
            "classification_violation_count mismatch: "
            f"{corpus_summary.get('classification_violation_count')} vs {class_violations}"
        )
    if int(corpus_summary.get("destination_missing_count", -1)) != destination_missing:
        failures.append(
            "destination_missing_count mismatch: "
            f"{corpus_summary.get('destination_missing_count')} vs {destination_missing}"
        )

    for row in corpus.get("policy_violation", []):
        kind = str(row.get("kind", "")).strip()
        path = str(row.get("path", "")).strip()
        failures.append(f"policy_violation present: {kind} -> {path}")

    if failures:
        print("ERROR: Wave 4 markdown corpus registry verification failed (strict TOML-only mode).")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("OK: Wave 4 markdown corpus registry invariants satisfied (no tracked markdown).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

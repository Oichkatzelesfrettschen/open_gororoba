#!/usr/bin/env python3
"""
Verify markdown origin audit invariants for docs/reports/data-artifacts.
"""

from __future__ import annotations

from pathlib import Path
import tomllib


ALLOWED_SCOPES = {"docs", "reports", "data_artifacts"}


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    audit_path = repo_root / "registry/markdown_origin_audit.toml"
    data = tomllib.loads(audit_path.read_text(encoding="utf-8"))

    failures: list[str] = []
    summary = data.get("markdown_origin_audit", {})
    docs = data.get("document", [])
    queue = data.get("consolidation_queue", [])

    if int(summary.get("document_count", -1)) != len(docs):
        failures.append(
            f"document_count mismatch: {summary.get('document_count')} vs {len(docs)}"
        )
    if int(summary.get("needs_consolidation_count", -1)) != len(queue):
        failures.append(
            "needs_consolidation_count mismatch: "
            f"{summary.get('needs_consolidation_count')} vs {len(queue)}"
        )
    if len(queue) != 0:
        failures.append(f"consolidation_queue not empty ({len(queue)} items)")

    for row in docs:
        path = str(row.get("path", "")).strip()
        scope = str(row.get("scope", "")).strip()
        status = str(row.get("origin_status", "")).strip()
        destination = str(row.get("toml_destination", "")).strip()
        destination_exists = bool(row.get("destination_exists", False))
        has_auto = bool(row.get("header_auto_generated", False))
        has_source = bool(row.get("header_source_of_truth", False))

        if scope not in ALLOWED_SCOPES:
            failures.append(f"{path}: invalid scope={scope}")
        if status != "generated_from_repo_process":
            failures.append(f"{path}: origin_status={status}")
        if not destination:
            failures.append(f"{path}: missing toml_destination")
        elif not destination_exists:
            failures.append(f"{path}: toml_destination missing on disk ({destination})")
        if not has_auto or not has_source:
            failures.append(
                f"{path}: missing headers auto={has_auto} source={has_source}"
            )

    if failures:
        print("ERROR: markdown origin audit verification failed.")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("OK: markdown origin audit verified (all in-scope markdown is TOML-generated).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

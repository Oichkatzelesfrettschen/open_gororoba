#!/usr/bin/env python3
"""
Build a Wave 4 markdown control-plane registry.

Input:
- registry/markdown_inventory.toml

Output:
- registry/markdown_corpus_registry.toml

Goal:
- Enforce one-source-of-truth TOML governance for project markdown.
- Keep only explicitly allowed tracked markdown entrypoints.
- Surface any lifecycle/policy drift as machine-readable violations.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from collections import Counter
from pathlib import Path

SAFE_CLASSIFICATIONS = {"toml_published_markdown", "third_party_markdown"}

ALLOWED_TRACKED_MARKDOWN = {
    "AGENTS.md",
    "CLAUDE.md",
    "GEMINI.md",
    "PANTHEON_PHYSICSFORGE_90_POINT_MIGRATION_PLAN.md",
    "PHASE10_11_ULTIMATE_ROADMAP.md",
    "PYTHON_REFACTORING_ROADMAP.md",
    "README.md",
    "curated/README.md",
    "curated/01_theory_frameworks/README_COQ.md",
    "data/artifacts/README.md",
    "data/csv/README.md",
}

POLICY_PREFIXES = (
    "docs/",
    "reports/",
    "data/artifacts/",
)


def _in_policy_scope(path: str) -> bool:
    if any(path.startswith(prefix) for prefix in POLICY_PREFIXES):
        return True
    return path in ALLOWED_TRACKED_MARKDOWN


def _assert_ascii(text: str, context: str) -> None:
    bad = sorted({ch for ch in text if ord(ch) > 127})
    if bad:
        raise SystemExit(f"ERROR: Non-ASCII output in {context}: {''.join(bad[:20])!r}")


def _esc(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def _bool(value: object) -> bool:
    return bool(value)


def _lifecycle(path: str, row: dict[str, object]) -> str:
    if _bool(row.get("third_party")):
        return "third_party_cache"
    if path in ALLOWED_TRACKED_MARKDOWN and str(row.get("git_status", "")) == "tracked":
        return "tracked_entrypoint"
    if path.startswith("docs/generated/"):
        return "generated_publish_mirror"
    if path.startswith("docs/book/src/"):
        return "generated_book_source"
    if path.startswith("docs/convos/"):
        return "generated_conversation_extract"
    if path.startswith("docs/"):
        return "generated_docs_tree"
    if path.startswith("data/artifacts/"):
        return "generated_artifact_report"
    if path.startswith("reports/"):
        return "generated_report_output"
    if path in {"NAVIGATOR.md", "REQUIREMENTS.md"}:
        return "generated_root_overlay"
    return "generated_other"


def _risk_score(path: str, row: dict[str, object], destination_exists: bool) -> int:
    score = 0
    classification = str(row.get("classification", ""))
    migration_action = str(row.get("migration_action", ""))
    git_status = str(row.get("git_status", ""))
    line_count = int(row.get("line_count", 0))
    generated = _bool(row.get("generated"))
    third_party = _bool(row.get("third_party"))

    if classification not in SAFE_CLASSIFICATIONS:
        score += 100
    if git_status == "tracked" and path not in ALLOWED_TRACKED_MARKDOWN:
        score += 80
    if (not generated) and (not third_party):
        score += 80
    if migration_action in {"migrate_to_new_registry", "port_body_to_toml_and_lock_mirror"}:
        score += 50
    if classification == "toml_published_markdown" and not destination_exists:
        score += 50
    score += min(line_count // 250, 20)
    return score


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[3]),
        help="Repository root.",
    )
    parser.add_argument(
        "--inventory",
        default="registry/markdown_inventory.toml",
        help="Input markdown inventory TOML.",
    )
    parser.add_argument(
        "--out",
        default="registry/markdown_corpus_registry.toml",
        help="Output Wave 4 markdown corpus registry TOML.",
    )
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    inv_path = root / args.inventory
    out_path = root / args.out

    inv = tomllib.loads(inv_path.read_text(encoding="utf-8"))
    docs = inv.get("document", [])

    git_status_counts: Counter[str] = Counter()
    classification_counts: Counter[str] = Counter()
    lifecycle_counts: Counter[str] = Counter()

    tracked_violations: list[str] = []
    classification_violations: list[str] = []
    destination_missing: list[str] = []
    risk_rows: list[tuple[int, int, str, dict[str, object], str, bool]] = []

    for row in docs:
        path = str(row.get("path", ""))
        git_status = str(row.get("git_status", ""))
        classification = str(row.get("classification", ""))
        line_count = int(row.get("line_count", 0))
        toml_destination = str(row.get("toml_destination", "")).strip()

        destination_exists = bool(toml_destination) and (root / toml_destination).is_file()
        lifecycle = _lifecycle(path, row)
        risk = _risk_score(path, row, destination_exists)

        git_status_counts[git_status] += 1
        classification_counts[classification] += 1
        lifecycle_counts[lifecycle] += 1

        if _in_policy_scope(path) and git_status == "tracked" and path not in ALLOWED_TRACKED_MARKDOWN:
            tracked_violations.append(path)
        if _in_policy_scope(path) and classification not in SAFE_CLASSIFICATIONS:
            classification_violations.append(path)
        if _in_policy_scope(path) and classification == "toml_published_markdown" and not destination_exists:
            destination_missing.append(path)

        if risk > 0:
            risk_rows.append((risk, line_count, path, row, lifecycle, destination_exists))

    risk_rows.sort(key=lambda item: (-item[0], -item[1], item[2]))

    lines: list[str] = []
    lines.append("# Wave 4 markdown corpus control-plane registry (TOML-first).")
    lines.append("# Generated by src/scripts/analysis/build_markdown_corpus_registry.py")
    lines.append("")
    lines.append("[markdown_corpus_registry]")
    lines.append('updated = "deterministic"')
    lines.append("authoritative = true")
    lines.append(f"source_inventory = {_esc(args.inventory)}")
    lines.append(f"document_count = {len(docs)}")
    lines.append(f"tracked_violation_count = {len(tracked_violations)}")
    lines.append(f"classification_violation_count = {len(classification_violations)}")
    lines.append(f"destination_missing_count = {len(destination_missing)}")
    lines.append(f"risk_item_count = {len(risk_rows)}")
    lines.append("")

    lines.append("[policy]")
    lines.append("toml_first_required = true")
    lines.append("allow_tracked_markdown_entrypoints_only = true")
    lines.append('safe_classifications = ["toml_published_markdown", "third_party_markdown"]')
    lines.append("")

    lines.append("allowed_tracked_markdown = [")
    for path in sorted(ALLOWED_TRACKED_MARKDOWN):
        lines.append(f"  {_esc(path)},")
    lines.append("]")
    lines.append("")

    lines.append("[git_status_counts]")
    for key in sorted(git_status_counts):
        lines.append(f"{key} = {git_status_counts[key]}")
    lines.append("")

    lines.append("[classification_counts]")
    for key in sorted(classification_counts):
        lines.append(f"{key} = {classification_counts[key]}")
    lines.append("")

    lines.append("[lifecycle_counts]")
    for key in sorted(lifecycle_counts):
        lines.append(f"{key} = {lifecycle_counts[key]}")
    lines.append("")

    for path in sorted(tracked_violations):
        lines.append("[[policy_violation]]")
        lines.append('kind = "tracked_markdown_outside_allowlist"')
        lines.append(f"path = {_esc(path)}")
        lines.append("")

    for path in sorted(classification_violations):
        lines.append("[[policy_violation]]")
        lines.append('kind = "classification_outside_safe_set"')
        lines.append(f"path = {_esc(path)}")
        lines.append("")

    for path in sorted(destination_missing):
        lines.append("[[policy_violation]]")
        lines.append('kind = "missing_toml_destination"')
        lines.append(f"path = {_esc(path)}")
        lines.append("")

    for i, (risk, _, path, row, lifecycle, destination_exists) in enumerate(risk_rows[:120], start=1):
        lines.append("[[risk_queue]]")
        lines.append(f"rank = {i}")
        lines.append(f"path = {_esc(path)}")
        lines.append(f"risk_score = {risk}")
        lines.append(f"classification = {_esc(str(row.get('classification', '')))}")
        lines.append(f"migration_action = {_esc(str(row.get('migration_action', '')))}")
        lines.append(f"migration_priority = {_esc(str(row.get('migration_priority', '')))}")
        lines.append(f"lifecycle = {_esc(lifecycle)}")
        lines.append(f"destination_exists = {'true' if destination_exists else 'false'}")
        if str(row.get("toml_destination", "")).strip():
            lines.append(f"toml_destination = {_esc(str(row.get('toml_destination', '')))}")
        lines.append(f"line_count = {int(row.get('line_count', 0))}")
        lines.append(f"rationale = {_esc(str(row.get('rationale', '')))}")
        lines.append("")

    for row in sorted(docs, key=lambda item: str(item.get("path", ""))):
        path = str(row.get("path", ""))
        lifecycle = _lifecycle(path, row)
        toml_destination = str(row.get("toml_destination", "")).strip()
        destination_exists = bool(toml_destination) and (root / toml_destination).is_file()
        risk = _risk_score(path, row, destination_exists)
        tracked_allowed = not (
            _in_policy_scope(path)
            and str(row.get("git_status", "")) == "tracked"
            and path not in ALLOWED_TRACKED_MARKDOWN
        )

        lines.append("[[document]]")
        lines.append(f"path = {_esc(path)}")
        lines.append(f"git_status = {_esc(str(row.get('git_status', '')))}")
        lines.append(f"classification = {_esc(str(row.get('classification', '')))}")
        lines.append(f"lifecycle = {_esc(lifecycle)}")
        lines.append(f"generated = {'true' if _bool(row.get('generated')) else 'false'}")
        lines.append(f"third_party = {'true' if _bool(row.get('third_party')) else 'false'}")
        lines.append(f"tracked_allowed = {'true' if tracked_allowed else 'false'}")
        lines.append(f"destination_exists = {'true' if destination_exists else 'false'}")
        lines.append(f"risk_score = {risk}")
        lines.append(f"size_bytes = {int(row.get('size_bytes', 0))}")
        lines.append(f"line_count = {int(row.get('line_count', 0))}")
        if toml_destination:
            lines.append(f"toml_destination = {_esc(toml_destination)}")
        lines.append("")

    rendered = "\n".join(lines)
    _assert_ascii(rendered, str(out_path))
    out_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {out_path} with {len(docs)} markdown corpus records.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

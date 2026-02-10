#!/usr/bin/env python3
"""
Build a strict origin audit for markdown under docs/reports/data-artifacts scopes.

The output captures whether each markdown file is verifiably generated from
repository TOML registries (with source headers) or needs consolidation into
the TOML scroll library.
"""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from collections import Counter
from pathlib import Path

SOURCE_RE = re.compile(r"Source of truth:\s*(.+?)\s*-->")
REGISTRY_PATH_RE = re.compile(r"registry/[A-Za-z0-9_./-]+\.toml")
IN_SCOPE_PREFIXES = ("docs/", "reports/", "data/artifacts/")


def _assert_ascii(text: str, context: str) -> None:
    bad = sorted({ch for ch in text if ord(ch) > 127})
    if bad:
        raise SystemExit(f"ERROR: Non-ASCII output in {context}: {''.join(bad[:20])!r}")


def _esc(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def _scope(path: str) -> str:
    if path.startswith("docs/"):
        return "docs"
    if path.startswith("reports/"):
        return "reports"
    if path.startswith("data/artifacts/"):
        return "data_artifacts"
    return "other"


def _origin_process(path: str, source_paths: list[str]) -> str:
    if path.startswith("docs/generated/"):
        return "src/scripts/analysis/export_registry_markdown_mirrors.py::mirror_exports"
    if source_paths:
        return "src/scripts/analysis/export_registry_markdown_mirrors.py::legacy_exports"
    return "unknown"


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
        help="Input markdown inventory TOML path.",
    )
    parser.add_argument(
        "--out",
        default="registry/markdown_origin_audit.toml",
        help="Output origin-audit TOML path.",
    )
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    inv_path = root / args.inventory
    out_path = root / args.out

    inv = tomllib.loads(inv_path.read_text(encoding="utf-8"))
    docs = [
        row
        for row in inv.get("document", [])
        if any(str(row.get("path", "")).startswith(prefix) for prefix in IN_SCOPE_PREFIXES)
    ]

    origin_status_counts: Counter[str] = Counter()
    scope_counts: Counter[str] = Counter()
    queue: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []

    for row in sorted(docs, key=lambda item: str(item.get("path", ""))):
        path = str(row.get("path", "")).strip()
        full = root / path
        text = full.read_text(encoding="utf-8", errors="ignore")
        head = "\n".join(text.splitlines()[:80])

        classification = str(row.get("classification", "")).strip()
        git_status = str(row.get("git_status", "")).strip()
        destination = str(row.get("toml_destination", "")).strip()
        destination_exists = bool(destination) and (root / destination).is_file()
        line_count = int(row.get("line_count", 0))

        has_auto = "AUTO-GENERATED" in head
        has_source = "Source of truth:" in head
        source_raw = ""
        source_paths: list[str] = []
        source_match = SOURCE_RE.search(head)
        if source_match:
            source_raw = source_match.group(1).strip()
            source_paths = sorted(set(REGISTRY_PATH_RE.findall(source_raw)))

        generated_class = classification == "toml_published_markdown"
        if not generated_class:
            origin_status = "non_generated_needs_consolidation"
            action = "migrate_to_toml_registry_and_regenerate"
        elif not destination_exists:
            origin_status = "missing_destination_registry"
            action = "repair_toml_destination_mapping"
        elif not (has_auto and has_source):
            origin_status = "missing_origin_headers"
            action = "regenerate_from_registry"
        elif not source_raw:
            origin_status = "missing_source_of_truth_value"
            action = "regenerate_with_source_header"
        else:
            origin_status = "generated_from_repo_process"
            action = "none"

        scope_name = _scope(path)
        origin_status_counts[origin_status] += 1
        scope_counts[scope_name] += 1

        entry = {
            "path": path,
            "scope": scope_name,
            "classification": classification,
            "git_status": git_status,
            "line_count": line_count,
            "toml_destination": destination,
            "destination_exists": destination_exists,
            "header_auto_generated": has_auto,
            "header_source_of_truth": has_source,
            "source_of_truth_raw": source_raw,
            "source_of_truth_paths": source_paths,
            "origin_process": _origin_process(path, source_paths),
            "origin_status": origin_status,
            "consolidation_action": action,
        }
        rows.append(entry)
        if action != "none":
            queue.append(entry)

    lines: list[str] = []
    lines.append("# Markdown origin audit registry (docs/reports/data-artifacts).")
    lines.append("# Generated by src/scripts/analysis/build_markdown_origin_audit.py")
    lines.append("")
    lines.append("[markdown_origin_audit]")
    lines.append('updated = "deterministic"')
    lines.append("authoritative = true")
    lines.append(f"source_inventory = {_esc(args.inventory)}")
    lines.append(f"document_count = {len(rows)}")
    lines.append(
        "generated_verified_count = "
        f"{origin_status_counts.get('generated_from_repo_process', 0)}"
    )
    lines.append(f"needs_consolidation_count = {len(queue)}")
    lines.append("")

    lines.append("[scope_counts]")
    for key in sorted(scope_counts):
        lines.append(f"{key} = {scope_counts[key]}")
    lines.append("")

    lines.append("[origin_status_counts]")
    for key in sorted(origin_status_counts):
        lines.append(f"{key} = {origin_status_counts[key]}")
    lines.append("")

    for i, item in enumerate(sorted(queue, key=lambda r: (-int(r["line_count"]), str(r["path"]))), start=1):
        lines.append("[[consolidation_queue]]")
        lines.append(f"rank = {i}")
        lines.append(f"path = {_esc(str(item['path']))}")
        lines.append(f"origin_status = {_esc(str(item['origin_status']))}")
        lines.append(f"classification = {_esc(str(item['classification']))}")
        lines.append(f"consolidation_action = {_esc(str(item['consolidation_action']))}")
        if str(item["toml_destination"]):
            lines.append(f"toml_destination = {_esc(str(item['toml_destination']))}")
        lines.append(f"line_count = {int(item['line_count'])}")
        lines.append("")

    for item in rows:
        lines.append("[[document]]")
        lines.append(f"path = {_esc(str(item['path']))}")
        lines.append(f"scope = {_esc(str(item['scope']))}")
        lines.append(f"classification = {_esc(str(item['classification']))}")
        lines.append(f"git_status = {_esc(str(item['git_status']))}")
        lines.append(f"origin_status = {_esc(str(item['origin_status']))}")
        lines.append(f"origin_process = {_esc(str(item['origin_process']))}")
        lines.append(
            "destination_exists = "
            f"{'true' if bool(item['destination_exists']) else 'false'}"
        )
        lines.append(
            "header_auto_generated = "
            f"{'true' if bool(item['header_auto_generated']) else 'false'}"
        )
        lines.append(
            "header_source_of_truth = "
            f"{'true' if bool(item['header_source_of_truth']) else 'false'}"
        )
        if str(item["toml_destination"]):
            lines.append(f"toml_destination = {_esc(str(item['toml_destination']))}")
        if str(item["source_of_truth_raw"]):
            lines.append(f"source_of_truth_raw = {_esc(str(item['source_of_truth_raw']))}")
        source_paths = list(item["source_of_truth_paths"])
        lines.append(
            "source_of_truth_paths = ["
            + ", ".join(_esc(str(path)) for path in source_paths)
            + "]"
        )
        lines.append(f"consolidation_action = {_esc(str(item['consolidation_action']))}")
        lines.append(f"line_count = {int(item['line_count'])}")
        lines.append("")

    rendered = "\n".join(lines)
    _assert_ascii(rendered, str(out_path))
    out_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {out_path} with {len(rows)} markdown origin records.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

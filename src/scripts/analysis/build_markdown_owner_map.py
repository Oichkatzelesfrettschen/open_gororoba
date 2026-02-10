#!/usr/bin/env python3
"""
Bootstrap the explicit markdown owner mapping table.

This script is intentionally NOT part of normal verification lanes.
It is a maintenance/bootstrap tool to create/update:
  registry/markdown_owner_map.toml
"""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

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


def _owner_group(path: str, destination: str) -> str:
    if destination == "registry/monograph.toml":
        return "monograph"
    if path.startswith("docs/book/src/"):
        return "book_docs"
    if path.startswith("docs/external_sources/"):
        return "external_sources"
    if path.startswith("docs/tickets/"):
        return "claim_tickets"
    if path.startswith("docs/claims/by_domain/"):
        return "claims_domains"
    if path.startswith("docs/convos/"):
        return "docs_convos"
    if path.startswith("docs/theory/") or path.startswith("docs/engineering/") or path.startswith("docs/research/"):
        return "research_narratives"
    if path.startswith("docs/") and destination == "registry/docs_root_narratives.toml":
        return "docs_root_narratives"
    if path.startswith("reports/"):
        return "reports_narratives"
    if path.startswith("data/artifacts/") and path != "data/artifacts/README.md":
        if destination == "registry/artifact_scrolls.toml":
            return "artifact_scrolls"
        return "data_artifact_narratives"
    return "general"


def _conversion_hint(path: str, destination: str) -> str:
    if destination == "registry/monograph.toml":
        return "Edit registry/monograph.toml document body_markdown and run: PYTHONWARNINGS=error MARKDOWN_EXPORT=1 make docs-publish"
    if path.startswith("docs/book/src/"):
        return (
            "Run: PYTHONWARNINGS=error python3 src/scripts/analysis/normalize_book_docs_registry.py "
            "--bootstrap-from-markdown && PYTHONWARNINGS=error MARKDOWN_EXPORT=1 make docs-publish"
        )
    if path.startswith("docs/external_sources/"):
        return (
            "Run: PYTHONWARNINGS=error python3 src/scripts/analysis/normalize_external_sources_registry.py "
            "--bootstrap-from-markdown && PYTHONWARNINGS=error MARKDOWN_EXPORT=1 make docs-publish"
        )
    if path.startswith("docs/tickets/") or path.startswith("docs/claims/by_domain/"):
        return (
            "Run: PYTHONWARNINGS=error python3 src/scripts/analysis/normalize_claims_support_registries.py "
            "--bootstrap-from-markdown && PYTHONWARNINGS=error MARKDOWN_EXPORT=1 make docs-publish"
        )
    if path.startswith("docs/convos/"):
        return (
            "Run: PYTHONWARNINGS=error python3 src/scripts/analysis/normalize_docs_convos_registry.py "
            "--bootstrap-from-markdown && PYTHONWARNINGS=error MARKDOWN_EXPORT=1 make docs-publish"
        )
    if path.startswith("docs/theory/") or path.startswith("docs/engineering/") or path.startswith("docs/research/"):
        return (
            "Run: PYTHONWARNINGS=error python3 src/scripts/analysis/normalize_research_narratives_registry.py "
            "--bootstrap-from-markdown && PYTHONWARNINGS=error MARKDOWN_EXPORT=1 make docs-publish"
        )
    if path.startswith("docs/"):
        return (
            "Run: PYTHONWARNINGS=error python3 src/scripts/analysis/normalize_docs_root_narratives_registry.py "
            "--bootstrap-from-markdown && PYTHONWARNINGS=error MARKDOWN_EXPORT=1 make docs-publish"
        )
    if path.startswith("reports/"):
        return (
            "Run: PYTHONWARNINGS=error python3 src/scripts/analysis/normalize_reports_narratives_registry.py "
            "--bootstrap-from-markdown && PYTHONWARNINGS=error MARKDOWN_EXPORT=1 make docs-publish"
        )
    if path.startswith("data/artifacts/") and path != "data/artifacts/README.md":
        return (
            "Run: PYTHONWARNINGS=error make registry-artifact-scrolls "
            "&& PYTHONWARNINGS=error MARKDOWN_EXPORT=1 make docs-publish"
        )
    return "Assign canonical registry owner and add a conversion pipeline."


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
        "--origin-audit",
        default="registry/markdown_origin_audit.toml",
        help="Input markdown origin audit TOML path.",
    )
    parser.add_argument(
        "--out",
        default="registry/markdown_owner_map.toml",
        help="Output owner mapping table.",
    )
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    inventory = tomllib.loads((root / args.inventory).read_text(encoding="utf-8"))
    origin = tomllib.loads((root / args.origin_audit).read_text(encoding="utf-8"))
    origin_by_path = {
        str(row.get("path", "")).strip(): row
        for row in origin.get("document", [])
    }
    docs = [
        row
        for row in inventory.get("document", [])
        if any(str(row.get("path", "")).startswith(prefix) for prefix in IN_SCOPE_PREFIXES)
    ]
    docs.sort(key=lambda row: str(row.get("path", "")))

    lines: list[str] = []
    lines.append("# Explicit markdown owner map for in-scope markdown files.")
    lines.append("# Generated by src/scripts/analysis/build_markdown_owner_map.py")
    lines.append("")
    lines.append("[markdown_owner_map]")
    lines.append('updated = "deterministic"')
    lines.append("authoritative = true")
    lines.append('scope = "docs reports data_artifacts"')
    lines.append(f"document_count = {len(docs)}")
    lines.append("")

    for i, row in enumerate(docs, start=1):
        path = str(row.get("path", "")).strip()
        destination = str(row.get("toml_destination", "")).strip()
        origin_row = origin_by_path.get(path, {})
        source_paths = [
            str(item).strip()
            for item in origin_row.get("source_of_truth_paths", [])
            if str(item).strip()
        ]
        if source_paths:
            destination = sorted(set(source_paths))[0]
        lines.append("[[owner]]")
        lines.append(f"id = {_esc(f'MOWN-{i:04d}')}")
        lines.append(f"path = {_esc(path)}")
        lines.append(f"scope = {_esc(_scope(path))}")
        lines.append(f"canonical_toml = {_esc(destination)}")
        lines.append(f"owner_group = {_esc(_owner_group(path, destination))}")
        lines.append(
            "requires_generated_header = true"
        )
        lines.append(f"conversion_hint = {_esc(_conversion_hint(path, destination))}")
        lines.append("")

    out_path = root / args.out
    rendered = "\n".join(lines)
    _assert_ascii(rendered, str(out_path))
    out_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {out_path} with {len(docs)} owner mappings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

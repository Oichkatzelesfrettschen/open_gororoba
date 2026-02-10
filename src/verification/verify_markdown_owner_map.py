#!/usr/bin/env python3
"""
Verify explicit markdown owner mapping.

Hard gate:
- Any in-scope markdown file must have an explicit owner-map entry.
- Owner map canonical TOML must exist and match inventory toml_destination.
- Generated headers must be present and reference the canonical TOML.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

IN_SCOPE_PREFIXES = ("docs/", "reports/", "data/artifacts/")


def _conversion_hint(path: str) -> str:
    if path.startswith("docs/book/src/"):
        return (
            "PYTHONWARNINGS=error python3 src/scripts/analysis/normalize_book_docs_registry.py "
            "--bootstrap-from-markdown && "
            "PYTHONWARNINGS=error MARKDOWN_EXPORT=1 MARKDOWN_EXPORT_EMIT_LEGACY=1 make docs-publish"
        )
    if path.startswith("docs/external_sources/"):
        return (
            "PYTHONWARNINGS=error python3 src/scripts/analysis/normalize_external_sources_registry.py "
            "--bootstrap-from-markdown && "
            "PYTHONWARNINGS=error MARKDOWN_EXPORT=1 MARKDOWN_EXPORT_EMIT_LEGACY=1 make docs-publish"
        )
    if path.startswith("docs/tickets/") or path.startswith("docs/claims/by_domain/"):
        return (
            "PYTHONWARNINGS=error python3 src/scripts/analysis/normalize_claims_support_registries.py "
            "--bootstrap-from-markdown && "
            "PYTHONWARNINGS=error MARKDOWN_EXPORT=1 MARKDOWN_EXPORT_EMIT_LEGACY=1 make docs-publish"
        )
    if path.startswith("docs/convos/"):
        return (
            "PYTHONWARNINGS=error python3 src/scripts/analysis/normalize_docs_convos_registry.py "
            "--bootstrap-from-markdown && "
            "PYTHONWARNINGS=error MARKDOWN_EXPORT=1 MARKDOWN_EXPORT_EMIT_LEGACY=1 make docs-publish"
        )
    if path.startswith("docs/theory/") or path.startswith("docs/engineering/") or path.startswith("docs/research/"):
        return (
            "PYTHONWARNINGS=error python3 src/scripts/analysis/normalize_research_narratives_registry.py "
            "--bootstrap-from-markdown && "
            "PYTHONWARNINGS=error MARKDOWN_EXPORT=1 MARKDOWN_EXPORT_EMIT_LEGACY=1 make docs-publish"
        )
    if path.startswith("docs/monograph/"):
        return "Edit registry/monograph.toml and run: PYTHONWARNINGS=error MARKDOWN_EXPORT=1 MARKDOWN_EXPORT_EMIT_LEGACY=1 make docs-publish"
    if path.startswith("docs/"):
        return (
            "PYTHONWARNINGS=error python3 src/scripts/analysis/normalize_docs_root_narratives_registry.py "
            "--bootstrap-from-markdown && "
            "PYTHONWARNINGS=error MARKDOWN_EXPORT=1 MARKDOWN_EXPORT_EMIT_LEGACY=1 make docs-publish"
        )
    if path.startswith("reports/"):
        return (
            "PYTHONWARNINGS=error python3 src/scripts/analysis/normalize_reports_narratives_registry.py "
            "--bootstrap-from-markdown && "
            "PYTHONWARNINGS=error MARKDOWN_EXPORT=1 MARKDOWN_EXPORT_EMIT_LEGACY=1 make docs-publish"
        )
    if path.startswith("data/artifacts/") and path != "data/artifacts/README.md":
        return (
            "PYTHONWARNINGS=error make registry-artifact-scrolls "
            "&& PYTHONWARNINGS=error MARKDOWN_EXPORT=1 MARKDOWN_EXPORT_EMIT_LEGACY=1 make docs-publish"
        )
    return "Assign canonical TOML owner in registry/markdown_owner_map.toml and regenerate."


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    inv = tomllib.loads((repo_root / "registry/markdown_inventory.toml").read_text(encoding="utf-8"))
    owner_map = tomllib.loads((repo_root / "registry/markdown_owner_map.toml").read_text(encoding="utf-8"))

    in_scope = [
        row
        for row in inv.get("document", [])
        if any(str(row.get("path", "")).startswith(prefix) for prefix in IN_SCOPE_PREFIXES)
    ]

    owner_rows = owner_map.get("owner", [])
    owner_by_path: dict[str, dict[str, object]] = {}
    duplicate_owner_paths: set[str] = set()
    for row in owner_rows:
        path = str(row.get("path", "")).strip()
        if path in owner_by_path:
            duplicate_owner_paths.add(path)
        owner_by_path[path] = row

    failures: list[str] = []
    if duplicate_owner_paths:
        for path in sorted(duplicate_owner_paths):
            failures.append(f"duplicate owner mapping entry: {path}")

    inventory_paths = {str(row.get("path", "")).strip() for row in in_scope}
    owner_paths = set(owner_by_path.keys())

    missing_owner = sorted(inventory_paths - owner_paths)
    stale_owner = sorted(owner_paths - inventory_paths)

    for path in missing_owner:
        failures.append(
            f"{path}: missing explicit owner mapping; conversion path: {_conversion_hint(path)}"
        )
    for path in stale_owner:
        failures.append(f"{path}: stale owner mapping (file not present in in-scope markdown)")

    for row in in_scope:
        path = str(row.get("path", "")).strip()
        classification = str(row.get("classification", "")).strip()

        owner = owner_by_path.get(path)
        if owner is None:
            continue
        canonical = str(owner.get("canonical_toml", "")).strip()
        if not canonical:
            failures.append(f"{path}: owner map canonical_toml is empty")
            continue
        if not (repo_root / canonical).is_file():
            failures.append(f"{path}: canonical_toml missing on disk: {canonical}")
        if classification != "toml_published_markdown":
            failures.append(f"{path}: classification={classification} (expected toml_published_markdown)")

        text = (repo_root / path).read_text(encoding="utf-8", errors="ignore")
        head = "\n".join(text.splitlines()[:80])
        if "AUTO-GENERATED" not in head:
            failures.append(f"{path}: missing AUTO-GENERATED header")
        if "Source of truth:" not in head:
            failures.append(f"{path}: missing Source of truth header")
        if canonical not in head:
            failures.append(
                f"{path}: Source of truth header does not reference canonical_toml ({canonical})"
            )

    if failures:
        print("ERROR: markdown owner map verification failed.")
        print("To map a new markdown file, add it to registry/markdown_owner_map.toml and use the suggested conversion command.")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("OK: markdown owner map verified. All in-scope markdown has explicit canonical TOML ownership.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Regression coverage for the SQLite-authored external-source compatibility exports.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_SOURCES_PATH = REPO_ROOT / "registry/external_sources.toml"

GENERATED_HEADER_PREFIX = "<!-- AUTO-GENERATED: READ-ONLY COMPATIBILITY EXPORT. -->"


def _load_registry() -> dict:
    return tomllib.loads(EXTERNAL_SOURCES_PATH.read_text(encoding="utf-8"))


def _strip_generated_header(raw: str) -> str:
    lines = raw.splitlines()
    while lines and lines[0].startswith("<!--"):
        lines.pop(0)
    while lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines).rstrip() + ("\n" if lines else "")


def test_source_markdown_covers_all_external_source_markdown_files() -> None:
    registry = _load_registry()
    markdown_names = {
        path.name for path in (REPO_ROOT / "docs" / "external_sources").glob("*.md")
    }
    assert markdown_names, "expected external-source markdown files to exist"

    source_markdown_names = {
        Path(str(row.get("source_markdown", ""))).name
        for row in registry.get("document", [])
        if str(row.get("source_markdown", "")).strip()
    }
    assert markdown_names == source_markdown_names, (
        "docs/external_sources coverage drifted from registry/external_sources.toml.\n"
        f"missing metadata: {sorted(markdown_names - source_markdown_names)}\n"
        f"stale metadata: {sorted(source_markdown_names - markdown_names)}"
    )


def test_strip_generated_header_removes_export_prefixes() -> None:
    raw = (
        GENERATED_HEADER_PREFIX
        + "\n<!-- Canonical write path: registry/canonical/control_plane.sqlite3 -->\n"
        + "\n# Flying Higher Than A Box-Kite\n\nBody line.\n"
    )
    stripped = _strip_generated_header(raw)
    assert stripped == "# Flying Higher Than A Box-Kite\n\nBody line.\n"


def test_registry_body_matches_generated_markdown_after_header_strip() -> None:
    registry = _load_registry()
    row = next(
        row
        for row in registry.get("document", [])
        if row.get("source_markdown")
        == "docs/external_sources/DE_MARRAIS_FLYING_HIGHER.md"
    )
    markdown_path = REPO_ROOT / "docs" / "external_sources" / "DE_MARRAIS_FLYING_HIGHER.md"
    raw = markdown_path.read_text(encoding="utf-8")
    assert raw.startswith(GENERATED_HEADER_PREFIX)
    assert _strip_generated_header(raw).rstrip() == str(row["body_markdown"]).rstrip()

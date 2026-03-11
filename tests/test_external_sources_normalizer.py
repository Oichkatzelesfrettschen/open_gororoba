"""
Regression coverage for the external-sources markdown normalizer.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "src/scripts/analysis/normalize_external_sources_registry.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "normalize_external_sources_registry",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_source_meta_covers_all_external_source_markdown_files() -> None:
    module = _load_module()
    docs_dir = REPO_ROOT / "docs/external_sources"
    markdown_names = {path.name for path in docs_dir.glob("*.md")}
    assert markdown_names, "expected external-source markdown files to exist"
    assert markdown_names == set(module.SOURCE_META), (
        "docs/external_sources coverage drifted from SOURCE_META.\n"
        f"missing metadata: {sorted(markdown_names - set(module.SOURCE_META))}\n"
        f"stale metadata: {sorted(set(module.SOURCE_META) - markdown_names)}"
    )


def test_strip_generated_header_removes_repeated_generated_prefixes() -> None:
    module = _load_module()
    raw = (
        module.GENERATED_HEADER_PREFIX
        + "\n"
        + module.GENERATED_HEADER_PREFIX
        + "\n# Flying Higher Than A Box-Kite\n\nBody line.\n"
    )
    stripped = module._strip_generated_header(raw)
    assert stripped == "# Flying Higher Than A Box-Kite\n\nBody line.\n"


def test_parse_doc_ignores_generated_header_when_deriving_title_and_body() -> None:
    module = _load_module()
    path = Path("docs/external_sources/DE_MARRAIS_FLYING_HIGHER.md")
    raw = (
        module.GENERATED_HEADER_PREFIX
        + "\n"
        + module.GENERATED_HEADER_PREFIX
        + "\n# Flying Higher Than A Box-Kite\n\n## Full Transcript\n\nBody line.\n"
    )

    record = module._parse_doc(5, path, raw)

    assert record.title == "Flying Higher Than A Box-Kite"
    assert record.has_full_transcript is True
    assert record.body_markdown == "# Flying Higher Than A Box-Kite\n\n## Full Transcript\n\nBody line."

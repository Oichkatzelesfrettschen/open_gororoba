#!/usr/bin/env python3
"""
Build markdown lifecycle governance registry.

The registry classifies each tracked markdown file into one of:
- toml_generated_mirror
- toml_manual_source
- generated_artifact
- manual_narrative
- immutable_transcript

This allows enforceable TOML-first policy and explicit exceptions.
"""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
import tomllib
from pathlib import Path

IMMUTABLE_AGENT_OVERLAYS = {"CLAUDE.md", "GEMINI.md"}
SAFE_CLASSIFICATIONS = [
    "toml_published_markdown",
    "toml_destination_exists_manual_markdown",
    "generated_artifact",
    "third_party_markdown",
]
TRACKED_ALLOWED_MODES = ["toml_generated_mirror", "toml_manual_source"]
TRACKED_ALLOWED_PATHS = [
    "docs/research/high_dimensional_algebra_unification_2026.md",
    "proofs/EPISTEMIC_BOUNDARIES.md",
]
EMBEDDED_MARKDOWN_PREFIXES = ["docs/", "reports/", "data/artifacts/"]
EMBEDDED_MARKDOWN_ROOT_PATHS = [
    "AGENTS.md",
    "CLAUDE.md",
    "GEMINI.md",
    "README.md",
    "PANTHEON_PHYSICSFORGE_90_POINT_MIGRATION_PLAN.md",
    "PHASE10_11_ULTIMATE_ROADMAP.md",
    "PYTHON_REFACTORING_ROADMAP.md",
    "SYNTHESIS_PIPELINE_PROGRESS.md",
    "crates/sign_imbalance/IMPLEMENTATION_NOTES.md",
    "curated/README.md",
    "curated/01_theory_frameworks/README_COQ.md",
    "data/csv/README.md",
    "data/artifacts/README.md",
    "NAVIGATOR.md",
    "REQUIREMENTS.md",
    "docs/REQUIREMENTS.md",
]
OWNER_SCOPE_PREFIXES = ["docs/", "reports/", "data/artifacts/"]
OWNER_SCOPE_PATHS = [
    "AGENTS.md",
    "CLAUDE.md",
    "GEMINI.md",
    "README.md",
    "apps/gororoba_studio/README.md",
    "crates/lbm_3d_cuda/README.md",
    "curated/README.md",
    "data/csv/README.md",
    "data/external/README.md",
    "proofs/EPISTEMIC_BOUNDARIES.md",
    "proofs/README.md",
]
GENERATED_PATTERNS = ["build/docs/generated/*.md", "docs/generated/*.md"]
SKIP_PREFIXES = [
    ".cache/",
    "reports/gates/",
    ".pytest_cache/",
    "venv/",
    ".venv/",
    ".venv_ingest/",
    ".horusec/",
    ".claude/",
    ".gemini/",
    ".playwright-mcp/",
    ".mamba/",
    "target/",
    "logs/",
    "build/",
    "dist/",
    "temp/",
    "tmp/",
]
SKIP_PATH_PARTS = [
    ".cache",
    "cargo-home",
    ".pytest_cache",
    "venv",
    ".venv",
    "target",
    "logs",
    "build",
    "dist",
    "temp",
    "tmp",
]
DISK_FORBIDDEN_MODES = ["deleted_mirror"]


def _escape(text: str) -> str:
    esc = (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{esc}"'


def _assert_ascii(text: str, context: str) -> None:
    bad = sorted({ch for ch in text if ord(ch) > 127})
    if bad:
        raise SystemExit(f"ERROR: Non-ASCII output in {context}: {''.join(bad[:20])!r}")


def _iter_registry_refs(root: Path) -> dict[str, set[str]]:
    refs: dict[str, set[str]] = {}
    reg_files = sorted((root / "registry").glob("*.toml"))
    declarative_only = {"knowledge_migration_plan.toml"}

    def add(path: str, src: str) -> None:
        path = path.strip()
        if not path.endswith(".md"):
            return
        refs.setdefault(path, set()).add(src)

    def walk(obj: object, src: str) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                lk = key.lower()
                if lk in {"source_markdown", "markdown", "path"}:
                    if isinstance(value, str):
                        add(value, src)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, str):
                                add(item, src)
                elif lk in {"source_markdown_glob", "source_markdown_globs"}:
                    if isinstance(value, str):
                        for p in root.glob(value):
                            if p.is_file():
                                add(p.relative_to(root).as_posix(), src)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, str):
                                for p in root.glob(item):
                                    if p.is_file():
                                        add(p.relative_to(root).as_posix(), src)
                else:
                    walk(value, src)
        elif isinstance(obj, list):
            for item in obj:
                walk(item, src)

    for reg in reg_files:
        if reg.name in declarative_only:
            continue
        data = tomllib.loads(reg.read_text(encoding="utf-8"))
        walk(data, reg.relative_to(root).as_posix())

    return refs


def _git_paths(root: Path, args: list[str]) -> set[str]:
    proc = subprocess.run(
        ["git", *args, "--", "*.md"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def _generated_mirror_patterns() -> list[str]:
    return [
        "AGENTS.md",
        "README.md",
        "curated/README.md",
        "curated/01_theory_frameworks/README_COQ.md",
        "data/csv/README.md",
        "data/artifacts/README.md",
        "NAVIGATOR.md",
        "docs/generated/*.md",
        "docs/CLAIMS_EVIDENCE_MATRIX.md",
        "docs/BIBLIOGRAPHY.md",
        "docs/INSIGHTS.md",
        "docs/EXPERIMENTS_PORTFOLIO_SHORTLIST.md",
        "docs/ROADMAP.md",
        "docs/TODO.md",
        "docs/NEXT_ACTIONS.md",
        "docs/CLAIMS_TASKS.md",
        "docs/claims/INDEX.md",
        "docs/claims/by_domain/*.md",
        "docs/tickets/*.md",
        "docs/tickets/INDEX.md",
        "docs/book/src/*.md",
        "docs/book/src/*/*.md",
        "docs/book/src/*/*/*.md",
        "docs/external_sources/*.md",
        "docs/external_sources/INDEX.md",
        "docs/theory/*.md",
        "docs/theory/INDEX.md",
        "docs/engineering/*.md",
        "docs/engineering/INDEX.md",
        "data/artifacts/ALGEBRAIC_FOUNDATIONS.md",
        "data/artifacts/BIBLIOGRAPHY.md",
        "data/artifacts/FINAL_REPORT.md",
        "data/artifacts/QUANTUM_REPORT.md",
        "data/artifacts/SIMULATION_REPORT.md",
        "data/artifacts/extracted_equations.md",
        "data/artifacts/reality_check_and_synthesis.md",
        "reports/*.md",
        "docs/convos/*.md",
        "REQUIREMENTS.md",
        "docs/REQUIREMENTS.md",
        "docs/requirements/*.md",
    ]


def _is_generated_mirror(path: str) -> bool:
    for pattern in _generated_mirror_patterns():
        if fnmatch.fnmatch(path, pattern):
            return True
    return False


def _render(records: list[dict]) -> str:
    by_mode: dict[str, int] = {}
    for rec in records:
        mode = rec["mode"]
        by_mode[mode] = by_mode.get(mode, 0) + 1

    lines: list[str] = []
    lines.append("# Markdown lifecycle governance registry (TOML-first).")
    lines.append("# Generated by src/scripts/analysis/build_markdown_governance_registry.py")
    lines.append("")
    lines.append("[markdown_governance]")
    lines.append('generated_at = "deterministic"')
    lines.append("authoritative = true")
    lines.append(f"document_count = {len(records)}")
    for key in sorted(by_mode):
        lines.append(f"{key}_count = {by_mode[key]}")
    lines.append("")
    lines.append("[policy]")
    lines.append(
        "safe_classifications = [" + ", ".join(_escape(item) for item in SAFE_CLASSIFICATIONS) + "]"
    )
    lines.append(
        "tracked_allowed_modes = ["
        + ", ".join(_escape(item) for item in TRACKED_ALLOWED_MODES)
        + "]"
    )
    lines.append(
        "tracked_allowed_paths = ["
        + ", ".join(_escape(item) for item in TRACKED_ALLOWED_PATHS)
        + "]"
    )
    lines.append(
        "embedded_markdown_prefixes = ["
        + ", ".join(_escape(item) for item in EMBEDDED_MARKDOWN_PREFIXES)
        + "]"
    )
    lines.append(
        "embedded_markdown_root_paths = ["
        + ", ".join(_escape(item) for item in EMBEDDED_MARKDOWN_ROOT_PATHS)
        + "]"
    )
    lines.append(
        "owner_scope_prefixes = [" + ", ".join(_escape(item) for item in OWNER_SCOPE_PREFIXES) + "]"
    )
    lines.append(
        "owner_scope_paths = [" + ", ".join(_escape(item) for item in OWNER_SCOPE_PATHS) + "]"
    )
    lines.append(
        "generated_patterns = [" + ", ".join(_escape(item) for item in GENERATED_PATTERNS) + "]"
    )
    lines.append("skip_prefixes = [" + ", ".join(_escape(item) for item in SKIP_PREFIXES) + "]")
    lines.append("skip_path_parts = [" + ", ".join(_escape(item) for item in SKIP_PATH_PARTS) + "]")
    lines.append(
        "disk_forbidden_modes = [" + ", ".join(_escape(item) for item in DISK_FORBIDDEN_MODES) + "]"
    )
    lines.append("")

    for rec in records:
        lines.append("[[document]]")
        lines.append(f"path = {_escape(rec['path'])}")
        lines.append(f"id = {_escape(rec['id'])}")
        lines.append(f"kind = {_escape(rec['kind'])}")
        lines.append(f"mode = {_escape(rec['mode'])}")
        lines.append(f"header_required = {'true' if rec['header_required'] else 'false'}")
        if rec["source_toml_refs"]:
            refs = ", ".join(_escape(x) for x in rec["source_toml_refs"])
            lines.append(f"source_toml_refs = [{refs}]")
        if rec["notes"]:
            lines.append(f"notes = {_escape(rec['notes'])}")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[3]),
        help="Repository root.",
    )
    parser.add_argument(
        "--knowledge-index",
        default="registry/knowledge_sources.toml",
        help="Knowledge source index path.",
    )
    parser.add_argument(
        "--inventory",
        default="registry/markdown_inventory.toml",
        help="Markdown inventory TOML path.",
    )
    parser.add_argument(
        "--out",
        default="registry/markdown_governance.toml",
        help="Output governance registry path.",
    )
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    knowledge = tomllib.loads((root / args.knowledge_index).read_text(encoding="utf-8"))
    inventory = tomllib.loads((root / args.inventory).read_text(encoding="utf-8"))
    refs = _iter_registry_refs(root)
    tracked_markdown = _git_paths(root, ["ls-files"])
    inventory_by_path = {
        str(row.get("path", "")).strip(): row
        for row in inventory.get("document", [])
        if str(row.get("path", "")).strip().endswith(".md")
    }
    knowledge_by_path = {
        str(row.get("path", "")).strip(): row
        for row in knowledge.get("document", [])
        if str(row.get("path", "")).strip().endswith(".md")
    }

    governed_paths: set[str] = set(knowledge_by_path)
    governed_paths.update(refs.keys())
    governed_paths.update(path for path in tracked_markdown if path.endswith(".md"))

    records: list[dict] = []
    for i, path in enumerate(sorted(governed_paths), start=1):
        row = knowledge_by_path.get(path, {})
        inv_row = inventory_by_path.get(path, {})
        if not path.endswith(".md"):
            continue
        classification = str(inv_row.get("classification", "")).strip()
        git_status = str(inv_row.get("git_status", "")).strip()
        if git_status and git_status != "tracked" and classification != "generated_artifact":
            continue
        if classification and classification not in SAFE_CLASSIFICATIONS:
            continue
        kind = str(row.get("kind", "")) or classification or "markdown"
        source_refs = sorted(refs.get(path, set()))
        toml_backing = (
            str(row.get("toml_backing", "")).strip()
            or str(inv_row.get("toml_destination", "")).strip()
        )
        if toml_backing:
            source_refs = [toml_backing] + [ref for ref in source_refs if ref != toml_backing]

        if classification == "third_party_markdown":
            mode = "third_party_markdown"
            header_required = False
            notes = "Third-party or cache markdown; allowed on disk but not authoritative."
        elif classification == "generated_artifact":
            mode = "generated_artifact"
            header_required = False
            notes = "Generated artifact/report; preserve reproducibility."
        elif path in IMMUTABLE_AGENT_OVERLAYS:
            mode = "toml_manual_source"
            header_required = False
            notes = "Manual compatibility stub; TOML pipelines must not rewrite this file."
        elif (
            classification == "toml_published_markdown"
            or (path.startswith("docs/") and path.count("/") == 1)
            or _is_generated_mirror(path)
        ):
            mode = "toml_generated_mirror"
            header_required = True
            notes = "Generated from TOML registries and overlays."
        elif kind == "transcript_input":
            mode = "immutable_transcript"
            header_required = False
            notes = "Immutable transcript input; not authoritative for claims."
        elif source_refs:
            mode = "toml_manual_source"
            header_required = False
            notes = "Manual source consumed by TOML normalizers."
        elif kind in {"generated_markdown", "artifact_report"}:
            mode = "generated_artifact"
            header_required = False
            notes = "Generated artifact/report; preserve reproducibility."
        else:
            mode = "manual_narrative"
            header_required = False
            notes = "Manual narrative source; raw-captured in registry/knowledge/docs."

        records.append(
            {
                "id": f"MDG-{i:04d}",
                "path": path,
                "kind": kind,
                "mode": mode,
                "header_required": header_required,
                "source_toml_refs": source_refs,
                "notes": notes,
            }
        )

    out_text = _render(records)
    out_path = root / args.out
    _assert_ascii(out_text, str(out_path))
    out_path.write_text(out_text, encoding="utf-8")
    print(f"Wrote {out_path} with {len(records)} entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
